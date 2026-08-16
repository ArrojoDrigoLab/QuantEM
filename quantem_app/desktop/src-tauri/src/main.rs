//! QuantEM desktop shell.
//!
//! The shell's entire job:
//!
//! 1. pick a free loopback port;
//! 2. spawn the `quantem-server` sidecar (a PyInstaller onedir build of the
//!    `quantem` Python package) as `quantem-server serve --port N`;
//! 3. show a loading page, wait until the port answers, then navigate the
//!    webview to `http://127.0.0.1:N/`;
//! 4. guarantee the sidecar dies with the window.
//!
//! **Storage (owner ruling 2026-08-09).** Everything the app stores -- sqlite
//! DB, models, HF cache, exports, logs, and the WebView2 profile -- lives with
//! the installation: `QUANTEM_DATA_DIR` if set (the one explicit override),
//! otherwise `<exe dir>\data`. The shell resolves that directory before doing
//! anything else, exports it as `QUANTEM_DATA_DIR` so the sidecar and its
//! workers inherit the same choice, and gives the webview an explicit profile
//! directory at `<data dir>\webview-profile` (without it, tauri forces the
//! profile into `%LOCALAPPDATA%\<identifier>`). There is deliberately NO
//! fallback to `%LOCALAPPDATA%`: an unwritable data directory is a hard error
//! that names the override.
//!
//! `TEMP`/`TMP` go with them, into `<data dir>\data\tmp`. Without that, an
//! import wrote the whole image to the user's `%TEMP%` on C: -- twice, once by
//! the web server buffering the request body and once by Django staging the
//! upload -- and then copied it across volumes into the data directory. That
//! is the "nothing is written to C:" rule broken by the shipped build, and
//! roughly 2 GiB of pointless I/O on a 1 GiB image. The path deliberately
//! matches `quantem.core.config.TMP_DIR`, so the staged upload is a sibling of
//! its destination and the move is a rename; a test on the Python side asserts
//! the two agree.
//!
//! **Process lifetime (Windows).** The sidecar is placed in a Job Object with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The job handle lives as long as this
//! process; when the shell exits -- cleanly, killed, or crashed -- the OS
//! closes the handle and terminates every process in the job, including the
//! spawned segmentation workers (`quantem.jobs.runner` re-executes the sidecar
//! binary via multiprocessing, and those children land in the same job). This
//! codebase has a history of leaked worker processes; the job object is the
//! belt, the `RunEvent::Exit` kill is the braces.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::Deserialize;
use tauri::{Manager, RunEvent, WebviewWindow, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;

/// How long the first launch may take before we give up. Cold start pays for
/// the frozen torch import plus the initial database migration.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Windows job object: kill the whole sidecar tree when the shell dies.
// ---------------------------------------------------------------------------
#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// Owns the job handle. Dropping it (or process death) kills the job.
    pub struct Job(HANDLE);

    // HANDLE is a raw pointer; the job handle is only ever used behind a
    // mutex-protected owner, and the OS side is thread-safe.
    unsafe impl Send for Job {}

    impl Job {
        pub fn new() -> Option<Job> {
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle == 0 as HANDLE {
                    return None;
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const core::ffi::c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 {
                    CloseHandle(handle);
                    return None;
                }
                Some(Job(handle))
            }
        }

        pub fn assign(&self, child: &std::process::Child) -> bool {
            unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle() as HANDLE) != 0 }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

/// Everything the exit handler needs to tear the sidecar down.
struct Sidecar {
    child: Option<Child>,
    #[cfg(windows)]
    _job: Option<job::Job>,
}

impl Sidecar {
    fn kill(&mut self) {
        if let Some(child) = self.child.as_mut() {
            // Best-effort explicit kill; the job object (dropped right after,
            // when `self` goes) sweeps the whole tree including workers.
            let _ = child.kill();
            let _ = child.wait();
        }
        self.child = None;
        #[cfg(windows)]
        {
            // Dropping the job handle fires KILL_ON_JOB_CLOSE for anything left.
            self._job = None;
        }
    }
}

/// Where is the sidecar? Checked in order:
///
/// 1. `QUANTEM_SERVER_EXE` -- explicit override, used by `tauri dev` and tests;
/// 2. `<exe dir>/quantem-server/quantem-server.exe` -- the shipped layout: the
///    PyInstaller onedir sits whole next to the shell (portable zip and the
///    Windows bootstrap installer both produce this);
/// 3. the Tauri resource dir, used by the macOS bundle.
fn find_sidecar(app: &tauri::AppHandle) -> Option<PathBuf> {
    let exe_name = if cfg!(windows) {
        "quantem-server.exe"
    } else {
        "quantem-server"
    };
    if let Ok(overridden) = env::var("QUANTEM_SERVER_EXE") {
        let p = PathBuf::from(overridden);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = env::current_exe() {
        if let Some(dir) = exe.parent() {
            let p = dir.join("quantem-server").join(exe_name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    if let Ok(resources) = app.path().resource_dir() {
        let p = resources.join("quantem-server").join(exe_name);
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Data directory (owner ruling 2026-08-09: all storage lives with the install)
// ---------------------------------------------------------------------------

/// Where `log_line` writes. Set once the data directory is known; until then
/// (or if the data directory is unwritable) the temp dir keeps the evidence.
static LOG_DIR: OnceLock<PathBuf> = OnceLock::new();

/// `QUANTEM_DATA_DIR` if set and non-empty (the explicit override), otherwise
/// `<exe dir>\data` -- except on macOS, where app translocation makes that
/// unusable and the location is `~/Library/Application Support/QuantEM`.
/// Returns the directory and whether it came from the override.
fn resolve_data_dir() -> Result<(PathBuf, bool), String> {
    if let Some(v) = env::var_os("QUANTEM_DATA_DIR") {
        if !v.is_empty() {
            let p = PathBuf::from(&v);
            if !p.is_absolute() {
                // The sidecar refuses relative paths too; failing here keeps
                // the shell and sidecar from resolving two different places.
                return Err(format!(
                    "QUANTEM_DATA_DIR must be an absolute path (got {}).",
                    p.display()
                ));
            }
            return Ok((p, true));
        }
    }
    // macOS is the one platform where storage-beside-the-executable cannot
    // work. Gatekeeper's app translocation runs a quarantined app from a
    // randomised read-only mount under /private/var/folders, so the bundle's
    // own directory is not writable on exactly the first launch that matters --
    // and the mount path changes between launches, so anything written before
    // the user clears quarantine would be orphaned where they will never find
    // it. This mirrors the same branch in `quantem.cli.default_data_dir`; the
    // two must agree, because whichever resolves first exports
    // QUANTEM_DATA_DIR for the other.
    #[cfg(target_os = "macos")]
    {
        let home = env::var_os("HOME")
            .ok_or_else(|| "cannot locate the home directory ($HOME is unset).".to_string())?;
        let dir = PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("QuantEM");
        return Ok((dir, false));
    }

    #[cfg(not(target_os = "macos"))]
    {
        let exe =
            env::current_exe().map_err(|e| format!("cannot locate the QuantEM executable: {e}"))?;
        let dir = exe
            .parent()
            .ok_or_else(|| "the QuantEM executable has no parent directory".to_string())?
            .join("data");
        Ok((dir, false))
    }
}

/// Create the directory and prove it is writable. No silent fallback: the
/// caller turns a failure into a visible error naming `QUANTEM_DATA_DIR`.
fn ensure_writable(dir: &PathBuf) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let probe = dir.join(".quantem-write-probe");
    std::fs::write(&probe, b"probe")
        .map_err(|e| format!("cannot write inside {}: {e}", dir.display()))?;
    let _ = std::fs::remove_file(&probe);
    Ok(())
}

fn free_port() -> std::io::Result<u16> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
    Ok(listener.local_addr()?.port())
}

fn port_open(port: u16) -> bool {
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_ok()
}

/// Append a line to the shell log (best effort; the sidecar's own output is
/// mirrored here too, since a windowed exe has no console to inherit). Lives
/// in `<data dir>\logs\quantem-desktop.log`; falls back to the temp dir only
/// while the data directory is unknown or unwritable.
fn log_line(line: &str) {
    let path = match LOG_DIR.get() {
        Some(dir) => dir.join("quantem-desktop.log"),
        None => env::temp_dir().join("quantem-desktop.log"),
    };
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(f, "{line}");
    }
}

fn eval_on_main(app: &tauri::AppHandle, js: String) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.eval(&js);
    }
}

fn set_status(app: &tauri::AppHandle, text: &str) {
    eval_on_main(
        app,
        format!(
            "window.__quantemStatus && window.__quantemStatus({})",
            serde_json_escape(text)
        ),
    );
}

fn fail(app: &tauri::AppHandle, text: &str) {
    log_line(&format!("[shell] FATAL: {text}"));
    eval_on_main(
        app,
        format!(
            "window.__quantemFailed && window.__quantemFailed({})",
            serde_json_escape(text)
        ),
    );
}

/// Minimal JS string literal escaping (we control every input; this guards
/// backslashes in Windows paths and quotes in error text).
fn serde_json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '<' => out.push_str("\\u003c"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// Native saves: both WKWebView and WebView2 use the OS dialog, then Rust owns
// the write. This avoids browser download behavior entirely in packaged apps.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveTextRequest {
    suggested_name: String,
    mime_type: String,
    contents: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SaveUrlRequest {
    suggested_name: String,
    mime_type: String,
    url: String,
}

fn safe_suggested_name(value: &str) -> String {
    let basename = value
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or("export");
    let cleaned: String = basename
        .chars()
        .filter(|character| !character.is_control())
        .take(180)
        .collect();
    if cleaned.trim().is_empty() {
        "export".to_owned()
    } else {
        cleaned
    }
}

fn file_filter(suggested_name: &str, mime_type: &str) -> Option<(String, String)> {
    let extension = Path::new(suggested_name)
        .extension()?
        .to_str()?
        .trim()
        .to_ascii_lowercase();
    if extension.is_empty()
        || extension.len() > 15
        || !extension.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
    {
        return None;
    }
    let label = match mime_type.split(';').next().unwrap_or("").trim() {
        "text/csv" => "CSV file".to_owned(),
        "image/png" => "PNG image".to_owned(),
        "application/json" => "JSON file".to_owned(),
        _ => format!("{} file", extension.to_ascii_uppercase()),
    };
    Some((label, extension))
}

fn choose_save_destination(
    window: &WebviewWindow,
    suggested_name: &str,
    mime_type: &str,
) -> Result<Option<PathBuf>, String> {
    let suggested_name = safe_suggested_name(suggested_name);
    let mut dialog = window
        .dialog()
        .file()
        .set_parent(window)
        .set_title("Save QuantEM export")
        .set_file_name(&suggested_name);
    if let Some((label, extension)) = file_filter(&suggested_name, mime_type) {
        dialog = dialog.add_filter(label, &[&extension]);
    }
    dialog
        .blocking_save_file()
        .map(|selection| {
            selection
                .into_path()
                .map_err(|error| format!("the selected destination is not a local file: {error}"))
        })
        .transpose()
}

fn stream_to_file_atomically(destination: &Path, source: &mut impl Read) -> Result<(), String> {
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| "the selected file has no parent directory".to_owned())?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
        format!(
            "could not create a temporary file beside {}: {error}",
            destination.display()
        )
    })?;
    std::io::copy(source, temporary.as_file_mut())
        .map_err(|error| format!("could not write {}: {error}", destination.display()))?;
    temporary
        .as_file_mut()
        .flush()
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| {
            format!(
                "could not finish writing {}: {error}",
                destination.display()
            )
        })?;
    temporary.persist(destination).map_err(|error| {
        format!(
            "could not replace {} with the completed export: {}",
            destination.display(),
            error.error
        )
    })?;
    Ok(())
}

fn validated_export_url(value: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(value).map_err(|_| "the export URL is invalid".to_owned())?;
    if url.scheme() != "http" {
        return Err("desktop exports must use the local HTTP server".to_owned());
    }
    if !url.username().is_empty() || url.password().is_some() || url.fragment().is_some() {
        return Err("the export URL contains unsupported credentials or a fragment".to_owned());
    }
    if !matches!(
        url.host_str(),
        Some("127.0.0.1" | "localhost" | "::1" | "[::1]")
    ) {
        return Err("desktop exports are restricted to the QuantEM loopback server".to_owned());
    }

    let segments: Vec<&str> = url.path().trim_matches('/').split('/').collect();
    let asset_export = matches!(
        segments.as_slice(),
        ["api", "assets", asset_id, "export-png"] if !asset_id.is_empty()
    );
    let analysis_export = matches!(
        segments.as_slice(),
        ["api", "analysis", run_id, "export", name]
            if !run_id.is_empty()
                && matches!(
                    *name,
                    "objects.csv" | "image_summary.csv" | "composition.csv" | "manifest.json"
                )
    );
    if !asset_export && !analysis_export {
        return Err("the URL is not a supported QuantEM export endpoint".to_owned());
    }
    Ok(url)
}

fn download_export(destination: &Path, url: reqwest::Url) -> Result<(), String> {
    // The updater deliberately enables reqwest's provider-neutral Rustls
    // feature. A reqwest Client still constructs its TLS connector for plain
    // loopback HTTP, so select the already-linked Ring provider explicitly.
    // If another plugin selected a provider first, keeping it is also valid.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(60 * 60))
        .build()
        .map_err(|error| format!("could not prepare the export request: {error}"))?;
    let mut response = client
        .get(url)
        .send()
        .map_err(|error| format!("the export request failed: {error}"))?;
    let status = response.status();
    if !status.is_success() {
        let mut detail = String::new();
        let _ = response.take(500).read_to_string(&mut detail);
        let detail = detail.trim();
        return Err(if detail.is_empty() {
            format!("the export request failed ({status})")
        } else {
            format!("the export request failed ({status}): {detail}")
        });
    }
    stream_to_file_atomically(destination, &mut response)
}

#[tauri::command]
async fn save_text_file(window: WebviewWindow, request: SaveTextRequest) -> Result<bool, String> {
    let Some(destination) =
        choose_save_destination(&window, &request.suggested_name, &request.mime_type)?
    else {
        return Ok(false);
    };
    tauri::async_runtime::spawn_blocking(move || {
        let mut contents = request.contents.as_bytes();
        stream_to_file_atomically(&destination, &mut contents)
    })
    .await
    .map_err(|error| format!("the save worker stopped unexpectedly: {error}"))??;
    Ok(true)
}

#[tauri::command]
async fn save_url_file(window: WebviewWindow, request: SaveUrlRequest) -> Result<bool, String> {
    let url = validated_export_url(&request.url)?;
    let Some(destination) =
        choose_save_destination(&window, &request.suggested_name, &request.mime_type)?
    else {
        return Ok(false);
    };
    tauri::async_runtime::spawn_blocking(move || download_export(&destination, url))
        .await
        .map_err(|error| format!("the export worker stopped unexpectedly: {error}"))??;
    Ok(true)
}

fn main() {
    // Resolve storage before anything else: the WebView2 profile location must
    // be decided before the webview exists, and the sidecar inherits the same
    // choice through the environment.
    let storage: Result<PathBuf, String> =
        resolve_data_dir().and_then(|(dir, explicit)| match ensure_writable(&dir) {
            Ok(()) => Ok(dir),
            Err(e) if explicit => Err(format!(
                "the data directory set by the QUANTEM_DATA_DIR environment \
                 variable is not writable:\n{e}\nUnset QUANTEM_DATA_DIR or \
                 point it at a writable folder."
            )),
            Err(e) => Err(format!(
                "the QuantEM data directory is not writable:\n{e}\nMove the \
                 installation to a writable folder, or set the QUANTEM_DATA_DIR \
                 environment variable to a writable location."
            )),
        });

    let webview_profile: Option<PathBuf> = match &storage {
        Ok(dir) => {
            let logs = dir.join("logs");
            let _ = std::fs::create_dir_all(&logs);
            let _ = LOG_DIR.set(logs);
            let profile = dir.join("webview-profile");
            // The sidecar and its spawned workers inherit this; an explicit
            // QUANTEM_DATA_DIR set by the user is what produced `dir` anyway.
            env::set_var("QUANTEM_DATA_DIR", dir);
            // Temp files too -- the sidecar's web server spools a large upload
            // here before Django ever sees it, and the default is the user's
            // %TEMP% on C:. Same path as quantem.core.config.TMP_DIR, so a
            // staged upload is a sibling of where it is going. Only exported
            // if the directory could actually be created: pointing TEMP at a
            // path that does not exist breaks every temporary file in the
            // process, which is worse than leaving it on C:.
            let temp = dir.join("data").join("tmp");
            match std::fs::create_dir_all(&temp) {
                Ok(()) => {
                    env::set_var("TEMP", &temp);
                    env::set_var("TMP", &temp);
                    env::set_var("TMPDIR", &temp);
                }
                Err(e) => log_line(&format!(
                    "[shell] WARNING: cannot create {} ({e}); temp files stay in the \
                     system temp directory",
                    temp.display()
                )),
            }
            // Belt (env var, honoured by WebView2 when no explicit folder is
            // passed) and braces (`data_directory` on the window builder in
            // setup, which is what tauri actually passes to WebView2 -- without
            // it the profile is forced into %LOCALAPPDATA%\<identifier>).
            env::set_var("WEBVIEW2_USER_DATA_FOLDER", &profile);
            Some(profile)
        }
        Err(e) => {
            log_line(&format!("[shell] FATAL data dir: {e}"));
            None
        }
    };
    if let Ok(dir) = &storage {
        log_line(&format!("[shell] data dir: {}", dir.display()));
    }

    let sidecar: Arc<Mutex<Sidecar>> = Arc::new(Mutex::new(Sidecar {
        child: None,
        #[cfg(windows)]
        _job: None,
    }));
    let sidecar_for_setup = sidecar.clone();
    let sidecar_for_exit = sidecar.clone();

    tauri::Builder::default()
        // The updater is deliberately configured only in the release overlay:
        // local development builds must never poll a public release channel.
        // Its webview permissions are restricted to the main window in
        // capabilities/default.json.
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .invoke_handler(tauri::generate_handler![save_text_file, save_url_file])
        .setup(move |app| {
            let handle = app.handle().clone();

            // The main window is created here rather than from config
            // (`"create": false` in tauri.conf.json) so the webview profile
            // can be placed inside the data directory.
            let win_cfg = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .expect("main window missing from tauri.conf.json")
                .clone();
            let mut wb = WebviewWindowBuilder::from_config(&handle, &win_cfg)?;
            if let Some(profile) = &webview_profile {
                wb = wb.data_directory(profile.clone());
            }
            wb.build()?;

            // An unwritable data directory is a hard, named error (owner
            // ruling: no silent fallback to %LOCALAPPDATA%). The window above
            // exists only to display it.
            if let Err(msg) = &storage {
                let h = handle.clone();
                let msg = msg.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(500));
                    fail(&h, &msg);
                });
                return Ok(());
            }

            // Dev loop: point the window at an already-running checkout server
            // (`quantem serve` / `npm run dev`) and spawn nothing.
            if let Ok(dev_url) = env::var("QUANTEM_DEV_SERVER_URL") {
                let js = format!("window.location.replace({})", serde_json_escape(&dev_url));
                let h = handle.clone();
                std::thread::spawn(move || {
                    // Give the webview a beat to finish loading the page.
                    std::thread::sleep(Duration::from_millis(300));
                    eval_on_main(&h, js);
                });
                return Ok(());
            }

            let Some(server_exe) = find_sidecar(&handle) else {
                let h = handle.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(500));
                    fail(
                        &h,
                        "quantem-server was not found next to the application.\n\
                         Expected quantem-server\\quantem-server.exe beside QuantEM.exe \
                         (or set QUANTEM_SERVER_EXE).",
                    );
                });
                return Ok(());
            };

            let port = free_port().unwrap_or(8722);
            log_line(&format!(
                "[shell] launching {} serve --port {port}",
                server_exe.display()
            ));

            let mut cmd = Command::new(&server_exe);
            cmd.arg("serve")
                .arg("--port")
                .arg(port.to_string())
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            #[cfg(windows)]
            {
                use std::os::windows::process::CommandExt;
                const CREATE_NO_WINDOW: u32 = 0x0800_0000;
                cmd.creation_flags(CREATE_NO_WINDOW);
            }

            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    let h = handle.clone();
                    let msg = format!("failed to launch {}: {e}", server_exe.display());
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(500));
                        fail(&h, &msg);
                    });
                    return Ok(());
                }
            };

            // Sweep the whole tree (server + spawned job workers) on exit.
            #[cfg(windows)]
            let job = {
                let job = job::Job::new();
                match &job {
                    Some(j) => {
                        if !j.assign(&child) {
                            log_line("[shell] WARNING: could not assign sidecar to job object");
                        }
                    }
                    None => log_line("[shell] WARNING: could not create job object"),
                }
                job
            };

            // Mirror sidecar output into the shell log.
            for (name, stream) in [
                ("stdout", child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)),
                ("stderr", child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>)),
            ] {
                if let Some(stream) = stream {
                    std::thread::spawn(move || {
                        let reader = BufReader::new(stream);
                        for line in reader.lines().map_while(Result::ok) {
                            log_line(&format!("[server {name}] {line}"));
                        }
                    });
                }
            }

            {
                let mut guard = sidecar_for_setup.lock().unwrap();
                guard.child = Some(child);
                #[cfg(windows)]
                {
                    guard._job = job;
                }
            }

            // Wait for the port on a worker thread, then swap the page over.
            let h = handle.clone();
            let sc = sidecar_for_setup.clone();
            std::thread::spawn(move || {
                let started = Instant::now();
                let url = format!("http://127.0.0.1:{port}/");
                loop {
                    if port_open(port) {
                        log_line(&format!("[shell] server is up at {url}"));
                        eval_on_main(&h, format!("window.location.replace({})", serde_json_escape(&url)));
                        return;
                    }
                    // A sidecar that died before opening the port will never
                    // come up; surface that instead of spinning to the timeout.
                    {
                        let mut guard = sc.lock().unwrap();
                        if let Some(child) = guard.child.as_mut() {
                            if let Ok(Some(status)) = child.try_wait() {
                                fail(
                                    &h,
                                    &format!(
                                        "the QuantEM server exited during startup (status {status}).\n\
                                         If a database migration failed, its pre-migration recovery snapshot was retained under backups\\pre-migration.\n\
                                         See logs\\quantem-desktop.log in the QuantEM data directory."
                                    ),
                                );
                                return;
                            }
                        }
                    }
                    if started.elapsed() > STARTUP_TIMEOUT {
                        fail(
                            &h,
                            "the QuantEM server did not start within five minutes.\n\
                             See logs\\quantem-desktop.log in the QuantEM data directory.",
                        );
                        return;
                    }
                    if started.elapsed() > Duration::from_secs(20) {
                        set_status(&h, "Still starting… the first launch migrates the database.");
                    }
                    std::thread::sleep(Duration::from_millis(300));
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the QuantEM shell")
        .run(move |_app, event| {
            if let RunEvent::Exit = event {
                sidecar_for_exit.lock().unwrap().kill();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn export_urls_are_limited_to_supported_loopback_endpoints() {
        assert!(validated_export_url(
            "http://127.0.0.1:8722/api/assets/asset-1/export-png/?source=original"
        )
        .is_ok());
        assert!(validated_export_url(
            "http://localhost:45174/api/analysis/run-1/export/objects.csv"
        )
        .is_ok());
        assert!(validated_export_url(
            "http://127.0.0.1:8722/api/analysis/run-1/export/image_summary.csv"
        )
        .is_ok());

        for invalid in [
            "https://127.0.0.1:8722/api/analysis/run-1/export/objects.csv",
            "http://example.com/api/analysis/run-1/export/objects.csv",
            "http://127.0.0.1:8722/api/analysis/run-1/",
            "http://127.0.0.1:8722/api/analysis/run-1/export/unknown.zip",
            "http://user@127.0.0.1:8722/api/analysis/run-1/export/objects.csv",
        ] {
            assert!(validated_export_url(invalid).is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn completed_save_atomically_replaces_an_existing_file() {
        let directory = tempfile::tempdir().expect("test directory");
        let destination = directory.path().join("objects.csv");
        std::fs::write(&destination, b"old contents").expect("seed destination");
        let mut source = Cursor::new(b"object_id,area\n1,42\n".to_vec());

        stream_to_file_atomically(&destination, &mut source).expect("atomic save");

        assert_eq!(
            std::fs::read(&destination).expect("saved file"),
            b"object_id,area\n1,42\n"
        );
        assert_eq!(
            std::fs::read_dir(directory.path())
                .expect("directory listing")
                .count(),
            1,
            "the temporary file should be consumed"
        );
    }

    #[test]
    fn server_export_is_streamed_into_the_destination() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("loopback listener");
        let port = listener.local_addr().expect("listener address").port();
        let body = b"object_id,area\n1,42\n";
        let server = std::thread::spawn(move || {
            let (mut connection, _) = listener.accept().expect("export request");
            let mut request = [0_u8; 2048];
            let _ = connection.read(&mut request);
            write!(
                connection,
                "HTTP/1.1 200 OK\r\nContent-Type: text/csv\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .expect("response headers");
            connection.write_all(body).expect("response body");
        });

        let directory = tempfile::tempdir().expect("test directory");
        let destination = directory.path().join("objects.csv");
        let url = validated_export_url(&format!(
            "http://127.0.0.1:{port}/api/analysis/run-1/export/objects.csv"
        ))
        .expect("valid export URL");
        download_export(&destination, url).expect("streamed export");
        server.join().expect("server thread");

        assert_eq!(std::fs::read(destination).expect("saved export"), body);
    }

    #[test]
    fn suggested_name_cannot_escape_the_dialog_default() {
        assert_eq!(
            safe_suggested_name("../../exports/objects.csv"),
            "objects.csv"
        );
        assert_eq!(
            safe_suggested_name(r"C:\\exports\\composition.csv"),
            "composition.csv"
        );
    }
}
