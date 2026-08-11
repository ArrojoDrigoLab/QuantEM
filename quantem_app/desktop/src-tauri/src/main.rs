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
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tauri::{Manager, RunEvent, WebviewWindowBuilder};

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
///    NSIS `bundle.resources` mapping both produce this);
/// 3. the Tauri resource dir, for any packaging that separates resources.
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
        let exe = env::current_exe()
            .map_err(|e| format!("cannot locate the QuantEM executable: {e}"))?;
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
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
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

fn main() {
    // Resolve storage before anything else: the WebView2 profile location must
    // be decided before the webview exists, and the sidecar inherits the same
    // choice through the environment.
    let storage: Result<PathBuf, String> = resolve_data_dir().and_then(|(dir, explicit)| {
        match ensure_writable(&dir) {
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
        }
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
