# QuantEM desktop shell

The Windows desktop channel for QuantEM. One double-clickable artifact that
opens the QuantEM window with nothing pre-installed — model weights are pulled
from Hugging Face at runtime (or installed from a release bundle with
`quantem models install`); they are **not** part of this distribution.

## Updating an installed copy

Updater-enabled releases check QuantEM's stable GitHub Release channel from
inside the installed desktop app. When a signed newer build is available, a
top banner offers **Update**, shows download progress, then waits for queued or
running tasks and any unsaved annotation draft to clear before restarting.
Application updates replace the shell and server only: the data directory
below, including images, segmentations, objects, analysis results, and model
cache, stays in place. The first updater-enabled version is installed normally;
later versions update from the banner. Release-key setup and the atomic publish
workflow are documented in [BUNDLING.md](BUNDLING.md).

Windows has one installer, not separate CPU and CUDA installers. It detects a
compatible NVIDIA driver, preselects CUDA when appropriate, allows an explicit
choice, and downloads the matching frozen runtime during installation. Later
automatic updates replace the embedded application layer while retaining a
byte-compatible Python/PyTorch/CUDA runtime. A large runtime download happens
again only when that layer actually changes or cannot be verified.

## Architecture

Three pieces, one process tree:

```
QuantEM.exe                  Tauri v2 shell (Rust, ~10 MB)
 └─ quantem-server.exe       PyInstaller onedir build of the `quantem` package
     └─ job workers          multiprocessing children the server spawns per job
```

* **The shell** (`src-tauri/`) resolves the data directory (below), picks a
  free loopback port, spawns `quantem-server serve --port N`, shows
  `ui/index.html` (a loading page) until the port answers, then navigates
  the webview to `http://127.0.0.1:N/`. It is deliberately dumb: no IPC, no
  state, no business logic. The UI it hosts is the same Vite bundle the pip
  channel serves.
* **The sidecar** (`../packaging/pyinstaller/`) is the entire `quantem`
  Python package — Django + DRF + torch + the segmentation stack — frozen
  with PyInstaller into a self-contained directory. It serves both the API
  and the built frontend (shipped inside it as data; see
  `quantem_server_settings.py`).
* **Models** come from the registry at runtime (Hugging Face download,
  `quantem models install <bundle>`, or the install-time selection below),
  land in the data directory, and are shared by every channel. Nothing
  model-sized ships in the installer.

### Where everything is stored (owner ruling 2026-08-09)

All application storage — the sqlite DB, models, HF cache, exports, logs,
and the WebView2 profile — lives **with the installation**:

* `QUANTEM_DATA_DIR`, if set, is the one explicit override (absolute path).
* Otherwise the shell uses `<exe dir>\data` — i.e. `<install>\data` for both
  the NSIS install and the unzipped portable build — and exports it as
  `QUANTEM_DATA_DIR` so the sidecar and its job workers inherit the same
  choice. The frozen sidecar computes the identical default on its own
  (`quantem.cli.default_data_dir`), so the two agree even when the sidecar
  is launched alone.
* The WebView2 profile goes to `<data dir>\webview-profile`: the shell both
  sets `WEBVIEW2_USER_DATA_FOLDER` and passes the directory explicitly to
  the webview builder (without the explicit pass, tauri forces the profile
  into `%LOCALAPPDATA%\<identifier>`).
* There is **no silent fallback**: if the computed location is not
  writable, the shell shows an error that names `QUANTEM_DATA_DIR` instead
  of relocating the data.

### Install-time model selection

After its hardware-acceleration page, the NSIS installer shows a "Model
downloads" page (eight packs, the four
OmniEM ones pre-checked, honest download sizes) and records the choice in
`<install>\data\pending-model-installs.json` (`{"packs": ["omniem:mito",
...]}`). The installer itself downloads nothing: on first launch the server
reads that file, queues the installs through the normal verified download
machinery (digest checks, progress, cancel, AV-retry), and deletes it. See
`src-tauri/nsis/hooks.nsh` and [BUNDLING.md](BUNDLING.md).

### Sidecar layout

Tauri's `externalBin` copies a *single file* next to the shell executable.
The server is a PyInstaller **onedir** build — an exe plus an `_internal/`
tree — because onefile would unpack multi-GB of torch to temp on every
launch. On macOS the whole directory ships via the platform-specific
`bundle.resources` mapping. On Windows, the bootstrap installer composes the
selected CPU/CUDA runtime and application layers into the same
`quantem-server/` layout; the portable builder stages it there directly. The
shell resolves the sidecar itself, in
order:

1. `QUANTEM_SERVER_EXE` (env override, used in dev),
2. `<exe dir>\quantem-server\quantem-server.exe` (portable zip and NSIS both
   produce this layout),
3. `<resource dir>\quantem-server\quantem-server.exe`.

### Process lifetime

The sidecar goes into a Windows Job Object with
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` immediately after spawn. When the shell
exits — cleanly, killed from Task Manager, or crashed — the OS terminates
every process in the job, including the multiprocessing job workers the
server spawns (they inherit the job). `RunEvent::Exit` additionally does an
explicit `kill()`. Verify after closing the window: no `quantem-server.exe`
or stray workers in `Get-Process`.

Sidecar stdout/stderr and shell events are appended to
`<data dir>\logs\quantem-desktop.log` (the temp dir only while the data
directory is unknown or unwritable). The server writes its own rotating
`logs\quantem-server.log` beside it.

## Dev loop

Run the server from the checkout (any of the usual ways), then point the
shell at it — the shell spawns nothing in this mode:

```powershell
# terminal 1: the server (checkout venv)
$env:QUANTEM_DATA_DIR = "D:\somewhere\devdata"
python -m quantem.cli serve --port 45174

# terminal 2: the shell
cd desktop
$env:QUANTEM_DEV_SERVER_URL = "http://127.0.0.1:45174/"
npm run dev
```

45174 is the port the dev SPA falls back to (`frontend/src/shared/api/core/http.ts`),
paired with the dev SPA's own 45173 (`frontend/vite.config.ts`). Neither is the
conventional 5173/8000 on purpose — those collide with whatever else on the
machine claimed them first, and some dev scripts free them by killing the
holder. Pass `--port` and set `VITE_API_BASE_URL` to use anything else.

Sharing the machine with another heavy job? The server sizes its worker pools
from total RAM and core count and otherwise assumes it owns the box, so two
such jobs oversubscribe it. `QUANTEM_MACHINE_PROFILE=standard` (or `small`)
forces a smaller row — 2 heavy slots and 4 raster workers instead of 4 and 8 —
and leaves the rest of the machine alone.

To exercise the real spawn path in dev, unset `QUANTEM_DEV_SERVER_URL` and
set `QUANTEM_SERVER_EXE` to a built sidecar exe instead.

## Building (this machine: everything off C:)

The lab build machine must not write to C:. Every cache the toolchain uses
is redirected; the MSVC compiler/linker already on the machine is used
in place (read-only), and the Windows SDK headers/libs live in an
[xwin](https://github.com/Jake-Shadle/xwin)-provisioned tree on D:.

```bash
# one-time bootstrap (already done on this machine; documented for rebuilds)
export RUSTUP_HOME='<repo>\.scratch\rustup'
export CARGO_HOME='<repo>\.scratch\cargo'
./rustup-init.exe -y --no-modify-path --profile minimal \
    --default-toolchain stable-x86_64-pc-windows-msvc
xwin.exe --accept-license --arch x86_64 \
    --cache-dir '<repo>\.scratch\xwin-cache' \
    splat --disable-symlinks --output '<repo>\.scratch\xwin-sdk'
```

Every build shell sources `<repo>\.scratch\desktop_env.sh`,
which exports: `RUSTUP_HOME`, `CARGO_HOME`, `CARGO_TARGET_DIR`,
`TMP`/`TEMP`/`TMPDIR`, `PIP_CACHE_DIR`, `npm_config_cache`,
`PYINSTALLER_CONFIG_DIR` (all under `<repo>\.scratch\`), plus
`LIB`/`INCLUDE` pointing at MSVC 14.28 (BuildTools 2019) and the xwin SDK,
and puts the MSVC `Hostx64\x64` bin dir on `PATH`.

Then, in order:

```powershell
# 1. frontend (once per UI change)
cd frontend; npm run build

# 2. server sidecar -> packaging\pyinstaller\dist\quantem-server
powershell -ExecutionPolicy Bypass -File packaging\pyinstaller\build.ps1

# 3. shell -> <CARGO_TARGET_DIR>\release\QuantEM.exe
cd desktop; npm install; npm run build:exe     # tauri build --no-bundle

# 4. portable zip -> desktop\dist-portable\QuantEM-portable-win64.zip
powershell -ExecutionPolicy Bypass -File packaging\make_portable.ps1
```

For the NSIS installer (`npm run build` without `--no-bundle`) see
[BUNDLING.md](BUNDLING.md) — including where the updater keypair and code
signing slot in before release.
