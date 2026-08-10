# Bundling the desktop channel

Two shippable artifacts, one build machine constraint (nothing may be
written to C:):

| Artifact | Command | Status on this machine |
| --- | --- | --- |
| Portable zip | `packaging\make_portable.ps1` | **Built here.** No installer tooling involved. |
| NSIS installer | `npm run build` in `desktop/` | Buildable here — see the NSIS cache note below. |

## The NSIS download hazard, and its override

`tauri-bundler` provisions NSIS by downloading it into the platform cache
directory — `dirs::cache_dir()`, which on Windows is `%LOCALAPPDATA%`
(C:), resolved through the known-folder API that environment variables do
**not** redirect.

The override exists and is config-level:
[`bundle.useLocalToolsDir: true`](https://tauri.app/reference/config/#uselocaltoolsdir)
(set in our `tauri.conf.json`) makes the bundler cache its tools in
`<cargo target dir>/.tauri/` instead. With `CARGO_TARGET_DIR` pointed at
`<repo>\.scratch\cargo-target`, the NSIS toolset lands on D:
and C: stays untouched. (Verified in the tauri-utils 2.9.3 source:
`use_local_tools_dir` switches `tauri_tools_path` from `dirs::cache_dir()`
to the project target directory; the WiX path is irrelevant — our only
bundle target is `nsis`.)

The bundler still *downloads* NSIS from the Tauri mirrors on first run. If
a fully offline build is ever needed, pre-seed
`<CARGO_TARGET_DIR>\.tauri\NSIS\` from a portable NSIS distribution and the
bundler will use it as-is (it checks for `makensis.exe` there before
downloading).

## The install flow (owner rulings 2026-08-09)

The installer runs in **current-user mode** (`installMode: "currentUser"` in
`tauri.conf.json` → `RequestExecutionLevel user`, no elevation, HKCU
registry) and **always shows the directory chooser** (`MUI_PAGE_DIRECTORY`
is unconditional in the Tauri template outside passive `/P` mode; the
default offered is `%LOCALAPPDATA%\QuantEM`, and whatever the user picks
becomes `$INSTDIR`). Nothing about the drive or folder is assumed.

Right after the directory page comes a **"Model downloads" page**: eight
checkboxes (OmniEM Mitochondria/ER/Nucleus/Lipid droplets checked by
default, the four QuantEM packs unchecked), real Hugging Face sizes in the
labels, and a running total that counts each family's shared encoder once —
the numbers come from `MEASURED_SIZES` in `src/quantem/registry/manifest.py`.
On install the selection is written to
`<INSTDIR>\data\pending-model-installs.json` as
`{"packs": ["omniem:mito", ...]}`. The installer downloads **nothing**; on
first launch the server consumes that file and queues the installs through
its verified download machinery (digest checks, progress, cancel, AV-retry),
then deletes it. Passive/update installs skip the page and leave any
existing selection untouched.

Uninstall: the app's data lives in `<INSTDIR>\data` (see the storage section
in [README.md](README.md)), so the uninstaller's delete-data checkbox
removes that tree too (hook `NSIS_HOOK_POSTUNINSTALL`); without the checkbox
the data survives the uninstall. The checkbox copy names what goes — the
`data\` folder with imported images, proofreading work, analyses, exports and
downloaded models (can be several GB) — and the directory-chooser page warns
that everything, multi-GB models included, lands under the chosen folder
(both texts live in `nsis/hooks.nsh`).

### The vendored NSIS template

The model page must appear *between* the directory and start-menu pages, but
NSIS page order is fixed at compile time and the `installerHooks` file is
included before any page is declared. So `src-tauri/nsis/installer.nsi` is a
**vendored copy of the template embedded in `@tauri-apps/cli` 2.11.4** with
exactly two fenced insertions (marked `QuantEM insertion`): one expands
`QUANTEM_MODEL_PAGE_FUNCTIONS` and declares `Page custom` after
`MUI_PAGE_DIRECTORY`; the other swaps the uninstaller's stock
`$(deleteAppData)` checkbox text for `QUANTEM_DELETE_DATA_TEXT` (the stock
LangString cannot be overridden from the hooks file — the bundler's language
files define it after the hooks are included) and makes the control tall
enough for the wrapped copy. All actual logic and copy — the page, the
running total, the pending-file writer, both hook macros, the directory-page
storage sentence (`MUI_DIRECTORYPAGE_TEXT_TOP`) and the checkbox text — lives
in `src-tauri/nsis/hooks.nsh`.

**When upgrading `@tauri-apps/cli`**: re-extract the embedded template
(it is the `installer.nsi` blob inside
`node_modules/@tauri-apps/cli-win32-x64-msvc/cli.win32-x64-msvc.node`, or
take `crates/tauri-bundler/src/bundle/windows/nsis/installer.nsi` from the
matching tauri tag), re-apply the fenced blocks, and rebuild. A stale
template compiles against a newer bundler only by luck — treat the pair as
pinned together.

## Layout produced

Both the portable zip and the NSIS install place the sidecar the same way
(`[[bin]] name = "QuantEM"` in `Cargo.toml` names the shell executable;
the data directory appears on first run, or `data\
pending-model-installs.json` first if models were selected at install):

```
QuantEM.exe
quantem-server\
    quantem-server.exe
    _internal\...
data\                       (created at install/run time -- DB, models,
    ...                      HF cache, exports, logs, webview-profile)
```

The installer gets the sidecar tree via `bundle.resources` (the map in
`tauri.conf.json` copies `packaging/pyinstaller/dist/quantem-server` to
`quantem-server/`); the portable zip is assembled by
`packaging\make_portable.ps1`. When first building the NSIS target, verify
the installed tree matches the layout above before shipping.

## Pre-release checklist items deliberately NOT done here

Per the owner's ruling (G), these are release-time steps and no keys or
signatures exist in this repo:

1. **Updater keypair** — when the update channel is turned on, generate the
   keypair with `npx tauri signer generate` *outside the repo*, put the
   public key in `tauri.conf.json` under `plugins.updater.pubkey`, add the
   `tauri-plugin-updater` dependency, and inject the private key into the
   release pipeline only (`TAURI_SIGNING_PRIVATE_KEY`,
   `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`). Nothing in the current config
   references an updater endpoint.
2. **Code signing** (Authenticode or ad-hoc) — unsigned binaries trip
   SmartScreen. Signing happens on the release machine over the finished
   artifacts (`QuantEM.exe`, `quantem-server.exe`, the NSIS exe);
   `bundle.windows.certificateThumbprint` / `signCommand` are the Tauri
   hooks when that lands. The lab build machine holds no certificates.

## Rebuilding elsewhere (a normal Windows box)

On a machine without the C:-prohibition, none of the redirects are needed:

```powershell
cd frontend; npm ci; npm run build
cd ..; pip install pyinstaller pyinstaller-hooks-contrib
powershell -File packaging\pyinstaller\build.ps1 -Python (Get-Command python).Source -Scratch $env:TEMP\quantem-build
cd desktop; npm ci; npm run build        # produces the NSIS installer under src-tauri\target\release\bundle\nsis\
```

Prerequisites there: Rust (MSVC toolchain), VS Build Tools + Windows 10/11
SDK, Node 20+, Python 3.12/3.13 with the `quantem` package's deps installed.

## Publishing a GitHub Release

Release assets are **never committed to git** — `dist/`, `dist-portable/` and
the cargo target directory are all ignored; the artifacts exist only as files
uploaded to the release. One release per version tag carries every platform's
assets (Windows now; macOS artifacts join the same tag when that lane lands).

The upload set for a version is exactly five files:

| Asset | Built by |
| --- | --- |
| `QuantEM_<version>_x64-setup.exe` | `npm run build` in `desktop/` (NSIS bundle) |
| `QuantEM-portable-win64.zip` | `packaging\make_portable.ps1` |
| `quantem-<version>-py3-none-any.whl` | `python -m build` (hatchling) |
| `quantem-<version>.tar.gz` | same sdist build |
| `SHA256SUMS.txt` | generated over the four files above, exact filenames |

```powershell
gh release create v<version> `
  QuantEM_<version>_x64-setup.exe QuantEM-portable-win64.zip `
  quantem-<version>-py3-none-any.whl quantem-<version>.tar.gz SHA256SUMS.txt `
  --title "QuantEM <version>" --notes-file notes.md
```

Release notes should carry the SmartScreen paragraph (the binaries are
unsigned: "More info" → "Run anyway", verify against `SHA256SUMS.txt` with
`certutil -hashfile <file> SHA256`). Pip users can install the wheel URL
directly; the sdist is the from-source fallback.
