# Release and self-update procedure

QuantEM desktop updates are signed application packages. On macOS they carry
the frozen server. Windows ships one small bootstrap installer plus CPU and
CUDA runtime payloads; the installer downloads the selected payload itself and
does not launch a second installer. No package contains model weights or user
data. The existing data root remains in place:

- Windows: the installer-selected `<install>/data` directory, unless
  `QUANTEM_DATA_DIR` explicitly selects another absolute directory.
- macOS: `~/Library/Application Support/QuantEM`, unless overridden.

That root contains the SQLite database, images, segmentations, annotations,
analysis outputs, models, caches, logs, and WebView profile. The NSIS update
path runs the old uninstaller with `/UPDATE`; its hooks deliberately skip all
data removal in that mode.

## One-time release setup

Generate one Ed25519 updater key pair with Tauri's signer. Store the private
key and its password outside the repository in the protected GitHub Actions
environment `quantem-desktop-release`, with an owner-controlled offline
recovery copy. Losing that key prevents future versions from being accepted by
already-installed apps.

Configure the protected environment with:

- Secret `TAURI_SIGNING_PRIVATE_KEY` and optional
  `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
- Variable `TAURI_UPDATER_PUBKEY` containing the matching public key.

The updater key is free and is not an Apple or Microsoft code-signing
certificate. Generate and configure it from PowerShell without putting the
private key on C: or in Git:

```powershell
. D:\Chris\QuantEM_repo\.scratch\quantem_env.ps1
$KeyDir = "D:\Chris\QuantEM_repo\.scratch\tauri-updater-key"
$Key = "$KeyDir\quantem-app.key"
New-Item -ItemType Directory -Force $KeyDir | Out-Null
Set-Location "D:\Chris\QuantEM_repo\quantem_app\desktop"
& $Npx tauri signer generate -w $Key
```

The signer asks for a password during key generation; enter the same password
in GitHub. In the repository, open **Settings > Secrets and variables >
Actions** and create these repository-level entries:

1. On **Secrets**, create `TAURI_SIGNING_PRIVATE_KEY`. Copy its value without
   printing it into shell history with `Get-Content $Key -Raw | Set-Clipboard`.
2. On **Variables**, create `TAURI_UPDATER_PUBKEY`. Copy its value with
   `Get-Content "$Key.pub" -Raw | Set-Clipboard`.
3. Only if a password was entered in the signer, create the Secret
   `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` with that password. Leave it absent for
   a passwordless key; the workflow passes the required empty value itself.

Keep an offline recovery copy of both key files and the password.

No paid platform certificate is required by this workflow. macOS applications
are ad-hoc signed, not notarized, and Windows executables are not Authenticode
signed. Those packages work, but users must approve the normal unidentified
developer/Gatekeeper or SmartScreen warning. Paid certificates can be added
later without changing the updater key.

The private updater key is never committed or written to an application asset.
The public key is injected into a temporary Tauri configuration by the release
workflow. This avoids shipping a placeholder or a disposable development key.

## One Windows installer, two runtimes

The release workflow freezes PyTorch 2.13 independently from the `cpu` and
`cu126` indexes. Each validated `quantem-server/` directory is placed in a
deterministic ZIP, signed with the Tauri release key, and described by a JSON
manifest. GitHub limits each release asset to 2 GiB, so an oversized archive is
split into sub-2-GB parts automatically; the full archive signature and digest
still cover the reconstructed byte stream.

The signed NSIS bootstrap embeds the exact part URLs and SHA-256 digests. It:

1. checks the installed NVIDIA CUDA driver API and preselects CUDA 12.6 when
   compatible, while retaining a visible CPU/CUDA choice;
2. downloads every part under `<install>/.quantem-install`, so temporary space
   is consumed on the drive the user selected;
3. verifies each part and the reconstructed archive, extracts to a staging
   directory, and transactionally replaces `<install>/quantem-server`; and
4. records the choice in both installer metadata and
   `<install>/.quantem-runtime-variant`.

Automatic updates read that marker and install the same flavor. Existing
installations from before this scheme have no marker and migrate safely to CPU.
The release always exposes one `QuantEM_<version>_x64-setup.exe` to users; the
runtime ZIP parts are implementation assets consumed by that installer.

## Publishing a release

1. Update all five version sources and push a matching `v<version>` tag.
2. The `quantem-app-desktop-release` workflow builds both Windows runtime
   payloads, the single Windows x64 installer, and native macOS x64/Apple
   Silicon bundles. It signs updater and runtime archives, ad-hoc signs the
   macOS applications, exercises a real local-mirror Windows install, and
   creates a draft GitHub release.
3. The workflow uploads the installers, hashes, signed updater packages, and
   `quantem-app-latest.json` before publishing the release. Apps check that
   JSON file, so they never see a partially populated release.
4. Download and install the normal installer for the first updater-enabled
   version. Later versions are offered from QuantEM's top update banner.

Model packs are intentionally independent of this process. Their registry and
cache update on their own schedule and are never replaced by an application
installer.

## Migration recovery

Before a launch applies pending Django migrations, QuantEM creates a consistent
SQLite snapshot at `backups/pre-migration/` under the active data root. The
snapshot manifest records the target version, pending migrations, and SHA-256.
Only the three newest migration snapshots are retained; images and models are
not copied or deleted. If migration fails, the application stops and the log
names the snapshot left for recovery.
