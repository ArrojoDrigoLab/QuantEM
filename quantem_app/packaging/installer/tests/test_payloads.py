from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

INSTALLER_SCRIPTS = Path(__file__).resolve().parents[1]
REPO_ROOT = INSTALLER_SCRIPTS.parents[2]
sys.path.insert(0, str(INSTALLER_SCRIPTS))

from build_payload_manifest import build_manifests  # noqa: E402
from build_windows_payload import build_payload  # noqa: E402
from split_windows_payload import split_payload  # noqa: E402


def test_windows_workflows_read_bundles_from_the_explicit_cargo_target() -> None:
    expected = "desktop/src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis"
    wrong_native_path = "desktop/src-tauri/target/release/bundle/nsis"
    workflows = {
        ".github/workflows/quantem-app.yml": 1,
        ".github/workflows/quantem-app-desktop-release.yml": 2,
    }
    for relative_path, expected_count in workflows.items():
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert wrong_native_path not in text
        assert text.count(expected) == expected_count


def test_windows_workflows_rebuild_and_wait_for_the_installer_mirror() -> None:
    workflows = (
        ".github/workflows/quantem-app.yml",
        ".github/workflows/quantem-app-desktop-release.yml",
    )
    for relative_path in workflows:
        text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "Remove-Item -Recurse -Force" in text
        assert "Local payload mirror did not become ready" in text
        assert "payload-server.stderr.log" in text
        assert ".quantem-install/failure.log" in text


def test_windows_workflows_retry_transient_tauri_tool_downloads() -> None:
    workflows = (
        REPO_ROOT / ".github/workflows/quantem-app.yml",
        REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml",
    )
    assert (
        sum(
            workflow.read_text(encoding="utf-8").count("scripts/invoke_tauri_build.ps1")
            for workflow in workflows
        )
        == 3
    )

    retry_script = (REPO_ROOT / "quantem_app/desktop/scripts/invoke_tauri_build.ps1").read_text(
        encoding="utf-8"
    )
    assert "[int] $Attempts = 3" in retry_script
    assert "for ($attempt = 1; $attempt -le $Attempts; $attempt++)" in retry_script
    assert "Start-Sleep -Seconds $delay" in retry_script
    assert "Tauri bundling failed after $Attempts attempts" in retry_script


def test_release_uses_tauris_signed_installer_as_the_windows_update() -> None:
    workflow = (REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml").read_text(
        encoding="utf-8"
    )
    collect_step = workflow.split("- name: Collect release installer and updater", 1)[1].split(
        "- name: Smoke-test a silent CPU install", 1
    )[0]
    assert 'Test-Path "$($installer.FullName).sig"' in collect_step
    assert "update_filename = $installerName" in collect_step
    assert 'signature_filename = "$installerName.sig"' in collect_step
    assert "*.nsis.zip" not in collect_step


def test_release_uses_supported_macos_build_toolchains() -> None:
    workflow = (REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml").read_text(
        encoding="utf-8"
    )
    assert "os: macos-15-intel" in workflow
    assert "macos-13" not in workflow
    assert 'python: "3.12"' in workflow
    assert 'torch: "2.2.2"' in workflow
    assert 'torchvision: "0.17.2"' in workflow
    assert "python-version: ${{ matrix.python }}" in workflow
    assert '"torch==${{ matrix.torch }}"' in workflow
    assert '"torchvision==${{ matrix.torchvision }}"' in workflow


def test_intel_macos_dmg_avoids_interactive_finder_automation() -> None:
    workflow = (REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml").read_text(
        encoding="utf-8"
    )
    assert '[[ "${{ matrix.platform }}" == "darwin-x86_64" ]]' in workflow
    assert "--bundles app --config" in workflow
    assert "bash scripts/build_macos_dmg.sh" in workflow

    dmg_script = (REPO_ROOT / "quantem_app/desktop/scripts/build_macos_dmg.sh").read_text(
        encoding="utf-8"
    )
    assert "hdiutil create" in dmg_script
    assert "hdiutil verify" in dmg_script
    assert "for attempt in 1 2 3" in dmg_script
    assert "osascript" not in dmg_script


def test_release_artifacts_are_short_lived_and_deleted_after_publication() -> None:
    release_workflow = (REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml").read_text(
        encoding="utf-8"
    )
    routine_workflow = (REPO_ROOT / ".github/workflows/quantem-app.yml").read_text(encoding="utf-8")

    assert release_workflow.count("retention-days: 1") == 3
    assert "retention-days: 7" not in release_workflow
    assert "actions: write" in release_workflow
    assert "Delete temporary workflow artifacts after publication" in release_workflow
    assert "/actions/runs/${GITHUB_RUN_ID}/artifacts" in release_workflow
    assert "/actions/artifacts/${artifact_id}" in release_workflow
    assert routine_workflow.count("retention-days: 1") == 1
    assert "name: QuantEM-macos-arm64" not in routine_workflow
    assert "name: clean temporary artifacts" in routine_workflow
    assert "Delete this run's temporary artifacts" in routine_workflow
    cleanup_step = routine_workflow.split("Delete this run's temporary artifacts", 1)[1]
    assert "working-directory: ." in cleanup_step


def test_release_notes_put_the_three_installers_first() -> None:
    script = REPO_ROOT / "quantem_app/desktop/scripts/build_release_notes.py"
    spec = importlib.util.spec_from_file_location("build_release_notes", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    notes = module.build_release_notes(
        version="0.1.1",
        tag="v0.1.1",
        generated_notes="**Full Changelog**: example",
    )

    assert notes.startswith("## Install QuantEM")
    assert "QuantEM_0.1.1_x64-setup.exe" in notes
    assert "QuantEM_0.1.1_darwin-aarch64.dmg" in notes
    assert "QuantEM_0.1.1_darwin-x86_64.dmg" in notes
    assert notes.index("## Install QuantEM") < notes.index("## Changes")
    assert "### [Download" not in notes
    assert "The remaining files" not in notes
    assert "—" not in notes
    assert "**Full Changelog**: example" in notes


def test_release_workflow_does_not_publish_build_only_manifests() -> None:
    workflow = (REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml").read_text(
        encoding="utf-8"
    )
    publish_step = workflow.split("- name: Publish only after every platform", 1)[1].split(
        "- name: Delete temporary workflow artifacts", 1
    )[0]

    assert "build_release_notes.py" in publish_step
    assert '--notes-file "$RUNNER_TEMP/public-release-notes.md"' in publish_step
    assert "rm -f ../release-assets/*.update-manifest.json" in publish_step
    assert "../release-assets/quantem-app-windows-payloads.json" in publish_step
    assert 'label_asset "QuantEM_${version}_x64-setup.exe" "Windows installer"' in publish_step
    assert "macOS installer (Apple silicon)" in publish_step
    assert "macOS installer (Intel)" in publish_step
    assert "—" not in publish_step


def test_release_workflow_captures_created_release_without_a_list_race() -> None:
    workflow = (REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml").read_text(
        encoding="utf-8"
    )
    publish_step = workflow.split("- name: Publish only after every platform", 1)[1].split(
        "- name: Delete temporary workflow artifacts", 1
    )[0]

    assert "releases/tags/${RELEASE_TAG}" in publish_step
    assert '--method POST "repos/${GITHUB_REPOSITORY}/releases"' in publish_step
    assert '> "$release_record"' in publish_step
    assert "releases?per_page=100" not in publish_step
    assert 'gh release create "$RELEASE_TAG"' not in publish_step
    assert 'gh release view "$RELEASE_TAG"' not in publish_step


def test_package_numpy_floor_supports_the_intel_macos_torch_build() -> None:
    pyproject = (REPO_ROOT / "quantem_app/pyproject.toml").read_text(encoding="utf-8")
    assert '"numpy>=1.26"' in pyproject
    assert '"numpy>=2.0"' not in pyproject


def test_installer_hash_verification_uses_the_base_windows_dotnet_api() -> None:
    hooks = (REPO_ROOT / "quantem_app/desktop/src-tauri/nsis/hooks.nsh").read_text(encoding="utf-8")
    assert "Security.Cryptography.SHA256" in hooks
    assert "Get-FileHash -LiteralPath" not in hooks


def test_windows_updates_are_layered_and_quiet() -> None:
    release_config = json.loads(
        (REPO_ROOT / "quantem_app/desktop/src-tauri/tauri.release.conf.template.json").read_text(
            encoding="utf-8"
        )
    )
    assert release_config["plugins"]["updater"]["windows"]["installMode"] == "quiet"

    workflows = (
        REPO_ROOT / ".github/workflows/quantem-app.yml",
        REPO_ROOT / ".github/workflows/quantem-app-desktop-release.yml",
    )
    for workflow in workflows:
        text = workflow.read_text(encoding="utf-8")
        assert "--application-output" in text
        assert ".quantem-runtime-id" in text
        assert 'Start-Process $installer -ArgumentList @("/S", "/UPDATE"' in text

    release_workflow = workflows[1].read_text(encoding="utf-8")
    assert "../release-assets/quantem-application_*.zip" in release_workflow


def _server_tree(root: Path, *, cuda: bool) -> Path:
    source = root / ("cuda" if cuda else "cpu") / "quantem-server"
    (source / "_internal" / "quantem").mkdir(parents=True)
    (source / "_internal" / "quantem_frontend").mkdir(parents=True)
    (source / "_internal" / "quantem_app-0.2.0.dist-info").mkdir(parents=True)
    (source / "quantem-server.exe").write_bytes(b"MZ-fake-server")
    (source / "_internal" / "library.zip").write_bytes(b"python")
    (source / "_internal" / "quantem" / "migrations.json").write_bytes(b"app-data")
    (source / "_internal" / "quantem_frontend" / "index.html").write_bytes(b"frontend")
    (source / "_internal" / "quantem_app-0.2.0.dist-info" / "METADATA").write_bytes(
        b"Version: 0.2.0"
    )
    if cuda:
        (source / "_internal" / "torch_cuda.dll").write_bytes(b"cuda")
    return source


def _make_payload(tmp_path: Path, variant: str) -> Path:
    manifest = tmp_path / variant / "payload-manifest.json"
    output = manifest.parent / f"quantem-runtime_0.2.0_windows-x64-{variant}.zip"
    application = manifest.parent / f"quantem-application_0.2.0_windows-x64-{variant}.zip"
    build_payload(
        source=_server_tree(tmp_path / "sources", cuda=variant == "cuda"),
        output=output,
        application_output=application,
        manifest_output=manifest,
        version="0.2.0",
        variant=variant,
        torch_version="2.13.0",
        cuda_runtime="cu126" if variant == "cuda" else None,
        cuda_driver_api=12060 if variant == "cuda" else None,
    )
    (manifest.parent / f"{output.name}.sig").write_text(f"signed-{variant}\n", encoding="utf-8")
    return manifest


def test_payload_archives_separate_runtime_and_application(tmp_path: Path) -> None:
    manifest_path = _make_payload(tmp_path, "cuda")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime_path = manifest_path.parent / manifest["filename"]
    application_path = manifest_path.parent / manifest["application_filename"]
    with zipfile.ZipFile(runtime_path) as archive:
        assert "quantem-server/_internal/torch_cuda.dll" in archive.namelist()
        assert "quantem-server/quantem-server.exe" not in archive.namelist()
        runtime_info = json.loads(archive.read("quantem-server/runtime-info.json"))
    with zipfile.ZipFile(application_path) as archive:
        assert "quantem-server/quantem-server.exe" in archive.namelist()
        assert "quantem-server/_internal/quantem_frontend/index.html" in archive.namelist()
        assert "quantem-server/_internal/torch_cuda.dll" not in archive.namelist()
        info = json.loads(archive.read("quantem-server/build-info.json"))
        files = json.loads(archive.read("quantem-layer/runtime-files.json"))
    assert info["variant"] == "cuda"
    assert info["cuda_driver_api"] == 12060
    assert info["runtime_id"] == runtime_info["runtime_id"] == manifest["runtime_id"]
    assert files["runtime_id"] == manifest["runtime_id"]
    assert len(manifest["sha256"]) == 64


def test_runtime_id_does_not_change_with_application_version(tmp_path: Path) -> None:
    source = _server_tree(tmp_path / "sources", cuda=False)
    runtime_ids = []
    for version in ("0.2.0", "0.2.1"):
        manifest = tmp_path / version / "payload.json"
        build_payload(
            source=source,
            output=manifest.parent / "runtime.zip",
            application_output=manifest.parent / "application.zip",
            manifest_output=manifest,
            version=version,
            variant="cpu",
            torch_version="2.13.0",
            cuda_runtime=None,
            cuda_driver_api=None,
        )
        runtime_ids.append(json.loads(manifest.read_text(encoding="utf-8"))["runtime_id"])
    assert runtime_ids[0] == runtime_ids[1]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows installer helper")
def test_existing_runtime_verifier_accepts_exact_files_and_rejects_changes(
    tmp_path: Path,
) -> None:
    manifest_path = _make_payload(tmp_path, "cpu")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    application = manifest_path.parent / payload["application_filename"]
    metadata = tmp_path / "metadata"
    with zipfile.ZipFile(application) as archive:
        archive.extract("quantem-layer/runtime-files.json", metadata)
    runtime_manifest = metadata / "quantem-layer/runtime-files.json"
    root = tmp_path / "sources/cpu/quantem-server"
    script = INSTALLER_SCRIPTS / "verify_existing_runtime.ps1"

    def verify() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-Root",
                str(root),
                "-Manifest",
                str(runtime_manifest),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert verify().returncode == 0
    (root / "_internal/library.zip").write_bytes(b"changed")
    assert verify().returncode == 1


def test_cpu_payload_rejects_cuda_libraries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpectedly contains CUDA"):
        build_payload(
            source=_server_tree(tmp_path, cuda=True),
            output=tmp_path / "bad.zip",
            application_output=tmp_path / "bad-application.zip",
            manifest_output=tmp_path / "bad.json",
            version="0.2.0",
            variant="cpu",
            torch_version="2.13.0",
            cuda_runtime=None,
            cuda_driver_api=None,
        )


def test_combined_manifest_pins_urls_hashes_and_cuda_driver_floor(tmp_path: Path) -> None:
    manifests = [_make_payload(tmp_path, "cpu"), _make_payload(tmp_path, "cuda")]
    nsis = tmp_path / "payload-manifest.nsh"
    public = tmp_path / "quantem-app-windows-payloads.json"
    result = build_manifests(
        manifests=manifests,
        tag="v0.2.0",
        nsis_output=nsis,
        json_output=public,
    )
    text = nsis.read_text(encoding="utf-8")
    assert '!define QPAYLOAD_VERSION "0.2.0"' in text
    assert '!define QPAYLOAD_CUDA_MIN_DRIVER_API "12060"' in text
    assert "quantem-runtime_0.2.0_windows-x64-cpu.zip" in text
    assert "QPAYLOAD_CPU_RUNTIME_ID" in text
    assert "QPAYLOAD_CPU_APPLICATION_PATH" in text
    assert "QPAYLOAD_RUNTIME_VERIFIER_PATH" in text
    assert result["variants"]["cuda"]["signature"] == "signed-cuda"


def test_oversized_payload_is_split_below_the_release_asset_limit(tmp_path: Path) -> None:
    manifest_path = _make_payload(tmp_path, "cpu")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest_path.parent / manifest["filename"]
    part_size = (archive.stat().st_size + 2) // 3
    parts = split_payload(
        archive=archive,
        manifest_path=manifest_path,
        max_part_bytes=part_size,
        remove_archive=True,
    )
    assert len(parts) > 1
    assert not archive.exists()
    assert all(
        (manifest_path.parent / str(part["filename"])).stat().st_size <= part_size for part in parts
    )
