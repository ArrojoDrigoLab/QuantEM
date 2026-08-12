from __future__ import annotations

import json
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
    assert sum(
        workflow.read_text(encoding="utf-8").count("scripts/invoke_tauri_build.ps1")
        for workflow in workflows
    ) == 3

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
    assert 'update_filename = $installerName' in collect_step
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


def test_package_numpy_floor_supports_the_intel_macos_torch_build() -> None:
    pyproject = (REPO_ROOT / "quantem_app/pyproject.toml").read_text(encoding="utf-8")
    assert '"numpy>=1.26"' in pyproject
    assert '"numpy>=2.0"' not in pyproject


def test_installer_hash_verification_uses_the_base_windows_dotnet_api() -> None:
    hooks = (REPO_ROOT / "quantem_app/desktop/src-tauri/nsis/hooks.nsh").read_text(encoding="utf-8")
    assert "Security.Cryptography.SHA256" in hooks
    assert "Get-FileHash -LiteralPath" not in hooks


def _server_tree(root: Path, *, cuda: bool) -> Path:
    source = root / ("cuda" if cuda else "cpu") / "quantem-server"
    source.mkdir(parents=True)
    (source / "quantem-server.exe").write_bytes(b"MZ-fake-server")
    (source / "library.zip").write_bytes(b"python")
    if cuda:
        (source / "torch_cuda.dll").write_bytes(b"cuda")
    return source


def _make_payload(tmp_path: Path, variant: str) -> Path:
    output = tmp_path / f"quantem-server_0.2.0_windows-x64-{variant}.zip"
    manifest = tmp_path / variant / "payload-manifest.json"
    build_payload(
        source=_server_tree(tmp_path / "sources", cuda=variant == "cuda"),
        output=output,
        manifest_output=manifest,
        version="0.2.0",
        variant=variant,
        torch_version="2.13.0",
        cuda_runtime="cu126" if variant == "cuda" else None,
        cuda_driver_api=12060 if variant == "cuda" else None,
    )
    (manifest.parent / f"{output.name}.sig").write_text(f"signed-{variant}\n", encoding="utf-8")
    return manifest


def test_payload_archive_has_one_stable_root_and_build_info(tmp_path: Path) -> None:
    manifest_path = _make_payload(tmp_path, "cuda")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_path = tmp_path / manifest["filename"]
    with zipfile.ZipFile(archive_path) as archive:
        assert "quantem-server/quantem-server.exe" in archive.namelist()
        info = json.loads(archive.read("quantem-server/build-info.json"))
    assert info["variant"] == "cuda"
    assert info["cuda_driver_api"] == 12060
    assert len(manifest["sha256"]) == 64


def test_cpu_payload_rejects_cuda_libraries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unexpectedly contains CUDA"):
        build_payload(
            source=_server_tree(tmp_path, cuda=True),
            output=tmp_path / "bad.zip",
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
    assert "windows-x64-cpu.zip" in text
    assert result["variants"]["cuda"]["signature"] == "signed-cuda"


def test_oversized_payload_is_split_below_the_release_asset_limit(tmp_path: Path) -> None:
    manifest_path = _make_payload(tmp_path, "cpu")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = tmp_path / manifest["filename"]
    part_size = (archive.stat().st_size + 2) // 3
    parts = split_payload(
        archive=archive,
        manifest_path=manifest_path,
        max_part_bytes=part_size,
        remove_archive=True,
    )
    assert len(parts) > 1
    assert not archive.exists()
    assert all((tmp_path / str(part["filename"])).stat().st_size <= part_size for part in parts)
