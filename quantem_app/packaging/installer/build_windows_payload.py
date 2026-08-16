"""Build deterministic Windows runtime and application layers.

PyInstaller's onedir layout mixes a small, frequently-changing application
surface with a very large, rarely-changing native runtime.  This module splits
the tree without changing the layout seen by ``quantem-server.exe``:

* the runtime archive contains Python, PyTorch, CUDA, and third-party files;
* the application archive contains the executable plus QuantEM-owned data.

The installer overlays the application archive on the runtime directory.  A
content-derived runtime ID lets routine upgrades retain an already-installed
runtime while still forcing a full replacement whenever any runtime file
changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[+.-][0-9A-Za-z.-]+)?$")
_CUDA_DLL_NAMES = (
    "c10_cuda.dll",
    "torch_cuda.dll",
    "cudart64",
    "cublas64",
    "cudnn64",
)
_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)
_APPLICATION_INTERNAL_DIRS = {"quantem", "quantem_frontend"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cuda_runtime_files(source: Path) -> list[str]:
    matches: list[str] = []
    for path in source.rglob("*.dll"):
        lowered = path.name.lower()
        if any(token in lowered for token in _CUDA_DLL_NAMES):
            matches.append(path.relative_to(source).as_posix())
    return sorted(matches)


def _zip_info(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, _ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    mode = stat.S_IFREG | (0o755 if executable else 0o644)
    info.external_attr = mode << 16
    info.create_system = 3
    return info


def _is_application_path(relative: Path) -> bool:
    """Return whether *relative* belongs to the replaceable application layer."""

    parts = relative.parts
    if parts == ("quantem-server.exe",):
        return True
    if len(parts) < 2 or parts[0] != "_internal":
        return False
    package = parts[1]
    return package in _APPLICATION_INTERNAL_DIRS or (
        package.startswith("quantem_app-") and package.endswith(".dist-info")
    )


def _file_manifest(
    source: Path, paths: list[Path], contract: dict[str, object]
) -> tuple[str, list[dict[str, object]]]:
    """Hash runtime files and return a stable compatibility ID plus manifest."""

    identity = hashlib.sha256()
    identity.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    identity.update(b"\0")
    entries: list[dict[str, object]] = []
    for path in paths:
        relative = path.relative_to(source).as_posix()
        digest = _sha256(path)
        size = path.stat().st_size
        entries.append({"path": relative, "size": size, "sha256": digest})
        identity.update(relative.encode("utf-8"))
        identity.update(b"\0")
        identity.update(str(size).encode("ascii"))
        identity.update(b"\0")
        identity.update(digest.encode("ascii"))
        identity.update(b"\0")
    return identity.hexdigest(), entries


def _write_archive(
    *,
    source: Path,
    paths: list[Path],
    output: Path,
    extras: dict[str, bytes],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True, compresslevel=6) as archive:
        for path in paths:
            relative = path.relative_to(source)
            archive_name = str(PurePosixPath("quantem-server", *relative.parts))
            info = _zip_info(
                archive_name, executable=path.suffix.lower() in {".exe", ".dll", ".pyd"}
            )
            with (
                path.open("rb") as source_stream,
                archive.open(info, "w", force_zip64=True) as target,
            ):
                shutil.copyfileobj(source_stream, target, length=1024 * 1024)
        for archive_name, content in sorted(extras.items()):
            archive.writestr(_zip_info(archive_name), content)


def build_payload(
    *,
    source: Path,
    output: Path,
    application_output: Path,
    manifest_output: Path,
    version: str,
    variant: str,
    torch_version: str,
    cuda_runtime: str | None,
    cuda_driver_api: int | None,
) -> dict[str, object]:
    if not _VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid release version {version!r}")
    if variant not in {"cpu", "cuda"}:
        raise ValueError(f"invalid runtime variant {variant!r}")
    source = source.resolve()
    if not (source / "quantem-server.exe").is_file():
        raise FileNotFoundError(
            f"missing frozen server executable: {source / 'quantem-server.exe'}"
        )

    cuda_files = _cuda_runtime_files(source)
    if variant == "cpu" and cuda_files:
        raise ValueError(f"CPU payload unexpectedly contains CUDA libraries: {cuda_files[:5]}")
    if variant == "cuda" and not cuda_files:
        raise ValueError("CUDA payload contains no CUDA runtime libraries")
    if variant == "cuda" and (not cuda_runtime or not cuda_driver_api):
        raise ValueError("CUDA payloads require --cuda-runtime and --cuda-driver-api")
    if variant == "cpu" and (cuda_runtime or cuda_driver_api):
        raise ValueError("CPU payloads must not declare a CUDA runtime")

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": 1,
        "variant": variant,
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_driver_api": cuda_driver_api,
    }

    paths = sorted(
        (path for path in source.rglob("*") if path.is_file()), key=lambda p: p.as_posix()
    )
    application_paths = [path for path in paths if _is_application_path(path.relative_to(source))]
    runtime_paths = [path for path in paths if path not in application_paths]
    if not application_paths or not runtime_paths:
        raise ValueError("server tree must contain both application and runtime files")

    runtime_id, runtime_files = _file_manifest(source, runtime_paths, contract)
    runtime_info = {**contract, "runtime_id": runtime_id}
    build_info = {
        **runtime_info,
        "schema": 2,
        # Keep ``version`` for the installed diagnostics contract.
        "version": version,
        "application_version": version,
    }
    runtime_manifest = {
        "schema": 1,
        "runtime_id": runtime_id,
        "files": runtime_files,
    }

    _write_archive(
        source=source,
        paths=runtime_paths,
        output=output,
        extras={
            "quantem-server/runtime-info.json": (
                json.dumps(runtime_info, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
        },
    )
    _write_archive(
        source=source,
        paths=application_paths,
        output=application_output,
        extras={
            "quantem-server/build-info.json": (
                json.dumps(build_info, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
            "quantem-layer/runtime-files.json": (
                json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
            "quantem-layer/runtime-info.json": (
                json.dumps(runtime_info, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        },
    )

    payload: dict[str, object] = {
        "schema": 2,
        "version": version,
        "variant": variant,
        "runtime_id": runtime_id,
        "filename": output.name,
        "signature_filename": f"{output.name}.sig",
        "sha256": _sha256(output),
        "size": output.stat().st_size,
        "size_mb": (output.stat().st_size + 1024 * 1024 - 1) // (1024 * 1024),
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_driver_api": cuda_driver_api,
        "cuda_files": cuda_files,
        "application_filename": application_output.name,
        "application_sha256": _sha256(application_output),
        "application_size": application_output.stat().st_size,
        "parts": [
            {
                "filename": output.name,
                "sha256": _sha256(output),
                "size": output.stat().st_size,
            }
        ],
    }
    manifest_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--application-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--variant", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--cuda-runtime")
    parser.add_argument("--cuda-driver-api", type=int)
    args = parser.parse_args()
    build_payload(
        source=args.source,
        output=args.output,
        application_output=args.application_output,
        manifest_output=args.manifest_output,
        version=args.version,
        variant=args.variant,
        torch_version=args.torch_version,
        cuda_runtime=args.cuda_runtime,
        cuda_driver_api=args.cuda_driver_api,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
