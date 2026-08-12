"""Build a deterministic, signed-release-ready Windows server payload.

The user-facing installer is deliberately runtime-agnostic.  CI builds one CPU
and one CUDA server directory, archives each beneath a ``quantem-server`` root,
and gives this script's JSON manifest to the installer-manifest generator.
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


def build_payload(
    *,
    source: Path,
    output: Path,
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

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    build_info = {
        "schema": 1,
        "version": version,
        "variant": variant,
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_driver_api": cuda_driver_api,
    }

    paths = sorted(
        (path for path in source.rglob("*") if path.is_file()), key=lambda p: p.as_posix()
    )
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
        build_info_bytes = (json.dumps(build_info, indent=2, sort_keys=True) + "\n").encode("utf-8")
        archive.writestr(_zip_info("quantem-server/build-info.json"), build_info_bytes)

    payload: dict[str, object] = {
        "schema": 1,
        "version": version,
        "variant": variant,
        "filename": output.name,
        "signature_filename": f"{output.name}.sig",
        "sha256": _sha256(output),
        "size": output.stat().st_size,
        "size_mb": (output.stat().st_size + 1024 * 1024 - 1) // (1024 * 1024),
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_driver_api": cuda_driver_api,
        "cuda_files": cuda_files,
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
