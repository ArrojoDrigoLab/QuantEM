"""Verify and stage the pinned PostgreSQL runtime for a desktop build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME_SPEC = HERE / "runtime.json"
REQUIRED_DIRECTORIES = ("bin", "lib", "share")
LICENSE_FILES = ("server_license.txt", "commandlinetools_3rd_party_licenses.txt")
REQUIRED_TOOLS = ("postgres", "initdb", "pg_ctl", "pg_isready", "psql", "createdb", "pg_dump", "pg_restore")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _platform_key() -> str:
    if os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"}:
        return "windows-x64"
    if sys_platform() == "darwin" and platform.machine().lower() in {"arm64", "x86_64"}:
        return "macos-universal"
    raise SystemExit(f"No packaged PostgreSQL runtime is defined for {platform.system()} {platform.machine()}")


def sys_platform() -> str:
    import sys

    return sys.platform


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise SystemExit(f"archive member escapes extraction root: {member.filename}")
        source.extractall(destination)


def _download(entry: dict[str, object], cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = str(entry["url"]).rsplit("/", 1)[-1]
    destination = cache_dir / filename
    if destination.is_file() and _sha256(destination) == entry["sha256"]:
        return destination
    temporary = destination.with_name(f".{destination.name}.tmp")
    with urllib.request.urlopen(str(entry["url"]), timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    if _sha256(temporary) != entry["sha256"]:
        temporary.unlink(missing_ok=True)
        raise SystemExit("downloaded PostgreSQL archive failed SHA-256 verification")
    os.replace(temporary, destination)
    return destination


def stage_runtime(*, archive: Path, output: Path, platform_key: str) -> dict[str, object]:
    specification = json.loads(RUNTIME_SPEC.read_text(encoding="utf-8"))
    entry = specification["platforms"][platform_key]
    actual_bytes = archive.stat().st_size
    if actual_bytes != entry["archive_bytes"]:
        raise SystemExit(
            "PostgreSQL archive size mismatch: "
            f"expected {entry['archive_bytes']}, got {actual_bytes}"
        )
    actual_hash = _sha256(archive)
    if actual_hash != entry["sha256"]:
        raise SystemExit(
            f"PostgreSQL archive SHA-256 mismatch: expected {entry['sha256']}, got {actual_hash}"
        )
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="quantem-postgresql-", dir=output.parent) as temp:
        extraction = Path(temp)
        _safe_extract(archive, extraction)
        root = extraction / str(entry["archive_root"])
        for name in REQUIRED_DIRECTORIES:
            source = root / name
            if not source.is_dir():
                raise SystemExit(f"PostgreSQL archive is missing {name}/")
            shutil.copytree(source, output / name)
        for name in LICENSE_FILES:
            source = root / name
            if not source.is_file():
                raise SystemExit(f"PostgreSQL archive is missing {name}")
            shutil.copy2(source, output / name)

    suffix = ".exe" if platform_key == "windows-x64" else ""
    for tool in REQUIRED_TOOLS:
        if not (output / "bin" / f"{tool}{suffix}").is_file():
            raise SystemExit(f"staged PostgreSQL runtime is missing {tool}{suffix}")
    postgres = output / "bin" / f"postgres{suffix}"
    result = subprocess.run(
        [str(postgres), "--version"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "DYLD_LIBRARY_PATH": str(output / "lib")},
    )
    expected_version = str(specification["postgresql_version"])
    if result.returncode or f" {expected_version}" not in result.stdout:
        raise SystemExit(f"staged postgres failed version check: {result.stdout} {result.stderr}")

    files = []
    total_bytes = 0
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "schema": 1,
        "platform": platform_key,
        "postgresql_version": expected_version,
        "source_archive_sha256": actual_hash,
        "total_bytes": total_bytes,
        "files": files,
    }
    (output / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--download-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows-x64", "macos-universal"))
    args = parser.parse_args()
    key = args.platform or _platform_key()
    specification = json.loads(RUNTIME_SPEC.read_text(encoding="utf-8"))
    entry = specification["platforms"][key]
    archive = args.archive
    if archive is None:
        if args.download_cache is None:
            parser.error("provide --archive or --download-cache")
        archive = _download(entry, args.download_cache)
    manifest = stage_runtime(archive=archive.resolve(strict=True), output=args.output.resolve(), platform_key=key)
    print(json.dumps({key: {"files": len(manifest["files"]), "bytes": manifest["total_bytes"]}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
