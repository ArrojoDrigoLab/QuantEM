"""``release verify`` must fail on a bundle that names the machine that built it.

The sanitiser and the ``scan`` command both worked, and a bundle built before
them still passed ``verify`` **41/41, exit 0**, while all eight packs carried
the lab's file-server UNC name, the developer's home drive and the training
box's paths -- in ``checkpoint_index.json`` and ``resolved_config.yaml``, both
covered by ``MANIFEST.json``, i.e. exactly the bytes a downloader is told to
hash and keep.

Hashes prove a bundle is *intact*. They say nothing about what is in it. Since
``verify`` is the one command the README puts in front of both the publisher
and the downloader, the check belongs there and not in a separate command
nobody is required to run. Publishing is not reversible.
"""

from __future__ import annotations

import json

import pytest

from quantem.registry import release


def _write_bundle(root, *, index_contents: str) -> None:
    """A minimal bundle whose MANIFEST matches its files exactly."""
    pack_dir = root / "packs" / "quantem__mito"
    pack_dir.mkdir(parents=True)
    files = {
        "packs/quantem__mito/head.pt": b"\x00\x01weights",
        "packs/quantem__mito/checkpoint_index.json": index_contents.encode("utf-8"),
    }
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.write_bytes(data)
        entries.append(
            {
                "path": rel,
                "sha256": release.cache.sha256_file(path),
                "size_bytes": len(data),
            }
        )
    (root / release.MANIFEST_NAME).write_text(
        json.dumps(
            {
                "kind": release.BUNDLE_KIND,
                "schema_version": release.BUNDLE_SCHEMA_VERSION,
                "release": "0.0.0-test",
                "generated_at": "2026-08-07T00:00:00Z",
                "generated_by": {"quantem": "test"},
                "total_bytes": sum(e["size_bytes"] for e in entries),
                "files": entries,
                "packs": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


CLEAN_INDEX = json.dumps({"encoder": {"run_id": "m1_dinov3_vitb", "step": 674999}})

LEAKY_INDEX = json.dumps(
    {
        "encoder": {"run_id": "m1_dinov3_vitb", "step": 674999},
        # The real thing, from the bundle that shipped: a lab file server.
        "checkpoints": [
            {"path": r"\\EXAMPLEHOST\share\example\m1_checkpoints\m1_teacher_674999.pth"},
            {"path": "/mnt/d/example/legacy/m1_checkpoints/m1_teacher_674999.pth"},
        ],
    }
)


def test_a_clean_bundle_verifies(tmp_path):
    root = tmp_path / "clean"
    root.mkdir()
    _write_bundle(root, index_contents=CLEAN_INDEX)
    assert release.scan_bundle_for_local_paths(root) == {}
    assert all(release.verify_bundle(root).values())


def test_a_leaky_bundle_is_hash_intact_but_must_not_verify(tmp_path):
    root = tmp_path / "leaky"
    root.mkdir()
    _write_bundle(root, index_contents=LEAKY_INDEX)

    # The distinction that let this ship: every hash matches.
    assert all(release.verify_bundle(root).values()), "the bundle is intact"
    assert release.scan_bundle_for_local_paths(root), "and it names the build machine"

    assert release.main(["verify", str(root)]) == 1, (
        "verify must refuse a bundle that names its build machine"
    )


def test_the_failure_names_the_offending_files_and_how_to_fix_it(tmp_path, capsys):
    root = tmp_path / "leaky"
    root.mkdir()
    _write_bundle(root, index_contents=LEAKY_INDEX)

    release.main(["verify", str(root)])
    err = capsys.readouterr().err

    assert "checkpoint_index.json" in err
    assert "EXAMPLEHOST" in err, "the reader has to see what leaked"
    assert "must not be published" in err
    assert "release build" in err, "and what to do about it"


#: Verbatim from the bundle that was built before the sanitiser existed. A bare
#: host name or a lone directory fragment is deliberately NOT in this list --
#: neither is a local path, and flagging them would make the gate cry wolf.
REAL_LEAKS = [
    r"\\EXAMPLEHOST\share\example\m1_checkpoints\m1_teacher_674999.pth",
    "/mnt/d/example/legacy/m1_checkpoints/m1_teacher_674999.pth",
    r"V:\example\fig3_seg_data",
    "/root/dino/fig3/configs/experiments/FIG4/F4_omni_er.yaml",
]


@pytest.mark.parametrize("needle", REAL_LEAKS)
def test_each_real_world_leak_shape_is_caught(tmp_path, needle):
    root = tmp_path / "leaky"
    root.mkdir()
    _write_bundle(root, index_contents=json.dumps({"p": needle}))
    assert release.scan_bundle_for_local_paths(root), f"{needle} went undetected"
