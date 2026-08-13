from __future__ import annotations

import json

from quantem.registry import cache


def _write_pack(root, pack_id: str, digest: str) -> None:
    pack_root = root / "packs" / cache.pack_dirname(pack_id)
    pack_root.mkdir(parents=True)
    (pack_root / cache.RECORD_NAME).write_text(
        json.dumps(
            {
                "pack_id": pack_id,
                "head": {"filename": cache.HEAD_NAME, "sha256": digest},
                "encoder": {"filename": cache.ENCODER_NAME, "sha256": digest},
            }
        ),
        encoding="utf-8",
    )


def test_remove_pack_keeps_a_blob_referenced_by_another_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "models_root", lambda: tmp_path)
    digest = "ab" * 32
    blob = cache.blob_path(digest)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"shared encoder")
    _write_pack(tmp_path, "quantem:mito", digest)
    _write_pack(tmp_path, "quantem:nucleus", digest)

    assert cache.remove_pack("quantem:mito") is True

    assert not cache.pack_dir("quantem:mito").exists()
    assert cache.pack_dir("quantem:nucleus").exists()
    assert blob.exists()


def test_remove_last_referencing_pack_reclaims_the_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "models_root", lambda: tmp_path)
    digest = "cd" * 32
    blob = cache.blob_path(digest)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"unshared head")
    _write_pack(tmp_path, "omniem:er", digest)

    assert cache.remove_pack("omniem:er") is True

    assert not cache.pack_dir("omniem:er").exists()
    assert not blob.exists()


def test_remove_missing_pack_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "models_root", lambda: tmp_path)

    assert cache.remove_pack("quantem:ld") is False
