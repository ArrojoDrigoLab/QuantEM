"""ONE live-network test against the real Hugging Face repository.

Marked ``requires_network`` and excluded from offline lanes; run it with::

    pytest src/quantem/registry/tests/test_hf_live.py -m requires_network

It downloads the smallest published head (``omniem-ld.safetensors``, 25.7 MB)
at the pinned revision and proves the whole verification chain end to end:
the model card's digest, the repository's LFS object id, and the re-hash of
the delivered bytes all agree. Everything else about the install path is
covered with the network mocked in ``test_hf_install.py``.
"""

from __future__ import annotations

import pytest

from quantem.registry import cache, hf

pytestmark = pytest.mark.requires_network


def test_smallest_head_downloads_and_verifies_end_to_end():
    card = hf.fetch_sidecar("omniem:ld")
    assert card.head_file == "omniem-ld.safetensors"
    assert card.head_sha256 and len(card.head_sha256) == 64
    assert card.head_bytes == 25_730_688  # the published size, pinned by revision

    # The LFS object id at the pinned revision is the same digest the card
    # publishes -- two independent sources, one truth.
    remote = hf.remote_file_info(card.head_file)
    assert remote.sha256 == card.head_sha256
    assert remote.size_bytes == card.head_bytes

    seen = []
    path = hf.download_file(
        card.head_file,
        expected_bytes=card.head_bytes,
        on_bytes=lambda done, total: seen.append((done, total)),
    )
    assert path.is_file()
    assert path.stat().st_size == card.head_bytes
    # The bytes on disk re-hash to the published digest.
    assert cache.sha256_file(path) == card.head_sha256
    # Progress ended at 100% of a known total.
    assert seen and seen[-1] == (card.head_bytes, card.head_bytes)
    # And the cache it landed in is the app's, not HF's home.
    assert str(hf.hf_cache_dir()) in str(path)
