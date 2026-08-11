"""A small, bounded LRU over crop embeddings.

Deliberately unremarkable, and deliberately capped. The implementation this was
ported from kept two module-level dicts and two on-disk ``.npz`` directories
with **no eviction in any of the four**, which in a desktop process that stays
open for days is simply a leak. There is no disk tier here: re-encoding costs
about half a second, and half a second is cheaper than a cache that grows until
the process dies.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from quantem.sam.backends.base import Embedding
from quantem.sam.config import EMBEDDING_CACHE_ENTRIES
from quantem.sam.geometry import Crop

#: ``(segmentation id, backend identity, crop window)``.
#:
#: The window, not the box -- see :data:`quantem.sam.config.CROP_GRID`. The
#: backend identity is in the key so new weights can never be handed an
#: embedding the old ones produced.
CacheKey = tuple[str, str, tuple[int, int, int, int]]


def cache_key(segmentation_id: str, backend_identity: str, crop: Crop) -> CacheKey:
    return (str(segmentation_id), str(backend_identity), crop.key())


class EmbeddingCache:
    """LRU with a fixed entry cap, safe to share between request threads."""

    def __init__(self, max_entries: int = EMBEDDING_CACHE_ENTRIES) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[CacheKey, Embedding] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: CacheKey) -> Embedding | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: CacheKey, embedding: Embedding) -> None:
        with self._lock:
            self._entries[key] = embedding
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def bytes_held(self) -> int:
        with self._lock:
            return sum(entry.nbytes for entry in self._entries.values())


#: The process-wide cache. One per process is right: the predictor is too.
EMBEDDINGS = EmbeddingCache()
