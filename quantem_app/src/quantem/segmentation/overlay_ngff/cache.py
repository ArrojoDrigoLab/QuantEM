"""Small process-local caches for immutable stores and revisioned palettes."""

from collections import OrderedDict
from threading import RLock
from time import monotonic


class BoundedCache:
    def __init__(self, *, max_bytes=32 * 1024 * 1024, max_entries=64, ttl=60):
        self.max_bytes, self.max_entries, self.ttl = max_bytes, max_entries, ttl
        self.entries = OrderedDict()
        self.size = 0
        self.lock = RLock()

    def get(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return None
            value, size, expiry = entry
            if expiry < monotonic():
                del self.entries[key]
                self.size -= size
                return None
            self.entries.move_to_end(key)
            return value

    def put(self, key, value, size=1):
        with self.lock:
            if key in self.entries:
                self.size -= self.entries.pop(key)[1]
            if size > self.max_bytes:
                return
            self.entries[key] = (value, size, monotonic() + self.ttl)
            self.size += size
            while self.size > self.max_bytes or len(self.entries) > self.max_entries:
                self.size -= self.entries.popitem(last=False)[1][1]

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.size = 0


store_validation_cache = BoundedCache(max_entries=128, ttl=30)
palette_cache = BoundedCache(ttl=120)
