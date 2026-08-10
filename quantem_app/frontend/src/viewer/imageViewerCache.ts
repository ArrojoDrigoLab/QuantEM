import { loadOmeZarr } from "@hms-dbmi/viv";

const VIV_OME_ZARR_CACHE_CAPACITY = 24;

type VivLoaderData = Awaited<ReturnType<typeof loadOmeZarr>>["data"];

type VivOmeZarrCacheEntry = {
  data?: VivLoaderData;
  promise: Promise<VivLoaderData>;
};

const vivOmeZarrCache = new Map<string, VivOmeZarrCacheEntry>();

function touchVivOmeZarrCacheEntry(url: string, entry: VivOmeZarrCacheEntry) {
  vivOmeZarrCache.delete(url);
  vivOmeZarrCache.set(url, entry);
}

function trimVivOmeZarrCache() {
  while (vivOmeZarrCache.size > VIV_OME_ZARR_CACHE_CAPACITY) {
    const oldestKey = vivOmeZarrCache.keys().next().value;
    if (!oldestKey) return;
    vivOmeZarrCache.delete(oldestKey);
  }
}

export function loadOmeZarrCached(url: string): Promise<VivLoaderData> {
  const existing = vivOmeZarrCache.get(url);
  if (existing) {
    touchVivOmeZarrCacheEntry(url, existing);
    return existing.data ? Promise.resolve(existing.data) : existing.promise;
  }

  const entry: VivOmeZarrCacheEntry = {
    promise: loadOmeZarr(url, { type: "multiscales" })
      .then((result) => {
        entry.data = result.data;
        touchVivOmeZarrCacheEntry(url, entry);
        trimVivOmeZarrCache();
        return result.data;
      })
      .catch((error) => {
        vivOmeZarrCache.delete(url);
        throw error;
      }),
  };
  vivOmeZarrCache.set(url, entry);
  trimVivOmeZarrCache();
  return entry.promise;
}

export function clearVivOmeZarrCache(): void {
  vivOmeZarrCache.clear();
}

export function resetVivOmeZarrCacheForTests(): void {
  clearVivOmeZarrCache();
}
