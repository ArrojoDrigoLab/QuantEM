import { describe, expect, it } from "vitest";
import {
  getTilesForBbox,
  segmentTileCoordKey,
  segmentTileKey,
  segmentTileLodFromZoom,
  withNeighborPrefetch,
} from "@/utils/segmentTiles";

describe("segmentTiles", () => {
  it("computes visible tile coords from an image-space bbox", () => {
    const tiles = getTilesForBbox(
      { x_min: 100, y_min: 50, x_max: 2300, y_max: 1400 },
      4096,
      4096,
      1024
    );
    expect(tiles).toEqual([
      { tileX: 0, tileY: 0 },
      { tileX: 1, tileY: 0 },
      { tileX: 2, tileY: 0 },
      { tileX: 0, tileY: 1 },
      { tileX: 1, tileY: 1 },
      { tileX: 2, tileY: 1 },
    ]);
  });

  it("adds neighbor prefetch tiles around visible set without duplicates", () => {
    const prefetched = withNeighborPrefetch(
      [{ tileX: 1, tileY: 1 }],
      4096,
      4096,
      1024
    );
    const keys = prefetched.map((tile) => segmentTileCoordKey(tile.tileX, tile.tileY));
    expect(keys).toHaveLength(9);
    expect(new Set(keys).size).toBe(9);
    expect(keys).toContain("0:0");
    expect(keys).toContain("1:1");
    expect(keys).toContain("2:2");
  });

  it("maps zoom to tile lod buckets", () => {
    expect(segmentTileLodFromZoom(4)).toBe(0);
    expect(segmentTileLodFromZoom(2)).toBe(1);
    expect(segmentTileLodFromZoom(1.5)).toBe(2);
    expect(segmentTileLodFromZoom(1.0)).toBe(3);
    expect(segmentTileLodFromZoom(0.4)).toBe(4);
  });

  it("builds deterministic cache keys regardless of states order", () => {
    const keyA = segmentTileKey({
      segmentationId: "seg-1",
      states: ["CONFIRMED", "CANDIDATE"],
      mode: "objects",
      tileSize: 1024,
      lod: 2,
      tileX: 3,
      tileY: 4,
    });
    const keyB = segmentTileKey({
      segmentationId: "seg-1",
      states: ["CANDIDATE", "CONFIRMED"],
      mode: "objects",
      tileSize: 1024,
      lod: 2,
      tileX: 3,
      tileY: 4,
    });
    expect(keyA).toBe(keyB);
  });
});
