import type { LabelState } from "@/shared/types";
import type { ViewportBbox } from "@/utils/viewportUtils";

export const SEGMENT_TILE_SIZE = 1024;

export interface SegmentTileCoord {
  tileX: number;
  tileY: number;
}

export function segmentTileCoordKey(tileX: number, tileY: number): string {
  return `${tileX}:${tileY}`;
}

export function segmentTileLodFromZoom(zoom: number): number {
  if (zoom >= 3) return 0;
  if (zoom >= 1.75) return 1;
  if (zoom >= 1.25) return 2;
  if (zoom >= 0.9) return 3;
  return 4;
}

function clampTileCoord(value: number, maxValue: number): number {
  if (maxValue < 0) return 0;
  return Math.min(maxValue, Math.max(0, value));
}

function getImageTileBounds(
  imageWidth: number,
  imageHeight: number,
  tileSize: number
): { maxTileX: number; maxTileY: number } {
  return {
    maxTileX: Math.max(0, Math.ceil(imageWidth / tileSize) - 1),
    maxTileY: Math.max(0, Math.ceil(imageHeight / tileSize) - 1),
  };
}

export function getTilesForBbox(
  bbox: ViewportBbox,
  imageWidth: number,
  imageHeight: number,
  tileSize: number = SEGMENT_TILE_SIZE
): SegmentTileCoord[] {
  const { maxTileX, maxTileY } = getImageTileBounds(imageWidth, imageHeight, tileSize);
  if (maxTileX < 0 || maxTileY < 0) return [];

  const minTileX = clampTileCoord(Math.floor(bbox.x_min / tileSize), maxTileX);
  const maxVisibleTileX = clampTileCoord(Math.floor((bbox.x_max - 1) / tileSize), maxTileX);
  const minTileY = clampTileCoord(Math.floor(bbox.y_min / tileSize), maxTileY);
  const maxVisibleTileY = clampTileCoord(Math.floor((bbox.y_max - 1) / tileSize), maxTileY);

  const tiles: SegmentTileCoord[] = [];
  for (let tileY = minTileY; tileY <= maxVisibleTileY; tileY += 1) {
    for (let tileX = minTileX; tileX <= maxVisibleTileX; tileX += 1) {
      tiles.push({ tileX, tileY });
    }
  }
  return tiles;
}

export function withNeighborPrefetch(
  visibleTiles: SegmentTileCoord[],
  imageWidth: number,
  imageHeight: number,
  tileSize: number = SEGMENT_TILE_SIZE
): SegmentTileCoord[] {
  const { maxTileX, maxTileY } = getImageTileBounds(imageWidth, imageHeight, tileSize);
  const seen = new Set<string>();
  const result: SegmentTileCoord[] = [];

  const push = (tileX: number, tileY: number) => {
    if (tileX < 0 || tileY < 0 || tileX > maxTileX || tileY > maxTileY) return;
    const key = segmentTileCoordKey(tileX, tileY);
    if (seen.has(key)) return;
    seen.add(key);
    result.push({ tileX, tileY });
  };

  for (const tile of visibleTiles) {
    push(tile.tileX, tile.tileY);
  }
  for (const tile of visibleTiles) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        push(tile.tileX + dx, tile.tileY + dy);
      }
    }
  }
  return result;
}

export function segmentTileKey(params: {
  segmentationId: string;
  states: LabelState[];
  mode: "objects" | "density";
  tileSize: number;
  lod: number;
  tileX: number;
  tileY: number;
}): string {
  const states = [...params.states].sort().join(",");
  return [
    params.segmentationId,
    params.mode,
    `s=${states}`,
    `ts=${params.tileSize}`,
    `lod=${params.lod}`,
    `x=${params.tileX}`,
    `y=${params.tileY}`,
  ].join("|");
}

