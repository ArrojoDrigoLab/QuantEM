import {
  SEGMENT_SMOOTHING_TOLERANCE,
} from "@/config";
import type { SegmentObject } from "@/shared/types";
import { simplifyPolygon, type Point } from "@/utils/geometry";

interface SmoothedGeometryCacheEntry {
  signature: string;
  coords: Array<[number, number]>;
}

const MAX_SMOOTHED_GEOMETRY_CACHE_SIZE = 50000;
const smoothedGeometryCache = new Map<string, SmoothedGeometryCacheEntry>();

function segmentGeometrySignature(segment: SegmentObject): string {
  const coords = segment.geometry_coords;
  if (coords.length === 0) {
    return `${segment.updated_at}:0`;
  }
  const first = coords[0];
  const last = coords[coords.length - 1];
  return `${segment.updated_at}:${coords.length}:${first[0]}:${first[1]}:${last[0]}:${last[1]}`;
}

function coordsToPoints(coords: Array<[number, number]>): Point[] {
  return coords.map(([x, y]) => ({ x, y }));
}

function pointsToCoords(points: Point[]): Array<[number, number]> {
  return points.map((point) => [point.x, point.y]);
}

function buildSmoothedGeometry(coords: Array<[number, number]>): Array<[number, number]> {
  if (coords.length < 4) {
    return coords;
  }

  const simplified = simplifyPolygon(
    coordsToPoints(coords),
    SEGMENT_SMOOTHING_TOLERANCE
  );
  if (simplified.length < 4) {
    return coords;
  }
  return pointsToCoords(simplified);
}

function getCachedSmoothedGeometry(segment: SegmentObject): Array<[number, number]> {
  const signature = segmentGeometrySignature(segment);
  const existing = smoothedGeometryCache.get(segment.id);
  if (existing && existing.signature === signature) {
    return existing.coords;
  }

  const coords = buildSmoothedGeometry(segment.geometry_coords);
  smoothedGeometryCache.set(segment.id, { signature, coords });
  if (smoothedGeometryCache.size > MAX_SMOOTHED_GEOMETRY_CACHE_SIZE) {
    smoothedGeometryCache.clear();
  }
  return coords;
}

export function withSmoothedSegmentGeometry<T extends SegmentObject>(segment: T): T {
  if (segment.smoothed_geometry_coords && segment.smoothed_geometry_coords.length >= 3) {
    return segment;
  }

  return {
    ...segment,
    smoothed_geometry_coords: getCachedSmoothedGeometry(segment),
  } as T;
}

export function withSmoothedSegmentGeometryBatch<T extends SegmentObject>(
  segments: T[]
): T[] {
  if (segments.length === 0) {
    return segments;
  }
  return segments.map(withSmoothedSegmentGeometry);
}

export function selectSegmentGeometryCoords(
  segment: SegmentObject,
  useSmoothedGeometry: boolean
): Array<[number, number]> {
  if (
    useSmoothedGeometry &&
    segment.smoothed_geometry_coords &&
    segment.smoothed_geometry_coords.length >= 3
  ) {
    return segment.smoothed_geometry_coords;
  }
  return segment.geometry_coords;
}
