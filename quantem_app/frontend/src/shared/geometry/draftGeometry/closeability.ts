import type { Point } from "@/utils/geometry";
import {
  collapseBacktracks,
  dedupeConsecutive,
  distanceBetween,
  dotProduct,
  EDGE_EPSILON,
  INVALID_CLOSE_POLYGON_MESSAGE,
  normalizeClosedRing,
  pointOnSegment,
  pointsEqual,
  projectPointToPolyline,
  resolveSingleAcceptedPolygonRing,
  signedRingArea,
  subtractPoints,
  type PolylineProjection,
  type SegmentLike,
} from "@/shared/geometry/draftGeometry/shared";

export type SingleCycleCloseability =
  | { kind: "not_ready" }
  | { kind: "ok"; ring: Point[] }
  | { kind: "error"; message: string };

type Segment = SegmentLike;

function segmentParameter(point: Point, start: Point, end: Point) {
  const direction = subtractPoints(end, start);
  const lengthSq = dotProduct(direction, direction);
  if (lengthSq <= EDGE_EPSILON) {
    return 0;
  }
  return Math.min(
    1,
    Math.max(0, dotProduct(subtractPoints(point, start), direction) / lengthSq)
  );
}

function findPointProjectionsOnPolyline(
  point: Point,
  polyline: Point[]
) {
  const exactMatches: PolylineProjection[] = [];
  let traversed = 0;

  for (let index = 0; index < polyline.length - 1; index += 1) {
    const start = polyline[index];
    const end = polyline[index + 1];
    const segmentLength = distanceBetween(start, end);
    if (pointOnSegment(point, start, end)) {
      exactMatches.push({
        point,
        segmentIndex: index,
        t: segmentParameter(point, start, end),
        distance: 0,
        offset: traversed + (segmentLength * segmentParameter(point, start, end)),
      });
    }
    traversed += segmentLength;
  }

  if (exactMatches.length > 0) {
    return exactMatches.sort((left, right) => left.offset - right.offset);
  }

  const projected = projectPointToPolyline(point, polyline);
  return projected && projected.distance <= EDGE_EPSILON * 4 ? [projected] : [];
}

function insertProjectionsIntoPolyline(
  polyline: Point[],
  projections: Array<PolylineProjection & { key: string }>
) {
  const path = polyline.length > 0 ? [polyline[0]] : [];
  const indexByKey = new Map<string, number>();

  for (let segmentIndex = 0; segmentIndex < polyline.length - 1; segmentIndex += 1) {
    const onSegment = projections
      .filter((projection) => projection.segmentIndex === segmentIndex)
      .sort((left, right) => left.t - right.t);

    for (const projection of onSegment) {
      if (!path.length || !pointsEqual(path[path.length - 1], projection.point)) {
        path.push(projection.point);
      }
      indexByKey.set(projection.key, path.length - 1);
    }

    const end = polyline[segmentIndex + 1];
    if (!path.length || !pointsEqual(path[path.length - 1], end)) {
      path.push(end);
    }
  }

  return { path, indexByKey };
}

function polylineTailFromProjection(
  polyline: Point[],
  projection: PolylineProjection
) {
  const { path, indexByKey } = insertProjectionsIntoPolyline(polyline, [
    { ...projection, key: "tail-start" },
  ]);
  const startIndex = indexByKey.get("tail-start");
  if (startIndex == null) {
    return [projection.point];
  }
  return dedupeConsecutive(path.slice(startIndex));
}

function normalizeCandidateRing(points: Point[]) {
  const ring = normalizeClosedRing(collapseBacktracks(points));
  if (!ring || Math.abs(signedRingArea(ring)) <= EDGE_EPSILON) {
    return null;
  }
  return ring;
}

function resolveOpenPathCloseability(
  pathInput: Point[],
  closurePoint: Point | null
): SingleCycleCloseability {
  const path = dedupeConsecutive(pathInput);
  if (path.length < 3 || !closurePoint) {
    return { kind: "not_ready" };
  }

  const closureProjections = findPointProjectionsOnPolyline(closurePoint, path);
  if (closureProjections.length === 0) {
    return { kind: "not_ready" };
  }

  let bestRing: Point[] | null = null;
  let bestArea = -1;
  for (const closureProjection of closureProjections) {
    const ring = resolveSingleAcceptedPolygonRing(
      polylineTailFromProjection(path, closureProjection)
    );
    if (!ring) {
      continue;
    }
    const area = Math.abs(signedRingArea(ring));
    if (!bestRing || area > bestArea + EDGE_EPSILON) {
      bestRing = ring;
      bestArea = area;
    }
  }

  if (!bestRing) {
    return { kind: "error", message: INVALID_CLOSE_POLYGON_MESSAGE };
  }

  return { kind: "ok", ring: bestRing };
}

export function resolveSingleCycleCloseability(
  paths: Point[][],
  closureEdge: Segment | null
): SingleCycleCloseability {
  if (!closureEdge || pointsEqual(closureEdge.start, closureEdge.end)) {
    return { kind: "not_ready" };
  }
  if (paths.length === 1) {
    return resolveOpenPathCloseability(paths[0] ?? [], closureEdge.end);
  }
  return { kind: "error", message: INVALID_CLOSE_POLYGON_MESSAGE };
}

export async function resolveSingleCycleCloseabilityAsync(
  paths: Point[][],
  closureEdge: Segment | null
): Promise<SingleCycleCloseability> {
  return resolveSingleCycleCloseability(paths, closureEdge);
}

export function normalizeRingToSingleCycle(ringInput: Point[]) {
  const ring = normalizeCandidateRing(ringInput);
  return ring ? resolveSingleAcceptedPolygonRing(ring) : null;
}
