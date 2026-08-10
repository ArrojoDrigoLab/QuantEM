import { pointInPolygon, type Point } from "@/utils/geometry";

export const EDGE_EPSILON = 1e-6;
export const SEGMENT_SEARCH_CELL_SIZE_PX = 300;
export const MIN_SIGNIFICANT_POLYGON_AREA_PX = 1000;
export const INVALID_CLOSE_POLYGON_MESSAGE =
  "That segment would make Close Polygon resolve to multiple polygons.";

export type SegmentIntersection =
  | { kind: "point"; point: Point }
  | { kind: "overlap" };

export interface SegmentLike {
  start: Point;
  end: Point;
}

export interface PolylineProjection {
  point: Point;
  segmentIndex: number;
  t: number;
  distance: number;
  offset: number;
}

export function nearlyEqual(left: number, right: number) {
  return Math.abs(left - right) <= EDGE_EPSILON;
}

export function pointsEqual(left: Point, right: Point) {
  return nearlyEqual(left.x, right.x) && nearlyEqual(left.y, right.y);
}

export function dedupeConsecutive(points: Point[]) {
  const deduped: Point[] = [];
  for (const point of points) {
    if (!deduped.length || !pointsEqual(point, deduped[deduped.length - 1])) {
      deduped.push(point);
    }
  }
  return deduped;
}

export function distanceBetween(left: Point, right: Point) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

export function dotProduct(left: Point, right: Point) {
  return (left.x * right.x) + (left.y * right.y);
}

export function crossProduct(left: Point, right: Point) {
  return (left.x * right.y) - (left.y * right.x);
}

export function subtractPoints(left: Point, right: Point): Point {
  return {
    x: left.x - right.x,
    y: left.y - right.y,
  };
}

export function collapseBacktracks(points: Point[]) {
  const collapsed: Point[] = [];
  for (const point of points) {
    if (
      collapsed.length >= 2 &&
      pointsEqual(collapsed[collapsed.length - 2], point)
    ) {
      collapsed.pop();
      continue;
    }
    if (!collapsed.length || !pointsEqual(collapsed[collapsed.length - 1], point)) {
      collapsed.push(point);
    }
  }
  return collapsed;
}

export function lerpPoint(start: Point, end: Point, ratio: number): Point {
  return {
    x: start.x + ((end.x - start.x) * ratio),
    y: start.y + ((end.y - start.y) * ratio),
  };
}

export function segmentIntersection(
  startA: Point,
  endA: Point,
  startB: Point,
  endB: Point
): SegmentIntersection | null {
  const directionA = subtractPoints(endA, startA);
  const directionB = subtractPoints(endB, startB);
  const denominator = crossProduct(directionA, directionB);
  const offset = subtractPoints(startB, startA);
  const offsetCrossA = crossProduct(offset, directionA);

  if (nearlyEqual(denominator, 0) && nearlyEqual(offsetCrossA, 0)) {
    const directionALengthSq = dotProduct(directionA, directionA);
    if (nearlyEqual(directionALengthSq, 0)) {
      return pointsEqual(startA, startB) ? { kind: "point", point: startA } : null;
    }

    const projectionStart = dotProduct(offset, directionA) / directionALengthSq;
    const projectionEnd =
      projectionStart + (dotProduct(directionB, directionA) / directionALengthSq);
    const overlapStart = Math.max(0, Math.min(projectionStart, projectionEnd));
    const overlapEnd = Math.min(1, Math.max(projectionStart, projectionEnd));
    if (overlapEnd < overlapStart - EDGE_EPSILON) {
      return null;
    }
    if (nearlyEqual(overlapStart, overlapEnd)) {
      return {
        kind: "point",
        point: lerpPoint(startA, endA, Math.min(1, Math.max(0, overlapStart))),
      };
    }
    return { kind: "overlap" };
  }

  if (nearlyEqual(denominator, 0)) {
    return null;
  }

  const ratioA = crossProduct(offset, directionB) / denominator;
  const ratioB = crossProduct(offset, directionA) / denominator;
  if (
    ratioA < -EDGE_EPSILON ||
    ratioA > 1 + EDGE_EPSILON ||
    ratioB < -EDGE_EPSILON ||
    ratioB > 1 + EDGE_EPSILON
  ) {
    return null;
  }

  return {
    kind: "point",
    point: lerpPoint(startA, endA, Math.min(1, Math.max(0, ratioA))),
  };
}

export function pointOnSegment(point: Point, start: Point, end: Point) {
  const segment = subtractPoints(end, start);
  const relative = subtractPoints(point, start);
  const cross = crossProduct(segment, relative);
  if (!nearlyEqual(cross, 0)) {
    return false;
  }
  const dot = dotProduct(relative, segment);
  if (dot < -EDGE_EPSILON) {
    return false;
  }
  const segmentLengthSq = dotProduct(segment, segment);
  return dot <= segmentLengthSq + EDGE_EPSILON;
}

export function projectPointToPolyline(
  point: Point,
  polyline: Point[]
): PolylineProjection | null {
  if (polyline.length === 0) {
    return null;
  }
  if (polyline.length === 1) {
    return {
      point: polyline[0],
      segmentIndex: 0,
      t: 0,
      distance: distanceBetween(point, polyline[0]),
      offset: 0,
    };
  }

  let best: PolylineProjection | null = null;
  let traversed = 0;
  for (let index = 0; index < polyline.length - 1; index += 1) {
    const start = polyline[index];
    const end = polyline[index + 1];
    const direction = subtractPoints(end, start);
    const lengthSq = dotProduct(direction, direction);
    if (nearlyEqual(lengthSq, 0)) {
      continue;
    }

    const rawT = dotProduct(subtractPoints(point, start), direction) / lengthSq;
    const t = Math.min(1, Math.max(0, rawT));
    const projected = lerpPoint(start, end, t);
    const distance = distanceBetween(point, projected);
    const offset = traversed + (distanceBetween(start, end) * t);
    if (!best || distance < best.distance) {
      best = { point: projected, segmentIndex: index, t, distance, offset };
    }
    traversed += distanceBetween(start, end);
  }

  return best;
}

export function collectPolylineClosureCandidates(
  point: Point,
  polyline: Point[],
  options: {
    vertexLimit?: number;
    vertexSnapDistance?: number;
  } = {}
) {
  const projection = projectPointToPolyline(point, polyline);
  if (!projection) {
    return [] as Point[];
  }

  const vertexLimit = Math.max(0, options.vertexLimit ?? 6);
  const vertexSnapDistance = Math.max(0, options.vertexSnapDistance ?? 8);
  const verticesByDistance = polyline
    .map((candidate) => ({
      point: candidate,
      distanceToInput: distanceBetween(point, candidate),
      distanceToProjection: distanceBetween(projection.point, candidate),
    }));
  const projectionAdjacentVertices = verticesByDistance
    .filter((candidate) => candidate.distanceToProjection <= vertexSnapDistance + EDGE_EPSILON)
    .sort((left, right) => left.distanceToInput - right.distanceToInput)
    .slice(0, vertexLimit)
    .map((candidate) => candidate.point);
  const nearestVertices = verticesByDistance
    .sort((left, right) => left.distanceToInput - right.distanceToInput)
    .slice(0, vertexLimit)
    .map((candidate) => candidate.point);

  const deduped: Point[] = [];
  for (const candidate of [projection.point, ...projectionAdjacentVertices, ...nearestVertices]) {
    if (!deduped.some((existing) => pointsEqual(existing, candidate))) {
      deduped.push(candidate);
    }
  }

  return deduped.sort(
    (left, right) => distanceBetween(point, left) - distanceBetween(point, right)
  );
}

interface SegmentBounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

function getSegmentBounds(segment: SegmentLike): SegmentBounds {
  return {
    minX: Math.min(segment.start.x, segment.end.x),
    minY: Math.min(segment.start.y, segment.end.y),
    maxX: Math.max(segment.start.x, segment.end.x),
    maxY: Math.max(segment.start.y, segment.end.y),
  };
}

function boundsOverlap(left: SegmentBounds, right: SegmentBounds) {
  return !(
    left.maxX < right.minX - EDGE_EPSILON ||
    right.maxX < left.minX - EDGE_EPSILON ||
    left.maxY < right.minY - EDGE_EPSILON ||
    right.maxY < left.minY - EDGE_EPSILON
  );
}

export function buildSpatialSegmentCandidatePairs(
  segments: SegmentLike[],
  options: {
    cellSize?: number;
  } = {}
) {
  const cellSize = Math.max(options.cellSize ?? SEGMENT_SEARCH_CELL_SIZE_PX, EDGE_EPSILON);
  const bounds = segments.map((segment) => getSegmentBounds(segment));
  const buckets = new Map<string, number[]>();

  for (let index = 0; index < segments.length; index += 1) {
    const segmentBounds = bounds[index];
    const minCellX = Math.floor(segmentBounds.minX / cellSize);
    const maxCellX = Math.floor(segmentBounds.maxX / cellSize);
    const minCellY = Math.floor(segmentBounds.minY / cellSize);
    const maxCellY = Math.floor(segmentBounds.maxY / cellSize);

    for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
        const bucketKey = `${cellX}:${cellY}`;
        const bucket = buckets.get(bucketKey);
        if (bucket) {
          bucket.push(index);
          continue;
        }
        buckets.set(bucketKey, [index]);
      }
    }
  }

  const pairs: Array<[number, number]> = [];
  const seenPairs = new Set<string>();
  for (const bucket of buckets.values()) {
    if (bucket.length < 2) {
      continue;
    }
    let minBucketX = Number.POSITIVE_INFINITY;
    let maxBucketX = Number.NEGATIVE_INFINITY;
    let minBucketY = Number.POSITIVE_INFINITY;
    let maxBucketY = Number.NEGATIVE_INFINITY;
    for (const segmentIndex of bucket) {
      const segmentBounds = bounds[segmentIndex];
      minBucketX = Math.min(minBucketX, segmentBounds.minX);
      maxBucketX = Math.max(maxBucketX, segmentBounds.maxX);
      minBucketY = Math.min(minBucketY, segmentBounds.minY);
      maxBucketY = Math.max(maxBucketY, segmentBounds.maxY);
    }

    const primaryAxis = maxBucketX - minBucketX >= maxBucketY - minBucketY ? "x" : "y";
    const secondaryAxis = primaryAxis === "x" ? "y" : "x";
    const orderedBucket = bucket.slice().sort((leftSegmentIndex, rightSegmentIndex) => {
      const leftBounds = bounds[leftSegmentIndex];
      const rightBounds = bounds[rightSegmentIndex];
      const primaryDelta =
        (primaryAxis === "x" ? leftBounds.minX : leftBounds.minY) -
        (primaryAxis === "x" ? rightBounds.minX : rightBounds.minY);
      if (!nearlyEqual(primaryDelta, 0)) {
        return primaryDelta;
      }
      const secondaryDelta =
        (secondaryAxis === "x" ? leftBounds.minX : leftBounds.minY) -
        (secondaryAxis === "x" ? rightBounds.minX : rightBounds.minY);
      if (!nearlyEqual(secondaryDelta, 0)) {
        return secondaryDelta;
      }
      return leftSegmentIndex - rightSegmentIndex;
    });

    for (let leftIndex = 0; leftIndex < orderedBucket.length; leftIndex += 1) {
      const leftSegmentIndex = orderedBucket[leftIndex];
      const leftBounds = bounds[leftSegmentIndex];
      const leftMaxPrimary = primaryAxis === "x" ? leftBounds.maxX : leftBounds.maxY;

      for (let rightIndex = leftIndex + 1; rightIndex < orderedBucket.length; rightIndex += 1) {
        const rightSegmentIndex = orderedBucket[rightIndex];
        const rightBounds = bounds[rightSegmentIndex];
        const rightMinPrimary = primaryAxis === "x" ? rightBounds.minX : rightBounds.minY;
        if (rightMinPrimary > leftMaxPrimary + EDGE_EPSILON) {
          break;
        }
        const pairStart = Math.min(leftSegmentIndex, rightSegmentIndex);
        const pairEnd = Math.max(leftSegmentIndex, rightSegmentIndex);
        if (!boundsOverlap(bounds[pairStart], bounds[pairEnd])) {
          continue;
        }
        const pairKey = `${pairStart}:${pairEnd}`;
        if (seenPairs.has(pairKey)) {
          continue;
        }
        seenPairs.add(pairKey);
        pairs.push([pairStart, pairEnd]);
      }
    }
  }

  return pairs;
}

export function normalizeClosedRing(points: Point[]) {
  const deduped = dedupeConsecutive(points);
  if (deduped.length < 3) {
    return null;
  }
  const closed = pointsEqual(deduped[0], deduped[deduped.length - 1])
    ? deduped
    : [...deduped, deduped[0]];
  return closed.length >= 4 ? closed : null;
}

export function openRing(points: Point[]) {
  if (points.length === 0) {
    return [];
  }
  return pointsEqual(points[0], points[points.length - 1]) ? points.slice(0, -1) : points.slice();
}

export function normalizeTrustedClosedRing(points: Point[]) {
  const ring = normalizeClosedRing(collapseBacktracks(points));
  if (!ring) {
    return null;
  }

  const vertices = openRing(ring);
  if (vertices.length < 3) {
    return null;
  }

  let total = 0;
  for (let index = 0; index < vertices.length; index += 1) {
    const current = vertices[index];
    const next = vertices[(index + 1) % vertices.length];
    total += (current.x * next.y) - (next.x * current.y);
  }

  return Math.abs(total / 2) > EDGE_EPSILON ? ring : null;
}

export function signedRingArea(points: Point[]) {
  const ring = normalizeClosedRing(points);
  if (!ring) {
    return 0;
  }
  const vertices = openRing(ring);
  if (vertices.length < 3) {
    return 0;
  }

  let total = 0;
  for (let index = 0; index < vertices.length; index += 1) {
    const current = vertices[index];
    const next = vertices[(index + 1) % vertices.length];
    total += (current.x * next.y) - (next.x * current.y);
  }
  return total / 2;
}

export function polygonEdges(points: Point[]) {
  const ring = normalizeClosedRing(points);
  if (!ring) {
    return [] as Array<[Point, Point]>;
  }
  const vertices = openRing(ring);
  const edges: Array<[Point, Point]> = [];
  for (let index = 0; index < vertices.length; index += 1) {
    edges.push([vertices[index], vertices[(index + 1) % vertices.length]]);
  }
  return edges;
}

export function pointInPolygonInclusive(point: Point, polygon: Point[]) {
  for (const [start, end] of polygonEdges(polygon)) {
    if (pointOnSegment(point, start, end)) {
      return true;
    }
  }
  return pointInPolygon(point, openRing(polygon));
}

export function uniqueSorted(values: number[]) {
  return values
    .sort((left, right) => left - right)
    .filter((value, index, sorted) => index === 0 || !nearlyEqual(value, sorted[index - 1]));
}

function buildPathSegments(points: Point[], closed: boolean) {
  const segments: SegmentLike[] = [];
  if (closed) {
    const ring = normalizeClosedRing(points);
    if (!ring) {
      return segments;
    }
    const vertices = openRing(ring);
    for (let index = 0; index < vertices.length; index += 1) {
      const start = vertices[index];
      const end = vertices[(index + 1) % vertices.length];
      if (!pointsEqual(start, end)) {
        segments.push({ start, end });
      }
    }
    return segments;
  }

  const path = dedupeConsecutive(points);
  for (let index = 0; index < path.length - 1; index += 1) {
    const start = path[index];
    const end = path[index + 1];
    if (!pointsEqual(start, end)) {
      segments.push({ start, end });
    }
  }
  return segments;
}

function segmentsAreAdjacent(
  leftIndex: number,
  rightIndex: number,
  segmentCount: number,
  closed: boolean
) {
  return (
    rightIndex === leftIndex + 1 ||
    (closed && leftIndex === 0 && rightIndex === segmentCount - 1)
  );
}

function isAllowedAdjacentIntersection(
  segments: SegmentLike[],
  leftIndex: number,
  rightIndex: number,
  intersection: SegmentIntersection,
  closed: boolean
) {
  if (intersection.kind !== "point") {
    return false;
  }
  if (!segmentsAreAdjacent(leftIndex, rightIndex, segments.length, closed)) {
    return false;
  }

  if (rightIndex === leftIndex + 1) {
    return (
      pointsEqual(intersection.point, segments[leftIndex].end) &&
      pointsEqual(intersection.point, segments[rightIndex].start)
    );
  }

  return (
    closed &&
    pointsEqual(intersection.point, segments[leftIndex].start) &&
    pointsEqual(intersection.point, segments[rightIndex].end)
  );
}

function hasInvalidSegmentSelfIntersection(
  segments: SegmentLike[],
  options: {
    closed?: boolean;
    cellSize?: number;
  } = {}
) {
  const closed = options.closed ?? false;
  const cellSize = Math.max(options.cellSize ?? SEGMENT_SEARCH_CELL_SIZE_PX, EDGE_EPSILON);
  const bounds = segments.map((segment) => getSegmentBounds(segment));
  const buckets = new Map<string, number[]>();

  for (let index = 0; index < segments.length; index += 1) {
    const segmentBounds = bounds[index];
    const minCellX = Math.floor(segmentBounds.minX / cellSize);
    const maxCellX = Math.floor(segmentBounds.maxX / cellSize);
    const minCellY = Math.floor(segmentBounds.minY / cellSize);
    const maxCellY = Math.floor(segmentBounds.maxY / cellSize);
    const seenCandidates = new Set<number>();

    for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
        const bucketKey = `${cellX}:${cellY}`;
        for (const candidateIndex of buckets.get(bucketKey) ?? []) {
          if (seenCandidates.has(candidateIndex)) {
            continue;
          }
          seenCandidates.add(candidateIndex);
          if (!boundsOverlap(segmentBounds, bounds[candidateIndex])) {
            continue;
          }

          const intersection = segmentIntersection(
            segments[candidateIndex].start,
            segments[candidateIndex].end,
            segments[index].start,
            segments[index].end
          );
          if (!intersection) {
            continue;
          }
          if (
            !isAllowedAdjacentIntersection(
              segments,
              candidateIndex,
              index,
              intersection,
              closed
            )
          ) {
            return true;
          }
        }
      }
    }

    for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
        const bucketKey = `${cellX}:${cellY}`;
        const bucket = buckets.get(bucketKey);
        if (bucket) {
          bucket.push(index);
          continue;
        }
        buckets.set(bucketKey, [index]);
      }
    }
  }

  return false;
}

export function hasInvalidPolylineSelfIntersection(
  points: Point[],
  options: {
    closed?: boolean;
    cellSize?: number;
  } = {}
) {
  return hasInvalidSegmentSelfIntersection(
    buildPathSegments(points, options.closed ?? false),
    options
  );
}

export function isSinglePolygonRing(points: Point[]) {
  const ring = normalizeClosedRing(points);
  if (!ring) {
    return false;
  }

  if (openRing(ring).length < 3 || Math.abs(signedRingArea(ring)) <= EDGE_EPSILON) {
    return false;
  }

  return !hasInvalidPolylineSelfIntersection(ring, { closed: true });
}

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

function pointKey(point: Point) {
  return `${point.x.toFixed(6)}:${point.y.toFixed(6)}`;
}

function findLastPointIndex(points: Point[], point: Point) {
  for (let index = points.length - 1; index >= 0; index -= 1) {
    if (pointsEqual(points[index], point)) {
      return index;
    }
  }
  return -1;
}

function canonicalCycleKey(points: Point[]) {
  const open = openRing(points);
  if (open.length === 0) {
    return "";
  }

  const forward = open.map(pointKey);
  const backward = [...forward].reverse();
  let best = "";

  const applyBestRotation = (values: string[]) => {
    for (let index = 0; index < values.length; index += 1) {
      const rotation = [...values.slice(index), ...values.slice(0, index)].join("|");
      if (!best || rotation < best) {
        best = rotation;
      }
    }
  };

  applyBestRotation(forward);
  applyBestRotation(backward);
  return best;
}

function overlapSplitPoints(left: SegmentLike, right: SegmentLike) {
  const overlapPoints = [left.start, left.end, right.start, right.end].filter(
    (point) =>
      pointOnSegment(point, left.start, left.end) &&
      pointOnSegment(point, right.start, right.end)
  );
  const unique = overlapPoints.filter(
    (point, index, points) =>
      points.findIndex((candidate) => pointsEqual(candidate, point)) === index
  );
  return unique.sort(
    (first, second) =>
      segmentParameter(first, left.start, left.end) -
      segmentParameter(second, left.start, left.end)
  );
}

function splitClosedWalkAtIntersections(points: Point[]) {
  const ring = normalizeClosedRing(points);
  if (!ring) {
    return null;
  }

  const segments = buildPathSegments(ring, true);
  if (segments.length === 0) {
    return null;
  }

  let hadOverlap = false;
  const splitPoints = segments.map((segment) => [segment.start, segment.end]);
  for (const [leftIndex, rightIndex] of buildSpatialSegmentCandidatePairs(segments)) {
    const left = segments[leftIndex];
    const right = segments[rightIndex];
    const intersection = segmentIntersection(
      left.start,
      left.end,
      right.start,
      right.end
    );
    if (!intersection) {
      continue;
    }
    if (intersection.kind === "overlap") {
      hadOverlap = true;
      for (const point of overlapSplitPoints(left, right)) {
        splitPoints[leftIndex]?.push(point);
        splitPoints[rightIndex]?.push(point);
      }
      continue;
    }

    splitPoints[leftIndex]?.push(intersection.point);
    splitPoints[rightIndex]?.push(intersection.point);
  }

  const walk: Point[] = [];
  for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
    const segment = segments[segmentIndex];
    const orderedPoints = (splitPoints[segmentIndex] ?? [])
      .slice()
      .sort(
        (left, right) =>
          segmentParameter(left, segment.start, segment.end) -
          segmentParameter(right, segment.start, segment.end)
      )
      .filter(
        (point, index, ordered) =>
          index === 0 || !pointsEqual(point, ordered[index - 1])
      );

    for (const point of orderedPoints) {
      if (!walk.length || !pointsEqual(walk[walk.length - 1], point)) {
        walk.push(point);
      }
    }
  }

  if (!walk.length) {
    return null;
  }
  if (!pointsEqual(walk[0], walk[walk.length - 1])) {
    walk.push(walk[0]);
  }
  return {
    walk: dedupeConsecutive(walk),
    hadOverlap,
  };
}

function extractSimpleCyclesFromWalk(points: Point[]) {
  const walk = dedupeConsecutive(points);
  if (walk.length < 4) {
    return [] as Point[][];
  }

  const stack: Point[] = [];
  const cycles: Point[][] = [];

  for (const point of walk) {
    const seenIndex = findLastPointIndex(stack, point);
    if (seenIndex === -1) {
      stack.push(point);
      continue;
    }

    const cycle = normalizeClosedRing([...stack.slice(seenIndex), point]);
    if (cycle) {
      cycles.push(cycle);
    }
    stack.splice(seenIndex + 1);
  }

  return cycles;
}

function resolvePolygonCycles(points: Point[]) {
  const ring = normalizeClosedRing(collapseBacktracks(points));
  if (!ring) {
    return {
      cycles: [] as Point[][],
      hadOverlap: false,
    };
  }
  if (isSinglePolygonRing(ring)) {
    return {
      cycles: [ring],
      hadOverlap: false,
    };
  }

  const split = splitClosedWalkAtIntersections(ring);
  if (!split) {
    return {
      cycles: [] as Point[][],
      hadOverlap: false,
    };
  }

  const uniqueCycles: Point[][] = [];
  const seen = new Set<string>();
  for (const cycle of extractSimpleCyclesFromWalk(split.walk)) {
    if (!isSinglePolygonRing(cycle)) {
      continue;
    }
    const key = canonicalCycleKey(cycle);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    uniqueCycles.push(cycle);
  }

  return {
    cycles: uniqueCycles.sort(
      (left, right) => Math.abs(signedRingArea(right)) - Math.abs(signedRingArea(left))
    ),
    hadOverlap: split.hadOverlap,
  };
}

export function resolveAcceptedPolygonRings(points: Point[]) {
  const { cycles, hadOverlap } = resolvePolygonCycles(points);
  if (
    hadOverlap &&
    cycles.every(
      (cycle) =>
        Math.abs(signedRingArea(cycle)) + EDGE_EPSILON < MIN_SIGNIFICANT_POLYGON_AREA_PX
    )
  ) {
    return [] as Point[][];
  }
  if (cycles.length <= 1) {
    return cycles;
  }
  return cycles.filter(
    (cycle) =>
      Math.abs(signedRingArea(cycle)) + EDGE_EPSILON >= MIN_SIGNIFICANT_POLYGON_AREA_PX
  );
}

export function resolveSingleAcceptedPolygonRing(points: Point[]) {
  const polygons = resolveAcceptedPolygonRings(points);
  return polygons[0] ?? null;
}
