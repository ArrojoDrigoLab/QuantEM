import type { Point } from "@/utils/geometry";
import {
  crossProduct,
  dedupeConsecutive,
  dotProduct,
  lerpPoint,
  nearlyEqual,
  normalizeClosedRing,
  normalizeTrustedClosedRing,
  openRing,
  pointInPolygonInclusive,
  pointsEqual,
  polygonEdges,
  subtractPoints,
  uniqueSorted,
  EDGE_EPSILON,
} from "@/shared/geometry/draftGeometry/shared";
import { buildClosedPolygonRing } from "@/shared/geometry/draftGeometry/polygons";
import { projectPointToPolyline } from "@/shared/geometry/draftGeometry/repair";
import type {
  CutDraftResult,
  RingCutResult,
} from "@/shared/geometry/draftGeometry/internalTypes";
import type {
  DraftPolygon,
  DraftRepairSession,
} from "@/shared/geometry/draftGeometry/types";

function segmentIntersectionTs(
  start: Point,
  end: Point,
  left: Point,
  right: Point
) {
  const directionA = subtractPoints(end, start);
  const directionB = subtractPoints(right, left);
  const diff = subtractPoints(left, start);
  const denominator = crossProduct(directionA, directionB);

  if (nearlyEqual(denominator, 0)) {
    if (!nearlyEqual(crossProduct(diff, directionA), 0)) {
      return [] as number[];
    }
    const lengthSq = dotProduct(directionA, directionA);
    if (nearlyEqual(lengthSq, 0)) {
      return [] as number[];
    }
    const t0 = dotProduct(subtractPoints(left, start), directionA) / lengthSq;
    const t1 = dotProduct(subtractPoints(right, start), directionA) / lengthSq;
    const overlapStart = Math.max(0, Math.min(t0, t1));
    const overlapEnd = Math.min(1, Math.max(t0, t1));
    if (overlapEnd < overlapStart - EDGE_EPSILON) {
      return [] as number[];
    }
    return uniqueSorted(
      [overlapStart, overlapEnd].filter((value) => value >= -EDGE_EPSILON && value <= 1 + EDGE_EPSILON)
    );
  }

  const t = crossProduct(diff, directionB) / denominator;
  const u = crossProduct(diff, directionA) / denominator;
  if (
    t < -EDGE_EPSILON ||
    t > 1 + EDGE_EPSILON ||
    u < -EDGE_EPSILON ||
    u > 1 + EDGE_EPSILON
  ) {
    return [] as number[];
  }
  return [Math.min(1, Math.max(0, t))];
}

function mergeOrderedRuns(runs: Point[][]) {
  if (runs.length === 0) {
    return [] as Point[][];
  }

  const merged = runs
    .map((run) => dedupeConsecutive(run))
    .filter((run) => run.length >= 2);
  if (merged.length <= 1) {
    return merged;
  }

  const first = merged[0];
  const last = merged[merged.length - 1];
  if (!first[0] || !last[last.length - 1]) {
    return merged;
  }
  if (!pointsEqual(first[0], last[last.length - 1])) {
    return merged;
  }

  return [
    dedupeConsecutive([
      ...last,
      ...first.slice(1),
    ]),
    ...merged.slice(1, -1),
  ].filter((run) => run.length >= 2);
}

export function cutClosedRingWithLasso(ringInput: Point[], lassoInput: Point[]): RingCutResult {
  const ring = normalizeTrustedClosedRing(ringInput);
  const lasso = normalizeTrustedClosedRing(lassoInput);
  if (!ring || !lasso) {
    return { kind: "untouched" };
  }

  const lassoSegments = polygonEdges(lasso);
  const ringVertices = openRing(ring);
  const runs: Point[][] = [];
  let currentRun: Point[] = [];
  let removedAnyBoundary = false;

  for (let index = 0; index < ringVertices.length; index += 1) {
    const start = ringVertices[index];
    const end = ringVertices[(index + 1) % ringVertices.length];
    const splitTs = [0, 1];

    for (const [lassoStart, lassoEnd] of lassoSegments) {
      splitTs.push(...segmentIntersectionTs(start, end, lassoStart, lassoEnd));
    }

    const sortedTs = uniqueSorted(splitTs);
    for (let splitIndex = 0; splitIndex < sortedTs.length - 1; splitIndex += 1) {
      const t0 = sortedTs[splitIndex];
      const t1 = sortedTs[splitIndex + 1];
      if (t1 - t0 <= EDGE_EPSILON) {
        continue;
      }

      const clippedStart = lerpPoint(start, end, t0);
      const clippedEnd = lerpPoint(start, end, t1);
      const midpoint = lerpPoint(start, end, (t0 + t1) / 2);
      const insideLasso = pointInPolygonInclusive(midpoint, lasso);

      if (insideLasso) {
        removedAnyBoundary = true;
        if (currentRun.length >= 2) {
          runs.push(dedupeConsecutive(currentRun));
        }
        currentRun = [];
        continue;
      }

      if (!currentRun.length) {
        currentRun = [clippedStart, clippedEnd];
      } else if (pointsEqual(currentRun[currentRun.length - 1], clippedStart)) {
        currentRun.push(clippedEnd);
      } else {
        runs.push(dedupeConsecutive(currentRun));
        currentRun = [clippedStart, clippedEnd];
      }
    }
  }

  if (currentRun.length >= 2) {
    runs.push(dedupeConsecutive(currentRun));
  }

  if (!removedAnyBoundary) {
    return { kind: "untouched" };
  }

  const mergedRuns = mergeOrderedRuns(runs);
  if (mergedRuns.length === 0) {
    return { kind: "error", message: "That cut would remove the entire polygon." };
  }
  if (mergedRuns.length !== 1) {
    return { kind: "error", message: "That cut would create multiple repair gaps." };
  }

  const remainingPath = dedupeConsecutive(mergedRuns[0]);
  if (
    remainingPath.length < 2 ||
    pointsEqual(remainingPath[0], remainingPath[remainingPath.length - 1])
  ) {
    return { kind: "error", message: "That cut would create multiple repair gaps." };
  }

  return { kind: "ok", remainingPath };
}

export function extractClosedRingPathWithinLasso(ringInput: Point[], lassoInput: Point[]) {
  const ring = normalizeTrustedClosedRing(ringInput);
  const lasso = normalizeTrustedClosedRing(lassoInput);
  if (!ring || !lasso) {
    return { kind: "untouched" as const };
  }

  const lassoSegments = polygonEdges(lasso);
  const ringVertices = openRing(ring);
  const runs: Point[][] = [];
  let currentRun: Point[] = [];
  let foundInsideBoundary = false;

  for (let index = 0; index < ringVertices.length; index += 1) {
    const start = ringVertices[index];
    const end = ringVertices[(index + 1) % ringVertices.length];
    const splitTs = [0, 1];

    for (const [lassoStart, lassoEnd] of lassoSegments) {
      splitTs.push(...segmentIntersectionTs(start, end, lassoStart, lassoEnd));
    }

    const sortedTs = uniqueSorted(splitTs);
    for (let splitIndex = 0; splitIndex < sortedTs.length - 1; splitIndex += 1) {
      const t0 = sortedTs[splitIndex];
      const t1 = sortedTs[splitIndex + 1];
      if (t1 - t0 <= EDGE_EPSILON) {
        continue;
      }

      const clippedStart = lerpPoint(start, end, t0);
      const clippedEnd = lerpPoint(start, end, t1);
      const midpoint = lerpPoint(start, end, (t0 + t1) / 2);
      const insideLasso = pointInPolygonInclusive(midpoint, lasso);

      if (!insideLasso) {
        if (currentRun.length >= 2) {
          runs.push(dedupeConsecutive(currentRun));
        }
        currentRun = [];
        continue;
      }

      foundInsideBoundary = true;
      if (!currentRun.length) {
        currentRun = [clippedStart, clippedEnd];
      } else if (pointsEqual(currentRun[currentRun.length - 1], clippedStart)) {
        currentRun.push(clippedEnd);
      } else {
        runs.push(dedupeConsecutive(currentRun));
        currentRun = [clippedStart, clippedEnd];
      }
    }
  }

  if (currentRun.length >= 2) {
    runs.push(dedupeConsecutive(currentRun));
  }

  if (!foundInsideBoundary) {
    return { kind: "untouched" as const };
  }

  const mergedRuns = mergeOrderedRuns(runs);
  if (mergedRuns.length === 0) {
    return {
      kind: "error" as const,
      message: "No replacement boundary remained inside the selected bbox.",
    };
  }
  if (mergedRuns.length !== 1) {
    return {
      kind: "error" as const,
      message: "The selected bbox crosses multiple replacement boundary sections.",
    };
  }

  const replacementPath = dedupeConsecutive(mergedRuns[0]);
  if (
    replacementPath.length < 2 ||
    pointsEqual(replacementPath[0], replacementPath[replacementPath.length - 1])
  ) {
    return {
      kind: "error" as const,
      message: "The selected bbox did not isolate one replacement boundary section.",
    };
  }

  return { kind: "ok" as const, replacementPath };
}

function activateNextRepairSession(repairSessions: DraftRepairSession[]) {
  return repairSessions.map((session, index) => ({
    ...session,
    active: index === 0,
  }));
}

export function cutDraftPolygonsWithLasso(
  polygons: DraftPolygon[],
  lassoPoints: Point[],
  lassoStart: Point,
  idFactory: (prefix: string) => string
): CutDraftResult {
  const lasso = normalizeClosedRing(lassoPoints);
  if (!lasso) {
    return { kind: "untouched" };
  }

  const untouchedPolygons: DraftPolygon[] = [];
  const touchedPolygons: Array<{
    polygon: DraftPolygon;
    remainingPath: Point[];
    distance: number;
    order: number;
  }> = [];

  for (const [index, polygon] of polygons.entries()) {
    if (!polygon.closed) {
      untouchedPolygons.push(polygon);
      continue;
    }

    const ring = buildClosedPolygonRing(polygon);
    if (!ring) {
      untouchedPolygons.push(polygon);
      continue;
    }

    const cutResult = cutClosedRingWithLasso(
      ring.map((point) => ({ x: point.x, y: point.y })),
      lasso
    );
    if (cutResult.kind === "error") {
      return cutResult;
    }
    if (cutResult.kind === "untouched") {
      untouchedPolygons.push(polygon);
      continue;
    }

    const projection = projectPointToPolyline(lassoStart, cutResult.remainingPath);
    touchedPolygons.push({
      polygon,
      remainingPath: cutResult.remainingPath,
      distance: projection?.distance ?? Number.POSITIVE_INFINITY,
      order: index,
    });
  }

  if (touchedPolygons.length === 0) {
    return { kind: "untouched" };
  }

  const repairSessions = touchedPolygons
    .sort((left, right) => left.distance - right.distance || left.order - right.order)
    .map((entry, index) => ({
      id: idFactory("repair-session"),
      sourcePolygonId: entry.polygon.id,
      remainingPath: entry.remainingPath,
      repairSegments: [],
      startAnchor: null,
      endAnchor: null,
      active: index === 0,
    }));

  return {
    kind: "ok",
    polygons: untouchedPolygons,
    repairSessions: activateNextRepairSession(repairSessions),
  };
}
