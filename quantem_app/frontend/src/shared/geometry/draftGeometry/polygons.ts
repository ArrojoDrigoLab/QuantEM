import type { DisplayGeometry } from "@/shared/geometry/draftGeometry/types";
import type { Point } from "@/utils/geometry";
import {
  resolveSingleCycleCloseabilityAsync,
  resolveSingleCycleCloseability,
} from "@/shared/geometry/draftGeometry/closeability";
import {
  collectPolylineClosureCandidates,
  dedupeConsecutive,
  INVALID_CLOSE_POLYGON_MESSAGE,
  normalizeClosedRing,
  normalizeTrustedClosedRing,
  pointsEqual,
  resolveSingleAcceptedPolygonRing,
  signedRingArea,
} from "@/shared/geometry/draftGeometry/shared";
import type {
  DraftPolygon,
  DraftSegment,
} from "@/shared/geometry/draftGeometry/types";

export type DraftPolygonCloseability =
  | { kind: "not_ready" }
  | { kind: "ok"; ring: Point[] }
  | { kind: "error"; message: string };

function appendUniquePoints(target: Point[], candidates: Array<Point | null | undefined>) {
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    if (!target.some((existing) => pointsEqual(existing, candidate))) {
      target.push(candidate);
    }
  }
  return target;
}

export function appendDraftSection(
  polygons: DraftPolygon[],
  section: DraftSegment
) {
  if (!polygons.length || polygons[polygons.length - 1].closed) {
    return [
      ...polygons,
      {
        id: section.id,
        segments: [section],
        closed: false,
      },
    ];
  }

  const next = polygons.slice();
  const last = next[next.length - 1];
  next[next.length - 1] = {
    ...last,
    segments: [...last.segments, section],
  };
  return next;
}

function buildDraftPath(segments: DraftSegment[]) {
  let path: Point[] = [];

  for (const segment of segments) {
    const segmentPoints = dedupeConsecutive(segment.points);
    if (segmentPoints.length < 2) {
      continue;
    }

    if (!path.length) {
      path = [...segmentPoints];
      continue;
    }

    const pathEnd = path[path.length - 1];
    const segmentStart = segmentPoints[0];
    if (!pointsEqual(pathEnd, segmentStart)) {
      path.push(segmentStart);
    }
    path.push(...segmentPoints.slice(1));
  }

  return dedupeConsecutive(path);
}

function createClosedDraftPolygonFromRing(
  ringInput: Point[],
  polygonId: string,
  idFactory: () => string
): DraftPolygon | null {
  const ring = normalizeClosedRing(ringInput);
  if (!ring) {
    return null;
  }

  const open = pointsEqual(ring[0], ring[ring.length - 1]) ? ring.slice(0, -1) : ring;
  if (open.length < 3) {
    return null;
  }

  const firstPoint = open[0];
  const lastPoint = open[open.length - 1];
  return {
    id: polygonId,
    closed: true,
    segments: [
      {
        id: idFactory(),
        kind: "section",
        points: open,
        endedOutsidePatch: false,
      },
      {
        id: idFactory(),
        kind: "closing",
        points: dedupeConsecutive([lastPoint, firstPoint]),
        endedOutsidePatch: false,
      },
    ],
  };
}

function resolveOpenDraftPolygonCloseability(
  polygon: DraftPolygon
): DraftPolygonCloseability {
  const path = buildDraftPath(polygon.segments);
  if (path.length < 3) {
    return { kind: "not_ready" };
  }

  const priorPath = buildDraftPath(polygon.segments.slice(0, -1));
  const endPoint = path[path.length - 1];
  const closureCandidates = appendUniquePoints(
    [],
    [
      ...(endPoint && priorPath.length >= 2
        ? collectPolylineClosureCandidates(endPoint, priorPath)
        : []),
      path[0] ?? null,
    ]
  );
  if (!endPoint || closureCandidates.length === 0) {
    return { kind: "not_ready" };
  }

  let bestCloseability: DraftPolygonCloseability | null = null;
  let bestArea = -1;
  let firstError: DraftPolygonCloseability | null = null;
  for (const closurePoint of closureCandidates) {
    const closeability = resolveSingleCycleCloseability([path], {
      start: endPoint,
      end: closurePoint,
    });
    if (closeability.kind === "ok") {
      const area = Math.abs(signedRingArea(closeability.ring));
      if (!bestCloseability || area > bestArea) {
        bestCloseability = closeability;
        bestArea = area;
      }
      continue;
    }
    if (!firstError && closeability.kind === "error") {
      firstError = closeability;
    }
  }

  return bestCloseability ?? firstError ?? { kind: "not_ready" };
}

async function resolveOpenDraftPolygonCloseabilityAsync(
  polygon: DraftPolygon
): Promise<DraftPolygonCloseability> {
  const path = buildDraftPath(polygon.segments);
  if (path.length < 3) {
    return { kind: "not_ready" };
  }

  const priorPath = buildDraftPath(polygon.segments.slice(0, -1));
  const endPoint = path[path.length - 1];
  const closureCandidates = appendUniquePoints(
    [],
    [
      ...(endPoint && priorPath.length >= 2
        ? collectPolylineClosureCandidates(endPoint, priorPath)
        : []),
      path[0] ?? null,
    ]
  );
  if (!endPoint || closureCandidates.length === 0) {
    return { kind: "not_ready" };
  }

  let bestCloseability: DraftPolygonCloseability | null = null;
  let bestArea = -1;
  let firstError: DraftPolygonCloseability | null = null;
  for (const closurePoint of closureCandidates) {
    const closeability = await resolveSingleCycleCloseabilityAsync([path], {
      start: endPoint,
      end: closurePoint,
    });
    if (closeability.kind === "ok") {
      const area = Math.abs(signedRingArea(closeability.ring));
      if (!bestCloseability || area > bestArea) {
        bestCloseability = closeability;
        bestArea = area;
      }
      continue;
    }
    if (!firstError && closeability.kind === "error") {
      firstError = closeability;
    }
  }

  return bestCloseability ?? firstError ?? { kind: "not_ready" };
}

export function resolveDraftPolygonCloseability(
  polygon: DraftPolygon | null
): DraftPolygonCloseability {
  if (!polygon) {
    return { kind: "not_ready" };
  }
  if (polygon.closed) {
    const ring = buildClosedPolygonRing(polygon);
    if (!ring) {
      return { kind: "not_ready" };
    }
    const resolvedRing = polygon.skipCycleResolution
      ? normalizeTrustedClosedRing(ring)
      : resolveSingleAcceptedPolygonRing(ring);
    return resolvedRing
      ? { kind: "ok", ring: resolvedRing }
      : { kind: "error", message: INVALID_CLOSE_POLYGON_MESSAGE };
  }
  return resolveOpenDraftPolygonCloseability(polygon);
}

export async function resolveDraftPolygonCloseabilityAsync(
  polygon: DraftPolygon | null
): Promise<DraftPolygonCloseability> {
  if (!polygon) {
    return { kind: "not_ready" };
  }
  if (polygon.closed) {
    return resolveDraftPolygonCloseability(polygon);
  }
  return resolveOpenDraftPolygonCloseabilityAsync(polygon);
}

export function canCloseDraftPolygon(polygon: DraftPolygon | null) {
  return resolveDraftPolygonCloseability(polygon).kind === "ok";
}

export function hasDraftPolygonCloseCandidate(
  polygon: DraftPolygon | null
) {
  if (!polygon || polygon.closed) {
    return false;
  }
  return buildDraftPath(polygon.segments).length >= 3;
}

export function closeDraftPolygon(
  polygons: DraftPolygon[],
  idFactory: () => string,
  ringOverride?: Point[]
) {
  if (!polygons.length) {
    return polygons;
  }
  const lastPolygon = polygons[polygons.length - 1];
  if (lastPolygon.closed || lastPolygon.segments.length === 0) {
    return polygons;
  }

  const closeability = ringOverride
    ? { kind: "ok" as const, ring: ringOverride }
    : resolveOpenDraftPolygonCloseability(lastPolygon);
  if (closeability.kind !== "ok") {
    return polygons;
  }

  const closedPolygon = createClosedDraftPolygonFromRing(
    closeability.ring,
    lastPolygon.id,
    idFactory
  );
  if (!closedPolygon) {
    return polygons;
  }

  return [...polygons.slice(0, -1), closedPolygon];
}

export function deleteLastDraftSegment(polygons: DraftPolygon[]) {
  if (!polygons.length) {
    return polygons;
  }

  const next = polygons.slice();
  const lastPolygon = next[next.length - 1];
  if (lastPolygon.segments.length === 0) {
    next.pop();
    return next;
  }

  const remainingSegments = lastPolygon.segments.slice(0, -1);
  if (remainingSegments.length === 0) {
    next.pop();
    return next;
  }

  next[next.length - 1] = {
    ...lastPolygon,
    closed: false,
    segments: remainingSegments,
  };
  return next;
}

export function getActiveDraftPolygon(polygons: DraftPolygon[]) {
  if (!polygons.length) {
    return null;
  }
  const lastPolygon = polygons[polygons.length - 1];
  return lastPolygon.closed ? null : lastPolygon;
}

export function getClosedDraftPolygons(polygons: DraftPolygon[]) {
  return polygons.filter((polygon) => polygon.closed);
}

export function countCompletedSections(polygons: DraftPolygon[]) {
  return polygons.reduce((total, polygon) => total + polygon.segments.length, 0);
}

export function flattenDraftSegments(polygons: DraftPolygon[]) {
  return polygons.flatMap((polygon) => polygon.segments);
}

export function buildClosedPolygonRing(polygon: DraftPolygon) {
  if (!polygon.closed) {
    return null;
  }

  let ring: Point[] = [];
  for (const segment of polygon.segments) {
    const segmentPoints = dedupeConsecutive(segment.points);
    if (segmentPoints.length < 2) {
      continue;
    }

    if (!ring.length) {
      ring = [...segmentPoints];
      continue;
    }

    const ringEnd = ring[ring.length - 1];
    const segmentStart = segmentPoints[0];
    if (!pointsEqual(ringEnd, segmentStart)) {
      ring.push(segmentStart);
    }
    ring.push(...segmentPoints.slice(1));
  }

  ring = dedupeConsecutive(ring);
  if (ring.length < 3) {
    return null;
  }

  const first = ring[0];
  const last = ring[ring.length - 1];
  if (!pointsEqual(first, last)) {
    ring.push(first);
  }

  ring = dedupeConsecutive(ring);
  if (ring.length < 4) {
    ring.push(first);
  }
  if (!pointsEqual(ring[0], ring[ring.length - 1])) {
    ring.push(ring[0]);
  }
  return ring.length >= 4 ? ring : null;
}

export function buildConfirmPolygons(polygons: DraftPolygon[]) {
  return getClosedDraftPolygons(polygons)
    .map((polygon) => {
      const ring = buildClosedPolygonRing(polygon);
      if (!ring) {
        return null;
      }
      return polygon.skipCycleResolution
        ? normalizeTrustedClosedRing(ring)
        : resolveSingleAcceptedPolygonRing(ring);
    })
    .filter((ring): ring is Point[] => Array.isArray(ring) && ring.length >= 4)
    .map((ring) => normalizeTrustedClosedRing(ring))
    .filter((ring): ring is Point[] => Array.isArray(ring) && ring.length >= 4)
    .map((ring) => ring.map((point) => [point.x, point.y] as [number, number]));
}

function createAutomaticPrefillPolygon(
  ring: Array<[number, number]>,
  idFactory: (prefix: string) => string
): DraftPolygon | null {
  const normalized = normalizeTrustedClosedRing(
    dedupeConsecutive(ring.map(([x, y]) => ({ x, y })))
  );
  if (!normalized) {
    return null;
  }

  const open = normalized.slice(0, -1);
  if (open.length < 3) {
    return null;
  }

  const firstPoint = open[0];
  const lastPoint = open[open.length - 1];
  return {
    id: idFactory("automatic-polygon"),
    closed: true,
    segments: [
      {
        id: idFactory("automatic-section"),
        kind: "section",
        points: open,
        endedOutsidePatch: false,
      },
      {
        id: idFactory("automatic-closing"),
        kind: "closing",
        points: dedupeConsecutive([lastPoint, firstPoint]),
        endedOutsidePatch: false,
      },
    ],
  };
}

export function buildAutomaticPrefillPolygons(
  displayGeometry: DisplayGeometry,
  idFactory: (prefix: string) => string
) {
  return displayGeometry.polygons
    .map((ring) => createAutomaticPrefillPolygon(ring, idFactory))
    .filter((polygon): polygon is DraftPolygon => polygon !== null);
}
