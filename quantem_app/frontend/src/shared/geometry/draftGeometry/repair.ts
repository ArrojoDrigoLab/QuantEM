import type { Point } from "@/utils/geometry";
import {
  dedupeConsecutive,
  distanceBetween,
  normalizeClosedRing,
  pointsEqual,
  projectPointToPolyline,
  type PolylineProjection,
} from "@/shared/geometry/draftGeometry/shared";
import type {
  DraftPolygon,
  DraftSegment,
  DraftRepairSession,
} from "@/shared/geometry/draftGeometry/types";

export type { PolylineProjection };
export { projectPointToPolyline };

export type RepairSessionCloseability =
  | { kind: "not_ready" }
  | { kind: "ok"; ring: Point[] }
  | { kind: "error"; message: string };

function createClosedDraftPolygonFromRing(
  ringInput: Point[],
  idFactory: (prefix: string) => string,
  polygonId: string
): DraftPolygon | null {
  const ring = normalizeClosedRing(ringInput);
  if (!ring) {
    return null;
  }

  const open = pointsEqual(ring[0], ring[ring.length - 1]) ? ring.slice(0, -1) : ring;
  if (open.length < 3) {
    return null;
  }
  const first = open[0];
  const last = open[open.length - 1];

  return {
    id: polygonId,
    closed: true,
    skipCycleResolution: true,
    segments: [
      {
        id: idFactory("repair-section"),
        kind: "section",
        points: open,
        endedOutsidePatch: false,
      },
      {
        id: idFactory("repair-closing"),
        kind: "closing",
        points: dedupeConsecutive([last, first]),
        endedOutsidePatch: false,
      },
    ],
  };
}

function buildRepairPath(segments: DraftSegment[]) {
  let path: Point[] = [];

  for (const segment of segments) {
    const deduped = dedupeConsecutive(segment.points);
    if (deduped.length < 2) {
      continue;
    }
    if (!path.length) {
      path = [...deduped];
      continue;
    }
    const last = path[path.length - 1];
    const first = deduped[0];
    if (!pointsEqual(last, first)) {
      path.push(first);
    }
    path.push(...deduped.slice(1));
  }

  return dedupeConsecutive(path);
}

function activateNextRepairSession(repairSessions: DraftRepairSession[]) {
  return repairSessions.map((session, index) => ({
    ...session,
    active: index === 0,
  }));
}

function getRepairSessionEnds(session: DraftRepairSession) {
  const first = session.remainingPath[0] ?? null;
  const last = session.remainingPath[session.remainingPath.length - 1] ?? null;
  if (!first || !last || pointsEqual(first, last)) {
    return null;
  }
  return { first, last };
}

function resolveRepairAnchors(
  session: DraftRepairSession,
  referencePoint: Point | null = null
) {
  const ends = getRepairSessionEnds(session);
  if (!ends) {
    return null;
  }

  const { first, last } = ends;
  if (session.startAnchor && session.endAnchor) {
    const matchesForward =
      pointsEqual(session.startAnchor, first) &&
      pointsEqual(session.endAnchor, last);
    const matchesReverse =
      pointsEqual(session.startAnchor, last) &&
      pointsEqual(session.endAnchor, first);
    if (matchesForward || matchesReverse) {
      return {
        start: session.startAnchor,
        end: session.endAnchor,
      };
    }
  }

  if (session.startAnchor) {
    if (pointsEqual(session.startAnchor, first)) {
      return { start: first, end: last };
    }
    if (pointsEqual(session.startAnchor, last)) {
      return { start: last, end: first };
    }
  }

  if (!referencePoint) {
    return { start: first, end: last };
  }

  const start =
    distanceBetween(referencePoint, first) <= distanceBetween(referencePoint, last)
      ? first
      : last;
  return {
    start,
    end: pointsEqual(start, first) ? last : first,
  };
}

function buildRepairReplacementPath(
  session: DraftRepairSession,
  repairPathInput?: Point[]
) {
  const repairPath = dedupeConsecutive(
    repairPathInput ?? buildRepairPath(session.repairSegments)
  );
  if (repairPath.length < 2) {
    return null;
  }

  const anchors = resolveRepairAnchors(session, repairPath[0] ?? null);
  const ends = getRepairSessionEnds(session);
  if (!anchors || !ends) {
    return null;
  }

  const anchoredRepairPath = pointsEqual(repairPath[0], anchors.start)
    ? repairPath
    : dedupeConsecutive([anchors.start, ...repairPath]);
  return dedupeConsecutive([...anchoredRepairPath, anchors.end]);
}

function buildRepairCloseRing(
  session: DraftRepairSession,
  repairPathInput?: Point[]
) {
  const replacementPath = buildRepairReplacementPath(session, repairPathInput);
  if (!replacementPath) {
    return null;
  }
  const anchors = resolveRepairAnchors(session, replacementPath[0] ?? null);
  const ends = getRepairSessionEnds(session);
  if (!anchors || !ends) {
    return null;
  }

  const boundaryPath =
    pointsEqual(anchors.start, ends.first) && pointsEqual(anchors.end, ends.last)
      ? [...session.remainingPath].reverse()
      : session.remainingPath.slice();

  return normalizeClosedRing(
    dedupeConsecutive([
      ...replacementPath,
      ...boundaryPath.slice(1),
    ])
  );
}

export function getActiveRepairSession(repairSessions: DraftRepairSession[]) {
  return repairSessions.find((session) => session.active) ?? repairSessions[0] ?? null;
}

export function countRepairSegments(repairSessions: DraftRepairSession[]) {
  return repairSessions.reduce((total, session) => total + session.repairSegments.length, 0);
}

export function resolveRepairStartAnchor(
  session: DraftRepairSession,
  point: Point
) {
  return resolveRepairAnchors(session, point)?.start ?? point;
}

export function hasRepairSessionCloseCandidate(
  session: DraftRepairSession | null
) {
  if (!session) {
    return false;
  }
  return buildRepairPath(session.repairSegments).length >= 2;
}

export function resolveRepairSessionCloseability(
  session: DraftRepairSession | null
): RepairSessionCloseability {
  if (!session) {
    return { kind: "not_ready" };
  }

  const ring = buildRepairCloseRing(session);
  if (!ring) {
    return { kind: "not_ready" };
  }
  return { kind: "ok", ring };
}

export function resolveRepairSessionReplacementPath(
  session: DraftRepairSession | null
) {
  if (!session) {
    return null;
  }
  return buildRepairReplacementPath(session);
}

export function appendRepairSection(
  repairSessions: DraftRepairSession[],
  section: DraftSegment
) {
  const activeSession = getActiveRepairSession(repairSessions);
  if (!activeSession) {
    return repairSessions;
  }

  return repairSessions.map((session) => {
    if (session.id !== activeSession.id) {
      return session;
    }

    const nextSegments = [...session.repairSegments, section];
    const anchors = resolveRepairAnchors(session, section.points[0] ?? null);

    return {
      ...session,
      repairSegments: nextSegments,
      startAnchor: anchors?.start ?? null,
      endAnchor: anchors?.end ?? null,
    };
  });
}

export function deleteLastRepairSegment(repairSessions: DraftRepairSession[]) {
  const activeSession = getActiveRepairSession(repairSessions);
  if (!activeSession || activeSession.repairSegments.length === 0) {
    return repairSessions;
  }

  return repairSessions.map((session) => {
    if (session.id !== activeSession.id) {
      return session;
    }

    const nextSegments = session.repairSegments.slice(0, -1);
    const anchors =
      nextSegments.length > 0
        ? resolveRepairAnchors(session, nextSegments[0]?.points[0] ?? null)
        : null;

    return {
      ...session,
      repairSegments: nextSegments,
      startAnchor: anchors?.start ?? null,
      endAnchor: anchors?.end ?? null,
    };
  });
}

export function closeActiveRepairSession(
  repairSessions: DraftRepairSession[],
  idFactory: (prefix: string) => string,
  ringOverride?: Point[]
) {
  const activeSession = getActiveRepairSession(repairSessions);
  if (!activeSession) {
    return null;
  }

  const ring = ringOverride ?? buildRepairCloseRing(activeSession);
  if (!ring) {
    return null;
  }

  const polygon = createClosedDraftPolygonFromRing(
    ring,
    idFactory,
    activeSession.sourcePolygonId
  );
  if (!polygon) {
    return null;
  }

  const remainingSessions = activateNextRepairSession(
    repairSessions.filter((session) => session.id !== activeSession.id)
  );
  return { polygon, repairSessions: remainingSessions };
}
