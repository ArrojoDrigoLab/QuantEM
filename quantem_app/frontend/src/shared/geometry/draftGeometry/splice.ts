import type {
  SpliceContract,
  SpliceSection,
} from "@/shared/geometry/draftGeometry/types";
import type { Point } from "@/utils/geometry";
import {
  dedupeConsecutive,
  dotProduct,
  lerpPoint,
  nearlyEqual,
  normalizeClosedRing,
  normalizeTrustedClosedRing,
  openRing,
  pointOnSegment,
  pointsEqual,
  subtractPoints,
} from "@/shared/geometry/draftGeometry/shared";

interface PathLocation {
  point: Point;
  segmentIndex: number;
  t: number;
}

interface MatchedSpliceSection {
  sourcePath: Point[];
  replacementPath: Point[];
  match: {
    start: PathLocation;
    end: PathLocation;
  };
  startOffset: number;
  endOffset: number;
}

function normalizePath(points: Point[]) {
  const normalized = dedupeConsecutive(points);
  if (normalized.length < 2) {
    return null;
  }
  return normalized;
}

function isClosedPath(points: Point[]) {
  return points.length >= 2 && pointsEqual(points[0], points[points.length - 1]);
}

function pointParameterOnSegment(point: Point, start: Point, end: Point) {
  const direction = subtractPoints(end, start);
  const lengthSq = dotProduct(direction, direction);
  if (nearlyEqual(lengthSq, 0)) {
    return 0;
  }
  return Math.min(
    1,
    Math.max(0, dotProduct(subtractPoints(point, start), direction) / lengthSq)
  );
}

function findPointLocationsOnRing(point: Point, ringOpen: Point[]) {
  const locations: PathLocation[] = [];
  for (let segmentIndex = 0; segmentIndex < ringOpen.length; segmentIndex += 1) {
    const start = ringOpen[segmentIndex];
    const end = ringOpen[(segmentIndex + 1) % ringOpen.length];
    if (!pointOnSegment(point, start, end)) {
      continue;
    }
    locations.push({
      point,
      segmentIndex,
      t: pointParameterOnSegment(point, start, end),
    });
  }
  return locations;
}

function buildForwardPath(ringOpen: Point[], start: PathLocation, end: PathLocation) {
  const path: Point[] = [start.point];
  let segmentIndex = start.segmentIndex;
  let completedLoop = false;

  for (let step = 0; step <= ringOpen.length; step += 1) {
    const segmentStart = ringOpen[segmentIndex];
    const segmentEnd = ringOpen[(segmentIndex + 1) % ringOpen.length];
    const startT =
      segmentIndex === start.segmentIndex && !completedLoop ? start.t : 0;
    const canFinishHere =
      segmentIndex === end.segmentIndex &&
      (completedLoop || end.segmentIndex !== start.segmentIndex || end.t >= startT - 1e-6);
    const endT = canFinishHere ? end.t : 1;

    if (endT > startT + 1e-6) {
      const clippedEnd = lerpPoint(segmentStart, segmentEnd, endT);
      if (!pointsEqual(path[path.length - 1], clippedEnd)) {
        path.push(clippedEnd);
      }
    }

    if (canFinishHere) {
      return dedupeConsecutive(path);
    }

    if (!pointsEqual(path[path.length - 1], segmentEnd)) {
      path.push(segmentEnd);
    }
    segmentIndex = (segmentIndex + 1) % ringOpen.length;
    if (segmentIndex === start.segmentIndex) {
      completedLoop = true;
    }
  }
  return null;
}

function pathsEqual(left: Point[], right: Point[]) {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((point, index) => pointsEqual(point, right[index]));
}

function pathLocationOffset(location: PathLocation, ringLength: number) {
  return Math.min(
    ringLength,
    Math.max(0, Number(location.segmentIndex) + Number(location.t))
  );
}

function appendPath(target: Point[], path: Point[]) {
  for (const point of path) {
    if (!target.length || !pointsEqual(target[target.length - 1], point)) {
      target.push(point);
    }
  }
}

function matchSourcePathExactly(ringOpen: Point[], sourcePath: Point[]) {
  const ringLength = ringOpen.length;
  for (let startIndex = 0; startIndex < ringLength; startIndex += 1) {
    if (!pointsEqual(ringOpen[startIndex], sourcePath[0])) {
      continue;
    }
    let matches = true;
    for (let offset = 1; offset < sourcePath.length; offset += 1) {
      const ringPoint = ringOpen[(startIndex + offset) % ringLength];
      if (!pointsEqual(ringPoint, sourcePath[offset])) {
        matches = false;
        break;
      }
    }
    if (!matches) {
      continue;
    }
    const endIndex = (startIndex + sourcePath.length - 1) % ringLength;
    return {
      start: { point: sourcePath[0], segmentIndex: startIndex, t: 0 },
      end: {
        point: sourcePath[sourcePath.length - 1],
        segmentIndex: endIndex,
        t: 0,
      },
    };
  }
  return null;
}

function matchSourcePathOnRing(ring: Point[], sourcePath: Point[]) {
  const ringOpen = openRing(ring);
  const exactMatch = matchSourcePathExactly(ringOpen, sourcePath);
  if (exactMatch) {
    const exactForward = buildForwardPath(ringOpen, exactMatch.start, exactMatch.end);
    if (exactForward && pathsEqual(exactForward, sourcePath)) {
      return exactMatch;
    }
  }

  const startLocations = findPointLocationsOnRing(sourcePath[0], ringOpen);
  const endLocations = findPointLocationsOnRing(sourcePath[sourcePath.length - 1], ringOpen);
  for (const start of startLocations) {
    for (const end of endLocations) {
      const forwardPath = buildForwardPath(ringOpen, start, end);
      if (forwardPath && pathsEqual(forwardPath, sourcePath)) {
        return { start, end };
      }
    }
  }
  return null;
}

function replaceClosedRingPath(
  currentRing: Point[],
  section: SpliceSection
) {
  const sourcePath = normalizePath(
    section.source_path.map(([x, y]) => ({ x, y }))
  );
  const replacementPath = normalizePath(
    section.replacement_path.map(([x, y]) => ({ x, y }))
  );
  if (!sourcePath || !replacementPath) {
    return {
      kind: "error" as const,
      message: "Splice section was not a valid path.",
    };
  }

  if (isClosedPath(sourcePath)) {
    const replacementRing = normalizeTrustedClosedRing(replacementPath);
    if (!replacementRing) {
      return {
        kind: "error" as const,
        message: "Full-ring replacement did not resolve to one closed polygon.",
      };
    }
    return { kind: "ok" as const, ring: replacementRing };
  }

  const match = matchSourcePathOnRing(currentRing, sourcePath);
  if (!match) {
    return {
      kind: "error" as const,
      message: "Source section no longer matched the current polygon.",
    };
  }

  const ringOpen = openRing(currentRing);
  const removedPath = buildForwardPath(ringOpen, match.start, match.end);
  if (!removedPath || !pathsEqual(removedPath, sourcePath)) {
    return {
      kind: "error" as const,
      message: "Source section no longer matched the current polygon.",
    };
  }

  const outsideForward = buildForwardPath(ringOpen, match.end, match.start);
  if (!outsideForward) {
    return {
      kind: "error" as const,
      message: "Splice could not reconstruct the polygon outside the selected bbox.",
    };
  }
  const remainingPath = dedupeConsecutive([...outsideForward].reverse());
  const normalizedReplacement = dedupeConsecutive(replacementPath);
  if (
    remainingPath.length < 2 ||
    normalizedReplacement.length < 2 ||
    !pointsEqual(remainingPath[0], normalizedReplacement[0]) ||
    !pointsEqual(
      remainingPath[remainingPath.length - 1],
      normalizedReplacement[normalizedReplacement.length - 1]
    )
  ) {
    return {
      kind: "error" as const,
      message: "Replacement section did not align to the current polygon endpoints.",
    };
  }

  const reconstructedRing = normalizeClosedRing([
    ...normalizedReplacement,
    ...[...remainingPath].reverse().slice(1),
  ]);
  const nextRing = reconstructedRing
    ? normalizeTrustedClosedRing(reconstructedRing)
    : null;
  if (!nextRing) {
    return {
      kind: "error" as const,
      message: "Splice did not resolve to one locally spliced polygon.",
    };
  }
  return { kind: "ok" as const, ring: nextRing };
}

export type SpliceResult =
  | { kind: "ok"; ring: Point[] }
  | { kind: "error"; message: string };

export function invertSpliceContract(
  splice: SpliceContract
): SpliceContract {
  return {
    bbox: splice.bbox,
    sections: [...splice.sections].reverse().map((section) => ({
      source_path: section.replacement_path,
      replacement_path: section.source_path,
      start_anchor: section.end_anchor ?? null,
      end_anchor: section.start_anchor ?? null,
    })),
  };
}

export function applySpliceContract(
  currentRingInput: Point[],
  splice: SpliceContract
): SpliceResult {
  let ring = normalizeTrustedClosedRing(currentRingInput);
  if (!ring) {
    return {
      kind: "error",
      message: "Splice did not resolve to one valid polygon ring.",
    };
  }

  if (!splice.sections.length) {
    return {
      kind: "error",
      message: "Splice did not return any sections.",
    };
  }

  for (const section of splice.sections) {
    const result = replaceClosedRingPath(ring, section);
    if (result.kind !== "ok") {
      return result;
    }
    ring = normalizeTrustedClosedRing(result.ring) ?? result.ring;
  }

  const finalRing = normalizeTrustedClosedRing(ring);
  if (!finalRing) {
    return {
      kind: "error",
      message: "Splice did not resolve to one locally spliced polygon.",
    };
  }
  return { kind: "ok", ring: finalRing };
}

function matchSpliceSectionOnBaseRing(
  baseRing: Point[],
  section: SpliceSection
):
  | { kind: "ok"; section: MatchedSpliceSection }
  | { kind: "error"; message: string } {
  const sourcePath = normalizePath(
    section.source_path.map(([x, y]) => ({ x, y }))
  );
  const replacementPath = normalizePath(
    section.replacement_path.map(([x, y]) => ({ x, y }))
  );
  if (!sourcePath || !replacementPath) {
    return {
      kind: "error",
      message: "Splice section was not a valid path.",
    };
  }
  if (isClosedPath(sourcePath)) {
    return {
      kind: "error",
      message: "Batch splice cannot apply full-ring replacement sections.",
    };
  }

  const match = matchSourcePathOnRing(baseRing, sourcePath);
  if (!match) {
    return {
      kind: "error",
      message: "Source section no longer matched the starting polygon.",
    };
  }

  const ringOpen = openRing(baseRing);
  const removedPath = buildForwardPath(ringOpen, match.start, match.end);
  if (!removedPath || !pathsEqual(removedPath, sourcePath)) {
    return {
      kind: "error",
      message: "Source section no longer matched the starting polygon.",
    };
  }
  if (
    !pointsEqual(replacementPath[0], sourcePath[0]) ||
    !pointsEqual(
      replacementPath[replacementPath.length - 1],
      sourcePath[sourcePath.length - 1]
    )
  ) {
    return {
      kind: "error",
      message: "Replacement section did not align to the source endpoints.",
    };
  }

  const ringLength = ringOpen.length;
  const startOffset = pathLocationOffset(match.start, ringLength);
  let endOffset = pathLocationOffset(match.end, ringLength);
  if (endOffset <= startOffset + 1e-6) {
    endOffset += ringLength;
  }

  return {
    kind: "ok",
    section: {
      sourcePath,
      replacementPath,
      match,
      startOffset,
      endOffset,
    },
  };
}

export function applySpliceContractsToBaseRing(
  baseRingInput: Point[],
  splices: SpliceContract[]
): SpliceResult {
  const baseRing = normalizeTrustedClosedRing(baseRingInput);
  if (!baseRing) {
    return {
      kind: "error",
      message: "Splice did not resolve to one valid polygon ring.",
    };
  }
  const ringOpen = openRing(baseRing);
  if (ringOpen.length < 3) {
    return {
      kind: "error",
      message: "Splice did not resolve to one valid polygon ring.",
    };
  }

  const matchedSections: MatchedSpliceSection[] = [];
  for (const splice of splices) {
    for (const section of splice.sections) {
      const matched = matchSpliceSectionOnBaseRing(baseRing, section);
      if (matched.kind !== "ok") {
        return matched;
      }
      matchedSections.push(matched.section);
    }
  }
  if (!matchedSections.length) {
    return {
      kind: "error",
      message: "Splice contract did not contain any sections.",
    };
  }

  matchedSections.sort((left, right) => left.startOffset - right.startOffset);
  for (let index = 1; index < matchedSections.length; index += 1) {
    const previous = matchedSections[index - 1];
    const current = matchedSections[index];
    if (current.startOffset < previous.endOffset - 1e-6) {
      return {
        kind: "error",
        message: "Batch sections overlapped on the starting polygon.",
      };
    }
  }
  const lastSection = matchedSections[matchedSections.length - 1];
  const firstSection = matchedSections[0];
  if (lastSection.endOffset - ringOpen.length > firstSection.startOffset + 1e-6) {
    return {
      kind: "error",
      message: "Batch sections overlapped across the ring closure.",
    };
  }

  const stitchedPath: Point[] = [];
  appendPath(stitchedPath, firstSection.replacementPath);
  let previousSection = firstSection;
  for (const currentSection of matchedSections.slice(1)) {
    const gap = buildForwardPath(
      ringOpen,
      previousSection.match.end,
      currentSection.match.start
    );
    if (!gap) {
      return {
        kind: "error",
        message: "Splice could not reconstruct a gap between sections.",
      };
    }
    appendPath(stitchedPath, gap);
    appendPath(stitchedPath, currentSection.replacementPath);
    previousSection = currentSection;
  }

  const closingGap = buildForwardPath(
    ringOpen,
    previousSection.match.end,
    firstSection.match.start
  );
  if (!closingGap) {
    return {
      kind: "error",
      message: "Splice could not reconstruct the closing gap between sections.",
    };
  }
  appendPath(stitchedPath, closingGap);

  const nextRing = normalizeTrustedClosedRing(stitchedPath);
  if (!nextRing) {
    return {
      kind: "error",
      message: "Splice did not resolve to one locally spliced polygon.",
    };
  }
  return { kind: "ok", ring: nextRing };
}
