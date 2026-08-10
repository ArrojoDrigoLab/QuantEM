import type { SegmentObject } from "@/shared/types";

export type PointActionMode = "confirm" | "reject";

function isPointActionableSegment(
  segment: SegmentObject,
  mode: PointActionMode
): boolean {
  if (mode === "reject") {
    return (
      segment.label_state === "CANDIDATE" ||
      segment.label_state === "INFERRED" ||
      segment.label_state === "CONFIRMED"
    );
  }
  return (
    segment.label_state === "CANDIDATE" || segment.label_state === "INFERRED"
  );
}

function actionableLabelPriority(
  segment: SegmentObject,
  mode: PointActionMode
): number {
  switch (segment.label_state) {
    case "CANDIDATE":
      return 0;
    case "INFERRED":
      return 1;
    case "CONFIRMED":
      return mode === "reject" ? 2 : 3;
    default:
      return 4;
  }
}

export function selectBestPointActionSegment(
  segments: SegmentObject[],
  mode: PointActionMode = "confirm"
): SegmentObject | null {
  const actionableSegments = segments.filter((segment) =>
    isPointActionableSegment(segment, mode)
  );
  if (actionableSegments.length === 0) {
    return null;
  }

  return [...actionableSegments].sort((a, b) => {
    const priorityDiff =
      actionableLabelPriority(a, mode) - actionableLabelPriority(b, mode);
    if (priorityDiff !== 0) {
      return priorityDiff;
    }
    if (a.confidence_score !== null && b.confidence_score !== null) {
      return b.confidence_score - a.confidence_score;
    }
    if (a.confidence_score !== null) return -1;
    if (b.confidence_score !== null) return 1;
    return a.id.localeCompare(b.id);
  })[0] ?? null;
}
