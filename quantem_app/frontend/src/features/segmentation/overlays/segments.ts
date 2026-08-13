import type { SegmentOverlay } from "@/viewer/types";
import type { SegmentObject } from "@/shared/types/segmentation";
import type { Point } from "@/utils/geometry";
import {
  isCellCandidateStatus,
  isCellConfirmedStatus,
  normalizeCellStatus,
} from "@/utils/cellStatus";
import {
  selectSegmentGeometryCoords,
  selectSegmentHoleCoords,
} from "@/utils/segmentGeometry";

const INSTANCE_COLOR_INTERNAL_NAMES = new Set([
  "quantem_internal_mito",
  "quantem_internal_mito_deepcontact_cell",
  "quantem_internal_mito_deepcontact_sem",
  "quantem_internal_mito_deepcontact_tem",
  "quantem_internal_nucleus",
  "quantem_internal_ld",
]);
const INSTANCE_COLOR_PALETTE = [
  "#2fb5a9",
  "#f56f52",
  "#3f83f8",
  "#f59e0b",
  "#10b981",
  "#ef4444",
  "#a855f7",
  "#f97316",
  "#0ea5e9",
  "#22c55e",
];
const RESERVED_EXCLUDED_COLOR = "#5c677d";
const CONFIRMED_FILL_COLOR = "#33cc66";
const CONFIRMED_FILL_OPACITY = 0.15;
const DEFAULT_CANDIDATE_STROKE_WIDTH = 2;
const DEFAULT_CONFIRMED_STROKE_WIDTH = 2;

export interface LeftPanelLayerStyles {
  candidateStrokeWidth: number;
  candidateFillOpacity: number;
  confirmedStrokeWidth: number;
  confirmedFillOpacity: number;
}

function clampOpacity(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function clampStrokeWidth(value: number): number {
  if (!Number.isFinite(value)) return 1;
  return Math.max(0.5, value);
}

function hashSegmentId(segmentId: string): number {
  let hash = 0;
  for (let index = 0; index < segmentId.length; index += 1) {
    hash = (hash * 31 + segmentId.charCodeAt(index)) >>> 0;
  }
  return hash;
}

function getDeterministicInstanceColor(segmentId: string): string {
  return INSTANCE_COLOR_PALETTE[hashSegmentId(segmentId) % INSTANCE_COLOR_PALETTE.length];
}

function segmentCellStatus(segment: SegmentObject) {
  return normalizeCellStatus(segment.status);
}

function isConfirmedSegment(segment: SegmentObject): boolean {
  const status = segmentCellStatus(segment);
  return status !== null
    ? isCellConfirmedStatus(status)
    : segment.label_state === "CONFIRMED";
}

function isCandidateSegment(segment: SegmentObject): boolean {
  const status = segmentCellStatus(segment);
  return status !== null
    ? isCellCandidateStatus(status)
    : segment.label_state === "CANDIDATE" || segment.label_state === "INFERRED";
}

function resolveSegmentColor(
  segment: SegmentObject,
  segmentationTypeInternalName?: string | null
): string {
  if (segment.label_state === "EXCLUDED") {
    return RESERVED_EXCLUDED_COLOR;
  }

  if (
    segmentationTypeInternalName &&
    INSTANCE_COLOR_INTERNAL_NAMES.has(segmentationTypeInternalName)
  ) {
    return getDeterministicInstanceColor(segment.id);
  }

  return isConfirmedSegment(segment) ? "#00ff00" : "#ff0000";
}

function segmentGeometry(segment: SegmentObject, useSmoothedGeometry: boolean): Point[] {
  return selectSegmentGeometryCoords(segment, useSmoothedGeometry).map(([x, y]) => ({
    x,
    y,
  }));
}

function segmentHoles(segment: SegmentObject): Point[][] {
  return selectSegmentHoleCoords(segment).map((ring) =>
    ring.map(([x, y]) => ({ x, y }))
  );
}

export function generateLeftPanelOverlays(
  segments: SegmentObject[],
  tooMany: boolean,
  segmentationTypeInternalName?: string | null,
  bboxHighlightedIds?: ReadonlySet<string>,
  useSmoothedGeometry: boolean = false,
  layerStyles?: LeftPanelLayerStyles
): SegmentOverlay[] {
  if (tooMany || segments.length === 0) {
    return [];
  }

  const candidateStrokeWidth = clampStrokeWidth(
    layerStyles?.candidateStrokeWidth ?? DEFAULT_CANDIDATE_STROKE_WIDTH
  );
  const confirmedStrokeWidth = clampStrokeWidth(
    layerStyles?.confirmedStrokeWidth ?? DEFAULT_CONFIRMED_STROKE_WIDTH
  );
  const candidateFillOpacity = clampOpacity(layerStyles?.candidateFillOpacity ?? 0);
  const confirmedFillOpacity = clampOpacity(
    layerStyles?.confirmedFillOpacity ?? CONFIRMED_FILL_OPACITY
  );

  return segments.map((segment) => {
    const isConfirmed = isConfirmedSegment(segment);
    const isExcluded = segment.label_state === "EXCLUDED";
    const isCandidate = isCandidateSegment(segment);
    const baseColor = resolveSegmentColor(segment, segmentationTypeInternalName);
    const fillColor = isConfirmed ? CONFIRMED_FILL_COLOR : baseColor;
    const strokeColor = baseColor;
    const fillOpacity = isConfirmed
      ? confirmedFillOpacity
      : isExcluded
        ? 0.05
        : isCandidate
          ? candidateFillOpacity
          : 0;
    const bboxHighlighted = bboxHighlightedIds?.has(segment.id) ?? false;

    return {
      id: segment.id,
      geometry: segmentGeometry(segment, useSmoothedGeometry),
      holes: segmentHoles(segment),
      fillColor,
      fillOpacity,
      strokeColor: bboxHighlighted ? "#00ffff" : strokeColor,
      strokeOpacity: isConfirmed || isExcluded ? 0.85 : 0.3,
      strokeWidth: bboxHighlighted
        ? 4
        : isConfirmed
          ? confirmedStrokeWidth
          : candidateStrokeWidth,
    };
  });
}

export function generateRightPanelOverlays(
  confirmedSegments: SegmentObject[],
  inferredSegments: SegmentObject[],
  excludedSegments: SegmentObject[],
  rightSelectedSegmentId: string | null,
  segmentationTypeInternalName?: string | null,
  bboxHighlightedIds?: ReadonlySet<string>,
  useSmoothedGeometry: boolean = false
): SegmentOverlay[] {
  return [...confirmedSegments, ...excludedSegments, ...inferredSegments].map((segment) => {
    const isSelected = segment.id === rightSelectedSegmentId;
    const isBboxHighlighted = bboxHighlightedIds?.has(segment.id) ?? false;
    const isConfirmed = isConfirmedSegment(segment);
    const isExcluded = segment.label_state === "EXCLUDED";

    let baseColor = resolveSegmentColor(segment, segmentationTypeInternalName);
    if (!isConfirmed && !isExcluded && baseColor === RESERVED_EXCLUDED_COLOR) {
      baseColor = "#ff0000";
    }

    const fillColor = isConfirmed ? CONFIRMED_FILL_COLOR : baseColor;
    const strokeColor = baseColor;
    const baseStrokeWidth = isSelected ? 4 : 2;

    return {
      id: segment.id,
      geometry: segmentGeometry(segment, useSmoothedGeometry),
      holes: segmentHoles(segment),
      fillColor,
      fillOpacity: isConfirmed ? CONFIRMED_FILL_OPACITY : isExcluded ? 0.05 : 0,
      strokeColor: isSelected ? "#00ffff" : strokeColor,
      strokeOpacity: 0.9,
      strokeWidth: isBboxHighlighted ? baseStrokeWidth * 2 : baseStrokeWidth,
    };
  });
}
