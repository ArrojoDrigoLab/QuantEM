import type { LabelState } from "@/shared/types/common";
import type { SegmentObject } from "@/shared/types/segmentation";

export function toSyntheticSegmentObject(
  segmentationId: string,
  segment: {
    id: string;
    label_state: LabelState;
    confidence_score: number | null;
    geometry_coords?: Array<[number, number]>;
    source_model?: string;
  }
): SegmentObject {
  return {
    id: segment.id,
    segmentation: segmentationId,
    label_state: segment.label_state,
    confidence_score: segment.confidence_score,
    source_model: segment.source_model,
    geometry_coords: segment.geometry_coords ?? [],
    created_at: "",
    updated_at: "",
  };
}

export function makeOptimisticOverlayId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function buildSyntheticSegmentsFromGeometries(
  segmentationId: string,
  geometries: Array<Array<[number, number]>>,
  labelState: LabelState,
  ids: readonly string[] = []
): SegmentObject[] {
  return geometries.map((geometryCoords, index) =>
    toSyntheticSegmentObject(segmentationId, {
      id:
        ids[index] ??
        makeOptimisticOverlayId(`optimistic-${labelState.toLowerCase()}`),
      label_state: labelState,
      confidence_score: null,
      geometry_coords: geometryCoords,
    })
  );
}

export function buildSyntheticSegmentsFromGeometryRings(
  segmentationId: string,
  geometries: Array<Array<Array<[number, number]>>>,
  labelState: LabelState,
  ids: readonly string[] = []
): SegmentObject[] {
  return geometries.map((rings, index) => ({
    ...toSyntheticSegmentObject(segmentationId, {
      id:
        ids[index] ??
        makeOptimisticOverlayId(`optimistic-${labelState.toLowerCase()}`),
      label_state: labelState,
      confidence_score: null,
      geometry_coords: rings[0] ?? [],
    }),
    geometry: { type: "Polygon" as const, coordinates: rings },
  }));
}
