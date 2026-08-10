import type { BBox } from "@/shared/types/common";
import type { Point } from "@/utils/geometry";

/**
 * A polygon / multi-polygon geometry as it is handed to the drawing tools —
 * plain image-space rings, no CRS and no server bookkeeping.
 */
export interface DisplayGeometry {
  geometry_type: "Polygon" | "MultiPolygon";
  polygons: Array<Array<[number, number]>>;
}

export interface SpliceAnchor {
  edge: "top" | "right" | "bottom" | "left";
  point: [number, number];
  perimeter_offset: number;
}

/**
 * One invertible edit: `source_path` (a contiguous run of the current ring) is
 * swapped for `replacement_path`. Inverting a contract swaps the two paths.
 */
export interface SpliceSection {
  source_path: Array<[number, number]>;
  replacement_path: Array<[number, number]>;
  start_anchor?: SpliceAnchor | null;
  end_anchor?: SpliceAnchor | null;
}

export interface SpliceContract {
  bbox: BBox;
  sections: SpliceSection[];
}

export interface DraftSegment {
  id: string;
  kind: "section" | "closing";
  points: Point[];
  endedOutsidePatch: boolean;
}

export interface DraftPolygon {
  id: string;
  segments: DraftSegment[];
  closed: boolean;
  skipCycleResolution?: boolean;
}

export type DraftEditorMode =
  | "navigate"
  | "attract"
  | "avoid"
  | "draw"
  | "tile_refine"
  | "cut";

export interface DraftCutOverlay {
  active: boolean;
  points: Point[];
}

export interface DraftRepairSession {
  id: string;
  sourcePolygonId: string;
  remainingPath: Point[];
  repairSegments: DraftSegment[];
  startAnchor: Point | null;
  endAnchor: Point | null;
  active: boolean;
}
