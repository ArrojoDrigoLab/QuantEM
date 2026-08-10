import type { Point } from "@/utils/geometry";
import type {
  DraftPolygon,
  DraftRepairSession,
} from "@/shared/geometry/draftGeometry/types";

export type RingCutResult =
  | { kind: "untouched" }
  | { kind: "error"; message: string }
  | { kind: "ok"; remainingPath: Point[] };

export type CutDraftResult =
  | { kind: "untouched" }
  | { kind: "error"; message: string }
  | {
      kind: "ok";
      polygons: DraftPolygon[];
      repairSessions: DraftRepairSession[];
    };

