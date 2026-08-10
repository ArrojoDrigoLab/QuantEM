import type { SegmentOverlay } from "@/viewer/types";

export interface OverlayScene {
  persistent: SegmentOverlay[];
  transient: SegmentOverlay[];
}

export const EMPTY_OVERLAY_SCENE: OverlayScene = {
  persistent: [],
  transient: [],
};
