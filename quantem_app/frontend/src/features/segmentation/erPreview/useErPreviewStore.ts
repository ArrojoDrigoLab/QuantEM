import { create } from "zustand";
import { pinErCandidates, runErModelPreview } from "@/shared/api/erPreview";
import { decodeProbImage, type ErProbOverlay } from "@/features/segmentation/erPreview/overlayCanvas";

interface ErPreviewRoi {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

interface ErPreviewState {
  overlay: ErProbOverlay | null;
  roiId: string | null;
  threshold: number;
  opacity: number;
  running: boolean;
  pinning: boolean;
  error: string | null;
  pinError: string | null;
  stats: { elapsed_s: number; frac: number } | null;
  run: (args: { assetId: string; sourceModel: string; roi: ErPreviewRoi }) => Promise<void>;
  pin: (args: { segmentationId: string }) => Promise<number | null>;
  setThreshold: (threshold: number) => void;
  setOpacity: (opacity: number) => void;
  clear: () => void;
}

export const useErPreviewStore = create<ErPreviewState>((set, get) => ({
  overlay: null,
  roiId: null,
  threshold: 0.3,
  opacity: 0.7,
  running: false,
  pinning: false,
  error: null,
  pinError: null,
  stats: null,
  async run({ assetId, sourceModel, roi }) {
    set({ running: true, error: null, pinError: null });
    try {
      const res = await runErModelPreview(assetId, {
        source_model: sourceModel,
        x: roi.x,
        y: roi.y,
        width: roi.width,
        height: roi.height,
        roi_id: roi.id,
      });
      const decoded = await decodeProbImage(res.prob_image);
      set({
        overlay: {
          probData: decoded.data,
          width: decoded.width,
          height: decoded.height,
          bounds: [res.bbox.x, res.bbox.y, res.bbox.width, res.bbox.height],
          color: res.color,
          sourceModel,
        },
        roiId: roi.id,
        threshold: typeof res.default_threshold === "number" ? res.default_threshold : 0.3,
        running: false,
        stats: res.stats ? { elapsed_s: res.stats.elapsed_s, frac: res.stats.frac } : null,
      });
    } catch (err) {
      set({ running: false, error: err instanceof Error ? err.message : "ER model run failed" });
    }
  },
  async pin({ segmentationId }) {
    const { overlay, roiId, threshold } = get();
    if (!segmentationId || !roiId || !overlay) {
      return null;
    }
    set({ pinning: true, pinError: null });
    try {
      const res = await pinErCandidates(segmentationId, {
        roi_id: roiId,
        source_model: overlay.sourceModel,
        threshold,
      });
      // Drop the transient preview; the pinned CANDIDATE segments now render
      // via the normal overlay/refresh machinery.
      set({ pinning: false, overlay: null, stats: null });
      return res.count;
    } catch (err) {
      set({ pinning: false, pinError: err instanceof Error ? err.message : "Pin failed" });
      return null;
    }
  },
  setThreshold(threshold) {
    set({ threshold });
  },
  setOpacity(opacity) {
    set({ opacity });
  },
  clear() {
    set({ overlay: null, roiId: null, error: null, pinError: null, stats: null });
  },
}));
