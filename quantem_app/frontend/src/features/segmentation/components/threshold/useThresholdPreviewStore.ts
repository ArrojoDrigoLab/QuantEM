import { create } from "zustand";
import type { ProbabilityOverlay } from "@/features/segmentation/erPreview/overlayCanvas";

interface ThresholdPreviewState {
  overlay: ProbabilityOverlay | null;
  threshold: number;
  opacity: number;
  setOverlay: (overlay: ProbabilityOverlay | null) => void;
  setThreshold: (threshold: number) => void;
  clear: () => void;
}

export const useThresholdPreviewStore = create<ThresholdPreviewState>((set) => ({
  overlay: null,
  threshold: 0.5,
  opacity: 0.72,
  setOverlay: (overlay) => set({ overlay }),
  setThreshold: (threshold) => set({ threshold }),
  clear: () => set({ overlay: null }),
}));
