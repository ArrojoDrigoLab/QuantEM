/**
 * Global state store for selected asset and segmentation.
 * Uses Zustand for lightweight state management.
 */

import { create } from "zustand";

interface SelectionState {
  selectedAssetId: string | null;
  selectedImageId: string | null;
  selectedSegmentationId: string | null;
  setSelectedAssetId: (id: string | null) => void;
  setSelectedImageId: (id: string | null) => void;
  setSelectedSegmentationId: (id: string | null) => void;
  clearSelection: () => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedAssetId: null,
  selectedImageId: null,
  selectedSegmentationId: null,
  setSelectedAssetId: (id) =>
    set((state) => ({
      selectedAssetId: id,
      selectedImageId: id,
      selectedSegmentationId:
        state.selectedAssetId === id ? state.selectedSegmentationId : null,
    })),
  setSelectedImageId: (id) =>
    set((state) => ({
      selectedAssetId: id,
      selectedImageId: id,
      selectedSegmentationId:
        state.selectedAssetId === id ? state.selectedSegmentationId : null,
    })),
  setSelectedSegmentationId: (id) => set({ selectedSegmentationId: id }),
  clearSelection: () =>
    set({ selectedAssetId: null, selectedImageId: null, selectedSegmentationId: null }),
}));
