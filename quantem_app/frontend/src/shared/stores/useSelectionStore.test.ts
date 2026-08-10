import { beforeEach, describe, expect, it } from "vitest";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";

describe("useSelectionStore", () => {
  beforeEach(() => {
    useSelectionStore.getState().clearSelection();
  });

  it("keeps segmentation selection when asset id is unchanged", () => {
    const store = useSelectionStore.getState();
    store.setSelectedAssetId("asset-1");
    store.setSelectedSegmentationId("seg-1");
    store.setSelectedAssetId("asset-1");

    const next = useSelectionStore.getState();
    expect(next.selectedAssetId).toBe("asset-1");
    expect(next.selectedImageId).toBe("asset-1");
    expect(next.selectedSegmentationId).toBe("seg-1");
  });

  it("clears segmentation selection when asset changes", () => {
    const store = useSelectionStore.getState();
    store.setSelectedAssetId("asset-1");
    store.setSelectedSegmentationId("seg-1");
    store.setSelectedAssetId("asset-2");

    const next = useSelectionStore.getState();
    expect(next.selectedAssetId).toBe("asset-2");
    expect(next.selectedImageId).toBe("asset-2");
    expect(next.selectedSegmentationId).toBeNull();
  });

  it("clears all selections", () => {
    const store = useSelectionStore.getState();
    store.setSelectedAssetId("asset-1");
    store.setSelectedSegmentationId("seg-1");
    store.clearSelection();

    const next = useSelectionStore.getState();
    expect(next.selectedAssetId).toBeNull();
    expect(next.selectedImageId).toBeNull();
    expect(next.selectedSegmentationId).toBeNull();
  });
});
