import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegmentation,
  overlayManifestState,
  setupSegmentationScreenTest,
} from "./SegmentationScreen.testUtils";
import { SegmentationScreen } from "./SegmentationScreen";
import { getAssetSegmentations } from "@/shared/api/assets";
import { unlockSegmentation } from "@/shared/api/segmentations/annotations";
import type { ImageViewerProps } from "@/viewer/imageViewerTypes";

const { viewerSpy } = vi.hoisted(() => ({ viewerSpy: vi.fn() }));
vi.mock("@/viewer/components/ImageViewer", () => ({
  ImageViewer: (props: ImageViewerProps) => {
    viewerSpy(props);
    return <div data-testid="viewer" />;
  },
}));
vi.mock("@/features/update/restartGuardHooks", () => ({ useRestartBlocker: vi.fn() }));
vi.mock("@/features/segmentation/screen/hooks/overlay/useOverlayLut", () => ({
  useOverlayLut: () => ({ rgba: new Uint8Array([0, 0, 0, 0, 0, 255, 0, 255]), maxLabel: 1, lutRevision: 2 }),
  useOverlayPickMap: () => null,
}));

function TestScreen() {
  return (
    <MemoryRouter initialEntries={["/assets/img-1/labeling/Mitochondria"]}>
      <Routes>
        <Route path="/assets/:assetId/labeling/:segmentationTypeName" element={<SegmentationScreen />} />
      </Routes>
    </MemoryRouter>
  );
}

function renderScreen() {
  return render(<TestScreen />);
}

function viewerProps(): ImageViewerProps {
  return viewerSpy.mock.lastCall![0];
}

describe("locked segmentation screen", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    vi.mocked(getAssetSegmentations).mockResolvedValue([
      makeSegmentation({ status_stage: "COMPLETED", is_complete: true }),
    ]);
    overlayManifestState.manifest = {
      status: "READY", ngff_url: "/ngff/confirmed.zarr", lut_url: "/lut/",
      arrays: ["labels", "border"], label_dtype: "uint32", source_model: null,
      bundle_version: 1, applied_revision: 3, desired_revision: 3, lut_revision: 2,
      chunk_size: [256, 256], level_count: 4, width: 1000, height: 800,
    };
  });

  it("opens completed images with editing and model controls available", async () => {
    renderScreen();
    expect(await screen.findByRole("button", {name: "Whole image segmented"})).toBeDisabled();
    expect(screen.queryByRole("button", {name: "Unlock segmentation"})).not.toBeInTheDocument();
    expect(screen.getByRole("button", {name: "Add New"})).toBeEnabled();
    expect(screen.getByRole("button", {name: "Remove objects"})).toBeEnabled();
    expect(unlockSegmentation).not.toHaveBeenCalled();
  });
  it("loads delayed overlays while preserving editing access", async () => {
    const saved = overlayManifestState.manifest;
    overlayManifestState.manifest = null;
    const {rerender} = renderScreen();
    await screen.findByRole("button", {name: "Whole image segmented"});
    overlayManifestState.manifest = saved;
    rerender(<TestScreen />);
    await waitFor(() => expect(viewerProps().overlays?.idMapOverlays?.length).toBeGreaterThan(0));
    expect(screen.getByRole("button", {name: "Add New"})).toBeEnabled();
    expect(unlockSegmentation).not.toHaveBeenCalled();
  });
});
