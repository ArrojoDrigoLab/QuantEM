import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegmentation,
  overlayManifestState,
  setupSegmentationScreenTest,
} from "./SegmentationScreen.testUtils";
import { SegmentationScreen } from "./SegmentationScreen";
import { getAssetSegmentations } from "@/shared/api/assets";
import { activateSegmentationRoi } from "@/shared/api/segmentations/rois";
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

  it("opens one viewer with confirmed objects and browsable ROIs, without editing controls", async () => {
    const user = userEvent.setup();
    renderScreen();
    await screen.findByRole("button", { name: "Unlock segmentation" });
    await screen.findAllByRole("button", { name: "Open" });
    expect(screen.getAllByTestId("viewer")).toHaveLength(1);
    expect(viewerProps().overlays?.idMapOverlays?.map((layer) => layer.id)).toEqual([
      "label-left-confirmed-idmap",
    ]);
    for (const name of ["Add New", "Edit Area", "Test", "Run model", "Remove area", "Remove objects", "Brush", "Polygon"]) {
      expect(screen.queryByRole("button", { name })).not.toBeInTheDocument();
    }
    expect(viewerProps().interactions?.brush?.enabled).toBe(false);
    await user.click(screen.getAllByRole("button", { name: "Open" })[1]);
    expect(viewerProps().viewport?.fitBounds).toEqual({ x: 300, y: 200, width: 100, height: 80 });
    expect(activateSegmentationRoi).not.toHaveBeenCalled();
    await user.click(screen.getByRole("checkbox", { name: "View confirmed" }));
    expect(viewerProps().overlays?.idMapOverlays).toEqual([]);
    expect(screen.getAllByTestId("viewer")).toHaveLength(1);
    await user.click(screen.getByRole("checkbox", { name: "View confirmed" }));
    expect(viewerProps().overlays?.idMapOverlays).toHaveLength(1);
  });

  it("restores the editing layout after unlocking without reloading", async () => {
    renderScreen();
    const unlock = await screen.findByRole("button", { name: "Unlock segmentation" });
    vi.mocked(getAssetSegmentations).mockResolvedValue([makeSegmentation()]);
    await userEvent.setup().click(unlock);
    await waitFor(() => expect(screen.getAllByTestId("viewer")).toHaveLength(2));
    expect(unlockSegmentation).toHaveBeenCalledWith("seg-1");
    expect(screen.getByRole("button", { name: "Remove objects" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add New" })).toBeInTheDocument();
  });

  it("loads a delayed confirmed overlay without unlocking or refreshing the page", async () => {
    const saved = overlayManifestState.manifest;
    overlayManifestState.manifest = null;
    const { rerender } = renderScreen();
    await screen.findByRole("button", { name: "Unlock segmentation" });
    expect(viewerProps().overlays?.idMapOverlays).toEqual([]);
    overlayManifestState.manifest = saved;
    rerender(<TestScreen />);
    await waitFor(() => expect(viewerProps().overlays?.idMapOverlays).toHaveLength(1));
    expect(screen.getAllByTestId("viewer")).toHaveLength(1);
    expect(unlockSegmentation).not.toHaveBeenCalled();
  });
});
