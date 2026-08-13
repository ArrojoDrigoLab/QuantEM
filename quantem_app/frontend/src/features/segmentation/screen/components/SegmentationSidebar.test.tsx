import type { ComponentProps } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SegmentationSidebar } from "@/features/segmentation/screen/components/SegmentationSidebar";
import { rebuildSegmentationOverlay } from "@/shared/api/segmentations/overlays";
import { ApiRequestError } from "@/shared/api/core/http";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

vi.mock("@/shared/api/segmentations/overlays", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/overlays")
  >("@/shared/api/segmentations/overlays");
  return { ...actual, rebuildSegmentationOverlay: vi.fn() };
});

const rebuildMock = vi.mocked(rebuildSegmentationOverlay);

type SidebarProps = ComponentProps<typeof SegmentationSidebar>;

function failedManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return {
    status: "FAILED",
    ngff_url: "/segmentation-overlays/seg-1.zarr",
    lut_url: "/api/segmentations/seg-1/overlay-lut/",
    arrays: ["labels", "border"],
    label_dtype: "uint32",
    source_model: "quantem:mito",
    bundle_version: 3,
    applied_revision: 5,
    desired_revision: 6,
    lut_revision: 6,
    chunk_size: [256, 256],
    level_count: 4,
    width: 1024,
    height: 1024,
    last_error:
      "Cannot create a file when that file already exists: " +
      "D:\\quantem-data\\tmp\\segmentation_overlays\\seg-1\\staging",
    ...overrides,
  };
}

function makeProps(layers: Partial<SidebarProps["layers"]> = {}): SidebarProps {
  return {
    tissue: {
      enabled: false,
      mode: "brush",
      brushSize: 24,
      hasMask: false,
      saving: false,
      onModeChange: vi.fn(),
      onBrushSizeChange: vi.fn(),
      onClear: vi.fn(),
      onSave: vi.fn(),
    } as unknown as SidebarProps["tissue"],
    review: {
      workflowMode: "review",
      reviewPhase: "model",
      correctionTool: "draw",
      draftOperation: "include",
      hoverActionMode: "confirm",
      drawBrushSize: 24,
      hasDrawStrokes: false,
      supportsPointFeedback: true,
      isErSegmentation: false,
      canApplyGroupAction: false,
      polygonHasDraft: false,
      polygonCanClose: false,
      onReviewPhaseChange: vi.fn(),
      onCorrectionToolChange: vi.fn(),
      onDraftOperationChange: vi.fn(),
      onHoverActionModeChange: vi.fn(),
      onDrawBrushSizeChange: vi.fn(),
      onClearDrawing: vi.fn(),
      onConfirmShape: vi.fn(),
      onClosePolygon: vi.fn(),
      onApplyGroupAction: vi.fn(),
    },
    layers: {
      usesRasterReviewOverlay: true,
      showCandidateBorders: true,
      onShowCandidateBordersChange: vi.fn(),
      showConfirmedBorders: true,
      onShowConfirmedBordersChange: vi.fn(),
      leftPanelLayerStyles: {
        candidateStrokeWidth: 2,
        candidateFillOpacity: 0.25,
        confirmedStrokeWidth: 2,
        confirmedFillOpacity: 0.25,
      } as SidebarProps["layers"]["leftPanelLayerStyles"],
      onCandidateStrokeWidthChange: vi.fn(),
      onCandidateFillOpacityChange: vi.fn(),
      onConfirmedStrokeWidthChange: vi.fn(),
      onConfirmedFillOpacityChange: vi.fn(),
      overlayUpdating: false,
      overlayBuildFailed: false,
      overlayManifest: null,
      overlaySegmentationId: "seg-1",
      onOverlayBuildRetried: vi.fn(),
      ...layers,
    },
    view: {
      leftNavigateMode: false,
      onLeftNavigateModeChange: vi.fn(),
      showConfirmedPanel: false,
      onShowConfirmedPanelChange: vi.fn(),
      isGroupActionMode: false,
      activeGroupActionVerb: "confirm",
      groupSelectionCount: 0,
    },
  };
}

/**
 * Finding V4. The viewer was taught to report a failed overlay build in wave
 * 0b; the labeling screen was not, and it is the screen someone spends an hour
 * on. The failure is silent there in the worst possible way: the review canvas
 * keeps drawing the last raster that built, so recent corrections are missing
 * from the picture and nothing says so.
 */
describe("SegmentationSidebar, when the overlay build has failed", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("says the build failed instead of saying nothing", () => {
    render(
      <SegmentationSidebar
        {...makeProps({ overlayBuildFailed: true, overlayManifest: failedManifest() })}
      />
    );

    expect(screen.getByText("Overlay could not be rebuilt")).toBeInTheDocument();
    expect(screen.getByText(/Your objects are safe/)).toBeInTheDocument();
  });

  it("shows the reason with a path a user can read", () => {
    render(
      <SegmentationSidebar
        {...makeProps({ overlayBuildFailed: true, overlayManifest: failedManifest() })}
      />
    );

    const reason = screen.getByText(/Cannot create a file/);
    expect(reason).toHaveTextContent(
      "D:\\quantem-data\\tmp\\segmentation_overlays\\seg-1\\staging"
    );
    // V5's defect, stated where it is read rather than where it is produced.
    expect(reason.textContent).not.toContain("\\\\");
  });

  it("warns that the picture on the canvas is out of date", () => {
    render(
      <SegmentationSidebar
        {...makeProps({ overlayBuildFailed: true, overlayManifest: failedManifest() })}
      />
    );

    expect(
      screen.getByText(/anything changed since then is missing from it/)
    ).toBeInTheDocument();
  });

  it("never claims the overlay is updating at the same time", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          overlayBuildFailed: true,
          overlayManifest: failedManifest(),
          overlayUpdating: false,
        })}
      />
    );

    expect(screen.queryByText("Overlay updating.")).not.toBeInTheDocument();
  });

  it("asks for the build again, for the bundle actually on screen", async () => {
    const user = userEvent.setup();
    const onOverlayBuildRetried = vi.fn();
    rebuildMock.mockResolvedValue(failedManifest({ status: "BUILDING" }));

    render(
      <SegmentationSidebar
        {...makeProps({
          overlayBuildFailed: true,
          overlayManifest: failedManifest(),
          onOverlayBuildRetried,
        })}
      />
    );

    await user.click(screen.getByRole("button", { name: /Retry overlay build/i }));

    expect(rebuildMock).toHaveBeenCalledWith("seg-1", "full", "quantem:mito");
    // Polling stopped when the build failed; only this puts it back.
    await waitFor(() => expect(onOverlayBuildRetried).toHaveBeenCalledTimes(1));
  });

  it("renders a refused retry rather than going quiet", async () => {
    const user = userEvent.setup();
    rebuildMock.mockRejectedValue(
      new ApiRequestError("This segmentation is marked done, so it is locked.", {
        status: 409,
      })
    );

    render(
      <SegmentationSidebar
        {...makeProps({ overlayBuildFailed: true, overlayManifest: failedManifest() })}
      />
    );

    await user.click(screen.getByRole("button", { name: /Retry overlay build/i }));

    expect(
      await screen.findByText(/This segmentation is marked done/)
    ).toBeInTheDocument();
  });

  it("says nothing is drawn when no bundle was ever built", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          overlayBuildFailed: true,
          overlayManifest: failedManifest({ ngff_url: null, bundle_version: 0 }),
        })}
      />
    );

    expect(
      screen.getByText(/no version of this overlay has ever finished building/)
    ).toBeInTheDocument();
  });

  it("stays out of the way while the build is healthy", () => {
    render(<SegmentationSidebar {...makeProps({ overlayUpdating: true })} />);

    expect(screen.getByText("Overlay updating.")).toBeInTheDocument();
    expect(
      screen.queryByText("Overlay could not be rebuilt")
    ).not.toBeInTheDocument();
  });

  it("does not offer a retry it cannot address", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          overlayBuildFailed: true,
          overlayManifest: failedManifest(),
          overlaySegmentationId: null,
        })}
      />
    );

    expect(
      screen.queryByText("Overlay could not be rebuilt")
    ).not.toBeInTheDocument();
  });
});
