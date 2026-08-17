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

/** The named-model preview bundle failed; the confirmed bundle is fine. */
function modelFailure(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SidebarProps["layers"]["failedOverlays"] {
  return [{ role: "model", manifest: failedManifest(overrides) }];
}

/** The source-less confirmed-display bundle failed. */
function confirmedFailure(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SidebarProps["layers"]["failedOverlays"] {
  return [
    {
      role: "confirmed",
      manifest: failedManifest({ source_model: null, ...overrides }),
    },
  ];
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
      modelOverlayUpdating: false,
      confirmedOverlayUpdating: false,
      failedOverlays: [],
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

  it("says the build failed instead of saying nothing, and says which display", () => {
    render(<SegmentationSidebar {...makeProps({ failedOverlays: modelFailure() })} />);

    expect(
      screen.getByText("The model preview display could not be rebuilt")
    ).toBeInTheDocument();
    expect(screen.getByText(/Your objects are safe/)).toBeInTheDocument();
  });

  it("shows the reason with a path a user can read", () => {
    render(<SegmentationSidebar {...makeProps({ failedOverlays: modelFailure() })} />);

    const reason = screen.getByText(/Cannot create a file/);
    expect(reason).toHaveTextContent(
      "D:\\quantem-data\\tmp\\segmentation_overlays\\seg-1\\staging"
    );
    // V5's defect, stated where it is read rather than where it is produced.
    expect(reason.textContent).not.toContain("\\\\");
  });

  it("warns that the picture on the canvas is out of date", () => {
    render(<SegmentationSidebar {...makeProps({ failedOverlays: modelFailure() })} />);

    expect(
      screen.getByText(/anything changed since then is missing from it/)
    ).toBeInTheDocument();
  });

  /**
   * The guard this replaces queried the exact string "Overlay updating.", which
   * stopped being rendered when the copy was split per display -- so it passed
   * for every possible state, including the regression it was written to catch.
   * The pattern has to cover all three sentences, plural "displays are" too.
   */
  it("never claims a failing display is updating at the same time", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          failedOverlays: modelFailure(),
          modelOverlayUpdating: false,
          confirmedOverlayUpdating: false,
        })}
      />
    );

    expect(
      screen.queryByText(/displays? (is|are) updating/i)
    ).not.toBeInTheDocument();
  });

  it("asks for the build again, for the bundle actually on screen", async () => {
    const user = userEvent.setup();
    const onOverlayBuildRetried = vi.fn();
    rebuildMock.mockResolvedValue(failedManifest({ status: "BUILDING" }));

    render(
      <SegmentationSidebar
        {...makeProps({
          failedOverlays: modelFailure(),
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

    render(<SegmentationSidebar {...makeProps({ failedOverlays: modelFailure() })} />);

    await user.click(screen.getByRole("button", { name: /Retry overlay build/i }));

    expect(
      await screen.findByText(/This segmentation is marked done/)
    ).toBeInTheDocument();
  });

  it("says nothing is drawn when no bundle was ever built", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          failedOverlays: modelFailure({ ngff_url: null, bundle_version: 0 }),
        })}
      />
    );

    expect(
      screen.getByText(/no version of this overlay has ever finished building/)
    ).toBeInTheDocument();
  });

  it("stays out of the way while the build is healthy", () => {
    render(
      <SegmentationSidebar {...makeProps({ modelOverlayUpdating: true })} />
    );

    expect(
      screen.getByText("Model preview display is updating.")
    ).toBeInTheDocument();
    expect(screen.queryByText("Layers")).not.toBeInTheDocument();
    expect(screen.queryByText(/could not be rebuilt/)).not.toBeInTheDocument();
  });

  /**
   * The reachable states, spelled out. `useOverlayManifestState` derives these
   * two flags from two separate manifests, so "confirmed alone" and "both" are
   * ordinary outcomes of a confirm -- which dirties both bundles and gets one
   * rebuild job each -- and neither had any coverage before.
   */
  it("says the confirmed display is rebuilding without blocking analysis", () => {
    render(
      <SegmentationSidebar {...makeProps({ confirmedOverlayUpdating: true })} />
    );

    expect(
      screen.getByText(
        "Confirmed display is updating. Saved objects remain ready for analysis."
      )
    ).toBeInTheDocument();
  });

  it("names both displays when both are rebuilding", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          modelOverlayUpdating: true,
          confirmedOverlayUpdating: true,
        })}
      />
    );

    expect(
      screen.getByText(
        "Model preview and confirmed displays are updating. " +
          "Saved objects remain ready for analysis."
      )
    ).toBeInTheDocument();
  });

  /**
   * Finding 9. The server rebuilds the two bundles one at a time, so a model
   * rebuild that fails while the confirmed one is still queued puts both of
   * these on screen at once. Unnamed, they read as two contradictory claims
   * about a single overlay; named, they are two true statements.
   */
  it("names the failing display apart from the one still rebuilding", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          confirmedOverlayUpdating: true,
          failedOverlays: modelFailure(),
        })}
      />
    );

    expect(
      screen.getByText(
        "Confirmed display is updating. Saved objects remain ready for analysis."
      )
    ).toBeInTheDocument();
    expect(
      screen.getByText("The model preview display could not be rebuilt")
    ).toBeInTheDocument();
  });

  it("surfaces a confirmed-display failure on its own, with its own retry", async () => {
    const user = userEvent.setup();
    rebuildMock.mockResolvedValue(failedManifest({ status: "BUILDING" }));

    render(
      <SegmentationSidebar {...makeProps({ failedOverlays: confirmedFailure() })} />
    );

    expect(
      screen.getByText("The confirmed display could not be rebuilt")
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Retry overlay build/i }));

    // The confirmed bundle is the source-less one: retrying it with the model
    // name would re-queue the wrong bundle and leave this card up for ever.
    expect(rebuildMock).toHaveBeenCalledWith("seg-1", "full", null);
  });

  it("shows a card per failed bundle rather than only the first", async () => {
    const user = userEvent.setup();
    rebuildMock.mockResolvedValue(failedManifest({ status: "BUILDING" }));

    render(
      <SegmentationSidebar
        {...makeProps({
          failedOverlays: [...modelFailure(), ...confirmedFailure()],
        })}
      />
    );

    expect(
      screen.getByText("The model preview display could not be rebuilt")
    ).toBeInTheDocument();
    expect(
      screen.getByText("The confirmed display could not be rebuilt")
    ).toBeInTheDocument();

    // Two cards, two retries, each aimed at its own bundle. Collapsing them
    // left the second display broken while the user believed both were fixed.
    const retries = screen.getAllByRole("button", { name: /Retry overlay build/i });
    expect(retries).toHaveLength(2);
    await user.click(retries[0]);
    await user.click(retries[1]);
    expect(rebuildMock).toHaveBeenCalledWith("seg-1", "full", "quantem:mito");
    expect(rebuildMock).toHaveBeenCalledWith("seg-1", "full", null);
  });

  it("does not offer a retry it cannot address", () => {
    render(
      <SegmentationSidebar
        {...makeProps({
          failedOverlays: modelFailure(),
          overlaySegmentationId: null,
        })}
      />
    );

    expect(screen.queryByText(/could not be rebuilt/)).not.toBeInTheDocument();
  });
});
