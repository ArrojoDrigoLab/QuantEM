import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { OverlaySelectionSidebar } from "@/features/viewer/components/OverlaySelectionSidebar";
import { rebuildSegmentationOverlay } from "@/shared/api/segmentations/overlays";
import type { ImageSegmentation, StatusStage } from "@/shared/types";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

vi.mock("@/shared/api/segmentations/overlays", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/overlays")
  >("@/shared/api/segmentations/overlays");
  return { ...actual, rebuildSegmentationOverlay: vi.fn() };
});

function makeSegmentation(
  stage: StatusStage,
  overrides: Partial<ImageSegmentation> = {}
): ImageSegmentation {
  return {
    id: "seg-1",
    asset: "img-1",
    segmentation_type: {
      id: "type-1",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: "Mitochondria",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    segment_counts: { CANDIDATE: 511 },
    status_stage: stage,
    status_progress: stage === "RUNNING_INFERENCE" ? 30 : 100,
    status_error: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  } as ImageSegmentation;
}

function failedManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return {
    status: "FAILED",
    last_error: "[Errno 28] No space left on device",
    ngff_url: "/segmentation-overlays/seg-1.zarr",
    lut_url: "/api/segmentations/seg-1/overlay-lut/",
    arrays: ["labels", "border"],
    label_dtype: "uint32",
    bundle_version: 3,
    applied_revision: 5,
    desired_revision: 6,
    lut_revision: 5,
    chunk_size: [256, 256],
    level_count: 4,
    width: 1024,
    height: 1024,
    ...overrides,
  };
}

function renderSidebar(
  segmentation: ImageSegmentation,
  overrides: {
    enabled?: boolean;
    overlayBuildFailures?: Record<string, SegmentationOverlayManifest>;
  } = {}
) {
  const handlers = {
    onToggle: vi.fn(),
    onColorChange: vi.fn(),
    onOpacityChange: vi.fn(),
    onEditSegmentation: vi.fn(),
    onDeleteSegmentation: vi.fn(),
    onOverlayBuildRetried: vi.fn(),
  };
  render(
    <OverlaySelectionSidebar
      overlays={[
        {
          segmentation,
          enabled: overrides.enabled ?? true,
          color: "#38bdf8",
          opacity: 0.25,
        },
      ]}
      overlayBuildFailures={overrides.overlayBuildFailures}
      {...handlers}
    />
  );
  return handlers;
}

describe("OverlaySelectionSidebar", () => {
  beforeEach(() => {
    vi.mocked(rebuildSegmentationOverlay).mockReset();
  });

  it("keeps rows compact until their accordion is opened", async () => {
    const user = userEvent.setup();
    const handlers = renderSidebar(makeSegmentation("CANDIDATES_READY"));

    expect(screen.getByRole("heading", { name: "Existing Segmentations" })).toBeInTheDocument();
    expect(screen.getByText("Mito")).toBeInTheDocument();
    expect(screen.queryByText("Color")).not.toBeInTheDocument();
    expect(screen.queryByText("Opacity")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("checkbox", { name: "Show Mitochondria" }));
    expect(handlers.onToggle).toHaveBeenCalledWith("seg-1");

    await user.click(
      screen.getByRole("button", { name: "Edit Labels for Mitochondria" })
    );
    expect(handlers.onEditSegmentation).toHaveBeenCalledWith("seg-1");

    await user.click(screen.getByRole("button", { name: "Expand Mitochondria" }));
    expect(screen.getByText("Color")).toBeInTheDocument();
    expect(screen.getByText("Opacity")).toBeInTheDocument();
    expect(screen.getByText("511 objects")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit Labels" }));
    expect(handlers.onEditSegmentation).toHaveBeenLastCalledWith("seg-1");
    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(handlers.onDeleteSegmentation).toHaveBeenCalledWith("seg-1");
  });

  it("keeps a failed segmentation visible and available for manual labeling", async () => {
    const user = userEvent.setup();
    renderSidebar(
      makeSegmentation("FAILED", {
        status_error: "Model pack 'quantem:er' is not installed.",
      })
    );

    const checkbox = screen.getByRole("checkbox", { name: "Show Mitochondria" });
    expect(checkbox).toBeDisabled();
    expect(screen.getByRole("button", { name: "Edit Labels for Mitochondria" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Expand Mitochondria" }));
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText(/Model pack 'quantem:er' is not installed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit Labels" })).toBeInTheDocument();
  });

  it("shows an overlay build failure only after the relevant row is opened", async () => {
    const user = userEvent.setup();
    renderSidebar(makeSegmentation("CANDIDATES_READY"), {
      overlayBuildFailures: { "seg-1": failedManifest() },
    });

    expect(screen.queryByText("Overlay could not be rebuilt")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Expand Mitochondria" }));
    expect(screen.getByText("Overlay could not be rebuilt")).toBeInTheDocument();
    expect(
      screen.getByText(/Reason from the server: \[Errno 28\] No space left on device/)
    ).toBeInTheDocument();
  });
});
