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

function makeFailedManifest(
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
  /**
   * Package 0.3. The overlay controls lived inside an `isCompleted(stage)`
   * branch, and `COMPLETED` is written only by "Mark Image Done" on another
   * screen — so the card for a run that had just finished carried no checkbox,
   * no colour and no opacity, and the canvas beside it stayed bare.
   */
  describe.each<[StatusStage, string]>([
    ["CANDIDATES_READY", "Run finished"],
    ["UPDATING", "Updating..."],
    ["COMPUTING_FEATURES", "Computing features..."],
  ])("a %s card", (stage, label) => {
    it("offers the checkbox, the colour and the opacity", async () => {
      const user = userEvent.setup();
      const handlers = renderSidebar(makeSegmentation(stage));

      expect(screen.getByText(label)).toBeInTheDocument();
      expect(screen.getByText("511 objects")).toBeInTheDocument();

      const checkbox = screen.getByRole("checkbox");
      expect(checkbox).toBeChecked();
      await user.click(checkbox);
      expect(handlers.onToggle).toHaveBeenCalledWith("seg-1");

      expect(screen.getByText("Color")).toBeInTheDocument();
      expect(screen.getByText("Opacity")).toBeInTheDocument();
      expect(screen.getByText("25%")).toBeInTheDocument();
    });

    it("still leads to proofreading and to delete", async () => {
      const user = userEvent.setup();
      const handlers = renderSidebar(makeSegmentation(stage));

      await user.click(screen.getByRole("button", { name: "Edit / Label" }));
      expect(handlers.onEditSegmentation).toHaveBeenCalledWith("seg-1");

      await user.click(
        screen.getByRole("button", { name: "Delete Mitochondria" })
      );
      expect(handlers.onDeleteSegmentation).toHaveBeenCalledWith("seg-1");
    });
  });

  it("keeps the quieter chrome on a COMPLETED card", () => {
    renderSidebar(makeSegmentation("COMPLETED"));

    expect(screen.getByRole("checkbox")).toBeInTheDocument();
    expect(screen.getByText("511 objects")).toBeInTheDocument();
    expect(screen.getByTitle("Open labeling")).toBeInTheDocument();
    // The prominent call to action belongs to a run nobody has checked yet.
    expect(
      screen.queryByRole("button", { name: "Edit / Label" })
    ).not.toBeInTheDocument();
    // "Completed" is the resting state, so it is not announced as news.
    expect(screen.queryByText("Completed")).not.toBeInTheDocument();
  });

  describe("a failure says what went wrong", () => {
    it("prints the server's reason verbatim, not the word Failed alone", () => {
      renderSidebar(
        makeSegmentation("FAILED", {
          status_error:
            "Model pack 'quantem:er' is not installed.\n" +
            "Model packs are downloaded on demand from Models.",
        })
      );

      expect(screen.getByText("Failed")).toBeInTheDocument();
      expect(
        screen.getByText(/Model pack 'quantem:er' is not installed\./)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/downloaded on demand from Models\./)
      ).toBeInTheDocument();
      // Nothing was written, so nothing is offered to draw.
      expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    });

    it("says the reason is missing rather than showing an empty box", () => {
      renderSidebar(makeSegmentation("FAILED", { status_error: "   " }));

      expect(
        screen.getByText("The server recorded no reason for it.")
      ).toBeInTheDocument();
    });

    it("carries a retry note on a run that has not failed yet", () => {
      renderSidebar(
        makeSegmentation("RUNNING_INFERENCE", {
          status_error:
            "Attempt 2 of 3 failed; retrying automatically. CUDA out of memory.",
        })
      );

      expect(screen.getByText("Running inference...")).toBeInTheDocument();
      expect(
        screen.getByText(/Attempt 2 of 3 failed; retrying automatically\./)
      ).toBeInTheDocument();
      // Still running: no overlay to switch on.
      expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    });

    it("stays quiet on a healthy run", () => {
      renderSidebar(makeSegmentation("RUNNING_INFERENCE"));

      expect(
        screen.queryByText(/The server recorded no reason/)
      ).not.toBeInTheDocument();
      expect(screen.getByText("Running inference...")).toBeInTheDocument();
    });
  });

  /**
   * Finding F1. A *successful* run whose overlay raster failed to rebuild:
   * `status_error` is empty, so `OverlayStatusError` renders nothing, and
   * until this card existed the manifest's `last_error` had no renderer at
   * all anywhere in the app.
   */
  describe("a failed overlay build", () => {
    beforeEach(() => {
      vi.mocked(rebuildSegmentationOverlay).mockReset();
    });

    it("is reported even though the run itself succeeded", () => {
      renderSidebar(makeSegmentation("CANDIDATES_READY"), {
        overlayBuildFailures: { "seg-1": makeFailedManifest() },
      });

      // The run is fine and still says so.
      expect(screen.getByText("Run finished")).toBeInTheDocument();
      expect(screen.getByText("511 objects")).toBeInTheDocument();
      // And the picture is not.
      expect(screen.getByText("Overlay could not be rebuilt")).toBeInTheDocument();
      expect(
        screen.getByText(/Reason from the server: \[Errno 28\] No space left on device/)
      ).toBeInTheDocument();
      // The controls stay: an existing bundle is still drawn and still tinted.
      expect(screen.getByRole("checkbox")).toBeChecked();
      expect(screen.getByText("Color")).toBeInTheDocument();
    });

    it("shows nothing on a card with no failure", () => {
      renderSidebar(makeSegmentation("CANDIDATES_READY"), {
        overlayBuildFailures: {},
      });

      expect(
        screen.queryByText("Overlay could not be rebuilt")
      ).not.toBeInTheDocument();
    });

    it("asks the server to build it again", async () => {
      const user = userEvent.setup();
      vi.mocked(rebuildSegmentationOverlay).mockResolvedValue(
        makeFailedManifest({ status: "BUILDING" })
      );
      const handlers = renderSidebar(makeSegmentation("CANDIDATES_READY"), {
        overlayBuildFailures: {
          "seg-1": makeFailedManifest({ source_model: "quantem:mito" }),
        },
      });

      await user.click(
        screen.getByRole("button", { name: /Retry overlay build/i })
      );

      expect(rebuildSegmentationOverlay).toHaveBeenCalledWith(
        "seg-1",
        "full",
        "quantem:mito"
      );
      expect(handlers.onOverlayBuildRetried).toHaveBeenCalledWith("seg-1");
    });
  });
});
