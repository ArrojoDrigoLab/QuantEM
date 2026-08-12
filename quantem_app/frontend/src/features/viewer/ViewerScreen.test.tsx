import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerScreen } from "@/features/viewer/ViewerScreen";
import { useSegmentationOverlayManifests } from "@/hooks/useSegmentationOverlayManifest";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import type { AssetDetail, StatusStage } from "@/shared/types";
import {
  createAssetSegmentation,
  getAsset,
  getAssetNgffUrl,
  getAssetSegmentations,
} from "@/shared/api/assets";
import {
  getSegmentationOverlayLutJson,
  rebuildSegmentationOverlay,
} from "@/shared/api/segmentations/overlays";
import {
  deleteSegmentation,
  getSegmentationDetail,
} from "@/shared/api/segmentations/lifecycle";
import { ApiRequestError } from "@/shared/api/core/http";
import type {
  SegmentationDeletePreview,
  SegmentationOverlayManifest,
} from "@/shared/types/segmentation";

const { navigateMock, viewerPropsSpy } = vi.hoisted(() => ({
  navigateMock: vi.fn(),
  viewerPropsSpy: vi.fn(),
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  );
  return {
    ...actual,
    useNavigate: () => navigateMock,
  };
});

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return {
    ...actual,
    getAsset: vi.fn(),
    getAssetNgffUrl: vi.fn(),
    getAssetSegmentations: vi.fn(),
    createAssetSegmentation: vi.fn(),
  };
});

vi.mock("@/shared/api/segmentations/overlays", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/overlays")
  >("@/shared/api/segmentations/overlays");
  return {
    ...actual,
    getSegmentationOverlayLutJson: vi.fn(),
    rebuildSegmentationOverlay: vi.fn(),
  };
});

vi.mock("@/shared/api/segmentations/lifecycle", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/lifecycle")
  >("@/shared/api/segmentations/lifecycle");
  return {
    ...actual,
    getSegmentationDetail: vi.fn(),
    deleteSegmentation: vi.fn(),
  };
});

vi.mock("@/viewer/components/ImageViewer", () => ({
  ImageViewer: (props: unknown) => {
    viewerPropsSpy(props);
    return <div data-testid="image-viewer" />;
  },
}));

// Only the fetching hook is faked. The predicates that decide what a manifest
// *means* live in `@/hooks/overlayManifestStatus` and are deliberately left
// real: stubbing them would make the F1 assertions below vacuous.
vi.mock("@/hooks/useSegmentationOverlayManifest", () => ({
  useSegmentationOverlayManifests: vi.fn(() => ({
    manifests: {},
    loading: false,
    refetching: false,
    refetch: vi.fn(),
  })),
}));

function makeImage(overrides: Partial<AssetDetail> = {}): AssetDetail {
  return {
    id: "img-1",
    file_path: "",
    original_filename: "source.tif",
    display_name: "Image 1",
    is_eval_set: false,
    width: 1024,
    height: 1024,
    channels: 1,
    bit_depth: 8,
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ngff_ready: true,
    ngff_url: "/ngff/img-1.zarr",
    ...overrides,
  };
}

function segmentation(
  status: StatusStage = "COMPLETED",
  overrides: {
    id?: string;
    longName?: string;
    image?: string;
    statusError?: string | null;
    segmentCounts?: Record<string, number>;
  } = {}
) {
  return {
    id: overrides.id ?? "seg-1",
    image: overrides.image ?? "img-1",
    segmentation_type: {
      id: "type-1",
      internal_name: "quantem_internal_mito",
      short_name: "Mito",
      long_name: overrides.longName ?? "Lipid Droplets",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    status_stage: status,
    status_progress: status === "COMPLETED" ? 100 : 30,
    status_error: overrides.statusError ?? null,
    segment_counts: overrides.segmentCounts ?? { CONFIRMED: 3 },
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

/** A built, up-to-date overlay bundle for `segmentationId`. */
function readyManifest(segmentationId: string) {
  return {
    status: "READY" as const,
    ngff_url: `/segmentation-overlays/${segmentationId}.zarr`,
    lut_url: `/api/segmentations/${segmentationId}/overlay-lut/`,
    arrays: ["labels", "border"],
    label_dtype: "uint32" as const,
    bundle_version: 1,
    applied_revision: 4,
    desired_revision: 4,
    lut_revision: 4,
    chunk_size: [256, 256] as [number, number],
    level_count: 4,
    width: 1024,
    height: 1024,
  };
}

/**
 * The shape finding F1 was reproduced from: the run succeeded and left
 * objects, then the overlay rebuild to revision 6 died on a stray file. The
 * server has stopped re-queueing, so `desired_revision > applied_revision`
 * here is permanent.
 */
const OVERLAY_BUILD_ERROR =
  "[WinError 183] Cannot create a file when that file already exists: " +
  "'D:\\data\\tmp\\segmentation_overlays\\seg-1\\staging'";

function failedManifest(
  segmentationId: string,
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return {
    ...readyManifest(segmentationId),
    status: "FAILED",
    applied_revision: 5,
    desired_revision: 6,
    last_error: OVERLAY_BUILD_ERROR,
    ...overrides,
  };
}

type IdMapOverlayProps = {
  id: string;
  ngffUrl: string;
  lut: Uint8Array;
  maxLabel: number;
  fillOpacity: number;
};

function lastIdMapOverlays(): IdMapOverlayProps[] {
  const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
    overlays?: { idMapOverlays?: IdMapOverlayProps[] };
  };
  return lastCall?.overlays?.idMapOverlays ?? [];
}

/** The alpha byte the LUT assigns `label`: 255 drawn, 0 invisible. */
function lutAlpha(lut: Uint8Array, label: number): number {
  return lut[label * 4 + 3];
}

/**
 * Drive the interval the component registered at `delay` ms, in place of real
 * time. Returns a handle that restores the spies.
 */
function captureIntervals() {
  const intervals: Array<{ id: number; delay: number; callback: () => void }> = [];
  let nextIntervalId = 1;
  const setIntervalSpy = vi.spyOn(window, "setInterval").mockImplementation(
    ((handler: TimerHandler, delay?: number) => {
      const id = nextIntervalId++;
      if (typeof handler === "function" && typeof delay === "number") {
        intervals.push({ id, delay, callback: handler as () => void });
      }
      return id as unknown as ReturnType<typeof window.setInterval>;
    }) as unknown as typeof window.setInterval
  );
  const clearIntervalSpy = vi
    .spyOn(window, "clearInterval")
    .mockImplementation((() => undefined) as typeof window.clearInterval);
  return {
    intervals,
    restore: () => {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    },
  };
}

describe("ViewerScreen", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useRealTimers();
    navigateMock.mockReset();
    viewerPropsSpy.mockReset();
    useSelectionStore.getState().clearSelection();
    useSelectionStore.getState().setSelectedImageId("img-1");
    vi.mocked(getAssetNgffUrl).mockReturnValue("/ngff/assets/img-1.zarr");
    vi.mocked(getAsset).mockResolvedValue(makeImage());
    vi.mocked(getAssetSegmentations).mockResolvedValue([segmentation()]);
    vi.mocked(getSegmentationOverlayLutJson).mockResolvedValue({
      lut_revision: 1,
      bundle_version: 1,
      max_label: 1,
      objects: [
        { label: 1, uuid: "obj-1", is_cell: false, state: "confirmed", color: "33CC66" },
      ],
    });
    vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
      manifests: {},
      loading: false,
      refetching: false,
      refetch: vi.fn(),
    });
  });

  it("shows disabled overlay message while preprocessing is incomplete", async () => {
    vi.mocked(getAsset).mockResolvedValue(
      makeImage({
        preprocess_stage: "ENCODING",
        preprocess_progress: 40,
        ngff_ready: false,
        ngff_url: null,
      })
    );

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    await screen.findByText("Segmentations");
    expect(
      screen.getByText(/Overlays are unavailable until preprocessing is complete/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/Encoding image \(40%\)/i)).toBeInTheDocument();
  });

  it("shows the experiment above the image name and returns to that experiment", async () => {
    const user = userEvent.setup();
    vi.mocked(getAsset).mockResolvedValue(
      makeImage({
        experiment_id: "exp-1",
        experiment_name: "Glucose infusion",
        pixel_size_nm: 5,
      })
    );

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    expect(await screen.findByText("Glucose infusion")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Image 1" })).toBeInTheDocument();
    expect(screen.queryByText("source.tif")).not.toBeInTheDocument();
    expect(screen.getByText("5 nm/px")).toBeInTheDocument();
    expect(screen.queryByText(/entered by hand|from file/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit pixel size" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Experiment/ }));
    expect(navigateMock).toHaveBeenCalledWith("/?experiment=exp-1");
  });

  it("navigates to labeling route when edit segmentation is clicked", async () => {
    const user = userEvent.setup();
    vi.mocked(getAsset).mockResolvedValue(makeImage());

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    await screen.findByRole("button", { name: "Edit Labels for Lipid Droplets" });
    await user.click(
      screen.getByRole("button", { name: "Edit Labels for Lipid Droplets" })
    );

    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith(
        "/assets/img-1/labeling/Lipid%20Droplets"
      );
    });
    expect(useSelectionStore.getState().selectedSegmentationId).toBe("seg-1");
  });

  it("creates a segmentation and navigates to the new labeling route", async () => {
    const user = userEvent.setup();
    vi.mocked(getAssetSegmentations).mockResolvedValue([]);
    vi.mocked(createAssetSegmentation).mockResolvedValue(
      segmentation("COMPLETED", {
        id: "seg-new",
        longName: "Mitochondria",
      })
    );

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    await user.click(await screen.findByRole("button", { name: "Mitochondria" }));
    // Creating an organelle segmentation queues a whole-image inference run, so
    // it now asks first. The navigation still has to happen once confirmed.
    await user.click(await screen.findByRole("button", { name: /Create and run/ }));

    await waitFor(() => {
      expect(createAssetSegmentation).toHaveBeenCalledWith("img-1", {
        segmentation_type_name: "Mitochondria",
      });
    });
    await waitFor(() => {
      expect(navigateMock).toHaveBeenCalledWith("/assets/img-1/labeling/Mitochondria");
    });

    expect(useSelectionStore.getState().selectedSegmentationId).toBe("seg-new");
    expect(vi.mocked(getAssetSegmentations).mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it("polls image status until preprocessing reaches a terminal stage", async () => {
    type IntervalEntry = { id: number; delay: number; callback: () => void };
    const intervals: IntervalEntry[] = [];
    let nextIntervalId = 1;
    const setIntervalSpy = vi.spyOn(window, "setInterval").mockImplementation(
      ((handler: TimerHandler, delay?: number) => {
        const id = nextIntervalId++;
        if (typeof handler === "function" && typeof delay === "number") {
          intervals.push({ id, delay, callback: handler as () => void });
        }
        return id as unknown as ReturnType<typeof window.setInterval>;
      }) as unknown as typeof window.setInterval
    );
    const clearIntervalSpy = vi.spyOn(window, "clearInterval").mockImplementation(
      (() => undefined) as typeof window.clearInterval
    );
    vi.mocked(getAsset)
      .mockResolvedValueOnce(
        makeImage({
          preprocess_stage: "ENCODING",
          preprocess_progress: 40,
          ngff_ready: false,
          ngff_url: null,
        })
      )
      .mockResolvedValue(makeImage());
    vi.mocked(getAssetSegmentations).mockResolvedValue([segmentation()]);

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    try {
      await screen.findByText(/Encoding image \(40%\)/i);
      const initialCalls = vi.mocked(getAsset).mock.calls.length;
      expect(initialCalls).toBeGreaterThan(0);
      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());
      const pollInterval = intervals.find((entry) => entry.delay === 2000);
      expect(pollInterval).toBeDefined();

      await act(async () => {
        pollInterval?.callback();
      });
      await waitFor(() =>
        expect(vi.mocked(getAsset).mock.calls.length).toBe(initialCalls + 1)
      );
      await waitFor(() =>
        expect(
          screen.queryByText(/Overlays are unavailable until preprocessing is complete/i)
        ).not.toBeInTheDocument()
      );
      await waitFor(() =>
        expect(clearIntervalSpy).toHaveBeenCalledWith(pollInterval?.id)
      );
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  it("polls segmentation status while processing and stops after completion", async () => {
    const user = userEvent.setup();
    type IntervalEntry = { id: number; delay: number; callback: () => void };
    const intervals: IntervalEntry[] = [];
    let nextIntervalId = 1;
    const setIntervalSpy = vi.spyOn(window, "setInterval").mockImplementation(
      ((handler: TimerHandler, delay?: number) => {
        const id = nextIntervalId++;
        if (typeof handler === "function" && typeof delay === "number") {
          intervals.push({ id, delay, callback: handler as () => void });
        }
        return id as unknown as ReturnType<typeof window.setInterval>;
      }) as unknown as typeof window.setInterval
    );
    const clearIntervalSpy = vi.spyOn(window, "clearInterval").mockImplementation(
      (() => undefined) as typeof window.clearInterval
    );
    vi.mocked(getAsset).mockResolvedValue(makeImage());
    vi.mocked(getAssetSegmentations)
      .mockResolvedValueOnce([segmentation("RUNNING_INFERENCE")])
      .mockResolvedValue([segmentation("COMPLETED")]);

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    try {
      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      expect(screen.getByText("Running inference...")).toBeInTheDocument();
      const initialCalls = vi.mocked(getAssetSegmentations).mock.calls.length;
      expect(initialCalls).toBeGreaterThan(0);
      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());
      const pollInterval = intervals.find((entry) => entry.delay === 3000);
      expect(pollInterval).toBeDefined();

      await act(async () => {
        pollInterval?.callback();
      });
      await waitFor(() =>
        expect(vi.mocked(getAssetSegmentations).mock.calls.length).toBe(initialCalls + 1)
      );
      await waitFor(() =>
        expect(screen.queryByText("Running inference...")).not.toBeInTheDocument()
      );
      await waitFor(() =>
        expect(clearIntervalSpy).toHaveBeenCalledWith(pollInterval?.id)
      );
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });

  /**
   * Paper-cut 2: there was no way to delete a segmentation, and its preset
   * left "Add segmentation" forever. The deletion is to the Mark-Done
   * standard: the dialog quotes counts read fresh from the server when it
   * opens, the DELETE carries the acknowledged object count, and refusals
   * are rendered in the dialog instead of closing it into silence.
   */
  describe("deleting a segmentation", () => {
    function makeDeletePreview(
      overrides: Partial<SegmentationDeletePreview> = {}
    ): SegmentationDeletePreview {
      return {
        segmentation_id: "seg-1",
        segmentation_type: "Lipid Droplets",
        object_count: 12,
        objects_by_label_state: {
          CONFIRMED: 3,
          EXCLUDED: 2,
          CANDIDATE: 5,
          INFERRED: 2,
        },
        probability_map_count: 1,
        overlay_count: 2,
        adapter_count: 1,
        analysis_run_count: 2,
        locked: false,
        ...overrides,
      };
    }

    function useDeletePreview(
      overrides: Partial<SegmentationDeletePreview> = {}
    ) {
      vi.mocked(getSegmentationDetail).mockResolvedValue({
        ...segmentation(),
        delete_preview: makeDeletePreview(overrides),
      } as never);
    }

    async function openDeleteDialog(user: ReturnType<typeof userEvent.setup>) {
      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );
      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      await user.click(screen.getByRole("button", { name: "Delete" }));
      return screen.findByRole("dialog");
    }

    it("asks first, quoting live counts of everything deletion destroys", async () => {
      const user = userEvent.setup();
      useDeletePreview();

      await openDeleteDialog(user);

      // The counts are the server's, read when the dialog opened — not the
      // possibly-stale list payload.
      expect(vi.mocked(getSegmentationDetail)).toHaveBeenCalledWith("seg-1");
      expect(
        await screen.findByText(
          /This deletes all 12 objects on this segmentation — 3 confirmed, 2 rejected and 7 nobody reviewed/
        )
      ).toBeInTheDocument();
      expect(
        screen.getByText(/2 overlay rasters, 1 probability map, 1 adapted model/)
      ).toBeInTheDocument();
      // What survives is said too: the analysis record is kept, marked.
      expect(
        screen.getByText(/The 2 analysis runs made from it are/)
      ).toBeInTheDocument();
      expect(screen.getByText(/"segmentation deleted"/)).toBeInTheDocument();
      // And what the delete frees: the preset comes back.
      expect(
        screen.getByText(/preset returns to "Add segmentation"/)
      ).toBeInTheDocument();
      expect(vi.mocked(deleteSegmentation)).not.toHaveBeenCalled();
    });

    it("deletes only after confirmation, sending the acknowledged count", async () => {
      const user = userEvent.setup();
      useDeletePreview();
      vi.mocked(deleteSegmentation).mockResolvedValue({
        deleted: makeDeletePreview(),
        analysis_runs_kept: 2,
      });

      await openDeleteDialog(user);
      const segmentationsCallsBefore =
        vi.mocked(getAssetSegmentations).mock.calls.length;
      await user.click(
        await screen.findByRole("button", {
          name: "Delete 12 objects and this segmentation",
        })
      );

      await waitFor(() => {
        expect(vi.mocked(deleteSegmentation)).toHaveBeenCalledWith("seg-1", 12);
      });
      // The list is re-read, which is what makes the preset reappear in
      // "Add segmentation".
      await waitFor(() => {
        expect(
          vi.mocked(getAssetSegmentations).mock.calls.length
        ).toBeGreaterThan(segmentationsCallsBefore);
      });
      await waitFor(() => {
        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      });
    });

    it("keeps the dialog open on a refusal and prints the server's reason", async () => {
      const user = userEvent.setup();
      useDeletePreview();
      vi.mocked(deleteSegmentation).mockRejectedValue(
        new ApiRequestError(
          JSON.stringify({
            detail:
              "This segmentation cannot be deleted while a " +
              "run_segmentation_full_task job is running on it (job j-1). " +
              "Cancel it (POST /api/jobs/j-1/cancel/) and delete again once " +
              "it has stopped.",
          }),
          { status: 409 }
        )
      );

      await openDeleteDialog(user);
      const detailCallsBefore =
        vi.mocked(getSegmentationDetail).mock.calls.length;
      await user.click(
        await screen.findByRole("button", {
          name: "Delete 12 objects and this segmentation",
        })
      );

      expect(
        await screen.findByText(/cannot be deleted while a/)
      ).toBeInTheDocument();
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      // The counts are re-read beside the refusal: the usual cause of a 409
      // is a run that just finished, i.e. the numbers moved.
      await waitFor(() => {
        expect(
          vi.mocked(getSegmentationDetail).mock.calls.length
        ).toBeGreaterThan(detailCallsBefore);
      });
    });

    it("says up front when the completion lock will refuse the delete", async () => {
      const user = userEvent.setup();
      useDeletePreview({ locked: true });

      await openDeleteDialog(user);

      expect(
        await screen.findByText(/This segmentation is locked\./)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Unlock it first/)
      ).toBeInTheDocument();
    });

    it("deletes nothing when the confirmation is cancelled", async () => {
      const user = userEvent.setup();
      useDeletePreview();

      await openDeleteDialog(user);
      await user.click(screen.getByRole("button", { name: "Cancel" }));

      expect(vi.mocked(deleteSegmentation)).not.toHaveBeenCalled();
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  /**
   * Package 0.3: a finished run is visible.
   *
   * A run that finishes leaves `CANDIDATES_READY`; `COMPLETED` is written only
   * by "Mark Image Done" on the labeling screen. The viewer gated both the
   * overlay's `enabled` flag and its sidebar controls on `COMPLETED`, and then
   * tinted only `{confirmed, refined, labeled}` — so the result of an 11-27
   * minute run was undrawn twice over, and the only way to see it was to leave
   * the viewer and declare the image finished first.
   */
  describe("a finished run draws itself", () => {
    beforeEach(() => {
      vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
        manifests: { "seg-1": readyManifest("seg-1") },
        loading: false,
        refetching: false,
        refetch: vi.fn(),
      });
    });

    it("draws the objects of a run sitting at CANDIDATES_READY", async () => {
      const user = userEvent.setup();
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        segmentation("CANDIDATES_READY", { segmentCounts: { CANDIDATE: 17 } }),
      ]);
      vi.mocked(getSegmentationOverlayLutJson).mockResolvedValue({
        lut_revision: 4,
        bundle_version: 1,
        max_label: 1,
        objects: [
          { label: 1, uuid: "obj-1", is_cell: false, state: "candidate", color: "FF0000" },
        ],
      });

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await screen.findByTestId("image-viewer");
      // No click, no navigation, no Mark Image Done: the canvas has the layer.
      await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(1));
      const overlay = lastIdMapOverlays()[0];
      expect(overlay.id).toBe("seg-1");
      expect(lutAlpha(overlay.lut, 1)).toBe(255);
      // And the segmentation is still CANDIDATES_READY — the card offers the
      // route to proofreading rather than having taken it.
      await user.click(
        screen.getByRole("button", { name: "Expand Lipid Droplets" })
      );
      expect(
        screen.getByRole("button", { name: "Edit Labels" })
      ).toBeInTheDocument();
      expect(screen.getByText("Run finished")).toBeInTheDocument();
    });

    it("gives the checkbox, colour and opacity to a CANDIDATES_READY card", async () => {
      const user = userEvent.setup();
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        segmentation("CANDIDATES_READY"),
      ]);

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      const checkbox = await screen.findByRole("checkbox");
      expect(checkbox).toBeChecked();
      expect(screen.queryByText("Color")).not.toBeInTheDocument();
      await user.click(
        screen.getByRole("button", { name: "Expand Lipid Droplets" })
      );
      expect(screen.getByText("Color")).toBeInTheDocument();
      expect(screen.getByText("Opacity")).toBeInTheDocument();
      expect(screen.getByText("25%")).toBeInTheDocument();

      // The checkbox is live: unticking it takes the layer off the canvas.
      await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(1));
      await user.click(checkbox);
      await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(0));
    });

    it("switches an overlay on the moment its run finishes", async () => {
      const user = userEvent.setup();
      const timers = captureIntervals();
      vi.mocked(getAssetSegmentations)
        .mockResolvedValueOnce([segmentation("RUNNING_INFERENCE")])
        .mockResolvedValue([segmentation("CANDIDATES_READY")]);

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      try {
        await user.click(
          await screen.findByRole("button", { name: "Expand Lipid Droplets" })
        );
        expect(screen.getByText("Running inference...")).toBeInTheDocument();
        expect(lastIdMapOverlays()).toHaveLength(0);

        const poll = timers.intervals.find((entry) => entry.delay === 3000);
        expect(poll).toBeDefined();
        await act(async () => {
          poll?.callback();
        });

        // The user did nothing but watch. The overlay appears.
        await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(1));
      } finally {
        timers.restore();
      }
    });

    it("leaves an overlay the user turned off turned off across polls", async () => {
      const user = userEvent.setup();
      const timers = captureIntervals();
      // Mito finished; nucleus is still running, which keeps the 3 s poll
      // alive and re-seeds the overlay config on every tick.
      const cards = () => [
        segmentation("CANDIDATES_READY", { id: "seg-1", longName: "Mitochondria" }),
        segmentation("RUNNING_INFERENCE", { id: "seg-2", longName: "Nucleus" }),
      ];
      vi.mocked(getAssetSegmentations).mockResolvedValue(cards());

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      try {
        const checkbox = await screen.findByRole("checkbox", {
          name: "Show Mitochondria",
        });
        await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(1));
        await user.click(checkbox);
        await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(0));

        const poll = timers.intervals.find((entry) => entry.delay === 3000);
        expect(poll).toBeDefined();
        await act(async () => {
          poll?.callback();
        });

        expect(
          screen.getByRole("checkbox", { name: "Show Mitochondria" })
        ).not.toBeChecked();
        expect(lastIdMapOverlays()).toHaveLength(0);
      } finally {
        timers.restore();
      }
    });

    it("draws guesses and kept objects, and not ones the user removed", async () => {
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        segmentation("CANDIDATES_READY", {
          segmentCounts: { CANDIDATE: 1, CONFIRMED: 1, EXCLUDED: 1 },
        }),
      ]);
      vi.mocked(getSegmentationOverlayLutJson).mockResolvedValue({
        lut_revision: 4,
        bundle_version: 1,
        max_label: 3,
        objects: [
          { label: 1, uuid: "obj-1", is_cell: false, state: "candidate", color: "FF0000" },
          { label: 2, uuid: "obj-2", is_cell: false, state: "confirmed", color: "33CC66" },
          { label: 3, uuid: "obj-3", is_cell: false, state: "excluded", color: "F59E0B" },
        ],
      });

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await screen.findByTestId("image-viewer");
      await waitFor(() => expect(lastIdMapOverlays()).toHaveLength(1));
      const { lut } = lastIdMapOverlays()[0];
      expect(lutAlpha(lut, 1)).toBe(255);
      expect(lutAlpha(lut, 2)).toBe(255);
      // A removal is a decision. Painting it back would report it as kept.
      expect(lutAlpha(lut, 3)).toBe(0);
    });

    it("says why a run failed instead of the bare word Failed", async () => {
      const user = userEvent.setup();
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        segmentation("FAILED", {
          statusError:
            "Model pack 'quantem:er' is not installed.\n" +
            "Model packs are downloaded on demand.",
        }),
      ]);

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      expect(screen.getByText("Failed")).toBeInTheDocument();
      expect(
        screen.getByText(/Model pack 'quantem:er' is not installed\./)
      ).toBeInTheDocument();
      // A failed run wrote nothing, so nothing is drawn and nothing is offered
      // to draw.
      expect(screen.getByRole("checkbox")).toBeDisabled();
      expect(lastIdMapOverlays()).toHaveLength(0);
    });
  });

  it("passes layered overlay props to ImageViewer", async () => {
    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    await screen.findByTestId("image-viewer");
    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      overlays?: {
        rasterLayers?: unknown[];
        persistent?: unknown[];
        transient?: unknown[];
      };
    };

    expect(lastCall.overlays?.rasterLayers).toBeUndefined();
    expect(lastCall.overlays?.persistent).toBeUndefined();
    expect(lastCall.overlays?.transient).toBeUndefined();
  });

  it("builds confirmed fill and border overlay layers from the manifest", async () => {
    vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
      manifests: {
        "seg-1": {
          status: "READY",
          ngff_url: "/segmentation-overlays/seg-1.zarr",
          lut_url: "/api/segmentations/seg-1/overlay-lut/",
          arrays: ["labels", "border"],
          label_dtype: "uint32",
          bundle_version: 1,
          applied_revision: 4,
          desired_revision: 4,
          lut_revision: 4,
          chunk_size: [256, 256],
          level_count: 4,
          width: 1024,
          height: 1024,
        },
      },
      loading: false,
      refetching: false,
      refetch: vi.fn(),
    });

    render(
      <MemoryRouter>
        <ViewerScreen />
      </MemoryRouter>
    );

    await screen.findByTestId("image-viewer");
    await waitFor(() => {
      const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
        overlays?: {
          rasterLayers?: unknown[];
          idMapOverlays?: Array<{
            id: string;
            ngffUrl: string;
            fillOpacity: number;
            borderOpacity: number;
            showBorders: boolean;
            maxLabel: number;
            lut: Uint8Array;
          }>;
        };
      };
      // Segmentation overlays are now ID-map (render-time LUT) overlays, not
      // pre-coloured channel rasters.
      expect(lastCall.overlays?.rasterLayers).toBeUndefined();
      expect(lastCall.overlays?.idMapOverlays).toHaveLength(1);
      expect(lastCall.overlays?.idMapOverlays?.[0]).toMatchObject({
        id: "seg-1",
        ngffUrl: "/segmentation-overlays/seg-1.zarr?rev=1-4",
        fillOpacity: 0.25,
        borderOpacity: 0.95,
        showBorders: true,
        maxLabel: 1,
      });
      expect(lastCall.overlays?.idMapOverlays?.[0].lut).toBeInstanceOf(Uint8Array);
    });
  });

  /**
   * Finding F1. A failed overlay build used to leave "Overlay updating…" on
   * screen for ever (53 consecutive FAILED polls over 6 min 19 s, including a
   * server restart) while the reason sat unread on every one of those
   * responses.
   */
  describe("a failed overlay build", () => {
    beforeEach(() => {
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        segmentation("CANDIDATES_READY", { segmentCounts: { CANDIDATE: 511 } }),
      ]);
      vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
        manifests: { "seg-1": failedManifest("seg-1") },
        loading: false,
        refetching: false,
        refetch: vi.fn(),
      });
    });

    it("stops claiming the overlay is updating", async () => {
      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await screen.findByRole("button", { name: "Expand Lipid Droplets" });
      expect(screen.queryByText(/Overlay updating/i)).not.toBeInTheDocument();
    });

    it("says on screen that the build failed, and why, verbatim", async () => {
      const user = userEvent.setup();
      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );

      // The header slot the eternal spinner used to occupy.
      expect(
        screen.getByText(/Overlay could not be rebuilt — see the card/i)
      ).toBeInTheDocument();
      // The card: the plain-language part...
      expect(
        screen.getByText(/Your objects are safe/i)
      ).toBeInTheDocument();
      // ...and the server's actual reason, not a paraphrase of it.
      expect(
        screen.getByText(new RegExp(escapeForRegExp(OVERLAY_BUILD_ERROR)))
      ).toBeInTheDocument();
      // Which picture they are looking at, and how far behind it is.
      expect(
        screen.getByText(/Revision 5 is on disk; revision 6 was requested/i)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/last overlay that built successfully/i)
      ).toBeInTheDocument();
    });

    it("says nothing is drawn when no bundle was ever built", async () => {
      const user = userEvent.setup();
      vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
        manifests: {
          "seg-1": failedManifest("seg-1", {
            ngff_url: null,
            bundle_version: 0,
            applied_revision: 0,
            desired_revision: 1,
          }),
        },
        loading: false,
        refetching: false,
        refetch: vi.fn(),
      });

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      expect(
        screen.getByText(/no version of this overlay has ever finished building/i)
      ).toBeInTheDocument();
      expect(
        screen.queryByText(/last overlay that built successfully/i)
      ).not.toBeInTheDocument();
    });

    it("names no reason honestly when the server recorded none", async () => {
      const user = userEvent.setup();
      vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
        manifests: { "seg-1": failedManifest("seg-1", { last_error: "" }) },
        loading: false,
        refetching: false,
        refetch: vi.fn(),
      });

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      expect(
        screen.getByText(/The server recorded no reason for it/i)
      ).toBeInTheDocument();
    });

    it("retries the build and refetches the manifest", async () => {
      const user = userEvent.setup();
      const refetch = vi.fn().mockResolvedValue(undefined);
      vi.mocked(useSegmentationOverlayManifests).mockReturnValue({
        manifests: { "seg-1": failedManifest("seg-1") },
        loading: false,
        refetching: false,
        refetch,
      });
      vi.mocked(rebuildSegmentationOverlay).mockResolvedValue(
        readyManifest("seg-1")
      );

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      await user.click(screen.getByRole("button", { name: /Retry overlay build/i }));

      await waitFor(() => {
        expect(rebuildSegmentationOverlay).toHaveBeenCalledWith(
          "seg-1",
          "full",
          null
        );
      });
      await waitFor(() => expect(refetch).toHaveBeenCalled());
    });

    it("renders a refused retry instead of swallowing it", async () => {
      const user = userEvent.setup();
      vi.mocked(rebuildSegmentationOverlay).mockRejectedValue(
        new ApiRequestError("Segmentation is complete and locked.", { status: 409 })
      );

      render(
        <MemoryRouter>
          <ViewerScreen />
        </MemoryRouter>
      );

      await user.click(
        await screen.findByRole("button", { name: "Expand Lipid Droplets" })
      );
      await user.click(screen.getByRole("button", { name: /Retry overlay build/i }));

      expect(
        await screen.findByText(/Segmentation is complete and locked/i)
      ).toBeInTheDocument();
    });
  });
});

/** Escape a literal (Windows paths are full of regex metacharacters). */
function escapeForRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
