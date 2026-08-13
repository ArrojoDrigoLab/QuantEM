import { vi } from "vitest";
import { useHoverSelection } from "@/hooks/useHoverSelection";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { useUserFeedbackStore } from "@/shared/stores/useUserFeedbackStore";
import type {
  AssetDetail,
  ImageSegmentation,
  JobQueueStatus,
  SegmentationOverlayManifest,
  SegmentObject,
  SegmentationRoi,
} from "@/shared/types";
import {
  getAsset,
  getAssetSegmentations,
} from "@/shared/api/assets";
import { getJobQueueStatus } from "@/shared/api/jobs";
import {
  confirmSegmentsBatch,
  getSegmentsAtPoint,
  listUserFeedback,
  markSegmentationComplete,
  unlockSegmentation,
  querySegmentsInRegion,
  updateSegmentLabelsBatch,
} from "@/shared/api/segmentations/annotations";
import {
  activateSegmentationRoi,
  createSegmentationRoi,
  getSegmentationRois,
  rerunSegmentationRoi,
} from "@/shared/api/segmentations/rois";
import {
  createCompletedRoi,
  getCompletedRois,
} from "@/shared/api/segmentations/completedRois";

const harness = vi.hoisted(() => ({
  overlayManifestHookSpy: vi.fn(),
  overlayManifestState: {
    manifest: null as SegmentationOverlayManifest | null,
  },
  overlayManifestRefetchMock: vi.fn(),
  thresholdState: {
    threshold: 0.99,
    setThreshold: vi.fn(),
  },
  viewportSyncState: {
    viewport: null,
    publishFromViewer: vi.fn(),
  },
  segmentsState: {
    uncertainSegments: [],
    refetchUncertainSegments: vi.fn(),
  },
  workflowModeState: {
    workflowMode: "review",
    leftMode: "hover",
    setWorkflowMode: vi.fn(),
    setLeftMode: vi.fn(),
  },
  hoverSelectionState: {
    hoverSegments: [] as SegmentObject[],
    hoverIndex: 0,
    highlightedSegmentId: null as string | null,
    hoverActionMode: "test" as const,
    hoverPoint: null as { x: number; y: number } | null,
    setHoverActionMode: vi.fn(),
    findSegmentsAtPoint: vi.fn(),
    cycleHoverIndex: vi.fn(),
    clearHover: vi.fn(),
  },
  drawingState: {
    pendingPolygon: null,
    pendingPolygonOperation: "include" as const,
    brushStrokes: [],
    brushSize: 24,
    draftOperation: "include" as const,
    clearDrawing: vi.fn(),
    setBrushSize: vi.fn(),
    setDraftOperation: vi.fn(),
    handleBrushStroke: vi.fn(),
    handleDrawComplete: vi.fn(),
    getBrushPolygons: vi.fn(() => []),
    getBrushPolygonRings: vi.fn(() => []),
    eraseBrushStrokesAt: vi.fn(),
  },
}));

export const {
  overlayManifestHookSpy,
  overlayManifestState,
  overlayManifestRefetchMock,
  thresholdState,
  viewportSyncState,
  segmentsState,
  workflowModeState,
  hoverSelectionState,
  drawingState,
} = harness;

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return {
    ...actual,
    getAsset: vi.fn(),
    getAssetSegmentations: vi.fn(),
  };
});

vi.mock("@/shared/api/jobs", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/jobs")>(
    "@/shared/api/jobs"
  );
  return {
    ...actual,
    getJobQueueStatus: vi.fn(),
  };
});

vi.mock("@/shared/api/segmentations/annotations", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/annotations")
  >(
    "@/shared/api/segmentations/annotations"
  );
  return {
    ...actual,
    getSegmentsAtPoint: vi.fn(),
    confirmSegmentsBatch: vi.fn(),
    listUserFeedback: vi.fn(),
    markSegmentationComplete: vi.fn(),
    unlockSegmentation: vi.fn(),
    querySegmentsInRegion: vi.fn(),
    updateSegmentLabelsBatch: vi.fn(),
  };
});

vi.mock("@/shared/api/segmentations/rois", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/segmentations/rois")>(
    "@/shared/api/segmentations/rois"
  );
  return {
    ...actual,
    activateSegmentationRoi: vi.fn(),
    getSegmentationRois: vi.fn(),
    createSegmentationRoi: vi.fn(),
    rerunSegmentationRoi: vi.fn(),
  };
});

vi.mock("@/shared/api/segmentations/completedRois", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/completedRois")
  >("@/shared/api/segmentations/completedRois");
  return {
    ...actual,
    getCompletedRois: vi.fn(),
    createCompletedRoi: vi.fn(),
  };
});

vi.mock("@/hooks/useSegmentationOverlayManifest", () => ({
  useSegmentationOverlayManifest: vi.fn((...args: unknown[]) => {
    overlayManifestHookSpy(...args);
    return {
      manifest: overlayManifestState.manifest,
      refetch: overlayManifestRefetchMock,
    };
  }),
}));

vi.mock("@/viewer/viewportSync/useViewportSyncGroup", () => ({
  useViewportSyncGroup: vi.fn(() => viewportSyncState),
}));

vi.mock("@/hooks/useThreshold", () => ({
  useThreshold: vi.fn(() => thresholdState),
}));

vi.mock("@/features/segmentation/hooks/useUncertainSegments", () => ({
  useUncertainSegments: vi.fn(() => segmentsState),
}));

vi.mock("@/features/segmentation/hooks/useSegmentationWorkflowMode", () => ({
  useSegmentationWorkflowMode: vi.fn(() => workflowModeState),
}));

vi.mock("@/hooks/useHoverSelection", () => ({
  useHoverSelection: vi.fn(() => hoverSelectionState),
}));

vi.mock("@/hooks/useDrawing", () => ({
  useDrawing: vi.fn(() => drawingState),
}));

let currentRois: SegmentationRoi[] = [];
let currentJobs: JobQueueStatus;

export function makeImage(overrides: Partial<AssetDetail> = {}): AssetDetail {
  return {
    id: "img-1",
    file_path: "",
    original_filename: "source.tif",
    display_name: "Image 1",
    is_eval_set: false,
    width: 1000,
    height: 800,
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

export function makeSegmentation(
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
    segment_counts: { CONFIRMED: 0, CANDIDATE: 0 },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    status_error: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    config: {
      supports_instance_params: false,
      instance_params: null,
    },
    ...overrides,
  };
}

export function makeErSegmentation(
  overrides: Partial<ImageSegmentation> = {}
): ImageSegmentation {
  return makeSegmentation({
    segmentation_type: {
      id: "type-er",
      internal_name: "quantem_internal_er",
      short_name: "ER",
      long_name: "Endoplasmic Reticulum",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    ...overrides,
  });
}

export function makeRoi(
  id: string,
  x: number,
  y: number,
  width: number,
  height: number,
  isActive: boolean
): SegmentationRoi {
  return {
    id,
    segmentation: "seg-1",
    x,
    y,
    width,
    height,
    source: "MANUAL",
    seed: null,
    is_active: isActive,
    is_complete: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

export function makeJobQueueStatus(
  running: JobQueueStatus["running"] = [],
  pending: JobQueueStatus["queues"][number]["pending"] = []
): JobQueueStatus {
  return {
    running,
    queues: [
      {
        queue_name: "cpu",
        display_name: "CPU",
        pending,
      },
    ],
    failed: [],
    completed: [],
    worker: {
      scheduler_in_process: true,
    },
    generated_at: "2026-01-01T00:00:00Z",
  };
}

export function makeSegment(overrides: Partial<SegmentObject> = {}): SegmentObject {
  return {
    id: "segment-1",
    segmentation: "seg-1",
    label_state: "CANDIDATE",
    confidence_score: 0.82,
    geometry_coords: [
      [10, 10],
      [20, 10],
      [20, 20],
      [10, 20],
      [10, 10],
    ],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

export function setupSegmentationScreenTest() {
  vi.clearAllMocks();
  vi.mocked(useHoverSelection).mockImplementation(() => hoverSelectionState);
  overlayManifestState.manifest = null;
  thresholdState.threshold = 0.99;
  viewportSyncState.viewport = null;
  useSelectionStore.getState().clearSelection();
  useSelectionStore.getState().setSelectedImageId("img-1");
  useSelectionStore.getState().setSelectedSegmentationId("seg-1");
  useUserFeedbackStore.setState({ feedbackBySegmentation: {} });

  currentRois = [
    makeRoi("roi-active", 0, 0, 100, 80, true),
    makeRoi("roi-existing", 300, 200, 100, 80, false),
  ];
  currentJobs = makeJobQueueStatus();

  vi.mocked(getAsset).mockResolvedValue(makeImage());
  vi.mocked(getAssetSegmentations).mockResolvedValue([makeSegmentation()]);
  vi.mocked(getSegmentsAtPoint).mockResolvedValue([]);
  vi.mocked(getSegmentationRois).mockImplementation(async () => currentRois);
  vi.mocked(getJobQueueStatus).mockImplementation(async () => currentJobs);
  vi.mocked(getCompletedRois).mockResolvedValue([]);
  vi.mocked(createCompletedRoi).mockImplementation(async (_segId, payload) => ({
    id: "completed-roi-1",
    segmentation: "seg-1",
    polygon_coords: payload.polygon_coords,
    bbox: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  }));
  vi.mocked(listUserFeedback).mockResolvedValue([]);
  vi.mocked(querySegmentsInRegion).mockResolvedValue({ segments: [] });
  vi.mocked(activateSegmentationRoi).mockImplementation(async (_segId, roiId) => {
    currentRois = currentRois.map((roi) => ({
      ...roi,
      is_active: roi.id === roiId,
    }));
    return currentRois.find((roi) => roi.id === roiId) as SegmentationRoi;
  });
  vi.mocked(createSegmentationRoi).mockImplementation(async (_segId, payload) => {
    const created: SegmentationRoi = {
      id: `roi-created-${currentRois.length + 1}`,
      segmentation: "seg-1",
      x: Number(payload.x),
      y: Number(payload.y),
      width: Number(payload.width),
      height: Number(payload.height),
      source: payload.source ?? "MANUAL",
      seed: payload.seed ?? null,
      is_active: true,
      is_complete: false,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    };
    currentRois = [created, ...currentRois.map((roi) => ({ ...roi, is_active: false }))];
    return created;
  });
  vi.mocked(rerunSegmentationRoi).mockImplementation(async (_segId, roiId) => {
    currentJobs = makeJobQueueStatus([], [
      {
        id: "job-roi-1",
        type: "run_segmentation_roi_task",
        task_label: "Run ROI",
        status: "PENDING",
        progress: 0,
        cancel_requested: false,
        queue_name: "cpu",
        resource_class: "cpu",
        created_at: "2026-01-01T00:00:00Z",
        image: { id: "img-1", display_name: "Image 1" },
        segmentation: {
          id: "seg-1",
          name: "Mitochondria",
          internal_name: "quantem_internal_mito",
          short_name: "Mito",
          long_name: "Mitochondria",
        },
      },
    ]);
    return { job_id: "job-roi-1", roi_id: roiId ?? "roi-active" };
  });
  vi.mocked(confirmSegmentsBatch).mockResolvedValue({
    created: 1,
    updated: 0,
    deleted: 0,
    confirmed_ids: ["segment-confirmed-1"],
    overlay: {
      desired_revision: 4,
      applied_revision: 3,
      sync_applied: false,
      rebuild_mode: "async_partial",
    },
  });
  vi.mocked(markSegmentationComplete).mockResolvedValue(makeSegmentation());
  vi.mocked(unlockSegmentation).mockResolvedValue(makeSegmentation());
  vi.mocked(updateSegmentLabelsBatch).mockResolvedValue({
    updated: 0,
    overlays: {},
  });

  workflowModeState.workflowMode = "review";
  workflowModeState.leftMode = "hover";
  hoverSelectionState.hoverActionMode = "test";
  hoverSelectionState.hoverSegments = [];
  hoverSelectionState.hoverIndex = 0;
  hoverSelectionState.highlightedSegmentId = null;
  hoverSelectionState.hoverPoint = null;
  drawingState.pendingPolygon = null;
  drawingState.brushStrokes = [];
  drawingState.brushSize = 24;
  drawingState.getBrushPolygonRings.mockReturnValue([]);
}
