import { render } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { SegmentationLeftPanel } from "@/features/segmentation/components/SegmentationLeftPanel";
import { getAssetNgffUrl } from "@/shared/api/assets";
import type { AssetDetail } from "@/shared/types/images";

const { viewerPropsSpy } = vi.hoisted(() => ({
  viewerPropsSpy: vi.fn(),
}));

vi.mock("@/viewer/components/ImageViewer", () => ({
  ImageViewer: (props: unknown) => {
    viewerPropsSpy(props);
    return <div data-testid="image-viewer" />;
  },
}));

vi.mock("@/shared/api/assets", async () => {
  const actual = await vi.importActual<typeof import("@/shared/api/assets")>(
    "@/shared/api/assets"
  );
  return {
    ...actual,
    getAssetNgffUrl: vi.fn((assetId: string) => `ngff:${assetId}`),
  };
});

function makeImage(): AssetDetail {
  return {
    id: "img-1",
    file_path: "",
    original_filename: "source.tif",
    display_name: "Image 1",
    is_eval_set: false,
    width: 1000,
    height: 1000,
    channels: 1,
    bit_depth: 8,
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    tags: [],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ngff_ready: true,
    ngff_url: "/ngff/img-1.zarr",
  };
}

function makeProps(
  overrides: Partial<ComponentProps<typeof SegmentationLeftPanel>> = {}
): ComponentProps<typeof SegmentationLeftPanel> {
  return {
    viewer: {
      image: makeImage(),
      segmentationTypeInternalName: "quantem_internal_mito",
      useSmoothedGeometry: false,
      layerStyles: {
        candidateStrokeWidth: 2,
        candidateFillOpacity: 0.18,
        confirmedStrokeWidth: 2,
        confirmedFillOpacity: 0.15,
      },
      viewport: null,
      onViewportChange: vi.fn(),
    },
    workflow: {
      mode: "review",
      leftMode: "hover",
      reviewPhase: "model",
      correctionTool: "draw",
      navigateMode: false,
      groupConfirmActive: false,
      targetCursorActive: true,
      roiPlacementActive: false,
    },
    segments: {
      items: [],
      highlightedSegmentId: null,
      hoverPoint: null,
      hoverCount: 0,
      tooMany: false,
      onClick: vi.fn(),
      onMouseMove: vi.fn(),
      onMouseLeave: vi.fn(),
      groupSelectionBBox: null,
      groupHighlightedSegmentIds: [],
    },
    roi: {
      activeRoi: null,
      rois: [],
      completedRois: [],
      roiPoints: [],
      roiPointsSubmitted: 0,
      roiComplete: false,
      roiLabelMode: "positive",
      brushSize: 24,
      brushColor: "#33cc66",
      roiStrokes: [],
      onRoiLabelModeChange: vi.fn(),
      onBrushSizeChange: vi.fn(),
      onBrushStroke: vi.fn(),
      onSubmitRoiLabels: vi.fn(),
      onClearRoiLabels: vi.fn(),
      onReselectRoi: vi.fn(),
      onMarkRoiComplete: vi.fn(),
    },
    drawing: {
      pendingPolygon: null,
      brushStrokes: [],
      brushSize: 24,
      onDrawComplete: vi.fn(),
      onBrushStroke: vi.fn(),
      onAddStroke: vi.fn(),
      onEraseStroke: vi.fn(),
      onAccept: vi.fn(),
      onCancel: vi.fn(),
    },
    uncertain: {
      limit: 50,
      onLimitChange: vi.fn(),
      onRefresh: vi.fn(),
    },
    completedRoi: {
      active: false,
      loading: false,
      mode: "include",
      items: [],
      polygons: [],
      liveSectionPoints: [],
      hasDraft: false,
      canClosePolygon: false,
      canSave: false,
      isSaving: false,
      onModeChange: vi.fn(),
      onClosePolygon: vi.fn(),
      onRequestSave: vi.fn(),
      onClear: vi.fn(),
    },
    feedback: {
      items: [],
    },
    overlays: {},
    ...overrides,
  };
}

describe("SegmentationLeftPanel", () => {
  it("passes target cursor mode through to the image viewer", () => {
    viewerPropsSpy.mockClear();
    render(<SegmentationLeftPanel {...makeProps()} />);

    expect(getAssetNgffUrl).toHaveBeenCalledWith("img-1", null);
    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      highlighting?: {
        cursorMode?: string;
        hoverCursor?: boolean;
      };
    };
    expect(lastCall.highlighting?.cursorMode).toBe("target");
    expect(lastCall.highlighting?.hoverCursor).toBe(false);
  });
});
