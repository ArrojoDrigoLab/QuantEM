import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { SegmentationRightPanel } from "@/features/segmentation/components/SegmentationRightPanel";
import { getAssetNgffUrl } from "@/shared/api/assets";
import type { AssetDetail } from "@/shared/types/images";
import type { SegmentObject } from "@/shared/types/segmentation";

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

function makeSegment(id: string, label_state: SegmentObject["label_state"]): SegmentObject {
  return {
    id,
    segmentation: "seg-1",
    label_state,
    confidence_score: 0.9,
    geometry_coords: [
      [10, 10],
      [20, 10],
      [20, 20],
      [10, 20],
    ],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

function makeProps(
  overrides: Partial<ComponentProps<typeof SegmentationRightPanel>> = {}
): ComponentProps<typeof SegmentationRightPanel> {
  return {
    image: makeImage(),
    segmentationTypeInternalName: null,
    useSmoothedGeometry: false,
    viewport: null,
    onViewportChange: vi.fn(),
    confirmedSegments: [makeSegment("c1", "CONFIRMED")],
    tooManyRight: false,
    activeRoi: null,
    rois: [],
    removeMode: "none",
    onRemoveModeChange: vi.fn(),
    onRemoveObjectClick: vi.fn(),
    removeAreaBrushSize: 24,
    onRemoveAreaBrushSizeChange: vi.fn(),
    removeAreaBrushStrokes: [],
    onRemoveAreaBrushStroke: vi.fn(),
    canApplyRemoveArea: false,
    onApplyRemoveArea: vi.fn(),
    removingArea: false,
    layerControls: {
      usesRasterOverlay: false,
      confirmed: {
        strokeWidth: 2,
        fillOpacity: 0.2,
        showBorders: true,
        onStrokeWidthChange: vi.fn(),
        onFillOpacityChange: vi.fn(),
        onShowBordersChange: vi.fn(),
      },
    },
    ...overrides,
  };
}

describe("SegmentationRightPanel", () => {
  it("renders confirmed objects only and points the viewer at the image ngff", () => {
    viewerPropsSpy.mockClear();
    render(<SegmentationRightPanel {...makeProps()} />);

    expect(screen.queryByText("Confirmed Objects")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove objects" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove area" })).toBeInTheDocument();
    expect(getAssetNgffUrl).toHaveBeenCalledWith("img-1", null);
    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      overlays?: {
        persistent?: Array<{ id: string }>;
        transient?: Array<{ id: string }>;
      };
      interactions?: {
        onShapeClick?: unknown;
      };
    };
    expect(lastCall.overlays?.persistent?.map((overlay) => overlay.id)).toEqual(["c1"]);
    expect(lastCall.overlays?.transient).toEqual([]);
    expect(lastCall.interactions?.onShapeClick).toBeUndefined();
    expect(screen.getByLabelText("Right pane overlay options")).toBeInTheDocument();
  });

  it("applies the right pane confirmed fill opacity independently", () => {
    viewerPropsSpy.mockClear();
    render(
      <SegmentationRightPanel
        {...makeProps({
          layerControls: {
            usesRasterOverlay: false,
            confirmed: {
              strokeWidth: 3,
              fillOpacity: 0,
              showBorders: true,
              onStrokeWidthChange: vi.fn(),
              onFillOpacityChange: vi.fn(),
              onShowBordersChange: vi.fn(),
            },
          },
        })}
      />
    );

    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      overlays?: {
        persistent?: Array<{
          id: string;
          fillOpacity: number;
          strokeWidth: number;
        }>;
      };
    };
    expect(
      lastCall.overlays?.persistent?.find((overlay) => overlay.id === "c1")
    ).toMatchObject({ fillOpacity: 0, strokeWidth: 3 });
  });

  it("shows many-shapes warning for confirmed overlays", () => {
    render(<SegmentationRightPanel {...makeProps({ tooManyRight: true })} />);

    expect(
      screen.getByText("Many confirmed shapes in view; rendering may be slower.")
    ).toBeInTheDocument();
  });

  it("shows a nonblocking notice while confirmed objects are being saved", () => {
    render(
      <SegmentationRightPanel
        {...makeProps({ confirmingObjects: true })}
      />
    );

    expect(screen.getByText("Saving confirmed objects…")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveClass("confirming-objects-veil");
    expect(screen.getByRole("status").parentElement).toHaveClass("right-viewer-stage");
    expect(screen.getByTestId("image-viewer").closest("section")).toHaveAttribute(
      "aria-busy",
      "true"
    );
  });

  it("includes roi overlay after confirmed overlays", () => {
    viewerPropsSpy.mockClear();
    render(
      <SegmentationRightPanel
        {...makeProps({
          activeRoi: {
            id: "roi-1",
            segmentation: "seg-1",
            x: 0,
            y: 0,
            width: 100,
            height: 100,
            source: "AUTO",
            seed: null,
            is_active: true,
            is_complete: false,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
          rois: [
            {
              id: "roi-2",
              segmentation: "seg-1",
              x: 200,
              y: 200,
              width: 100,
              height: 100,
              source: "MANUAL",
              seed: null,
              is_active: false,
              is_complete: false,
              created_at: "2026-01-02T00:00:00Z",
              updated_at: "2026-01-02T00:00:00Z",
            },
          ],
        })}
      />
    );

    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      overlays?: {
        persistent?: Array<{
          id: string;
          strokeOpacity?: number;
          strokeDasharray?: string;
        }>;
      };
    };
    expect(lastCall.overlays?.persistent?.at(0)?.id).toBe("c1");
    expect(lastCall.overlays?.persistent?.at(-1)?.id).toBe("roi-frame");
    expect(
      lastCall.overlays?.persistent?.find((overlay) => overlay.id === "roi-frame-roi-2")
    ).toMatchObject({ strokeOpacity: 0.4, strokeDasharray: "8 6" });
  });

  it("toggles remove objects mode off when pressed while active", async () => {
    const user = userEvent.setup();
    const onRemoveModeChange = vi.fn();
    viewerPropsSpy.mockClear();

    render(
      <SegmentationRightPanel
        {...makeProps({
          removeMode: "objects",
          onRemoveModeChange,
        })}
      />
    );

    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      interactions?: {
        onShapeClick?: unknown;
        onShapeHover?: unknown;
      };
    };
    expect(lastCall.interactions?.onShapeClick).toBeTypeOf("function");
    expect(lastCall.interactions?.onShapeHover).toBeTypeOf("function");
    expect(screen.getByText(/Click once to permanently delete/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Remove objects" }));
    expect(onRemoveModeChange).toHaveBeenCalledWith("none");
  });

  it("passes the picked UUID straight to deletion with no keyboard step", () => {
    const onRemoveObjectClick = vi.fn();
    viewerPropsSpy.mockClear();
    render(
      <SegmentationRightPanel
        {...makeProps({
          removeMode: "objects",
          onRemoveObjectClick,
        })}
      />
    );

    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      interactions?: { onShapeClick?: (segmentId: string | null) => void };
    };
    act(() => {
      lastCall.interactions?.onShapeClick?.("confirmed-1");
    });

    expect(onRemoveObjectClick).toHaveBeenCalledWith("confirmed-1");
  });

  it("shows remove-area controls and brush mode when active", () => {
    viewerPropsSpy.mockClear();

    render(
      <SegmentationRightPanel
        {...makeProps({
          removeMode: "area",
          removeAreaBrushStrokes: [
            {
              id: "stroke-1",
              label: 1,
              size: 24,
              points: [
                { x: 10, y: 10 },
                { x: 20, y: 20 },
              ],
            },
          ],
          canApplyRemoveArea: true,
        })}
      />
    );

    expect(screen.getByRole("button", { name: "Remove area" })).toBeInTheDocument();
    expect(screen.getByRole("slider", { name: /Brush diameter/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove" })).toBeInTheDocument();

    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      interactions?: {
        brush?: {
          enabled?: boolean;
          onStroke?: unknown;
        };
        onImageClick?: unknown;
        onShapeClick?: unknown;
      };
    };
    expect(lastCall.interactions?.brush?.enabled).toBe(true);
    expect(lastCall.interactions?.brush?.onStroke).toBeTypeOf("function");
    expect(lastCall.interactions?.onImageClick).toBeUndefined();
    expect(lastCall.interactions?.onShapeClick).toBeUndefined();
  });

  it("highlights the independently hovered confirmed object", () => {
    viewerPropsSpy.mockClear();
    render(
      <SegmentationRightPanel
        {...makeProps({
          removeMode: "objects",
          confirmedSegments: [makeSegment("hovered", "CONFIRMED")],
          idMapOverlays: [
            {
              id: "confirmed-idmap",
              ngffUrl: "/overlay.zarr",
              lut: new Uint8Array(8),
              maxLabel: 1,
              lutRevision: 1,
              fillOpacity: 0.2,
              borderOpacity: 1,
              showBorders: true,
              pickMap: new Map([[1, "hovered"]]),
            },
          ],
        })}
      />
    );

    const initialCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      interactions?: { onShapeHover?: (segmentId: string | null) => void };
    };
    act(() => {
      initialCall.interactions?.onShapeHover?.("hovered");
    });
    const lastCall = viewerPropsSpy.mock.calls.at(-1)?.[0] as {
      overlays?: {
        persistent?: Array<{ id: string; strokeColor: string; strokeWidth?: number }>;
        idMapOverlays?: Array<{
          highlightedSegmentId?: string | null;
          highlightRevision?: number;
        }>;
      };
    };
    expect(lastCall.overlays?.persistent?.find((item) => item.id === "hovered"))
      .toMatchObject({ strokeColor: "#00ffff", strokeWidth: 4 });
    expect(lastCall.overlays?.idMapOverlays?.[0]).toMatchObject({
      highlightedSegmentId: "hovered",
      highlightRevision: 1,
    });
  });
});
