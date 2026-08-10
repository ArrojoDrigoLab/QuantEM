import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import { removeSegmentationArea } from "@/shared/api/segmentations/annotations";
import { useTissueLabeling } from "@/features/segmentation/screen/hooks/useTissueLabeling";
import { makeSegmentation } from "@/features/segmentation/SegmentationScreen.testUtils";
import type { useDrawing } from "@/hooks/useDrawing";
import type { Point } from "@/utils/geometry";
import type { ConfirmBatchResponse } from "@/shared/types/segmentation";

type SubmitOptions = {
  geometries: Array<Array<[number, number]>>;
  samScores?: Array<number | null | undefined>;
  mergeOverlaps?: boolean;
  manualCreation?: boolean;
};

/** One object stored, measured, nothing to report. */
const STORED_ONE: ConfirmBatchResponse = {
  created: 1,
  updated: 0,
  deleted: 0,
  confirmed_ids: ["seg-obj-1"],
  overlay: null,
  outlines: null,
  measurement: null,
};

// Typed so `.mock.calls[0]` exposes the committed options payload.
const makeSubmitSpy = (
  response: ConfirmBatchResponse | null = STORED_ONE
): Mock<(options: SubmitOptions) => Promise<ConfirmBatchResponse | null>> =>
  vi.fn(async () => response);

vi.mock("@/shared/api/segmentations/annotations", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/annotations")
  >("@/shared/api/segmentations/annotations");
  return {
    ...actual,
    removeSegmentationArea: vi.fn(),
  };
});

/** A minimal stand-in for the screen-level drawing state. */
function makeDrawingStub(
  brushPolygons: Point[][] = []
): ReturnType<typeof useDrawing> {
  return {
    pendingPolygon: null,
    brushSize: 24,
    setBrushSize: vi.fn(),
    brushStrokes: brushPolygons.length
      ? [{ id: "s1", label: 1, size: 24, points: brushPolygons[0] }]
      : [],
    handleDrawComplete: vi.fn(),
    handleBrushStroke: vi.fn(),
    getBrushPolygons: vi.fn(() => brushPolygons),
    eraseBrushStrokesAt: vi.fn(),
    clearDrawing: vi.fn(),
  } as unknown as ReturnType<typeof useDrawing>;
}

function makeArgs(
  overrides: Partial<Parameters<typeof useTissueLabeling>[0]> = {}
): Parameters<typeof useTissueLabeling>[0] {
  return {
    currentSegmentation: makeSegmentation({
      segmentation_type: {
        id: "type-tissue",
        internal_name: "quantem_internal_tissue",
        short_name: "Tissue",
        long_name: "Tissue Mask",
        default_color: null,
        tags: [],
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    }),
    currentSegmentationId: "seg-1",
    enabled: true,
    isPointInsideImageBounds: () => true,
    registerAnnotationActivity: vi.fn(),
    showErrorToast: vi.fn(),
    showNoticeToast: vi.fn(),
    drawing: makeDrawingStub(),
    submitConfirmedGeometriesOptimistically: makeSubmitSpy(),
    refreshSegmentViews: vi.fn(async () => {}),
    setOverlayManifestPollingEnabled: vi.fn(),
    clearHoverInteraction: vi.fn(),
    ...overrides,
  };
}

/** Trace a closing triangle through a polygon-trace tool. */
async function traceAndClose(tool: {
  handlePolygonClick: (p: Point) => void;
  handlePolygonMouseMove: (p: Point) => void;
  handleClosePolygon: () => Promise<void>;
}) {
  act(() => {
    tool.handlePolygonClick({ x: 10, y: 10 });
  });
  act(() => {
    tool.handlePolygonMouseMove({ x: 20, y: 20 });
  });
  act(() => {
    tool.handlePolygonClick({ x: 30, y: 10 });
  });
  await act(async () => {
    await tool.handleClosePolygon();
  });
}

describe("useTissueLabeling", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("defaults to the brush tool", () => {
    const { result } = renderHook(() => useTissueLabeling(makeArgs()));
    expect(result.current.tool).toBe("brush");
    expect(result.current.activePolygonTool).toBeNull();
  });

  it("confirms brushed strokes as a merged manual region", async () => {
    const submit = makeSubmitSpy();
    const drawing = makeDrawingStub([
      [
        { x: 0, y: 0 },
        { x: 40, y: 0 },
        { x: 40, y: 40 },
        { x: 0, y: 40 },
      ],
    ]);
    const { result } = renderHook(() =>
      useTissueLabeling(
        makeArgs({ drawing, submitConfirmedGeometriesOptimistically: submit })
      )
    );

    expect(result.current.canConfirmBrush).toBe(true);

    await act(async () => {
      await result.current.handleConfirmBrush();
    });

    expect(submit).toHaveBeenCalledTimes(1);
    const [payload] = submit.mock.calls[0];
    expect(payload.mergeOverlaps).toBe(true);
    expect(payload.manualCreation).toBe(true);
    expect(payload.geometries).toHaveLength(1);
    expect(drawing.clearDrawing).toHaveBeenCalled();
  });

  it("adds a closed polygon into the mask via confirm-batch", async () => {
    const submit = makeSubmitSpy();
    const { result } = renderHook(() =>
      useTissueLabeling(
        makeArgs({ submitConfirmedGeometriesOptimistically: submit })
      )
    );

    act(() => {
      result.current.setTool("polygon");
    });
    await waitFor(() => expect(result.current.activePolygonTool).not.toBeNull());

    await traceAndClose(result.current.addPolygon);

    expect(submit).toHaveBeenCalledTimes(1);
    const [payload] = submit.mock.calls[0];
    expect(payload.mergeOverlaps).toBe(true);
    expect(payload.geometries).toHaveLength(1);
    const ring = payload.geometries[0];
    expect(ring.length).toBeGreaterThanOrEqual(4);
    expect(ring[0]).toEqual(ring[ring.length - 1]);
    expect(removeSegmentationArea).not.toHaveBeenCalled();
  });

  it("excludes a closed polygon from the mask via remove-area", async () => {
    vi.mocked(removeSegmentationArea).mockResolvedValue({
      created: 0,
      updated: 1,
      deleted: 0,
      created_ids: [],
      updated_ids: ["seg-obj-1"],
      deleted_ids: [],
      overlay: null,
    });
    const submit = makeSubmitSpy();
    const refreshSegmentViews = vi.fn(async () => {});
    const setOverlayManifestPollingEnabled = vi.fn();
    const { result } = renderHook(() =>
      useTissueLabeling(
        makeArgs({
          submitConfirmedGeometriesOptimistically: submit,
          refreshSegmentViews,
          setOverlayManifestPollingEnabled,
        })
      )
    );

    act(() => {
      result.current.setTool("exclude");
    });
    await waitFor(() => expect(result.current.activePolygonTool).not.toBeNull());

    await traceAndClose(result.current.excludePolygon);

    expect(removeSegmentationArea).toHaveBeenCalledTimes(1);
    const [segId, payload] = vi.mocked(removeSegmentationArea).mock.calls[0];
    expect(segId).toBe("seg-1");
    expect(payload.areas).toHaveLength(1);
    expect(payload.areas[0].geometry_coords.length).toBeGreaterThanOrEqual(4);
    // The cut must re-enable polling and trigger an immediate overlay refresh so
    // the asynchronously-rebuilt hole is picked up.
    expect(setOverlayManifestPollingEnabled).toHaveBeenCalledWith(true);
    expect(refreshSegmentViews).toHaveBeenCalled();
    // Excluding never adds to the mask.
    expect(submit).not.toHaveBeenCalled();
  });

  /**
   * Every tissue tool clears its draft the moment the request returns, so the
   * screen after a store that kept nothing is pixel-identical to the screen
   * after one that worked. All three used to discard the response body.
   */
  describe("what the response says happened", () => {
    it("repeats the server's sentence when the brushed area was refused", async () => {
      const showNoticeToast = vi.fn();
      const drawing = makeDrawingStub([
        [
          { x: 0, y: 0 },
          { x: 40, y: 0 },
          { x: 40, y: 0.4 },
          { x: 0, y: 0.4 },
        ],
      ]);
      const { result } = renderHook(() =>
        useTissueLabeling(
          makeArgs({
            drawing,
            showNoticeToast,
            submitConfirmedGeometriesOptimistically: makeSubmitSpy({
              created: 0,
              updated: 0,
              deleted: 0,
              confirmed_ids: [],
              outlines: {
                separated: [],
                dropped: [{ index: 0, areas: 1, kept: 0 }],
                detail:
                  "segments[0] was not stored: the outline spans 1 pixel or less in one dimension, so there is no area to measure.",
              },
            }),
          })
        )
      );

      await act(async () => {
        await result.current.handleConfirmBrush();
      });

      expect(showNoticeToast).toHaveBeenCalledWith(
        expect.stringContaining("was not stored")
      );
    });

    it("says so when a merge batch stored nothing and gave no reason", async () => {
      // The merge path makes no per-outline claim (a thin lobe can survive
      // inside the object it fuses with), so 0/0/0 is the only signal there is.
      const showNoticeToast = vi.fn();
      const { result } = renderHook(() =>
        useTissueLabeling(
          makeArgs({
            showNoticeToast,
            submitConfirmedGeometriesOptimistically: makeSubmitSpy({
              created: 0,
              updated: 0,
              deleted: 0,
              confirmed_ids: [],
              outlines: null,
              measurement: null,
            }),
          })
        )
      );

      act(() => {
        result.current.setTool("polygon");
      });
      await waitFor(() => expect(result.current.activePolygonTool).not.toBeNull());
      await traceAndClose(result.current.addPolygon);

      expect(showNoticeToast).toHaveBeenCalledWith(
        expect.stringContaining("Nothing was added to the tissue mask")
      );
    });

    it("says nothing when the polygon landed", async () => {
      const showNoticeToast = vi.fn();
      const { result } = renderHook(() =>
        useTissueLabeling(makeArgs({ showNoticeToast }))
      );

      act(() => {
        result.current.setTool("polygon");
      });
      await waitFor(() => expect(result.current.activePolygonTool).not.toBeNull());
      await traceAndClose(result.current.addPolygon);

      expect(showNoticeToast).not.toHaveBeenCalled();
    });

    it("reports an exclude ring that cut nothing", async () => {
      vi.mocked(removeSegmentationArea).mockResolvedValue({
        created: 0,
        updated: 0,
        deleted: 0,
        created_ids: [],
        updated_ids: [],
        deleted_ids: [],
        overlay: null,
      });
      const showNoticeToast = vi.fn();
      const { result } = renderHook(() =>
        useTissueLabeling(makeArgs({ showNoticeToast }))
      );

      act(() => {
        result.current.setTool("exclude");
      });
      await waitFor(() => expect(result.current.activePolygonTool).not.toBeNull());
      await traceAndClose(result.current.excludePolygon);

      expect(showNoticeToast).toHaveBeenCalledWith(
        expect.stringContaining("Nothing was excluded")
      );
    });

    it("reports a cut whose objects could not be re-measured", async () => {
      // 207. The hole is committed; the stored area still describes the shape
      // before the cut, and that is the number that reaches objects.csv.
      vi.mocked(removeSegmentationArea).mockResolvedValue({
        created: 0,
        updated: 1,
        deleted: 0,
        created_ids: [],
        updated_ids: ["seg-obj-1"],
        deleted_ids: [],
        overlay: null,
        measurement: {
          measured: 0,
          unmeasured_ids: ["seg-obj-1"],
          detail: "The image could not be opened, so these were not measured.",
        },
      });
      const showNoticeToast = vi.fn();
      const { result } = renderHook(() =>
        useTissueLabeling(makeArgs({ showNoticeToast }))
      );

      act(() => {
        result.current.setTool("exclude");
      });
      await waitFor(() => expect(result.current.activePolygonTool).not.toBeNull());
      await traceAndClose(result.current.excludePolygon);

      expect(showNoticeToast).toHaveBeenCalledWith(
        expect.stringContaining("were not measured")
      );
    });
  });
});
