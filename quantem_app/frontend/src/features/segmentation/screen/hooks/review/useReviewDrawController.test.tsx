import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import {
  drawingState,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useReviewDrawController } from "@/features/segmentation/screen/hooks/review/useReviewDrawController";

describe("useReviewDrawController", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    drawingState.clearDrawing.mockClear();
  });

  it("confirms drawn geometry through the draw controller", async () => {
    const reviewDrawing = {
      ...drawingState,
      // Closed, because `useDrawing.handleDrawComplete` closes the freehand
      // path before it ever reaches this hook.
      pendingPolygon: [
        { x: 10, y: 10 },
        { x: 20, y: 10 },
        { x: 10, y: 20 },
        { x: 10, y: 10 },
      ],
    } as unknown as typeof drawingState & {
      pendingPolygon: Array<{ x: number; y: number }> | null;
    };
    const submitConfirmedGeometriesOptimistically = vi.fn(async () => null);

    const { result } = renderHook(() =>
      useReviewDrawController({
        currentSegmentation: {
          id: "seg-1",
        } as never,
        activeSourceModel: null,
        isErSegmentation: false,
        drawing: reviewDrawing as never,
        registerAnnotationActivity: vi.fn(),
        handleOverlayMutationRefresh: vi.fn(),
        showErrorToast: vi.fn(),
        showNoticeToast: vi.fn(),
        submitConfirmedGeometriesOptimistically,
      })
    );

    await act(async () => {
      await result.current.handleAcceptPolygon();
    });

    expect(submitConfirmedGeometriesOptimistically).toHaveBeenCalledWith({
      geometryRings: [
        [[
          [10, 10],
          [20, 10],
          [10, 20],
          [10, 10],
        ]],
      ],
      operations: ["include"],
      mergeOverlaps: false,
      manualCreation: true,
    });
    expect(drawingState.clearDrawing).toHaveBeenCalled();
  });

  it("merges drawn geometry into overlapping confirmed objects for ER", async () => {
    const reviewDrawing = {
      ...drawingState,
      pendingPolygon: [
        { x: 10, y: 10 },
        { x: 20, y: 10 },
        { x: 10, y: 20 },
        { x: 10, y: 10 },
      ],
    } as unknown as typeof drawingState & {
      pendingPolygon: Array<{ x: number; y: number }> | null;
    };
    const submitConfirmedGeometriesOptimistically = vi.fn(async () => null);

    const { result } = renderHook(() =>
      useReviewDrawController({
        currentSegmentation: { id: "seg-1" } as never,
        activeSourceModel: null,
        isErSegmentation: true,
        drawing: reviewDrawing as never,
        registerAnnotationActivity: vi.fn(),
        handleOverlayMutationRefresh: vi.fn(),
        showErrorToast: vi.fn(),
        showNoticeToast: vi.fn(),
        submitConfirmedGeometriesOptimistically,
      })
    );

    await act(async () => {
      await result.current.handleAcceptPolygon();
    });

    expect(submitConfirmedGeometriesOptimistically).toHaveBeenCalledWith(
      expect.objectContaining({ mergeOverlaps: true, manualCreation: true })
    );
  });

  /**
   * The ring the server measures is the ring the user drew.
   *
   * The backend rasteriser (`quantem.seg_core.rasterize`) was rewritten to a
   * pixel-centre convention precisely so a hand-drawn object and a model-found
   * object of the same shape measure the same, and the server-side simplify
   * was removed because it moved the outline before measurement. The client
   * kept doing it: `SIMPLIFY_POLYGONS` ran Douglas-Peucker at 1.0 px over every
   * ring on its way out.
   *
   * Squares hid it. Curves did not — measured through the server's own fill,
   * against the ring as drawn:
   *
   *   20 px lipid droplet   -7.8%   (brushed, the same droplet: +4.4%)
   *   60 px mitochondrion   -2.5%
   *   8 px object           -15.6%
   *
   * Opposite signs by tool, so it is not a bias anything downstream could undo.
   */
  describe("the outline is posted as drawn", () => {
    /** A freehand lasso round a 20 px droplet, at ~0.75 px pointer spacing. */
    function droplet(radius: number) {
      const count = Math.round((2 * Math.PI * radius) / 0.75);
      const points = Array.from({ length: count }, (_, index) => {
        const angle = (index / count) * 2 * Math.PI;
        return {
          x: 200 + radius * Math.cos(angle),
          y: 200 + radius * Math.sin(angle),
        };
      });
      return [...points, { x: points[0].x, y: points[0].y }];
    }

    /** Shoelace area of a closed ring. */
    function ringArea(ring: Array<[number, number]>) {
      let total = 0;
      for (let index = 0; index < ring.length - 1; index += 1) {
        total +=
          ring[index][0] * ring[index + 1][1] -
          ring[index + 1][0] * ring[index][1];
      }
      return Math.abs(total) / 2;
    }

    it.each([
      ["20 px lipid droplet", 10],
      ["60 px mitochondrion", 30],
    ])("posts every vertex of a %s", async (_label, radius) => {
      const drawn = droplet(radius);
      const reviewDrawing = {
        ...drawingState,
        pendingPolygon: drawn,
      } as unknown as typeof drawingState;
      const submit: Mock<
        (options: {
          geometries?: Array<Array<[number, number]>>;
          geometryRings?: Array<Array<Array<[number, number]>>>;
          operations?: Array<"include" | "exclude">;
        }) => Promise<null>
      > = vi.fn(async () => null);

      const { result } = renderHook(() =>
        useReviewDrawController({
          currentSegmentation: { id: "seg-1" } as never,
          activeSourceModel: null,
          isErSegmentation: false,
          drawing: reviewDrawing as never,
          registerAnnotationActivity: vi.fn(),
          handleOverlayMutationRefresh: vi.fn(),
          showErrorToast: vi.fn(),
          showNoticeToast: vi.fn(),
          submitConfirmedGeometriesOptimistically: submit,
        })
      );

      await act(async () => {
        await result.current.handleAcceptPolygon();
      });

      const posted = submit.mock.calls[0][0].geometryRings?.[0]?.[0] as Array<
        [number, number]
      >;
      // Vertex for vertex, not "close enough": any dropped vertex is area the
      // server will not measure.
      expect(posted).toEqual(drawn.map((point) => [point.x, point.y]));
      expect(ringArea(posted)).toBeCloseTo(
        ringArea(drawn.map((point) => [point.x, point.y])),
        10
      );
    });
  });

  /**
   * One stroke, several objects.
   *
   * A freehand path that crosses itself does not enclose one area, and
   * `confirm-batch` now stores every area it encloses instead of keeping the
   * largest lobe and silently dropping the rest. That is the right behaviour
   * and it is not what the gesture looks like, so the response says the outline
   * separated and this is the only place that repeats it to the user.
   */
  describe("what the response says happened", () => {
    function renderWithResponse(response: unknown) {
      const reviewDrawing = {
        ...drawingState,
        pendingPolygon: [
          { x: 10, y: 10 },
          { x: 20, y: 10 },
          { x: 10, y: 20 },
        ],
      } as unknown as typeof drawingState;
      const showNoticeToast = vi.fn();
      const showErrorToast = vi.fn();
      const { result } = renderHook(() =>
        useReviewDrawController({
          currentSegmentation: { id: "seg-1" } as never,
          activeSourceModel: null,
          isErSegmentation: false,
          drawing: reviewDrawing as never,
          registerAnnotationActivity: vi.fn(),
          handleOverlayMutationRefresh: vi.fn(),
          showErrorToast,
          showNoticeToast,
          submitConfirmedGeometriesOptimistically: vi.fn(
            async () => response as never
          ),
        })
      );
      return { result, showNoticeToast, showErrorToast };
    }

    it("reports an outline that separated into several objects", async () => {
      const { result, showNoticeToast, showErrorToast } = renderWithResponse({
        created: 2,
        updated: 0,
        deleted: 0,
        confirmed_ids: ["a", "b"],
        outlines: {
          separated: [{ index: 0, areas: 2, kept: 2 }],
          detail:
            "segments[0] crosses itself: it encloses 2 separate areas rather than one. All 2 were kept, each as its own object.",
        },
      });

      await act(async () => {
        await result.current.handleAcceptPolygon();
      });

      expect(showNoticeToast).toHaveBeenCalledWith(
        expect.stringContaining("encloses 2 separate areas")
      );
      // The drawing worked exactly as asked; reporting it as a failure would
      // say the opposite.
      expect(showErrorToast).not.toHaveBeenCalled();
    });

    it("reports objects whose outline was stored but not measured", async () => {
      // 207: the geometry is committed and cannot be taken back, but the
      // morphometric columns those objects reach objects.csv with are missing.
      const { result, showNoticeToast } = renderWithResponse({
        created: 1,
        updated: 0,
        deleted: 0,
        confirmed_ids: ["a"],
        measurement: {
          measured: 0,
          unmeasured_ids: ["a"],
          detail: "The image could not be opened, so these were not measured.",
        },
      });

      await act(async () => {
        await result.current.handleAcceptPolygon();
      });

      expect(showNoticeToast).toHaveBeenCalledWith(
        expect.stringContaining("were not measured")
      );
    });

    it("says nothing on the ordinary case", async () => {
      const { result, showNoticeToast } = renderWithResponse({
        created: 1,
        updated: 0,
        deleted: 0,
        confirmed_ids: ["a"],
        outlines: null,
        measurement: null,
      });

      await act(async () => {
        await result.current.handleAcceptPolygon();
      });

      expect(showNoticeToast).not.toHaveBeenCalled();
    });
  });
});
