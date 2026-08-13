import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useErPolygonWorkflow } from "@/features/segmentation/screen/hooks/useErPolygonWorkflow";
import { makeSegmentation } from "@/features/segmentation/SegmentationScreen.testUtils";
import type { ConfirmBatchResponse } from "@/shared/types/segmentation";
import type { Point } from "@/utils/geometry";

function makeArgs(
  overrides: Partial<Parameters<typeof useErPolygonWorkflow>[0]> = {}
): Parameters<typeof useErPolygonWorkflow>[0] {
  return {
    currentSegmentation: makeSegmentation(),
    active: true,
    isPointInsideImageBounds: () => true,
    registerAnnotationActivity: vi.fn(),
    showErrorToast: vi.fn(),
    showNoticeToast: vi.fn(),
    draftOperation: "include",
    submitConfirmedGeometriesOptimistically: vi.fn(async () => ({
      created: 1,
      updated: 0,
      deleted: 0,
      confirmed_ids: ["seg-obj-1"],
      outlines: null,
      measurement: null,
    })),
    ...overrides,
  };
}

async function traceAndClose(tool: {
  handlePolygonClick: (point: Point) => void;
  handlePolygonMouseMove: (point: Point) => void;
  handleClosePolygon: () => Promise<void>;
}) {
  act(() => tool.handlePolygonClick({ x: 10, y: 10 }));
  act(() => tool.handlePolygonMouseMove({ x: 20, y: 20 }));
  act(() => tool.handlePolygonClick({ x: 30, y: 10 }));
  await act(async () => {
    await tool.handleClosePolygon();
  });
}

/**
 * Closing the polygon clears the draft whatever the server said, so a ring it
 * stored nothing for leaves exactly the same screen as one it stored. This
 * hook used to `await` the submit and discard the result.
 */
describe("useErPolygonWorkflow: what the response says happened", () => {
  it("repeats the server's sentence when the ring was refused", async () => {
    const showNoticeToast = vi.fn();
    const { result } = renderHook(() =>
      useErPolygonWorkflow(
        makeArgs({
          showNoticeToast,
          submitConfirmedGeometriesOptimistically: vi.fn(
            async (): Promise<ConfirmBatchResponse> => ({
              created: 0,
              updated: 0,
              deleted: 0,
              confirmed_ids: [],
              outlines: {
                separated: [],
                dropped: [{ index: 0, areas: 1, kept: 0 }],
                detail:
                  "segments[0] was not stored: the outline spans 1 pixel or less in one dimension.",
              },
            })
          ),
        })
      )
    );

    await traceAndClose(result.current);

    expect(showNoticeToast).toHaveBeenCalledWith(
      expect.stringContaining("was not stored")
    );
  });

  it("says nothing changed when the merge stored nothing and gave no reason", async () => {
    const showNoticeToast = vi.fn();
    const { result } = renderHook(() =>
      useErPolygonWorkflow(
        makeArgs({
          showNoticeToast,
          submitConfirmedGeometriesOptimistically: vi.fn(
            async (): Promise<ConfirmBatchResponse> => ({
              created: 0,
              updated: 0,
              deleted: 0,
              confirmed_ids: [],
              outlines: null,
              measurement: null,
            })
          ),
        })
      )
    );

    await traceAndClose(result.current);

    expect(showNoticeToast).toHaveBeenCalledWith(
      expect.stringContaining("Nothing was stored")
    );
  });

  it("says nothing when the ring landed", async () => {
    const showNoticeToast = vi.fn();
    const showErrorToast = vi.fn();
    const { result } = renderHook(() =>
      useErPolygonWorkflow(makeArgs({ showNoticeToast, showErrorToast }))
    );

    await traceAndClose(result.current);

    expect(showNoticeToast).not.toHaveBeenCalled();
    expect(showErrorToast).not.toHaveBeenCalled();
  });
});
