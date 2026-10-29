import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useOptimisticOverlayState } from "./useOptimisticOverlayState";
import type { SegmentObject } from "@/shared/types/segmentation";

const styles = { candidateStrokeWidth: 1, candidateFillOpacity: 0.3, confirmedStrokeWidth: 1, confirmedFillOpacity: 0.3 };
const hidden = new Set<string>();

describe("optimistic display retirement", () => {
  it("waits for both the confirmed geometry and palette revisions", () => {
    const { result, rerender } = renderHook(({ geometry, palette }) => useOptimisticOverlayState({
      currentSegmentationId: "seg", segmentationInternalName: "mito",
      useSmoothedSegmentGeometry: false, leftPanelLayerStyles: styles, hiddenSegmentIds: hidden,
      settledOverlayRevision: geometry, settledOverlayLutRevision: palette,
    }), { initialProps: { geometry: 5, palette: 10 } });
    const segment = { id: "object", label_state: "CONFIRMED", geometry_coords: [[0, 0], [10, 0], [10, 10], [0, 0]] } as SegmentObject;
    act(() => result.current.stageOptimisticSegments([segment], { revision: 6, lutRevision: 12 }));
    rerender({ geometry: 100, palette: 10 });
    expect(result.current.optimisticConfirmed).toHaveLength(1);
    rerender({ geometry: 5, palette: 12 });
    expect(result.current.optimisticConfirmed).toHaveLength(1);
    rerender({ geometry: 6, palette: 12 });
    expect(result.current.optimisticConfirmed).toHaveLength(0);
  });
});
