import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeSegment,
  overlayManifestHookSpy,
  overlayManifestRefetchMock,
  overlayManifestState,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useSegmentationOverlayState } from "@/features/segmentation/screen/hooks/useSegmentationOverlayState";
import { OVERLAY_REFRESH_IDLE_DELAY_MS } from "@/features/segmentation/screen/utils/constants";
import type { SegmentationOverlayManifest } from "@/shared/types";

function makeOverlayManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return {
    status: "READY",
    ngff_url: "/ngff/overlay.zarr",
    lut_url: "/api/segmentations/seg-1/overlay-lut/",
    arrays: ["labels", "border"],
    label_dtype: "uint32",
    bundle_version: 1,
    applied_revision: 3,
    desired_revision: 3,
    lut_revision: 2,
    chunk_size: [256, 256],
    level_count: 4,
    width: 1000,
    height: 1000,
    ...overrides,
  };
}

function makeArgs(
  overrides: Partial<Parameters<typeof useSegmentationOverlayState>[0]> = {}
): Parameters<typeof useSegmentationOverlayState>[0] {
  return {
    currentSegmentationId: "seg-1",
    activeSourceModel: null,
    segmentationInternalName: "quantem_internal_mito",
    refetchSegmentations: vi.fn().mockResolvedValue(undefined),
    refetchLeftSegments: vi.fn().mockResolvedValue(undefined),
    useSmoothedSegmentGeometry: false,
    ...overrides,
  };
}

describe("useSegmentationOverlayState", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    overlayManifestState.manifest = makeOverlayManifest();
    overlayManifestRefetchMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("enables manifest polling", async () => {
    renderHook(() => useSegmentationOverlayState(makeArgs()));

    await waitFor(() => {
      expect(overlayManifestHookSpy).toHaveBeenLastCalledWith("seg-1", true, true, null);
    });
  });

  it("defers overlay manifest refresh after annotation activity and cleans up on unmount", () => {
    vi.useFakeTimers();
    overlayManifestState.manifest = makeOverlayManifest({
      status: "BUILDING",
      desired_revision: 4,
    });
    const refetchSegmentations = vi.fn().mockResolvedValue(undefined);
    const refetchLeftSegments = vi.fn().mockResolvedValue(undefined);

    const { result, unmount } = renderHook(() =>
      useSegmentationOverlayState(
        makeArgs({ refetchSegmentations, refetchLeftSegments })
      )
    );

    act(() => {
      result.current.refresh.registerAnnotationActivity();
    });

    expect(overlayManifestHookSpy).toHaveBeenLastCalledWith("seg-1", true, false, null);
    act(() => {
      vi.advanceTimersByTime(OVERLAY_REFRESH_IDLE_DELAY_MS);
    });
    expect(overlayManifestRefetchMock).toHaveBeenCalledTimes(1);

    act(() => {
      result.current.refresh.registerAnnotationActivity();
    });
    unmount();
    act(() => {
      vi.advanceTimersByTime(OVERLAY_REFRESH_IDLE_DELAY_MS);
    });

    expect(overlayManifestRefetchMock).toHaveBeenCalledTimes(1);
    expect(refetchSegmentations).not.toHaveBeenCalled();
    expect(refetchLeftSegments).not.toHaveBeenCalled();
  });

  it("waits the full idle delay before refreshing a dirty overlay", () => {
    vi.useFakeTimers();
    overlayManifestState.manifest = makeOverlayManifest({
      status: "DIRTY",
      desired_revision: 5,
    });

    const { result } = renderHook(() => useSegmentationOverlayState(makeArgs()));

    act(() => {
      result.current.refresh.registerAnnotationActivity();
      vi.advanceTimersByTime(OVERLAY_REFRESH_IDLE_DELAY_MS - 1);
    });
    expect(overlayManifestRefetchMock).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(overlayManifestRefetchMock).toHaveBeenCalledTimes(1);
  });

  it("applies, rolls back, and settles optimistic label overrides", async () => {
    overlayManifestState.manifest = makeOverlayManifest({
      applied_revision: 4,
      desired_revision: 4,
    });
    const sourceSegment = makeSegment({
      id: "segment-1",
      label_state: "CANDIDATE",
    });

    const { result, rerender } = renderHook(() =>
      useSegmentationOverlayState(makeArgs())
    );

    act(() => {
      result.current.optimistic.stageOptimisticSegments([sourceSegment], 5);
      result.current.optimistic.applyOptimisticLabel(
        "segment-1",
        "CONFIRMED",
        sourceSegment
      );
    });

    await waitFor(() => {
      expect(
        result.current.optimistic.applyLabelOverrides([sourceSegment])[0]?.label_state
      ).toBe("CONFIRMED");
      expect(result.current.optimistic.optimisticConfirmed).toHaveLength(1);
    });

    act(() => {
      result.current.optimistic.rollbackOptimisticLabel("segment-1");
    });
    await waitFor(() => {
      expect(
        result.current.optimistic.applyLabelOverrides([sourceSegment])[0]?.label_state
      ).toBe("CANDIDATE");
    });

    act(() => {
      result.current.optimistic.stageOptimisticSegments([sourceSegment], 5);
      result.current.optimistic.stageOptimisticRevisionTargets(["segment-1"], 5);
      result.current.optimistic.applyOptimisticLabel(
        "segment-1",
        "EXCLUDED",
        sourceSegment
      );
    });
    await waitFor(() => {
      expect(result.current.optimistic.optimisticExcluded).toHaveLength(1);
    });

    overlayManifestState.manifest = makeOverlayManifest({
      applied_revision: 7,
      desired_revision: 7,
    });
    rerender();

    // A bundle being ready on disk is not the handoff point: retain the vector
    // until the canvas confirms that it loaded the replacement revision.
    expect(result.current.optimistic.optimisticExcluded).toHaveLength(1);
    act(() => {
      result.current.manifest.handleLeftOverlayRevisionDisplayed(7);
    });

    await waitFor(() => {
      expect(result.current.optimistic.optimisticExcluded).toHaveLength(0);
    });
  });

  // Detailed candidate/confirmed/right ID-map spec assertions live with
  // useOverlayLayerControls. Here we verify this composition hook preserves the
  // controls it exposes.
  it("retains overlay border toggle state", () => {
    const { result } = renderHook(() => useSegmentationOverlayState(makeArgs()));

    act(() => {
      result.current.layers.setShowCandidateBorders(false);
    });

    expect(result.current.manifest.usesRasterReviewOverlay).toBe(true);
    expect(result.current.layers.showCandidateBorders).toBe(false);
    expect(result.current.layers.showConfirmedBorders).toBe(true);
  });

  it("removes a hard-deleted object from vector data immediately and can roll it back", () => {
    const segment = makeSegment({ id: "delete-me", label_state: "CONFIRMED" });
    const { result } = renderHook(() => useSegmentationOverlayState(makeArgs()));

    act(() => {
      result.current.optimistic.stageOptimisticSegments([segment]);
      result.current.deletion.hideSegment(segment.id);
    });

    expect(result.current.optimistic.applyLabelOverrides([segment])).toEqual([]);
    expect(result.current.optimistic.optimisticConfirmed).toEqual([]);

    act(() => {
      result.current.deletion.rollbackSegment(segment.id);
    });
    expect(result.current.optimistic.applyLabelOverrides([segment])).toEqual([segment]);
  });
});
