import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getSegmentationOverlayManifest } from "@/shared/api/segmentations/overlays";
import { useSegmentationOverlayManifest } from "@/hooks/useSegmentationOverlayManifest";
import {
  overlayBuildFailed,
  overlayBuildFailureReason,
  overlayIsUpdating,
} from "@/hooks/overlayManifestStatus";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

vi.mock("@/shared/api/segmentations/overlays", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/overlays")
  >(
    "@/shared/api/segmentations/overlays"
  );
  return {
    ...actual,
    getSegmentationOverlayManifest: vi.fn(),
  };
});

const getSegmentationOverlayManifestMock = vi.mocked(getSegmentationOverlayManifest);

function makeManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return {
    status: "READY",
    ngff_url: "/segmentation-overlays/seg-1.zarr",
    lut_url: "/api/segmentations/seg-1/overlay-lut/",
    arrays: ["labels", "border"],
    label_dtype: "uint32",
    bundle_version: 1,
    applied_revision: 1,
    desired_revision: 1,
    lut_revision: 1,
    chunk_size: [256, 256],
    level_count: 1,
    width: 1024,
    height: 1024,
    ...overrides,
  };
}

/**
 * A build that failed at revision 6 while revision 5 is on disk.
 *
 * `desired_revision > applied_revision` is *permanently* true in this shape:
 * `ensure_overlay_manifest` refuses to re-queue once a failure with a reason
 * is recorded, so nothing will ever move `applied_revision` up again.
 */
function failedManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return makeManifest({
    status: "FAILED",
    applied_revision: 5,
    desired_revision: 6,
    last_error:
      "[WinError 183] Cannot create a file when that file already exists: " +
      "'D:\\\\data\\\\tmp\\\\segmentation_overlays\\\\seg-1\\\\staging'",
    ...overrides,
  });
}

describe("overlay manifest predicates", () => {
  it("does not call a FAILED build an update in progress", () => {
    // The whole of finding F1 in one assertion: the third clause
    // (desired > applied) is true here and used to carry the predicate on its
    // own, which is why "Overlay updating..." never came down.
    const manifest = failedManifest();
    expect(manifest.desired_revision).toBeGreaterThan(manifest.applied_revision);
    expect(overlayIsUpdating(manifest)).toBe(false);
    expect(overlayBuildFailed(manifest)).toBe(true);
  });

  it("still calls a genuine build an update in progress", () => {
    expect(
      overlayIsUpdating(
        makeManifest({ status: "BUILDING", applied_revision: 1, desired_revision: 2 })
      )
    ).toBe(true);
    expect(
      overlayIsUpdating(
        makeManifest({ status: "DIRTY", applied_revision: 1, desired_revision: 1 })
      )
    ).toBe(true);
    expect(
      overlayIsUpdating(
        makeManifest({ status: "MISSING", applied_revision: 0, desired_revision: 1 })
      )
    ).toBe(true);
    expect(overlayIsUpdating(makeManifest())).toBe(false);
    expect(overlayIsUpdating(undefined)).toBe(false);
  });

  it("reports the server's reason, and distinguishes 'no reason recorded'", () => {
    expect(overlayBuildFailureReason(failedManifest())).toContain("WinError 183");
    expect(overlayBuildFailureReason(failedManifest({ last_error: "   " }))).toBeNull();
    expect(overlayBuildFailureReason(failedManifest({ last_error: undefined }))).toBeNull();
    // A stale string on a healthy manifest is not a failure to report.
    expect(
      overlayBuildFailureReason(makeManifest({ last_error: "old news" }))
    ).toBeNull();
  });
});

describe("useSegmentationOverlayManifest", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("stops polling once the build has FAILED", async () => {
    const pollIntervals: number[] = [];
    const nativeSetInterval = window.setInterval.bind(window);
    const setIntervalSpy = vi.spyOn(window, "setInterval").mockImplementation(
      ((handler: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (typeof handler === "function" && delay === 1500) {
          pollIntervals.push(delay);
        }
        return nativeSetInterval(handler, delay, ...args);
      }) as typeof window.setInterval
    );
    getSegmentationOverlayManifestMock.mockResolvedValue(failedManifest());

    try {
      const { result } = renderHook(() =>
        useSegmentationOverlayManifest("seg-1", true, true)
      );

      await waitFor(() => expect(result.current.loading).toBe(false));
      // Give the polling effect every chance to register a timer.
      await act(async () => {
        await Promise.resolve();
      });

      expect(pollIntervals).toHaveLength(0);
      expect(getSegmentationOverlayManifestMock).toHaveBeenCalledTimes(1);
    } finally {
      setIntervalSpy.mockRestore();
    }
  });

  it("skips polling when polling is disabled", async () => {
    const pollIntervals: number[] = [];
    const nativeSetInterval = window.setInterval.bind(window);
    const setIntervalSpy = vi.spyOn(window, "setInterval").mockImplementation(
      ((handler: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (typeof handler === "function" && delay === 1500) {
          pollIntervals.push(delay);
        }
        return nativeSetInterval(
          handler,
          delay,
          ...args
        );
      }) as typeof window.setInterval
    );
    getSegmentationOverlayManifestMock.mockResolvedValue(
      makeManifest({
        status: "DIRTY",
        applied_revision: 1,
        desired_revision: 2,
      })
    );

    try {
      const { result } = renderHook(() =>
        useSegmentationOverlayManifest("seg-1", true, false)
      );

      await waitFor(() => expect(result.current.loading).toBe(false));

      expect(getSegmentationOverlayManifestMock).toHaveBeenCalledTimes(1);
      expect(pollIntervals).toHaveLength(0);
    } finally {
      setIntervalSpy.mockRestore();
    }
  });

  it("continues polling when polling is enabled", async () => {
    type IntervalEntry = { id: number; delay: number; callback: () => void };
    const intervals: IntervalEntry[] = [];
    let nextIntervalId = 1;
    const nativeSetInterval = window.setInterval.bind(window);
    const nativeClearInterval = window.clearInterval.bind(window);
    const setIntervalSpy = vi.spyOn(window, "setInterval").mockImplementation(
      ((handler: TimerHandler, delay?: number, ...args: unknown[]) => {
        if (typeof handler === "function" && delay === 1500) {
          const id = nextIntervalId++;
          intervals.push({ id, delay, callback: handler as () => void });
          return id as unknown as ReturnType<typeof window.setInterval>;
        }
        return nativeSetInterval(
          handler,
          delay,
          ...args
        );
      }) as unknown as typeof window.setInterval
    );
    const clearIntervalSpy = vi.spyOn(window, "clearInterval").mockImplementation(
      ((intervalId?: number) => {
        if (typeof intervalId === "number" && intervals.some((entry) => entry.id === intervalId)) {
          return undefined;
        }
        return nativeClearInterval(intervalId);
      }) as typeof window.clearInterval
    );
    getSegmentationOverlayManifestMock.mockResolvedValue(
      makeManifest({
        status: "BUILDING",
        applied_revision: 1,
        desired_revision: 2,
      })
    );

    try {
      const { result, unmount } = renderHook(() =>
        useSegmentationOverlayManifest("seg-1", true, true)
      );

      await waitFor(() => expect(result.current.loading).toBe(false));
      await waitFor(() => expect(setIntervalSpy).toHaveBeenCalled());

      const pollInterval = intervals.find((entry) => entry.delay === 1500);
      expect(pollInterval).toBeDefined();

      await act(async () => {
        pollInterval?.callback();
      });

      await waitFor(() =>
        expect(getSegmentationOverlayManifestMock).toHaveBeenCalledTimes(2)
      );

      unmount();
      expect(clearIntervalSpy).toHaveBeenCalledWith(pollInterval?.id);
    } finally {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
    }
  });
});
