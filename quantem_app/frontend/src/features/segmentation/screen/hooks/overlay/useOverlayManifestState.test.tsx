import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useOverlayManifestState } from "@/features/segmentation/screen/hooks/overlay/useOverlayManifestState";
import { getSegmentationOverlayManifest } from "@/shared/api/segmentations/overlays";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";

vi.mock("@/shared/api/segmentations/overlays", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/overlays")
  >("@/shared/api/segmentations/overlays");
  return { ...actual, getSegmentationOverlayManifest: vi.fn() };
});

const getManifestMock = vi.mocked(getSegmentationOverlayManifest);

function makeManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return {
    status: "READY",
    ngff_url: "/segmentation-overlays/seg-1.zarr",
    lut_url: "/api/segmentations/seg-1/overlay-lut/",
    arrays: ["labels", "border"],
    label_dtype: "uint32",
    bundle_version: 3,
    applied_revision: 5,
    desired_revision: 5,
    lut_revision: 5,
    chunk_size: [256, 256],
    level_count: 4,
    width: 1024,
    height: 1024,
    ...overrides,
  };
}

/** The F1 shape: failed at 6, revision 5 on disk, nothing will re-queue it. */
function failedManifest(
  overrides: Partial<SegmentationOverlayManifest> = {}
): SegmentationOverlayManifest {
  return makeManifest({
    status: "FAILED",
    applied_revision: 5,
    desired_revision: 6,
    last_error:
      "[WinError 183] Cannot create a file when that file already exists: " +
      "'D:\\data\\tmp\\segmentation_overlays\\seg-1\\staging'",
    ...overrides,
  });
}

function renderState() {
  return renderHook(() =>
    useOverlayManifestState({
      currentSegmentationId: "seg-1",
      activeSourceModel: "quantem:mito",
    })
  );
}

describe("useOverlayManifestState", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("stops calling a failed build an update in progress", async () => {
    getManifestMock.mockResolvedValue(failedManifest());
    const { result } = renderState();

    await waitFor(() => expect(result.current.overlayManifest).not.toBeNull());

    // The clause that used to carry the predicate on its own is still true...
    expect(result.current.overlayManifest?.desired_revision).toBeGreaterThan(
      result.current.overlayManifest?.applied_revision ?? 0
    );
    // ...and the labeling sidebar no longer says "Overlay updating." for it.
    expect(result.current.overlayUpdating).toBe(false);
    expect(result.current.overlayManifestNeedsPolling).toBe(false);
    expect(result.current.overlayBuildFailed).toBe(true);
  });

  it("hands the labeling screen the server's reason", async () => {
    getManifestMock.mockResolvedValue(failedManifest());
    const { result } = renderState();

    await waitFor(() => expect(result.current.overlayBuildFailed).toBe(true));
    expect(result.current.overlayBuildError).toContain("WinError 183");
    expect(result.current.overlayBuildError).toContain("staging");
    // A bundle is on disk, so the canvas is stale rather than empty.
    expect(result.current.usesRasterReviewOverlay).toBe(true);
  });

  it("says 'no reason recorded' apart from 'no failure'", async () => {
    getManifestMock.mockResolvedValue(failedManifest({ last_error: "" }));
    const { result } = renderState();

    await waitFor(() => expect(result.current.overlayBuildFailed).toBe(true));
    expect(result.current.overlayBuildError).toBeNull();
  });

  it("keeps polling a build that is genuinely running", async () => {
    getManifestMock.mockResolvedValue(
      makeManifest({ status: "BUILDING", applied_revision: 5, desired_revision: 6 })
    );
    const { result } = renderState();

    await waitFor(() => expect(result.current.overlayManifest).not.toBeNull());
    expect(result.current.overlayUpdating).toBe(true);
    expect(result.current.overlayBuildFailed).toBe(false);
    expect(result.current.overlayBuildError).toBeNull();
  });

  it("records the revision each labeling pane has actually loaded", async () => {
    getManifestMock.mockResolvedValue(makeManifest());
    const { result } = renderState();

    await waitFor(() => expect(result.current.overlayManifest).not.toBeNull());
    expect(result.current.leftDisplayedOverlayRevision).toBeNull();
    expect(result.current.rightDisplayedOverlayRevision).toBeNull();

    act(() => result.current.handleLeftOverlayRevisionDisplayed(5));
    act(() => result.current.handleRightOverlayRevisionDisplayed(4));

    expect(result.current.leftDisplayedOverlayRevision).toBe(5);
    expect(result.current.rightDisplayedOverlayRevision).toBe(4);
  });

  /**
   * The request itself belongs to `OverlayBuildFailureNotice`, which both the
   * viewer and the labeling sidebar mount, and is covered where it is rendered
   * (`SegmentationSidebar.test.tsx`, `OverlaySelectionSidebar.test.tsx`). What
   * only this hook can do is undo the thing it did when the build failed: it
   * stopped the poll, and a card that never notices the retry succeeded is the
   * same dead end in a new costume.
   */
  it("puts polling back after a retry is accepted", async () => {
    getManifestMock.mockResolvedValue(failedManifest());
    const { result } = renderState();

    await waitFor(() => expect(result.current.overlayBuildFailed).toBe(true));
    expect(result.current.overlayManifestNeedsPolling).toBe(false);
    const callsBefore = getManifestMock.mock.calls.length;

    getManifestMock.mockResolvedValue(
      makeManifest({ status: "BUILDING", applied_revision: 5, desired_revision: 6 })
    );
    await act(async () => {
      result.current.handleOverlayBuildRetried();
    });

    await waitFor(() => expect(result.current.overlayBuildFailed).toBe(false));
    expect(getManifestMock.mock.calls.length).toBeGreaterThan(callsBefore);
    expect(result.current.overlayManifestNeedsPolling).toBe(true);
  });
});
