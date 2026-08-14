import { describe, expect, it } from "vitest";
import { buildLeftPanelViewerConfig } from "@/features/segmentation/components/leftPanel/buildLeftPanelViewerConfig";
import { makeLeftPanelProps } from "@/features/segmentation/components/leftPanel/leftPanelTestUtils";

describe("buildLeftPanelViewerConfig", () => {
  it("preserves transient fit bounds", () => {
    const props = makeLeftPanelProps({
      workflow: {
        mode: "review",
        leftMode: "draw",
        reviewPhase: "correction",
        correctionTool: "draw",
        navigateMode: false,
        groupConfirmActive: false,
        targetCursorActive: false,
        roiPlacementActive: false,
      },
      viewer: {
        ...makeLeftPanelProps().viewer,
        transientFitBounds: {
          x: 10,
          y: 20,
          width: 30,
          height: 40,
        },
        transientFitBoundsKey: "focus-1",
      },
    });

    const viewerProps = buildLeftPanelViewerConfig({
      ...props,
      overlayScene: { persistent: [], transient: [] },
    });

    expect(viewerProps.viewport?.disablePan).toBe(false);
    expect(viewerProps.viewport?.fitBounds).toEqual({
      x: 10,
      y: 20,
      width: 30,
      height: 40,
    });
    expect(viewerProps.viewport?.fitBoundsKey).toBe("focus-1");
    expect(viewerProps.viewport?.fitBoundsPaddingRatio).toBe(1);
  });

  it("does not fit an active ROI until Open is requested", () => {
    const props = makeLeftPanelProps({
      roi: {
        ...makeLeftPanelProps().roi,
        activeRoi: {
          id: "roi-1",
          segmentation: "seg-1",
          x: 100,
          y: 200,
          width: 512,
          height: 512,
          source: "MANUAL",
          is_active: true,
          is_complete: false,
          completed_for_segmentation: false,
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
        },
      },
    });

    const viewerProps = buildLeftPanelViewerConfig({
      ...props,
      overlayScene: { persistent: [], transient: [] },
    });

    expect(viewerProps.viewport?.fitBounds).toBeNull();
    expect(viewerProps.viewport?.fitBoundsKey).toBeNull();
    expect(viewerProps.viewport?.fitBoundsPaddingRatio).toBeUndefined();
  });

  it("uses group mode and target cursor state in highlighting", () => {
    const props = makeLeftPanelProps({
      workflow: {
        ...makeLeftPanelProps().workflow,
        groupConfirmActive: true,
        targetCursorActive: true,
      },
      segments: {
        ...makeLeftPanelProps().segments,
        highlightedSegmentId: "segment-1",
        hoverPoint: { x: 4, y: 5 },
        hoverCount: 2,
      },
    });

    const viewerProps = buildLeftPanelViewerConfig({
      ...props,
      overlayScene: { persistent: [], transient: [] },
    });

    expect(viewerProps.viewport?.disablePan).toBe(true);
    expect(viewerProps.highlighting?.highlightedSegmentId).toBeNull();
    expect(viewerProps.highlighting?.cursorMode).toBe("target");
    expect(viewerProps.highlighting?.hoverCursor).toBe(false);
    expect(viewerProps.highlighting?.hoverBadge).toEqual({
      point: { x: 4, y: 5 },
      count: 2,
    });
  });

  it("keeps Navigate exclusive while ROI placement remains armed", () => {
    const props = makeLeftPanelProps({
      workflow: {
        ...makeLeftPanelProps().workflow,
        mode: "review",
        reviewPhase: "correction",
        correctionTool: "draw",
        leftMode: "draw",
        navigateMode: true,
        roiPlacementActive: true,
      },
    });

    const viewerProps = buildLeftPanelViewerConfig({
      ...props,
      overlayScene: { persistent: [], transient: [] },
    });

    expect(viewerProps.interactions?.mode).toBe("navigate");
    expect(viewerProps.interactions?.onImageClick).toBeUndefined();
    expect(viewerProps.interactions?.onImagePress).toBeUndefined();
    expect(viewerProps.interactions?.onImageDrag).toBeUndefined();
    expect(viewerProps.interactions?.onImageRelease).toBeUndefined();
    expect(viewerProps.interactions?.brush?.enabled).toBe(false);
    expect(viewerProps.interactions?.draw?.enabled).toBe(false);
    expect(viewerProps.viewport?.disablePan).toBe(false);
  });

  it("routes ROI placement and disables pan after Navigate is turned off", () => {
    const props = makeLeftPanelProps({
      workflow: {
        ...makeLeftPanelProps().workflow,
        navigateMode: false,
        roiPlacementActive: true,
      },
    });

    const viewerProps = buildLeftPanelViewerConfig({
      ...props,
      overlayScene: { persistent: [], transient: [] },
    });

    expect(viewerProps.interactions?.mode).toBeUndefined();
    expect(viewerProps.interactions?.onImageClick).toBe(props.segments.onClick);
    expect(viewerProps.interactions?.onImagePress).toBe(props.segments.onPress);
    expect(viewerProps.interactions?.onImageDrag).toBe(props.segments.onDrag);
    expect(viewerProps.interactions?.onImageRelease).toBe(props.segments.onRelease);
    expect(viewerProps.viewport?.disablePan).toBe(true);
  });

  it("reserves the drag for box-to-object instead of the correction brush", () => {
    const props = makeLeftPanelProps({
      workflow: {
        ...makeLeftPanelProps().workflow,
        mode: "review",
        reviewPhase: "correction",
        correctionTool: "sam",
        leftMode: "hover",
        navigateMode: false,
        samBoxActive: true,
      },
    });

    const viewerProps = buildLeftPanelViewerConfig({
      ...props,
      overlayScene: { persistent: [], transient: [] },
    });

    expect(viewerProps.interactions?.brush?.enabled).toBe(false);
    expect(viewerProps.interactions?.draw?.enabled).toBe(false);
    expect(viewerProps.interactions?.onImageClick).toBeUndefined();
    expect(viewerProps.viewport?.disablePan).toBe(true);
    expect(viewerProps.highlighting?.hoverCursor).toBe(false);
  });
});
