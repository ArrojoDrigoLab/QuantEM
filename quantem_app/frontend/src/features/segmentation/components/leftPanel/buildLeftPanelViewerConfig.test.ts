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

  it("keeps clicks live and forces brush/draw/pan off during ROI placement", () => {
    const props = makeLeftPanelProps({
      workflow: {
        ...makeLeftPanelProps().workflow,
        // Even with navigate mode on and a brush tool selected, placement wins.
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

    expect(viewerProps.interactions?.onImageClick).toBe(props.segments.onClick);
    expect(viewerProps.interactions?.brush?.enabled).toBe(false);
    expect(viewerProps.interactions?.draw?.enabled).toBe(false);
    expect(viewerProps.viewport?.disablePan).toBe(true);
  });
});
