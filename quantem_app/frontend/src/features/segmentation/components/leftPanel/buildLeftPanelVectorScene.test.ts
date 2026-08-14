import { describe, expect, it } from "vitest";
import { buildLeftPanelVectorScene } from "@/features/segmentation/components/leftPanel/buildLeftPanelVectorScene";
import {
  makeLeftPanelFeedback,
  makeLeftPanelProps,
} from "@/features/segmentation/components/leftPanel/leftPanelTestUtils";
import { makeSegment } from "@/features/segmentation/SegmentationScreen.testUtils";

describe("buildLeftPanelVectorScene", () => {
  it("composes group, drawing, roi, feedback, and extra overlays", () => {
    const scene = buildLeftPanelVectorScene(
      makeLeftPanelProps({
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
        segments: {
          ...makeLeftPanelProps().segments,
          items: [makeSegment({ id: "segment-1", label_state: "CONFIRMED" })],
          groupSelectionBBox: { x0: 1, y0: 2, x1: 5, y1: 6 },
          groupHighlightedSegmentIds: ["segment-1"],
        },
        roi: {
          ...makeLeftPanelProps().roi,
          activeRoi: {
            id: "roi-1",
            segmentation: "seg-1",
            x: 100,
            y: 120,
            width: 80,
            height: 60,
            source: "MANUAL",
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
              x: 300,
              y: 320,
              width: 80,
              height: 60,
              source: "MANUAL",
              seed: null,
              is_active: false,
              is_complete: false,
              created_at: "2026-01-02T00:00:00Z",
              updated_at: "2026-01-02T00:00:00Z",
            },
          ],
        },
        drawing: {
          ...makeLeftPanelProps().drawing,
          pendingPolygon: [
            { x: 50, y: 50 },
            { x: 70, y: 50 },
            { x: 50, y: 70 },
          ],
        },
        feedback: {
          items: [makeLeftPanelFeedback()],
        },
        overlays: {
          extraTransientOverlays: [
            {
              id: "extra-overlay",
              geometry: [
                { x: 1, y: 1 },
                { x: 2, y: 1 },
                { x: 1, y: 2 },
              ],
              fillColor: "#fff",
              fillOpacity: 0.1,
              strokeColor: "#000",
              strokeOpacity: 0.5,
              strokeWidth: 1,
            },
          ],
        },
      })
    );

    const overlayIds = new Set([...scene.persistent, ...scene.transient].map((overlay) => overlay.id));

    expect(Array.from(overlayIds)).toEqual(
      expect.arrayContaining([
        "segment-1",
        "roi-frame",
        "roi-frame-roi-2",
        "right-selection-bbox",
        "pending-polygon",
        "user-feedback-feedback-1",
        "extra-overlay",
      ])
    );
    expect(scene.transient.find((overlay) => overlay.id === "roi-frame-roi-2")).toMatchObject({
      strokeOpacity: 0.4,
      strokeDasharray: "8 6",
    });
  });

  it("hides only the ROI overlay being area-edited", () => {
    const scene = buildLeftPanelVectorScene(
      makeLeftPanelProps({
        roi: {
          ...makeLeftPanelProps().roi,
          activeRoi: {
            id: "roi-1",
            segmentation: "seg-1",
            x: 0,
            y: 0,
            width: 10,
            height: 10,
            source: "MANUAL",
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
              x: 20,
              y: 20,
              width: 10,
              height: 10,
              source: "MANUAL",
              seed: null,
              is_active: false,
              is_complete: false,
              created_at: "2026-01-02T00:00:00Z",
              updated_at: "2026-01-02T00:00:00Z",
            },
          ],
        },
        overlays: {
          hideRoiOverlayId: "roi-1",
        },
      })
    );

    expect(scene.transient.map((overlay) => overlay.id)).not.toContain("roi-frame");
    expect(scene.transient.map((overlay) => overlay.id)).toContain("roi-frame-roi-2");
  });

  it("renders completed ROI overlays only while the mode is active", () => {
    const activeScene = buildLeftPanelVectorScene(
      makeLeftPanelProps({
        completedRoi: {
          ...makeLeftPanelProps().completedRoi,
          active: true,
          items: [
            {
              id: "completed-roi-1",
              segmentation: "seg-1",
              polygon_coords: [
                [5, 5],
                [25, 5],
                [25, 25],
                [5, 25],
                [5, 5],
              ],
              bbox: { x0: 5, y0: 5, x1: 25, y1: 25 },
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-01T00:00:00Z",
            },
          ],
          liveSectionPoints: [
            { x: 40, y: 40 },
            { x: 60, y: 40 },
          ],
          hasDraft: true,
        },
      })
    );
    const inactiveScene = buildLeftPanelVectorScene(
      makeLeftPanelProps({
        completedRoi: {
          ...makeLeftPanelProps().completedRoi,
          items: [
            {
              id: "completed-roi-1",
              segmentation: "seg-1",
              polygon_coords: [
                [5, 5],
                [25, 5],
                [25, 25],
                [5, 25],
                [5, 5],
              ],
              bbox: { x0: 5, y0: 5, x1: 25, y1: 25 },
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-01T00:00:00Z",
            },
          ],
          liveSectionPoints: [
            { x: 40, y: 40 },
            { x: 60, y: 40 },
          ],
          hasDraft: true,
        },
      })
    );

    expect(activeScene.transient.map((overlay) => overlay.id)).toEqual(
      expect.arrayContaining(["completed-roi-completed-roi-1", "completed-roi-draft-live"])
    );
    expect(inactiveScene.transient.map((overlay) => overlay.id)).not.toEqual(
      expect.arrayContaining(["completed-roi-completed-roi-1", "completed-roi-draft-live"])
    );
  });
});
