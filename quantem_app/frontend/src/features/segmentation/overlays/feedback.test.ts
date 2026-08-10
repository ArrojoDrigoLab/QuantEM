import { describe, expect, it } from "vitest";
import { generateUserFeedbackPointOverlays } from "@/features/segmentation/overlays/feedback";

describe("segmentation overlay feedback", () => {
  it("colors feedback points by utilized status", () => {
    const overlays = generateUserFeedbackPointOverlays([
      {
        id: "queued",
        segmentation: "seg-1",
        input_type: "point",
        point: { x: 1, y: 1 },
        polygon_coords: null,
        feedback_type: "CONFIRMED",
        utilized_status: "QUEUED",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
      {
        id: "success",
        segmentation: "seg-1",
        input_type: "point",
        point: { x: 2, y: 2 },
        polygon_coords: null,
        feedback_type: "CONFIRMED",
        utilized_status: "SUCCESS",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
      {
        id: "failed",
        segmentation: "seg-1",
        input_type: "point",
        point: { x: 3, y: 3 },
        polygon_coords: null,
        feedback_type: "REJECTED",
        utilized_status: "FAILED",
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);

    expect(overlays.find((overlay) => overlay.id === "user-feedback-queued")?.fillColor).toBe(
      "#ffd400"
    );
    expect(overlays.find((overlay) => overlay.id === "user-feedback-success")?.fillColor).toBe(
      "#33cc66"
    );
    expect(overlays.find((overlay) => overlay.id === "user-feedback-failed")?.fillColor).toBe(
      "#ff5d5d"
    );
  });
});
