import { beforeEach, describe, expect, it } from "vitest";
import { useUserFeedbackStore } from "@/shared/stores/useUserFeedbackStore";
import type { UserFeedback } from "@/shared/types";

function makeFeedback(id: string, status: UserFeedback["utilized_status"]): UserFeedback {
  return {
    id,
    segmentation: "seg-1",
    input_type: "point",
    point: { x: 1, y: 1 },
    polygon_coords: null,
    feedback_type: "CONFIRMED",
    utilized_status: status,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

describe("useUserFeedbackStore", () => {
  beforeEach(() => {
    useUserFeedbackStore.setState({ feedbackBySegmentation: {} });
  });

  it("upserts single and batch feedback items", () => {
    const store = useUserFeedbackStore.getState();
    store.upsertFeedback("seg-1", makeFeedback("one", "QUEUED"));
    store.upsertFeedbackBatch("seg-1", [
      makeFeedback("two", "PROCESSING"),
      makeFeedback("three", "SUCCESS"),
    ]);

    const feedback = useUserFeedbackStore.getState().feedbackBySegmentation["seg-1"];
    expect(Object.keys(feedback)).toHaveLength(3);
    expect(feedback.one.utilized_status).toBe("QUEUED");
    expect(feedback.two.utilized_status).toBe("PROCESSING");
    expect(feedback.three.utilized_status).toBe("SUCCESS");
  });

  it("removes and clears feedback safely", () => {
    const store = useUserFeedbackStore.getState();
    store.upsertFeedback("seg-1", makeFeedback("one", "QUEUED"));
    store.upsertFeedback("seg-1", makeFeedback("two", "FAILED"));
    store.removeFeedback("seg-1", "one");

    let feedback = useUserFeedbackStore.getState().feedbackBySegmentation["seg-1"];
    expect(Object.keys(feedback)).toEqual(["two"]);

    store.clearFeedback("seg-1");
    feedback = useUserFeedbackStore.getState().feedbackBySegmentation["seg-1"];
    expect(feedback).toEqual({});
  });
});
