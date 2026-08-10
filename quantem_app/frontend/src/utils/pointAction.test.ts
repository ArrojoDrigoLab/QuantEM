import { describe, expect, it } from "vitest";
import { selectBestPointActionSegment } from "@/utils/pointAction";
import type { SegmentObject } from "@/shared/types";

function makeSegment(
  id: string,
  labelState: SegmentObject["label_state"],
  confidenceScore: number | null
): SegmentObject {
  return {
    id,
    segmentation: "seg-1",
    label_state: labelState,
    confidence_score: confidenceScore,
    geometry_coords: [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
    ],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

describe("selectBestPointActionSegment", () => {
  it("prefers candidates over inferred segments at the same point", () => {
    const result = selectBestPointActionSegment([
      makeSegment("inferred", "INFERRED", 0.99),
      makeSegment("candidate", "CANDIDATE", 0.25),
    ]);

    expect(result?.id).toBe("candidate");
  });

  it("chooses the highest-confidence actionable candidate", () => {
    const result = selectBestPointActionSegment([
      makeSegment("low", "CANDIDATE", 0.2),
      makeSegment("high", "CANDIDATE", 0.9),
      makeSegment("null-score", "CANDIDATE", null),
    ]);

    expect(result?.id).toBe("high");
  });

  it("ignores non-actionable segment states", () => {
    const result = selectBestPointActionSegment([
      makeSegment("confirmed", "CONFIRMED", 0.9),
      makeSegment("excluded", "EXCLUDED", 0.8),
    ]);

    expect(result).toBeNull();
  });

  it("allows confirmed segments to be rejected when no candidate is selected", () => {
    const result = selectBestPointActionSegment(
      [
        makeSegment("confirmed", "CONFIRMED", 0.9),
        makeSegment("excluded", "EXCLUDED", 0.8),
      ],
      "reject"
    );

    expect(result?.id).toBe("confirmed");
  });
});
