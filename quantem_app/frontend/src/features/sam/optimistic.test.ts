import { describe, expect, it } from "vitest";

import { optimisticSegmentForSamResponse } from "./optimistic";
import type { SamBoxResponse } from "./types";

function response(overrides: Partial<SamBoxResponse> = {}): SamBoxResponse {
  return {
    created: 1,
    updated: 0,
    deleted: 0,
    confirmed_ids: ["confirmed-1"],
    overlay: {
      desired_revision: 8,
      applied_revision: 7,
      sync_applied: false,
      rebuild_mode: "async_partial",
    },
    object: {
      geometry_coords: [
        [10, 10],
        [20, 10],
        [20, 20],
        [10, 10],
      ],
      score: 0.91,
      area: 100,
    },
    other_candidates: [],
    timing: {
      cache_hit: true,
      encode_ms: 0,
      decode_ms: 18,
      device: "cuda",
    },
    ...overrides,
  };
}

describe("optimisticSegmentForSamResponse", () => {
  it("uses the stored object id and returned polygon for an immediate confirmed overlay", () => {
    expect(optimisticSegmentForSamResponse("seg-1", response())).toMatchObject({
      id: "confirmed-1",
      segmentation: "seg-1",
      label_state: "CONFIRMED",
      confidence_score: null,
      geometry_coords: [
        [10, 10],
        [20, 10],
        [20, 20],
        [10, 10],
      ],
    });
  });

  it("does not invent an overlay when SAM stored nothing", () => {
    expect(
      optimisticSegmentForSamResponse(
        "seg-1",
        response({ created: 0, confirmed_ids: [] })
      )
    ).toBeNull();
  });
});
