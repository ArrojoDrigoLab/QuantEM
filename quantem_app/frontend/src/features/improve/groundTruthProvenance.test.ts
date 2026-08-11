import { describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import {
  EMPTY_PROVENANCE,
  fetchGroundTruthProvenance,
  mergeProvenance,
  modelDerivedFraction,
  needsSelfAgreementCaveat,
} from "@/features/improve/groundTruthProvenance";
import { server } from "@/test/msw/server";

const SEG = "seg-1";
const BASE = "http://127.0.0.1:8000";

function roi(id: string) {
  return {
    id,
    segmentation: SEG,
    polygon_coords: [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100],
    ],
    bbox: { x0: 0, y0: 0, x1: 100, y1: 100 },
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

function segment(label_state: string, source_model: string) {
  return {
    id: `${label_state}-${source_model}-${Math.random()}`,
    label_state,
    source_model,
    confidence_score: 0.9,
  };
}

describe("modelDerivedFraction", () => {
  it("is the share of positives that started as model output", () => {
    // The real session: 86 model candidates confirmed, 4 drawn by hand.
    expect(
      modelDerivedFraction({
        confirmedFromModel: 86,
        drawnByHand: 4,
        rejected: 0,
        totalConfirmed: 90,
        regions: 1,
      })
    ).toBeCloseTo(86 / 90);
  });

  it("is null rather than 0 when nothing is confirmed", () => {
    expect(modelDerivedFraction(EMPTY_PROVENANCE)).toBeNull();
  });
});

describe("needsSelfAgreementCaveat", () => {
  it("fires when almost all the ground truth is the model's own output", () => {
    expect(
      needsSelfAgreementCaveat({
        confirmedFromModel: 86,
        drawnByHand: 4,
        rejected: 0,
        totalConfirmed: 90,
        regions: 1,
      })
    ).toBe(true);
  });

  it("does not fire for a mixed reference", () => {
    // Confirming model output is the intended workflow; it is only a problem
    // when there is almost nothing independent left to score against.
    expect(
      needsSelfAgreementCaveat({
        confirmedFromModel: 40,
        drawnByHand: 60,
        rejected: 0,
        totalConfirmed: 100,
        regions: 1,
      })
    ).toBe(false);
  });

  it("does not fire when there is nothing to judge", () => {
    expect(needsSelfAgreementCaveat(EMPTY_PROVENANCE)).toBe(false);
  });
});

describe("mergeProvenance", () => {
  it("sums across segmentations", () => {
    const merged = mergeProvenance([
      { confirmedFromModel: 10, drawnByHand: 2, rejected: 1, totalConfirmed: 12, regions: 1 },
      { confirmedFromModel: 5, drawnByHand: 3, rejected: 0, totalConfirmed: 8, regions: 2 },
    ]);
    expect(merged).toEqual({
      confirmedFromModel: 15,
      drawnByHand: 5,
      rejected: 1,
      totalConfirmed: 20,
      regions: 3,
    });
  });
});

describe("fetchGroundTruthProvenance", () => {
  it("splits confirmed objects by whether a model proposed them", async () => {
    server.use(
      http.get(`${BASE}/api/segmentations/${SEG}/completed-rois/`, () =>
        HttpResponse.json([roi("roi-1")])
      ),
      http.post(`${BASE}/api/segmentations/${SEG}/segments/query-region`, () =>
        HttpResponse.json({
          segments: [
            ...Array.from({ length: 86 }, () =>
              segment("CONFIRMED", "quantem:mito")
            ),
            ...Array.from({ length: 4 }, () => segment("CONFIRMED", "manual")),
            segment("EXCLUDED", "quantem:mito"),
          ],
        })
      )
    );

    const result = await fetchGroundTruthProvenance(SEG);

    expect(result).toEqual({
      confirmedFromModel: 86,
      drawnByHand: 4,
      rejected: 1,
      totalConfirmed: 90,
      regions: 1,
    });
  });

  it("is empty when there is no completed area", async () => {
    server.use(
      http.get(`${BASE}/api/segmentations/${SEG}/completed-rois/`, () =>
        HttpResponse.json([])
      )
    );

    expect(await fetchGroundTruthProvenance(SEG)).toEqual(EMPTY_PROVENANCE);
  });

  it("sums over every completed area", async () => {
    let call = 0;
    server.use(
      http.get(`${BASE}/api/segmentations/${SEG}/completed-rois/`, () =>
        HttpResponse.json([roi("roi-1"), roi("roi-2")])
      ),
      http.post(`${BASE}/api/segmentations/${SEG}/segments/query-region`, () => {
        call += 1;
        return HttpResponse.json({
          segments: [
            segment("CONFIRMED", "quantem:mito"),
            segment("CONFIRMED", call === 1 ? "manual" : "quantem:mito"),
          ],
        });
      })
    );

    const result = await fetchGroundTruthProvenance(SEG);

    expect(result.regions).toBe(2);
    expect(result.totalConfirmed).toBe(4);
    expect(result.confirmedFromModel).toBe(3);
    expect(result.drawnByHand).toBe(1);
  });
});
