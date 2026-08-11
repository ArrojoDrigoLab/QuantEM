/**
 * The domain-shift nudge, against numbers a real model produced.
 *
 * The acceptance for this heuristic is not "does the arithmetic work" — it is
 * **fires on a synthetic noise image, silent on a good one** — so the fixtures
 * here are measured, not invented. `quantem:mito` on CPU at threshold 0.5,
 * through the product's own closing and minimum-area filters:
 *
 *  - two synthetic noise images (uniform, and smoothed gaussian) produced
 *    **zero objects each**, with a mean probability of 0.0002 over the whole
 *    field. Out-of-domain input does not produce a pile of uncertain objects on
 *    this build; it produces nothing;
 *  - four real EM images from four different datasets in the fig4
 *    ground-truth set produced 15, 22, 73 and 118 objects, with mean
 *    confidences of 0.708, 0.608, 0.814 and 0.705.
 *
 * `GOOD_PLANT_IMAGE` below is the *worst* of the four — the one that comes
 * closest to firing — and it is here verbatim because it is what set
 * `lowMeanMargin`. Its mean of 0.608 cleared the originally-written 0.10 margin
 * by 0.008; a fixture that only just passes is a fixture that will fail on
 * somebody else's crop, so the margin moved rather than the test.
 */

import { describe, expect, it } from "vitest";
import {
  DOMAIN_SHIFT_LABEL,
  DOMAIN_SHIFT_MESSAGE,
  DOMAIN_SHIFT_THRESHOLDS,
  assessDomainShift,
} from "./domainShift";

/**
 * Per-object mean probability, `orgsegnet_plant tr_00219`, 22 objects,
 * mean 0.608, 32% of them within 0.05 of the cut-off. A correct result on a
 * plant sample: the nudge must stay silent.
 */
const GOOD_PLANT_IMAGE = [
  0.5103, 0.5108, 0.5222, 0.5229, 0.5289, 0.5293, 0.5394, 0.5523, 0.5554,
  0.5587, 0.5595, 0.564, 0.5672, 0.5699, 0.5717, 0.6379, 0.6487, 0.6505,
  0.6845, 0.721, 0.9128, 0.9616,
];

/**
 * `deeppi_em_skeletal_muscle te_00021`, 15 objects, mean 0.708. Below the
 * distribution floor, so it is also the case that proves the floor holds.
 */
const GOOD_MUSCLE_IMAGE = [
  0.5061, 0.5483, 0.5539, 0.5844, 0.5881, 0.6131, 0.6212, 0.6472, 0.655,
  0.6809, 0.8646, 0.9121, 0.9298, 0.9459, 0.9624,
];

describe("the acceptance: noise fires it, a good image does not", () => {
  it("fires on a synthetic noise image, which produced no objects at all", () => {
    // Both noise images measured identically here: 0 objects.
    const nudge = assessDomainShift({ objectCount: 0, runFinished: true });

    expect(nudge).not.toBeNull();
    expect(nudge!.reason).toBe("no_objects");
    expect(nudge!.message).toBe(DOMAIN_SHIFT_MESSAGE);
    expect(nudge!.label).toBe(DOMAIN_SHIFT_LABEL);
  });

  it("stays silent on the real image that comes closest to firing", () => {
    expect(
      assessDomainShift({
        objectCount: GOOD_PLANT_IMAGE.length,
        runFinished: true,
        confidences: GOOD_PLANT_IMAGE,
        threshold: 0.5,
      })
    ).toBeNull();
  });

  it("stays silent on the other real images too", () => {
    for (const [name, confidences] of [
      ["muscle", GOOD_MUSCLE_IMAGE],
      ["plant", GOOD_PLANT_IMAGE],
    ] as const) {
      expect(
        assessDomainShift({
          objectCount: confidences.length,
          runFinished: true,
          confidences,
          threshold: 0.5,
        }),
        name
      ).toBeNull();
    }
  });

  it("clears the plant image's mean by more than the whole band", () => {
    // The measurement that moved `lowMeanMargin` from 0.10 to 0.05, asserted
    // so a later tightening has to look at this number first.
    const mean =
      GOOD_PLANT_IMAGE.reduce((sum, value) => sum + value, 0) /
      GOOD_PLANT_IMAGE.length;

    expect(mean).toBeGreaterThan(0.6);
    expect(mean - 0.5).toBeGreaterThan(
      DOMAIN_SHIFT_THRESHOLDS.lowMeanMargin + DOMAIN_SHIFT_THRESHOLDS.cutOffBand
    );
  });
});

describe("the arms that need a distribution", () => {
  it("fires when almost everything sits on the cut-off", () => {
    // The intermediate case the real measurement did not produce: 40 objects,
    // all just over the bar. Constructed, and labelled as constructed.
    const piled = Array.from({ length: 40 }, (_, index) => 0.5 + index * 0.001);

    const nudge = assessDomainShift({
      objectCount: piled.length,
      runFinished: true,
      confidences: piled,
      threshold: 0.5,
    });

    expect(nudge?.reason).toBe("confidence_at_the_cut_off");
    expect(nudge?.evidence).toMatch(/100% of the 40 objects/);
  });

  it("fires on a low mean that is not piled tightly enough for arm 2", () => {
    // Half at 0.52, half at 0.56: only 100% within 0.05... so spread it wider
    // than the band while keeping the mean low.
    const spread = [
      ...Array.from({ length: 20 }, () => 0.52),
      ...Array.from({ length: 20 }, () => 0.57),
    ];

    const nudge = assessDomainShift({
      objectCount: spread.length,
      runFinished: true,
      confidences: spread,
      threshold: 0.5,
    });

    expect(nudge?.reason).toBe("low_mean_confidence");
    expect(nudge?.evidence).toMatch(/Average confidence/);
  });

  it("says nothing when there are too few objects to be a distribution", () => {
    // Six objects all at the cut-off is a small run, not evidence. This is the
    // real 6-object case from the first measurement pass, rounded down.
    expect(
      assessDomainShift({
        objectCount: 6,
        runFinished: true,
        confidences: [0.51, 0.51, 0.52, 0.52, 0.53, 0.53],
        threshold: 0.5,
      })
    ).toBeNull();
  });

  it("says nothing when it has no confidences to look at", () => {
    // The labeling header's situation: object count only. Arm 1 is all it can
    // do, and guessing the rest would be an invented result.
    expect(
      assessDomainShift({ objectCount: 40, runFinished: true })
    ).toBeNull();
    expect(
      assessDomainShift({
        objectCount: 40,
        runFinished: true,
        confidences: null,
        threshold: 0.5,
      })
    ).toBeNull();
  });

  it("says nothing when the threshold is unknown", () => {
    // Every confidence arm is relative to the cut-off. Without it there is no
    // question to ask, and assuming 0.5 would be wrong on an adapted model.
    expect(
      assessDomainShift({
        objectCount: 40,
        runFinished: true,
        confidences: Array.from({ length: 40 }, () => 0.5),
        threshold: null,
      })
    ).toBeNull();
  });
});

describe("when it must not speak at all", () => {
  it("is silent while the run is still going", () => {
    // Zero objects halfway through a run is not evidence of anything.
    expect(
      assessDomainShift({ objectCount: 0, runFinished: false })
    ).toBeNull();
  });

  it("is silent on a normal result", () => {
    const confident = Array.from({ length: 60 }, () => 0.85);

    expect(
      assessDomainShift({
        objectCount: confident.length,
        runFinished: true,
        confidences: confident,
        threshold: 0.5,
      })
    ).toBeNull();
  });

  it("refuses a distribution with a hole in it rather than averaging around it", () => {
    // A confidence array shorter than the object count, or holding a NaN, means
    // the caller does not have the distribution it thinks it has.
    const withHole = [...Array.from({ length: 39 }, () => 0.51), Number.NaN];

    expect(
      assessDomainShift({
        objectCount: 40,
        runFinished: true,
        confidences: withHole,
        threshold: 0.5,
      })
    ).toBeNull();
  });
});

describe("what it says", () => {
  it("always carries the label, and the label says it is a guess", () => {
    const nudge = assessDomainShift({ objectCount: 0, runFinished: true });

    expect(nudge!.label).toMatch(/guess/i);
    expect(nudge!.label).toMatch(/not a measurement/i);
  });

  it("offers the two things that actually work", () => {
    expect(DOMAIN_SHIFT_MESSAGE).toMatch(/other model family/);
    expect(DOMAIN_SHIFT_MESSAGE).toMatch(/mark up one box/);
  });

  it("never claims to have detected anything", () => {
    // The word this surface must not use about itself.
    expect(`${DOMAIN_SHIFT_LABEL} ${DOMAIN_SHIFT_MESSAGE}`).not.toMatch(
      /detect|out-of-distribution|confidence score|certain/i
    );
  });
});
