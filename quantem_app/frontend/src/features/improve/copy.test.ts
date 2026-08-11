/**
 * The plain-language layer, and the rule that keeps it honest.
 *
 * I-4 in UX_PLAN §7: *"a test asserts the plain sentence and the drawer never
 * disagree in direction."* That is the last test in this file, and it is the
 * reason these sentences are functions rather than JSX.
 */

import { describe, expect, it } from "vitest";
import {
  calibrationReport,
  checkedAreasSentence,
  evidenceSentence,
  improvementDirection,
  levelChanged,
  summariseCheckedAreas,
} from "@/features/improve/copy";
import type { AdaptCrop, Adapter, AdapterSweep } from "@/shared/types/finetune";

function crop(overrides: Partial<AdaptCrop>): AdaptCrop {
  return {
    id: "c",
    name: "abcd1234_0",
    image_key: "asset-1",
    width: 256,
    height: 256,
    n_objects: 4,
    annotated_px: 1000,
    has_probability: true,
    is_this_image: true,
    image_name: "Grid2 Cell10",
    ...overrides,
  };
}

function sweep(overrides: Partial<AdapterSweep> = {}): AdapterSweep {
  return {
    thresholds: [0.4, 0.45, 0.5],
    train_dice: [0.8, 0.9, 0.85],
    calibrated_threshold: 0.65,
    train_dice_at_calibrated: 0.9,
    train_dice_at_default: 0.85,
    heldout_dice_at_calibrated: 0.83,
    heldout_dice_at_default: 0.78,
    heldout_oracle: 0.86,
    improvement: 0.05,
    per_crop: { "abcd1234_0": 0.83 },
    train_crop_names: ["abcd1234_0"],
    heldout_crop_names: ["efgh5678_0"],
    ...overrides,
  };
}

function adapter(overrides: Partial<Adapter> = {}): Adapter {
  const s = (overrides.sweep as AdapterSweep) ?? sweep();
  return {
    id: "ad-1",
    base_model: "quantem:mito",
    name: "Liver 01",
    status: "SUCCESS",
    mode: "threshold_only",
    steps: 0,
    trainable_params: 0,
    segmentation_id: "seg-1",
    split_mode: "image-disjoint",
    train_crop_names: s.train_crop_names,
    heldout_crop_names: s.heldout_crop_names,
    sweep: s,
    calibrated_threshold: s.calibrated_threshold,
    default_threshold: 0.5,
    heldout_dice: s.heldout_dice_at_calibrated,
    verified_reload: false,
    train_seconds: null,
    applied_at: null,
    created_at: "2026-01-01T00:00:00Z",
    error: "",
    caveats: [],
    ...overrides,
  };
}

describe("checkedAreasSentence", () => {
  it("names the other images, because crops are pooled across all of them", () => {
    // UX_PLAN §1.9: the old copy said the app learns "only from what you mark"
    // on this image. It pools every image with the same organelle segmented,
    // and naming them is what makes that checkable.
    const summary = summariseCheckedAreas([
      crop({ id: "a" }),
      crop({ id: "b" }),
      crop({ id: "c", is_this_image: false, image_name: "Grid2 Cell11" }),
    ]);
    expect(checkedAreasSentence(summary)).toBe(
      "I'll look at the 3 areas you've marked as checked — 2 on this image and " +
        "1 on Grid2 Cell11 — and match my cut-off to what you kept in them."
    );
  });

  it("stays a sentence when everything is on this image", () => {
    const summary = summariseCheckedAreas([crop({ id: "a" })]);
    expect(checkedAreasSentence(summary)).toBe(
      "I'll look at the 1 area you've marked as checked on this image, and " +
        "match my cut-off to what you kept in it."
    );
  });

  it("says outright when nothing being learned from is on this image", () => {
    const summary = summariseCheckedAreas([
      crop({ id: "a", is_this_image: false, image_name: "Grid2 Cell11" }),
    ]);
    expect(checkedAreasSentence(summary)).toContain("none on this image");
  });

  it("caps a long image list rather than printing twelve names", () => {
    const summary = summariseCheckedAreas(
      ["A", "B", "C", "D", "E"].map((name, index) =>
        crop({ id: name, is_this_image: false, image_name: `Grid ${name}`, name: `${index}` })
      )
    );
    expect(checkedAreasSentence(summary)).toContain(
      "Grid A, Grid B and Grid C and 2 more"
    );
  });

  it("does not pretend there is something to learn from when there is not", () => {
    expect(checkedAreasSentence(summariseCheckedAreas([]))).toContain(
      "nothing for me to learn from"
    );
  });
});

describe("evidenceSentence", () => {
  it("carries the sample size in the same sentence as the claim (I-4)", () => {
    expect(evidenceSentence("image-disjoint", 2)).toBe(
      "Checked against 2 checked areas on a different image that I did not fit to."
    );
    expect(evidenceSentence("within-image", 1)).toContain("1 checked area");
  });

  it("says a within-image score does not measure a new image", () => {
    expect(evidenceSentence("within-image", 1)).toContain(
      "does not tell you how it will do on a new image"
    );
  });

  it("refuses to imply evidence when everything was fitted to", () => {
    const sentence = evidenceSentence("no-heldout", 0);
    expect(sentence).toContain("nothing left over to check it against");
    expect(sentence).not.toContain("Checked against");
  });
});

describe("calibrationReport", () => {
  it("states the new include level with its number and the one it replaces", () => {
    const report = calibrationReport(adapter(), 0.42);
    expect(report.level).toBe(
      "New include level 0.65, where my default is 0.50."
    );
    expect(report.timing).toBe("Done in 0.4 seconds.");
  });

  it("names which way the model was wrong", () => {
    // Higher include level means fewer objects: it was including too much.
    expect(calibrationReport(adapter(), null).adjustment).toBe(
      "I was including a little too much."
    );
    const lower = adapter({
      sweep: sweep({ calibrated_threshold: 0.35 }),
      calibrated_threshold: 0.35,
    });
    expect(calibrationReport(lower, null).adjustment).toBe(
      "I was leaving things out."
    );
  });

  it("does not invite a pointless re-run when the level did not move", () => {
    const unchanged = adapter({
      sweep: sweep({ calibrated_threshold: 0.5 }),
      calibrated_threshold: 0.5,
    });
    const report = calibrationReport(unchanged, null);
    expect(report.level).toContain("already matched your marks best");
    expect(report.adjustment).toBeNull();
    expect(levelChanged(0.5, 0.5)).toBe(false);
  });

  it("will not claim 'better' when nothing was held back to check against", () => {
    const noHeldout = adapter({
      split_mode: "no-heldout",
      sweep: sweep({ improvement: null, heldout_crop_names: [] }),
      heldout_crop_names: [],
    });
    const report = calibrationReport(noHeldout, null);
    expect(report.direction).toBe("unknown");
    expect(report.verdict).toContain("cannot say it is better");
    expect(report.verdict).not.toContain("better than");
  });

  it("says so out loud when the held-out score got worse", () => {
    const worse = adapter({ sweep: sweep({ improvement: -0.04 }) });
    const report = calibrationReport(worse, null);
    expect(report.direction).toBe("worse");
    expect(report.verdict).toContain("my default did better");
  });

  it("always carries the preservation sentence", () => {
    expect(calibrationReport(adapter(), null).preservation).toContain(
      "kept, removed or drawn by hand"
    );
  });

  /**
   * I-4, the whole point of this file.
   *
   * `AboutTheNumbers` renders `sweep.improvement` with a sign. This asserts the
   * plain sentence never contradicts that sign — for every branch, including
   * the two where the honest answer is "no" and "I cannot tell".
   */
  it("never disagrees in direction with the number the drawer prints", () => {
    const cases: Array<[number | null, string]> = [
      [0.05, "better"],
      [0.0005, "same"],
      [0, "same"],
      [-0.0005, "same"],
      [-0.04, "worse"],
      [null, "unknown"],
    ];
    for (const [improvement, expected] of cases) {
      const s = sweep({ improvement });
      expect(improvementDirection(s)).toBe(expected);
      const report = calibrationReport(
        adapter({ sweep: s, calibrated_threshold: s.calibrated_threshold }),
        null
      );
      expect(report.direction).toBe(expected);
      // The words that assert a direction only appear for that direction.
      const claimsBetter = report.verdict.includes("better than my default");
      const claimsWorse = report.verdict.includes("my default did better");
      expect(claimsBetter).toBe(expected === "better");
      expect(claimsWorse).toBe(expected === "worse");
    }
  });

  /**
   * What is left on this list is **internal names that contradict the word on
   * screen** -- `EXCLUDED` for something the buttons call rejected, and
   * `CompletedROI` for something they call a confirmed area. Leaking one of
   * those hands the reader a second name for a thing they already have a name
   * for, which is the failure this test exists to catch.
   *
   * The list used to be much longer and was wrong about why. "adapter" and
   * "fine-tune" came off when the owner retired that convention (R15a) and
   * asked for a button called Fine-Tune. "threshold" and "Dice" came off
   * because they are the real names of a real control and a real metric, and
   * the owner's position is that this product should use them rather than
   * talk around them. "candidate" and "confirmed" came off because the app
   * already says both out loud -- the segmentation header says "are unconfirmed
   * candidates" -- so forbidding them here and printing them two screens away
   * was not a vocabulary rule, just an inconsistency.
   */
  it("uses none of the forbidden vocabulary", () => {
    const report = calibrationReport(adapter(), 1.2);
    const text = [
      report.timing,
      report.level,
      report.verdict,
      report.adjustment,
      report.evidence,
      report.preservation,
    ]
      .filter(Boolean)
      .join(" ");
    for (const word of ["excluded", "completed ROI"]) {
      expect(text.toLowerCase()).not.toContain(word.toLowerCase());
    }
  });
});
