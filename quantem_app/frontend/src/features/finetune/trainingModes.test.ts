/**
 * Three choices on screen, two fields on the wire, and no way to express the
 * combination that means nothing.
 */

import { describe, expect, it } from "vitest";
import {
  FINE_TUNE_MODE_OPTIONS,
  TRAINING_MODE_HELP,
  USE_ALL_TILE_CEILING,
  modeChoiceFromDefault,
  modeChoicePayload,
} from "@/features/finetune/trainingModes";

describe("features/finetune/trainingModes", () => {
  it("offers exactly the three the owner asked for", () => {
    expect(FINE_TUNE_MODE_OPTIONS.map((option) => option.value)).toEqual([
      "use_all",
      "holdout_1",
      "holdout_1_cv",
    ]);
  });

  it("maps each choice onto the two wire fields", () => {
    expect(modeChoicePayload("use_all")).toEqual({
      mode: "use_all",
      cv_benchmark: false,
    });
    expect(modeChoicePayload("holdout_1")).toEqual({
      mode: "holdout_1",
      cv_benchmark: false,
    });
    expect(modeChoicePayload("holdout_1_cv")).toEqual({
      mode: "holdout_1",
      cv_benchmark: true,
    });
  });

  it("never asks for cross-validation over a run that holds nothing back", () => {
    for (const option of FINE_TUNE_MODE_OPTIONS) {
      const payload = modeChoicePayload(option.value);
      if (payload.mode === "use_all") expect(payload.cv_benchmark).toBe(false);
    }
  });

  it("reads the server's default without inventing a third value", () => {
    expect(modeChoiceFromDefault("use_all")).toBe("use_all");
    expect(modeChoiceFromDefault("holdout_1")).toBe("holdout_1");
  });

  it("explains the tile threshold the default is decided on", () => {
    expect(TRAINING_MODE_HELP.join(" ")).toContain(
      `${USE_ALL_TILE_CEILING} tiles or fewer`
    );
  });
});
