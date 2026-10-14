import { expect, it } from "vitest";
import { parseTrainingInputs } from "./trainingOptions";

const defaults = { schedule: "quick" as const, steps: "", lr: "0.0001", seed: "0",
  validation: false, evalInterval: "100", patience: "5" };

it.each([
  { lr: "" }, { seed: "" }, { steps: "1.5" }, { steps: "20001" },
  { lr: "Infinity" }, { lr: "0" }, { seed: "-1" }, { seed: "4294967296" },
  { evalInterval: "" }, { validation: true, patience: "1001" },
])("rejects invalid controls before they become JSON defaults: %o", (inputs) => {
  expect(parseTrainingInputs({ ...defaults, ...inputs }).error).toBeTruthy();
});

it("keeps automatic steps and ignores inactive early stopping", () => {
  expect(parseTrainingInputs({ ...defaults, patience: "" })).toEqual({ error: null, options: {
    schedule: "quick", steps: undefined, lr: 0.0001, seed: 0,
    validation_fraction: 0, eval_interval: 100, patience: 0,
  } });
});
