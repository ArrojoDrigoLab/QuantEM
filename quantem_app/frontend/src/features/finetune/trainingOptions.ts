import type { FineTuneRunPayload } from "@/shared/types/finetune";

interface TrainingInputs {
  schedule: "quick" | "extended";
  steps: string;
  lr: string;
  seed: string;
  validation: boolean;
  evalInterval: string;
  patience: string;
}

type TrainingOptions = Pick<FineTuneRunPayload,
  "schedule" | "steps" | "lr" | "seed" | "validation_fraction" | "eval_interval" | "patience">;

/** Validate editable controls before sending them; the server checks tile coverage. */
export function parseTrainingInputs(inputs: TrainingInputs):
  { options: TrainingOptions; error: null } | { options: null; error: string } {
  function number(label: string, text: string, min: number, max: number, integer = true) {
    const value = Number(text);
    if (!text.trim() || !Number.isFinite(value) || value < min || value > max ||
        (integer && !Number.isInteger(value))) {
      throw new Error(`${label} must be ${integer ? "a whole number" : "a number"} from ${min} to ${max}.`);
    }
    return value;
  }
  try {
    return { error: null, options: {
      schedule: inputs.schedule,
      steps: inputs.steps.trim() ? number("Custom steps", inputs.steps, 1, 20000) : undefined,
      lr: number("Learning rate", inputs.lr, 1e-7, 0.01, false),
      seed: number("Random seed", inputs.seed, 0, 4294967295),
      validation_fraction: inputs.validation ? 0.2 : 0,
      eval_interval: number("Evaluation interval", inputs.evalInterval, 1, 20000),
      patience: inputs.validation ? number("Patience", inputs.patience, 0, 1000) : 0,
    } };
  } catch (error) {
    return { options: null, error: (error as Error).message };
  }
}
