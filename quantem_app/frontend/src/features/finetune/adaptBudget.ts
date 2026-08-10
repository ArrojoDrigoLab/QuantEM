/**
 * What a head training is allowed to cost, and what counts as a legal answer.
 *
 * Separate from the step that renders it because both the form and the wizard's
 * start handler need the rule, and a shared rule is the only way the field and
 * the refusal can be made to say the same thing.
 */

import type { AdaptMode } from "@/shared/types/finetune";

export interface AdaptBudget {
  steps: number;
  lr: number;
  seed: number;
  name: string;
}

/**
 * Why this budget cannot be run, or null when it can.
 *
 * `min={1}` on a number input is decorative in every browser, and the server
 * reads the budget as `int(steps or 300)` / `float(lr or 1e-4)` -- so a typed
 * zero is not refused, it is *replaced*, and the adapter row then reports a
 * budget nobody asked for as though they had. The step count and the learning
 * rate are the two numbers a methods section quotes about a fine-tune, so the
 * form refuses rather than letting the server substitute silently.
 *
 * Only checked for head training: threshold calibration ignores both.
 */
export function adaptBudgetError(
  mode: AdaptMode,
  budget: AdaptBudget
): string | null {
  if (!Number.isInteger(budget.seed)) {
    return "The seed must be a whole number.";
  }
  if (mode !== "head") return null;
  if (!Number.isInteger(budget.steps) || budget.steps < 1) {
    return "Steps must be a whole number of at least 1.";
  }
  if (!Number.isFinite(budget.lr) || budget.lr <= 0) {
    return "The learning rate must be greater than zero.";
  }
  return null;
}
