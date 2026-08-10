/**
 * Step 3 — how far to go, and how much to spend.
 *
 * Two rungs. Threshold calibration fits a single scalar against probability
 * maps that already exist: seconds of CPU, no weights changed, and it is the
 * honest default because most "the model is wrong on my data" is a threshold.
 * Head training fits the neck and decoder with the encoder frozen; it needs
 * torch, and the server says whether this machine has it.
 */

import { Badge, Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import type { AdaptBudget } from "@/features/finetune/adaptBudget";
import type {
  AdaptCropsResponse,
  AdapterStatus,
  AdaptMode,
} from "@/shared/types/finetune";
import type { Runnability } from "@/features/models/runnable";

const FIELD_CLASS =
  "h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500";
const LABEL_CLASS =
  "block text-xs font-semibold uppercase tracking-wide text-slate-500";

export type { AdaptBudget } from "@/features/finetune/adaptBudget";

export interface StepModeProps {
  crops: AdaptCropsResponse | null;
  mode: AdaptMode;
  onModeChange: (mode: AdaptMode) => void;
  budget: AdaptBudget;
  onBudgetChange: (budget: AdaptBudget) => void;
  /**
   * Whether the chosen base model can be loaded here.
   *
   * Only head training loads it — threshold calibration sweeps an existing
   * probability map — so this narrows one rung, not the step.
   */
  baseModelRunnability: Runnability;
  baseModel: string;
  /**
   * The run these settings already belong to, when one exists.
   *
   * Once a run has been started this step stops being a form and becomes a
   * report of what was asked for: the wizard cannot start a second run on the
   * same segmentation, so leaving the controls live invited the reader to
   * "change" a mode that nothing would act on. Null before anything is started.
   */
  existingRun?: { status: AdapterStatus; mode: AdaptMode } | null;
}

const RUN_STATE_COPY: Record<AdapterStatus, string> = {
  PENDING: "is queued",
  RUNNING: "is running now",
  SUCCESS: "has finished",
  FAILED: "failed",
};

const MODE_COPY: Record<AdaptMode, { title: string; detail: string }> = {
  threshold_only: {
    title: "Calibrate the threshold",
    detail:
      "Sweeps the decision threshold over probability maps the app has already computed and keeps the value that maximises mean Dice on the fitted crops. No weights change. Seconds on a CPU.",
  },
  head: {
    title: "Train the head",
    detail:
      "Freezes the encoder and fits the neck and decoder to your crops, then calibrates the threshold on the adapted model. Needs PyTorch; minutes to tens of minutes depending on the device.",
  },
};

export function StepMode({
  crops,
  mode,
  onModeChange,
  budget,
  onBudgetChange,
  baseModelRunnability,
  baseModel,
  existingRun = null,
}: StepModeProps) {
  const available = crops?.modes ?? ["threshold_only"];
  const headBlockedByModel = baseModelRunnability.state === "blocked";
  const locked = existingRun !== null;
  const patch = (updates: Partial<AdaptBudget>) =>
    onBudgetChange({ ...budget, ...updates });

  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-4">
        <h2 className="m-0 text-base font-semibold text-slate-950">
          {locked ? "What this run fitted" : "Choose what to fit"}
        </h2>
        {existingRun ? (
          // The trap this closes: a reload during a head training used to land
          // back here showing "Calibrate the threshold" selected, because the
          // form had gone back to its default while the real run carried on.
          <p
            className="m-0 mt-2 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700"
            role="status"
          >
            An adaptation for this segmentation{" "}
            {RUN_STATE_COPY[existingRun.status]}, and it was started as{" "}
            <strong>{MODE_COPY[existingRun.mode].title.toLowerCase()}</strong>.
            These settings describe that run, so they cannot be changed here.
          </p>
        ) : null}
        <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {(["threshold_only", "head"] as AdaptMode[]).map((candidate) => {
            const offered =
              available.includes(candidate) &&
              !(candidate === "head" && headBlockedByModel);
            const selected = mode === candidate;
            return (
              <button
                key={candidate}
                type="button"
                disabled={!offered || locked}
                aria-pressed={selected}
                onClick={() => onModeChange(candidate)}
                className={cx(
                  "rounded-md border px-3 py-3 text-left transition-colors",
                  selected
                    ? "border-cyan-500 bg-cyan-50"
                    : "border-slate-200 bg-white",
                  !offered || locked ? "cursor-not-allowed" : "hover:bg-slate-50",
                  (!offered || (locked && !selected)) && "opacity-60"
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-sm font-medium text-slate-900">
                    {MODE_COPY[candidate].title}
                  </span>
                  {locked && selected ? (
                    <Badge tone="info">this run</Badge>
                  ) : offered ? null : (
                    <Badge tone="warning">unavailable here</Badge>
                  )}
                </div>
                <p className="m-0 mt-1 text-xs text-slate-600">
                  {MODE_COPY[candidate].detail}
                </p>
              </button>
            );
          })}
        </div>
        {/* Both of these explain why a rung is not on offer, which is only a
            live question while a rung is still being chosen. */}
        {locked ? null : !available.includes("head") ? (
          <p className="m-0 mt-3 text-xs text-amber-700">
            Head training is not offered because PyTorch is not installed on this
            machine. Threshold calibration works without it.
          </p>
        ) : headBlockedByModel ? (
          <p className="m-0 mt-3 text-xs text-amber-700">
            Head training is not offered because {baseModel} cannot be loaded
            here: {baseModelRunnability.reason} Threshold calibration does not
            load the model, so it is still available.
          </p>
        ) : null}
      </Panel>

      <Panel className="p-4">
        <h3 className="m-0 text-sm font-semibold text-slate-900">Budget</h3>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div>
            <label className={LABEL_CLASS} htmlFor="adapt-name">
              Name
            </label>
            <input
              id="adapt-name"
              type="text"
              className={`${FIELD_CLASS} mt-1`}
              value={budget.name}
              placeholder="mito @ my-liver-set"
              disabled={locked}
              onChange={(event) => patch({ name: event.target.value })}
            />
          </div>
          <div>
            <label className={LABEL_CLASS} htmlFor="adapt-seed">
              Seed
            </label>
            <input
              id="adapt-seed"
              type="number"
              className={`${FIELD_CLASS} mt-1`}
              value={budget.seed}
              disabled={locked}
              onChange={(event) =>
                patch({ seed: Number.parseInt(event.target.value, 10) || 0 })
              }
            />
          </div>
          {mode === "head" ? (
            <>
              <div>
                <label className={LABEL_CLASS} htmlFor="adapt-steps">
                  Steps
                </label>
                <input
                  id="adapt-steps"
                  type="number"
                  min={1}
                  className={`${FIELD_CLASS} mt-1`}
                  value={budget.steps}
                  disabled={locked}
                  onChange={(event) =>
                    patch({ steps: Number.parseInt(event.target.value, 10) || 0 })
                  }
                />
                <p className="m-0 mt-1 text-xs text-slate-500">
                  300 is the reference recipe; more steps on a handful of crops
                  fits the crops, not the organelle.
                </p>
                {/* At the field, while it is still focused. The server would
                    otherwise swap a zero for 300 and record that as what was
                    asked for. */}
                {!Number.isInteger(budget.steps) || budget.steps < 1 ? (
                  <p className="m-0 mt-1 text-xs text-red-700" role="alert">
                    Steps must be a whole number of at least 1.
                  </p>
                ) : null}
              </div>
              <div>
                <label className={LABEL_CLASS} htmlFor="adapt-lr">
                  Learning rate
                </label>
                <input
                  id="adapt-lr"
                  type="number"
                  step="0.00001"
                  min={0}
                  className={`${FIELD_CLASS} mt-1`}
                  value={budget.lr}
                  disabled={locked}
                  onChange={(event) =>
                    patch({ lr: Number.parseFloat(event.target.value) || 0 })
                  }
                />
                <p className="m-0 mt-1 text-xs text-slate-500">
                  1e-4 with AdamW, as published.
                </p>
                {!Number.isFinite(budget.lr) || budget.lr <= 0 ? (
                  <p className="m-0 mt-1 text-xs text-red-700" role="alert">
                    The learning rate must be greater than zero.
                  </p>
                ) : null}
              </div>
            </>
          ) : (
            <p className="m-0 text-xs text-slate-500 sm:col-span-2">
              Threshold calibration has nothing to tune: it sweeps every
              threshold and keeps the best on the fitted crops.
            </p>
          )}
        </div>
      </Panel>
    </div>
  );
}
