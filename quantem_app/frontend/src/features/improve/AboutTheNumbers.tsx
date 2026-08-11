/**
 * "About the numbers" — the whole statistics layer, one click away.
 *
 * This is `StepResults.tsx` from the six-step wizard, moved here **verbatim**
 * and renamed. Nothing was cut in the move and nothing may be: `HonestScore`,
 * `SweepCurve`, `GroundTruthProvenance`, the split-mode badge, the oracle
 * wording and the server-generated "Read before quoting these numbers" caveats
 * all survive exactly as they were (UX_PLAN §1.6). What changed is where they
 * live — behind one disclosure on the Improve panel instead of behind four
 * wizard steps — and who has to read them: nobody, until they ask.
 *
 * The plain-language layer above it is not allowed to contradict this one. That
 * is a rule with a test (`copy.test.ts`), not an aspiration: the sentence
 * "matches your marks better than my default did" is derived from the same
 * `improvement` figure rendered here, so the two cannot disagree in direction.
 *
 * Layout follows the honesty rules rather than the drama: the split mode sits
 * above the numbers, the held-out score is never rendered apart from it, the
 * oracle is a labelled ceiling, and the per-crop table badges which crops the
 * threshold was fitted on. Improvement is shown as held-out at the calibrated
 * threshold minus held-out at the default — a training-set improvement is not
 * an improvement.
 */

import { Badge, Panel } from "@/shared/ui/design";
import { formatDuration, formatInteger, formatNumber } from "@/shared/ui/format";
import type { Adapter, AdapterSweep } from "@/shared/types/finetune";
import {
  HeldoutDice,
  OracleCeiling,
  SplitModeNote,
} from "@/features/improve/components/HonestScore";
import { SweepCurve } from "@/features/improve/components/SweepCurve";
import { GroundTruthProvenancePanel } from "@/features/improve/components/GroundTruthProvenance";
import type { GroundTruthProvenance } from "@/features/improve/groundTruthProvenance";

function isSweep(value: unknown): value is AdapterSweep {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as AdapterSweep).thresholds)
  );
}

export interface AboutTheNumbersProps {
  adapter: Adapter;
  /** The base model's sweep, which only the job result carries. */
  baseSweep: AdapterSweep | null;
  /**
   * Composition of the annotations the score was measured against.
   *
   * Shown directly under the numbers rather than in a separate step: a
   * held-out Dice against mostly-self-confirmed labels is a self-agreement
   * score, and that has to be visible at the moment the number is read.
   */
  provenance: GroundTruthProvenance | null;
  provenanceLoading: boolean;
  provenanceError: string | null;
}

export function AboutTheNumbers({
  adapter,
  baseSweep,
  provenance,
  provenanceLoading,
  provenanceError,
}: AboutTheNumbersProps) {
  const sweep = isSweep(adapter.sweep) ? adapter.sweep : null;

  if (adapter.status !== "SUCCESS" || !sweep) {
    return (
      <Panel className="p-4">
        <p className="m-0 text-sm text-slate-600">
          Results appear once the run succeeds.
        </p>
      </Panel>
    );
  }

  const trainNames = new Set(sweep.train_crop_names ?? adapter.train_crop_names);
  const cropNames = Object.keys(sweep.per_crop ?? {}).sort();
  const hasHeldout = adapter.split_mode !== "no-heldout";

  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="m-0 text-base font-semibold text-slate-950">
              {adapter.name || "Adapted model"}
            </h2>
            <p className="m-0 mt-1 text-xs text-slate-500">
              {adapter.base_model} · {adapter.mode.replace("_", " ")}
              {adapter.mode === "head"
                ? ` · ${formatInteger(adapter.steps)} steps · ${formatInteger(
                    adapter.trainable_params
                  )} trainable params`
                : ""}
              {adapter.train_seconds
                ? ` · ${formatDuration(adapter.train_seconds)}`
                : ""}
            </p>
          </div>
          {adapter.mode === "head" ? (
            <Badge tone={adapter.verified_reload ? "good" : "warning"}>
              {adapter.verified_reload
                ? "saved head reloaded and re-scored"
                : "not re-scored after saving"}
            </Badge>
          ) : (
            // `verified_reload` is false for every threshold-only run, and used
            // to be rendered as nothing at all. Silence reads as an omission;
            // it is actually false by construction, because calibration saves
            // no weights and so has nothing to reload. Saying so is the
            // difference between "unverified" and "not applicable".
            <Badge
              tone="default"
              title="Threshold calibration changes no weights: it fits a single scalar against probability maps that already exist. There is no saved head, so there is nothing to reload and re-score."
            >
              no weights to re-score
            </Badge>
          )}
        </div>

        <div className="mt-3">
          <SplitModeNote mode={adapter.split_mode} />
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <p className="m-0 text-xs uppercase tracking-wide text-slate-500">
              Chosen threshold
            </p>
            <p className="m-0 text-lg font-semibold tabular-nums text-slate-900">
              {formatNumber(sweep.calibrated_threshold, 2)}
            </p>
          </div>
          <div>
            <p className="m-0 text-xs uppercase tracking-wide text-slate-500">
              Fitted-crop Dice
            </p>
            <p className="m-0 text-lg font-semibold tabular-nums text-slate-900">
              {formatNumber(sweep.train_dice_at_calibrated, 3)}
            </p>
            <p className="m-0 text-xs text-slate-500">
              at default 0.50: {formatNumber(sweep.train_dice_at_default, 3)}
            </p>
          </div>
          <HeldoutDice
            value={sweep.heldout_dice_at_calibrated}
            mode={adapter.split_mode}
            label="Held-out, chosen threshold"
          />
          <HeldoutDice
            value={sweep.heldout_dice_at_default}
            mode={adapter.split_mode}
            label="Held-out, default 0.50"
          />
        </div>

        {hasHeldout ? (
          <p className="m-0 mt-3 text-sm text-slate-700">
            Calibration moved the held-out Dice by{" "}
            <span className="font-semibold tabular-nums">
              {sweep.improvement === null || sweep.improvement === undefined
                ? "an amount that could not be computed"
                : `${sweep.improvement >= 0 ? "+" : ""}${formatNumber(sweep.improvement, 3)}`}
            </span>
            . That is the number worth quoting; the fitted-crop figures above it
            are what the threshold was chosen on.
          </p>
        ) : null}

        <div className="mt-3">
          <OracleCeiling value={sweep.heldout_oracle} />
        </div>

        <div className="mt-4 border-t border-slate-200 pt-4">
          <GroundTruthProvenancePanel
            provenance={provenance}
            loading={provenanceLoading}
            error={provenanceError}
            standalone={false}
          />
        </div>

        {adapter.caveats.length > 0 ? (
          <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-amber-800">
              Read before quoting these numbers
            </p>
            <ul className="m-0 mt-2 list-disc space-y-1 pl-5 text-sm text-amber-900">
              {adapter.caveats.map((caveat) => (
                <li key={caveat}>{caveat}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </Panel>

      <Panel className="p-4">
        <h3 className="m-0 text-sm font-semibold text-slate-900">
          Threshold sweep
        </h3>
        <div className="mt-3">
          <SweepCurve
            sweep={sweep}
            baseSweep={baseSweep}
            splitMode={adapter.split_mode}
            downloadStem={adapter.id}
          />
        </div>
        {!baseSweep && adapter.mode === "head" ? (
          <p className="m-0 mt-2 text-xs text-slate-500">
            The base model&apos;s curve is only available from the job result of
            the run that produced this adapter; it is not stored on the adapter
            itself.
          </p>
        ) : null}
      </Panel>

      {cropNames.length > 0 ? (
        <Panel className="p-4">
          <h3 className="m-0 text-sm font-semibold text-slate-900">
            Per-region Dice at the chosen threshold
          </h3>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[420px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-1 pr-3 font-semibold">Region</th>
                  <th className="py-1 pr-3 font-semibold">Role</th>
                  <th className="py-1 font-semibold">Dice</th>
                </tr>
              </thead>
              <tbody>
                {cropNames.map((name) => (
                  <tr key={name} className="border-b border-slate-100">
                    <td className="py-1 pr-3 text-slate-700">{name}</td>
                    <td className="py-1 pr-3">
                      {trainNames.has(name) ? (
                        <Badge tone="info">threshold fitted on this</Badge>
                      ) : (
                        <Badge tone="good">held out</Badge>
                      )}
                    </td>
                    <td className="py-1 tabular-nums text-slate-900">
                      {formatNumber(sweep.per_crop[name] ?? null, 3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="m-0 mt-3 text-xs text-slate-500">
            A region the threshold was fitted on will score higher than one it
            was not, by construction. Only the held-out rows say anything about
            new data, and only as far as the split mode above allows.
          </p>
        </Panel>
      ) : null}
    </div>
  );
}
