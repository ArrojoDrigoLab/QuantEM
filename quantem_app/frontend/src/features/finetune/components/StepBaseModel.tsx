/**
 * Step 1 — which released model to start from.
 *
 * When `GET /api/models/` is unavailable the ids are still offered (they are
 * fixed by the contract) but the card says install state is unknown instead of
 * claiming "not installed". Guessing there would send someone to download 660 MB
 * they already have, or start a run that cannot find its weights.
 */

import { Link } from "react-router-dom";
import { Panel } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { formatBytes, formatNumber, formatTimestamp } from "@/shared/ui/format";
import type { AdaptedModelEntry, ModelCatalogue } from "@/shared/types/finetune";
import type { ModelChoice } from "@/features/finetune/models";
import { SplitModeBadge } from "@/features/finetune/components/HonestScore";
import { noPackIsRunnable, packRunnability } from "@/features/models/runnable";
import {
  RunnabilityBadge,
  RunnabilityReason,
} from "@/features/models/components/RunnabilityBadge";

export interface StepBaseModelProps {
  choices: ModelChoice[];
  catalogue: ModelCatalogue | null;
  catalogueError: string | null;
  value: string;
  onChange: (packId: string) => void;
  adapted: AdaptedModelEntry[];
}

export function StepBaseModel({
  choices,
  catalogue,
  catalogueError,
  value,
  onChange,
  adapted,
}: StepBaseModelProps) {
  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-4">
        <h2 className="m-0 text-base font-semibold text-slate-950">
          Choose a base model
        </h2>
        <p className="m-0 mt-1 text-sm text-slate-600">
          Fine-tuning starts from one of the released packs. Its encoder stays
          frozen; only the small head on top is fitted to your annotations.
        </p>

        {catalogueError ? (
          <p className="m-0 mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
            The model catalogue did not answer ({catalogueError}). The packs
            below are the released ids from the API contract; whether each one is
            installed, and how large a download it would need, is unknown until
            the catalogue is reachable.
          </p>
        ) : null}

        {catalogue?.device ? (
          <p className="m-0 mt-3 text-xs text-slate-500">
            Device: {catalogue.device.name}
            {catalogue.device.cuda ? " (CUDA)" : catalogue.device.mps ? " (MPS)" : ""}
          </p>
        ) : null}

        {noPackIsRunnable(catalogue) ? (
          // The clean-install case. Without this the wizard happily walked the
          // user to step 4 and the run died on a missing encoder.
          <div className="m-0 mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2">
            <p className="m-0 text-xs font-semibold text-amber-900">
              None of these packs can run on this machine yet.
            </p>
            <p className="m-0 mt-1 text-xs text-amber-900">
              Each card says why below.{" "}
              <Link className="underline" to="/models">
                Install one from the models screen
              </Link>{" "}
              to train a head. Threshold calibration also needs a stored
              probability map, which only a completed run produces.
            </p>
          </div>
        ) : null}

        <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
          {choices.map((choice) => {
            const selected = choice.id === value;
            const runnability = packRunnability(choice.pack);
            const blocked = runnability.state === "blocked";
            return (
              <button
                key={choice.id}
                type="button"
                // Still selectable when blocked: the reason is often "not
                // installed yet", and letting the user select it is how they
                // find the install route. What must not happen is the wizard
                // pretending it will work -- hence the badge, the reason, and
                // the block on starting a run in StepRun.
                onClick={() => onChange(choice.id)}
                aria-describedby={blocked ? `${choice.id}-reason` : undefined}
                className={cx(
                  "rounded-md border px-3 py-2 text-left transition-colors",
                  selected
                    ? "border-cyan-500 bg-cyan-50"
                    : blocked
                      ? "border-slate-200 bg-slate-50 hover:bg-slate-100"
                      : "border-slate-200 bg-white hover:bg-slate-50"
                )}
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span
                    className={cx(
                      "text-sm font-medium",
                      blocked ? "text-slate-600" : "text-slate-900"
                    )}
                  >
                    {choice.title}
                  </span>
                  <RunnabilityBadge runnability={runnability} />
                </div>
                <p className="m-0 mt-1 font-mono text-xs text-slate-500">
                  {choice.id}
                </p>
                <div id={blocked ? `${choice.id}-reason` : undefined}>
                  <RunnabilityReason
                    runnability={runnability}
                    className="m-0 mt-1 text-xs text-amber-800"
                  />
                </div>
                {choice.pack ? (
                  <p className="m-0 mt-1 text-xs text-slate-500">
                    {choice.pack.installed
                      ? "installed"
                      : `${formatBytes(choice.pack.download_bytes)} to install`}{" "}
                    · {choice.pack.neck} · {choice.pack.decoder} · tile{" "}
                    {choice.pack.tile_size} ·{" "}
                    {choice.pack.canonical_nm === null
                      ? "native resolution"
                      : `${formatNumber(choice.pack.canonical_nm, 1)} nm/px`}
                  </p>
                ) : (
                  <p className="m-0 mt-1 text-xs text-slate-500">
                    Install state unknown — the catalogue did not mention this
                    pack.
                  </p>
                )}
              </button>
            );
          })}
        </div>
      </Panel>

      {adapted.length > 0 ? (
        <Panel className="p-4">
          <h3 className="m-0 text-sm font-semibold text-slate-900">
            Already adapted from this base
          </h3>
          <ul className="m-0 mt-2 flex list-none flex-col gap-2 p-0">
            {adapted.map((entry) => (
              <li
                key={entry.id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-slate-200 px-3 py-2"
              >
                <div>
                  <p className="m-0 text-sm text-slate-900">
                    {entry.name || entry.id}
                  </p>
                  <p className="m-0 text-xs text-slate-500">
                    {formatTimestamp(entry.created_at)} · threshold{" "}
                    {formatNumber(entry.calibrated_threshold, 2)}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm tabular-nums text-slate-900">
                    Dice {formatNumber(entry.heldout_dice, 3)}
                  </span>
                  <SplitModeBadge mode={entry.split_mode} />
                </div>
              </li>
            ))}
          </ul>
        </Panel>
      ) : null}
    </div>
  );
}
