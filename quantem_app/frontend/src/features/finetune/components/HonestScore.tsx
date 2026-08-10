/**
 * The pieces that make a Dice safe to display.
 *
 * `API_CONTRACT.md` honesty rule 1: a held-out score is never shown without its
 * split mode, because `within-image` and `image-disjoint` are different claims
 * and only one of them is about generalising to a new image. Rule 3: the oracle
 * is a ceiling computed with the answers, never a target.
 *
 * These are components rather than format helpers so the two values cannot get
 * separated by a later edit — there is no way to render the number here without
 * also rendering what it means.
 */

import { Badge } from "@/shared/ui/design";
import { formatNumber, NOT_MEASURED } from "@/shared/ui/format";
import type { SplitMode } from "@/shared/types/finetune";
import {
  SPLIT_MODE_LABELS,
  SPLIT_MODE_SENTENCES,
  splitModeTone,
} from "@/features/finetune/splitMode";

export function SplitModeBadge({ mode }: { mode: SplitMode }) {
  return <Badge tone={splitModeTone(mode)}>{SPLIT_MODE_LABELS[mode]}</Badge>;
}

export function SplitModeNote({ mode }: { mode: SplitMode }) {
  return (
    <div className="flex flex-col gap-1">
      <SplitModeBadge mode={mode} />
      <p className="m-0 text-xs text-slate-600">{SPLIT_MODE_SENTENCES[mode]}</p>
    </div>
  );
}

export interface HeldoutDiceProps {
  value: number | null | undefined;
  mode: SplitMode;
  label?: string;
}

/** A held-out Dice and its split mode, inseparably. */
export function HeldoutDice({ value, mode, label = "Held-out Dice" }: HeldoutDiceProps) {
  const missing = value === null || value === undefined || mode === "no-heldout";
  return (
    <div>
      <p className="m-0 text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="m-0 flex items-baseline gap-2">
        <span className="text-lg font-semibold tabular-nums text-slate-900">
          {missing ? NOT_MEASURED : formatNumber(value, 3)}
        </span>
        <SplitModeBadge mode={mode} />
      </p>
    </div>
  );
}

/** The per-crop oracle: what a cheat could reach, stated as the ceiling it is. */
export function OracleCeiling({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return null;
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
      <p className="m-0 text-xs uppercase tracking-wide text-slate-500">
        Oracle ceiling — not a target
      </p>
      <p className="m-0 mt-1 text-sm text-slate-700">
        <span className="font-semibold tabular-nums">{formatNumber(value, 3)}</span>{" "}
        is the best a per-crop threshold could reach if it were chosen using the
        answers. No procedure that has not seen the answers can be expected to
        reach it, so it bounds the numbers above rather than describing what to
        aim for.
      </p>
    </div>
  );
}
