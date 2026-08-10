/**
 * Step 6 — save and apply.
 *
 * The adapter row is already saved by the time the job finishes; "apply" is the
 * separate decision to use it for subsequent runs on this segmentation. Keeping
 * them apart means a run that produced a worse held-out score can be inspected
 * and left alone rather than silently becoming the model.
 */

import { Badge, Button, Panel } from "@/shared/ui/design";
import { formatNumber, formatTimestamp } from "@/shared/ui/format";
import type { Adapter } from "@/shared/types/finetune";
import { SplitModeBadge } from "@/features/finetune/components/HonestScore";

export interface StepApplyProps {
  adapter: Adapter;
  onApply: () => void;
  applying: boolean;
  applyError: string | null;
  /** Where to go once the adapter is in use. */
  segmentationHref: string;
}

export function StepApply({
  adapter,
  onApply,
  applying,
  applyError,
  segmentationHref,
}: StepApplyProps) {
  const applied = Boolean(adapter.applied_at);

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="m-0 text-base font-semibold text-slate-950">
          Use this adapter
        </h2>
        {applied ? (
          <Badge tone="good">applied {formatTimestamp(adapter.applied_at)}</Badge>
        ) : (
          <Badge>not applied</Badge>
        )}
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Base model
          </dt>
          <dd className="m-0 font-mono text-sm text-slate-900">
            {adapter.base_model}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Threshold
          </dt>
          <dd className="m-0 text-sm tabular-nums text-slate-900">
            {formatNumber(adapter.calibrated_threshold, 2)}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Held-out Dice
          </dt>
          <dd className="m-0 flex items-center gap-2 text-sm tabular-nums text-slate-900">
            {adapter.split_mode === "no-heldout"
              ? "—"
              : formatNumber(adapter.heldout_dice, 3)}
            <SplitModeBadge mode={adapter.split_mode} />
          </dd>
        </div>
      </dl>

      <p className="m-0 mt-3 text-sm text-slate-600">
        Applying makes this the model used for subsequent runs on this
        segmentation. It changes nothing that has already been produced: existing
        objects, probability maps and analysis runs keep the model they were made
        with.
      </p>

      {applyError ? (
        <p className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {applyError}
        </p>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          onClick={onApply}
          disabled={applying || adapter.status !== "SUCCESS"}
        >
          {applying ? "Applying…" : applied ? "Apply again" : "Apply this adapter"}
        </Button>
        <a
          className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50"
          href={segmentationHref}
        >
          Back to proofreading
        </a>
      </div>
    </Panel>
  );
}
