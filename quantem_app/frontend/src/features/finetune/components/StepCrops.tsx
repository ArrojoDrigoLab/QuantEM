/**
 * Step 2 — what has been annotated, and whether it is enough.
 *
 * The readiness verdict comes from the server, not from counting crops here:
 * `blockers` is the API telling the UI it must not proceed. The one that
 * matters is the completed ROI — inside it, anything that is not a confirmed
 * object is true background; without it, "background" and "not looked at yet"
 * are the same pixels and every Dice on the later steps would be a fiction.
 *
 * Honesty rule 2: the crops the threshold will be fitted on are badged here,
 * before the run, not only in the results table afterwards.
 */

import { Badge, Button, Panel } from "@/shared/ui/design";
import { formatInteger } from "@/shared/ui/format";
import {
  CONFIRMED_AREA_HOW_TO,
  CONFIRMED_AREA_LABEL,
  ROI_REVIEWED_LABEL,
} from "@/shared/constants/confirmedArea";
import type { AdaptCropsResponse } from "@/shared/types/finetune";
import { SplitModeNote } from "@/features/finetune/components/HonestScore";

/**
 * The blocker whose advice was, read literally, "do the thing you just did".
 *
 * The server's text is *"No completed ROI on this image. Mark the area you have
 * finished annotating as complete."* -- and the labeling screen has a tick box
 * that used to be called **Done (ER)**, which marks a ROI window as finished
 * and does not create a `CompletedROI`. Ticking it and coming here got you this
 * sentence. The wording on the labeling screen is fixed at the source (the tick
 * box is "{ROI_REVIEWED_LABEL}" now), and this is the other end: the blocker
 * names the API term, so the screen has to say which control produces one.
 *
 * Matched on the phrase rather than on a code, because `blockers` is a list of
 * sentences; the note is additive, so a miss loses an explanation and never
 * hides the blocker itself.
 */
function needsConfirmedArea(blockers: string[]): boolean {
  return blockers.some((blocker) => /completed (roi|area)/i.test(blocker));
}

export interface StepCropsProps {
  crops: AdaptCropsResponse | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export function StepCrops({ crops, loading, error, onRefresh }: StepCropsProps) {
  if (loading && !crops) {
    return (
      <Panel className="p-4">
        <p className="m-0 text-sm text-slate-600">Reading your annotations…</p>
      </Panel>
    );
  }

  if (error) {
    return (
      <Panel className="border-red-200 bg-red-50 p-4">
        <p className="m-0 text-sm text-red-800">{error}</p>
        <div className="mt-3">
          <Button size="sm" onClick={onRefresh}>
            Try again
          </Button>
        </div>
      </Panel>
    );
  }

  if (!crops) return null;

  const trainNames = new Set(crops.train_crop_names);
  const heldoutNames = new Set(crops.heldout_crop_names);
  const withoutProbability = crops.crops.filter(
    (crop) => crop.has_probability === false
  ).length;

  return (
    <div className="flex flex-col gap-4">
      <Panel className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="m-0 text-base font-semibold text-slate-950">
            Your annotated regions
          </h2>
          <div className="flex items-center gap-2">
            <Badge tone={crops.ready ? "good" : "warning"}>
              {crops.ready ? "ready" : "not ready"}
            </Badge>
            <Button size="sm" onClick={onRefresh}>
              Refresh
            </Button>
          </div>
        </div>

        <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Regions
            </dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {crops.crops.length}
            </dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Images
            </dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {crops.n_images}
            </dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Will fit on
            </dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {crops.train_crop_names.length}
            </dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Held out
            </dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {crops.heldout_crop_names.length}
            </dd>
          </div>
        </dl>

        <div className="mt-3">
          <SplitModeNote mode={crops.split_mode} />
        </div>

        {crops.blockers.length > 0 ? (
          <div className="mt-3 rounded-md border border-red-200 bg-red-50 p-3">
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-red-800">
              Cannot proceed
            </p>
            <ul className="m-0 mt-2 list-disc space-y-1 pl-5 text-sm text-red-900">
              {crops.blockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
            {needsConfirmedArea(crops.blockers) ? (
              <p className="m-0 mt-2 text-sm text-red-900">
                <strong>&ldquo;Completed ROI&rdquo; means one specific shape.</strong>{" "}
                It is the polygon the <em>{CONFIRMED_AREA_LABEL}</em> tool draws
                on the labeling screen. {CONFIRMED_AREA_HOW_TO} The per-ROI
                &ldquo;{ROI_REVIEWED_LABEL}&rdquo; tick box beside the ER ROI
                list is a different thing: it records your own progress and does
                not satisfy this.
              </p>
            ) : null}
          </div>
        ) : null}

        {crops.warnings.length > 0 ? (
          <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-amber-800">
              Worth knowing before you run
            </p>
            <ul className="m-0 mt-2 list-disc space-y-1 pl-5 text-sm text-amber-900">
              {crops.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {withoutProbability > 0 ? (
          <p className="m-0 mt-3 text-xs text-amber-700">
            {withoutProbability} region
            {withoutProbability === 1 ? " has" : "s have"} no probability map.
            Threshold calibration can only use regions the model has already been
            run over; head training predicts its own.
          </p>
        ) : null}
      </Panel>

      <Panel className="p-4">
        <h3 className="m-0 text-sm font-semibold text-slate-900">Regions</h3>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="py-1 pr-3 font-semibold">Region</th>
                <th className="py-1 pr-3 font-semibold">Role</th>
                <th className="py-1 pr-3 font-semibold">Image</th>
                <th className="py-1 pr-3 font-semibold">Size</th>
                <th className="py-1 pr-3 font-semibold">Objects</th>
                <th className="py-1 pr-3 font-semibold">Annotated px</th>
                <th className="py-1 font-semibold">Prob. map</th>
              </tr>
            </thead>
            <tbody>
              {crops.crops.map((crop) => (
                <tr key={crop.id} className="border-b border-slate-100">
                  <td className="py-1 pr-3 text-slate-700">{crop.name}</td>
                  <td className="py-1 pr-3">
                    {trainNames.has(crop.name) ? (
                      <Badge tone="info">fitted on</Badge>
                    ) : heldoutNames.has(crop.name) ? (
                      <Badge tone="good">held out</Badge>
                    ) : (
                      <Badge>unused</Badge>
                    )}
                  </td>
                  <td className="py-1 pr-3 font-mono text-xs text-slate-500">
                    {crop.image_key.slice(0, 8)}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {crop.width}×{crop.height}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatInteger(crop.n_objects)}
                  </td>
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatInteger(crop.annotated_px)}
                  </td>
                  <td className="py-1">
                    {crop.has_probability === false ? (
                      <span className="text-xs text-amber-700">none</span>
                    ) : (
                      <span className="text-xs text-slate-500">present</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="m-0 mt-3 text-xs text-slate-500">
          Inside a completed region, anything that is not a confirmed object
          counts as background; everything outside it is ignored entirely. That
          is what makes a Dice on these crops mean something.
        </p>
      </Panel>
    </div>
  );
}
