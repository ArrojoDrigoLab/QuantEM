/**
 * What happens after a fine-tune finishes: the result, and the offer.
 *
 * The offer is the point. Owner R13: a success message, then **the option** to
 * run the new model on some or all of the images it was scoped to — never
 * automatic. So nothing on this panel queues anything until the button is
 * pressed, the images start out picked but the pick is editable, and the button
 * says how many images it is about to run on.
 *
 * The scores, when cross-validation ran, are shown per image as well as
 * averaged. A mean over four annotations hides the image that scored 0.2, and
 * that image is the one worth looking at.
 */

import { useState } from "react";
import { Badge, Button, Panel } from "@/shared/ui/design";
import type {
  FineTuneApplyResponse,
  FineTuneCvResults,
  FineTunePreviewImage,
  FineTuneRunDetail,
} from "@/shared/types/finetune";

function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toFixed(3);
}

function cvResultsOf(run: FineTuneRunDetail | null): FineTuneCvResults | null {
  const results = run?.cv_results;
  if (!results || !("per_image" in results)) return null;
  return results as FineTuneCvResults;
}

export function FineTuneSuccess({
  name,
  run,
  scopedImages,
  applying,
  applyError,
  applyResult,
  onApply,
  onClose,
}: {
  name: string;
  run: FineTuneRunDetail | null;
  scopedImages: FineTunePreviewImage[];
  applying: boolean;
  applyError: string | null;
  applyResult: FineTuneApplyResponse | null;
  onApply: (assetIds: string[]) => void;
  onClose: () => void;
}) {
  const [picked, setPicked] = useState<Set<string>>(
    () => new Set(scopedImages.map((image) => image.asset_id))
  );
  const cv = cvResultsOf(run);
  const caveats = run?.caveats ?? [];

  const toggle = (assetId: string, on: boolean) => {
    setPicked((current) => {
      const next = new Set(current);
      if (on) next.add(assetId);
      else next.delete(assetId);
      return next;
    });
  };

  if (applyResult) {
    return (
      <div className="flex flex-col gap-3" data-testid="finetune-applied">
        <p className="m-0 text-sm text-slate-800">
          Queued on {applyResult.queued.length}{" "}
          {applyResult.queued.length === 1 ? "image" : "images"}. Progress is in
          Tasks &amp; Queues, and each image shows it too.
        </p>
        <div className="flex justify-end">
          <Button variant="primary" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4" data-testid="finetune-success">
      <div className="flex items-center gap-2">
        <Badge tone="good">Finished</Badge>
        <p className="m-0 text-sm text-slate-800">
          &ldquo;{name}&rdquo; is trained and saved.
        </p>
      </div>

      {cv ? (
        <Panel className="p-3" data-testid="finetune-cv-results">
          <p className="m-0 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Cross-validation
          </p>
          <p className="m-0 mt-1 text-sm text-slate-800">
            Average Dice {score(cv.mean?.dice)}, IoU {score(cv.mean?.iou)} over{" "}
            {cv.folds?.length ?? 0} rounds.
          </p>
          {cv.per_image?.length ? (
            <ul className="m-0 mt-2 list-none p-0">
              {cv.per_image.map((row) => (
                <li
                  key={row.asset_id}
                  className="flex items-center justify-between gap-3 py-0.5 text-xs text-slate-700"
                >
                  <span className="truncate" title={row.name}>
                    {row.name}
                  </span>
                  <span className="shrink-0 tabular-nums">
                    Dice {score(row.dice)} · IoU {score(row.iou)} · {row.n_tiles}{" "}
                    {row.n_tiles === 1 ? "tile" : "tiles"}
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </Panel>
      ) : null}

      {caveats.length > 0 ? (
        <ul className="m-0 list-disc pl-5 text-xs text-amber-800">
          {caveats.map((caveat) => (
            <li key={caveat}>{caveat}</li>
          ))}
        </ul>
      ) : null}

      <div>
        <div className="flex items-baseline justify-between gap-2">
          <p className="m-0 text-sm font-semibold text-slate-900">
            Run it on these images?
          </p>
          <button
            type="button"
            className="text-xs text-cyan-700 underline hover:text-cyan-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            onClick={() =>
              setPicked((current) =>
                current.size === scopedImages.length
                  ? new Set()
                  : new Set(scopedImages.map((image) => image.asset_id))
              )
            }
          >
            {picked.size === scopedImages.length ? "Clear all" : "Select all"}
          </button>
        </div>
        <p className="m-0 mt-1 text-xs text-slate-600">
          Optional, and nothing has been queued. Anything you drew, confirmed or
          removed on these images stays exactly as it is.
        </p>
        <ul
          className="m-0 mt-2 max-h-40 list-none overflow-y-auto rounded-md border border-slate-200 p-2"
          data-testid="finetune-apply-images"
        >
          {scopedImages.map((image) => (
            <li key={image.asset_id} className="py-0.5">
              <label className="flex items-center gap-2 text-xs text-slate-700">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5"
                  checked={picked.has(image.asset_id)}
                  onChange={(event) => toggle(image.asset_id, event.target.checked)}
                />
                <span className="truncate" title={image.name}>
                  {image.name}
                </span>
              </label>
            </li>
          ))}
        </ul>
      </div>

      {applyError ? (
        <p className="m-0 text-sm text-red-700" role="alert">
          {applyError}
        </p>
      ) : null}

      <div className="flex justify-end gap-2">
        <Button onClick={onClose}>Not now</Button>
        <Button
          variant="primary"
          disabled={picked.size === 0 || applying}
          onClick={() => onApply([...picked])}
        >
          {applying
            ? "Queueing…"
            : `Run on ${picked.size} ${picked.size === 1 ? "image" : "images"}`}
        </Button>
      </div>
    </div>
  );
}
