/** The completed fine-tune, its optional application targets, and live status. */

import { useEffect, useState } from "react";
import { getFineTuneApplyProgress } from "@/shared/api/finetune";
import { Badge, Button, Panel } from "@/shared/ui/design";
import type { ScopeDatasetNode } from "@/features/finetune/scopeTree";
import type {
  FineTuneApplyImageProgress,
  FineTuneApplyProgress,
  FineTuneApplyResponse,
  FineTuneCvResults,
  FineTunePreviewImage,
  FineTuneRunDetail,
} from "@/shared/types/finetune";

function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(3);
}

function cvResultsOf(run: FineTuneRunDetail | null): FineTuneCvResults | null {
  const results = run?.cv_results;
  if (!results || !("per_image" in results)) return null;
  return results as FineTuneCvResults;
}

function applyStatusTone(
  status: FineTuneApplyImageProgress["status"]
): "default" | "good" | "warning" | "info" {
  if (status === "SUCCESS") return "good";
  if (status === "FAILED" || status === "CANCELLED") return "warning";
  if (status === "RUNNING") return "info";
  return "default";
}

export function FineTuneSuccess({
  name,
  run,
  scopedImages,
  availableDatasets,
  applying,
  applyError,
  applyResult,
  onApply,
  onClose,
}: {
  name: string;
  run: FineTuneRunDetail | null;
  scopedImages: FineTunePreviewImage[];
  availableDatasets: ScopeDatasetNode[];
  applying: boolean;
  applyError: string | null;
  applyResult: FineTuneApplyResponse | null;
  onApply: (assetIds: string[], datasetIds: string[]) => void;
  onClose: () => void;
}) {
  const [picked, setPicked] = useState<Set<string>>(
    () => new Set(scopedImages.map((image) => image.asset_id))
  );
  const [pickedDatasets, setPickedDatasets] = useState<Set<string>>(
    () => new Set()
  );
  const [applyProgress, setApplyProgress] = useState<FineTuneApplyProgress | null>(
    null
  );
  const [progressError, setProgressError] = useState<string | null>(null);
  const cv = cvResultsOf(run);
  const caveats = run?.caveats ?? [];

  useEffect(() => {
    if (scopedImages.length === 0) return;
    setPicked((current) =>
      current.size === 0
        ? new Set(scopedImages.map((image) => image.asset_id))
        : current
    );
  }, [scopedImages]);

  const toggle = (assetId: string, on: boolean) => {
    setPicked((current) => {
      const next = new Set(current);
      if (on) next.add(assetId);
      else next.delete(assetId);
      return next;
    });
  };

  const toggleDataset = (datasetId: string, on: boolean) => {
    setPickedDatasets((current) => {
      const next = new Set(current);
      if (on) next.add(datasetId);
      else next.delete(datasetId);
      return next;
    });
  };

  useEffect(() => {
    if (!applyResult?.batch_id || !applyResult.adapter_id) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const next = await getFineTuneApplyProgress(
          applyResult.adapter_id,
          applyResult.batch_id
        );
        if (cancelled) return;
        setApplyProgress(next);
        setProgressError(null);
        if (next.complete < next.total) timer = setTimeout(poll, 1000);
      } catch {
        if (cancelled) return;
        setProgressError("Could not refresh per-image progress.");
        timer = setTimeout(poll, 2000);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [applyResult]);

  if (applyResult) {
    return (
      <div className="flex flex-col gap-3" data-testid="finetune-applied">
        <p className="m-0 text-sm text-slate-800">
          Queued on {applyResult.queued.length}{" "}
          {applyResult.queued.length === 1 ? "image" : "images"}. Progress is in
          Tasks &amp; Queues and is reported here per image.
        </p>
        {applyProgress ? (
          <Panel className="p-3" data-testid="finetune-apply-progress">
            <p className="m-0 text-xs font-semibold text-slate-800">
              {applyProgress.complete} of {applyProgress.total} complete
              {applyProgress.failed > 0
                ? `; ${applyProgress.failed} failed`
                : ""}
            </p>
            <ul className="m-0 mt-2 max-h-56 list-none overflow-y-auto p-0">
              {applyProgress.images.map((image) => (
                <li
                  key={image.job_id}
                  className="border-t border-slate-100 py-2 first:border-t-0"
                >
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="truncate text-slate-700" title={image.asset_name}>
                      {image.asset_name || image.asset_id}
                    </span>
                    <Badge tone={applyStatusTone(image.status)}>
                      {image.status.toLowerCase()}
                    </Badge>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded bg-slate-100">
                    <div
                      className="h-full bg-cyan-600"
                      style={{ width: `${Math.max(0, Math.min(100, image.progress))}%` }}
                      aria-label={`${image.asset_name || image.asset_id} progress`}
                    />
                  </div>
                  <p className="m-0 mt-1 text-xs text-slate-500">
                    {image.units_done !== null && image.units_total !== null
                      ? `${image.units_done} of ${image.units_total} tiles`
                      : `${Math.round(image.progress)}%`}
                    {image.stage ? ` - ${image.stage}` : ""}
                  </p>
                  {image.failure ? (
                    <p className="m-0 mt-1 text-xs text-red-700" role="alert">
                      {image.failure}
                    </p>
                  ) : null}
                </li>
              ))}
            </ul>
          </Panel>
        ) : (
          <p className="m-0 text-xs text-slate-500">Loading per-image progress...</p>
        )}
        {progressError ? (
          <p className="m-0 text-xs text-red-700" role="alert">
            {progressError}
          </p>
        ) : null}
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
                    Dice {score(row.dice)} / IoU {score(row.iou)} / {row.n_tiles}{" "}
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

      {availableDatasets.length > 0 ? (
        <div>
          <div className="flex items-baseline justify-between gap-2">
            <p className="m-0 text-sm font-semibold text-slate-900">
              Or run it across a Dataset
            </p>
            <button
              type="button"
              className="text-xs text-cyan-700 underline hover:text-cyan-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              onClick={() =>
                setPickedDatasets((current) =>
                  current.size === availableDatasets.length
                    ? new Set()
                    : new Set(availableDatasets.map((dataset) => dataset.id))
                )
              }
            >
              {pickedDatasets.size === availableDatasets.length
                ? "Clear datasets"
                : "Select all datasets"}
            </button>
          </div>
          <p className="m-0 mt-1 text-xs text-slate-600">
            Datasets are the existing groups in this Experiment. Every active
            image in a selected Dataset is queued, including images that were
            not part of training.
          </p>
          <ul
            className="m-0 mt-2 max-h-32 list-none overflow-y-auto rounded-md border border-slate-200 p-2"
            data-testid="finetune-apply-datasets"
          >
            {availableDatasets.map((dataset) => (
              <li key={dataset.id} className="py-0.5">
                <label className="flex items-center justify-between gap-2 text-xs text-slate-700">
                  <span className="flex min-w-0 items-center gap-2">
                    <input
                      type="checkbox"
                      className="h-3.5 w-3.5"
                      checked={pickedDatasets.has(dataset.id)}
                      onChange={(event) =>
                        toggleDataset(dataset.id, event.target.checked)
                      }
                    />
                    <span className="truncate" title={dataset.name}>
                      {dataset.name}
                    </span>
                  </span>
                  <span className="shrink-0 text-slate-500">
                    {dataset.imageCount} images
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {applyError ? (
        <p className="m-0 text-sm text-red-700" role="alert">
          {applyError}
        </p>
      ) : null}

      <div className="flex justify-end gap-2">
        <Button onClick={onClose}>Not now</Button>
        <Button
          variant="primary"
          disabled={(picked.size === 0 && pickedDatasets.size === 0) || applying}
          onClick={() => onApply([...picked], [...pickedDatasets])}
        >
          {applying
            ? "Queueing..."
            : pickedDatasets.size > 0
              ? `Run ${picked.size} selected images + ${pickedDatasets.size} ${
                  pickedDatasets.size === 1 ? "Dataset" : "Datasets"
                }`
              : `Run on ${picked.size} ${picked.size === 1 ? "image" : "images"}`}
        </Button>
      </div>
    </div>
  );
}
