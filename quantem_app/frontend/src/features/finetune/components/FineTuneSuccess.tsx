/** The completed fine-tune, its optional application targets, and live status. */

import { useEffect, useRef, useState } from "react";
import { getFineTuneApplyProgress } from "@/shared/api/finetune";
import { Badge, Button, Panel } from "@/shared/ui/design";
import type { ScopeDatasetNode } from "@/features/finetune/scopeTree";
import type {
  FineTuneApplyImageProgress,
  FineTuneApplyProgress,
  FineTuneApplyResponse,
  FineTuneAppliedImageEvent,
  FineTuneCvPerRoi,
  FineTuneCvResults,
  FineTunePreviewImage,
  FineTuneRunDetail,
} from "@/shared/types/finetune";
import { downloadText } from "@/utils/downloadText";

function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(3);
}

function cvResultsOf(run: FineTuneRunDetail | null): FineTuneCvResults | null {
  const results = run?.cv_results;
  if (!results || !("per_image" in results)) return null;
  return results as FineTuneCvResults;
}

type CvTableRow = Pick<
  FineTuneCvPerRoi,
  "fold" | "roi_name" | "roi_label" | "asset_id" | "name" | "threshold" | "dice" | "iou"
>;

function cvRowsOf(
  run: FineTuneRunDetail | null,
  cv: FineTuneCvResults | null,
  scopedImages: FineTunePreviewImage[]
): CvTableRow[] {
  if (!cv) return [];
  if (cv.per_roi?.length) return cv.per_roi;
  if (cv.per_image?.length) {
    return cv.per_image.map((row, index) => ({
      fold: index,
      roi_name: `image-${index + 1}`,
      roi_label: "All held-out ROIs",
      asset_id: row.asset_id,
      name: row.name,
      threshold: row.threshold ?? null,
      dice: row.dice,
      iou: row.iou,
    }));
  }

  // Compatibility for CV runs saved before per-ROI rows were persisted. A
  // completed CV's final sweep still names every crop: the final round's train
  // set followed by its held-out crop reconstructs the rotation order used by
  // the single-image flow. Per-round thresholds did not survive those builds,
  // so they remain blank rather than borrowing the final round's value.
  const cropNames = [
    ...(run?.train_crop_names ?? []),
    ...(run?.heldout_crop_names ?? []),
  ].filter((value, index, values) => values.indexOf(value) === index);
  const roiNumberByAsset = new Map<string, number>();
  return (cv.folds ?? []).map((fold, index) => {
    const roiName = cropNames[index] ?? `fold-${index + 1}`;
    const prefix = roiName.split("_", 1)[0];
    const image = scopedImages.find((candidate) => candidate.asset_id.startsWith(prefix));
    const assetId = image?.asset_id ?? fold.held_out_asset_id ?? "";
    const nextNumber = (roiNumberByAsset.get(assetId) ?? 0) + 1;
    roiNumberByAsset.set(assetId, nextNumber);
    return {
      fold: fold.fold,
      roi_name: roiName,
      roi_label: `ROI ${nextNumber}`,
      asset_id: assetId,
      name: image?.name ?? "Held-out image",
      threshold: fold.threshold ?? null,
      dice: fold.dice,
      iou: fold.iou,
    };
  });
}

function csvCell(value: string | number | null | undefined): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function fineTuneCvCsv(
  rows: CvTableRow[],
  mean: FineTuneCvResults["mean"]
): string {
  const lines = [
    ["image", "roi", "threshold", "dice", "iou"],
    ...rows.map((row) => [
      row.name,
      row.roi_label,
      row.threshold,
      row.dice,
      row.iou,
    ]),
    ["Average", "", mean.threshold ?? null, mean.dice, mean.iou],
  ];
  return `${lines.map((line) => line.map(csvCell).join(",")).join("\r\n")}\r\n`;
}

function safeFileName(name: string): string {
  const cleaned = name.trim().replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "");
  return cleaned || "fine-tune";
}

function downloadCvCsv(name: string, rows: CvTableRow[], cv: FineTuneCvResults) {
  downloadText(
    `${safeFileName(name)}-cross-validation.csv`,
    fineTuneCvCsv(rows, cv.mean),
    "text/csv"
  );
}

function applyStatusTone(
  status: FineTuneApplyImageProgress["status"]
): "default" | "good" | "warning" | "info" {
  if (status === "SUCCESS") return "good";
  if (status === "FAILED" || status === "CANCELLED") return "warning";
  if (status === "RUNNING") return "info";
  return "default";
}

function ApplyProgressPanel({
  progress,
  current,
  baseModel,
  adapterId,
  segmentationTypeName,
}: {
  progress: FineTuneApplyProgress;
  current: boolean;
  baseModel: string;
  adapterId: string;
  segmentationTypeName: string;
}) {
  return (
    <Panel className="p-3" data-testid="finetune-apply-progress">
      <p className="m-0 text-xs font-semibold text-slate-800">
        {current ? "Current run" : "Latest run"}: {progress.complete} of{" "}
        {progress.total} complete
        {progress.failed > 0 ? `; ${progress.failed} failed` : ""}
      </p>
      {!current && progress.total > 0 ? (
        <p className="m-0 mt-1 text-xs text-slate-600">
          These results are saved on their images. Open one to review its
          probability map and choose the include level.
        </p>
      ) : null}
      <ul className="m-0 mt-2 max-h-56 list-none overflow-y-auto p-0">
        {progress.images.map((image) => {
          const href = segmentationTypeName
            ? `#/assets/${image.asset_id}/labeling/${encodeURIComponent(
                segmentationTypeName
              )}?seg=${encodeURIComponent(image.segmentation_id)}&source_model=${encodeURIComponent(
                baseModel
              )}&adapter_id=${encodeURIComponent(adapterId)}`
            : null;
          return (
            <li
              key={image.job_id}
              className="border-t border-slate-100 py-2 first:border-t-0"
            >
              <div className="flex items-center justify-between gap-2 text-xs">
                {href ? (
                  <a
                    className="truncate text-cyan-700 underline hover:text-cyan-900"
                    href={href}
                    title={`Open ${image.asset_name || image.asset_id}`}
                  >
                    {image.asset_name || image.asset_id}
                  </a>
                ) : (
                  <span className="truncate text-slate-700" title={image.asset_name}>
                    {image.asset_name || image.asset_id}
                  </span>
                )}
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
          );
        })}
      </ul>
    </Panel>
  );
}

export function FineTuneSuccess({
  adapterId,
  name,
  run,
  baseModel,
  segmentationTypeName,
  scopedImages,
  availableDatasets,
  applying,
  applyError,
  applyResult,
  onApply,
  onClose,
  onAppliedImageCompleted,
}: {
  adapterId: string;
  name: string;
  run: FineTuneRunDetail | null;
  baseModel: string;
  segmentationTypeName: string;
  scopedImages: FineTunePreviewImage[];
  availableDatasets: ScopeDatasetNode[];
  applying: boolean;
  applyError: string | null;
  applyResult: FineTuneApplyResponse | null;
  onApply: (assetIds: string[], datasetIds: string[]) => void;
  onClose: () => void;
  onAppliedImageCompleted?: (event: FineTuneAppliedImageEvent) => void;
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
  const [methodologyOpen, setMethodologyOpen] = useState(false);
  const notifiedJobsRef = useRef<Set<string>>(new Set());
  const cv = cvResultsOf(run);
  const cvRows = cvRowsOf(run, cv, scopedImages);

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
    if (!adapterId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const next = await getFineTuneApplyProgress(
          adapterId,
          applyResult?.batch_id
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
  }, [adapterId, applyResult?.batch_id]);

  useEffect(() => {
    if (!applyResult || !applyProgress || !onAppliedImageCompleted) return;
    for (const image of applyProgress.images) {
      if (image.status !== "SUCCESS" || notifiedJobsRef.current.has(image.job_id)) {
        continue;
      }
      notifiedJobsRef.current.add(image.job_id);
      onAppliedImageCompleted({
        adapterId,
        baseModel,
        assetId: image.asset_id,
        segmentationId: image.segmentation_id,
      });
    }
  }, [
    adapterId,
    applyProgress,
    applyResult,
    baseModel,
    onAppliedImageCompleted,
  ]);

  if (applyResult) {
    return (
      <div className="flex flex-col gap-3" data-testid="finetune-applied">
        <p className="m-0 text-sm text-slate-800">
          Queued on {applyResult.queued.length}{" "}
          {applyResult.queued.length === 1 ? "image" : "images"}. Progress is in
          Tasks &amp; Queues and is reported here per image.
        </p>
        {applyProgress ? (
          <ApplyProgressPanel
            progress={applyProgress}
            current
            baseModel={baseModel}
            adapterId={adapterId}
            segmentationTypeName={segmentationTypeName}
          />
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
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="m-0 text-xs font-semibold uppercase tracking-wide text-slate-500">
              Cross-validation
            </p>
            <div className="flex gap-2">
              <Button size="sm" onClick={() => downloadCvCsv(name, cvRows, cv)}>
                Download CSV
              </Button>
              <Button size="sm" onClick={() => setMethodologyOpen(true)}>
                Methodology
              </Button>
            </div>
          </div>
          <div className="mt-3 overflow-x-auto rounded-md border border-slate-200">
            <table className="w-full border-collapse text-left text-xs">
              <thead className="bg-slate-50 text-slate-600">
                <tr>
                  <th className="px-2 py-1.5 font-semibold">Image</th>
                  <th className="px-2 py-1.5 font-semibold">ROI</th>
                  <th className="px-2 py-1.5 text-right font-semibold">Threshold</th>
                  <th className="px-2 py-1.5 text-right font-semibold">Dice</th>
                  <th className="px-2 py-1.5 text-right font-semibold">IoU</th>
                </tr>
              </thead>
              <tbody>
                {cvRows.map((row) => (
                  <tr
                    key={`${row.fold}:${row.roi_name}`}
                    className="border-t border-slate-100 text-slate-700"
                  >
                    <td className="max-w-56 truncate px-2 py-1.5" title={row.name}>
                      {row.name}
                    </td>
                    <td className="whitespace-nowrap px-2 py-1.5">{row.roi_label}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {score(row.threshold)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {score(row.dice)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {score(row.iou)}
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot className="border-t border-slate-300 bg-slate-50 font-semibold text-slate-900">
                <tr>
                  <td className="px-2 py-1.5">Average</td>
                  <td className="px-2 py-1.5">{cv.folds.length} rounds</td>
                  <td className="px-2 py-1.5 text-right tabular-nums">
                    {score(cv.mean?.threshold)}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums">
                    {score(cv.mean?.dice)}
                  </td>
                  <td className="px-2 py-1.5 text-right tabular-nums">
                    {score(cv.mean?.iou)}
                  </td>
                </tr>
              </tfoot>
            </table>
          </div>
        </Panel>
      ) : null}

      {applyProgress && applyProgress.total > 0 ? (
        <ApplyProgressPanel
          progress={applyProgress}
          current={false}
          baseModel={baseModel}
          adapterId={adapterId}
          segmentationTypeName={segmentationTypeName}
        />
      ) : null}

      {methodologyOpen ? (
        <div
          className="finetune-methodology-overlay"
          onClick={() => setMethodologyOpen(false)}
        >
          <div
            className="finetune-methodology-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="finetune-methodology-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3">
              <h3
                id="finetune-methodology-title"
                className="m-0 text-base font-semibold text-slate-950"
              >
                Cross-validation methodology
              </h3>
              <Button
                size="sm"
                variant="ghost"
                aria-label="Close methodology"
                onClick={() => setMethodologyOpen(false)}
              >
                X
              </Button>
            </div>
            <p className="m-0 mt-3 text-sm leading-6 text-slate-700">
              QuantEM repeats training once for each held-out unit. When the
              annotations span multiple images, one whole image is held out per
              round. When they all come from one image, one annotated ROI is held
              out per round; that is a within-image estimate and does not measure
              performance on a new image.
            </p>
            <p className="m-0 mt-3 text-sm leading-6 text-slate-700">
              Each round fits its threshold using only the training ROIs, then
              applies that threshold unchanged to the held-out ROI or image. The
              table reports Dice and IoU for each held-out ROI and averages the
              round scores at the bottom. The saved head is the model from the
              final round; the cross-validation rows describe the round models
              before they were discarded.
            </p>
            <div className="mt-4 flex justify-end">
              <Button variant="primary" onClick={() => setMethodologyOpen(false)}>
                Close
              </Button>
            </div>
          </div>
        </div>
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
