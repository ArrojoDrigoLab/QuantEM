/**
 * Configure one analysis run.
 *
 * Two design choices worth stating. Compartment *names* are editable because
 * they become the exported column headers (`area_fraction_mito`), and a user
 * analysing a segmentation type they invented needs to be able to say what to
 * call it. And the tissue mask defaults to the tissue segmentation when one
 * exists, because "no tissue mask" silently changes the denominator of every
 * fraction on the results screen.
 */

import type { ChangeEvent } from "react";
import { Button, Panel } from "@/shared/ui/design";
import { describeObjectsPixelSize } from "@/shared/objectsPixelSize";
import type { ImageSegmentation } from "@/shared/types/images";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import {
  MAX_REPLICATES,
  parseBandEdges,
  replicatesError,
  seedError,
  type AnalysisFormState,
  type PointsSourceChoice,
} from "@/features/analysis/analysisOptions";

const FIELD_CLASS =
  "h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500 disabled:bg-slate-100 disabled:text-slate-500";
const LABEL_CLASS =
  "block text-xs font-semibold uppercase tracking-wide text-slate-500";

export interface AnalysisConfigFormProps {
  segmentations: ImageSegmentation[];
  /**
   * The segmentation this run is recorded against — the one whose objects the
   * bundle measures. Its `objects_pixel_size` decides the warning above the
   * Run button; null (or an older backend) renders no warning.
   */
  selectedSegmentation?: ImageSegmentation | null;
  state: AnalysisFormState;
  onChange: (next: AnalysisFormState) => void;
  onSubmit: () => void;
  submitting: boolean;
  error: string | null;
}

export function AnalysisConfigForm({
  segmentations,
  selectedSegmentation = null,
  state,
  onChange,
  onSubmit,
  submitting,
  error,
}: AnalysisConfigFormProps) {
  const patch = (updates: Partial<AnalysisFormState>) => {
    onChange({ ...state, ...updates });
  };

  const enabledNames = state.compartments
    .filter((entry) => entry.enabled)
    .map((entry) => entry.name.trim())
    .filter(Boolean);
  const bandParse = parseBandEdges(state.bandEdgesText);
  /**
   * Validated where they are typed, not only where they are submitted.
   *
   * The band-edge field has always done this and the numeric ones had not, so
   * `min={1} max={1000}` on Replicates was decorative: 20000 typed into it
   * looked accepted right up until Run analysis quietly refused to fire, with
   * the only explanation rendered below the fold beside a button that had just
   * moved down the page to make room for it.
   */
  const replicatesMessage = replicatesError(state.replicates);
  const seedMessage = seedError(state.seed);
  /**
   * Said before the run is spent, not in the finished bundle.
   *
   * `run_analysis` blanks every physical unit when the objects predate the
   * image's calibration, and until this notice existed the first place that
   * said so was the bundle itself — blank micron columns and
   * `calibrated: false`, minutes after the click this notice now sits above.
   * The verdict is the server's (`objects_pixel_size.predates_calibration`);
   * these are the same sentences the labeling header's chip carries.
   */
  const objectsPixelSize = describeObjectsPixelSize(selectedSegmentation);

  const segmentationLabel = (segmentationId: string): string => {
    const seg = segmentations.find((entry) => entry.id === segmentationId);
    return seg?.segmentation_type.long_name ?? segmentationId;
  };

  const handleCsvFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    patch({ pointsCsv: text, pointsSource: "csv" });
    event.target.value = "";
  };

  return (
    <Panel className="p-4">
      <h2 className="m-0 text-base font-semibold text-slate-950">
        Configure the run
      </h2>

      <fieldset className="mt-4 border-0 p-0">
        <legend className={LABEL_CLASS}>Compartments</legend>
        <p className="m-0 mt-1 text-xs text-slate-500">
          Each one is rasterised at full resolution and becomes a column in the
          export. The name is the column header.
        </p>
        <div className="mt-2 flex flex-col gap-2">
          {state.compartments.length === 0 ? (
            <p className="m-0 text-sm text-slate-500">
              This image has no segmentations to measure.
            </p>
          ) : null}
          {state.compartments.map((entry, index) => (
            <div key={entry.segmentationId} className="flex items-center gap-2">
              <input
                type="checkbox"
                id={`compartment-${entry.segmentationId}`}
                className="h-4 w-4"
                checked={entry.enabled}
                disabled={submitting}
                onChange={(event) => {
                  const next = [...state.compartments];
                  next[index] = { ...entry, enabled: event.target.checked };
                  patch({ compartments: next });
                }}
              />
              <label
                htmlFor={`compartment-${entry.segmentationId}`}
                className="min-w-0 flex-1 truncate text-sm text-slate-700"
                title={segmentationLabel(entry.segmentationId)}
              >
                {segmentationLabel(entry.segmentationId)}
              </label>
              <input
                type="text"
                aria-label={`Name for ${segmentationLabel(entry.segmentationId)}`}
                className={`${FIELD_CLASS} w-28`}
                value={entry.name}
                disabled={submitting || !entry.enabled}
                onChange={(event) => {
                  const next = [...state.compartments];
                  next[index] = { ...entry, name: event.target.value };
                  patch({ compartments: next });
                }}
              />
            </div>
          ))}
        </div>
      </fieldset>

      <div className="mt-4">
        <label className={LABEL_CLASS} htmlFor="analysis-tissue">
          Tissue mask
        </label>
        <select
          id="analysis-tissue"
          className={`${FIELD_CLASS} mt-1`}
          value={state.tissueSegmentationId}
          disabled={submitting}
          onChange={(event) => patch({ tissueSegmentationId: event.target.value })}
        >
          <option value="">Whole image (no tissue mask)</option>
          {segmentations.map((seg) => (
            <option key={seg.id} value={seg.id}>
              {segmentationDisplayName(seg)}
            </option>
          ))}
        </select>
        {state.tissueSegmentationId === "" ? (
          <p className="m-0 mt-1 text-xs text-amber-700">
            Without a tissue mask every fraction is relative to the whole image,
            including empty resin. The run will say so in its caveats.
          </p>
        ) : null}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div>
          <label className={LABEL_CLASS} htmlFor="analysis-points-source">
            Point source
          </label>
          <select
            id="analysis-points-source"
            className={`${FIELD_CLASS} mt-1`}
            value={state.pointsSource}
            disabled={submitting}
            onChange={(event) =>
              patch({
                pointsSource: event.target.value as PointsSourceChoice,
                distanceTarget:
                  event.target.value === "none" ? "" : state.distanceTarget,
              })
            }
          >
            <option value="none">None</option>
            <option value="centroids">Object centroids</option>
            <option value="csv">Imported x,y CSV</option>
          </select>
        </div>
        <div>
          <label className={LABEL_CLASS} htmlFor="analysis-distance-target">
            Distance target
          </label>
          <select
            id="analysis-distance-target"
            className={`${FIELD_CLASS} mt-1`}
            value={state.distanceTarget}
            disabled={submitting || state.pointsSource === "none"}
            onChange={(event) => patch({ distanceTarget: event.target.value })}
          >
            <option value="">None</option>
            {enabledNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {state.pointsSource === "csv" ? (
        <div className="mt-3">
          <label className={LABEL_CLASS} htmlFor="analysis-points-csv">
            Points CSV (x,y in image pixels)
          </label>
          <textarea
            id="analysis-points-csv"
            className="mt-1 h-28 w-full rounded-md border border-slate-300 bg-white p-2 font-mono text-xs text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            value={state.pointsCsv}
            disabled={submitting}
            placeholder={"x,y\n1024,512\n980,640"}
            onChange={(event) => patch({ pointsCsv: event.target.value })}
          />
          <input
            type="file"
            accept=".csv,text/csv"
            aria-label="Load points CSV"
            className="mt-1 text-xs text-slate-600"
            disabled={submitting}
            onChange={(event) => {
              void handleCsvFile(event);
            }}
          />
        </div>
      ) : null}

      {state.pointsSource === "centroids" ? (
        <p className="m-0 mt-2 text-xs text-slate-500">
          Centroids come from the confirmed objects of this segmentation. If that
          same organelle is also one of the compartments above, its enrichment is
          1 / area fraction by construction — the run names it in its caveats.
        </p>
      ) : null}

      <div className="mt-4">
        <label className={LABEL_CLASS} htmlFor="analysis-bands">
          Distance band edges (nm)
        </label>
        <input
          id="analysis-bands"
          type="text"
          className={`${FIELD_CLASS} mt-1`}
          value={state.bandEdgesText}
          disabled={submitting}
          onChange={(event) => patch({ bandEdgesText: event.target.value })}
        />
        {bandParse.error ? (
          <p className="m-0 mt-1 text-xs text-red-700">{bandParse.error}</p>
        ) : null}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div>
          <label className={LABEL_CLASS} htmlFor="analysis-replicates">
            Replicates
          </label>
          <input
            id="analysis-replicates"
            type="number"
            min={1}
            max={MAX_REPLICATES}
            className={`${FIELD_CLASS} mt-1`}
            value={state.replicates}
            disabled={submitting}
            aria-invalid={replicatesMessage !== null}
            aria-describedby="analysis-replicates-error"
            onChange={(event) =>
              patch({ replicates: Number.parseInt(event.target.value, 10) || 0 })
            }
          />
          <p
            id="analysis-replicates-error"
            className="m-0 mt-1 text-xs text-red-700"
            role={replicatesMessage ? "alert" : undefined}
          >
            {replicatesMessage}
          </p>
        </div>
        <div>
          <label className={LABEL_CLASS} htmlFor="analysis-seed">
            Seed
          </label>
          <input
            id="analysis-seed"
            type="number"
            className={`${FIELD_CLASS} mt-1`}
            value={state.seed}
            disabled={submitting}
            aria-invalid={seedMessage !== null}
            aria-describedby="analysis-seed-error"
            onChange={(event) =>
              patch({ seed: Number.parseInt(event.target.value, 10) || 0 })
            }
          />
          <p
            id="analysis-seed-error"
            className="m-0 mt-1 text-xs text-red-700"
            role={seedMessage ? "alert" : undefined}
          >
            {seedMessage}
          </p>
        </div>
        <div>
          <label className={LABEL_CLASS} htmlFor="analysis-group">
            Group label
          </label>
          <input
            id="analysis-group"
            type="text"
            className={`${FIELD_CLASS} mt-1`}
            value={state.group}
            placeholder="e.g. fasted"
            disabled={submitting}
            onChange={(event) => patch({ group: event.target.value })}
          />
        </div>
      </div>

      {objectsPixelSize ? (
        <div
          className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2"
          role="status"
          data-testid="analysis-objects-pixel-size"
        >
          <p className="m-0 text-sm font-semibold text-amber-900">
            {objectsPixelSize.summary}
          </p>
          <p className="m-0 mt-1 text-sm text-amber-900">
            {objectsPixelSize.detail} {objectsPixelSize.consequence}
          </p>
        </div>
      ) : null}

      {error ? (
        <p className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      ) : null}

      <div className="mt-4 flex items-center gap-3">
        <Button variant="primary" onClick={onSubmit} disabled={submitting}>
          {submitting ? "Starting..." : "Run analysis"}
        </Button>
        <p className="m-0 text-xs text-slate-500">
          Runs in the background; the queue keeps going if you navigate away.
        </p>
      </div>
    </Panel>
  );
}
