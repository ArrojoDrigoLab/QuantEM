/** Configure one analysis run with its measured compartments and analysis mask. */

import { Button, Panel } from "@/shared/ui/design";
import type { ImageSegmentation } from "@/shared/types/images";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import {
  ANALYSIS_MASK_INTERNAL_NAME,
  type AnalysisFormState,
} from "@/features/analysis/analysisOptions";

const FIELD_CLASS =
  "h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500 disabled:bg-slate-100 disabled:text-slate-500";
const LABEL_CLASS =
  "block text-xs font-semibold uppercase tracking-wide text-slate-500";

export interface AnalysisConfigFormProps {
  segmentations: ImageSegmentation[];
  state: AnalysisFormState;
  onChange: (next: AnalysisFormState) => void;
  onSubmit: () => void;
  submitting: boolean;
  error: string | null;
}

export function AnalysisConfigForm({
  segmentations,
  state,
  onChange,
  onSubmit,
  submitting,
  error,
}: AnalysisConfigFormProps) {
  const patch = (updates: Partial<AnalysisFormState>) => {
    onChange({ ...state, ...updates });
  };
  const analysisMasks = segmentations.filter(
    (segmentation) =>
      segmentation.segmentation_type.internal_name === ANALYSIS_MASK_INTERNAL_NAME
  );
  const segmentationLabel = (segmentationId: string): string => {
    const segmentation = segmentations.find((entry) => entry.id === segmentationId);
    return segmentation?.segmentation_type.long_name ?? segmentationId;
  };

  return (
    <Panel className="p-4">
      <h2 className="m-0 text-base font-semibold text-slate-950">
        Configure the run
      </h2>

      <fieldset className="mt-4 border-0 p-0">
        <legend className={LABEL_CLASS}>Compartments</legend>
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
        <label className={LABEL_CLASS} htmlFor="analysis-mask">
          Analysis Mask
        </label>
        <select
          id="analysis-mask"
          className={`${FIELD_CLASS} mt-1`}
          value={state.tissueSegmentationId}
          disabled={submitting}
          onChange={(event) => patch({ tissueSegmentationId: event.target.value })}
        >
          <option value="">Whole image (no analysis mask)</option>
          {analysisMasks.map((segmentation) => (
            <option key={segmentation.id} value={segmentation.id}>
              {segmentationDisplayName(segmentation)}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-4">
        <Button variant="primary" onClick={onSubmit} disabled={submitting}>
          {submitting ? "Starting…" : "Run Analysis"}
        </Button>
      </div>

      {error ? (
        <p className="m-0 mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
          {error}
        </p>
      ) : null}
    </Panel>
  );
}
