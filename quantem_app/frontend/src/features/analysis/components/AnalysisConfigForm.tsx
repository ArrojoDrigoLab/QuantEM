import { defaultOutputFiles } from "../outputSelection";
import { AnalysisFileSelection } from "../AnalysisFileSelection";
/** Configure one analysis run with its measured compartments and analysis mask. */

import { Button, Panel } from "@/shared/ui/design";
import { HelpDisclosure } from "@/shared/ui/HelpDisclosure";
import type { ImageSegmentation } from "@/shared/types/images";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import {
  CELLS_INTERNAL_NAME,
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
  const sampleAreas = segmentations.filter(
    (segmentation) =>
      segmentation.segmentation_type.category === "area"
  );
  const cellLayers = segmentations.filter(
    (segmentation) =>
      segmentation.segmentation_type.internal_name === CELLS_INTERNAL_NAME
  );
  const segmentationLabel = (segmentationId: string): string => {
    const segmentation = segmentations.find((entry) => entry.id === segmentationId);
    return segmentation ? segmentationDisplayName(segmentation) : segmentationId;
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
                disabled={submitting || entry.ready === false}
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
                {segmentationLabel(entry.segmentationId)}{entry.ready === false && " — whole-image segmentation required"}
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
        <div className="flex items-center gap-2">
          <label className={LABEL_CLASS} htmlFor="sample-area">Measurement area</label>
          {state.tissueSegmentationId && <HelpDisclosure label="masked measurements" title="Measuring within a mask">
            An object is included when its centre is inside the mask; its full outline is measured. Compartment areas are clipped to the mask.
          </HelpDisclosure>}
        </div>
        <select
          id="sample-area"
          className={`${FIELD_CLASS} mt-1`}
          value={state.tissueSegmentationId}
          disabled={submitting}
          onChange={(event) =>
            patch({
              tissueSegmentationId: event.target.value,
              outputs: {
                ...state.outputs,
                sample_area: event.target.value
                  ? true
                  : false,
              },
            })
          }
        >
          <option value="">No area restriction</option>
          {sampleAreas.map((segmentation) => (
            <option key={segmentation.id} value={segmentation.id}>
              {segmentationDisplayName(segmentation)}
            </option>
          ))}
        </select>
      </div>

      <div className="mt-4">
        <label className={LABEL_CLASS} htmlFor="cells-layer">Cells</label>
        <select
          id="cells-layer"
          className={`${FIELD_CLASS} mt-1`}
          value={state.cellSegmentationId}
          disabled={submitting}
          onChange={(event) =>
            patch({
              cellSegmentationId: event.target.value,
              outputs: {
                ...state.outputs,
                all_cells: event.target.value ? state.outputs.all_cells : false,
                per_cell: event.target.value ? state.outputs.per_cell : false,
              },
            })
          }
        >
          <option value="">No cell breakdown</option>
          {cellLayers.map((segmentation) => (
            <option key={segmentation.id} value={segmentation.id}>
              {segmentationDisplayName(segmentation)}
            </option>
          ))}
        </select>
      </div>

      <fieldset className="mt-4 border-0 p-0">
        <legend className={LABEL_CLASS}>Outputs</legend>
        <div className="mt-2 flex flex-col gap-2 text-sm text-slate-700">
          {[
            ["whole_image", "Whole image summary"],
            ["sample_area", "Selected area summary"],
            ["all_cells", "All cells summary"],
            ["per_cell", "Per-cell breakdown"],
            ["individual_objects", "Individual organelle table"],
          ].map(([key, label]) => {
            const outputKey = key as keyof AnalysisFormState["outputs"];
            const requiresSample = outputKey === "sample_area";
            const requiresCells = outputKey === "all_cells" || outputKey === "per_cell";
            const locked = outputKey === "whole_image";
            const unavailable =
              (requiresSample && !state.tissueSegmentationId) ||
              (requiresCells && !state.cellSegmentationId);
            return (
              <label key={key} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={state.outputs[outputKey]}
                  disabled={submitting || locked || unavailable}
                  onChange={(event) =>
                    patch({
                      outputs: {
                        ...state.outputs,
                        [outputKey]: event.target.checked,
                      },
                    })
                  }
                />
                {label}
                {locked ? " (always included)" : ""}
              </label>
            );
          })}
        </div>
      </fieldset>

      <details className="mt-4 rounded-md border border-slate-200 px-3 py-2 text-sm">
        <summary className="cursor-pointer font-medium text-slate-700">
          Plain-language preview
        </summary>
        <p className="mb-0 mt-2 text-slate-600">
          QuantEM will report every selected organelle across the whole image
          {state.outputs.sample_area && state.tissueSegmentationId
            ? ", within the selected measurement area"
            : ""}
          {state.outputs.all_cells && state.cellSegmentationId
            ? ", across the union of all cells"
            : ""}
          {state.outputs.per_cell && state.cellSegmentationId
            ? ", and for each individual cell"
            : ""}
          . Object-based organelles are assigned to cells by centroid and keep
          their full-outline measurements.
        </p>
      </details>

      <div className="mt-4">
        <AnalysisFileSelection value={state.outputFiles ?? defaultOutputFiles()} onChange={(outputFiles) => patch({ outputFiles })} />
        <Button variant="primary" onClick={onSubmit} disabled={submitting || !state.compartments.some((entry) => entry.enabled) || state.compartments.some((entry) => entry.enabled && entry.ready === false)}>
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
