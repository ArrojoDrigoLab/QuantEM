import type { SegmentationInstanceParams } from "@/shared/types/images";

export interface InstanceParamsSection {
  enabled: boolean;
  draft: SegmentationInstanceParams | null;
  isSaving: boolean;
  hasQueuedOrRunningOrganelleTask: boolean;
  onChange: (
    key: keyof SegmentationInstanceParams,
    value: number | null
  ) => void;
  onSave: () => void;
}

/**
 * Full-width instance-parameter toolbar shown above the labeling panes for
 * segmentation types that expose tunable instance params (e.g. mitochondria).
 *
 * This is rendered as a sibling of the header/main grid -- never inside the
 * `.segmentation-main` grid -- so it does not consume a grid column and create
 * a spurious "second sidebar".
 */
export function SegmentationInstanceParamsBar({
  enabled,
  draft,
  isSaving,
  hasQueuedOrRunningOrganelleTask,
  onChange,
  onSave,
}: InstanceParamsSection) {
  if (!enabled || !draft) {
    return null;
  }
  return (
    <section className="instance-params-toolbar">
      <div className="instance-params-fields">
        <label htmlFor="seg-param-threshold">
          Segmentation Threshold
          <input
            id="seg-param-threshold"
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={draft.segmentation_threshold}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isNaN(value)) return;
              onChange("segmentation_threshold", value);
            }}
          />
        </label>
        <label htmlFor="seg-param-center-threshold">
          Center Confidence
          <input
            id="seg-param-center-threshold"
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={draft.center_confidence_threshold}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isNaN(value)) return;
              onChange("center_confidence_threshold", value);
            }}
          />
        </label>
        <label htmlFor="seg-param-min-distance">
          Center Min Distance
          <input
            id="seg-param-min-distance"
            type="number"
            min={1}
            step={1}
            value={draft.center_min_distance}
            onChange={(event) => {
              const value = Number(event.target.value);
              if (Number.isNaN(value)) return;
              onChange("center_min_distance", value);
            }}
          />
        </label>
        <label htmlFor="seg-param-downsampling">
          Downsampling Factor
          <input
            id="seg-param-downsampling"
            type="number"
            min={1}
            step={1}
            placeholder="Auto"
            value={draft.downsampling_factor ?? ""}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw.trim() === "") {
                onChange("downsampling_factor", null);
                return;
              }
              const value = Number(raw);
              if (Number.isNaN(value)) return;
              onChange("downsampling_factor", value);
            }}
          />
        </label>
      </div>
      <div className="instance-params-actions">
        <button
          type="button"
          className="instance-params-button"
          onClick={onSave}
          disabled={isSaving || hasQueuedOrRunningOrganelleTask}
        >
          {isSaving ? "Saving..." : "Save Params"}
        </button>
      </div>
    </section>
  );
}
