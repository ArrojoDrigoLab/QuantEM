export interface Tag {
  id: string;
  name: string;
  group: string;
  created_at: string;
  updated_at: string;
}

/**
 * A named subset of one experiment's images.
 *
 * Mirrors `quantem.library.serializers.serialize_dataset`. `experiment` is that
 * experiment's id and is never null: a dataset cannot exist outside one.
 */
export interface Dataset {
  id: string;
  experiment: string;
  name: string;
  notes: string;
  /** Images in this dataset that are still in the library. */
  asset_count: number;
  created_at: string | null;
  updated_at: string | null;
}

/**
 * One experiment, with its datasets.
 *
 * Mirrors `quantem.library.serializers.serialize_experiment`. This replaces the
 * corpus-catalogue `Experiment` that was carried over into the frontend types
 * and never had a backend: it declared `start_date`, `origin`, `doi`,
 * `citation`, `raw_metadata` and a `confirmed_assets` flag, and
 * `shared/api/assets.ts` called `/api/experiments/<id>/`, an endpoint the server
 * did not mount. None of those fields exist here. This is a local desktop app
 * organising a scientist's own images, not a catalogue federation.
 *
 * The datasets are nested rather than fetched separately because every caller
 * wants both at once: the library filter, the import form's two pickers, and
 * the tree a fine-tune's scope is chosen from.
 */
export interface Experiment {
  id: string;
  name: string;
  notes: string;
  datasets: Dataset[];
  /** Images in this experiment that are still in the library. */
  asset_count: number;
  /** Of those, the ones in none of its datasets. A real bucket, not a gap. */
  ungrouped_asset_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export type PreprocessStage =
  | "NONE"
  | "ENCODING"
  | "SAM"
  | "FEATURES"
  | "DONE"
  | "FAILED"
  | "CANCELLED"
  | "SKIPPED";

export type LabelState = "CONFIRMED" | "EXCLUDED" | "INFERRED" | "CANDIDATE";
export type SegmentLabelState = LabelState;
export type SegmentCounts = Partial<Record<SegmentLabelState, number>>;
export type CellStatus = 0 | 1 | 2 | 10;
export type CellStatusLabel =
  | "CANDIDATE"
  | "INITIAL_CONFIRM"
  | "MODEL_OUTPUT_GEOMETRY"
  | "REFINED";
export type CellStatusCounts = Partial<Record<CellStatusLabel, number>>;
export type SegmentStatus = 0 | 1 | 10;
export type SegmentStatusLabel = "CANDIDATE" | "CONFIRMED" | "REFINED";

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type RefinementStatus = "UNREFINED" | "MANUAL" | "AUTOMATIC";

export type JobStatus =
  | "PENDING"
  | "RUNNING"
  | "SUCCESS"
  | "FAILED"
  | "CANCELLED"
  | "RETRY";

export type JobPriority = "high" | "default";
export type JobResourceClass = "cpu" | "gpu";
