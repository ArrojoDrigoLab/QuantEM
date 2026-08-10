export interface Tag {
  id: string;
  name: string;
  group: string;
  created_at: string;
  updated_at: string;
}

export interface Experiment {
  id: string;
  name: string;
  start_date: string | null;
  end_date: string | null;
  notes?: string;
  origin?: "LOCAL" | "CATALOG";
  catalog_source_key?: string;
  external_id?: string;
  external_version?: string;
  source_url?: string;
  doi?: string;
  license?: string;
  citation?: string;
  confirmed_assets: boolean;
  raw_metadata?: Record<string, unknown>;
  normalized_metadata?: Record<string, unknown>;
  tags?: Tag[];
  created_at: string;
  updated_at: string;
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
