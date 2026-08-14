/**
 * Quantitative analysis runs, as returned by `quantem.analysis`.
 *
 * Shapes follow `API_CONTRACT.md` §Analysis and the flattening done by
 * `AnalysisRunSerializer`: `composition`/`objects`/`points`/`distances`/
 * `monte_carlo` sit at the top level of the run, not under `results`.
 *
 * Sections that were not computed come back as `null` rather than being
 * omitted. "Not computed" and "computed and empty" are different states — no
 * points imported is not the same as every point falling off the tissue — so
 * every optional section is `... | null` here and the UI branches on it.
 */

export type AnalysisRunStatus = "PENDING" | "RUNNING" | "SUCCESS" | "FAILED";

export interface GlobalAreaRow {
  segmentation_id: string | null;
  name: string;
  foreground_pixels: number;
  denominator_pixels: number;
  foreground_percent: number | null;
}

export interface GlobalAreaReport {
  measurement_mode: "global";
  metric: "foreground_area_percent";
  segmentation_id: string;
  whole_image: GlobalAreaRow;
  analysis_masks: GlobalAreaRow[];
}

/** How a point set was obtained. `null` means the run measured no points. */
export type AnalysisPointsSource = "centroids" | "csv" | null;

/** Distribution summary for one morphometric. `n` alone when nothing was finite. */
export interface AnalysisMetricSummary {
  n: number;
  mean?: number | null;
  sd?: number | null;
  median?: number | null;
  iqr?: number | null;
  min?: number | null;
  max?: number | null;
  /** Confirmed objects this metric could have been measured on. */
  n_objects?: number | null;
  /** Of those, how many have no value here. */
  n_missing?: number | null;
  /** Missing values that were measured but rejected as estimator failures. */
  n_unreportable?: number | null;
  /** Why an estimator-produced value was rejected. */
  unreportable_reason?: string | null;
  /** Coverage explanation for partial measurements. */
  note?: string | null;
  /** Legacy saved runs may include a long estimator explanation here. */
  estimator_note?: string | null;
}

export interface AnalysisComposition {
  tissue_px: number;
  /** Null whenever pixel size is unset — do not render µm² in that state. */
  tissue_um2: number | null;
  area_fractions: Record<string, number>;
  areas_px: Record<string, number>;
  areas_um2: Record<string, number> | null;
  /** Individually named objects from the selected Analysis Mask. */
  regions?: AnalysisCompositionRegion[];
}

export interface AnalysisCompositionRegion {
  id: string;
  name: string;
  tissue_px: number;
  tissue_um2: number | null;
  area_fractions: Record<string, number>;
  areas_px: Record<string, number>;
  areas_um2: Record<string, number> | null;
}

export interface AnalysisObjectDensity {
  count: number;
  tissue_um2?: number | null;
  per_um2: number | null;
}

export interface AnalysisObjects {
  n: number;
  summary: Record<string, AnalysisMetricSummary>;
  density: AnalysisObjectDensity;
}

export interface AnalysisPoints {
  n_total: number;
  n_on_tissue: number;
  /**
   * Excluded from every fraction. Honesty rule 5: always shown when non-zero.
   *
   * `n_total == n_on_tissue + n_off_tissue + n_unreadable` — an unreadable row
   * is *not* counted as off-tissue, because a point that cannot be read is
   * nowhere rather than outside something.
   */
  n_off_tissue: number;
  /**
   * Rows whose coordinate is missing or infinite, dropped before any
   * measurement. In no count, fraction or enrichment.
   *
   * Optional because a run stored before `assign_points` reported it has no
   * such key; absent is "this build did not say", not zero.
   */
  n_unreadable?: number;
  /**
   * Points outside the image, clipped onto its border and counted *there*.
   *
   * A whole point set landing on one edge is what a CSV in nanometres looks
   * like, so this is the number that catches a unit error before an enrichment
   * gets quoted. Optional for the same reason as `n_unreadable`.
   */
  n_out_of_bounds?: number;
  counts: Record<string, number>;
  fractions: Record<string, number>;
  /** `null` for a compartment with zero area: the ratio is undefined, not infinite. */
  enrichment: Record<string, number | null>;
}

export interface AnalysisDistances {
  target: string;
  band_labels: string[];
  band_counts: number[];
  band_fractions: number[];
  median_nm: number | null;
  n_inside: number;
  /**
   * What the median and the bands are actually over.
   *
   * Not `points.n_total`: `distance_to_boundary` drops the rows that are not
   * positions, so with any unreadable row the two differ and the screen was
   * quoting the wrong denominator ("41 of 60 inside" over a section that
   * measured 55). `dist.n` server-side.
   *
   * Optional because a run stored before the server sent it has no such key.
   */
  n_measured?: number;
  /** Points with no readable coordinate, so no distance to anything. */
  n_unreadable?: number;
  /**
   * Measured points that lie outside the image and were clipped onto the
   * border, so their distance is from that border pixel and not from the
   * coordinates given. They are inside `n_measured`, not beside it.
   */
  n_out_of_image?: number;
}

export interface AnalysisMonteCarlo {
  replicates: number;
  seed: number;
  observed: Record<string, number | null>;
  null_mean: Record<string, number | null>;
  null_sd: Record<string, number | null>;
  z: Record<string, number | null>;
  p_two_sided: Record<string, number | null>;
}

/**
 * The randomised-input control: uniform points over the user's own masks must
 * give enrichment ~1.0 everywhere, or the normalisation is biased for this
 * geometry.
 */
export interface AnalysisMonteCarloSelfCheck {
  n_points: number;
  smallest_compartment_fraction: number;
  enrichment: Record<string, number | null>;
  max_abs_deviation: number;
  [key: string]: unknown;
}

/** Echo of the validated request, as stored on the run. */
export interface AnalysisRunParams {
  compartments?: Record<string, string>;
  tissue_segmentation_id?: string | null;
  points_source?: AnalysisPointsSource;
  points_csv?: string;
  distance_target?: string | null;
  band_edges_nm?: number[];
  replicates?: number;
  seed?: number;
  group?: string;
  [key: string]: unknown;
}

export interface AnalysisRun {
  id: string;
  segmentation_id: string;
  status: AnalysisRunStatus;
  group: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  params: AnalysisRunParams;
  /** Null when the image has no pixel size: report pixels, never µm. */
  pixel_size_nm: number | null;
  calibrated: boolean | null;
  composition: AnalysisComposition | null;
  objects: AnalysisObjects | null;
  points: AnalysisPoints | null;
  distances: AnalysisDistances | null;
  monte_carlo: AnalysisMonteCarlo | null;
  monte_carlo_self_check: AnalysisMonteCarloSelfCheck | null;
  caveats: string[];
  export_dir: string;
  /** Bundle files that exist on disk right now, e.g. `["manifest.json", ...]`. */
  exports: string[];
  error: string;
}

/** One row of the run history: enough to pick a run, not enough to plot one. */
export interface AnalysisRunSummary {
  id: string;
  status: AnalysisRunStatus;
  group: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  export_dir: string;
  error: string;
  n_objects: number | null;
  calibrated: boolean | null;
  n_caveats: number;
}

export interface AnalysisRunCreatePayload {
  /** organelle name -> segmentation id, e.g. `{"mito": "<uuid>"}`. */
  compartments?: Record<string, string>;
  tissue_segmentation_id?: string | null;
  points_source?: AnalysisPointsSource;
  points_csv?: string;
  distance_target?: string | null;
  band_edges_nm?: number[];
  replicates?: number;
  seed?: number;
  group?: string;
}

export interface AnalysisRunCreateResponse {
  job_id: string;
  analysis_run_id: string;
}

/** The three files `write_bundle` produces. */
export type AnalysisExportName = "objects.csv" | "image_summary.csv" | "manifest.json";
