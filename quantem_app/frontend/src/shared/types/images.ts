import type {
  CellStatusCounts,
  Dataset,
  Experiment,
  PreprocessStage,
  SegmentCounts,
  Tag,
} from "@/shared/types/common";

/**
 * Where one image sits in the library, if anywhere.
 *
 * Emitted by `serialize_asset_grouping` on **both** the list entry and the
 * detail payload, so the library can group and filter sixty cards without sixty
 * round trips.
 *
 * Every field is null or empty for an unorganised image, and that is the
 * ordinary case rather than a missing value: it is the state every library that
 * exists today is in, and nothing here may be rendered as an incomplete setup.
 *
 * The names travel beside the ids so a card can be labelled without resolving
 * anything. They are a snapshot: a rename lands on the next list fetch.
 */
export interface AssetGrouping {
  experiment_id?: string | null;
  experiment_name?: string | null;
  dataset_ids?: string[];
  dataset_names?: string[];
}

// Denormalized mirror of the tile exporter's source registry for one asset.
// Present (with populated counts) once a tile export run has touched the asset;
// has_tiles is true only when a run completed and produced >=1 accepted tile.
export interface TileStatusSummary {
  run_id?: string;
  run_dir?: string;
  manifest_path?: string;
  status?: string;
  is_3d?: boolean;
  config_digest?: string;
  source_export_key?: string;
  tile_records?: number;
  accepted_tiles?: number;
  borderline_tiles?: number;
  rejected_tiles?: number;
  completed_at?: string;
}

/**
 * One stored file behind an asset, as returned by `serialize_rendition`.
 *
 * The FULL rendition's `metadata` is the only place the *file's own* declared
 * pixel size survives: `source_metadata.pixel_size_nm` for a 2D import,
 * `volume_metadata.voxel_size_nm` (z, y, x) for a volume. `Asset.pixel_size_nm`
 * is the effective value and may have been typed by hand, so comparing the two
 * is how the UI tells "read from file" from "entered by hand".
 */
export interface AssetRendition {
  id: string;
  type: "FULL" | "PREVIEW" | string;
  derived_from?: string | null;
  storage_root?: string;
  stored_path?: string;
  path_exists?: boolean;
  is_directory?: boolean;
  stored_width?: number | null;
  stored_height?: number | null;
  stored_depth?: number | null;
  stored_channels?: number | null;
  stored_bit_depth?: number | null;
  z_plane_indices?: number[];
  metadata?: Record<string, unknown>;
}

export interface AssetDetail extends AssetGrouping {
  id: string;
  asset_id?: string;
  asset_key?: string;
  asset_type?: AvailabilityFilter;
  file_path: string;
  original_filename: string;
  display_name: string;
  is_eval_set: boolean;
  notes?: string;
  width: number;
  height: number;
  channels: number;
  bit_depth: number;
  // Nanometres per pixel, or null/absent when the image is uncalibrated.
  // Emitted by serialize_asset_entry, which serialize_asset_detail extends.
  // Null is propagated, never defaulted: a wrong scale silently corrupts every
  // physical-unit measurement downstream.
  pixel_size_nm?: number | null;
  pixel_size_nm_z?: number | null;
  // 3D volume metadata (null/absent for 2D images). depth is the original
  // acquired z-plane count; stored_depth is how many planes the canonical
  // OME-TIFF / NGFF actually contains after z decimation; z_plane_indices maps
  // stored slice index -> original source plane index for true-depth labels.
  depth?: number | null;
  stored_depth?: number | null;
  z_plane_indices?: number[];
  z_sampling?: Record<string, unknown>;
  volume_metadata?: Record<string, unknown>;
  preprocess_stage: PreprocessStage;
  preprocess_stage_display?: string;
  preprocess_progress: number;
  preprocess_error?: string;
  ngff_ready?: boolean;
  ngff_status?: "ready" | "missing" | "queued" | "processing" | "unavailable";
  ngff_url?: string | null;
  is_workable?: boolean;
  can_view?: boolean;
  can_segment?: boolean;
  has_tiles?: boolean;
  tiles_generated_at?: string | null;
  tiles_summary?: TileStatusSummary;
  // Authoritative accepted-tile count (= Asset.accepted_tiles, deduped/png-verified).
  // Use this for the displayed count, not tiles_summary.accepted_tiles.
  tile_count?: number | null;
  /**
   * Stored files behind this asset. Only `serialize_asset_detail` emits them
   * (the list entry does not), and the FULL entry carries the source file's own
   * metadata -- see `resolvePixelSize`.
   */
  renditions?: AssetRendition[];
  tags: Tag[];
  tag_ids?: string[];
  created_at: string;
  updated_at: string;
}

/**
 * One image in the local library, as returned by `serialize_asset_entry`.
 *
 * QuantEM port: this interface previously carried the corpus-catalogue payload
 * (experiment/species/tissue/organ names, tile status, cell counts, eval-set
 * flags). None of those fields exist on `Asset` any more and the serializer
 * does not emit them. Kept in sync with
 * `src/quantem/assets/serializers.py::serialize_asset_entry`.
 */
export interface HomeImage extends AssetGrouping {
  id: string;
  display_name: string;
  original_filename: string;
  notes?: string;
  width?: number | null;
  height?: number | null;
  depth?: number | null;
  stored_depth?: number | null;
  pixel_size_nm?: number | null;
  pixel_size_nm_z?: number | null;
  /**
   * In-plane pixel size the source file itself declared, or null when it was
   * silent. Emitted by `serialize_asset_entry` so a list entry can resolve
   * "read from file" vs "entered by hand" without the renditions the detail
   * payload carries. Absent on an older backend, which resolves to "unknown
   * provenance" rather than a guess.
   */
  file_declared_pixel_size_nm?: number | null;
  metadata_summary: string;
  created_at: string;
  updated_at: string;
  preprocess_stage: PreprocessStage;
  preprocess_progress: number;
  preprocess_error?: string;
  ngff_ready?: boolean;
  ngff_url?: string | null;
  ngff_status?: 'ready' | 'missing' | 'queued' | 'processing' | 'unavailable';
  is_workable?: boolean;
  can_view?: boolean;
  can_segment?: boolean;
}

export type AvailabilityFilter = "all" | "local" | "catalog";

export interface AssetEntry extends HomeImage {
  asset_id?: string;
  asset_key?: string;
  asset_type?: "local" | "catalog";
  entry_type?: "local" | "catalog";
  can_open: boolean;
  notes?: string;
  landing_url?: string;
  download_url?: string;
  preview_thumbnail_url?: string | null;
  // For 3-plane subsets derived from a larger catalog volume: the source asset's
  // accepted tile count and its original (pre-subset) WxHxD dimensions.
  tile_count?: number | null;
  source_dimensions?: string | null;
}

export type HomeEntry = AssetEntry;

export interface HomeEntryPage {
  results: AssetEntry[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export type HomeFilterParamValue = string | string[];

/**
 * The literal the grouping filters accept for "in none of them".
 *
 * Unassigned cannot be named by an id, because there is no row to have one, and
 * it is not a gap that can be left out: it is the bucket every image starts in.
 * Mirrors `UNASSIGNED` in `quantem/assets/views.py`.
 */
export const UNASSIGNED_FILTER = "none";

/**
 * What `/api/assets/` actually filters on.
 *
 * The facets this used to declare -- `tag`, `kingdom`, `species`, `tissue`,
 * `organ`, `confirmed`, `tile_status` -- were the corpus catalogue's, and
 * `_filtered_asset_queryset` reads none of them; they were serialised into the
 * query string on every library fetch and discarded on arrival. `experiment`
 * and `dataset` were in the same dead set and are now real.
 *
 * Both accept a repeated value for a union, and both accept
 * {@link UNASSIGNED_FILTER} alongside real ids, so "this experiment, plus
 * everything not yet filed" is one request.
 */
export interface HomeImagesParams {
  search?: string;
  dataset?: HomeFilterParamValue;
  experiment?: HomeFilterParamValue;
  ordering?: string;
}

export interface HomeEntriesParams extends HomeImagesParams {
  availability?: AvailabilityFilter;
  limit?: number;
  offset?: number;
}

/**
 * What one grouping write did.
 *
 * `dataset_links_dropped` is the consequence of a move: a dataset belongs to
 * exactly one experiment, so images moved out of theirs cannot stay in its
 * datasets. The count travels back so the screen can report it rather than
 * leaving the user to notice later.
 */
export interface AssetGroupingResult {
  assets_changed: number;
  dataset_links_dropped: number;
  assets_moved_out_of_datasets: number;
  datasets_left: string[];
  experiment: Experiment | null;
  datasets: Dataset[];
}

/**
 * A grouping change to apply to a selection of images.
 *
 * Each of `experiment` and `datasets` is a **tri-state**: leave the key out to
 * keep what the images have, send `null` (or `[]`) to clear it, send a value to
 * set it. `experiment_name` / `dataset_name` are the "type a new name" halves of
 * the two pickers and create the row on the way past.
 */
export interface AssetGroupingRequest {
  asset_ids: string[];
  experiment?: string | null;
  experiment_name?: string;
  datasets?: string[] | null;
  dataset_name?: string;
  datasets_mode?: "replace" | "add";
}

export interface SegmentationTypeTag {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface SegmentationType {
  id: string;
  internal_name: string;
  short_name: string;
  long_name: string;
  default_color?: string | null;
  tags: SegmentationTypeTag[];
  tag_ids?: string[];
  created_at: string;
  updated_at: string;
}

export type StatusStage =
  | "UNSTARTED"
  | "RUNNING_INFERENCE"
  | "EXTRACTING_CANDIDATES"
  | "CANDIDATES_READY"
  | "UPDATING"
  | "COMPUTING_FEATURES"
  | "COMPLETED"
  | "FAILED";

export type PipelineStage = "PENDING" | "RUNNING" | "READY" | "FAILED" | "SKIPPED";

export interface SegmentationInstanceParams {
  center_min_distance: number;
  center_confidence_threshold: number;
  segmentation_threshold: number;
  downsampling_factor: number | null;
}

/**
 * What `ImageSegmentationSerializer.get_config` actually emits.
 *
 * The `mitonet_model_*` quartet was removed with MitoNet: the serializer never
 * sent it and the header was rendering a model name out of a hard-coded
 * fallback. Provenance on this screen now comes from `source_models` /
 * `segment_counts_by_source_model` and the overlay manifest, which are real.
 */
export interface ImageSegmentationConfig {
  supports_instance_params?: boolean;
  instance_params?: SegmentationInstanceParams | null;
}

export interface SourceModelOption {
  value: string;
  label: string;
  model_family: string;
  variant?: string;
  is_default?: boolean;
  count?: number;
}

/**
 * What `status_stage` does not say, when the stage alone would mislead.
 *
 * `CANDIDATES_READY` is the stage a finished run leaves behind whether it
 * produced two hundred objects or none, so a run that found nothing looked
 * exactly like a run that worked. `ImageSegmentationSerializer.get_run_notice`
 * derives this from the stage *and* the object count and sends it on the same
 * payload as the stage it qualifies; it is `null` in every other case, so a
 * client renders it whenever it is present rather than re-deciding when it
 * applies.
 *
 * `kind` is open-ended on purpose — a build that grows a second kind must not
 * make this one un-renderable — so nothing here branches on it.
 */
export interface SegmentationRunNotice {
  /** `"no_objects"` or `"no_new_objects"` today. Not switched on: the message is the payload. */
  kind: string;
  /**
   * One short line for the chip, written by the server
   * (`RUN_NOTICE_SUMMARIES`) so the chip and the box beneath it cannot say
   * different things. "Ran and found no objects" is false over a proofread
   * segmentation holding twelve confirmed objects — which is exactly the case
   * the second kind fires on — so the chip must not compose its own. Optional
   * only for an older backend that predates it; the client then falls back to
   * the empty-run wording, the only kind such a backend emits.
   */
  summary?: string;
  /** One sentence, fit to stand next to the stage. */
  message: string;
  /**
   * What to check, in the order the server ranked them. The pixel size leads:
   * it decides what apparent size the model sees an organelle at, and lowering
   * the threshold on a wrongly-scaled run does not bring the objects back.
   */
  next_steps: string[];
}

/**
 * What the image's pixel size was **when this segmentation's objects were
 * made**, read off the objects' own run stamps.
 * `ImageSegmentationSerializer.get_objects_pixel_size`; null when the
 * segmentation holds no objects.
 *
 * `Asset.pixel_size_nm` says what the image records *now*; nothing else on the
 * payload says whether the objects on top of it were produced at that number.
 * They routinely were not: a user imports uncalibrated, runs inference, then
 * types the pixel size in — and every analysis of those objects reports pixels,
 * not µm², because converting them with a number that did not exist when they
 * were measured is arithmetic on the wrong objects.
 */
export interface SegmentationObjectsPixelSize {
  /**
   * Every distinct `native_pixel_size_nm` the stamps recorded, numbers
   * ascending and `null` last. `null` is a real member — a run that had no
   * pixel size to resample with — not a gap. Two entries means two runs at
   * different scales, and the objects are not one population. `unknown`
   * because a damaged stamp can hold anything and the server reports it
   * unchanged rather than failing the read.
   */
  produced_nm: unknown[];
  /**
   * The predicate `run_analysis` blanks its physical units on
   * (`calibrated_after_the_fact`): the image records a usable pixel size now,
   * and at least one object here was produced without one. Sent rather than
   * re-derived client-side so the labeling screen and the finished bundle
   * cannot disagree.
   */
  predates_calibration: boolean;
  /**
   * Objects carrying no run stamp at all — hand-drawn outlines, or objects
   * made before stamping existed. Not folded into `produced_nm`: "not
   * produced by a model" is not "produced at an unknown scale".
   */
  unstamped_count: number;
}

export interface ImageSegmentation {
  id: string;
  asset?: string | null;
  segmentation_type: SegmentationType;
  segmentation_type_id?: string;
  segment_counts?: SegmentCounts;
  source_models?: SourceModelOption[];
  segment_counts_by_source_model?: Record<string, SegmentCounts> | null;
  cell_status_counts?: CellStatusCounts | null;
  config?: ImageSegmentationConfig | null;
  /**
   * The include level the objects on screen were found at, or null when nobody
   * has moved the dial. Deliberately not defaulted to the run's own threshold:
   * that is a different fact, and showing it here would claim a dial position
   * the user never set.
   */
  include_level?: number | null;
  is_complete?: boolean;
  /**
   * Present only when the stage would mislead on its own — a run finished here
   * and left no objects at all, or a re-run over a proofread segmentation
   * added none. See {@link SegmentationRunNotice}.
   */
  run_notice?: SegmentationRunNotice | null;
  /**
   * The pixel size these objects were actually produced at, off their own run
   * stamps. Null when the segmentation holds no objects; absent on an older
   * backend. See {@link SegmentationObjectsPixelSize}.
   */
  objects_pixel_size?: SegmentationObjectsPixelSize | null;
  status_stage: StatusStage;
  status_stage_display?: string;
  status_progress: number;
  status_error?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ImageSegmentationCreatePayload {
  segmentation_type_id?: string;
  segmentation_type_name?: string;
  source_model?: string;
}

export interface UploadImageOptions {
  displayName?: string;
  /**
   * Nanometres per pixel typed at import. Overrides whatever the file declares;
   * omit it (or pass null) to let the reader take the file's own value.
   */
  pixelSizeNm?: number | null;
  /**
   * Free text stored on `Asset.notes` and matched by the library's search box.
   *
   * This replaces `tagNames`, which `uploadAsset` posted as `tag_names` and
   * `AssetUploadView` never read: there is no tag field on `Asset` and no tag
   * anywhere in the Python tree, so every word typed into the import form's
   * Tags box was discarded on arrival.
   */
  notes?: string;
  segmentMito?: boolean;
  segmentEr?: boolean;
  segmentNucleus?: boolean;
  segmentLd?: boolean;
  /**
   * Where in the library these images go, if anywhere.
   *
   * Optional, and optional is load-bearing: an import that names none of these
   * behaves exactly as it did before they existed. Each pair is "an existing
   * one" or "a new name to create"; the id wins when both are sent.
   *
   * A dataset without an experiment is refused at the door, because a dataset
   * lives inside exactly one experiment and the combination describes nothing.
   */
  experimentId?: string;
  experimentName?: string;
  datasetId?: string;
  datasetName?: string;
}

export interface ProbabilityMap {
  id: string;
  name: string;
  file_path: string;
  channel_index: number;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ImageROI {
  id: string;
  asset: string;
  display_name: string;
  x: number;
  y: number;
  width: number;
  height: number;
  source: "AUTO" | "MANUAL";
  created_at: string;
  updated_at: string;
}
