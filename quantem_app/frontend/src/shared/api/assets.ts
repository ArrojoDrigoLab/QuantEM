// Canonical image-like API contract. Public callers should use asset IDs.
import { apiRequest, apiRequestFormData } from "@/shared/api/core/http";
import type { Dataset, Experiment } from "@/shared/types/common";
import type {
  AssetDetail,
  AssetEntry,
  AssetGroupingRequest,
  AssetGroupingResult,
  HomeEntriesParams,
  HomeEntryPage,
  HomeEntry,
  HomeImage,
  HomeImagesParams,
  ImageSegmentation,
  ImageSegmentationCreatePayload,
  SegmentationType,
  UploadImageOptions,
} from "@/shared/types/images";
import { resolveApiUrl } from "@/shared/api/core/http";

/**
 * The orderings `/api/assets/` actually implements.
 *
 * Mirrors `ASSET_ORDERINGS` in `quantem/assets/views.py`. Anything else is
 * **silently** replaced by `display_name` server-side, which is what made a new
 * import disappear: the library paged 60 rows with no `ordering` at all, the
 * server sorted them alphabetically, the client then sorted *those 60* by
 * import date, and the newest asset -- row 61 of 62 by name -- was never
 * fetched. The card simply did not exist, under a sort control reading
 * "Imported / Descending" and a footer reading "Showing 60 of 62 images".
 *
 * So every paged caller sends one of these, and the list is exported rather
 * than spelled out at the call site so a typo cannot silently become
 * alphabetical order again.
 *
 * Note what is *not* here: `status`. The library used to offer it as a sort and
 * arrange it client-side inside a window fetched by a *different* ordering,
 * which reorders one page and calls it a library sort. The option is gone from
 * the control rather than faked; see `LibraryPage.tsx`'s `SORT_FIELDS`.
 */
export const ASSET_ORDERINGS = [
  "display_name",
  "-display_name",
  "created_at",
  "-created_at",
  "updated_at",
  "-updated_at",
] as const;

export type AssetOrdering = (typeof ASSET_ORDERINGS)[number];

export function getHomeImages(params: HomeImagesParams = {}): Promise<HomeImage[]> {
  const query = new URLSearchParams();
  appendHomeImageParams(query, params);
  if (params.ordering) query.set("ordering", params.ordering);
  const qs = query.toString();
  return apiRequest<HomeImage[]>(`/api/assets/${qs ? `?${qs}` : ""}`);
}

function appendHomeParam(
  query: URLSearchParams,
  key: string,
  value: string | string[] | undefined
) {
  if (Array.isArray(value)) {
    for (const item of value) {
      if (item) query.append(key, item);
    }
    return;
  }
  if (value) query.set(key, value);
}

/**
 * Exactly the parameters `_filtered_asset_queryset` reads.
 *
 * It used to send seven more -- `tag`, `kingdom`, `species`, `tissue`, `organ`,
 * `confirmed`, `tile_status` -- which were the corpus catalogue's facets. The
 * server has never looked at any of them, so every library fetch carried a
 * query string describing filters that did nothing. Same defect as the import
 * form's Tags box: a control that is accepted and discarded reads, from the
 * call site, as a feature that works. `experiment` and `dataset` were in that
 * dead set too, and are now real.
 */
function appendHomeImageParams(query: URLSearchParams, params: HomeImagesParams) {
  if (params.search) query.set("search", params.search);
  appendHomeParam(query, "dataset", params.dataset);
  appendHomeParam(query, "experiment", params.experiment);
  if (params.ordering) query.set("ordering", params.ordering);
}

export function getAssetEntries(params: HomeEntriesParams = {}): Promise<AssetEntry[]> {
  const query = new URLSearchParams();
  appendHomeImageParams(query, params);
  if (params.availability) query.set("availability", params.availability);
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  const qs = query.toString();
  return apiRequest<AssetEntry[]>(`/api/assets/${qs ? `?${qs}` : ""}`);
}

export function getHomeEntries(params: HomeEntriesParams = {}): Promise<HomeEntry[]> {
  return getAssetEntries(params);
}

/**
 * One page of the library.
 *
 * `params.ordering` decides which rows exist on this page, not merely their
 * arrangement — see {@link ASSET_ORDERINGS}. A paged caller that omits it gets
 * the alphabetically first `limit` rows.
 */
export function getAssetEntryPage(params: HomeEntriesParams = {}): Promise<HomeEntryPage> {
  const query = new URLSearchParams();
  appendHomeImageParams(query, params);
  if (params.availability) query.set("availability", params.availability);
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.offset !== undefined) query.set("offset", String(params.offset));
  const qs = query.toString();
  return apiRequest<HomeEntryPage>(`/api/assets/${qs ? `?${qs}` : ""}`);
}

export function getHomeEntryPage(params: HomeEntriesParams = {}): Promise<HomeEntryPage> {
  return getAssetEntryPage(params);
}

/**
 * Experiments and datasets: the optional grouping over the image library.
 *
 * These three functions existed before any of it did. `getExperiment`,
 * `updateExperiment` and `deleteExperiment` called `/api/experiments/<id>/`,
 * which the server did not mount, against an `Experiment` type carrying
 * `confirmed_assets`, `doi` and `citation` -- the corpus catalogue's shape, left
 * behind when that product was cut. They are real now, and the shape is what
 * this application actually stores.
 *
 * The list carries each experiment's datasets and counts in one response. The
 * library is a desktop library, so one call is right; do not paginate it.
 */
export function getExperiments(): Promise<Experiment[]> {
  return apiRequest<Experiment[]>("/api/experiments/");
}

export function getExperiment(experimentId: string): Promise<Experiment> {
  return apiRequest<Experiment>(`/api/experiments/${experimentId}/`);
}

export function createExperiment(payload: {
  name: string;
  notes?: string;
}): Promise<Experiment> {
  return apiRequest<Experiment>("/api/experiments/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateExperiment(
  experimentId: string,
  updates: Partial<Pick<Experiment, "name" | "notes">>
): Promise<Experiment> {
  return apiRequest<Experiment>(`/api/experiments/${experimentId}/`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

/**
 * Delete an experiment. **The images survive.**
 *
 * `Asset.experiment` is `SET_NULL`: the images become unassigned, which is an
 * ordinary state, and the experiment's datasets go with it because a dataset
 * cannot exist outside one. Any confirmation copy in front of this must say so
 * -- "delete" over a group of images reads as "delete the images".
 */
export function deleteExperiment(experimentId: string): Promise<void> {
  return apiRequest<void>(`/api/experiments/${experimentId}/`, {
    method: "DELETE",
  });
}

export function getDatasets(experimentId?: string): Promise<Dataset[]> {
  const query = experimentId
    ? `?experiment=${encodeURIComponent(experimentId)}`
    : "";
  return apiRequest<Dataset[]>(`/api/datasets/${query}`);
}

export function createDataset(payload: {
  experiment: string;
  name: string;
  notes?: string;
}): Promise<Dataset> {
  return apiRequest<Dataset>("/api/datasets/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateDataset(
  datasetId: string,
  updates: Partial<Pick<Dataset, "name" | "notes">>
): Promise<Dataset> {
  return apiRequest<Dataset>(`/api/datasets/${datasetId}/`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

/** Delete a dataset. Its images stay in the experiment. */
export function deleteDataset(datasetId: string): Promise<void> {
  return apiRequest<void>(`/api/datasets/${datasetId}/`, { method: "DELETE" });
}

/**
 * Put a selection of images into an experiment and a dataset, or take them out.
 *
 * One route for one image and for forty: the library acts on a selection, and
 * a selection of one is still a selection. See {@link AssetGroupingRequest} for
 * the tri-state each field has, and note `dataset_links_dropped` on the reply --
 * moving images to another experiment takes them out of the datasets that
 * experiment does not contain, and the screen has to say so.
 */
export function assignAssetGrouping(
  payload: AssetGroupingRequest
): Promise<AssetGroupingResult> {
  return apiRequest<AssetGroupingResult>("/api/assets/grouping/", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getAsset(assetId: string): Promise<AssetDetail> {
  return apiRequest<AssetDetail>(`/api/assets/${assetId}/`);
}

export function getAssetPreviewPngUrl(assetId: string, cacheKey?: string | null): string {
  const base = resolveApiUrl(`/api/assets/${assetId}/preview-png`);
  if (!cacheKey) return base;
  return `${base}?v=${encodeURIComponent(cacheKey)}`;
}

export function getAssetNgffThumbnailUrl(assetId: string, cacheKey?: string | null): string {
  const base = resolveApiUrl(`/api/assets/${assetId}/ngff-thumbnail/`);
  if (!cacheKey) return base;
  return `${base}?v=${encodeURIComponent(cacheKey)}`;
}

export function getAssetPreviewThumbnailUrl(
  assetId: string,
  cacheKey?: string | null
): string {
  const base = resolveApiUrl(`/api/assets/${assetId}/preview-thumbnail/`);
  if (!cacheKey) return base;
  return `${base}?v=${encodeURIComponent(cacheKey)}`;
}

/**
 * Roughly what a multipart envelope adds to the file's own bytes.
 *
 * The request body is not the file: it is boundary lines, a
 * `Content-Disposition` header per part, and the other form fields. A file one
 * byte under the server's limit therefore still produces a body over it, and
 * waitress compares with `>=`, so the refusal is by *body* size. 4 KiB is far
 * more than the real envelope (a few hundred bytes) and far too small to make
 * the stated limit wrong when it is printed — `formatBytes` renders
 * 64 GiB and 64 GiB − 4 KiB identically.
 */
const UPLOAD_ENVELOPE_ALLOWANCE_BYTES = 4096;

/**
 * The largest upload this server will accept, in bytes, or `null` when it has
 * not said.
 *
 * `/api/system/status/` reports `max_upload_bytes` — the same
 * `QUANTEM_MAX_UPLOAD_BYTES` that `quantem serve` hands to waitress as
 * `max_request_body_size`. Without it the client could not refuse an impossible
 * file locally: waitress rejects from the request headers and closes the socket
 * while the browser is still streaming, which the browser reports as a plain
 * network error, minutes into an upload that was never going to be accepted.
 *
 * It is read defensively rather than typed on `SystemStatus`
 * (`shared/types/jobs.ts`) because that file is not this package's to edit; an
 * older server, or a stubbed one, simply says nothing and no local check runs.
 * `null` therefore means "unknown", never "unlimited" — a caller must not
 * invent a limit of its own.
 */
export function readMaxUploadBytes(status: unknown): number | null {
  if (!status || typeof status !== "object") return null;
  const value = (status as { max_upload_bytes?: unknown }).max_upload_bytes;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value;
}

/**
 * Whether this file is too large for the server to accept at all.
 *
 * False when the limit is unknown: refusing on a guess would block imports the
 * server would have taken.
 */
export function exceedsUploadLimit(
  fileSizeBytes: number,
  maxUploadBytes: number | null
): boolean {
  if (maxUploadBytes === null) return false;
  return fileSizeBytes >= maxUploadBytes - UPLOAD_ENVELOPE_ALLOWANCE_BYTES;
}

export function uploadAsset(
  file: File,
  options: UploadImageOptions = {}
): Promise<AssetDetail> {
  const formData = new FormData();
  formData.append("file", file);
  if (options.displayName) {
    formData.append("display_name", options.displayName);
  }
  // A typed pixel size wins over whatever the file declares (see
  // `create_uploaded_asset`), so only send it when the user actually typed one.
  if (options.pixelSizeNm !== undefined && options.pixelSizeNm !== null) {
    formData.append("pixel_size_nm", String(options.pixelSizeNm));
  }
  // `notes`, not the `tag_names` this used to post. `AssetUploadView` never read
  // `tag_names` -- there is no tag field on `Asset` and no tag anywhere in the
  // Python tree -- so the import form's Tags box accepted text and dropped it.
  // `notes` is a real column, it survives the round trip, and
  // `_filtered_asset_queryset` searches it.
  if (options.notes) {
    formData.append("notes", options.notes);
  }
  if (options.segmentMito !== undefined) {
    // Only `segment_mito`. The companion `use_mitonet` field went out with
    // MitoNet -- no backend view reads it, so it was a dead parameter named
    // after a model this product does not ship.
    formData.append("segment_mito", options.segmentMito ? "true" : "false");
  }
  if (options.segmentEr !== undefined) {
    formData.append("segment_er", options.segmentEr ? "true" : "false");
  }
  if (options.segmentNucleus !== undefined) {
    formData.append("segment_nucleus", options.segmentNucleus ? "true" : "false");
  }
  if (options.segmentLd !== undefined) {
    formData.append("segment_ld", options.segmentLd ? "true" : "false");
  }
  // Optional grouping. Only sent when the user actually chose or typed
  // something: an import that names nothing must produce exactly the request
  // this function produced before these fields existed.
  if (options.experimentId) {
    formData.append("experiment_id", options.experimentId);
  } else if (options.experimentName) {
    formData.append("experiment_name", options.experimentName);
  }
  if (options.datasetId) {
    formData.append("dataset_id", options.datasetId);
  } else if (options.datasetName) {
    formData.append("dataset_name", options.datasetName);
  }
  return apiRequestFormData<AssetDetail>("/api/assets/upload/", formData);
}

/**
 * Patch mutable asset fields.
 *
 * `pixel_size_nm` is in the accepted set (`update_asset`'s `allowed`): `null`
 * clears it back to uncalibrated, and a non-positive value is a 400. This is
 * the only route by which an image that arrived without a resolution tag can
 * ever become measurable.
 *
 * The signature is exactly `update_asset`'s `allowed` set and nothing more. It
 * used to offer `tag_ids`, `experiment_id` and `is_eval_set` as well, none of
 * which that function copies -- the same defect as the import form's Tags box
 * one function up, and worth removing for the same reason: a parameter the
 * server silently drops reads, from the call site, as a feature that works.
 *
 * Experiment and dataset are deliberately **not** back here. They go through
 * {@link assignAssetGrouping}, which is the only route that enforces the rule
 * that an image's datasets all belong to its experiment. A second door onto the
 * same two columns is a second place for that rule to be forgotten.
 */
export function updateAsset(
  assetId: string,
  updates: Partial<Pick<AssetDetail, "display_name" | "notes" | "pixel_size_nm">>
): Promise<AssetDetail> {
  return apiRequest<AssetDetail>(`/api/assets/${assetId}/`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export function deleteAsset(assetId: string): Promise<void> {
  return apiRequest<void>(`/api/assets/${assetId}/`, {
    method: "DELETE",
  });
}

interface SegmentationTypePage {
  results: SegmentationType[];
}

export async function getSegmentationTypes(): Promise<SegmentationType[]> {
  const response = await apiRequest<SegmentationType[] | SegmentationTypePage>(
    "/api/segmentation-types/"
  );

  // The DRF viewset is paginated in installed builds. Keep accepting a raw
  // list as well so this client remains compatible with unpaginated servers.
  if (Array.isArray(response)) return response;
  return Array.isArray(response.results) ? response.results : [];
}

export function createSegmentationType(
  data: Partial<SegmentationType>
): Promise<SegmentationType> {
  return apiRequest<SegmentationType>("/api/segmentation-types/", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function getAssetSegmentations(assetId: string): Promise<ImageSegmentation[]> {
  return apiRequest<ImageSegmentation[]>(`/api/assets/${assetId}/segmentations/`);
}

export function createAssetSegmentation(
  assetId: string,
  payload: ImageSegmentationCreatePayload
): Promise<ImageSegmentation> {
  return apiRequest<ImageSegmentation>(`/api/assets/${assetId}/segmentations/`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getSegmentation(segmentationId: string): Promise<ImageSegmentation> {
  return apiRequest<ImageSegmentation>(`/api/segmentations/${segmentationId}/`);
}

export function getAssetNgffUrl(assetId: string, cacheKey?: string | null): string {
  const params = new URLSearchParams();
  if (cacheKey) params.set("v", cacheKey);
  const base = resolveApiUrl(`/ngff/assets/${assetId}.zarr`);
  const query = params.toString();
  return query ? `${base}?${query}` : base;
}
