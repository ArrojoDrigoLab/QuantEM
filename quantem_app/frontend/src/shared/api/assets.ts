// Canonical image-like API contract. Public callers should use asset IDs.
import { apiRequest, apiRequestFormData } from "@/shared/api/core/http";
import type { Experiment } from "@/shared/types/common";
import type {
  AssetDetail,
  AssetEntry,
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

function appendHomeImageParams(query: URLSearchParams, params: HomeImagesParams) {
  if (params.search) query.set("search", params.search);
  appendHomeParam(query, "tag", params.tag);
  appendHomeParam(query, "kingdom", params.kingdom);
  appendHomeParam(query, "species", params.species);
  appendHomeParam(query, "tissue", params.tissue);
  appendHomeParam(query, "organ", params.organ);
  appendHomeParam(query, "dataset", params.dataset);
  appendHomeParam(query, "experiment", params.experiment);
  appendHomeParam(query, "confirmed", params.confirmed);
  appendHomeParam(query, "tile_status", params.tile_status);
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

export function getExperiment(experimentId: string): Promise<Experiment> {
  return apiRequest<Experiment>(`/api/experiments/${experimentId}/`);
}

export function updateExperiment(
  experimentId: string,
  updates: Partial<Pick<Experiment, "confirmed_assets">>
): Promise<Experiment> {
  return apiRequest<Experiment>(`/api/experiments/${experimentId}/`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
}

export function deleteExperiment(experimentId: string): Promise<void> {
  return apiRequest<void>(`/api/experiments/${experimentId}/`, {
    method: "DELETE",
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

export function getSegmentationTypes(): Promise<SegmentationType[]> {
  return apiRequest<SegmentationType[]>("/api/segmentation-types/");
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
