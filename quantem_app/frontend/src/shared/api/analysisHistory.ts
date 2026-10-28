import { apiRequest } from "./core/http";

export interface AnalysisHistoryEntry {
  id: string; kind: "batch" | "run"; name: string; created_at: string; status: string;
  images: { id: string; name: string }[]; structures: string[];
  superseded_count?: number; superseded_scope?: "all" | "part" | null;
  measurement_count: number; measurement_area: string; completed_image_ids: string[];
}
export interface BatchIdentity {
  structure?: string;
  segmentation_type_id?: string | null;
  image: string; asset_id: string; organelle: string; sample_id: string; group: string;
  measurement_area: string; pixel_size_nm: number | null;
}
export interface BatchItem {
  identity: BatchIdentity; reason: string; ready?: boolean; status?: string;
  current?: boolean | null; reused?: boolean; run_id?: string;
  metrics?: Record<string, number | null>; params?: Record<string, unknown>;
  notes?: string[];
}
export interface AnalysisBatch {
  id: string; name: string; created_at: string; settings: Record<string, unknown>; items: BatchItem[];
}
export interface AnalysisHistoryPage {
  results: AnalysisHistoryEntry[];
  count: number; offset: number; limit: number; next_offset: number | null;
}
export const getAnalysisHistory = (experimentId: string, imageId?: string, offset = 0) => {
  const params = new URLSearchParams({ offset: String(offset) });
  if (imageId) params.set("asset_id", imageId);
  return apiRequest<AnalysisHistoryPage>(`/api/experiments/${experimentId}/analysis-history/?${params}`);
};
export const getAnalysisBatch = (id: string) => apiRequest<AnalysisBatch>(`/api/analysis-batches/${id}/`);
