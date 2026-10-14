import { apiRequest } from "./core/http";

export interface AnalysisHistoryEntry {
  id: string; kind: "batch" | "run"; name: string; created_at: string; status: string;
  images: { id: string; name: string }[]; structures: string[];
  measurement_count: number; measurement_area: string; completed_image_ids: string[];
}
export interface BatchIdentity {
  image: string; asset_id: string; organelle: string; sample_id: string; group: string;
  measurement_area: string; pixel_size_nm: number | null;
}
export interface BatchItem {
  identity: BatchIdentity; reason: string; ready?: boolean; status?: string;
  current?: boolean | null; reused?: boolean; run_id?: string;
  metrics?: Record<string, number | null>; params?: Record<string, unknown>;
}
export interface AnalysisBatch {
  id: string; name: string; created_at: string; settings: Record<string, unknown>; items: BatchItem[];
}
export const getAnalysisHistory = (experimentId: string, imageId?: string) =>
  apiRequest<AnalysisHistoryEntry[]>(`/api/experiments/${experimentId}/analysis-history/${imageId ? `?asset_id=${encodeURIComponent(imageId)}` : ""}`);
export const getAnalysisBatch = (id: string) => apiRequest<AnalysisBatch>(`/api/analysis-batches/${id}/`);
