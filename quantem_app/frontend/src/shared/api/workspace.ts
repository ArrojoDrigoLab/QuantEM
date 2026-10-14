import { apiRequest } from "./core/http";

export interface WorkspaceSegmentation {
  id: string; type_id: string; type: string; name: string;
  measurement_mode: string; stage: string; confirmed_count: number;
  review_complete: boolean; provenance: Record<string, unknown>;
  latest_analysis?: { id: string; status: string; current: boolean | null } | null;
}
export interface WorkspaceImage {
  id: string; name: string; pixel_size_nm: number | null; stage: string;
  datasets: { id: string; name: string }[];
  sample_id: string; group: string; segmentations: WorkspaceSegmentation[];
}
export const getWorkspace = (id: string) =>
  apiRequest<{ images: WorkspaceImage[] }>(`/api/experiments/${id}/workspace/`);
export const getNavigationImages = (id: string) =>
  apiRequest<{ images: Pick<WorkspaceImage, "id" | "name">[] }>(`/api/experiments/${id}/workspace/?navigation=true`);
export const recordVisit = (id: string, route: string) =>
  apiRequest(`/api/experiments/${id}/navigation/`, {
    method: "POST", body: JSON.stringify({ route }),
  });
export const saveSample = (experiment: string, asset_id: string, sample_id: string, group: string) =>
  apiRequest(`/api/experiments/${experiment}/workspace/`, {
    method: "PATCH", body: JSON.stringify({ asset_id, sample_id, group }),
  });
