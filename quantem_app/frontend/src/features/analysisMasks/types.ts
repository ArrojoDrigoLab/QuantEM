export type AnalysisMaskOperation = "include" | "exclude";

export interface CellTag {
  name: string;
  value: string;
}

export interface CellTagSuggestion {
  name: string;
  values: string[];
}

export type AnalysisMaskGeometry =
  | {
      type: "Polygon";
      coordinates: number[][][];
    }
  | {
      type: "MultiPolygon";
      coordinates: number[][][][];
    };

export interface AnalysisMaskObject {
  id: string;
  segmentation: string;
  name: string;
  color: string;
  sort_order: number;
  tags: CellTag[];
  geometry: AnalysisMaskGeometry | null;
  created_at: string;
  updated_at: string;
}

export interface AnalysisMaskObjectListResponse {
  objects: AnalysisMaskObject[];
  foreground_pixels: number;
  tag_suggestions?: CellTagSuggestion[];
}

export interface AnalysisMaskObjectSaveResponse
  extends AnalysisMaskObjectListResponse {
  overlay: unknown;
}

export interface AnalysisMaskObjectMutationResponse {
  object: AnalysisMaskObject;
  foreground_pixels: number;
  overlay: unknown;
}

export interface AnalysisMaskObjectDeleteResponse {
  deleted_id: string;
  foreground_pixels: number;
  overlay: unknown;
}

export interface AnalysisMaskShape {
  rings: Array<Array<[number, number]>>;
}
