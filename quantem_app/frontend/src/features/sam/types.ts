/** Wire types for box-prompted object adding. Mirrors `quantem.sam.views`. */

import type { SegmentationOverlayMutationState } from "@/shared/types/segmentation";

export interface SamCandidate {
  geometry_coords: Array<[number, number]>;
  score: number;
  area: number;
}

export interface SamBoxTiming {
  /** False for the first box in a crop window -- that one paid the encoder. */
  cache_hit: boolean;
  encode_ms: number;
  decode_ms: number;
  device: string;
}

export interface SamBoxResponse {
  created: number;
  updated: number;
  deleted: number;
  confirmed_ids: string[];
  overlay: SegmentationOverlayMutationState;
  measurement?: unknown;
  /** The mask that was stored. */
  object: SamCandidate;
  /** The masks that were not, kept so a future "try the next one" is cheap. */
  other_candidates: SamCandidate[];
  timing: SamBoxTiming;
}

export interface SamModelDownloadState {
  status: "IDLE" | "RUNNING" | "SUCCESS" | "FAILED";
  bytes_done: number;
  bytes_total: number;
  error: string;
  percent: number | null;
}

export interface SamModelStatus {
  model: string;
  installed: boolean;
  download: SamModelDownloadState;
  size_bytes: number;
  stub_mode?: boolean;
}
