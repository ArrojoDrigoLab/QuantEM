/** HTTP for box-prompted object adding. One function per route, like the rest. */

import { apiRequest } from "@/shared/api/core/http";
import type { BBox } from "@/shared/types/common";
import type { SamBoxResponse, SamModelStatus } from "./types";

/** Segment whatever the box encloses and store the best mask as an object. */
export function promptSamBox(
  segmentationId: string,
  bbox: BBox
): Promise<SamBoxResponse> {
  return apiRequest<SamBoxResponse>(
    `/api/sam/segmentations/${segmentationId}/box/`,
    {
      method: "POST",
      body: JSON.stringify({ box: bbox }),
    }
  );
}

/** Are the weights on this machine, and if a download is running, how far along. */
export function getSamModelStatus(): Promise<SamModelStatus> {
  return apiRequest<SamModelStatus>("/api/sam/model/");
}

/** Start the download. Safe to call twice -- an in-flight one is not restarted. */
export function startSamModelDownload(): Promise<SamModelStatus> {
  return apiRequest<SamModelStatus>("/api/sam/model/download/", {
    method: "POST",
  });
}
