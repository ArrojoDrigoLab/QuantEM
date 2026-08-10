import type { HomeEntry } from "@/shared/types/images";

export function getStageDisplay(image: HomeEntry): string {
  if (image.is_workable && !image.ngff_ready) return "NGFF pending";
  if (image.preprocess_stage === "DONE" || image.preprocess_stage === "SKIPPED") {
    return "Ready";
  }
  if (
    image.preprocess_stage === "ENCODING" ||
    image.preprocess_stage === "SAM" ||
    image.preprocess_stage === "FEATURES"
  ) {
    return `Encoding image (${Math.round(image.preprocess_progress)}%)`;
  }
  if (image.preprocess_stage === "FAILED") return "Failed";
  if (image.preprocess_stage === "CANCELLED") return "Cancelled";
  return image.ngff_ready ? "Ready" : "Preprocessing queued";
}

export function isLibraryEntryProcessing(image: HomeEntry): boolean {
  return !["DONE", "FAILED", "CANCELLED", "NONE", "SKIPPED"].includes(
    image.preprocess_stage
  );
}
