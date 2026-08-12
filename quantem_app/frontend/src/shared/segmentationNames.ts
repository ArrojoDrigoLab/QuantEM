import type { ImageSegmentation } from "@/shared/types/images";

/** The name shown to people: an analysis-mask label, or its reusable type. */
export function segmentationDisplayName(segmentation: ImageSegmentation): string {
  return segmentation.display_name?.trim() || segmentation.segmentation_type.long_name;
}

/** A compact label for rows in the viewer's segmentation list. */
export function segmentationShortName(segmentation: ImageSegmentation): string {
  return segmentation.display_name?.trim() || segmentation.segmentation_type.short_name;
}

/**
 * Preserve readable paths for type-named segmentations while keeping each
 * image-specific analysis mask unambiguous, even if its label matches a type.
 */
export function segmentationRouteToken(segmentation: ImageSegmentation): string {
  return segmentation.display_name?.trim()
    ? segmentation.id
    : segmentation.segmentation_type.long_name;
}
