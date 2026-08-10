/**
 * Which source model the labeling screen should show first.
 *
 * The bug this exists for: `is_default` marks the app's default family
 * (QuantEM, on every organelle), so reopening the labeling screen reset the
 * family toggle to QuantEM even when every object in the segmentation came
 * from OmniEM — 34 fresh candidates invisible behind "No objects from QuantEM
 * yet" until the user happened to click the toggle. The default must follow
 * the objects, not the catalogue.
 *
 * Rules, in order (an explicit `?source_model=` URL param still wins upstream
 * of this module — this only decides the fallback when the URL says nothing):
 *
 * 1. If any option owns objects (`count > 0`), the answer is one of those
 *    owners: the remembered per-segmentation choice if it is an owner, else
 *    the catalogue default if it is an owner, else the owner with the most
 *    objects. A family with zero objects is never defaulted to over one that
 *    has them.
 * 2. With no objects anywhere (a fresh segmentation), the remembered choice
 *    for THIS segmentation wins, then the catalogue default, then the first
 *    option.
 *
 * The remembered choice is written only when the user actually changes the
 * toggle (`handleSourceModelChange`), keyed by segmentation id in
 * localStorage — per segmentation, because "last family used on the mito seg"
 * says nothing about the ER seg next door.
 */

import { NONE_SOURCE_MODEL } from "@/features/segmentation/components/segmentationHeaderProvenance";
import type { SourceModelOption } from "@/shared/types/images";

const STORAGE_KEY_PREFIX = "quantem.labeling.source-model.";

function storageKey(segmentationId: string): string {
  return `${STORAGE_KEY_PREFIX}${segmentationId}`;
}

/** Persist the user's explicit toggle choice for one segmentation. */
export function rememberSourceModel(
  segmentationId: string | null,
  value: string
): void {
  if (!segmentationId || !value) return;
  try {
    window.localStorage.setItem(storageKey(segmentationId), value);
  } catch {
    // Storage can be full or disabled; losing the memory only means the
    // objects-first default below decides, which is never wrong data.
  }
}

/** The last value the user explicitly chose for this segmentation, if any. */
export function recallSourceModel(segmentationId: string | null): string | null {
  if (!segmentationId) return null;
  try {
    return window.localStorage.getItem(storageKey(segmentationId));
  } catch {
    return null;
  }
}

function isSelectable(options: SourceModelOption[], value: string): boolean {
  // "none" is the synthetic "confirmed/manual only" selection the toggle also
  // offers; it never appears in `options` but is a real remembered choice.
  return (
    value === NONE_SOURCE_MODEL ||
    options.some((option) => option.value === value)
  );
}

/**
 * The source model to select when the URL names none.
 *
 * Pure given its inputs plus the per-segmentation memory; see the module
 * comment for the ordering.
 */
export function defaultSourceModel(
  options: SourceModelOption[],
  segmentationId: string | null
): string | null {
  if (options.length === 0) return null;

  const remembered = recallSourceModel(segmentationId);
  const owners = options.filter((option) => (option.count ?? 0) > 0);

  if (owners.length > 0) {
    if (remembered && owners.some((option) => option.value === remembered)) {
      return remembered;
    }
    const defaultOwner = owners.find((option) => option.is_default);
    if (defaultOwner) return defaultOwner.value;
    // Most objects wins; a tie keeps the server's order.
    return owners.reduce((best, option) =>
      (option.count ?? 0) > (best.count ?? 0) ? option : best
    ).value;
  }

  if (remembered && isSelectable(options, remembered)) {
    return remembered;
  }
  return (
    options.find((option) => option.is_default)?.value ??
    options[0]?.value ??
    null
  );
}
