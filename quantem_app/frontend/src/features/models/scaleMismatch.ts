/**
 * Running a pack that declares a working resolution on an image that has none.
 *
 * Six of the eight released packs resample to a `canonical_nm` before they run.
 * `predict_region` logs "expects 8.0 nm/px but the asset has no pixel_size_nm;
 * running at native scale" and carries on, which is the exact condition that
 * makes every downstream number wrong: the objects are whatever that mismatch
 * produced, and nothing measured from them can be reported in µm².
 *
 * This lives on its own because there are **three** doors into that run and they
 * were not guarded alike. Creating a segmentation opened a written confirmation
 * naming the pack and the resolution; "Run Full Segmentation" on the labeling
 * screen fired instantly with no dialog at all, on the same image, for the same
 * inference pass; and the import form — the screen *everybody starts on* — ticks
 * four organelles by default and queues the same passes with nothing said. One
 * check, one sentence, all three doors.
 *
 * `canonical_nm === null` (both ER packs) genuinely runs at native scale by
 * design, so there is nothing to warn about there — warning anyway would train
 * people to click through the warning that matters.
 */

import type { ModelCatalogue } from "@/shared/types/finetune";

export interface ScaleMismatch {
  packId: string;
  /** The resolution the pack was trained at, in nm/px. */
  canonicalNm: number;
}

/**
 * The mismatch, or null when there is none to report.
 *
 * `pixelSizeNm === undefined` means the caller does not know the image's pixel
 * size, which is a different thing from knowing it is unset: say nothing rather
 * than guess. A catalogue that has not answered also yields null — an unknown
 * pack cannot be claimed to declare a canonical resolution.
 */
export function scaleMismatchForPack(
  catalogue: ModelCatalogue | null | undefined,
  packId: string | null | undefined,
  pixelSizeNm: number | null | undefined
): ScaleMismatch | null {
  if (!catalogue || !packId) return null;
  if (pixelSizeNm === undefined) return null;
  if (typeof pixelSizeNm === "number" && pixelSizeNm > 0) return null;
  const pack = catalogue.packs.find((candidate) => candidate.id === packId);
  if (!pack || pack.canonical_nm == null) return null;
  return { packId: pack.id, canonicalNm: pack.canonical_nm };
}

/**
 * The pack an organelle's run is actually queued with, keyed by the id the UI
 * uses for that organelle.
 *
 * Mirrors `default_source_model_for_organelle` on the server, which is what
 * both `POST /api/assets/<id>/segmentations/` and the upload pipeline's
 * `segment_*` flags resolve to. Kept in one place because two screens now need
 * it — the create dialog and the import form — and a private copy in each is
 * how they would come to name different models for the same run.
 *
 * The server remains the authority on what actually runs; this only decides
 * what the warning is allowed to claim.
 */
export const DEFAULT_PACK_FOR_ORGANELLE: Record<string, string> = {
  mito: "quantem:mito",
  er: "quantem:er",
  nucleus: "quantem:nucleus",
  ld: "quantem:ld",
};

/**
 * Every mismatch a batch of organelles would produce, in the order given.
 *
 * The import form queues up to four runs from one button, and they do not all
 * declare a working resolution: ER runs at native scale by design and must not
 * appear here, or the warning stops meaning anything. Duplicate packs are
 * collapsed so the sentence never names the same model twice.
 */
export function scaleMismatchesForOrganelles(
  catalogue: ModelCatalogue | null | undefined,
  organelleIds: readonly string[],
  pixelSizeNm: number | null | undefined
): ScaleMismatch[] {
  const seen = new Set<string>();
  const mismatches: ScaleMismatch[] = [];
  for (const organelleId of organelleIds) {
    const packId = DEFAULT_PACK_FOR_ORGANELLE[organelleId];
    if (!packId || seen.has(packId)) continue;
    const mismatch = scaleMismatchForPack(catalogue, packId, pixelSizeNm);
    if (!mismatch) continue;
    seen.add(packId);
    mismatches.push(mismatch);
  }
  return mismatches;
}
