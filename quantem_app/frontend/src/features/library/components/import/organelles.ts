/**
 * The four runs the import form can queue, and what is ticked to begin with.
 *
 * Split out of `ImageUploadPanel.tsx` unchanged, and kept apart from
 * `ImportRunOptions.tsx` so that file exports components only.
 */

/**
 * The four runs this form can queue, in the order they are offered.
 *
 * `id` matches `DEFAULT_PACK_FOR_ORGANELLE`, which mirrors the server's
 * `default_source_model_for_organelle`, so the pack named in a warning is the
 * pack the import will actually use.
 */
export const ORGANELLE_CHOICES = [
  { id: "mito", inputId: "segment-mito", label: "Segment mitochondria" },
  { id: "er", inputId: "segment-er", label: "Segment ER" },
  { id: "nucleus", inputId: "segment-nucleus", label: "Segment nucleus" },
  { id: "ld", inputId: "segment-ld", label: "Segment lipid droplets" },
] as const;

export type OrganelleId = (typeof ORGANELLE_CHOICES)[number]["id"];

/**
 * What to tick before the user has touched anything: nothing.
 *
 * Each box is one whole-image inference pass on the CPU, minutes to tens of
 * minutes, *per image*, and the only way to stop one once it is queued is the
 * Library's job sidebar. The person importing has not yet seen the images, let
 * alone decided which organelles they care about. Every organelle is one tick
 * away here, and "Run Full Segmentation" on the labeling screen queues the
 * identical pass once an image is open and calibrated -- which is the point at
 * which the choice can actually be made well.
 */
export const NO_ORGANELLES_TICKED: OrganelleId[] = [];
