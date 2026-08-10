/**
 * The base models a user can adapt, and how to describe them when the
 * catalogue endpoint is not there.
 *
 * `GET /api/models/` is the authority on what is installed, how big a download
 * would be, and what device is available. When it does not answer, the wizard
 * still has to let someone choose a base model — the ids are fixed by the
 * contract and by `quantem.inference.specs.MODEL_SPECS` — but it must not
 * invent the parts only the server knows. Hence `ModelChoice.pack === null`
 * meaning "install state unknown", rendered as exactly that.
 */

import type {
  AdaptedModelEntry,
  ModelCatalogue,
  ModelFamily,
  ModelPack,
  OrganelleKey,
} from "@/shared/types/finetune";

export const FAMILY_LABELS: Record<string, string> = {
  quantem: "QuantEM",
  omniem: "OmniEM",
};

export const ORGANELLE_LABELS: Record<string, string> = {
  mito: "Mitochondria",
  er: "Endoplasmic reticulum",
  nucleus: "Nucleus",
  ld: "Lipid droplets",
};

/** The eight released packs, ordered family-major. */
export const RELEASED_PACK_IDS: string[] = [
  "quantem:mito",
  "quantem:er",
  "quantem:nucleus",
  "quantem:ld",
  "omniem:mito",
  "omniem:er",
  "omniem:nucleus",
  "omniem:ld",
];

export interface ModelChoice {
  id: string;
  title: string;
  family: string;
  organelle: string;
  /** Null when the catalogue is unavailable: nothing about install state is known. */
  pack: ModelPack | null;
}

export function splitPackId(id: string): { family: string; organelle: string } {
  const [family = "", organelle = ""] = id.split(":");
  return { family, organelle };
}

export function packTitle(id: string): string {
  const { family, organelle } = splitPackId(id);
  const familyLabel = FAMILY_LABELS[family] ?? family;
  const organelleLabel = ORGANELLE_LABELS[organelle] ?? organelle;
  return `${familyLabel} — ${organelleLabel}`;
}

/**
 * Merge what the catalogue said with the fixed list of released ids.
 *
 * A pack the server knows about wins; one it did not mention is still offered
 * (the id is valid whether or not the catalogue is up), with `pack: null` so
 * the UI can say its install state is unknown rather than guess "not
 * installed".
 */
export function toModelChoices(catalogue: ModelCatalogue | null): ModelChoice[] {
  const byId = new Map<string, ModelPack>();
  for (const pack of catalogue?.packs ?? []) {
    byId.set(pack.id, pack);
  }
  const ids = [
    ...RELEASED_PACK_IDS,
    ...[...byId.keys()].filter((id) => !RELEASED_PACK_IDS.includes(id)),
  ];
  return ids.map((id) => {
    const pack = byId.get(id) ?? null;
    const { family, organelle } = splitPackId(id);
    return {
      id,
      title: pack?.title ?? packTitle(id),
      family: (pack?.family as ModelFamily | undefined) ?? family,
      organelle: (pack?.organelle as OrganelleKey | undefined) ?? organelle,
      pack,
    };
  });
}

/**
 * The base model that matches a segmentation's organelle, if any.
 *
 * Used only to preselect a sensible default — the user can pick anything, and
 * adapting a mitochondria model on nucleus annotations is their call to make.
 */
export function suggestBaseModel(
  internalName: string | undefined,
  choices: ModelChoice[]
): string | null {
  if (!internalName) return null;
  const organelle = internalName.replace(/^quantem_internal_/, "");
  const match = choices.find((choice) => choice.organelle === organelle);
  return match?.id ?? null;
}

export function adaptedForBase(
  catalogue: ModelCatalogue | null,
  baseModel: string
): AdaptedModelEntry[] {
  return (catalogue?.adapted ?? []).filter((entry) => entry.base === baseModel);
}
