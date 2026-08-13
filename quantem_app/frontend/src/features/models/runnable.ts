/**
 * Can this model actually run here, and if not, what do we tell the user?
 *
 * `GET /api/models/` answers this per pack with `runnable` / `reason`, and
 * `probe_runnable`'s own docstring says it exists "so the picker can grey the
 * pack out and say why". Nothing in the client read those fields, so on a clean
 * install the Adapt wizard offered all eight packs as available, the user
 * picked QuantEM — Mitochondria, and the run died several seconds in with a
 * message the job queue then replaced.
 *
 * The four states are deliberately distinct:
 *
 *  - **runnable** — `engine.load_model` would succeed.
 *  - **downloadable** — its files are missing, and the first run will install
 *    and verify them before loading the model.
 *  - **blocked** — it would fail, and `reason` says why. Never offer this as a
 *    selectable option.
 *  - **unknown** — the catalogue did not answer, or answered without the field
 *    (an older backend). Offer it, but do not claim it works.
 *
 * Collapsing "unknown" into either of the others is the bug this module exists
 * to prevent: guessing "blocked" hides a working model, and guessing "runnable"
 * reproduces the original failure.
 */

import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";

export type RunnabilityState =
  | "runnable"
  | "downloadable"
  | "blocked"
  | "unknown";

export interface Runnability {
  state: RunnabilityState;
  /** The server's sentence, when it gave one. */
  reason: string | null;
  /** Short label for a badge. */
  label: string;
}

const UNKNOWN: Runnability = {
  state: "unknown",
  reason: null,
  label: "run state unknown",
};

/** Runnability of one pack, from the catalogue entry (null = not in catalogue). */
export function packRunnability(pack: ModelPack | null | undefined): Runnability {
  if (!pack) return UNKNOWN;
  // A clean install reports `runnable: false` because there are no weights to
  // load yet. That is preparation work, not an incompatibility: every picker
  // keeps the pack selectable and the run path downloads it before inference.
  if (!pack.installed) {
    return {
      state: "downloadable",
      reason: null,
      label: "downloads on first run",
    };
  }
  if (typeof pack.runnable !== "boolean") return UNKNOWN;
  if (pack.runnable) {
    return { state: "runnable", reason: null, label: "ready to run" };
  }
  return {
    state: "blocked",
    reason: pack.reason ?? null,
    label: "cannot run here",
  };
}

/** Runnability of a pack id, looked up in a catalogue that may be missing. */
export function runnabilityForPackId(
  catalogue: ModelCatalogue | null | undefined,
  packId: string | null | undefined
): Runnability {
  if (!catalogue || !packId) return UNKNOWN;
  return packRunnability(catalogue.packs.find((pack) => pack.id === packId));
}

/**
 * A pack id derived from a source-model value.
 *
 * Source models and pack ids share the `family:organelle` spelling
 * (`quantem:mito`), so a labeling screen's selected source model maps straight
 * onto a catalogue entry. `"manual"` and `"none"` are not models and have no
 * pack.
 */
export function packIdForSourceModel(
  sourceModel: string | null | undefined
): string | null {
  if (!sourceModel) return null;
  if (!sourceModel.includes(":")) return null;
  return sourceModel;
}

/** Every released pack that cannot run, for a one-line summary. */
export function blockedPacks(catalogue: ModelCatalogue | null): ModelPack[] {
  return (catalogue?.packs ?? []).filter(
    (pack) => packRunnability(pack).state === "blocked"
  );
}

/** True when no released pack on this machine can run. */
export function noPackIsRunnable(catalogue: ModelCatalogue | null): boolean {
  const packs = catalogue?.packs ?? [];
  if (packs.length === 0) return false;
  return packs.every((pack) => packRunnability(pack).state === "blocked");
}

/**
 * The catalogue's device line, e.g. `"CPU"` or `"NVIDIA RTX A5000 (CUDA)"`.
 *
 * Worth showing next to a run button: inference on CPU is the difference
 * between seconds and a minute per image, and the user is about to wait.
 */
export function describeDevice(catalogue: ModelCatalogue | null): string | null {
  const device = catalogue?.device;
  if (!device) return null;
  const accelerator = device.cuda ? " (CUDA)" : device.mps ? " (MPS)" : "";
  return `${device.name}${accelerator}`;
}
