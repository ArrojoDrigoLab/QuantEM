/**
 * Form state for an analysis run, and the rules that turn it into a request.
 *
 * The validation here deliberately mirrors
 * `quantem.analysis.loaders.normalise_params`. The server stays authoritative —
 * its message is what gets shown when it refuses — but a desktop app that
 * queues a job and fails it three minutes later for a mistyped band edge is
 * worse than one that says so while the field is still focused.
 */

import type {
  AnalysisRunCreatePayload,
  AnalysisPointsSource,
} from "@/shared/types/analysis";
import type { ImageSegmentation } from "@/shared/types/images";

export const DEFAULT_BAND_EDGES_NM = [0, 50, 100, 200];
export const DEFAULT_REPLICATES = 20;
export const DEFAULT_SEED = 12345;
/** `quantem.analysis.loaders.MAX_REPLICATES`. */
export const MAX_REPLICATES = 1000;

export const TISSUE_INTERNAL_NAME = "quantem_internal_tissue";
export const ANALYSIS_MASK_INTERNAL_NAME = "quantem_internal_analysis_mask";

/**
 * Mirror of `quantem.analysis.loaders.BUILTIN_COMPARTMENT_NAMES`.
 *
 * These names decide the exported column headers (`area_fraction_mito`), so
 * the default offered here has to be the same string the backend would have
 * chosen on its own.
 */
export const BUILTIN_COMPARTMENT_NAMES: Record<string, string> = {
  quantem_internal_mito: "mito",
  quantem_internal_er: "er",
  quantem_internal_nucleus: "nucleus",
  quantem_internal_ld: "ld",
  quantem_internal_tissue: "tissue",
};

/** A type the user created themselves keeps its own internal name. */
export function compartmentNameFor(segmentation: ImageSegmentation): string {
  const internal = segmentation.segmentation_type.internal_name;
  return BUILTIN_COMPARTMENT_NAMES[internal] ?? internal;
}

export interface CompartmentSelection {
  segmentationId: string;
  /** Label the compartment carries into every exported column. */
  name: string;
  enabled: boolean;
}

export type PointsSourceChoice = "none" | "centroids" | "csv";

export interface AnalysisFormState {
  compartments: CompartmentSelection[];
  /** "" means no analysis mask: the whole image is the denominator. */
  tissueSegmentationId: string;
  pointsSource: PointsSourceChoice;
  pointsCsv: string;
  /** "" means no distance analysis. */
  distanceTarget: string;
  bandEdgesText: string;
  replicates: number;
  seed: number;
  group: string;
}

export function formatBandEdges(edges: number[]): string {
  return edges.join(", ");
}

export function defaultFormState(
  segmentations: ImageSegmentation[],
  primarySegmentationId: string
): AnalysisFormState {
  const analysisMask = segmentations.find(
    (seg) => seg.segmentation_type.internal_name === ANALYSIS_MASK_INTERNAL_NAME
  );
  return {
    compartments: segmentations
      .filter(
        (seg) =>
          seg.segmentation_type.internal_name !== TISSUE_INTERNAL_NAME &&
          seg.segmentation_type.internal_name !== ANALYSIS_MASK_INTERNAL_NAME
      )
      .map((seg) => ({
        segmentationId: seg.id,
        name: compartmentNameFor(seg),
        // Only the segmentation the screen was opened on is on by default:
        // every extra compartment is another full-resolution rasterisation.
        enabled: seg.id === primarySegmentationId,
      })),
    tissueSegmentationId: analysisMask?.id ?? "",
    pointsSource: "none",
    pointsCsv: "",
    distanceTarget: "",
    bandEdgesText: formatBandEdges(DEFAULT_BAND_EDGES_NM),
    replicates: DEFAULT_REPLICATES,
    seed: DEFAULT_SEED,
    group: "",
  };
}

/**
 * Why the replicate count is out of range, or null when it is fine.
 *
 * Exported because the *field* has to be able to say this while it is still
 * focused, not only the submit handler. `min`/`max` on a number input are
 * decorative in every browser -- typing 20000 into `max={1000}` is accepted,
 * looks accepted, and the only thing that ever objected was a message rendered
 * beside the Run button, which on a 1280x720 window is 65px below the fold with
 * the page unscrolled. A limit nothing enforces where you can see it is a limit
 * you find out about by having your run not happen.
 *
 * `buildAnalysisPayload` calls this too, so the field and the refusal cannot
 * word the same rule differently.
 */
export function replicatesError(replicates: number): string | null {
  if (!Number.isInteger(replicates) || replicates < 1) {
    return "Replicates must be a whole number of at least 1.";
  }
  if (replicates > MAX_REPLICATES) {
    return `Replicates must be ${MAX_REPLICATES} or fewer; beyond that this wants a batch script, not a desktop button.`;
  }
  return null;
}

/** Same shape for the seed: the API takes any whole number, including negatives. */
export function seedError(seed: number): string | null {
  return Number.isInteger(seed) ? null : "The seed must be a whole number.";
}

export interface BandEdgeParse {
  edges: number[] | null;
  error: string | null;
}

export function parseBandEdges(text: string): BandEdgeParse {
  const parts = text
    .split(/[,\s]+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (parts.length < 2) {
    return { edges: null, error: "Distance bands need at least two edges." };
  }
  const edges: number[] = [];
  for (const part of parts) {
    const value = Number(part);
    if (!Number.isFinite(value)) {
      return { edges: null, error: `"${part}" is not a number.` };
    }
    edges.push(value);
  }
  for (let index = 1; index < edges.length; index += 1) {
    if (edges[index] <= edges[index - 1]) {
      return { edges: null, error: "Band edges must increase." };
    }
  }
  return { edges, error: null };
}

export interface PayloadResult {
  payload: AnalysisRunCreatePayload | null;
  error: string | null;
}

export function buildAnalysisPayload(state: AnalysisFormState): PayloadResult {
  const enabled = state.compartments.filter((entry) => entry.enabled);
  if (enabled.length === 0) {
    return { payload: null, error: "Choose at least one compartment to measure." };
  }

  const compartments: Record<string, string> = {};
  for (const entry of enabled) {
    const name = entry.name.trim();
    if (!name) {
      return { payload: null, error: "A compartment name cannot be blank." };
    }
    if (compartments[name]) {
      return {
        payload: null,
        error: `Two compartments are both called "${name}". Names become column headers, so they have to differ.`,
      };
    }
    compartments[name] = entry.segmentationId;
  }

  const { edges, error: bandError } = parseBandEdges(state.bandEdgesText);
  if (bandError || !edges) {
    return { payload: null, error: bandError };
  }

  const replicates = replicatesError(state.replicates);
  if (replicates) {
    return { payload: null, error: replicates };
  }
  const seed = seedError(state.seed);
  if (seed) {
    return { payload: null, error: seed };
  }

  const pointsSource: AnalysisPointsSource =
    state.pointsSource === "none" ? null : state.pointsSource;
  if (pointsSource === "csv" && !state.pointsCsv.trim()) {
    return { payload: null, error: "Paste or load an x,y CSV before running." };
  }

  const distanceTarget = state.distanceTarget.trim();
  if (distanceTarget) {
    if (!(distanceTarget in compartments)) {
      return {
        payload: null,
        error: `The distance target "${distanceTarget}" is not one of the selected compartments.`,
      };
    }
    if (pointsSource === null) {
      return {
        payload: null,
        error: "A distance target needs a point set; choose a point source first.",
      };
    }
  }

  return {
    payload: {
      compartments,
      tissue_segmentation_id: state.tissueSegmentationId || null,
      points_source: pointsSource,
      points_csv: pointsSource === "csv" ? state.pointsCsv : "",
      distance_target: distanceTarget || null,
      band_edges_nm: edges,
      replicates: state.replicates,
      seed: state.seed,
      group: state.group.trim(),
    },
    error: null,
  };
}
