import { describe, expect, it } from "vitest";
import {
  buildAnalysisPayload,
  compartmentNameFor,
  defaultFormState,
  parseBandEdges,
  type AnalysisFormState,
} from "@/features/analysis/analysisOptions";
import type { ImageSegmentation } from "@/shared/types/images";

function makeSegmentation(
  id: string,
  internalName: string,
  longName = internalName,
  category: "organelle" | "area" = "organelle"
): ImageSegmentation {
  return {
    id,
    segmentation_type: {
      id: `type-${id}`,
      internal_name: internalName,
      short_name: longName,
      long_name: longName,
      category,
      representation: category === "area" ? "semantic" : "instances",
      topology_policy: "non_overlapping",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    has_saved_area: true,
    is_complete: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}

const MITO = makeSegmentation("seg-mito", "quantem_internal_mito", "Mitochondria");
const NUCLEUS = makeSegmentation("seg-nuc", "quantem_internal_nucleus", "Nucleus");
const TISSUE = makeSegmentation("seg-tis", "quantem_internal_tissue", "Sample Area", "area");
const CELLS = makeSegmentation("seg-cells", "quantem_internal_cells", "Cells", "area");
const ANALYSIS_MASK = makeSegmentation(
  "seg-mask",
  "quantem_internal_analysis_mask",
  "Analysis Mask",
  "area"
);
const CUSTOM = makeSegmentation("seg-x", "my_own_thing", "My own thing");

function baseState(overrides: Partial<AnalysisFormState> = {}): AnalysisFormState {
  return {
    ...defaultFormState([MITO, NUCLEUS, TISSUE, CELLS, ANALYSIS_MASK], MITO.id),
    ...overrides,
  };
}

describe("compartmentNameFor", () => {
  it("uses the analysis vocabulary for built-in types", () => {
    expect(compartmentNameFor(MITO)).toBe("mito");
    expect(compartmentNameFor(NUCLEUS)).toBe("nucleus");
  });

  it("leaves a user-created type's own name alone", () => {
    expect(compartmentNameFor(CUSTOM)).toBe("my_own_thing");
  });
});

describe("defaultFormState", () => {
  it("includes complete zero-object predictions and excludes partial manual organelles", () => {
    const zero = { ...MITO, has_saved_area: false, is_complete: true };
    const partial = { ...NUCLEUS, has_saved_area: true, is_complete: false };
    const state = defaultFormState([zero, partial], zero.id);
    expect(state.compartments.map((entry) => entry.enabled)).toEqual([true, false]);
    expect(buildAnalysisPayload(state).payload?.compartments).toEqual({ mito: zero.id });
    state.compartments[1].enabled = true;
    expect(buildAnalysisPayload(state).error).toMatch(/whole image/);
  });

  it("carries explicit structured-file choices into the request", () => {
    const state = defaultFormState([MITO], MITO.id);
    state.outputFiles = { ...state.outputFiles, "organelle_objects.csv": false };
    expect(buildAnalysisPayload(state).payload?.output_files?.["organelle_objects.csv"]).toBe(false);
  });

  it("preselects populated Sample Area and Cells layers", () => {
    const state = defaultFormState([MITO, NUCLEUS, TISSUE, CELLS], MITO.id);
    expect(state.tissueSegmentationId).toBe(TISSUE.id);
    expect(state.cellSegmentationId).toBe(CELLS.id);
    expect(state.outputs).toEqual({
      whole_image: true,
      sample_area: true,
      all_cells: true,
      per_cell: true,
      individual_objects: true,
    });
  });

  it("keeps areas out of compartments and enables every organelle with data", () => {
    const state = defaultFormState([MITO, NUCLEUS, TISSUE, CELLS, ANALYSIS_MASK], MITO.id);
    expect(state.compartments.map((entry) => entry.segmentationId)).toEqual([
      MITO.id,
      NUCLEUS.id,
    ]);
    expect(state.compartments.filter((entry) => entry.enabled)).toHaveLength(2);
    expect(state.compartments[0].enabled).toBe(true);
  });
});

describe("parseBandEdges", () => {
  it("accepts commas and whitespace", () => {
    expect(parseBandEdges("0, 50 100,200").edges).toEqual([0, 50, 100, 200]);
  });

  it("rejects a single edge, non-numbers and a non-increasing sequence", () => {
    expect(parseBandEdges("50").error).toMatch(/two edges/);
    expect(parseBandEdges("0, fifty").error).toMatch(/not a number/);
    expect(parseBandEdges("0, 100, 50").error).toMatch(/increase/);
  });
});

describe("buildAnalysisPayload", () => {
  it("maps enabled compartments to name -> segmentation id", () => {
    const { payload, error } = buildAnalysisPayload(baseState());
    expect(error).toBeNull();
    expect(payload?.compartments).toEqual({ mito: MITO.id, nucleus: NUCLEUS.id });
    expect(payload?.tissue_segmentation_id).toBe(TISSUE.id);
    expect(payload?.cell_segmentation_id).toBe(CELLS.id);
    expect(payload?.outputs?.per_cell).toBe(true);
    expect(payload?.band_edges_nm).toEqual([0, 50, 100, 200]);
  });

  it("refuses two compartments with the same column header", () => {
    const state = baseState();
    state.compartments = state.compartments.map((entry) => ({
      ...entry,
      enabled: true,
      name: "mito",
    }));
    expect(buildAnalysisPayload(state).error).toMatch(/both called/);
  });

  it("refuses a blank compartment name", () => {
    const state = baseState();
    state.compartments = [{ segmentationId: MITO.id, name: "  ", enabled: true }];
    expect(buildAnalysisPayload(state).error).toMatch(/cannot be blank/);
  });

  it("refuses a distance target with no point source", () => {
    const state = baseState({ distanceTarget: "mito" });
    expect(buildAnalysisPayload(state).error).toMatch(/needs a point set/);
  });

  it("refuses a distance target that is not a selected compartment", () => {
    const state = baseState({ pointsSource: "centroids", distanceTarget: "missing" });
    expect(buildAnalysisPayload(state).error).toMatch(/not one of the selected/);
  });

  it("refuses an empty CSV when the source is csv", () => {
    const state = baseState({ pointsSource: "csv", pointsCsv: "  " });
    expect(buildAnalysisPayload(state).error).toMatch(/x,y CSV/);
  });

  it("drops the CSV body when the source is not csv", () => {
    const state = baseState({ pointsSource: "centroids", pointsCsv: "x,y\n1,2" });
    expect(buildAnalysisPayload(state).payload?.points_csv).toBe("");
    expect(buildAnalysisPayload(state).payload?.points_source).toBe("centroids");
  });

  it("sends null rather than a source when no points were chosen", () => {
    expect(buildAnalysisPayload(baseState()).payload?.points_source).toBeNull();
  });

  it("enforces the same replicate ceiling as the server", () => {
    expect(buildAnalysisPayload(baseState({ replicates: 0 })).error).toMatch(
      /at least 1/
    );
    expect(buildAnalysisPayload(baseState({ replicates: 1001 })).error).toMatch(
      /1000 or fewer/
    );
  });

  it("refuses to run with nothing selected", () => {
    const state = baseState();
    state.compartments = state.compartments.map((entry) => ({
      ...entry,
      enabled: false,
    }));
    expect(buildAnalysisPayload(state).error).toMatch(/at least one compartment/);
  });
});
