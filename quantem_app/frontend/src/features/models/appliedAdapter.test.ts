import { describe, expect, it } from "vitest";
import { appliedAdapterState, formatThreshold } from "@/features/models/appliedAdapter";
import type {
  AdaptedModelEntry,
  ModelCatalogue,
  ModelPack,
} from "@/shared/types/finetune";

function pack(overrides: Partial<ModelPack> = {}): ModelPack {
  return {
    id: "quantem:mito",
    family: "quantem",
    organelle: "mito",
    title: "QuantEM — Mitochondria",
    installed: true,
    download_bytes: 1,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "affinity_mws",
    neck: "naive_1x1",
    adapt: "last_n",
    licence: "see NOTICE",
    notes: "",
    runnable: true,
    reason: null,
    ...overrides,
  };
}

function adapted(overrides: Partial<AdaptedModelEntry> = {}): AdaptedModelEntry {
  return {
    id: "adapted:a1",
    base: "quantem:mito",
    name: "mito @ liver",
    created_at: "2026-01-01T00:00:00Z",
    calibrated_threshold: 0.45,
    heldout_dice: 0.9,
    split_mode: "image-disjoint",
    mode: "threshold_only",
    segmentation_id: "seg-1",
    applied_at: "2026-01-02T00:00:00Z",
    ...overrides,
  };
}

function catalogue(adapters: AdaptedModelEntry[]): ModelCatalogue {
  return {
    packs: [pack(), pack({ id: "omniem:mito", family: "omniem" })],
    adapted: adapters,
    device: null,
  };
}

describe("appliedAdapterState", () => {
  it("finds the adapter applied to this segmentation", () => {
    const state = appliedAdapterState(
      catalogue([adapted()]),
      "seg-1",
      "quantem:mito"
    );

    expect(state?.active).toBe(true);
    expect(state?.publishedThreshold).toBe(0.5);
    expect(state?.adapter.calibrated_threshold).toBe(0.45);
  });

  it("finds a named fine-tune through its applied target views", () => {
    const state = appliedAdapterState(
      catalogue([
        adapted({
          segmentation_id: null,
          segmentation_ids: ["seg-1", "seg-2"],
          name: "Test CV",
        }),
      ]),
      "seg-2",
      "quantem:mito"
    );

    expect(state?.active).toBe(true);
    expect(state?.adapter.name).toBe("Test CV");
  });

  it("ignores an adapter that was trained but never applied", () => {
    // `apply_active_adapter` filters on `applied_at`; training alone changes
    // nothing about a run.
    expect(
      appliedAdapterState(catalogue([adapted({ applied_at: null })]), "seg-1", "quantem:mito")
    ).toBeNull();
  });

  it("ignores an adapter applied to a different segmentation", () => {
    expect(
      appliedAdapterState(
        catalogue([adapted({ segmentation_id: "seg-other" })]),
        "seg-1",
        "quantem:mito"
      )
    ).toBeNull();
  });

  /**
   * The silent case. A threshold calibrated on quantem:mito describes that
   * model's probability distribution, so the backend refuses to reuse it for
   * omniem:mito and runs the released pack instead. The adapter is still
   * applied, so this must report it rather than return null.
   */
  it("reports an applied adapter the selected model will bypass", () => {
    const state = appliedAdapterState(
      catalogue([adapted()]),
      "seg-1",
      "omniem:mito"
    );

    expect(state).not.toBeNull();
    expect(state?.active).toBe(false);
    expect(state?.selectedSourceModel).toBe("omniem:mito");
  });

  it("takes the most recently applied adapter when several have been", () => {
    const state = appliedAdapterState(
      catalogue([
        adapted({ id: "adapted:old", applied_at: "2026-01-02T00:00:00Z" }),
        adapted({ id: "adapted:new", applied_at: "2026-03-09T00:00:00Z" }),
      ]),
      "seg-1",
      "quantem:mito"
    );

    expect(state?.adapter.id).toBe("adapted:new");
  });

  it("reports a trained head separately from a calibrated threshold", () => {
    expect(
      appliedAdapterState(catalogue([adapted({ mode: "head" })]), "seg-1", "quantem:mito")
        ?.trainedHead
    ).toBe(true);
    expect(
      appliedAdapterState(catalogue([adapted()]), "seg-1", "quantem:mito")?.trainedHead
    ).toBe(false);
  });

  it("says nothing without a catalogue", () => {
    expect(appliedAdapterState(null, "seg-1", "quantem:mito")).toBeNull();
  });
});

describe("formatThreshold", () => {
  it("always shows two places, so 0.5 and 0.45 are comparable at a glance", () => {
    expect(formatThreshold(0.5)).toBe("0.50");
    expect(formatThreshold(0.45)).toBe("0.45");
  });

  it("returns null rather than a fake number", () => {
    expect(formatThreshold(null)).toBeNull();
    expect(formatThreshold(undefined)).toBeNull();
  });
});
