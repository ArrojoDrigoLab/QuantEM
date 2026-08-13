import { describe, expect, it } from "vitest";
import {
  describeDevice,
  noPackIsRunnable,
  packIdForSourceModel,
  packRunnability,
  runnabilityForPackId,
} from "@/features/models/runnable";
import type { ModelCatalogue, ModelPack } from "@/shared/types/finetune";

function pack(overrides: Partial<ModelPack> = {}): ModelPack {
  return {
    id: "quantem:mito",
    family: "quantem",
    organelle: "mito",
    title: "QuantEM — Mitochondria",
    installed: true,
    download_bytes: 662337373,
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
    encoder_tier: "exported",
    ...overrides,
  };
}

function catalogue(packs: ModelPack[]): ModelCatalogue {
  return { packs, adapted: [], device: { kind: "cpu", name: "CPU", cuda: false, mps: false } };
}

describe("packRunnability", () => {
  it("reports a runnable pack", () => {
    expect(packRunnability(pack()).state).toBe("runnable");
  });

  it("carries the server's reason for a blocked pack", () => {
    // This is the real message for the four QuantEM packs on a machine with no
    // exported encoder: they install fine and then fail seconds into a run.
    const reason =
      "This pack's ViT-B needs Meta's `dinov3` package, which QuantEM does not " +
      "redistribute.";
    const result = packRunnability(pack({ runnable: false, reason }));

    expect(result.state).toBe("blocked");
    expect(result.reason).toBe(reason);
    expect(result.label).toBe("cannot run here");
  });

  it("treats a not-installed pack as downloadable, not blocked", () => {
    const result = packRunnability(
      pack({ installed: false, runnable: false, reason: "Not installed yet." })
    );
    expect(result.state).toBe("downloadable");
    expect(result.label).toBe("downloads on first run");
  });

  it("says 'unknown' when the backend omits the field, never 'blocked'", () => {
    // Guessing "blocked" would hide a model that works; guessing "runnable"
    // reproduces the original clean-install failure.
    const withoutField: Partial<ModelPack> = pack();
    delete withoutField.runnable;
    expect(packRunnability(withoutField as ModelPack).state).toBe("unknown");
  });

  it("says 'unknown' for a pack the catalogue never mentioned", () => {
    expect(packRunnability(null).state).toBe("unknown");
    expect(packRunnability(undefined).state).toBe("unknown");
  });
});

describe("runnabilityForPackId", () => {
  it("finds the pack in the catalogue", () => {
    const cat = catalogue([pack({ runnable: false, reason: "Not installed yet." })]);
    expect(runnabilityForPackId(cat, "quantem:mito").state).toBe("blocked");
  });

  it("is unknown without a catalogue", () => {
    expect(runnabilityForPackId(null, "quantem:mito").state).toBe("unknown");
  });

  it("is unknown for an id the catalogue does not list", () => {
    expect(runnabilityForPackId(catalogue([pack()]), "omniem:er").state).toBe("unknown");
  });
});

describe("packIdForSourceModel", () => {
  it("maps a source model onto the pack of the same name", () => {
    expect(packIdForSourceModel("quantem:mito")).toBe("quantem:mito");
  });

  it("has no pack for manual or none", () => {
    expect(packIdForSourceModel("manual")).toBeNull();
    expect(packIdForSourceModel("none")).toBeNull();
    expect(packIdForSourceModel(null)).toBeNull();
  });
});

describe("noPackIsRunnable", () => {
  it("is false on a clean install because packs download on first run", () => {
    const packs = ["quantem:mito", "omniem:mito"].map((id) =>
      pack({ id, installed: false, runnable: false, reason: "Not installed yet." })
    );
    expect(noPackIsRunnable(catalogue(packs))).toBe(false);
  });

  it("is false when at least one pack runs", () => {
    expect(noPackIsRunnable(catalogue([pack(), pack({ id: "omniem:mito", runnable: false })]))).toBe(
      false
    );
  });

  it("is false when there is no catalogue to judge", () => {
    expect(noPackIsRunnable(null)).toBe(false);
    expect(noPackIsRunnable(catalogue([]))).toBe(false);
  });
});

describe("describeDevice", () => {
  it("names the accelerator when there is one", () => {
    const cat = catalogue([]);
    cat.device = { kind: "cuda", name: "RTX A5000", cuda: true, mps: false };
    expect(describeDevice(cat)).toBe("RTX A5000 (CUDA)");
  });

  it("names a plain CPU without decoration", () => {
    expect(describeDevice(catalogue([]))).toBe("CPU");
  });

  it("is null when the catalogue did not answer", () => {
    expect(describeDevice(null)).toBeNull();
  });
});
