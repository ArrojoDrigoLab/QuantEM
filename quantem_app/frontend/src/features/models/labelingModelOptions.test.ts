import { describe, expect, it } from "vitest";
import {
  buildLabelingModelOptions,
  defaultLabelingModel,
} from "@/features/models/labelingModelOptions";
import type { SourceModelOption } from "@/shared/types/images";
import type {
  AdaptedModelEntry,
  ModelCatalogue,
  ModelPack,
} from "@/shared/types/finetune";

const SOURCES: SourceModelOption[] = [
  { value: "quantem:mito", label: "QuantEM", model_family: "quantem" },
  { value: "omniem:mito", label: "OmniEM", model_family: "omniem" },
  { value: "manual", label: "Manual", model_family: "manual" },
];

function pack(id: "quantem:mito" | "omniem:mito", installed: boolean): ModelPack {
  const family = id.startsWith("quantem:") ? "quantem" : "omniem";
  return {
    id,
    family,
    organelle: "mito",
    title: family,
    installed,
    download_bytes: 1,
    canonical_nm: 8,
    tile_size: 512,
    default_threshold: 0.5,
    decoder: "decoder",
    neck: "neck",
    adapt: "head",
    licence: "licence",
    notes: "",
  };
}

function adapted(overrides: Partial<AdaptedModelEntry> = {}): AdaptedModelEntry {
  return {
    id: "adapted:a1",
    base: "omniem:mito",
    name: "TESTFT",
    created_at: "2026-01-01T00:00:00Z",
    calibrated_threshold: 0.25,
    heldout_dice: 0.9,
    split_mode: "image-disjoint",
    scope_segmentation_ids: ["seg-1"],
    ...overrides,
  };
}

function catalogue(
  adapters: AdaptedModelEntry[],
  installed: [boolean, boolean] = [true, true]
): ModelCatalogue {
  return {
    packs: [pack("quantem:mito", installed[0]), pack("omniem:mito", installed[1])],
    adapted: adapters,
    device: null,
  };
}

describe("labeling model options", () => {
  it("lists each applicable fine-tune separately without replacing its base", () => {
    const models = catalogue([
      adapted(),
      adapted({ id: "adapted:other", name: "Other", scope_segmentation_ids: ["seg-2"] }),
    ]);
    const options = buildLabelingModelOptions(SOURCES, models, "seg-1");

    expect(options.map((option) => [option.value, option.label])).toContainEqual([
      "omniem:mito",
      "OmniEM",
    ]);
    expect(options.map((option) => [option.value, option.label])).toContainEqual([
      "adapted:a1",
      "TESTFT (fine-tuned model)",
    ]);
    expect(options.some((option) => option.value === "adapted:other")).toBe(false);
  });

  it("prefers the most recently run fine-tune for this image", () => {
    const models = catalogue([
      adapted({
        id: "adapted:old",
        created_at: "2026-01-01T00:00:00Z",
        last_run_at_by_segmentation: { "seg-1": "2026-03-01T00:00:00Z" },
      }),
      adapted({
        id: "adapted:new",
        created_at: "2026-02-01T00:00:00Z",
        last_run_at_by_segmentation: { "seg-1": "2026-02-15T00:00:00Z" },
      }),
    ]);
    const options = buildLabelingModelOptions(SOURCES, models, "seg-1");
    expect(defaultLabelingModel(options, models, "seg-1")).toBe("adapted:old");
  });

  it("then prefers the newest applicable fine-tune", () => {
    const models = catalogue([
      adapted({ id: "adapted:old", created_at: "2026-01-01T00:00:00Z" }),
      adapted({ id: "adapted:new", created_at: "2026-02-01T00:00:00Z" }),
    ]);
    const options = buildLabelingModelOptions(SOURCES, models, "seg-1");
    expect(defaultLabelingModel(options, models, "seg-1")).toBe("adapted:new");
  });

  it("uses the sole downloaded base and OmniEM when both or neither are downloaded", () => {
    for (const [installed, expected] of [
      [[true, false], "quantem:mito"],
      [[false, true], "omniem:mito"],
      [[true, true], "omniem:mito"],
      [[false, false], "omniem:mito"],
    ] as const) {
      const models = catalogue([], [...installed]);
      const options = buildLabelingModelOptions(SOURCES, models, "seg-1");
      expect(defaultLabelingModel(options, models, "seg-1")).toBe(expected);
    }
  });
});
