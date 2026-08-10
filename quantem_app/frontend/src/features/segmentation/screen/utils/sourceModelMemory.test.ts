/**
 * The family toggle's default must follow the objects, not the catalogue.
 *
 * The bug: `is_default` marks QuantEM on every organelle, so reopening the
 * labeling screen reset the toggle to QuantEM even when all 34 candidates came
 * from an OmniEM run — invisible behind "No objects from QuantEM yet" until
 * the user clicked the toggle. When no family owns the objects outright, the
 * last family the user chose for THAT segmentation wins, remembered in
 * localStorage per segmentation id.
 */

import { beforeEach, describe, expect, it } from "vitest";
import {
  defaultSourceModel,
  recallSourceModel,
  rememberSourceModel,
} from "@/features/segmentation/screen/utils/sourceModelMemory";
import type { SourceModelOption } from "@/shared/types/images";

function option(overrides: Partial<SourceModelOption> = {}): SourceModelOption {
  return {
    value: "quantem:mito",
    label: "QuantEM",
    model_family: "quantem",
    variant: "",
    is_default: true,
    count: 0,
    ...overrides,
  };
}

const QUANTEM = option();
const OMNIEM = option({
  value: "omniem:mito",
  label: "OmniEM",
  model_family: "omniem",
  is_default: false,
});
const MANUAL = option({
  value: "manual",
  label: "Manual",
  model_family: "manual",
  is_default: false,
});

describe("defaultSourceModel", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("is null with no options at all", () => {
    expect(defaultSourceModel([], "seg-1")).toBeNull();
  });

  it("falls back to the catalogue default on a fresh segmentation", () => {
    expect(defaultSourceModel([QUANTEM, OMNIEM, MANUAL], "seg-1")).toBe(
      "quantem:mito"
    );
  });

  it("defaults to the family that owns the objects, not is_default", () => {
    // The uat13 #4 repro: an OmniEM run made every object, QuantEM has none.
    const options = [QUANTEM, { ...OMNIEM, count: 34 }, MANUAL];
    expect(defaultSourceModel(options, "seg-1")).toBe("omniem:mito");
  });

  it("never prefers a remembered family with zero objects over the one that has them", () => {
    rememberSourceModel("seg-1", "quantem:mito");
    const options = [QUANTEM, { ...OMNIEM, count: 34 }, MANUAL];
    expect(defaultSourceModel(options, "seg-1")).toBe("omniem:mito");
  });

  it("uses the remembered choice when both families own objects", () => {
    rememberSourceModel("seg-1", "omniem:mito");
    const options = [
      { ...QUANTEM, count: 17 },
      { ...OMNIEM, count: 34 },
      MANUAL,
    ];
    expect(defaultSourceModel(options, "seg-1")).toBe("omniem:mito");
  });

  it("prefers the default family among owners when nothing is remembered", () => {
    const options = [
      { ...QUANTEM, count: 17 },
      { ...OMNIEM, count: 34 },
      MANUAL,
    ];
    expect(defaultSourceModel(options, "seg-1")).toBe("quantem:mito");
  });

  it("picks the biggest owner when the default family owns nothing", () => {
    const options = [QUANTEM, { ...OMNIEM, count: 34 }, { ...MANUAL, count: 2 }];
    expect(defaultSourceModel(options, "seg-1")).toBe("omniem:mito");
  });

  it("uses the remembered choice on a fresh segmentation with no objects", () => {
    rememberSourceModel("seg-1", "omniem:mito");
    expect(defaultSourceModel([QUANTEM, OMNIEM, MANUAL], "seg-1")).toBe(
      "omniem:mito"
    );
  });

  it("keeps memories apart per segmentation", () => {
    rememberSourceModel("seg-1", "omniem:mito");
    expect(defaultSourceModel([QUANTEM, OMNIEM, MANUAL], "seg-2")).toBe(
      "quantem:mito"
    );
  });

  it("accepts a remembered synthetic 'none' selection", () => {
    rememberSourceModel("seg-1", "none");
    expect(defaultSourceModel([QUANTEM, OMNIEM, MANUAL], "seg-1")).toBe("none");
  });

  it("ignores a remembered value that is no longer offered", () => {
    rememberSourceModel("seg-1", "mitonet:mito");
    expect(defaultSourceModel([QUANTEM, OMNIEM, MANUAL], "seg-1")).toBe(
      "quantem:mito"
    );
  });

  it("treats a missing count as zero (older backend)", () => {
    const options = [
      { ...QUANTEM, count: undefined },
      { ...OMNIEM, count: undefined },
    ];
    expect(defaultSourceModel(options, "seg-1")).toBe("quantem:mito");
  });
});

describe("rememberSourceModel / recallSourceModel", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("round-trips a choice", () => {
    rememberSourceModel("seg-1", "omniem:mito");
    expect(recallSourceModel("seg-1")).toBe("omniem:mito");
  });

  it("does nothing without a segmentation id", () => {
    rememberSourceModel(null, "omniem:mito");
    expect(recallSourceModel(null)).toBeNull();
  });
});
