import { describe, expect, it } from "vitest";
import { buildTintedLut } from "@/viewer/overlays/labelLut";

describe("buildTintedLut", () => {
  it("tints labels whose state is in the visible set at full alpha", () => {
    const { rgba } = buildTintedLut(
      [{ label: 2, state: "confirmed" }],
      "#ff8000",
      new Set(["confirmed"])
    );
    const offset = 2 * 4;
    expect(rgba[offset]).toBe(0xff);
    expect(rgba[offset + 1]).toBe(0x80);
    expect(rgba[offset + 2]).toBe(0x00);
    expect(rgba[offset + 3]).toBe(255);
  });

  it("leaves out-of-set labels fully transparent", () => {
    const { rgba } = buildTintedLut(
      [
        { label: 1, state: "confirmed" },
        { label: 3, state: "candidate" },
      ],
      "#00ff00",
      new Set(["confirmed"])
    );
    // Label 3 (candidate) is not in the visible set -> alpha 0 and no colour.
    const offset = 3 * 4;
    expect(rgba[offset]).toBe(0);
    expect(rgba[offset + 1]).toBe(0);
    expect(rgba[offset + 2]).toBe(0);
    expect(rgba[offset + 3]).toBe(0);
    // Label 1 (confirmed) is tinted.
    expect(rgba[1 * 4 + 3]).toBe(255);
  });

  it("reports the maximum label seen and sizes the palette accordingly", () => {
    const { rgba, maxLabel } = buildTintedLut(
      [
        { label: 0, state: "confirmed" },
        { label: 5, state: "candidate" },
        { label: 2, state: "refined" },
      ],
      "#123456",
      new Set(["confirmed", "refined"])
    );
    expect(maxLabel).toBe(5);
    expect(rgba.length).toBe((5 + 1) * 4);
  });

  it("returns a single-entry palette and maxLabel 0 for no objects", () => {
    const { rgba, maxLabel } = buildTintedLut([], "#ffffff", new Set(["confirmed"]));
    expect(maxLabel).toBe(0);
    expect(rgba.length).toBe(4);
    expect(rgba[3]).toBe(0);
  });
});
