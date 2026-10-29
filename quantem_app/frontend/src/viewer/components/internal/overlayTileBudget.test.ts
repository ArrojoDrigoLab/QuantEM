import { describe, expect, it, vi } from "vitest";
import { accountImageTile, accountOverlayTile, OverlayTileBudget } from "./overlayTileBudget";
import { BudgetedImageLayer } from "./BudgetedImageLayer";
import type { VivMultiscaleLayerProps } from "./vivUtils";

describe("shared overlay tile budget", () => {
  it("charges both panes and old/replacement generations to the same allowance", () => {
    const budget = new OverlayTileBudget(128 * 1024 * 1024);
    const owners = Array.from({ length: 8 }, () => Symbol());
    const changed = vi.fn();
    for (const owner of owners) budget.register(owner, changed);
    expect(budget.bytesPerLayer * owners.length).toBe(budget.totalBytes);
    budget.register(owners[0], changed);
    expect(budget.bytesPerLayer).toBe(16 * 1024 * 1024);
    for (const owner of owners.slice(1)) budget.release(owner);
    expect(budget.bytesPerLayer).toBe(budget.totalBytes);
    expect(changed).toHaveBeenCalled();
  });

  it("accounts for decoded arrays and the derived rendering allocations", () => {
    const content = { width: 16, height: 32,
      data: [new Uint32Array(512), new Uint8Array(512)], byteLength: 0 };
    accountOverlayTile({ content });
    expect(content.byteLength).toBe(512 * 18);
    accountOverlayTile({ content });
    expect(content.byteLength).toBe(512 * 18);
  });

  it("charges base-image channels and float textures too", () => {
    const content = { width: 16, height: 32, data: [new Uint16Array(512)], byteLength: 0 };
    accountImageTile({ content });
    expect(content.byteLength).toBe(512 * 6);
  });

  it("accounts for interleaved RGB tiles without NaN or per-scalar textures", () => {
    const content = { width: 16, height: 32, data: new Uint8Array(512 * 3), byteLength: 0 };
    accountImageTile({ content });
    expect(content.byteLength).toBe(512 * 7);
  });

  it("preserves Viv's color shader extension in the budgeted wrapper", () => {
    const layer = new BudgetedImageLayer({ id: "base" } as VivMultiscaleLayerProps);
    const rendered = layer.renderLayers();
    expect(rendered.props.extensions?.length).toBeGreaterThan(0);
    expect(rendered.props.extensions?.[0].constructor.name).toContain("ColorPaletteExtension");
  });
});
