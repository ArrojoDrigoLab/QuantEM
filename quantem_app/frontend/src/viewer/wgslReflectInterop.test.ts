/**
 * Guards the `wgsl_reflect` pin.
 *
 * `@luma.gl/shadertools` does `import { WgslReflect } from 'wgsl_reflect'`.
 * From 1.3.0 that package declares `"type": "module"` while its `main` still
 * points at a CommonJS build and it publishes no `exports` map, so Node's ESM
 * loader parses `wgsl_reflect.node.js` as ESM, finds no `export` statements,
 * and throws
 *
 *   SyntaxError: The requested module 'wgsl_reflect' does not provide an
 *   export named 'WgslReflect'
 *
 * Vitest externalises bare dependencies to that loader, so the failure took out
 * every suite that transitively imports Viv or deck.gl -- all five viewer
 * suites, i.e. the viewer's entire test coverage. Vite's browser build picks
 * the `module` entry and never sees it, which is why `npm run build` stayed
 * green throughout.
 *
 * `package.json` therefore pins `wgsl_reflect` to 1.2.0 via `overrides`. 1.2.0
 * is inside luma's own `^1.2.0` range, omits `"type": "module"` so Node reads
 * the CJS build as CommonJS, and exposes the identical `uniforms` / `textures`
 * / `samplers` / `entry` surface that `getShaderLayoutFromWGSL` consumes.
 *
 * This test fails in one line if the pin is lifted, instead of five suites
 * failing to load with a message that points at Viv.
 */

import { describe, expect, it } from "vitest";
import { getShaderLayoutFromWGSL } from "@luma.gl/shadertools";

const WGSL_SOURCE = `
  @group(0) @binding(0) var<uniform> scale: vec2<f32>;
  @vertex fn main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos * scale, 0.0, 1.0);
  }
`;

describe("wgsl_reflect interop", () => {
  it("loads @luma.gl/shadertools, which is what the export break blocked", async () => {
    // Importing the module at all is the assertion: the SyntaxError was thrown
    // while linking, before any of its exports could be touched.
    expect(typeof getShaderLayoutFromWGSL).toBe("function");
  });

  it("still reflects the shader surface luma reads out of wgsl_reflect", () => {
    // `getShaderLayoutFromWGSL` logs and returns an empty layout on a parse
    // failure, so a populated layout is what proves the pinned version's
    // uniforms/entry API is intact.
    const layout = getShaderLayoutFromWGSL(WGSL_SOURCE);

    expect(layout.bindings.map((binding) => binding.name)).toContain("scale");
    expect(layout.attributes.map((attribute) => attribute.name)).toContain("pos");
  });
});
