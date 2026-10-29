import { BitmapLayer } from "@deck.gl/layers";
import type { LayerContext, UpdateParameters } from "@deck.gl/core";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

export const NEAREST_SAMPLER = { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" } as const;

const idmap = {
  name: "idmap",
  fs: `precision highp int;
uniform sampler2D bitmapTexture;
uniform sampler2D labelPalette;
uniform sampler2D labelBorder;
uniform idmapUniforms {
  uint maxLabel;
  uint highlightedLabel;
  float fillOpacity;
  float borderOpacity;
  float showBorders;
} idmap;
`,
  uniformTypes: {
    maxLabel: "u32", highlightedLabel: "u32", fillOpacity: "f32",
    borderOpacity: "f32", showBorders: "f32",
  },
} as const satisfies ShaderModule;

interface PaletteProps {
  palette: Texture;
  border: { data: Uint8Array; width: number; height: number };
  maxLabel: number;
  highlightedLabel: number;
  fillOpacity: number;
  borderOpacity: number;
  showBorders: boolean;
  idLabels?: Uint32Array | Uint16Array | Uint8Array;
  idTileWidth?: number;
}

/** Labels are packed losslessly into RGBA8 once; changes only update uniforms/LUT. */
export class PaletteBitmapLayer extends BitmapLayer<PaletteProps> {
  static layerName = "PaletteBitmapLayer";
  declare state: BitmapLayer<PaletteProps>["state"] & { borderTexture?: Texture };

  getShaders() {
    const shaders = super.getShaders();
    return {
      ...shaders,
      // Hook functions precede the layer's source. Declare its existing sampler
      // in our module so it is in scope without binding a second texture.
      fs: shaders.fs.replace("uniform sampler2D bitmapTexture;", ""),
      modules: [...shaders.modules, idmap],
      inject: {
        ...shaders.inject,
        "fs:DECKGL_FILTER_COLOR": `
  ivec2 size = textureSize(bitmapTexture, 0);
  ivec2 pixel = clamp(ivec2(geometry.uv * vec2(size)), ivec2(0), size - 1);
  // Reuse BitmapLayer's texture binding instead of binding the public image
  // prop a second time; that prop may be CPU texture data rather than a GPU
  // Texture. Sampling directly also avoids BitmapLayer's tint/opacity pass,
  // which would corrupt the four packed label bytes.
  uvec4 bytes = uvec4(round(texelFetch(bitmapTexture, pixel, 0) * 255.0));
  uint label = bytes.r | (bytes.g << 8u) | (bytes.b << 16u) | (bytes.a << 24u);
  if (label == 0u || label > idmap.maxLabel) discard;
  uint paletteWidth = uint(textureSize(labelPalette, 0).x);
  vec4 tint = texelFetch(labelPalette, ivec2(int(label % paletteWidth), int(label / paletteWidth)), 0);
  if (tint.a == 0.0) discard;
  bool border = texelFetch(labelBorder, pixel, 0).r > 0.0;
  if (label == idmap.highlightedLabel) {
    color = vec4(0.0, 1.0, 1.0, border ? 1.0 : max(idmap.fillOpacity, 0.25));
  } else if (border && idmap.showBorders > 0.5) {
    color = vec4(tint.rgb * 0.5, idmap.borderOpacity * tint.a);
  } else {
    color = vec4(tint.rgb, idmap.fillOpacity * tint.a);
  }
  color.a *= layer.opacity;
  // Deck's picking pass replaces alpha with an object identifier. Discard
  // invisible fragments first so they cannot intercept objects beneath them.
  if (color.a <= 0.0) discard;
`,
      },
    };
  }

  updateState(params: UpdateParameters<this>) {
    super.updateState(params);
    if (params.props.border !== params.oldProps.border || !this.state.borderTexture) {
      this.state.borderTexture?.destroy();
      const { data, width, height } = this.props.border;
      this.setState({ borderTexture: this.context.device.createTexture({
        data, width, height, format: "r8unorm", sampler: NEAREST_SAMPLER,
      }) });
    }
  }

  draw(options: Parameters<BitmapLayer["draw"]>[0]) {
    if (!this.state.borderTexture) return;
    this.state.model?.shaderInputs.setProps({ idmap: {
      labelPalette: this.props.palette, labelBorder: this.state.borderTexture,
      maxLabel: this.props.maxLabel, highlightedLabel: this.props.highlightedLabel,
      fillOpacity: this.props.fillOpacity, borderOpacity: this.props.borderOpacity,
      showBorders: this.props.showBorders ? 1 : 0,
    } });
    super.draw(options);
  }

  finalizeState(context: LayerContext) {
    this.state.borderTexture?.destroy();
    super.finalizeState(context);
  }
}
