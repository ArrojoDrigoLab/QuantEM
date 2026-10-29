import { BitmapLayer } from "@deck.gl/layers";
import type { ShaderModule } from "@luma.gl/shadertools";

const probability = {
  name: "probability",
  fs: `uniform sampler2D bitmapTexture;
uniform probabilityUniforms { float threshold; vec3 tint; } probability;`,
  uniformTypes: { threshold: "f32", tint: "vec3<f32>" },
} as const satisfies ShaderModule;

/** Threshold changes are uniform updates; the grayscale texture stays resident. */
export class ProbabilityBitmapLayer extends BitmapLayer<{
  threshold: number;
  color: [number, number, number];
}> {
  static layerName = "ProbabilityBitmapLayer";
  getShaders() {
    const shaders = super.getShaders();
    return { ...shaders, fs: shaders.fs.replace("uniform sampler2D bitmapTexture;", ""),
      modules: [...shaders.modules, probability], inject: {
      ...shaders.inject,
      "fs:DECKGL_FILTER_COLOR": `
        // Use BitmapLayer's existing texture binding. The public image prop may
        // be CPU texture data, so attempting to bind it as another sampler is
        // invalid on some WebGL implementations.
        vec4 sampleValue = texture(bitmapTexture, geometry.uv);
        if (sampleValue.a == 0.0 || sampleValue.r < probability.threshold) discard;
        float confidence = clamp((sampleValue.r - probability.threshold) / max(0.000001, 1.0 - probability.threshold), 0.0, 1.0);
        color = vec4(probability.tint, sampleValue.a * (0.4 + 0.5 * confidence) * layer.opacity);
      `,
    } };
  }
  draw(options: Parameters<BitmapLayer["draw"]>[0]) {
    this.state.model?.shaderInputs.setProps({ probability: {
      threshold: Math.min(1, Math.max(0, this.props.threshold)),
      tint: this.props.color.map((value) => value / 255),
    } });
    super.draw(options);
  }
}
