/**
 * The run readout has to be readable.
 *
 * `.apply-full-progress` is the percentage beside Run Full Segmentation and the
 * only number on the labeling header that moves while a run goes. It shipped as
 * `color: #d1fae5` with no background of its own, on a `#f5f5f5` header: a
 * measured contrast of **1.04:1**, which is not "low contrast" so much as "not
 * on screen". Two waves of progress work went into making that number honest
 * and nobody could read it.
 *
 * This asserts the ratio rather than the hex, so the rule can be restyled
 * freely and can never go back to being invisible.
 */

import { describe, expect, it } from "vitest";
// The stylesheet as text rather than as a computed style: jsdom's cascade only
// sees rules the component under test imported, and the file is the thing that
// ships.
import css from "./SegmentationHeader.css?raw";

/** The labeling header's own background (`.segmentation-header`). */
const HEADER_BACKGROUND = "#f5f5f5";

function channel(value: number): number {
  const c = value / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

/** WCAG 2.x contrast ratio, 1:1 to 21:1. */
function contrastRatio(foreground: string, background: string): number {
  const a = luminance(foreground);
  const b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

function ruleBody(css: string, selector: string): string {
  const start = css.indexOf(`${selector} {`);
  if (start === -1) throw new Error(`no rule for ${selector}`);
  const end = css.indexOf("}", start);
  return css.slice(start, end);
}

function declaration(body: string, property: string): string | null {
  const match = new RegExp(`(?:^|[;{\\n])\\s*${property}:\\s*(#[0-9a-fA-F]{6})`).exec(
    body
  );
  return match ? match[1] : null;
}

describe("the run percentage on the labeling header", () => {
  it("is legible against whatever it actually sits on", () => {
    const body = ruleBody(css, ".apply-full-progress");
    const color = declaration(body, "color");
    expect(color).not.toBeNull();
    // No background of its own means it sits on the header itself, which is
    // exactly the case that measured 1.04:1.
    const background = declaration(body, "background") ?? HEADER_BACKGROUND;

    expect(contrastRatio(color!, background)).toBeGreaterThanOrEqual(4.5);
  });

  it("takes a tinted chip, like every other status on this header", () => {
    // The convention on this header is a tinted background plus dark text, not
    // a bare tint of text: `.source-model-provenance`, `.header-model-blocked`,
    // `.header-locked-notice` and `.header-adapter-notice` are all that shape.
    const body = ruleBody(css, ".apply-full-progress");
    expect(declaration(body, "background")).not.toBeNull();
  });

  it("keeps the spinner beside it visible too", () => {
    const body = ruleBody(css, ".apply-full-spinner");
    const leading = declaration(body, "border-top-color");
    expect(leading).not.toBeNull();
    const chip = declaration(ruleBody(css, ".apply-full-progress"), "background");
    expect(contrastRatio(leading!, chip ?? HEADER_BACKGROUND)).toBeGreaterThanOrEqual(
      3
    );
  });
});
