/**
 * The failure-action rule `notices.css` adds has to be readable.
 *
 * The labeling header has already lost a number to a 1.04:1 contrast ratio
 * once (`.apply-full-progress`, invisible through two waves of progress work),
 * and the fix that time was to assert the *ratio* rather than the hex so the
 * rule can be restyled freely and can never go back. Same treatment here: the
 * failure action is a small link inside the failed-run notice.
 */

import { describe, expect, it } from "vitest";
import css from "./notices.css?raw";

function channel(value: number): number {
  const c = value / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminance(hex: string): number {
  const h = hex.replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

function contrastRatio(foreground: string, background: string): number {
  const a = luminance(foreground);
  const b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

function ruleBody(selector: string): string {
  const start = css.indexOf(`${selector} {`);
  if (start === -1) throw new Error(`no rule for ${selector}`);
  return css.slice(start, css.indexOf("}", start));
}

function declaration(body: string, property: string): string {
  const match = new RegExp(`(?:^|[;{\\n])\\s*${property}:\\s*(#[0-9a-fA-F]{6})`).exec(
    body
  );
  if (!match) throw new Error(`no ${property} in ${body}`);
  return match[1];
}

/** `.header-failed-notice` in SegmentationHeader.css, which this link sits in. */
const FAILED_NOTICE_BACKGROUND = "#fee2e2";

describe("the surfaces notices.css added", () => {
  it("keeps the failure action legible on the red notice it sits in", () => {
    // Not on the header's own grey: this link is inside `.header-failed-notice`,
    // and checking it against the wrong background is how a link ends up
    // passing a test and disappearing on screen.
    const colour = declaration(ruleBody(".header-failure-action"), "color");

    expect(contrastRatio(colour, FAILED_NOTICE_BACKGROUND)).toBeGreaterThanOrEqual(
      4.5
    );
  });
});
