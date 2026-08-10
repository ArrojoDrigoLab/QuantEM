/**
 * No unit may be rendered under `text-transform: uppercase`.
 *
 * CSS uppercase maps U+00B5 MICRO SIGN to U+039C GREEK CAPITAL MU, whose glyph
 * is indistinguishable from a Latin M in every font this app ships. The
 * composition table's `Area (µm²)` header therefore rendered as **AREA (MM²)**
 * beside a value of 4.848 µm² — a factor of 10^6 on the number a reader copies
 * straight into a figure legend. The biologist who found it read it as mm²
 * before inspecting anything.
 *
 * It is invisible in code review: the source says µ and only the render says M.
 * So the guard is a test over the rendered DOM rather than a comment, and it
 * covers every panel rather than the one line that was wrong.
 */

import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CompositionPanel } from "./CompositionPanel";

/** Characters that change meaning, not just case, under text-transform. */
const CASE_UNSAFE = ["µ"]; // MICRO SIGN -> GREEK CAPITAL MU

function uppercasedTextIn(container: HTMLElement): string[] {
  const found: string[] = [];
  for (const el of Array.from(container.querySelectorAll<HTMLElement>("*"))) {
    // jsdom does not cascade text-transform, so walk the class lists the way
    // Tailwind applies them: an ancestor's `uppercase` reaches this text.
    let node: HTMLElement | null = el;
    let uppercased = false;
    while (node) {
      if (node.classList.contains("normal-case")) break;
      if (node.classList.contains("uppercase")) {
        uppercased = true;
        break;
      }
      node = node.parentElement;
    }
    if (!uppercased) continue;
    for (const child of Array.from(el.childNodes)) {
      if (child.nodeType === Node.TEXT_NODE && child.textContent?.trim()) {
        found.push(child.textContent);
      }
    }
  }
  return found;
}

const CALIBRATED = {
  tissue_px: 1_000_000,
  tissue_um2: 25.0,
  area_fractions: { mito: 0.05, nucleus: 0.1 },
  areas_px: { mito: 50_000, nucleus: 100_000 },
  areas_um2: { mito: 1.25, nucleus: 2.5 },
};

describe("units are never uppercased", () => {
  it("does not render a micro sign under text-transform: uppercase", () => {
    const { container } = render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator={false}
      />,
    );

    const offenders = uppercasedTextIn(container).filter((text) =>
      CASE_UNSAFE.some((ch) => text.includes(ch)),
    );

    expect(
      offenders,
      "µ becomes M under CSS uppercase — µm² renders as MM², a factor of 10^6",
    ).toEqual([]);
  });

  it("still shows the µm² column when the run is calibrated", () => {
    const { container } = render(
      <CompositionPanel
        composition={CALIBRATED}
        calibrated
        pixelSizeNm={5}
        wholeImageDenominator={false}
      />,
    );
    expect(container.textContent).toContain("µm²");
    expect(container.textContent).not.toContain("MM²");
  });
});
