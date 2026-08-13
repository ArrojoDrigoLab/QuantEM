/**
 * Area composition: how much of the tissue each compartment occupies.
 *
 * Honesty rule 6 is enforced structurally here — the µm² column only exists
 * when the run reported a pixel size. An uncalibrated run shows pixels and says
 * so, rather than showing a physical unit it cannot support.
 */

import { Badge, Panel } from "@/shared/ui/design";
import { formatInteger, formatNumber, formatPercent } from "@/shared/ui/format";
import { PixelSizeTag } from "@/shared/ui/PixelSize";
import type { AnalysisComposition } from "@/shared/types/analysis";

export interface CompositionPanelProps {
  composition: AnalysisComposition;
  calibrated: boolean;
  pixelSizeNm: number | null;
  /** True when the run had no tissue mask: every fraction includes resin. */
  wholeImageDenominator: boolean;
}

export function CompositionPanel({
  composition,
  calibrated,
  pixelSizeNm,
  wholeImageDenominator,
}: CompositionPanelProps) {
  const names = Object.keys(composition.area_fractions).sort();
  const hasDerivedCytoplasm =
    "cytoplasm" in composition.area_fractions &&
    "nucleus" in composition.area_fractions;

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-sm font-semibold text-slate-900">Composition</h3>
        <div className="flex flex-wrap items-center gap-2">
          <PixelSizeTag valueNm={calibrated ? pixelSizeNm : null} />
          {wholeImageDenominator ? (
            <Badge tone="warning">Whole image as denominator</Badge>
          ) : (
            <Badge tone="info">Restricted to the tissue mask</Badge>
          )}
        </div>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Tissue area
          </dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {formatInteger(composition.tissue_px)} px
          </dd>
        </div>
        {calibrated ? (
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">
              Tissue area
            </dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {formatNumber(composition.tissue_um2, 2)} µm²
            </dd>
          </div>
        ) : null}
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">
            Compartments
          </dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {names.length}
          </dd>
        </div>
      </dl>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[420px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1 pr-3 font-semibold">Compartment</th>
              <th className="py-1 pr-3 font-semibold">Area fraction</th>
              <th className="py-1 pr-3 font-semibold">Area (px)</th>
              {/* normal-case on the header below is load-bearing, not styling:
                  CSS text-transform: uppercase maps U+00B5 MICRO SIGN to
                  U+039C GREEK CAPITAL MU, glyph-identical to a Latin M, so this
                  rendered as "AREA (MM²)" beside a value in µm² -- a factor of
                  10^6 on the number a reader copies into a figure legend, and
                  invisible in review because the source says µ and only the
                  render says M. Guarded by unitCase.test.tsx. */}
              {calibrated ? (
                <th className="py-1 font-semibold normal-case">Area (µm²)</th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name} className="border-b border-slate-100">
                <td className="py-1 pr-3 text-slate-700">{name}</td>
                <td className="py-1 pr-3 tabular-nums text-slate-900">
                  {formatPercent(composition.area_fractions[name])}
                </td>
                <td className="py-1 pr-3 tabular-nums text-slate-900">
                  {formatInteger(composition.areas_px?.[name])}
                </td>
                {calibrated ? (
                  <td className="py-1 tabular-nums text-slate-900">
                    {formatNumber(composition.areas_um2?.[name] ?? null, 3)}
                  </td>
                ) : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* The footer must agree with the header badge. With no tissue mask the
          denominator IS the whole image, and "Fractions are of tissue area,
          not of the image" beside "Whole image as denominator" is a flat
          contradiction — so the sentence flips with the badge. */}
      <p className="mt-3 mb-0 text-xs text-slate-500">
        {wholeImageDenominator
          ? "Fractions are of the whole image — this run had no tissue mask, so resin and background count in the denominator."
          : "Fractions are of tissue area, not of the image."}
        {hasDerivedCytoplasm
          ? " cytoplasm is derived as tissue minus nucleus; organelle compartments are subsets of it, so these fractions do not sum to 1."
          : ""}
      </p>
    </Panel>
  );
}
