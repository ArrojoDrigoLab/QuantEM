/** Composition measured within each named Analysis Mask object. */

import { Button, Panel } from "@/shared/ui/design";
import { formatInteger, formatNumber, formatPercent } from "@/shared/ui/format";
import { PixelSizeTag } from "@/shared/ui/PixelSize";
import type { AnalysisComposition } from "@/shared/types/analysis";
import { downloadUrl } from "@/utils/downloadText";

export interface CompositionPanelProps {
  composition: AnalysisComposition;
  calibrated: boolean;
  pixelSizeNm: number | null;
  wholeImageDenominator: boolean;
  compositionCsvUrl?: string | null;
}

export function CompositionPanel({
  composition,
  calibrated,
  pixelSizeNm,
  wholeImageDenominator,
  compositionCsvUrl = null,
}: CompositionPanelProps) {
  const regions =
    composition.regions && composition.regions.length > 0
      ? composition.regions
      : [
          {
            id: "",
            name: wholeImageDenominator ? "Whole image" : "Combined analysis area",
            tissue_px: composition.tissue_px,
            tissue_um2: composition.tissue_um2,
            area_fractions: composition.area_fractions,
            areas_px: composition.areas_px,
            areas_um2: composition.areas_um2,
          },
        ];
  const names = Array.from(
    new Set(regions.flatMap((region) => Object.keys(region.area_fractions)))
  ).sort();
  const hasDerivedCytoplasm = names.includes("cytoplasm") && names.includes("nucleus");

  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 text-sm font-semibold text-slate-900">Composition</h3>
        <div className="flex flex-wrap items-center gap-2">
          <PixelSizeTag valueNm={calibrated ? pixelSizeNm : null} />
          {compositionCsvUrl ? (
            <Button
              size="sm"
              onClick={() =>
                downloadUrl("composition.csv", compositionCsvUrl, "text/csv")
              }
            >
              Download Composition Metrics
            </Button>
          ) : null}
        </div>
      </div>

      <dl className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">Analysis area</dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {formatInteger(composition.tissue_px)} px
          </dd>
        </div>
        {calibrated ? (
          <div>
            <dt className="text-xs uppercase tracking-wide text-slate-500">Analysis area</dt>
            <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
              {formatNumber(composition.tissue_um2, 2)} µm²
            </dd>
          </div>
        ) : null}
        <div>
          <dt className="text-xs uppercase tracking-wide text-slate-500">Compartments</dt>
          <dd className="m-0 text-sm font-semibold tabular-nums text-slate-900">
            {names.length}
          </dd>
        </div>
      </dl>

      <div className="mt-3 overflow-x-auto">
        <table className="w-full min-w-[640px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
              <th className="py-1 pr-3 font-semibold">Analysis Mask object</th>
              <th className="py-1 pr-3 font-semibold normal-case">Total area (px)</th>
              {calibrated ? (
                <th className="py-1 pr-3 font-semibold normal-case">Total area (µm²)</th>
              ) : null}
              {names.flatMap((name) => [
                <th key={`${name}-fraction`} className="py-1 pr-3 font-semibold normal-case">
                  {name} area fraction
                </th>,
                <th key={`${name}-px`} className="py-1 pr-3 font-semibold normal-case">
                  {name} area (px)
                </th>,
                ...(calibrated
                  ? [
                      <th key={`${name}-um2`} className="py-1 pr-3 font-semibold normal-case">
                        {name} area (µm²)
                      </th>,
                    ]
                  : []),
              ])}
            </tr>
          </thead>
          <tbody>
            {regions.map((region) => (
              <tr key={region.id || region.name} className="border-b border-slate-100">
                <td className="py-1 pr-3 font-medium text-slate-700">{region.name}</td>
                <td className="py-1 pr-3 tabular-nums text-slate-900">
                  {formatInteger(region.tissue_px)}
                </td>
                {calibrated ? (
                  <td className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatNumber(region.tissue_um2, 3)}
                  </td>
                ) : null}
                {names.flatMap((name) => [
                  <td key={`${name}-fraction`} className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatPercent(region.area_fractions[name])}
                  </td>,
                  <td key={`${name}-px`} className="py-1 pr-3 tabular-nums text-slate-900">
                    {formatInteger(region.areas_px[name])}
                  </td>,
                  ...(calibrated
                    ? [
                        <td key={`${name}-um2`} className="py-1 pr-3 tabular-nums text-slate-900">
                          {formatNumber(region.areas_um2?.[name] ?? null, 3)}
                        </td>,
                      ]
                    : []),
                ])}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="mt-3 mb-0 text-xs text-slate-500">
        {wholeImageDenominator
          ? "Fractions are of the whole image because this run had no Analysis Mask."
          : "Each row is measured independently within that named Analysis Mask object."}
        {hasDerivedCytoplasm
          ? " Cytoplasm is derived as the analysis area minus nucleus; organelle compartments are subsets of it."
          : ""}
      </p>
    </Panel>
  );
}
