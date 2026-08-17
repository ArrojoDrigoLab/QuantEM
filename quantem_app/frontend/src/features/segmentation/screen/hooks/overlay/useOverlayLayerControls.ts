import { useMemo, useState } from "react";
import { RASTER_BORDER_OPACITY } from "@/shared/constants/segmentation";
import { clamp } from "@/features/segmentation/screen/utils/bbox";
import {
  useOverlayLut,
  useOverlayPickMap,
  type OverlayLutState,
} from "@/features/segmentation/screen/hooks/overlay/useOverlayLut";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { SegmentationOverlayManifest } from "@/shared/types/segmentation";
import type {
  ConfirmedLayerStyle,
  LeftPanelLayerStyles,
} from "@/features/segmentation/overlays/segments";

interface UseOverlayLayerControlsArgs {
  segmentationId: string;
  modelManifest: SegmentationOverlayManifest | null;
  confirmedManifest: SegmentationOverlayManifest | null;
  hiddenSegmentIds?: ReadonlySet<string>;
  hiddenSegmentVisualRevision?: number;
}

const DEFAULT_LEFT_PANEL_LAYER_STYLES: LeftPanelLayerStyles = {
  candidateStrokeWidth: 2,
  candidateFillOpacity: 0.18,
  confirmedStrokeWidth: 2,
  confirmedFillOpacity: 0.2,
};

const DEFAULT_RIGHT_PANEL_CONFIRMED_STYLE: ConfirmedLayerStyle = {
  strokeWidth: 2,
  fillOpacity: 0.2,
};

// Stable module ref (useOverlayLut keys its effect on the joined value).
const CANDIDATE_LAYER_HIDDEN_STATES = ["confirmed", "excluded"];
const CONFIRMED_LAYER_HIDDEN_STATES = ["candidate", "excluded"];
const NO_HIDDEN_SEGMENTS = new Set<string>();

/**
 * Appends/overwrites a `?rev=` cache key on the bundle URL so the browser and
 * the zarr loader refetch the raster whenever its pixels change.
 *
 * The raster changes on BOTH full rebuilds (which bump `bundle_version` and
 * renumber labels) AND in-place partial geometry updates (which edit the active
 * bundle and bump `applied_revision` but NOT `bundle_version`). Keying on
 * `bundle_version` alone -- as this once did -- left partial updates serving the
 * stale, pre-edit raster (newly drawn/confirmed objects invisible until a hard
 * refresh). Keying on `<bundle_version>-<applied_revision>` busts on either.
 * LUT-only state edits (confirm/recolour/show-hide) bump `lut_revision` only --
 * neither token here -- so they still never reload the rasters.
 */
function withRasterRevision(
  url: string,
  bundleVersion: number,
  appliedRevision: number
): string {
  const hashIndex = url.indexOf("#");
  const pathAndSearch = hashIndex >= 0 ? url.slice(0, hashIndex) : url;
  const hash = hashIndex >= 0 ? url.slice(hashIndex) : "";
  const queryIndex = pathAndSearch.indexOf("?");
  const path = queryIndex >= 0 ? pathAndSearch.slice(0, queryIndex) : pathAndSearch;
  const search = queryIndex >= 0 ? pathAndSearch.slice(queryIndex + 1) : "";
  const params = new URLSearchParams(search);
  params.set("rev", `${bundleVersion}-${appliedRevision}`);
  return `${path}?${params.toString()}${hash}`;
}

/** Zeroes only the locally deleted UUIDs without waiting for a new server LUT. */
function hideSegmentsInLut(
  lut: OverlayLutState | null,
  pickMap: Map<number, string> | null,
  hiddenSegmentIds: ReadonlySet<string>
): OverlayLutState | null {
  if (!lut || !pickMap || hiddenSegmentIds.size === 0) return lut;
  let rgba: Uint8Array | null = null;
  for (const [label, segmentId] of pickMap) {
    if (!hiddenSegmentIds.has(segmentId) || label > lut.maxLabel) continue;
    rgba ??= lut.rgba.slice();
    rgba[label * 4 + 3] = 0;
  }
  return rgba ? { ...lut, rgba } : lut;
}

export function useOverlayLayerControls({
  segmentationId,
  modelManifest,
  confirmedManifest,
  hiddenSegmentIds = NO_HIDDEN_SEGMENTS,
  hiddenSegmentVisualRevision = 0,
}: UseOverlayLayerControlsArgs) {
  const [showCandidateBorders, setShowCandidateBorders] = useState(true);
  const [showConfirmedBorders, setShowConfirmedBorders] = useState(true);
  const [showRightConfirmedBorders, setShowRightConfirmedBorders] = useState(true);
  const [leftPanelLayerStyles, setLeftPanelLayerStyles] = useState<LeftPanelLayerStyles>(
    DEFAULT_LEFT_PANEL_LAYER_STYLES
  );
  const [rightPanelConfirmedStyle, setRightPanelConfirmedStyle] =
    useState<ConfirmedLayerStyle>(DEFAULT_RIGHT_PANEL_CONFIRMED_STYLE);

  // One ID-map layer has one scalar fill opacity, so candidates and confirmed
  // objects need separate state-filtered LUTs for their controls to be real.
  const candidateLut = useOverlayLut({
    segmentationId,
    sourceModel: modelManifest?.source_model ?? null,
    lutRevision: modelManifest?.lut_revision ?? null,
    enabled: !!modelManifest?.ngff_url,
    hiddenStates: CANDIDATE_LAYER_HIDDEN_STATES,
  });
  const confirmedLut = useOverlayLut({
    segmentationId,
    sourceModel: null,
    lutRevision: confirmedManifest?.lut_revision ?? null,
    enabled: !!confirmedManifest?.ngff_url,
    hiddenStates: CONFIRMED_LAYER_HIDDEN_STATES,
  });

  const candidatePickMap = useOverlayPickMap({
    segmentationId,
    sourceModel: modelManifest?.source_model ?? null,
    rasterRevision:
      modelManifest != null
        ? `${modelManifest.bundle_version}-${modelManifest.applied_revision}`
        : null,
    enabled: !!modelManifest?.ngff_url,
  });
  const confirmedPickMap = useOverlayPickMap({
    segmentationId,
    sourceModel: null,
    rasterRevision:
      confirmedManifest != null
        ? `${confirmedManifest.bundle_version}-${confirmedManifest.applied_revision}`
        : null,
    enabled: !!confirmedManifest?.ngff_url,
  });
  const visibleCandidateLut = useMemo(
    () => hideSegmentsInLut(candidateLut, candidatePickMap, hiddenSegmentIds),
    [candidateLut, candidatePickMap, hiddenSegmentIds]
  );
  const visibleConfirmedLut = useMemo(
    () => hideSegmentsInLut(confirmedLut, confirmedPickMap, hiddenSegmentIds),
    [confirmedLut, confirmedPickMap, hiddenSegmentIds]
  );

  const candidateNgffUrl = modelManifest?.ngff_url
    ? withRasterRevision(
        modelManifest.ngff_url,
        modelManifest.bundle_version,
        modelManifest.applied_revision
      )
    : null;
  const confirmedNgffUrl = confirmedManifest?.ngff_url
    ? withRasterRevision(
        confirmedManifest.ngff_url,
        confirmedManifest.bundle_version,
        confirmedManifest.applied_revision
      )
    : null;

  const leftIdMapOverlays = useMemo<ViewerIdMapOverlaySpec[]>(() => {
    const overlays: ViewerIdMapOverlaySpec[] = [];
    if (candidateNgffUrl && visibleCandidateLut) {
      overlays.push({
        id: "label-left-candidates-idmap",
        ngffUrl: candidateNgffUrl,
        revision: modelManifest?.applied_revision,
        lut: visibleCandidateLut.rgba,
        maxLabel: visibleCandidateLut.maxLabel,
        lutRevision: visibleCandidateLut.lutRevision,
        visualRevision: hiddenSegmentVisualRevision,
        fillOpacity: leftPanelLayerStyles.candidateFillOpacity,
        borderOpacity: RASTER_BORDER_OPACITY,
        showBorders: showCandidateBorders,
        pickMap: candidatePickMap ?? undefined,
      });
    }
    if (confirmedNgffUrl && visibleConfirmedLut) {
      overlays.push({
        id: "label-left-confirmed-idmap",
        ngffUrl: confirmedNgffUrl,
        revision: confirmedManifest?.applied_revision,
        lut: visibleConfirmedLut.rgba,
        maxLabel: visibleConfirmedLut.maxLabel,
        lutRevision: visibleConfirmedLut.lutRevision,
        visualRevision: hiddenSegmentVisualRevision,
        fillOpacity: leftPanelLayerStyles.confirmedFillOpacity,
        borderOpacity: RASTER_BORDER_OPACITY,
        showBorders: showConfirmedBorders,
        pickMap: confirmedPickMap ?? undefined,
      });
    }
    return overlays;
  }, [
    candidateNgffUrl,
    confirmedNgffUrl,
    modelManifest?.applied_revision,
    confirmedManifest?.applied_revision,
    visibleCandidateLut,
    visibleConfirmedLut,
    hiddenSegmentVisualRevision,
    candidatePickMap,
    confirmedPickMap,
    leftPanelLayerStyles.candidateFillOpacity,
    leftPanelLayerStyles.confirmedFillOpacity,
    showCandidateBorders,
    showConfirmedBorders,
  ]);

  const rightIdMapOverlays = useMemo<ViewerIdMapOverlaySpec[]>(() => {
    if (!confirmedNgffUrl || !visibleConfirmedLut) return [];
    return [
      {
        id: "label-right-confirmed-idmap",
        ngffUrl: confirmedNgffUrl,
        revision: confirmedManifest?.applied_revision,
        lut: visibleConfirmedLut.rgba,
        maxLabel: visibleConfirmedLut.maxLabel,
        lutRevision: visibleConfirmedLut.lutRevision,
        visualRevision: hiddenSegmentVisualRevision,
        fillOpacity: rightPanelConfirmedStyle.fillOpacity,
        borderOpacity: RASTER_BORDER_OPACITY,
        showBorders: showRightConfirmedBorders,
        pickMap: confirmedPickMap ?? undefined,
      },
    ];
  }, [
    confirmedNgffUrl,
    confirmedManifest?.applied_revision,
    visibleConfirmedLut,
    hiddenSegmentVisualRevision,
    confirmedPickMap,
    rightPanelConfirmedStyle.fillOpacity,
    showRightConfirmedBorders,
  ]);

  return {
    leftPanelLayerStyles,
    showCandidateBorders,
    setShowCandidateBorders,
    showConfirmedBorders,
    setShowConfirmedBorders,
    rightPanelConfirmedStyle,
    showRightConfirmedBorders,
    setShowRightConfirmedBorders,
    leftIdMapOverlays,
    rightIdMapOverlays,
    updateLayerStyles: {
      setCandidateStrokeWidth: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          candidateStrokeWidth: clamp(value, 0.5, 8),
        }));
      },
      setCandidateFillOpacity: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          candidateFillOpacity: clamp(value, 0, 1),
        }));
      },
      setConfirmedStrokeWidth: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          confirmedStrokeWidth: clamp(value, 0.5, 8),
        }));
      },
      setConfirmedFillOpacity: (value: number) => {
        if (Number.isNaN(value)) return;
        setLeftPanelLayerStyles((prev) => ({
          ...prev,
          confirmedFillOpacity: clamp(value, 0, 1),
        }));
      },
    },
    updateRightLayerStyle: {
      setStrokeWidth: (value: number) => {
        if (Number.isNaN(value)) return;
        setRightPanelConfirmedStyle((prev) => ({
          ...prev,
          strokeWidth: clamp(value, 0.5, 8),
        }));
      },
      setFillOpacity: (value: number) => {
        if (Number.isNaN(value)) return;
        setRightPanelConfirmedStyle((prev) => ({
          ...prev,
          fillOpacity: clamp(value, 0, 1),
        }));
      },
    },
  };
}
