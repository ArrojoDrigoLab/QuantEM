import { useEffect, useId, useRef, useState } from "react";
import {
  getAssetRasterExportUrl,
  getAssetSegmentations,
  type AssetRasterExportSelection,
} from "@/shared/api/assets";
import { segmentationDisplayName } from "@/shared/segmentationNames";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { saveUrlFile } from "@/utils/downloadText";

const ANALYSIS_MASK_INTERNAL_NAME = "quantem_internal_analysis_mask";

export function ExportDialog({
  asset,
  onClose,
}: {
  asset: { id: string; displayName: string };
  onClose: () => void;
}) {
  const titleId = useId();
  const firstInputRef = useRef<HTMLInputElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const [selection, setSelection] = useState<AssetRasterExportSelection>({
    source: "original",
  });
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const { data: segmentations, loading, error } = useApiQuery(
    () => getAssetSegmentations(asset.id),
    [asset.id]
  );

  useEffect(() => {
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    firstInputRef.current?.focus();
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      restoreFocusRef.current?.focus();
    };
  }, [onClose]);

  const hasAnalysisMasks = Boolean(
    segmentations?.some(
      (segmentation) =>
        segmentation.segmentation_type.internal_name === ANALYSIS_MASK_INTERNAL_NAME
    )
  );

  const exportFilename = () => {
    const filenamePart = (value: string, fallback: string) =>
      value
        .trim()
        .replace(/[^A-Za-z0-9._-]+/g, "_")
        .replace(/^[._]+|[._]+$/g, "")
        .slice(0, 120) || fallback;
    const imageName = filenamePart(asset.displayName, "image");
    if (selection.source === "original") return `${imageName}_EM_8bit.png`;
    const segmentation = segmentations?.find(
      (candidate) => candidate.id === selection.segmentationId
    );
    const segmentationName = filenamePart(
      segmentation?.display_name ||
        segmentation?.segmentation_type.long_name ||
        "segmentation",
      "segmentation"
    );
    return `${imageName}_${segmentationName}_8bit.png`;
  };

  const startDownload = async () => {
    setSaving(true);
    setSaveError(null);
    try {
      const result = await saveUrlFile(
        exportFilename(),
        getAssetRasterExportUrl(asset.id, selection),
        "image/png"
      );
      if (result !== "cancelled") onClose();
    } catch (cause) {
      setSaveError(
        cause instanceof Error
          ? cause.message
          : "The selected image could not be saved."
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4"
      onClick={onClose}
    >
      <section
        className="w-full max-w-lg rounded-2xl bg-white p-6 text-slate-900 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id={titleId} className="m-0 text-xl font-semibold text-slate-950">
          Export {asset.displayName}
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          Choose one source. The download is an 8-bit grayscale PNG.
        </p>

        <fieldset className="mt-5 max-h-72 overflow-auto rounded-lg border border-slate-200 p-3">
          <legend className="px-1 text-sm font-semibold text-slate-800">Source</legend>
          <label className="flex cursor-pointer items-start gap-3 rounded-md px-2 py-2 hover:bg-slate-50">
            <input
              ref={firstInputRef}
              type="radio"
              name="raster-export-source"
              className="mt-0.5 h-4 w-4"
              checked={selection.source === "original"}
              onChange={() => setSelection({ source: "original" })}
            />
            <span>
              <span className="block text-sm font-medium text-slate-900">
                Original EM image
              </span>
              <span className="block text-xs text-slate-500">
                Canonical 8-bit grayscale pixels
              </span>
            </span>
          </label>

          {loading ? <p className="px-2 text-sm text-slate-500">Loading segmentations…</p> : null}
          {segmentations?.map((segmentation) => {
            const isAnalysisMask =
              segmentation.segmentation_type.internal_name ===
              ANALYSIS_MASK_INTERNAL_NAME;
            const checked =
              selection.source === "segmentation" &&
              selection.segmentationId === segmentation.id;
            return (
              <label
                key={segmentation.id}
                className="flex cursor-pointer items-start gap-3 rounded-md px-2 py-2 hover:bg-slate-50"
              >
                <input
                  type="radio"
                  name="raster-export-source"
                  className="mt-0.5 h-4 w-4"
                  checked={checked}
                  onChange={() =>
                    setSelection({
                      source: "segmentation",
                      segmentationId: segmentation.id,
                    })
                  }
                />
                <span>
                  <span className="block text-sm font-medium text-slate-900">
                    {segmentationDisplayName(segmentation)}
                  </span>
                  <span className="block text-xs text-slate-500">
                    {isAnalysisMask
                      ? "Analysis Mask objects as discrete labels"
                      : segmentation.segmentation_type.measurement_mode === "objects"
                        ? "Objects as discrete labels"
                        : "Binary mask"}
                  </span>
                </span>
              </label>
            );
          })}
        </fieldset>

        {error ? (
          <p className="mt-3 text-sm text-red-700" role="alert">
            {extractApiErrorMessage(error, "The segmentations could not be loaded.")}
          </p>
        ) : null}

        {saveError ? (
          <p className="mt-3 text-sm text-red-700" role="alert">
            {saveError}
          </p>
        ) : null}

        <div className="mt-4 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600">
          Background is 0. Object labels count from 1 through 255, then restart at 1.
          {hasAnalysisMasks ? (
            <span className="mt-1 block">
              Where Analysis Mask objects overlap, the object later in the list replaces
              the earlier object in the exported pixels.
            </span>
          ) : null}
        </div>

        <div className="mt-6 flex justify-end gap-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="primary" disabled={saving} onClick={() => void startDownload()}>
            {saving ? "Saving…" : "Export"}
          </Button>
        </div>
      </section>
    </div>
  );
}
