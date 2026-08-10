import { useErPreviewStore } from "@/features/segmentation/erPreview/useErPreviewStore";

interface ErPreviewControlsProps {
  assetId: string | null | undefined;
  segmentationId: string | null | undefined;
  sourceModel: string | null | undefined;
  roi: { id: string; x: number; y: number; width: number; height: number } | null | undefined;
  /** Called after candidates are pinned, so the parent can refresh segment views. */
  onPinned?: () => void;
}

/**
 * "Run model on ROI" control for the ER labeling view. Runs the selected Source
 * Model on the active ROI and shows the result as a replaceable overlay with a
 * live threshold slider (re-thresholded client-side, no re-run). "Pin
 * candidates" then persists the thresholded result as CANDIDATE segments.
 */
export function ErPreviewControls({
  assetId,
  segmentationId,
  sourceModel,
  roi,
  onPinned,
}: ErPreviewControlsProps) {
  const {
    run,
    pin,
    running,
    pinning,
    error,
    pinError,
    stats,
    overlay,
    threshold,
    opacity,
    setThreshold,
    setOpacity,
    clear,
  } = useErPreviewStore();
  const canRun = Boolean(assetId && sourceModel && roi);

  const onRun = () => {
    if (assetId && sourceModel && roi) {
      void run({ assetId, sourceModel, roi });
    }
  };

  const onPin = async () => {
    if (!segmentationId) return;
    const count = await pin({ segmentationId });
    if (count !== null) {
      onPinned?.();
    }
  };

  return (
    <div
      className="er-preview-controls"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 14,
        padding: "6px 16px",
        borderBottom: "1px solid var(--border-color, #2a2a2a)",
        flexWrap: "wrap",
        fontSize: 13,
      }}
    >
      <button onClick={onRun} disabled={!canRun || running} title={!roi ? "Select an ROI first" : undefined}>
        {running ? "Running model…" : "Run model on ROI"}
      </button>
      {overlay && (
        <>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            Threshold
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              style={{ width: 120 }}
            />
            <span style={{ fontVariantNumeric: "tabular-nums", width: 30 }}>{threshold.toFixed(2)}</span>
          </label>
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            Opacity
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              style={{ width: 90 }}
            />
          </label>
          <button
            onClick={onPin}
            disabled={pinning || !segmentationId}
            title={!segmentationId ? "No active segmentation" : "Persist thresholded result as candidates"}
          >
            {pinning ? "Pinning…" : "Pin candidates"}
          </button>
          <button onClick={clear} disabled={pinning}>
            Clear
          </button>
          <span style={{ opacity: 0.7 }}>model: {overlay.sourceModel}</span>
        </>
      )}
      {pinError && <span style={{ color: "var(--error-color, #ff6b6b)" }}>{pinError}</span>}
      {stats && (
        <span style={{ opacity: 0.7 }}>
          {stats.elapsed_s}s · {(stats.frac * 100).toFixed(1)}% @ default
        </span>
      )}
      {!roi && <span style={{ opacity: 0.6 }}>Select an ROI to enable.</span>}
      {error && <span style={{ color: "var(--error-color, #ff6b6b)" }}>{error}</span>}
    </div>
  );
}
