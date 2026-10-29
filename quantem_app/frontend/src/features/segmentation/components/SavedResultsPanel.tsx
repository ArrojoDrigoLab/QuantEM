import { useEffect, useMemo, useRef, useState } from "react";
import { apiRequest, resolveApiUrl } from "@/shared/api/core/http";
import { getAssetNgffUrl } from "@/shared/api/assets";
import type { AssetDetail } from "@/shared/types";
import { Button } from "@/shared/ui/design";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { createViewportActionResolver } from "@/utils/viewportActions";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { useViewportSyncGroup } from "@/viewer/viewportSync/useViewportSyncGroup";
import "./SavedResultsPanel.css";

export type SavedResult = {
  id: string;
  name: string;
  source_model: string;
  threshold: number;
  is_canonical: boolean;
  available: boolean;
  created_at: string;
  preview_url: string;
};

interface SavedResultsPanelProps {
  segmentationId: string;
  image: AssetDetail;
  revision: string;
  open: boolean;
  onClose: () => void;
  onResultsChange: (results: SavedResult[]) => void;
  onQueued: () => void;
}

/** A two-viewer, full-resolution comparison for retained model runs. */
export function SavedResultsPanel({
  segmentationId,
  image,
  revision,
  open,
  onClose,
  onResultsChange,
  onQueued,
}: SavedResultsPanelProps) {
  const [results, setResults] = useState<SavedResult[]>([]);
  const [left, setLeft] = useState("");
  const [right, setRight] = useState("");
  const [opacity, setOpacity] = useState(0.3);
  const [confirm, setConfirm] = useState<SavedResult | null>(null);
  const [deletions, setDeletions] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [actionError, setActionError] = useState("");
  const [queued, setQueued] = useState("");
  const canonicalIdRef = useRef("");
  const viewportResolver = useMemo(
    () => createViewportActionResolver(image.width, image.height),
    [image.height, image.width]
  );
  const { viewport, publishFromViewer } = useViewportSyncGroup(
    `saved-results:${segmentationId}`,
    { resolveAction: viewportResolver }
  );

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    const load = async () => {
      try {
        const response = await apiRequest<{ results: SavedResult[] }>(
          `/api/segmentations/${segmentationId}/results/`
        );
        if (stopped) return;
        setResults(response.results);
        onResultsChange(response.results);
        setError("");
        if (queued) {
          const job = await apiRequest<{ status: string; message: string }>(
            `/api/jobs/${queued}/`
          );
          if (stopped) return;
          if (!["PENDING", "RETRY", "RUNNING"].includes(job.status)) {
            setQueued("");
            if (job.status !== "SUCCESS") {
              setActionError(
                job.message ||
                  "The result could not be applied. Retry from Tasks & Queues."
              );
            }
            onQueued();
          }
        }
      } catch (cause) {
        if (!stopped) {
          setError(
            extractApiErrorMessage(cause, "Saved results could not be loaded.")
          );
        }
      }
      if (!stopped && queued) timer = window.setTimeout(() => void load(), 4000);
    };
    void load();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [segmentationId, revision, queued, onQueued, onResultsChange]);

  const availableResults = useMemo(
    () => results.filter((result) => result.available),
    [results]
  );

  // Reset to the useful pairing when the workspace opens, then preserve the
  // person's choices across background refreshes. Replacing the result array
  // every poll used to snap both dropdowns back while they were being used.
  useEffect(() => {
    if (!open) {
      setLeft("");
      setRight("");
      canonicalIdRef.current = "";
      return;
    }
    if (availableResults.length < 2) return;
    const canonical = availableResults.find((row) => row.is_canonical);
    if (!canonical) return;
    const alternative =
      availableResults.find(
        (row) => !row.is_canonical && row.id !== canonical.id
      ) ?? availableResults.find((row) => row.id !== canonical.id);
    const availableIds = new Set(availableResults.map((row) => row.id));
    const canonicalChanged =
      canonicalIdRef.current !== "" && canonicalIdRef.current !== canonical.id;
    const nextLeft =
      !canonicalChanged && left && availableIds.has(left)
        ? left
        : canonical.id;
    const nextRight =
      !canonicalChanged && right && availableIds.has(right) && right !== nextLeft
        ? right
        : alternative?.id === nextLeft
          ? availableResults.find((row) => row.id !== nextLeft)?.id ?? ""
          : alternative?.id ?? "";
    canonicalIdRef.current = canonical.id;
    if (left !== nextLeft) setLeft(nextLeft);
    if (right !== nextRight) setRight(nextRight);
  }, [availableResults, left, open, right]);

  const applyResult = async () => {
    if (!confirm) return;
    setBusy(true);
    setError("");
    setActionError("");
    try {
      const response = await apiRequest<{ job_id: string }>(
        `/api/segmentations/${segmentationId}/results/`,
        {
          method: "POST",
          body: JSON.stringify({
            result_id: confirm.id,
            delete_result_ids: deletions,
          }),
        }
      );
      setConfirm(null);
      setQueued(response.job_id);
      onQueued();
    } catch (cause) {
      setActionError(
        extractApiErrorMessage(cause, "This result could not be selected.")
      );
    } finally {
      setBusy(false);
    }
  };

  const pane = (
    id: string,
    setId: (value: string) => void,
    otherId: string,
    label: string,
    viewerId: string
  ) => {
    const result = results.find((row) => row.id === id);
    return (
      <section className="saved-results-pane" role="group" aria-label={`${label} view`}>
        <div className="saved-results-pane-toolbar">
          <label>
            <span>{label}</span>
            <select value={id} onChange={(event) => setId(event.target.value)}>
              {availableResults.map((row) => (
                <option
                  key={row.id}
                  value={row.id}
                  disabled={row.id === otherId}
                >
                  {row.is_canonical ? "✓ " : ""}
                  {row.name}
                  {row.is_canonical ? " · Confirmed" : ""} ·{" "}
                  {new Date(row.created_at).toLocaleString()}
                </option>
              ))}
            </select>
          </label>
          {result && !result.is_canonical ? (
            <Button
              className="shrink-0"
              disabled={busy || Boolean(queued) || !result.available}
              onClick={() => {
                setConfirm(result);
                setDeletions(
                  results.filter((row) => row.id !== result.id).map((row) => row.id)
                );
              }}
            >
              Confirm this model
            </Button>
          ) : null}
        </div>
        <div className="saved-results-viewer">
          <ImageViewer
            image={{
              ngffUrl: getAssetNgffUrl(image.id, null),
              width: image.width,
              height: image.height,
              storedDepth: image.stored_depth,
              zPlaneIndices: image.z_plane_indices,
              pixelSizeNm: image.pixel_size_nm,
            }}
            className="viewer-container"
            viewport={{
              state: viewport ?? undefined,
              onChange: (next) => publishFromViewer(viewerId, next),
            }}
            overlays={
              result
                ? {
                    bitmapOverlays: [
                      {
                        id: `saved-result-${result.id}`,
                        image: resolveApiUrl(result.preview_url),
                        bounds: [0, 0, image.width, image.height],
                        opacity,
                      },
                    ],
                  }
                : undefined
            }
            interactions={{ mode: "navigate" }}
          />
        </div>
      </section>
    );
  };

  if (!open) return null;

  return (
    <main className="saved-results-comparison" aria-label="Compare model runs">
      <div className="saved-results-toolbar">
        <div>
          <h3>Compare runs</h3>
          <p>The confirmed run is shown first. Choose a different retained run on either side.</p>
          <p>Overlays are reduced-resolution thresholded predictions. Confirming a run applies object filtering and preserves your completed annotations.</p>
        </div>
        <label className="saved-results-opacity">
          Opacity
          <input
            aria-label="Model result opacity"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
          <span>{Math.round(opacity * 100)}%</span>
        </label>
        <Button onClick={onClose}>Back to labeling</Button>
      </div>
      {actionError || error ? (
        <p role="alert" className="saved-results-error">
          {actionError || error}
        </p>
      ) : null}
      {queued ? (
        <p role="status" className="saved-results-status">
          Applying the selected result…
        </p>
      ) : null}
      <div className="saved-results-panes">
        {pane(left, setLeft, right, "Confirmed model", "saved-left")}
        {pane(right, setRight, left, "Comparison model", "saved-right")}
      </div>
      <ConfirmDialog
        isOpen={Boolean(confirm)}
        title="Confirm this model"
        message="Use this model's outputs as the default for analysis? Your manually annotated areas will still override model outputs in ROIs marked as 'Done'."
        details={
          <fieldset disabled={busy}>
            <legend>Delete old model outputs</legend>
            {results
              .filter((row) => row.id !== confirm?.id)
              .map((row) => (
                <label className="block" key={row.id}>
                  <input
                    type="checkbox"
                    checked={deletions.includes(row.id)}
                    onChange={(event) =>
                      setDeletions((ids) =>
                        event.target.checked
                          ? [...ids, row.id]
                          : ids.filter((id) => id !== row.id)
                      )
                    }
                  />{" "}
                  {row.name}
                </label>
              ))}
            <p>
              Clear a checkbox to keep that version for comparison. Checked outputs are
              deleted only after the selected model's outputs are successfully applied.
            </p>
          </fieldset>
        }
        confirmText={busy ? "Queuing…" : "Confirm"}
        confirmDisabled={busy}
        cancelText="Cancel"
        onCancel={() => {
          if (!busy) setConfirm(null);
        }}
        onConfirm={() => {
          if (!busy) void applyResult();
        }}
      />
    </main>
  );
}
