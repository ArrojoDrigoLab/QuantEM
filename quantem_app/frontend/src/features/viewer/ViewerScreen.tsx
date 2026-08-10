import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useSegmentationOverlayManifests } from "@/hooks/useSegmentationOverlayManifest";
import { useViewportSync } from "@/hooks/useViewportSync";
import {
  getAsset,
  getAssetNgffUrl,
  getAssetSegmentations,
} from "@/shared/api/assets";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { OverlaySelectionSidebar } from "@/features/viewer/components/OverlaySelectionSidebar";
import { SegmentationCreatePanel } from "@/features/viewer/components/SegmentationCreatePanel";
import { getSegmentationOverlayLutJson } from "@/shared/api/segmentations/overlays";
import {
  deleteSegmentation,
  getSegmentationDetail,
} from "@/shared/api/segmentations/lifecycle";
import { pluraliseObjects } from "@/features/segmentation/components/segmentationCompletionLoss";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { PixelSizeEditor } from "@/shared/ui/PixelSize";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { buildTintedLut } from "@/viewer/overlays/labelLut";
import type { PreprocessStage } from "@/shared/types/common";
import type { ImageSegmentation, StatusStage } from "@/shared/types/images";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type {
  OverlayLutJson,
  SegmentationDeletePreview,
} from "@/shared/types/segmentation";
import "./ViewerScreen.css";

const IMAGE_READY_STAGES: PreprocessStage[] = ["DONE", "SKIPPED"];
const IMAGE_TERMINAL_STAGES: PreprocessStage[] = ["DONE", "SKIPPED", "FAILED", "CANCELLED"];

const PROCESSING_STATUS_STAGES: StatusStage[] = [
  "UNSTARTED",
  "RUNNING_INFERENCE",
  "EXTRACTING_CANDIDATES",
];

function getStageLabel(stage: PreprocessStage): string {
  switch (stage) {
    case "ENCODING": return "Encoding image";
    case "SAM": return "Running segmentation model";
    case "FEATURES": return "Extracting features";
    case "FAILED": return "Preprocessing failed";
    case "CANCELLED": return "Preprocessing cancelled";
    case "NONE": return "Queued for preprocessing";
    default: return "Processing";
  }
}

const DEFAULT_COLORS = [
  "#38bdf8",
  "#22c55e",
  "#f97316",
  "#a855f7",
  "#f43f5e",
  "#eab308",
  "#14b8a6",
  "#6366f1",
];

function getFallbackColor(index: number) {
  return DEFAULT_COLORS[index % DEFAULT_COLORS.length];
}

interface ViewerOverlayConfig {
  segmentation: ImageSegmentation;
  enabled: boolean;
  color: string;
  opacity: number;
}

export function ViewerScreen() {
  const { selectedAssetId, setSelectedSegmentationId, clearSelection } =
    useSelectionStore();
  const navigate = useNavigate();

  const handleBackToLibrary = useCallback(() => {
    clearSelection();
    navigate("/");
  }, [clearSelection, navigate]);
  const { viewport, setViewport } = useViewportSync();
  const [overlayConfig, setOverlayConfig] = useState<
    Record<string, ViewerOverlayConfig>
  >({});
  // Per-segmentation label -> object LUT JSON, used to build a client tinted LUT
  // that shows confirmed objects in the user's chosen overlay colour.
  const [overlayLutJsonById, setOverlayLutJsonById] = useState<
    Record<string, OverlayLutJson>
  >({});

  const { data: image, refetch: refetchImage } = useApiQuery(
    () => {
      if (!selectedAssetId) throw new Error("No asset selected");
      return getAsset(selectedAssetId);
    },
    [selectedAssetId]
  );

  const imageReady = image
    ? IMAGE_READY_STAGES.includes(image.preprocess_stage)
    : false;

  const imageTerminal = image
    ? IMAGE_TERMINAL_STAGES.includes(image.preprocess_stage)
    : false;

  // Poll image status while preprocessing is still running
  useEffect(() => {
    if (!image || imageTerminal) return undefined;
    const interval = setInterval(() => {
      void refetchImage();
    }, 2000);
    return () => clearInterval(interval);
  }, [image, imageTerminal, refetchImage]);

  const { data: segmentations, refetch: refetchSegmentations } = useApiQuery<
    ImageSegmentation[]
  >(
    () => {
      if (!selectedAssetId) throw new Error("No asset selected");
      return getAssetSegmentations(selectedAssetId);
    },
    [selectedAssetId]
  );

  const handleSegmentationCreated = async (segmentation: ImageSegmentation) => {
    const assetId = segmentation.asset ?? selectedAssetId;
    if (!assetId) return;
    setSelectedSegmentationId(segmentation.id);
    await refetchSegmentations();
    const encodedName = encodeURIComponent(
      segmentation.segmentation_type.long_name
    );
    navigate(`/assets/${assetId}/labeling/${encodedName}`);
  };

  const visibleSegmentations = useMemo(() => {
    if (!segmentations) return [];
    return segmentations;
  }, [segmentations]);

  // Poll segmentation statuses while any are still processing
  const hasProcessingSegmentations = useMemo(() => {
    return visibleSegmentations.some((seg) =>
      PROCESSING_STATUS_STAGES.includes(seg.status_stage)
    );
  }, [visibleSegmentations]);

  useEffect(() => {
    if (!hasProcessingSegmentations) return;
    const interval = setInterval(() => void refetchSegmentations(), 3000);
    return () => clearInterval(interval);
  }, [hasProcessingSegmentations, refetchSegmentations]);

  useEffect(() => {
    if (!visibleSegmentations.length) {
      setOverlayConfig({});
      return;
    }
    setOverlayConfig((prev) => {
      const next: Record<string, ViewerOverlayConfig> = { ...prev };
      const seen = new Set<string>();

      visibleSegmentations.forEach((seg, index) => {
        seen.add(seg.id);
        const isCompleted = seg.status_stage === "COMPLETED";
        if (!next[seg.id]) {
          next[seg.id] = {
            segmentation: seg,
            enabled: isCompleted,
            color: seg.segmentation_type.default_color || getFallbackColor(index),
            opacity: 0.25,
          };
          return;
        }
        next[seg.id] = {
          ...next[seg.id],
          segmentation: seg,
          enabled: isCompleted ? next[seg.id].enabled : false,
        };
      });

      Object.keys(next).forEach((key) => {
        if (!seen.has(key)) {
          delete next[key];
        }
      });

      return next;
    });
  }, [visibleSegmentations]);

  const overlayList = useMemo(
    () => Object.values(overlayConfig),
    [overlayConfig]
  );
  const enabledOverlayList = useMemo(
    () => overlayList.filter((overlay) => overlay.enabled),
    [overlayList]
  );

  const { manifests: overlayManifests, loading: overlayManifestLoading, refetching: overlayManifestRefetching } =
    useSegmentationOverlayManifests({
      segmentationIds: enabledOverlayList.map((overlay) => overlay.segmentation.id),
      enabled: imageReady,
    });

  // Load (and refresh on lut_revision changes) the label -> object LUT JSON for
  // each enabled segmentation overlay so we can build a client tinted LUT.
  const enabledOverlayLutKey = useMemo(
    () =>
      enabledOverlayList
        .map((overlay) => {
          const manifest = overlayManifests[overlay.segmentation.id];
          return `${overlay.segmentation.id}:${manifest?.lut_revision ?? "?"}`;
        })
        .join("|"),
    [enabledOverlayList, overlayManifests]
  );

  useEffect(() => {
    const targets = enabledOverlayList.filter(
      (overlay) => overlayManifests[overlay.segmentation.id]?.ngff_url
    );
    if (!imageReady || targets.length === 0) {
      setOverlayLutJsonById({});
      return undefined;
    }
    let cancelled = false;
    void Promise.all(
      targets.map(async (overlay) => {
        const manifest = overlayManifests[overlay.segmentation.id];
        const json = await getSegmentationOverlayLutJson(
          overlay.segmentation.id,
          manifest?.source_model ?? null
        );
        return [overlay.segmentation.id, json] as const;
      })
    )
      .then((entries) => {
        if (cancelled) return;
        setOverlayLutJsonById(Object.fromEntries(entries));
      })
      .catch((error) => {
        console.error("Failed to load segmentation overlay LUT JSON", error);
        if (!cancelled) setOverlayLutJsonById({});
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabledOverlayLutKey, imageReady]);

  const CONFIRMED_OVERLAY_STATES = useMemo(
    () => new Set(["confirmed", "refined", "labeled"]),
    []
  );

  const viewerIdMapOverlays = useMemo<ViewerIdMapOverlaySpec[]>(() => {
    return enabledOverlayList.flatMap((overlay) => {
      const manifest = overlayManifests[overlay.segmentation.id];
      const json = overlayLutJsonById[overlay.segmentation.id];
      if (!manifest?.ngff_url || !json) return [];
      const { rgba, maxLabel } = buildTintedLut(
        json.objects,
        overlay.color,
        CONFIRMED_OVERLAY_STATES
      );
      return [
        {
          id: overlay.segmentation.id,
          // Bust the raster cache on any pixel change: full rebuilds bump
          // bundle_version, in-place partial geometry edits bump applied_revision
          // (state-only recolours bump neither, so they keep the cached raster).
          ngffUrl: `${manifest.ngff_url}?rev=${manifest.bundle_version}-${manifest.applied_revision}`,
          lut: rgba,
          maxLabel,
          lutRevision: manifest.lut_revision,
          fillOpacity: overlay.opacity,
          borderOpacity: 0.95,
          showBorders: true,
        },
      ];
    });
  }, [
    enabledOverlayList,
    overlayManifests,
    overlayLutJsonById,
    CONFIRMED_OVERLAY_STATES,
  ]);

  const overlayUpdating = useMemo(() => {
    return enabledOverlayList.some((overlay) => {
      const manifest = overlayManifests[overlay.segmentation.id];
      return (
        manifest &&
        (manifest.status === "BUILDING" ||
          manifest.status === "DIRTY" ||
          manifest.desired_revision > manifest.applied_revision)
      );
    });
  }, [enabledOverlayList, overlayManifests]);

  const handleToggle = (segmentationId: string) => {
    setOverlayConfig((prev) => ({
      ...prev,
      [segmentationId]: {
        ...prev[segmentationId],
        enabled: !prev[segmentationId]?.enabled,
      },
    }));
  };

  const handleColorChange = (segmentationId: string, color: string) => {
    setOverlayConfig((prev) => ({
      ...prev,
      [segmentationId]: {
        ...prev[segmentationId],
        color,
      },
    }));
  };

  const handleOpacityChange = (segmentationId: string, opacity: number) => {
    setOverlayConfig((prev) => ({
      ...prev,
      [segmentationId]: {
        ...prev[segmentationId],
        opacity,
      },
    }));
  };

  const handleEditSegmentation = (segmentationId: string) => {
    if (!selectedAssetId) return;
    const target = visibleSegmentations.find((seg) => seg.id === segmentationId);
    const encodedName = target
      ? encodeURIComponent(target.segmentation_type.long_name)
      : "new";
    setSelectedSegmentationId(segmentationId);
    navigate(`/assets/${selectedAssetId}/labeling/${encodedName}`);
  };

  /**
   * Deleting a segmentation, to the Mark-Done standard.
   *
   * The dialog quotes counts read fresh from
   * `GET /api/segmentations/<id>/` when it opens — not `segment_counts` off
   * the list payload, which can be a poll behind — and the DELETE carries the
   * object count the user was shown. The server refuses a stale count, an
   * active job and the completion lock with a 409, and each refusal is
   * rendered in the dialog rather than closing it into silence.
   */
  const [deleteTarget, setDeleteTarget] = useState<ImageSegmentation | null>(null);
  const [deletePreview, setDeletePreview] =
    useState<SegmentationDeletePreview | null>(null);
  const [deletePreviewError, setDeletePreviewError] = useState<string | null>(null);
  const [deleteSubmitError, setDeleteSubmitError] = useState<string | null>(null);
  const [deletePreviewNonce, setDeletePreviewNonce] = useState(0);
  const [deleting, setDeleting] = useState(false);

  const handleRequestDeleteSegmentation = useCallback(
    (segmentationId: string) => {
      const target =
        visibleSegmentations.find((seg) => seg.id === segmentationId) ?? null;
      if (!target) return;
      setDeletePreview(null);
      setDeletePreviewError(null);
      setDeleteSubmitError(null);
      setDeleteTarget(target);
      setDeletePreviewNonce((current) => current + 1);
    },
    [visibleSegmentations]
  );

  const deleteTargetId = deleteTarget?.id ?? null;
  useEffect(() => {
    if (!deleteTargetId) return undefined;
    let cancelled = false;
    void getSegmentationDetail(deleteTargetId)
      .then((detail) => {
        if (!cancelled) setDeletePreview(detail.delete_preview);
      })
      .catch((error) => {
        if (cancelled) return;
        setDeletePreviewError(
          extractApiErrorMessage(
            error,
            "What this would delete could not be counted."
          )
        );
      });
    return () => {
      cancelled = true;
    };
  }, [deleteTargetId, deletePreviewNonce]);

  const confirmDeleteSegmentation = useCallback(async () => {
    if (!deleteTargetId) return;
    setDeleting(true);
    setDeleteSubmitError(null);
    try {
      await deleteSegmentation(deleteTargetId, deletePreview?.object_count);
      setDeleteTarget(null);
      setDeletePreview(null);
      await refetchSegmentations();
    } catch (error) {
      setDeleteSubmitError(
        extractApiErrorMessage(
          error,
          "The segmentation could not be deleted; nothing was changed."
        )
      );
      // Re-read, so the numbers beside the refusal are the ones that would
      // now go — the usual cause of the 409 is a run that just finished.
      setDeletePreviewNonce((current) => current + 1);
    } finally {
      setDeleting(false);
    }
  }, [deleteTargetId, deletePreview?.object_count, refetchSegmentations]);

  const existingSegmentationTypes = useMemo(
    () => visibleSegmentations.map((seg) => seg.segmentation_type.long_name),
    [visibleSegmentations]
  );

  if (!image) {
    return <div className="viewer-loading">Loading viewer...</div>;
  }

  return (
    <div className="viewer-screen">
      <div className="viewer-main">
        <div className="viewer-header">
          <div className="viewer-header-identity">
            {/* In the header, not floating over it. The app used to render a
                `position: fixed` back button at (16, 16) as a sibling of this
                screen, which landed exactly on top of the <h2> and the filename
                -- you could not read which image you were looking at. */}
            <button
              type="button"
              className="viewer-back-button"
              onClick={handleBackToLibrary}
            >
              ← Back to Library
            </button>
            <h2>{image.display_name}</h2>
            <span className="viewer-filename">{image.original_filename}</span>
            {/* The viewer is the first screen where the image is a single
                concrete thing, so it is where calibration belongs. Editing it
                PATCHes the asset and refetches, which is the only route by
                which an untagged EM export ever becomes measurable. */}
            <PixelSizeEditor
              className="viewer-pixel-size"
              asset={image}
              onSaved={() => {
                void refetchImage();
              }}
            />
          </div>
          <div className="viewer-header-actions">
            {(overlayManifestLoading || overlayManifestRefetching) && (
              <span className="viewer-loading-indicator">Loading overlays…</span>
            )}
            {overlayUpdating && (
              <span className="viewer-loading-indicator">Overlay updating…</span>
            )}
            {selectedAssetId ? (
              <>
                <Link
                  className="viewer-header-link"
                  to={`/assets/${selectedAssetId}/analysis`}
                >
                  Analysis
                </Link>
                <Link
                  className="viewer-header-link"
                  to={`/assets/${selectedAssetId}/adapt`}
                >
                  Adapt a model
                </Link>
              </>
            ) : null}
          </div>
        </div>
        <div className="viewer-canvas">
          <ImageViewer
            image={{
              ngffUrl: getAssetNgffUrl(image.id, null),
              width: image.width,
              height: image.height,
            }}
            className="viewer-container"
            viewport={{
              state: viewport ?? undefined,
              onChange: setViewport,
            }}
            overlays={{
              idMapOverlays: viewerIdMapOverlays,
            }}
          />
        </div>
      </div>
      {/* Deleting a run's output is at least as destructive as Mark Image
          Done's discard, so it gets the same standard: live counts read when
          the dialog opens, refusals rendered in the dialog, and the confirm
          button naming the number it deletes. */}
      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title={`Delete ${deleteTarget?.segmentation_type.long_name ?? "this segmentation"} from this image?`}
        message={
          "This permanently deletes the segmentation and everything it produced. " +
          "Nothing is archived, so it cannot be undone — getting the objects " +
          "back means running the model again."
        }
        details={
          <SegmentationDeleteNotice
            preview={deletePreview}
            previewError={deletePreviewError}
            submitError={deleteSubmitError}
          />
        }
        detailsTone="warning"
        confirmText={
          deleting
            ? "Deleting…"
            : deletePreview && deletePreview.object_count > 0
              ? `Delete ${pluraliseObjects(deletePreview.object_count)} and this segmentation`
              : "Delete segmentation"
        }
        cancelText="Cancel"
        onConfirm={() => {
          void confirmDeleteSegmentation();
        }}
        onCancel={() => setDeleteTarget(null)}
      />
      <OverlaySelectionSidebar
        overlays={overlayList}
        onToggle={handleToggle}
        onColorChange={handleColorChange}
        onOpacityChange={handleOpacityChange}
        onEditSegmentation={handleEditSegmentation}
        onDeleteSegmentation={handleRequestDeleteSegmentation}
        disabled={!imageReady}
        statusMessage={
          image && !imageReady
            ? `${getStageLabel(image.preprocess_stage)}${image.preprocess_progress > 0 ? ` (${Math.round(image.preprocess_progress)}%)` : ""}.\nOverlays are unavailable until preprocessing is complete.`
            : undefined
        }
        createPanel={
          selectedAssetId ? (
            <SegmentationCreatePanel
              imageId={selectedAssetId}
              onCreated={handleSegmentationCreated}
              existingSegmentationTypes={existingSegmentationTypes}
              title="Add segmentation"
              description="Choose a preset or create a custom segmentation."
              // So the confirmation can say what a missing pixel size costs the
              // run it is about to queue. `null` here is a fact ("this image is
              // uncalibrated"), not a missing value.
              pixelSizeNm={image.pixel_size_nm ?? null}
            />
          ) : null
        }
      />
    </div>
  );
}

/**
 * What deleting this segmentation destroys, what it keeps, and what it frees.
 *
 * The counts are the server's, read when the dialog opened; the DELETE carries
 * the object count so a run finishing mid-dialog is refused rather than
 * silently included. Analysis runs are named as *kept* because deleting the
 * numbers a paper may already cite would be the greater destruction — they
 * survive marked "segmentation deleted" and can no longer be traced back to
 * objects in the app.
 */
function SegmentationDeleteNotice({
  preview,
  previewError,
  submitError,
}: {
  preview: SegmentationDeletePreview | null;
  previewError: string | null;
  submitError: string | null;
}) {
  const refusal = submitError ? (
    // Same class as Mark Done's refusal box — styled in ConfirmDialog.css.
    <p className="segmentation-discard-refusal" role="alert">
      {submitError}
    </p>
  ) : null;

  if (previewError) {
    return (
      <>
        {refusal}
        <p>
          {previewError} Deleting will still remove every object, overlay
          raster, probability map and adapted model this segmentation holds.
        </p>
      </>
    );
  }

  if (!preview) {
    return (
      <>
        {refusal}
        <p>Counting what this segmentation holds…</p>
      </>
    );
  }

  const confirmed = preview.objects_by_label_state.CONFIRMED ?? 0;
  const excluded = preview.objects_by_label_state.EXCLUDED ?? 0;
  const unreviewed = Math.max(preview.object_count - confirmed - excluded, 0);

  const alsoDeleted: string[] = [];
  if (preview.overlay_count > 0) {
    alsoDeleted.push(
      `${preview.overlay_count} overlay raster${preview.overlay_count === 1 ? "" : "s"}`
    );
  }
  if (preview.probability_map_count > 0) {
    alsoDeleted.push(
      `${preview.probability_map_count} probability map${
        preview.probability_map_count === 1 ? "" : "s"
      }`
    );
  }
  if (preview.adapter_count > 0) {
    alsoDeleted.push(
      `${preview.adapter_count} adapted model${
        preview.adapter_count === 1 ? "" : "s"
      } (including any trained weights)`
    );
  }

  return (
    <>
      {refusal}
      {preview.locked ? (
        <p>
          <strong>This segmentation is locked.</strong> It was marked done, and
          the server refuses to delete it in that state. Unlock it first
          ("Unlock segmentation" on the labeling screen), then delete.
        </p>
      ) : null}
      <p>
        {preview.object_count === 0
          ? "This segmentation holds no objects"
          : `This deletes all ${pluraliseObjects(preview.object_count)} on this segmentation — ${confirmed} confirmed, ${excluded} rejected and ${unreviewed} nobody reviewed`}
        {alsoDeleted.length > 0
          ? `${preview.object_count === 0 ? ", but deleting it removes" : " — together with"} its ${alsoDeleted.join(", ")}`
          : ""}
        .
      </p>
      {excluded > 0 ? (
        <p>
          Rejections are ground truth: "Adapt a model" trains against them as
          negative examples, and deleting them deletes that record.
        </p>
      ) : null}
      {preview.analysis_run_count > 0 ? (
        <p>
          The {preview.analysis_run_count} analysis run
          {preview.analysis_run_count === 1 ? "" : "s"} made from it{" "}
          {preview.analysis_run_count === 1 ? "is" : "are"} <strong>kept</strong>,
          with {preview.analysis_run_count === 1 ? "its" : "their"} export
          bundle{preview.analysis_run_count === 1 ? "" : "s"}: the numbers are
          the record of an analysis that happened. They are marked "segmentation
          deleted" and can no longer be traced back to objects in the app.
        </p>
      ) : null}
      <p>
        Afterwards the {preview.segmentation_type} preset returns to "Add
        segmentation", so it can be recreated — recreating it queues a fresh
        model run.
      </p>
    </>
  );
}
