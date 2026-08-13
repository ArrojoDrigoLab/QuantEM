import { useCallback, useEffect, useMemo, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { getAssetNgffUrl } from "@/shared/api/assets";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { useViewportSync } from "@/hooks/useViewportSync";
import { usePolygonTraceWorkflow } from "@/features/segmentation/screen/hooks/usePolygonTraceWorkflow";
import { generateCompletedRoiDraftOverlays } from "@/features/segmentation/overlays/completedRois";
import { generateRoiStrokeOverlays } from "@/features/segmentation/overlays/roi";
import { useOverlayConfig } from "@/features/viewer/state/useOverlayConfig";
import { useViewerAssetState } from "@/features/viewer/state/useViewerAssetState";
import { brushStrokesToConnectedPolygonRings } from "@/utils/brushMask";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { Point } from "@/utils/geometry";
import { ImageViewer } from "@/viewer/components/ImageViewer";
import { panKeyState } from "@/viewer/panKeyState";
import type { SegmentOverlay } from "@/viewer/types";
import {
  deleteAnalysisMaskObject,
  listAnalysisMaskObjects,
  patchAnalysisMaskObject,
  renameAnalysisMaskObject,
  saveAnalysisMaskObjects,
} from "./api";
import {
  AnalysisMaskSidebar,
  type AnalysisMaskTool,
} from "./AnalysisMaskSidebar";
import { analysisMaskObjectsOverlays } from "./overlays";
import type {
  AnalysisMaskObject,
  AnalysisMaskOperation,
  AnalysisMaskShape,
} from "./types";
import "./AnalysisMaskScreen.css";

const ANALYSIS_MASK_INTERNAL_NAME = "quantem_internal_analysis_mask";

function upsertObject(
  objects: AnalysisMaskObject[],
  updated: AnalysisMaskObject
): AnalysisMaskObject[] {
  const existing = objects.findIndex((object) => object.id === updated.id);
  if (existing < 0) {
    return [...objects, updated].sort(
      (left, right) => left.sort_order - right.sort_order
    );
  }
  return objects.map((object) => (object.id === updated.id ? updated : object));
}

export function AnalysisMaskScreen() {
  const navigate = useNavigate();
  const { segmentationTypeName } = useParams();
  const {
    selectedAssetId,
    selectedSegmentationId,
    setSelectedSegmentationId,
  } = useSelectionStore();
  const { viewport, setViewport } = useViewportSync();
  const {
    image,
    imageReady,
    visibleSegmentations,
    refetchSegmentations,
  } = useViewerAssetState(selectedAssetId);

  const currentSegmentation = useMemo(
    () =>
      visibleSegmentations.find(
        (segmentation) => segmentation.id === segmentationTypeName
      ) ??
      visibleSegmentations.find(
        (segmentation) => segmentation.id === selectedSegmentationId
      ) ??
      null,
    [segmentationTypeName, selectedSegmentationId, visibleSegmentations]
  );
  const currentSegmentationId = currentSegmentation?.id ?? null;
  const otherAnalysisMasks = useMemo(
    () =>
      visibleSegmentations.filter(
        (segmentation) =>
          segmentation.id !== currentSegmentationId &&
          segmentation.segmentation_type.internal_name ===
            ANALYSIS_MASK_INTERNAL_NAME
      ),
    [currentSegmentationId, visibleSegmentations]
  );
  const existingMaskOverlays = useOverlayConfig({
    visibleSegmentations: otherAnalysisMasks,
    imageReady,
  });

  const {
    data: objectResponse,
    loading: objectsLoading,
    error: objectsError,
  } = useApiQuery(
    () => {
      if (!currentSegmentationId) {
        throw new Error("No analysis mask is selected.");
      }
      return listAnalysisMaskObjects(currentSegmentationId);
    },
    [currentSegmentationId]
  );
  const [objects, setObjects] = useState<AnalysisMaskObject[]>([]);
  const [activeObjectId, setActiveObjectId] = useState<string | null>(null);
  const [tool, setTool] = useState<AnalysisMaskTool>("polygon");
  const [operation, setOperation] =
    useState<AnalysisMaskOperation>("include");
  const [brushSize, setBrushSize] = useState(24);
  const [navigateMode, setNavigateMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pendingBrush, setPendingBrush] = useState<{
    id: string;
    label: number;
    size: number;
    points: Point[];
  } | null>(null);
  const [deleteTarget, setDeleteTarget] =
    useState<AnalysisMaskObject | null>(null);
  const [message, setMessage] = useState<{
    tone: "error" | "notice";
    text: string;
  } | null>(null);

  useEffect(() => {
    setObjects(objectResponse?.objects ?? []);
  }, [objectResponse]);

  useEffect(() => {
    if (!currentSegmentationId) return;
    setSelectedSegmentationId(currentSegmentationId);
    setActiveObjectId(null);
    setTool("polygon");
    setOperation("include");
    setNavigateMode(false);
  }, [currentSegmentationId, setSelectedSegmentationId]);

  useEffect(() => {
    if (activeObjectId === null && operation !== "include") {
      setOperation("include");
    }
  }, [activeObjectId, operation]);

  const showError = useCallback((text: string) => {
    setMessage({ tone: "error", text });
  }, []);

  const isPointInsideImageBounds = useCallback(
    (point: Point) =>
      Boolean(
        image &&
          Number.isFinite(point.x) &&
          Number.isFinite(point.y) &&
          point.x >= 0 &&
          point.y >= 0 &&
          point.x <= image.width &&
          point.y <= image.height
      ),
    [image]
  );

  const applyShapes = useCallback(
    async (shapes: AnalysisMaskShape[], requestedOperation = operation) => {
      if (!currentSegmentationId || busy || shapes.length === 0) return;
      const effectiveOperation = activeObjectId
        ? requestedOperation
        : "include";
      setBusy(true);
      try {
        const response = await patchAnalysisMaskObject(currentSegmentationId, {
          objectId: activeObjectId,
          operation: effectiveOperation,
          shapes,
        });
        setObjects((current) => upsertObject(current, response.object));
        setActiveObjectId(response.object.id);
      } catch (error) {
        showError(
          extractApiErrorMessage(error, "The analysis-mask object could not be saved.")
        );
        throw error;
      } finally {
        setBusy(false);
      }
    },
    [
      activeObjectId,
      busy,
      currentSegmentationId,
      operation,
      showError,
    ]
  );

  const polygon = usePolygonTraceWorkflow({
    active: tool === "polygon" && !navigateMode,
    idPrefix: "analysis-mask-polygon",
    isPointInsideImageBounds,
    registerAnnotationActivity: () => {},
    showErrorToast: showError,
    resetKey: currentSegmentationId,
    commitErrorMessage: "The polygon could not be saved to this analysis-mask object.",
    onCommit: useCallback(
      (ring: Array<[number, number]>) => applyShapes([{ rings: [ring] }]),
      [applyShapes]
    ),
  });

  const handleBrushStroke = useCallback(
    async (points: Point[]) => {
      if (busy || points.length === 0) return;
      const stroke = {
        id: `analysis-mask-brush-${Date.now()}`,
        label: operation === "include" ? 1 : 0,
        size: brushSize,
        points,
        operation,
      };
      setPendingBrush(stroke);
      const polygons = brushStrokesToConnectedPolygonRings([stroke]);
      const shapes = polygons.map((polygon) => ({
        rings: [polygon.exterior, ...polygon.holes].map((ring) =>
          ring.map((point) => [point.x, point.y] as [number, number])
        ),
      }));
      try {
        await applyShapes(shapes);
      } finally {
        setPendingBrush(null);
      }
    },
    [applyShapes, brushSize, busy, operation]
  );

  const saveActiveObject = useCallback(async (): Promise<boolean> => {
    if (!currentSegmentationId || busy || polygon.hasDraft) return false;
    setBusy(true);
    try {
      const response = await saveAnalysisMaskObjects(currentSegmentationId);
      setObjects(response.objects);
      setActiveObjectId(null);
      setOperation("include");
      void refetchSegmentations();
      return true;
    } catch (error) {
      showError(
        extractApiErrorMessage(error, "The analysis mask could not be saved.")
      );
      return false;
    } finally {
      setBusy(false);
    }
  }, [
    busy,
    currentSegmentationId,
    polygon.hasDraft,
    refetchSegmentations,
    showError,
  ]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable
      ) {
        return;
      }
      const plain =
        !event.repeat && !event.ctrlKey && !event.metaKey && !event.altKey;
      const lower = event.key.toLowerCase();
      if (plain && lower === "a") {
        event.preventDefault();
        setNavigateMode((current) => !current);
        return;
      }
      if (
        plain &&
        lower === "r" &&
        tool === "polygon" &&
        !navigateMode &&
        polygon.canClosePolygon
      ) {
        event.preventDefault();
        void polygon.handleClosePolygon();
        return;
      }
      if (
        (event.key === "Delete" || event.key === "Backspace") &&
        polygon.hasDraft
      ) {
        event.preventDefault();
        polygon.clearDraft();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key !== " " && event.code !== "Space") return;
      const panned = panKeyState.consumeSpacePan();
      if (panned || !activeObjectId || busy || polygon.hasDraft) return;
      const target = event.target as HTMLElement | null;
      if (
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable
      ) {
        return;
      }
      event.preventDefault();
      void saveActiveObject();
    };
    window.addEventListener("keydown", onKeyDown, true);
    window.addEventListener("keyup", onKeyUp, true);
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("keyup", onKeyUp, true);
    };
  }, [
    activeObjectId,
    busy,
    navigateMode,
    polygon,
    saveActiveObject,
    tool,
  ]);

  const currentObjectOverlays = useMemo(
    () => analysisMaskObjectsOverlays(objects, activeObjectId),
    [activeObjectId, objects]
  );
  const polygonOverlays = useMemo(
    () =>
      generateCompletedRoiDraftOverlays(
        polygon.polygons,
        polygon.liveSectionPoints,
        activeObjectId ? operation : "include"
      ).map((overlay) => ({ ...overlay, id: `analysis-mask-${overlay.id}` })),
    [activeObjectId, operation, polygon.liveSectionPoints, polygon.polygons]
  );
  const pendingBrushOverlays = useMemo<SegmentOverlay[]>(
    () => (pendingBrush ? generateRoiStrokeOverlays([pendingBrush]) : []),
    [pendingBrush]
  );

  const renameObject = useCallback(
    async (objectId: string, name: string) => {
      if (!currentSegmentationId) return;
      try {
        const updated = await renameAnalysisMaskObject(
          currentSegmentationId,
          objectId,
          name
        );
        setObjects((current) => upsertObject(current, updated));
      } catch (error) {
        showError(extractApiErrorMessage(error, "The object name could not be saved."));
        throw error;
      }
    },
    [currentSegmentationId, showError]
  );

  const confirmDeleteObject = useCallback(async () => {
    if (!currentSegmentationId || !deleteTarget || busy) return;
    setBusy(true);
    try {
      await deleteAnalysisMaskObject(currentSegmentationId, deleteTarget.id);
      setObjects((current) =>
        current.filter((object) => object.id !== deleteTarget.id)
      );
      if (activeObjectId === deleteTarget.id) {
        setActiveObjectId(null);
        setOperation("include");
      }
      setDeleteTarget(null);
    } catch (error) {
      showError(extractApiErrorMessage(error, "The object could not be deleted."));
    } finally {
      setBusy(false);
    }
  }, [
    activeObjectId,
    busy,
    currentSegmentationId,
    deleteTarget,
    showError,
  ]);

  const openViewer = useCallback(async () => {
    if (!selectedAssetId || busy || polygon.hasDraft) return;
    if (await saveActiveObject()) {
      navigate(`/assets/${selectedAssetId}/viewer`);
    }
  }, [busy, navigate, polygon.hasDraft, saveActiveObject, selectedAssetId]);

  if (!image || objectsLoading || !currentSegmentation) {
    return <div className="analysis-mask-loading">Loading analysis mask...</div>;
  }
  if (
    currentSegmentation.segmentation_type.internal_name !==
    ANALYSIS_MASK_INTERNAL_NAME
  ) {
    return <Navigate to={`/assets/${image.id}/viewer`} replace />;
  }

  return (
    <div className="analysis-mask-screen">
      <header className="analysis-mask-header">
        <div className="analysis-mask-header-left">
          <button type="button" onClick={() => void openViewer()}>
            ← Back to Viewer
          </button>
          <h2>{image.display_name}</h2>
        </div>
        <button
          type="button"
          className="analysis-mask-save-all"
          disabled={busy || polygon.hasDraft}
          title={
            polygon.hasDraft
              ? "Close or clear the unfinished polygon first."
              : undefined
          }
          onClick={() => void openViewer()}
        >
          Save Analysis Masks
        </button>
        <div className="analysis-mask-header-spacer" aria-hidden />
      </header>

      <main className="analysis-mask-main">
        <AnalysisMaskSidebar
          tool={tool}
          onToolChange={(nextTool) => {
            setTool(nextTool);
            setNavigateMode(false);
          }}
          operation={operation}
          onOperationChange={setOperation}
          canExclude={activeObjectId !== null}
          brushSize={brushSize}
          onBrushSizeChange={setBrushSize}
          polygonHasDraft={polygon.hasDraft}
          polygonCanClose={polygon.canClosePolygon}
          polygonSaving={polygon.submitting}
          onClosePolygon={() => {
            void polygon.handleClosePolygon();
          }}
          onClearPolygon={polygon.clearDraft}
          navigateMode={navigateMode}
          onNavigateModeChange={setNavigateMode}
          objects={objects}
          activeObjectId={activeObjectId}
          busy={busy}
          onEditObject={(objectId) => {
            setActiveObjectId(objectId);
            setOperation("include");
            setNavigateMode(false);
          }}
          onSaveObject={() => {
            void saveActiveObject();
          }}
          onRenameObject={renameObject}
          onRequestDeleteObject={setDeleteTarget}
          existingMaskLayers={existingMaskOverlays.overlayList}
          onToggleExistingMask={existingMaskOverlays.handleToggle}
        />
        <section className="analysis-mask-canvas">
          <ImageViewer
            image={{
              ngffUrl: getAssetNgffUrl(image.id, null),
              width: image.width,
              height: image.height,
            }}
            className="viewer-container"
            viewport={{ state: viewport ?? undefined, onChange: setViewport }}
            overlays={{
              persistent: currentObjectOverlays,
              transient: [...polygonOverlays, ...pendingBrushOverlays],
              idMapOverlays: existingMaskOverlays.viewerIdMapOverlays,
            }}
            interactions={{
              onImageClick:
                tool === "polygon" && !navigateMode
                  ? polygon.handlePolygonClick
                  : undefined,
              onImageMouseMove:
                tool === "polygon" && !navigateMode
                  ? polygon.handlePolygonMouseMove
                  : undefined,
              brush: {
                enabled: tool === "brush" && !navigateMode && !busy,
                size: brushSize,
                color:
                  (activeObjectId ? operation : "include") === "exclude"
                    ? "#ef4444"
                    : "#22c55e",
                onStroke: (points) => {
                  void handleBrushStroke(points);
                },
              },
            }}
          />
          {busy ? <div className="analysis-mask-saving">Saving shape...</div> : null}
        </section>
      </main>

      <ConfirmDialog
        isOpen={deleteTarget !== null}
        title="Delete analysis-mask object?"
        message={
          deleteTarget
            ? `Delete ${deleteTarget.name}? Its area will be removed from this analysis mask.`
            : ""
        }
        confirmText={busy ? "Deleting..." : "Delete object"}
        cancelText="Cancel"
        confirmDisabled={busy}
        onConfirm={() => {
          void confirmDeleteObject();
        }}
        onCancel={() => setDeleteTarget(null)}
      />

      {objectsError ? (
        <div className="analysis-mask-toast error" role="alert">
          {objectsError.message}
        </div>
      ) : null}
      {message ? (
        <div
          className={`analysis-mask-toast ${message.tone}`}
          role={message.tone === "error" ? "alert" : "status"}
        >
          <span>{message.text}</span>
          <button type="button" onClick={() => setMessage(null)}>
            Dismiss
          </button>
        </div>
      ) : null}
    </div>
  );
}
