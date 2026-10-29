import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Navigate, useLocation, useNavigate, useParams } from "react-router-dom";
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
import type { AssetDetail, ImageSegmentation } from "@/shared/types/images";
import {
  deleteAnalysisMaskObject,
  listAnalysisMaskObjects,
  patchAnalysisMaskObject,
  renameAnalysisMaskObject,
  saveAnalysisMaskObjects,
  updateAnalysisMaskObjectMetadata,
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
  CellTag,
  CellTagSuggestion,
} from "./types";
import "./AnalysisMaskScreen.css";

const CELLS_INTERNAL_NAME = "quantem_internal_cells";
const OPTIMISTIC_CELL_COLORS = [
  "#38bdf8",
  "#22c55e",
  "#f97316",
  "#a855f7",
  "#f43f5e",
  "#eab308",
  "#14b8a6",
  "#6366f1",
  "#ec4899",
  "#84cc16",
] as const;

interface AnalysisMaskRouteState {
  analysisMaskHandoff?: {
    image: AssetDetail;
    segmentation: ImageSegmentation;
    created: boolean;
  };
}

function isOptimisticCell(object: AnalysisMaskObject): boolean {
  return object.id.startsWith("pending-cell-");
}

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
  const location = useLocation();
  const { segmentationTypeName } = useParams();
  const {
    selectedAssetId,
    selectedSegmentationId,
    setSelectedSegmentationId,
  } = useSelectionStore();
  const { viewport, setViewport } = useViewportSync();
  const {
    image: fetchedImage,
    imageReady,
    visibleSegmentations,
    refetchSegmentations,
  } = useViewerAssetState(selectedAssetId);

  const routeHandoff = (location.state as AnalysisMaskRouteState | null)
    ?.analysisMaskHandoff;
  const handoff =
    routeHandoff &&
    routeHandoff.image.id === selectedAssetId &&
    routeHandoff.segmentation.id === segmentationTypeName
      ? routeHandoff
      : null;
  const image = fetchedImage ?? handoff?.image ?? null;
  const segmentationsWithHandoff = useMemo(() => {
    if (!handoff) return visibleSegmentations;
    if (
      visibleSegmentations.some(
        (segmentation) => segmentation.id === handoff.segmentation.id
      )
    ) {
      return visibleSegmentations;
    }
    return [...visibleSegmentations, handoff.segmentation];
  }, [handoff, visibleSegmentations]);

  const currentSegmentation = useMemo(
    () =>
      segmentationsWithHandoff.find(
        (segmentation) => segmentation.id === segmentationTypeName
      ) ??
      segmentationsWithHandoff.find(
        (segmentation) => segmentation.id === selectedSegmentationId
      ) ??
      null,
    [segmentationTypeName, selectedSegmentationId, segmentationsWithHandoff]
  );
  const currentSegmentationId = currentSegmentation?.id ?? null;
  const isCells =
    currentSegmentation?.segmentation_type.internal_name === CELLS_INTERNAL_NAME;
  const otherAnalysisMasks = useMemo(
    () =>
      segmentationsWithHandoff.filter(
        (segmentation) =>
          segmentation.id !== currentSegmentationId &&
          segmentation.segmentation_type.category === "area"
      ),
    [currentSegmentationId, segmentationsWithHandoff]
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
  const [metadataEditingObjectId, setMetadataEditingObjectId] = useState<
    string | null
  >(null);
  const [tool, setTool] = useState<AnalysisMaskTool>("polygon");
  const [operation, setOperation] =
    useState<AnalysisMaskOperation>("include");
  const [brushSize, setBrushSize] = useState(24);
  const [navigateMode, setNavigateMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [pendingCellSaves, setPendingCellSaves] = useState(0);
  const cellSaveQueueRef = useRef<Promise<void>>(Promise.resolve());
  const optimisticCellCounterRef = useRef(0);
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
    dismissAfterMs?: number;
  } | null>(null);

  useEffect(() => {
    if (!objectResponse) return;
    setObjects((current) => {
      const pending = current.filter(isOptimisticCell);
      return [...objectResponse.objects, ...pending].sort(
        (left, right) => left.sort_order - right.sort_order
      );
    });
  }, [objectResponse]);

  const cellTagSuggestions = useMemo<CellTagSuggestion[]>(() => {
    const byName = new Map<
      string,
      { name: string; values: Map<string, string> }
    >();
    const include = (name: string, values: string[]) => {
      const trimmedName = name.trim();
      if (!trimmedName) return;
      const nameKey = trimmedName.toLowerCase();
      const entry = byName.get(nameKey) ?? {
        name: trimmedName,
        values: new Map<string, string>(),
      };
      for (const value of values) {
        const trimmedValue = value.trim();
        if (trimmedValue) {
          entry.values.set(
            trimmedValue.toLowerCase(),
            entry.values.get(trimmedValue.toLowerCase()) ?? trimmedValue
          );
        }
      }
      byName.set(nameKey, entry);
    };

    for (const suggestion of objectResponse?.tag_suggestions ?? []) {
      include(suggestion.name, suggestion.values);
    }
    for (const object of objects) {
      for (const tag of object.tags) include(tag.name, [tag.value]);
    }
    return Array.from(byName.values()).map((entry) => ({
      name: entry.name,
      values: Array.from(entry.values.values()),
    }));
  }, [objectResponse?.tag_suggestions, objects]);

  useEffect(() => {
    if (!currentSegmentationId) return;
    setSelectedSegmentationId(currentSegmentationId);
    setActiveObjectId(null);
    setMetadataEditingObjectId(null);
    setTool("polygon");
    setOperation("include");
    setNavigateMode(false);
  }, [currentSegmentationId, setSelectedSegmentationId]);

  useEffect(() => {
    if (activeObjectId === null && operation !== "include") {
      setOperation("include");
    }
    if (isCells && activeObjectId === null && tool !== "polygon") {
      setTool("polygon");
    }
  }, [activeObjectId, isCells, operation, tool]);

  const showError = useCallback((text: string) => {
    setMessage({ tone: "error", text });
  }, []);

  const showShapeError = useCallback(
    (text: string) => {
      setMessage({
        tone: "error",
        text,
        ...(isCells ? { dismissAfterMs: 3000 } : {}),
      });
    },
    [isCells]
  );

  useEffect(() => {
    if (!message?.dismissAfterMs) return undefined;
    const shownMessage = message;
    const timeout = window.setTimeout(() => {
      setMessage((current) => (current === shownMessage ? null : current));
    }, message.dismissAfterMs);
    return () => window.clearTimeout(timeout);
  }, [message]);

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

      // A newly closed Cell is already a complete object. Put it on screen
      // immediately and serialize the small vector request behind any prior
      // Cell save; the next click can begin the next Cell without waiting for
      // disk, rasterization, or a round trip.
      if (isCells && activeObjectId === null) {
        optimisticCellCounterRef.current += 1;
        const optimisticNumber = optimisticCellCounterRef.current;
        const existingMaxOrder = objects.reduce(
          (maximum, object) => Math.max(maximum, object.sort_order),
          0
        );
        const optimisticOrder = existingMaxOrder + 1;
        const optimisticId = `pending-cell-${Date.now()}-${optimisticNumber}`;
        const now = new Date().toISOString();
        const optimisticObject: AnalysisMaskObject = {
          id: optimisticId,
          segmentation: currentSegmentationId,
          name: `Cell ${optimisticOrder}`,
          color:
            OPTIMISTIC_CELL_COLORS[
              (optimisticOrder - 1) % OPTIMISTIC_CELL_COLORS.length
            ],
          sort_order: optimisticOrder,
          tags: [],
          geometry: {
            type: "Polygon",
            coordinates: shapes[0].rings,
          },
          created_at: now,
          updated_at: now,
        };
        setObjects((current) => [...current, optimisticObject]);
        setPendingCellSaves((current) => current + 1);

        const request = cellSaveQueueRef.current.then(() =>
          patchAnalysisMaskObject(currentSegmentationId, {
            objectId: null,
            operation: "include",
            shapes,
          })
        );
        cellSaveQueueRef.current = request.then(
          () => undefined,
          () => undefined
        );
        void request
          .then((response) => {
            setObjects((current) =>
              current
                .filter(
                  (object) =>
                    object.id !== response.object.id ||
                    object.id === optimisticId
                )
                .map((object) =>
                  object.id === optimisticId ? response.object : object
                )
                .sort((left, right) => left.sort_order - right.sort_order)
            );
          })
          .catch((error) => {
            setObjects((current) =>
              current.filter((object) => object.id !== optimisticId)
            );
            showShapeError(
              extractApiErrorMessage(
                error,
                "The cell could not be saved. Its outline was removed so it can be retried."
              )
            );
          })
          .finally(() => {
            setPendingCellSaves((current) => Math.max(0, current - 1));
          });
        return;
      }

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
        showShapeError(
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
      isCells,
      objects,
      operation,
      showShapeError,
    ]
  );

  const polygon = usePolygonTraceWorkflow({
    active:
      tool === "polygon" &&
      !navigateMode &&
      metadataEditingObjectId === null,
    idPrefix: "analysis-mask-polygon",
    isPointInsideImageBounds,
    registerAnnotationActivity: () => {},
    showErrorToast: showShapeError,
    resetKey: currentSegmentationId,
    commitErrorMessage: "The polygon could not be saved to this analysis-mask object.",
    onCommit: useCallback(
      (ring: Array<[number, number]>) => applyShapes([{ rings: [ring] }]),
      [applyShapes]
    ),
  });

  const handleBrushStroke = useCallback(
    async (points: Point[]) => {
      if (busy || metadataEditingObjectId !== null || points.length === 0) return;
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
    [applyShapes, brushSize, busy, metadataEditingObjectId, operation]
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

  const finishActiveObject = useCallback(async (): Promise<boolean> => {
    if (isCells) {
      if (busy || polygon.hasDraft) return false;
      setActiveObjectId(null);
      setOperation("include");
      setTool("polygon");
      return true;
    }
    return saveActiveObject();
  }, [busy, isCells, polygon.hasDraft, saveActiveObject]);

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
      if (metadataEditingObjectId !== null) return;
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
      void finishActiveObject();
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
    finishActiveObject,
    metadataEditingObjectId,
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

  const updateObjectMetadata = useCallback(
    async (objectId: string, name: string, tags: CellTag[]) => {
      if (!currentSegmentationId) return;
      try {
        const updated = isCells
          ? await updateAnalysisMaskObjectMetadata(
              currentSegmentationId,
              objectId,
              { name, tags }
            )
          : await renameAnalysisMaskObject(
              currentSegmentationId,
              objectId,
              name
            );
        setObjects((current) => upsertObject(current, updated));
      } catch (error) {
        showError(
          extractApiErrorMessage(
            error,
            "The cell name and tags could not be saved."
          )
        );
        throw error;
      }
    },
    [currentSegmentationId, isCells, showError]
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
      if (metadataEditingObjectId === deleteTarget.id) {
        setMetadataEditingObjectId(null);
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
    metadataEditingObjectId,
    showError,
  ]);

  const openViewer = useCallback(async () => {
    if (!selectedAssetId || busy || polygon.hasDraft) return;
    if (isCells) {
      await cellSaveQueueRef.current;
      void refetchSegmentations();
      navigate(`/assets/${selectedAssetId}/viewer`);
      return;
    }
    if (await finishActiveObject()) {
      navigate(`/assets/${selectedAssetId}/viewer`);
    }
  }, [
    busy,
    finishActiveObject,
    isCells,
    navigate,
    polygon.hasDraft,
    refetchSegmentations,
    selectedAssetId,
  ]);

  if (
    !image ||
    !currentSegmentation ||
    (objectsLoading && !handoff?.created)
  ) {
    return <div className="analysis-mask-loading">Loading analysis mask...</div>;
  }
  if (currentSegmentation.segmentation_type.category !== "area") {
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
        {isCells ? (
          <div
            className={`analysis-mask-autosave ${pendingCellSaves > 0 ? "is-saving" : ""}`}
            role="status"
          >
            <span aria-hidden="true" />
            Cells save automatically
          </div>
        ) : (
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
            Save Analysis Mask
          </button>
        )}
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
          objectNamesRequired={
            currentSegmentation.segmentation_type.internal_name ===
            CELLS_INTERNAL_NAME
          }
          cellsMode={isCells}
          activeObjectId={activeObjectId}
          metadataEditingObjectId={metadataEditingObjectId}
          onMetadataEditingObjectChange={setMetadataEditingObjectId}
          metadataEditDisabled={polygon.hasDraft}
          tagSuggestions={cellTagSuggestions}
          busy={busy}
          onEditObject={(objectId) => {
            setActiveObjectId(objectId);
            setOperation("include");
            setTool("polygon");
            setNavigateMode(false);
          }}
          onSaveObject={() => {
            void finishActiveObject();
          }}
          onUpdateObjectMetadata={updateObjectMetadata}
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
              mode:
                navigateMode || metadataEditingObjectId !== null
                  ? "navigate"
                  : undefined,
              onImageClick:
                tool === "polygon" &&
                !navigateMode &&
                metadataEditingObjectId === null
                  ? polygon.handlePolygonClick
                  : undefined,
              onImageMouseMove:
                tool === "polygon" &&
                !navigateMode &&
                metadataEditingObjectId === null
                  ? polygon.handlePolygonMouseMove
                  : undefined,
              brush: {
                enabled:
                  tool === "brush" &&
                  !navigateMode &&
                  !busy &&
                  metadataEditingObjectId === null,
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
          {busy ? <div className="analysis-mask-saving">Saving edit...</div> : null}
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
        confirmTone="danger"
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
