import { useCallback, useEffect, useMemo } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useViewportSyncGroup } from "@/viewer/viewportSync/useViewportSyncGroup";
import { calculateViewportBbox } from "@/utils/viewportUtils";
import { createViewportActionResolver } from "@/utils/viewportActions";
import {
  SEGMENT_SMOOTHING_VIEWPORT_DIMENSION_THRESHOLD_PX,
} from "@/config";
import {
  POINT_FEEDBACK_SEGMENTATION_TYPES,
  STATUS_POLL_MS,
  TISSUE_INTERNAL_NAME,
} from "@/features/segmentation/screen/utils/constants";
import {
  defaultSourceModel,
  rememberSourceModel,
} from "@/features/segmentation/screen/utils/sourceModelMemory";

function normalizeSegmentationName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function formatStage(stage: string, progress: number) {
  if (stage === "DONE" || stage === "READY") return "Complete";
  if (stage === "FAILED") return "Failed";
  if (stage === "CANCELLED") return "Cancelled";
  if (stage === "SKIPPED") return "Not applicable";
  if (stage === "NONE") return "Not started";
  return `${stage} (${Math.round(progress)}%)`;
}

export function useSegmentationRouteState() {
  const navigate = useNavigate();
  const { segmentationTypeName } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    selectedAssetId,
    selectedSegmentationId,
    setSelectedSegmentationId,
    clearSelection,
  } = useSelectionStore();

  const { data: image, loading: imageLoading, refetch: refetchImage } = useApiQuery(
    () => {
      if (!selectedAssetId) throw new Error("No asset selected");
      return getAsset(selectedAssetId);
    },
    [selectedAssetId]
  );

  const {
    data: segmentations,
    loading: segmentationsLoading,
    refetch: refetchSegmentations,
  } = useApiQuery(
    () => {
      if (!selectedAssetId) throw new Error("No asset selected");
      return getAssetSegmentations(selectedAssetId);
    },
    [selectedAssetId]
  );

  const viewportGroupId = selectedAssetId ? `labeling:${selectedAssetId}` : "labeling:unknown";
  const viewportResolver = useMemo(() => {
    if (!image) return null;
    return createViewportActionResolver(image.width, image.height);
  }, [image]);

  const { viewport, publishFromViewer } = useViewportSyncGroup(viewportGroupId, {
    resolveAction: viewportResolver,
  });

  const useSmoothedSegmentGeometry = useMemo(() => {
    if (!image) return false;

    const bbox = calculateViewportBbox(viewport, image.width, image.height);
    const spanX = bbox ? bbox.x_max - bbox.x_min : image.width;
    const spanY = bbox ? bbox.y_max - bbox.y_min : image.height;
    return (
      spanX > SEGMENT_SMOOTHING_VIEWPORT_DIMENSION_THRESHOLD_PX ||
      spanY > SEGMENT_SMOOTHING_VIEWPORT_DIMENSION_THRESHOLD_PX
    );
  }, [image, viewport]);

  const visibleSegmentations = useMemo(() => segmentations ?? [], [segmentations]);

  const normalizedSegmentationName = useMemo(() => {
    if (!segmentationTypeName) return null;
    try {
      return normalizeSegmentationName(decodeURIComponent(segmentationTypeName));
    } catch {
      return normalizeSegmentationName(segmentationTypeName);
    }
  }, [segmentationTypeName]);

  const segmentationFromParam = useMemo(() => {
    if (!normalizedSegmentationName) return null;
    return (
      visibleSegmentations.find((seg) => {
        const candidateNames = [
          seg.segmentation_type?.long_name ?? "",
          seg.segmentation_type?.short_name ?? "",
          seg.segmentation_type?.internal_name ?? "",
        ];
        return candidateNames.some(
          (candidate) => normalizeSegmentationName(candidate) === normalizedSegmentationName
        );
      }) ?? null
    );
  }, [normalizedSegmentationName, visibleSegmentations]);

  const currentSegmentation =
    segmentationFromParam ??
    visibleSegmentations.find((seg) => seg.id === selectedSegmentationId) ??
    visibleSegmentations[0] ??
    null;
  const currentSegmentationId = currentSegmentation?.id ?? null;
  const segmentationInternalName =
    currentSegmentation?.segmentation_type?.internal_name ?? null;
  const isTissueSegmentation = segmentationInternalName === TISSUE_INTERNAL_NAME;
  const isErSegmentation =
    segmentationInternalName === "quantem_internal_er" ||
    segmentationInternalName === "quantem_internal_er_deepcontact_cell" ||
    segmentationInternalName === "quantem_internal_er_deepcontact_sem" ||
    segmentationInternalName === "quantem_internal_er_deepcontact_tem";
  const supportsPointFeedback = Boolean(
    segmentationInternalName &&
      POINT_FEEDBACK_SEGMENTATION_TYPES.has(segmentationInternalName)
  );
  const supportsInstanceParams = Boolean(currentSegmentation?.config?.instance_params);
  const sourceModelOptions = useMemo(
    () => currentSegmentation?.source_models ?? [],
    [currentSegmentation]
  );
  const sourceModelQueryParam = searchParams.get("source_model")?.trim() || null;
  const activeSourceModel = useMemo(() => {
    if (sourceModelOptions.length === 0) return null;
    // "none" is a synthetic selection (not a real backend source model). The
    // backend filter treats an unrecognized source model as "confirmed/manual
    // only", so the left pane shows no model-derived candidates.
    if (sourceModelQueryParam === "none") return "none";
    if (
      sourceModelQueryParam &&
      sourceModelOptions.some((option) => option.value === sourceModelQueryParam)
    ) {
      return sourceModelQueryParam;
    }
    // No usable URL param: follow the objects, not `is_default`. The catalogue
    // default is QuantEM on every organelle, so a reopened screen used to show
    // "No objects from QuantEM yet" while every candidate sat under OmniEM.
    return defaultSourceModel(sourceModelOptions, currentSegmentationId);
  }, [currentSegmentationId, sourceModelOptions, sourceModelQueryParam]);
  const currentInstanceParams = useMemo(
    () => currentSegmentation?.config?.instance_params ?? null,
    [currentSegmentation]
  );

  const handleOpenViewer = useCallback(() => {
    if (!image) return;
    navigate(`/assets/${image.id}/viewer`);
  }, [image, navigate]);

  const handleBackToHome = useCallback(() => {
    clearSelection();
    navigate("/");
  }, [clearSelection, navigate]);

  const handleSegmentationChange = useCallback(
    (segmentationId: string) => {
      const target = visibleSegmentations.find((seg) => seg.id === segmentationId);
      if (!target || !selectedAssetId) return;
      setSelectedSegmentationId(segmentationId);
      const encodedName = encodeURIComponent(target.segmentation_type.long_name);
      const params = new URLSearchParams();
      // Same objects-first rule as the mount-time default: switching to a
      // segmentation whose objects all came from OmniEM must not land on the
      // QuantEM view of it.
      const defaultSource = defaultSourceModel(
        target.source_models ?? [],
        target.id
      );
      if (defaultSource) {
        params.set("source_model", defaultSource);
      }
      const qs = params.toString();
      navigate(`/assets/${selectedAssetId}/labeling/${encodedName}${qs ? `?${qs}` : ""}`, {
        replace: true,
      });
    },
    [navigate, selectedAssetId, setSelectedSegmentationId, visibleSegmentations]
  );

  const handleSourceModelChange = useCallback(
    (sourceModel: string) => {
      // An explicit toggle is the one signal worth remembering: next time
      // this segmentation opens with no URL param and no single owner of the
      // objects, this choice is the default.
      rememberSourceModel(currentSegmentationId, sourceModel);
      const next = new URLSearchParams(searchParams);
      if (sourceModel) {
        next.set("source_model", sourceModel);
      } else {
        next.delete("source_model");
      }
      setSearchParams(next, { replace: true });
    },
    [currentSegmentationId, searchParams, setSearchParams]
  );

  useEffect(() => {
    if (!segmentationFromParam) return;
    if (segmentationFromParam.id === selectedSegmentationId) return;
    setSelectedSegmentationId(segmentationFromParam.id);
  }, [segmentationFromParam, selectedSegmentationId, setSelectedSegmentationId]);

  useEffect(() => {
    if (visibleSegmentations.length === 0) {
      setSelectedSegmentationId(null);
      return;
    }
    if (segmentationTypeName) {
      return;
    }
    if (
      selectedSegmentationId &&
      visibleSegmentations.some((seg) => seg.id === selectedSegmentationId)
    ) {
      return;
    }
    setSelectedSegmentationId(visibleSegmentations[0].id);
  }, [
    segmentationTypeName,
    selectedSegmentationId,
    setSelectedSegmentationId,
    visibleSegmentations,
  ]);

  useEffect(() => {
    if (!selectedAssetId) return;
    if (segmentationsLoading) return;
    if (!currentSegmentation) {
      if (segmentationTypeName !== "new") {
        navigate(`/assets/${selectedAssetId}/labeling/new`, { replace: true });
      }
      return;
    }

    const encodedName = encodeURIComponent(currentSegmentation.segmentation_type.long_name);
    const normalizedParam = normalizedSegmentationName ?? "";
    const normalizedCurrent = normalizeSegmentationName(
      currentSegmentation.segmentation_type.long_name
    );
    if (normalizedParam !== normalizedCurrent && segmentationTypeName !== encodedName) {
      navigate(
        `/assets/${selectedAssetId}/labeling/${encodedName}${
          searchParams.toString() ? `?${searchParams.toString()}` : ""
        }`,
        { replace: true }
      );
    }
  }, [
    currentSegmentation,
    navigate,
    normalizedSegmentationName,
    searchParams,
    selectedAssetId,
    segmentationsLoading,
    segmentationTypeName,
  ]);

  useEffect(() => {
    if (!activeSourceModel || sourceModelQueryParam === activeSourceModel) return;
    const next = new URLSearchParams(searchParams);
    next.set("source_model", activeSourceModel);
    setSearchParams(next, { replace: true });
  }, [activeSourceModel, searchParams, setSearchParams, sourceModelQueryParam]);

  const preprocessReady =
    image?.preprocess_stage === "DONE" ||
    image?.preprocess_stage === "FAILED" ||
    image?.preprocess_stage === "SKIPPED";
  const preprocessLabel =
    image ? formatStage(image.preprocess_stage, image.preprocess_progress) : "";

  useEffect(() => {
    if (!image || preprocessReady) return undefined;
    const interval = window.setInterval(() => {
      void refetchImage();
      void refetchSegmentations();
    }, STATUS_POLL_MS);
    return () => clearInterval(interval);
  }, [image, preprocessReady, refetchImage, refetchSegmentations]);

  return {
    selectedImageId: selectedAssetId,
    selectedAssetId,
    image,
    imageLoading,
    refetchImage,
    segmentationsLoading,
    visibleSegmentations,
    currentSegmentation,
    currentSegmentationId,
    segmentationInternalName,
    isTissueSegmentation,
    isErSegmentation,
    supportsPointFeedback,
    supportsInstanceParams,
    currentInstanceParams,
    sourceModelOptions,
    activeSourceModel,
    handleSourceModelChange,
    viewport,
    publishFromViewer,
    useSmoothedSegmentGeometry,
    refetchSegmentations,
    handleOpenViewer,
    handleSegmentationChange,
    handleBackToHome,
    preprocessReady,
    preprocessLabel,
  };
}
