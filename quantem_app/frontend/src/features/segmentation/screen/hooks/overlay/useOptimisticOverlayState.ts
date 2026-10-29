import type { OverlayRevisionTarget } from "@/shared/types/segmentation";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  generateLeftPanelOverlays,
  type LeftPanelLayerStyles,
} from "@/features/segmentation/overlays/segments";
import type {
  SegmentationOverlayMutationState,
  SegmentObject,
} from "@/shared/types/segmentation";
import type { LabelState } from "@/shared/types/common";

interface UseOptimisticOverlayStateArgs {
  currentSegmentationId: string | null;
  segmentationInternalName: string | null;
  useSmoothedSegmentGeometry: boolean;
  leftPanelLayerStyles: LeftPanelLayerStyles;
  hiddenSegmentIds: ReadonlySet<string>;
  settledOverlayRevision: number | null;
  settledOverlayLutRevision?: number | null;
}

export function useOptimisticOverlayState({
  currentSegmentationId,
  segmentationInternalName,
  useSmoothedSegmentGeometry,
  leftPanelLayerStyles,
  hiddenSegmentIds,
  settledOverlayRevision,
  settledOverlayLutRevision,
}: UseOptimisticOverlayStateArgs) {
  const [labelOverrides, setLabelOverrides] = useState<Record<string, LabelState>>({});
  const [optimisticSegments, setOptimisticSegments] = useState<
    Record<string, SegmentObject>
  >({});
  const [optimisticOverlayRevisionTargets, setOptimisticOverlayRevisionTargets] =
    useState<Record<string, OverlayRevisionTarget>>({});

  useEffect(() => {
    setLabelOverrides({});
    setOptimisticSegments({});
    setOptimisticOverlayRevisionTargets({});
  }, [currentSegmentationId]);

  const applyLabelOverrides = useCallback(
    (items: SegmentObject[]) =>
      items.filter((segment) => !hiddenSegmentIds.has(segment.id)).map((segment) => {
        const override = labelOverrides[segment.id];
        if (!override || override === segment.label_state) {
          return segment;
        }
        return { ...segment, label_state: override };
      }),
    [hiddenSegmentIds, labelOverrides]
  );

  const visibleOptimisticSegments = useMemo(
    () =>
      Object.values(optimisticSegments).filter(
        (segment) => !hiddenSegmentIds.has(segment.id)
      ),
    [hiddenSegmentIds, optimisticSegments]
  );

  const clearOptimisticSegments = useCallback((segmentIds: string[]) => {
    if (segmentIds.length === 0) return;
    const idsToClear = new Set(segmentIds);
    setOptimisticSegments((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const segmentId of idsToClear) {
        if (!next[segmentId]) continue;
        changed = true;
        delete next[segmentId];
      }
      return changed ? next : prev;
    });
    setOptimisticOverlayRevisionTargets((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const segmentId of idsToClear) {
        if (next[segmentId] === undefined) continue;
        changed = true;
        delete next[segmentId];
      }
      return changed ? next : prev;
    });
    setLabelOverrides((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const segmentId of idsToClear) {
        if (next[segmentId] === undefined) continue;
        changed = true;
        delete next[segmentId];
      }
      return changed ? next : prev;
    });
  }, []);

  const stageOptimisticSegments = useCallback(
    (items: SegmentObject[], targetRevision?: OverlayRevisionTarget | null) => {
      if (items.length === 0) return;
      setOptimisticSegments((prev) => {
        const next = { ...prev };
        for (const segment of items) {
          next[segment.id] = segment;
        }
        return next;
      });
      if (targetRevision != null) {
        setOptimisticOverlayRevisionTargets((prev) => {
          const next = { ...prev };
          for (const segment of items) {
            next[segment.id] = targetRevision;
          }
          return next;
        });
      }
    },
    []
  );

  const stageOptimisticRevisionTargets = useCallback(
    (segmentIds: string[], targetRevision?: OverlayRevisionTarget | null) => {
      if (
        segmentIds.length === 0 ||
        targetRevision == null
      ) {
        return;
      }
      setOptimisticOverlayRevisionTargets((prev) => {
        const next = { ...prev };
        for (const segmentId of segmentIds) {
          next[segmentId] = targetRevision;
        }
        return next;
      });
    },
    []
  );

  const getOptimisticTargetRevision = useCallback(
    (overlay: SegmentationOverlayMutationState | null | undefined): OverlayRevisionTarget | null => {
      if (!overlay) return null;
      if (typeof overlay.confirmed_display_desired_revision === "number") {
        return { revision: overlay.confirmed_display_desired_revision,
          lutRevision: overlay.confirmed_display_lut_revision ?? 0 };
      }
      return overlay.sync_applied ? overlay.applied_revision : overlay.desired_revision;
    },
    []
  );

  const rollbackOptimisticLabel = useCallback((segmentId: string) => {
    setLabelOverrides((prev) => {
      if (!prev[segmentId]) return prev;
      const next = { ...prev };
      delete next[segmentId];
      return next;
    });
    setOptimisticSegments((prev) => {
      if (!prev[segmentId]) return prev;
      const next = { ...prev };
      delete next[segmentId];
      return next;
    });
    setOptimisticOverlayRevisionTargets((prev) => {
      if (prev[segmentId] === undefined) return prev;
      const next = { ...prev };
      delete next[segmentId];
      return next;
    });
  }, []);

  const applyOptimisticLabel = useCallback(
    (
      segmentId: string,
      labelState: LabelState,
      sourceSegment?: SegmentObject | null,
      options?: { stageOverlay?: boolean }
    ) => {
      const stageOverlay = options?.stageOverlay ?? true;
      setLabelOverrides((prev) => ({ ...prev, [segmentId]: labelState }));
      if (!stageOverlay) {
        setOptimisticSegments((prev) => {
          if (!prev[segmentId]) return prev;
          const next = { ...prev };
          delete next[segmentId];
          return next;
        });
        return;
      }
      const resolvedSourceSegment = sourceSegment ?? optimisticSegments[segmentId] ?? null;
      if (!resolvedSourceSegment) return;
      setOptimisticSegments((prev) => ({
        ...prev,
        [segmentId]: { ...resolvedSourceSegment, label_state: labelState },
      }));
    },
    [optimisticSegments]
  );

  const optimisticConfirmed = useMemo(
    () =>
      visibleOptimisticSegments.filter(
        (segment) => segment.label_state === "CONFIRMED"
      ),
    [visibleOptimisticSegments]
  );
  const optimisticExcluded = useMemo(
    () =>
      visibleOptimisticSegments.filter(
        (segment) => segment.label_state === "EXCLUDED"
      ),
    [visibleOptimisticSegments]
  );

  const optimisticTransientOverlays = useMemo(
    () =>
      generateLeftPanelOverlays(
        visibleOptimisticSegments,
        false,
        segmentationInternalName,
        undefined,
        useSmoothedSegmentGeometry,
        leftPanelLayerStyles
      ),
    [
      leftPanelLayerStyles,
      segmentationInternalName,
      useSmoothedSegmentGeometry,
      visibleOptimisticSegments,
    ]
  );

  useEffect(() => {
    if (typeof settledOverlayRevision !== "number") return;
    const settledIds = Object.entries(optimisticOverlayRevisionTargets)
      .filter(([, target]) => typeof target === "number" ? settledOverlayRevision >= target :
        settledOverlayRevision >= target.revision && (settledOverlayLutRevision ?? -1) >= target.lutRevision)
      .map(([segmentId]) => segmentId);
    if (settledIds.length === 0) return;
    clearOptimisticSegments(settledIds);
  }, [clearOptimisticSegments, optimisticOverlayRevisionTargets, settledOverlayRevision, settledOverlayLutRevision]);

  return {
    applyLabelOverrides,
    applyOptimisticLabel,
    rollbackOptimisticLabel,
    stageOptimisticSegments,
    clearOptimisticSegments,
    stageOptimisticRevisionTargets,
    getOptimisticTargetRevision,
    optimisticConfirmed,
    optimisticExcluded,
    optimisticTransientOverlays,
  };
}
