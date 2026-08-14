import { useCallback, useEffect, useRef, useState } from "react";

interface OptimisticDeletionState {
  ids: ReadonlySet<string>;
  visualRevision: number;
}

const EMPTY_STATE: OptimisticDeletionState = {
  ids: new Set<string>(),
  visualRevision: 0,
};

/**
 * Objects hidden while their hard-delete request is being persisted.
 *
 * Keeping this separate from label overrides is deliberate: deletion is not a
 * fourth label state. The UUID is removed from every client-side presentation
 * immediately and stays hidden after a successful request while the rebuilt
 * overlay catches up. Only a failed request puts it back.
 */
export function useOptimisticDeletionState(currentSegmentationId: string | null) {
  const [state, setState] = useState<OptimisticDeletionState>(EMPTY_STATE);
  const hiddenIdsRef = useRef<ReadonlySet<string>>(EMPTY_STATE.ids);

  useEffect(() => {
    const ids = new Set<string>();
    hiddenIdsRef.current = ids;
    setState((previous) => ({
      ids,
      visualRevision: previous.visualRevision + 1,
    }));
  }, [currentSegmentationId]);

  const hideSegment = useCallback((segmentId: string): boolean => {
    if (hiddenIdsRef.current.has(segmentId)) return false;
    const ids = new Set(hiddenIdsRef.current);
    ids.add(segmentId);
    hiddenIdsRef.current = ids;
    setState((previous) => ({
      ids,
      visualRevision: previous.visualRevision + 1,
    }));
    return true;
  }, []);

  const rollbackSegment = useCallback((segmentId: string) => {
    if (!hiddenIdsRef.current.has(segmentId)) return;
    const ids = new Set(hiddenIdsRef.current);
    ids.delete(segmentId);
    hiddenIdsRef.current = ids;
    setState((previous) => ({
      ids,
      visualRevision: previous.visualRevision + 1,
    }));
  }, []);

  return {
    hiddenSegmentIds: state.ids,
    visualRevision: state.visualRevision,
    hideSegment,
    rollbackSegment,
  };
}
