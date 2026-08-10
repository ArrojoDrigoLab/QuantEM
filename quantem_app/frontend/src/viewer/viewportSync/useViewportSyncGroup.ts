import { useCallback, useEffect, useMemo, useState } from "react";
import type { ViewportState } from "@/viewer/types";
import {
  getViewportSyncGroup,
  type ViewportAction,
  type ViewportActionResolver,
} from "@/viewer/viewportSync/viewportSyncStore";

interface UseViewportSyncGroupOptions {
  resolveAction?: ViewportActionResolver | null;
}

export function useViewportSyncGroup(groupId: string, options: UseViewportSyncGroupOptions = {}) {
  const group = useMemo(() => getViewportSyncGroup(groupId), [groupId]);
  const [viewport, setViewportState] = useState<ViewportState | null>(group.getState());

  useEffect(() => group.subscribe(setViewportState), [group]);

  useEffect(() => {
    group.setResolver(options.resolveAction ?? null);
  }, [group, options.resolveAction]);

  const publishFromViewer = useCallback(
    (viewerId: string, state: ViewportState) => {
      group.publishFromViewer(viewerId, state);
    },
    [group]
  );

  const setViewport = useCallback(
    (state: ViewportState) => {
      group.setViewport(state);
    },
    [group]
  );

  const applyAction = useCallback(
    (action: ViewportAction) => {
      group.applyAction(action);
    },
    [group]
  );

  return {
    viewport,
    publishFromViewer,
    setViewport,
    applyAction,
  };
}

