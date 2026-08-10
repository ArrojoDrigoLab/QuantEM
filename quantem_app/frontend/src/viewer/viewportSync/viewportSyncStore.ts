import type { ViewportState } from "@/viewer/types";

export type ViewportAction =
  | {
      type: "fitToBounds";
      x: number;
      y: number;
      width: number;
      height: number;
      padding?: number;
    }
  | { type: "centerOnPoint"; x: number; y: number; keepZoom?: boolean; zoom?: number }
  | { type: "setZoom"; zoom: number; centerX?: number; centerY?: number }
  | { type: "panTo"; centerX: number; centerY: number; keepZoom?: boolean; zoom?: number }
  | { type: "setViewport"; viewport: ViewportState };

export type ViewportActionResolver = (
  action: ViewportAction,
  currentViewport: ViewportState | null
) => ViewportState | null;

type ViewportListener = (state: ViewportState | null) => void;

function isViewportClose(a: ViewportState | null, b: ViewportState | null, eps = 1e-5) {
  if (!a || !b) return a === b;
  return (
    Math.abs(a.centerX - b.centerX) < eps &&
    Math.abs(a.centerY - b.centerY) < eps &&
    Math.abs(a.zoom - b.zoom) < eps
  );
}

class ViewportSyncGroup {
  private viewport: ViewportState | null = null;
  private listeners = new Set<ViewportListener>();
  private resolver: ViewportActionResolver | null = null;
  private pendingState: ViewportState | null = null;
  private rafId: number | null = null;

  getState() {
    return this.viewport;
  }

  setResolver(resolver: ViewportActionResolver | null) {
    this.resolver = resolver;
  }

  subscribe(listener: ViewportListener) {
    this.listeners.add(listener);
    listener(this.viewport);
    return () => {
      this.listeners.delete(listener);
    };
  }

  publishFromViewer(_viewerId: string, state: ViewportState) {
    this.scheduleUpdate(state);
  }

  setViewport(state: ViewportState) {
    this.scheduleUpdate(state);
  }

  applyAction(action: ViewportAction) {
    if (action.type === "setViewport") {
      this.scheduleUpdate(action.viewport);
      return;
    }
    if (!this.resolver) return;
    const next = this.resolver(action, this.viewport);
    if (next) {
      this.scheduleUpdate(next);
    }
  }

  private scheduleUpdate(next: ViewportState) {
    if (isViewportClose(next, this.viewport)) return;
    this.pendingState = next;
    if (this.rafId !== null) return;
    this.rafId = window.requestAnimationFrame(() => {
      this.rafId = null;
      const pending = this.pendingState;
      this.pendingState = null;
      if (!pending || isViewportClose(pending, this.viewport)) return;
      this.viewport = pending;
      this.listeners.forEach((listener) => listener(this.viewport));
    });
  }
}

const groups = new Map<string, ViewportSyncGroup>();

export function getViewportSyncGroup(groupId: string) {
  let group = groups.get(groupId);
  if (!group) {
    group = new ViewportSyncGroup();
    groups.set(groupId, group);
  }
  return group;
}

