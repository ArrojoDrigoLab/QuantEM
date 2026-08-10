import type { SegmentOverlay } from "@/viewer/types";
import type { OverlayScene } from "@/viewer/overlays/types";

type OverlayLayer = SegmentOverlay[] | null | undefined;

function flattenLayers(layers: OverlayLayer[]): SegmentOverlay[] {
  return layers.flatMap((layer) => layer ?? []);
}

export function composeOverlayScene({
  persistentLayers = [],
  transientLayers = [],
}: {
  persistentLayers?: OverlayLayer[];
  transientLayers?: OverlayLayer[];
}): OverlayScene {
  return {
    persistent: flattenLayers(persistentLayers),
    transient: flattenLayers(transientLayers),
  };
}

export function sceneToOverlayList(scene: OverlayScene): SegmentOverlay[] {
  return [...scene.persistent, ...scene.transient];
}
