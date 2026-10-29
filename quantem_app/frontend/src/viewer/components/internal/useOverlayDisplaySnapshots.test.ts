import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Layer } from "@deck.gl/core";
import type { ViewerIdMapOverlaySpec } from "@/viewer/types";
import type { IdMapLoaderData } from "./useViewerIdMapLoader";
import type { VivLoaderData } from "./vivUtils";
import { useOverlayDisplaySnapshots } from "./useOverlayDisplaySnapshots";

const spec = (revision: number): ViewerIdMapOverlaySpec => ({
  id: "confirmed",
  ngffUrl: `/overlay?rev=${revision}-${revision}`,
  revision,
  lut: new Uint8Array([0, 0, 0, 0, revision, 255, 0, 255]),
  maxLabel: 1,
  lutRevision: revision,
  pickMap: new Map([[1, `object-${revision}`]]),
  fillOpacity: 0.3,
  borderOpacity: 1,
  showBorders: true,
  highlightedSegmentId: "object-2",
});
const loaded = (value: ViewerIdMapOverlaySpec): Record<string, IdMapLoaderData> => ({
  [value.id]: {
    ngffUrl: value.ngffUrl,
    revision: value.revision!,
    labelsData: [{}] as VivLoaderData,
    borderData: [{}] as VivLoaderData,
  },
});
const ready = (id: string) => [{ id: `idmap-overlay-${id}`, isLoaded: true }] as Layer[];

describe("overlay display handoff", () => {
  it("retains the old pixels, palette and picking until replacement viewport tiles load", () => {
    const first = spec(1);
    const second = spec(2);
    const { result, rerender } = renderHook(({ desired, data }) =>
      useOverlayDisplaySnapshots("image", desired, data, {}),
      { initialProps: { desired: [first], data: loaded(first) } },
    );
    act(() => result.current.acceptLoaded(ready(first.id)));
    rerender({ desired: [{ ...second, pending: true }], data: {} });
    expect(result.current.visible[0].spec).toBe(first);
    expect(result.current.specs[0].pickMap).toBe(first.pickMap);
    rerender({
      desired: [{ ...second, pending: true, fillOpacity: 0, showBorders: false }],
      data: {},
    });
    expect(result.current.specs[0]).toMatchObject({ fillOpacity: 0, showBorders: false });
    expect(result.current.specs[0].lut).toBe(first.lut);
    expect(result.current.specs[0].pickMap).toBe(first.pickMap);
    rerender({ desired: [second], data: loaded(second) });
    expect(result.current.visible[0].spec).toBe(first);
    const staging = result.current.specs[1];
    expect(staging.fillOpacity).toBe(0);
    expect(staging.highlightedSegmentId).toBeNull();
    expect(staging.pickMap).toBeUndefined();
    act(() => result.current.acceptLoaded([]));
    expect(result.current.visible[0].spec).toBe(first);
    act(() => result.current.acceptLoaded(ready(staging.id)));
    expect(result.current.visible[0].spec).toBe(second);
    expect(result.current.specs).toHaveLength(1);
    expect(result.current.specs[0].id).toBe(staging.id);
    expect(result.current.specs[0].pickMap).toBe(second.pickMap);
  });

  it("keeps the last good generation on failure and clears it on hide or source changes", () => {
    const first = spec(1);
    const { result, rerender } = renderHook(({ desired, data, errors }) =>
      useOverlayDisplaySnapshots("image", desired, data, errors),
      {
        initialProps: {
          desired: [first],
          data: loaded(first),
          errors: {} as Record<string, string>,
        },
      },
    );
    act(() => result.current.acceptLoaded(ready(first.id)));
    const second = spec(2);
    rerender({
      desired: [second],
      data: loaded(second),
      errors: { [second.ngffUrl]: "Tile failed" },
    });
    expect(result.current.visible[0].spec).toBe(first);
    rerender({
      desired: [{ ...second, ngffUrl: "/another-organelle?rev=2", pending: true }],
      data: {},
      errors: {},
    });
    expect(result.current.specs).toHaveLength(0);
    rerender({ desired: [], data: {}, errors: {} });
    expect(result.current.specs).toHaveLength(0);
  });
});
