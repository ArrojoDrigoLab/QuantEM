/**
 * The drag, and the threshold that decides whether it was one.
 *
 * The bug being guarded against: the tool this was ported from measured the
 * dead zone as 2% of the viewport width, so on a narrow viewport a deliberate
 * small box was swallowed as a click and nothing happened. The threshold here
 * is a fixed 8 screen px, and a box smaller than the old adaptive dead zone
 * must still be sent.
 */

import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SamBoxResponse } from "./types";
import { useSamBoxTool } from "./useSamBoxTool";

const promptSamBox = vi.fn();
const getSamModelStatus = vi.fn();
const startSamModelDownload = vi.fn();

vi.mock("./api", () => ({
  promptSamBox: (...args: unknown[]) => promptSamBox(...args),
  getSamModelStatus: (...args: unknown[]) => getSamModelStatus(...args),
  startSamModelDownload: (...args: unknown[]) => startSamModelDownload(...args),
}));

const READY = {
  model: "micro-SAM EM organelles (ViT-B)",
  installed: true,
  size_bytes: 375_023_499,
  download: {
    status: "SUCCESS" as const,
    bytes_done: 0,
    bytes_total: 0,
    error: "",
    percent: null,
  },
};

const RESPONSE: SamBoxResponse = {
  created: 1,
  updated: 0,
  deleted: 0,
  confirmed_ids: ["abc"],
  overlay: {
    desired_revision: 1,
    applied_revision: 1,
    sync_applied: true,
    rebuild_mode: "sync_partial",
  },
  object: { geometry_coords: [], score: 0.9, area: 100 },
  other_candidates: [],
  timing: { cache_hit: false, encode_ms: 500, decode_ms: 20, device: "cuda" },
};

function setup(overrides: Partial<Parameters<typeof useSamBoxTool>[0]> = {}) {
  const onObjectCreated = vi.fn();
  const onError = vi.fn();
  const hook = renderHook(() =>
    useSamBoxTool({
      segmentationId: "seg-1",
      available: true,
      onObjectCreated,
      onError,
      ...overrides,
    })
  );
  return { hook, onObjectCreated, onError };
}

function pendingOverlays(
  result: { current: ReturnType<typeof useSamBoxTool> }
) {
  return result.current.overlays.filter((overlay) =>
    overlay.id.startsWith("sam-box-pending")
  );
}

/** Press at the origin, move to (dx, dy) in both spaces, release. */
async function drag(
  result: { current: ReturnType<typeof useSamBoxTool> },
  dx: number,
  dy: number
) {
  await act(async () => {
    result.current.handleImagePress({ x: 0, y: 0 }, { x: 0, y: 0 });
  });
  await act(async () => {
    result.current.handleImageDrag({ x: dx, y: dy }, { x: dx, y: dy });
  });
  await act(async () => {
    result.current.handleImageRelease({ x: dx, y: dy }, { x: dx, y: dy });
  });
}

describe("useSamBoxTool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getSamModelStatus.mockResolvedValue(READY);
    promptSamBox.mockResolvedValue(RESPONSE);
  });

  it("starts switched off", () => {
    const { hook } = setup();
    expect(hook.result.current.isActive).toBe(false);
  });

  it("sends the box once the drag passes the threshold", async () => {
    const { hook, onObjectCreated } = setup();
    act(() => hook.result.current.setActive(true));

    await drag(hook.result, 40, 30);

    await waitFor(() => expect(promptSamBox).toHaveBeenCalledTimes(1));
    expect(promptSamBox).toHaveBeenCalledWith("seg-1", {
      x0: 0,
      y0: 0,
      x1: 40,
      y1: 30,
    });
    await waitFor(() => expect(onObjectCreated).toHaveBeenCalledWith(RESPONSE));
  });

  it("treats a movement under the threshold as a click, not a box", async () => {
    const { hook } = setup();
    act(() => hook.result.current.setActive(true));

    await drag(hook.result, 3, 2);

    expect(promptSamBox).not.toHaveBeenCalled();
  });

  it("sends a small but deliberate box that a 2%-of-viewport dead zone would eat", async () => {
    const { hook } = setup();
    act(() => hook.result.current.setActive(true));

    // 14 px: under 2% of a 1000 px viewport (20 px), over the fixed 8 px.
    await drag(hook.result, 14, 14);

    await waitFor(() => expect(promptSamBox).toHaveBeenCalledTimes(1));
  });

  it("normalises a box dragged up and to the left", async () => {
    const { hook } = setup();
    act(() => hook.result.current.setActive(true));

    await act(async () => {
      hook.result.current.handleImagePress({ x: 100, y: 100 }, { x: 100, y: 100 });
    });
    await act(async () => {
      hook.result.current.handleImageRelease({ x: 40, y: 30 }, { x: 40, y: 30 });
    });

    await waitFor(() =>
      expect(promptSamBox).toHaveBeenCalledWith("seg-1", {
        x0: 40,
        y0: 30,
        x1: 100,
        y1: 100,
      })
    );
  });

  it("shows a pending rectangle while the request is in flight", async () => {
    let settle: (value: unknown) => void = () => {};
    promptSamBox.mockReturnValue(
      new Promise((resolve) => {
        settle = resolve;
      })
    );
    const { hook } = setup();
    act(() => hook.result.current.setActive(true));

    await drag(hook.result, 50, 50);

    await waitFor(() => expect(hook.result.current.isSubmitting).toBe(true));
    expect(pendingOverlays(hook.result)).toHaveLength(1);

    await act(async () => {
      settle(RESPONSE);
    });
    await waitFor(() => expect(hook.result.current.isSubmitting).toBe(false));
    expect(hook.result.current.overlays).toHaveLength(0);
  });

  it("keeps the box visible until the returned object has been staged", async () => {
    let finishStaging: () => void = () => {};
    const staging = new Promise<void>((resolve) => {
      finishStaging = resolve;
    });
    const onObjectCreated = vi.fn(() => staging);
    const { hook } = setup({ onObjectCreated });
    act(() => hook.result.current.setActive(true));

    await drag(hook.result, 50, 50);

    await waitFor(() => expect(onObjectCreated).toHaveBeenCalledWith(RESPONSE));
    expect(pendingOverlays(hook.result)).toHaveLength(1);

    await act(async () => finishStaging());
    await waitFor(() => expect(hook.result.current.isSubmitting).toBe(false));
    expect(hook.result.current.overlays).toHaveLength(0);
  });

  it("keeps every rapid box until its own request and staging finish", async () => {
    const requests: Array<{
      resolve: (response: typeof RESPONSE) => void;
    }> = [];
    promptSamBox.mockImplementation(
      () =>
        new Promise((resolve) => {
          requests.push({ resolve });
        })
    );
    const stagingResolvers = new Map<string, () => void>();
    const onObjectCreated = vi.fn(
      (response: typeof RESPONSE) =>
        new Promise<void>((resolve) => {
          stagingResolvers.set(response.confirmed_ids[0], resolve);
        })
    );
    const { hook } = setup({ onObjectCreated });
    act(() => hook.result.current.setActive(true));

    for (let index = 1; index <= 5; index += 1) {
      await drag(hook.result, 20 + index, 20 + index);
    }

    await waitFor(() => expect(requests).toHaveLength(5));
    expect(pendingOverlays(hook.result)).toHaveLength(5);
    expect(hook.result.current.pendingCount).toBe(5);
    expect(
      new Set(pendingOverlays(hook.result).map(({ id }) => id)).size
    ).toBe(5);

    const secondResponse = {
      ...RESPONSE,
      confirmed_ids: ["second"],
    };
    await act(async () => requests[1].resolve(secondResponse));
    await waitFor(() =>
      expect(onObjectCreated).toHaveBeenCalledWith(secondResponse)
    );

    // Its box stays until its own mask is staged, and the other four are not
    // affected by this request finishing out of order.
    expect(pendingOverlays(hook.result)).toHaveLength(5);
    await act(async () => stagingResolvers.get("second")?.());
    await waitFor(() => expect(pendingOverlays(hook.result)).toHaveLength(4));
    expect(hook.result.current.isSubmitting).toBe(true);
    expect(hook.result.current.pendingCount).toBe(4);

    const firstResponse = {
      ...RESPONSE,
      confirmed_ids: ["first"],
    };
    await act(async () => requests[0].resolve(firstResponse));
    await waitFor(() =>
      expect(onObjectCreated).toHaveBeenCalledWith(firstResponse)
    );
    expect(pendingOverlays(hook.result)).toHaveLength(4);
    await act(async () => stagingResolvers.get("first")?.());
    await waitFor(() => expect(pendingOverlays(hook.result)).toHaveLength(3));

    for (const index of [4, 2, 3]) {
      const response = {
        ...RESPONSE,
        confirmed_ids: [`request-${index}`],
      };
      await act(async () => requests[index].resolve(response));
      await waitFor(() => expect(onObjectCreated).toHaveBeenCalledWith(response));
      await act(async () => stagingResolvers.get(`request-${index}`)?.());
    }
    await waitFor(() => expect(hook.result.current.isSubmitting).toBe(false));
    expect(pendingOverlays(hook.result)).toHaveLength(0);
  });

  it("draws a live rectangle during the drag", async () => {
    const { hook } = setup();
    act(() => hook.result.current.setActive(true));

    await act(async () => {
      hook.result.current.handleImagePress({ x: 0, y: 0 }, { x: 0, y: 0 });
    });
    await act(async () => {
      hook.result.current.handleImageDrag({ x: 60, y: 60 }, { x: 60, y: 60 });
    });

    expect(hook.result.current.overlays.some((o) => o.id === "sam-box-live")).toBe(
      true
    );
  });

  it("reports a failure through onError and clears the pending box", async () => {
    promptSamBox.mockRejectedValue(new Error("no weights"));
    const { hook, onError } = setup();
    act(() => hook.result.current.setActive(true));

    await drag(hook.result, 50, 50);

    await waitFor(() => expect(onError).toHaveBeenCalledTimes(1));
    expect(hook.result.current.overlays).toHaveLength(0);
    expect(hook.result.current.isSubmitting).toBe(false);
  });

  it("does nothing while the tool is off", async () => {
    const { hook } = setup();
    await drag(hook.result, 50, 50);
    expect(promptSamBox).not.toHaveBeenCalled();
  });

  it("switches itself off when the screen says it is unavailable", async () => {
    const { hook } = setup();
    act(() => hook.result.current.setActive(true));
    expect(hook.result.current.isActive).toBe(true);

    hook.rerender();
    const off = renderHook(() =>
      useSamBoxTool({
        segmentationId: "seg-1",
        available: false,
        onObjectCreated: vi.fn(),
        onError: vi.fn(),
      })
    );
    expect(off.result.current.isActive).toBe(false);
  });
});
