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

const RESPONSE = {
  created: 1,
  updated: 0,
  deleted: 0,
  confirmed_ids: ["abc"],
  overlay: null,
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
    expect(
      hook.result.current.overlays.some((o) => o.id === "sam-box-pending")
    ).toBe(true);

    await act(async () => {
      settle(RESPONSE);
    });
    await waitFor(() => expect(hook.result.current.isSubmitting).toBe(false));
    expect(hook.result.current.overlays).toHaveLength(0);
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
