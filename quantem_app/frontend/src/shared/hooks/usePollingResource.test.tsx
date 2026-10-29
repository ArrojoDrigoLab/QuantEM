import { act, renderHook } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { usePollingResource } from "./usePollingResource";

type Progress = { id: string; status: string };
const terminal = (value: Progress) => value.status === "SUCCESS";
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}
afterEach(() => vi.useRealTimers());

it("backs off repeated connection failures and resets on recovery", async () => {
  vi.useFakeTimers();
  const load = vi.fn<() => Promise<Progress>>().mockRejectedValue(new Error("Offline"));
  const { result } = renderHook(() => usePollingResource("job", load, terminal, 100));
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(load).toHaveBeenCalledTimes(2);
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(load).toHaveBeenCalledTimes(2);
  load.mockResolvedValue({ id: "job", status: "RUNNING" });
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(result.current.error).toBeNull();
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(load).toHaveBeenCalledTimes(4);
});

it("stops automatic retries for a removed record but allows an explicit retry", async () => {
  vi.useFakeTimers();
  const load = vi.fn<() => Promise<Progress>>().mockRejectedValue(Object.assign(new Error("Removed"), { status: 404 }));
  const { result } = renderHook(() => usePollingResource("job", load, terminal, 100));
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(load).toHaveBeenCalledTimes(1);
  expect(result.current.retrying).toBe(false);
  load.mockResolvedValue({ id: "job", status: "SUCCESS" });
  await act(async () => { await result.current.refresh(); });
  expect(result.current.settled).toBe(true);
});

it("serializes slow polls and rejects an old job's terminal response after switching jobs", async () => {
  vi.useFakeTimers();
  const old = deferred<Progress>();
  const load = vi.fn((id: string) => id === "old" ? old.promise : Promise.resolve({ id, status: "RETRY" }));
  const { result, rerender } = renderHook(({ id }) => usePollingResource(id, load, terminal, 100), { initialProps: { id: "old" } });
  await act(async () => { await vi.advanceTimersByTimeAsync(500); });
  expect(load).toHaveBeenCalledTimes(1);
  rerender({ id: "new" });
  await act(async () => {});
  expect(result.current.data).toEqual({ id: "new", status: "RETRY" });
  await act(async () => { old.resolve({ id: "old", status: "SUCCESS" }); });
  expect(result.current.settled).toBe(false);
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(load).toHaveBeenLastCalledWith("new");
  expect(load).toHaveBeenCalledTimes(3);
});

it("retains progress through a connection error, retries, and stops scheduled polls on completion", async () => {
  vi.useFakeTimers();
  const load = vi.fn<() => Promise<Progress>>()
    .mockResolvedValueOnce({ id: "job", status: "RUNNING" })
    .mockRejectedValueOnce(new Error("Offline"))
    .mockResolvedValue({ id: "job", status: "SUCCESS" });
  const { result } = renderHook(() => usePollingResource("job", load, terminal, 100));
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(100); });
  expect(result.current.error?.message).toBe("Offline");
  expect(result.current.data?.status).toBe("RUNNING");
  await act(async () => { window.dispatchEvent(new Event("online")); });
  expect(result.current.error).toBeNull();
  expect(result.current.settled).toBe(true);
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(load).toHaveBeenCalledTimes(3);
});

it("ignores an unresolved request after unmount and coalesces manual refreshes", async () => {
  const pending = deferred<Progress>();
  const load = vi.fn(() => pending.promise);
  const { result, unmount } = renderHook(() => usePollingResource("job", load, terminal));
  await act(async () => {});
  const first = result.current.refresh();
  const second = result.current.refresh();
  expect(first).toBe(second);
  unmount();
  pending.resolve({ id: "job", status: "RUNNING" });
  await first;
  expect(load).toHaveBeenCalledTimes(1);
});
