import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useApiQuery } from "@/shared/hooks/useApiQuery";

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("useApiQuery", () => {
  it("settles initial loading when reconnect supersedes the first request", async () => {
    const first = deferred<string>();
    const fetcher = vi.fn().mockReturnValueOnce(first.promise).mockResolvedValue("reconnected");
    const { result } = renderHook(() => useApiQuery(fetcher));
    await act(async () => { window.dispatchEvent(new Event("online")); });
    expect(result.current.data).toBe("reconnected");
    expect(result.current.loading).toBe(false);
    expect(result.current.refetching).toBe(false);
    await act(async () => { first.resolve("obsolete"); });
    expect(result.current.data).toBe("reconnected");
  });

  it("clears refetching when a new context supersedes a background request", async () => {
    const old = deferred<string>();
    const fetcher = vi.fn().mockResolvedValueOnce("first").mockReturnValueOnce(old.promise).mockResolvedValue("second");
    const { result, rerender } = renderHook(({ id }) => useApiQuery(fetcher, [id]), { initialProps: { id: "a" } });
    await waitFor(() => expect(result.current.data).toBe("first"));
    act(() => { void result.current.refetch(); });
    rerender({ id: "b" });
    await waitFor(() => expect(result.current.data).toBe("second"));
    expect(result.current.refetching).toBe(false);
    await act(async () => { old.resolve("obsolete"); });
  });

  it("loads data successfully", async () => {
    const fetcher = vi.fn().mockResolvedValue("ok");

    const { result } = renderHook(() => useApiQuery(fetcher, []));

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toBe("ok");
    expect(result.current.error).toBeNull();
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("ignores stale responses when deps change quickly", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const fetcher = vi.fn((id: string) => (id === "a" ? first.promise : second.promise));

    const { result, rerender } = renderHook(
      ({ id }) => useApiQuery(() => fetcher(id), [id]),
      { initialProps: { id: "a" } }
    );

    act(() => {
      rerender({ id: "b" });
    });

    await act(async () => {
      first.resolve("stale");
      await Promise.resolve();
    });

    expect(result.current.data).toBeNull();

    await act(async () => {
      second.resolve("fresh");
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.data).toBe("fresh");
  });

  it("tracks refetching separately from initial loading", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const fetcher = vi.fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);

    const { result } = renderHook(() => useApiQuery(fetcher, []));

    await act(async () => {
      first.resolve("initial");
    });
    await waitFor(() => {
      expect(result.current.loading).toBe(false);
      expect(result.current.data).toBe("initial");
    });

    act(() => {
      void result.current.refetch();
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.refetching).toBe(true);

    await act(async () => {
      second.resolve("updated");
    });

    await waitFor(() => {
      expect(result.current.refetching).toBe(false);
    });
    expect(result.current.data).toBe("updated");
  });

  it("preserves existing data when a refetch fails", async () => {
    const first = deferred<string>();
    const second = deferred<string>();
    const fetcher = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const { result } = renderHook(() => useApiQuery(fetcher, []));

      await act(async () => {
        first.resolve("initial");
      });
      await waitFor(() => {
        expect(result.current.loading).toBe(false);
        expect(result.current.data).toBe("initial");
      });

      act(() => {
        void result.current.refetch();
      });

      await act(async () => {
        second.reject(new Error("refetch failed"));
        await Promise.resolve();
      });

      await waitFor(() => {
        expect(result.current.refetching).toBe(false);
      });

      expect(result.current.data).toBe("initial");
      expect(result.current.error?.message).toBe("refetch failed");
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });

  it("keeps a refetch error visible until a later request succeeds", async () => {
    const retry = deferred<string>();
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce("initial")
      .mockRejectedValueOnce(new Error("connection lost"))
      .mockReturnValueOnce(retry.promise);
    const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const { result } = renderHook(() => useApiQuery(fetcher, []));
      await waitFor(() => expect(result.current.data).toBe("initial"));

      await act(async () => {
        await result.current.refetch();
      });
      expect(result.current.error?.message).toBe("connection lost");

      act(() => {
        void result.current.refetch();
      });
      expect(result.current.refetching).toBe(true);
      expect(result.current.error?.message).toBe("connection lost");

      await act(async () => {
        retry.resolve("reconnected");
      });
      await waitFor(() => expect(result.current.refetching).toBe(false));
      expect(result.current.data).toBe("reconnected");
      expect(result.current.error).toBeNull();
    } finally {
      consoleErrorSpy.mockRestore();
    }
  });
});
