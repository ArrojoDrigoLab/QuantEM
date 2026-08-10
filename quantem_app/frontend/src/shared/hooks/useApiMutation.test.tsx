import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useApiMutation } from "@/shared/hooks/useApiMutation";

describe("useApiMutation", () => {
  it("calls onSuccess and returns the mutation result", async () => {
    const mutateFn = vi.fn(async (payload: { name: string }) => ({ id: payload.name }));
    const onSuccess = vi.fn();

    const { result } = renderHook(() =>
      useApiMutation<{ name: string }, { id: string }>(mutateFn, { onSuccess })
    );

    let response: { id: string } | undefined;
    await act(async () => {
      response = await result.current.mutate({ name: "A" });
    });

    expect(mutateFn).toHaveBeenCalledWith({ name: "A" });
    expect(onSuccess).toHaveBeenCalledWith({ id: "A" });
    expect(response).toEqual({ id: "A" });
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("sets error and calls onError when mutation fails", async () => {
    const mutateFn = vi.fn(async (payload: { name: string }) => {
      void payload;
      throw new Error("boom");
    });
    const onError = vi.fn();

    const { result } = renderHook(() =>
      useApiMutation<{ name: string }, { id: string }>(mutateFn, { onError })
    );

    let response: { id: string } | undefined;
    await act(async () => {
      response = await result.current.mutate({ name: "A" });
    });

    expect(response).toBeUndefined();
    expect(onError).toHaveBeenCalled();
    expect(result.current.error?.message).toBe("boom");
    expect(result.current.loading).toBe(false);

    act(() => {
      result.current.reset();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });
});
