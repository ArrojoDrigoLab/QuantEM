import { act, renderHook } from "@testing-library/react";
import { expect, it } from "vitest";
import { useWorkspaceState } from "./useWorkspaceState";

it("restores each context and ignores setters retained by a previous context", () => {
  const { result, rerender, unmount } = renderHook(({ id }) => useWorkspaceState(id, "default"), { initialProps: { id: "a" } });
  expect(sessionStorage.getItem("quantem.workspace.a")).toBeNull();
  const setA = result.current[1];
  act(() => { setA("choice a"); });
  rerender({ id: "b" });
  act(() => { result.current[1]("choice b"); });
  act(() => { setA("late response"); });
  expect(result.current[0]).toBe("choice b");
  rerender({ id: "a" });
  expect(result.current[0]).toBe("choice a");
  unmount();
  const reopened = renderHook(() => useWorkspaceState("b", "default"));
  expect(reopened.result.current[0]).toBe("choice b");
});
