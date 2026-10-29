import { useCallback, useEffect, useMemo, useState, type SetStateAction } from "react";
import { readWorkspaceMemory, rememberWorkspace } from "@/shared/state/workspaceMemory";

/** Session-scoped workflow choices only. Server records remain authoritative. */
export function useWorkspaceState<T>(key: string, initial: T | (() => T)) {
  const restored = useMemo(() => readWorkspaceMemory<T>(key,
    typeof initial === "function" ? (initial as () => T)() : initial),
    // Defaults apply when entering a context, not when its server data refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [key]);
  const [state, setState] = useState({ key, value: restored, dirty: false });
  if (state.key !== key) setState({ key, value: restored, dirty: false });
  const value = state.key === key ? state.value : restored;
  useEffect(() => {
    if (state.key === key && state.dirty) rememberWorkspace(key, state.value);
  }, [key, state]);
  const setValue = useCallback((update: SetStateAction<T>) => {
    setState((previous) => {
      // A callback retained by an old request must not replace the new context.
      if (previous.key !== key) return previous;
      const current = previous.value;
      const next = typeof update === "function" ? (update as (value: T) => T)(current) : update;
      if (Object.is(current, next) && previous.dirty) return previous;
      return { key, value: next, dirty: true };
    });
  }, [key]);
  return [value, setValue] as const;
}

export function useWorkspaceSet(key: string, initial: () => string[]) {
  const [values, setValues] = useWorkspaceState<string[]>(key, initial);
  const value = useMemo(() => new Set(values), [values]);
  const setValue = useCallback((update: SetStateAction<Set<string>>) => setValues((previous) =>
    [...(typeof update === "function" ? update(new Set(previous)) : update)]), [setValues]);
  return [value, setValue] as const;
}
