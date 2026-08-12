import {
  useCallback,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  RestartGuardContext,
  type RestartBlocker,
  type RestartGuardValue,
} from "@/features/update/restartGuardContext";

export function RestartGuardProvider({ children }: { children: ReactNode }) {
  const [blockers, setBlockers] = useState<RestartBlocker[]>([]);
  const setBlocker = useCallback((id: string, message: string | null) => {
    setBlockers((current) => {
      const existing = current.find((blocker) => blocker.id === id);
      if ((existing?.message ?? null) === message) {
        return current;
      }
      const withoutCurrent = current.filter((blocker) => blocker.id !== id);
      return message ? [...withoutCurrent, { id, message }] : withoutCurrent;
    });
  }, []);
  const value = useMemo<RestartGuardValue>(
    () => ({
      blockers,
      setBlocker,
    }),
    [blockers, setBlocker]
  );
  return <RestartGuardContext.Provider value={value}>{children}</RestartGuardContext.Provider>;
}
