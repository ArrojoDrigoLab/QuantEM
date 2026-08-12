import { useContext, useEffect, useId } from "react";
import { RestartGuardContext } from "@/features/update/restartGuardContext";

export function useRestartGuard() {
  const context = useContext(RestartGuardContext);
  if (!context) {
    throw new Error("useRestartGuard must be used inside RestartGuardProvider.");
  }
  return context;
}

/** Register a local-only draft that must not be lost to an application restart. */
export function useRestartBlocker(active: boolean, message: string): void {
  const { setBlocker } = useRestartGuard();
  const id = useId();
  useEffect(() => {
    setBlocker(id, active ? message : null);
    return () => setBlocker(id, null);
  }, [active, id, message, setBlocker]);
}
