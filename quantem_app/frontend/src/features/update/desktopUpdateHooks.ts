import { useContext } from "react";
import { DesktopUpdateContext } from "@/features/update/desktopUpdateContext";

export function useDesktopUpdate() {
  const context = useContext(DesktopUpdateContext);
  if (!context) {
    throw new Error("useDesktopUpdate must be used inside DesktopUpdateProvider.");
  }
  return context;
}
