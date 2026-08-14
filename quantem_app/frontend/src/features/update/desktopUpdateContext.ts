import { createContext } from "react";
import type { DesktopUpdate } from "@/features/update/desktopUpdater";

export type DesktopUpdatePhase =
  | "idle"
  | "checking"
  | "up-to-date"
  | "available"
  | "downloading"
  | "downloaded"
  | "waiting"
  | "blocked"
  | "applying"
  | "error";

export interface DownloadProgress {
  downloaded: number;
  total: number | null;
}

export interface DesktopUpdateValue {
  enabled: boolean;
  phase: DesktopUpdatePhase;
  update: DesktopUpdate | null;
  progress: DownloadProgress;
  openJobs: number;
  restartBlocker: string | null;
  error: string | null;
  checkNow: () => Promise<void>;
  upgradeNow: () => Promise<void>;
}

export const DesktopUpdateContext = createContext<DesktopUpdateValue | null>(null);
