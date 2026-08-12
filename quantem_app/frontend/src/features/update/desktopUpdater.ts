import { isDesktopTauriBuild } from "@/features/update/desktopRuntime";

export type UpdateDownloadEvent =
  | { event: "Started"; data: { contentLength?: number } }
  | { event: "Progress"; data: { chunkLength: number } }
  | { event: "Finished" };

export interface DesktopUpdate {
  version: string;
  body?: string | null;
  date?: string | null;
  download: (callback: (event: UpdateDownloadEvent) => void) => Promise<void>;
  install: () => Promise<void>;
}

interface DesktopUpdaterApi {
  check: () => Promise<DesktopUpdate | null>;
  relaunch: () => Promise<void>;
}

/**
 * Load Tauri-only APIs lazily. The web/pip app remains a normal browser app
 * and must neither make release-network requests nor attempt native commands.
 */
export async function getDesktopUpdater(): Promise<DesktopUpdaterApi | null> {
  if (!isDesktopTauriBuild()) {
    return null;
  }
  const [updater, process] = await Promise.all([
    import("@tauri-apps/plugin-updater"),
    import("@tauri-apps/plugin-process"),
  ]);
  return {
    check: updater.check as DesktopUpdaterApi["check"],
    relaunch: process.relaunch,
  };
}
