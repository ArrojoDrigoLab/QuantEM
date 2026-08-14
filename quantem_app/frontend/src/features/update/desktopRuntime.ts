/** Detect the bundled Tauri webview without invoking a native command in web builds. */
export function isDesktopTauriBuild(): boolean {
  if (typeof globalThis === "undefined") return false;

  const tauriGlobal = globalThis as typeof globalThis & {
    isTauri?: boolean;
    __TAURI_INTERNALS__?: unknown;
  };

  // Tauri 2 exposes `isTauri` even when `app.withGlobalTauri` is disabled.
  // Keep the internals check as a compatibility fallback for older shells.
  return tauriGlobal.isTauri === true || tauriGlobal.__TAURI_INTERNALS__ != null;
}
