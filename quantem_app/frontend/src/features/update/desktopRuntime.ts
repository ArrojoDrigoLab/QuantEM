/** Detect the bundled Tauri webview without importing a native module in web builds. */
export function isDesktopTauriBuild(): boolean {
  return (
    typeof window !== "undefined" &&
    "__TAURI_INTERNALS__" in (window as Window & { __TAURI_INTERNALS__?: unknown })
  );
}
