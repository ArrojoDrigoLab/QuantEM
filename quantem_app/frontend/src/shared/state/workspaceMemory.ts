/** Small session-scoped preferences; never annotations or server records. */
export function readWorkspaceMemory<T>(key: string, fallback: T): T {
  try {
    const value = sessionStorage.getItem("quantem.workspace." + key);
    return value ? JSON.parse(value) as T : fallback;
  } catch { return fallback; }
}

export function rememberWorkspace(key: string, value: unknown) {
  try { sessionStorage.setItem("quantem.workspace." + key, JSON.stringify(value)); }
  catch { /* Navigation remains available when storage is unavailable. */ }
}
