/**
 * Client-side file downloads for chart data.
 *
 * Every chart in the analysis and fine-tuning screens plots an array that the
 * user must also be able to take away, and some of those arrays (the
 * distance-band table, the threshold sweep) are not columns in any server-side
 * export. Rather than plot numbers nobody can extract, the chart and the
 * download read the same array in the same component.
 */

/** Quote a CSV field only when it needs it, and double any embedded quotes. */
export function toCsvField(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(headers: string[], rows: Array<Array<unknown>>): string {
  const lines = [headers.map(toCsvField).join(",")];
  for (const row of rows) {
    lines.push(row.map(toCsvField).join(","));
  }
  return `${lines.join("\n")}\n`;
}

interface SaveFilePickerType {
  description?: string;
  accept: Record<string, string[]>;
}

interface SaveFilePickerOptions {
  suggestedName?: string;
  types?: SaveFilePickerType[];
}

declare global {
  interface Window {
    /** Chromium/WebView2 File System Access API. */
    showSaveFilePicker?: (
      options?: SaveFilePickerOptions
    ) => Promise<FileSystemFileHandle>;
  }
}

export type SaveFileResult = "saved" | "cancelled" | "download-started";

function isTauriRuntime(): boolean {
  if (typeof globalThis === "undefined") return false;
  const tauriGlobal = globalThis as typeof globalThis & {
    isTauri?: boolean;
    __TAURI_INTERNALS__?: unknown;
  };
  return tauriGlobal.isTauri === true || tauriGlobal.__TAURI_INTERNALS__ != null;
}

async function invokeNativeSave(
  command: "save_text_file" | "save_url_file",
  request: Record<string, string>
): Promise<SaveFileResult> {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const saved = await invoke<boolean>(command, { request });
    return saved ? "saved" : "cancelled";
  } catch (cause) {
    // Rust command errors cross the IPC bridge as strings. Normalize them so
    // dialogs and alert handlers can show the actual filesystem/server reason.
    throw cause instanceof Error ? cause : new Error(String(cause));
  }
}

function extensionOf(filename: string): string | null {
  const basename = filename.split(/[\\/]/).at(-1) ?? filename;
  const dot = basename.lastIndexOf(".");
  if (dot <= 0 || dot === basename.length - 1) return null;
  const extension = basename.slice(dot);
  return /^\.[A-Za-z0-9_-]{1,15}$/.test(extension) ? extension : null;
}

function pickerOptions(
  filename: string,
  mimeType: string
): SaveFilePickerOptions {
  const extension = extensionOf(filename);
  const bareMimeType = mimeType.split(";", 1)[0].trim() || "application/octet-stream";
  return {
    suggestedName: filename,
    ...(extension
      ? {
          types: [
            {
              description: `${extension.slice(1).toUpperCase()} file`,
              accept: { [bareMimeType]: [extension] },
            },
          ],
        }
      : {}),
  };
}

function isCancelledPicker(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    error.name === "AbortError"
  );
}

async function pickedHandle(
  filename: string,
  mimeType: string
): Promise<FileSystemFileHandle | null | undefined> {
  if (typeof window === "undefined" || typeof window.showSaveFilePicker !== "function") {
    return undefined;
  }
  try {
    return await window.showSaveFilePicker(pickerOptions(filename, mimeType));
  } catch (error) {
    if (isCancelledPicker(error)) return null;
    throw error;
  }
}

function clickDownload(filename: string, href: string): void {
  if (typeof document === "undefined") {
    throw new Error("File saving is unavailable outside a browser window.");
  }
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.hidden = true;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

async function responseError(response: Response): Promise<Error> {
  let detail = "";
  try {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const body = (await response.json()) as { error?: unknown; detail?: unknown };
      detail = String(body.error ?? body.detail ?? "").trim();
    } else {
      detail = (await response.text()).trim().slice(0, 500);
    }
  } catch {
    // The status line below is still actionable if an error body is unreadable.
  }
  return new Error(
    detail || `The export request failed (${response.status} ${response.statusText}).`
  );
}

/**
 * Save generated text through the native desktop dialog when bundled, or the
 * browser picker when the File System Access API is available.
 *
 * The picker must be called directly from the click handler: Chromium requires
 * transient user activation. Other browsers retain the ordinary download
 * fallback.
 */
export async function saveTextFile(
  filename: string,
  text: string,
  mimeType = "text/csv"
): Promise<SaveFileResult> {
  if (isTauriRuntime()) {
    return invokeNativeSave("save_text_file", {
      suggestedName: filename,
      mimeType,
      contents: text,
    });
  }

  const handle = await pickedHandle(filename, mimeType);
  if (handle === null) return "cancelled";
  if (handle) {
    const writable = await handle.createWritable();
    try {
      await writable.write(new Blob([text], { type: `${mimeType};charset=utf-8` }));
      await writable.close();
    } catch (error) {
      await writable.abort(error).catch(() => undefined);
      throw error;
    }
    return "saved";
  }

  const blob = new Blob([text], { type: `${mimeType};charset=utf-8` });
  if (typeof URL.createObjectURL === "function") {
    const url = URL.createObjectURL(blob);
    try {
      clickDownload(filename, url);
    } finally {
      URL.revokeObjectURL(url);
    }
  } else {
    clickDownload(
      filename,
      `data:${mimeType};charset=utf-8,${encodeURIComponent(text)}`
    );
  }
  return "download-started";
}

/**
 * Save a server export. The response body is piped into the selected file so a
 * full-resolution PNG never has to be held in WebView memory.
 */
export async function saveUrlFile(
  filename: string,
  url: string,
  mimeType = "application/octet-stream"
): Promise<SaveFileResult> {
  if (isTauriRuntime()) {
    const absoluteUrl = new URL(url, window.location.href).href;
    return invokeNativeSave("save_url_file", {
      suggestedName: filename,
      mimeType,
      url: absoluteUrl,
    });
  }

  const handle = await pickedHandle(filename, mimeType);
  if (handle === null) return "cancelled";
  if (!handle) {
    clickDownload(filename, url);
    return "download-started";
  }

  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) throw await responseError(response);

  const writable = await handle.createWritable();
  try {
    if (response.body) {
      // pipeTo closes the destination after the response has been written.
      await response.body.pipeTo(writable);
    } else {
      await writable.write(await response.blob());
      await writable.close();
    }
  } catch (error) {
    await writable.abort(error).catch(() => undefined);
    throw error;
  }
  return "saved";
}

function reportSaveFailure(error: unknown): void {
  const detail = error instanceof Error ? error.message : String(error);
  window.alert(`QuantEM could not save the file. ${detail}`);
}

/** Start a generated-text save and make any failure visible to the user. */
export function downloadText(
  filename: string,
  text: string,
  mimeType = "text/csv"
): void {
  void saveTextFile(filename, text, mimeType).catch(reportSaveFailure);
}

/** Start a server-backed save and make any failure visible to the user. */
export function downloadUrl(
  filename: string,
  url: string,
  mimeType = "application/octet-stream"
): void {
  void saveUrlFile(filename, url, mimeType).catch(reportSaveFailure);
}

export function downloadCsv(
  filename: string,
  headers: string[],
  rows: Array<Array<unknown>>
): void {
  downloadText(filename, toCsv(headers, rows), "text/csv");
}
