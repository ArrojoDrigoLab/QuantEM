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

/** Save `text` as `filename`. No-op outside a browser (jsdom-safe). */
export function downloadText(
  filename: string,
  text: string,
  mimeType = "text/csv"
): void {
  if (typeof document === "undefined" || typeof URL.createObjectURL !== "function") {
    return;
  }
  const blob = new Blob([text], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function downloadCsv(
  filename: string,
  headers: string[],
  rows: Array<Array<unknown>>
): void {
  downloadText(filename, toCsv(headers, rows), "text/csv");
}
