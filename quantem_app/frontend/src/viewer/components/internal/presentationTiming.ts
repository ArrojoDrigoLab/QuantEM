/** Diagnostic timing ends only when the caller acknowledges a coherent draw. */
export class PresentationTiming {
  private pending = new Map<string, number>();
  private completed = new Set<string>();

  requested(key: string, now = performance.now()) {
    if (this.pending.has(key) || this.completed.has(key)) return;
    this.pending.set(key, now);
    if (this.pending.size > 64) this.pending.delete(this.pending.keys().next().value!);
  }

  displayed(key: string, now = performance.now()): number | undefined {
    const start = this.pending.get(key);
    if (start === undefined) return;
    this.pending.delete(key);
    this.completed.add(key);
    if (this.completed.size > 64) this.completed.delete(this.completed.keys().next().value!);
    if (typeof performance.measure === "function") {
      // Keep only the latest entry; diagnostic collection must also be bounded.
      performance.clearMeasures("quantem.overlay.presentation");
      performance.measure("quantem.overlay.presentation", { start, end: now, detail: { revision: key } });
    }
    return now - start;
  }
}
