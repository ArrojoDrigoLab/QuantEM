/** Retry transient display reads without keeping an obsolete viewer alive. */
export async function retryOverlayRequest<T>(request: () => Promise<T>, signal: AbortSignal): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    signal.throwIfAborted();
    try { return await request(); }
    catch (error) {
      if (attempt >= 2 || signal.aborted) throw error;
      await new Promise<void>((resolve, reject) => {
        const abort = () => { clearTimeout(timer); reject(signal.reason); };
        const timer = setTimeout(() => {
          signal.removeEventListener("abort", abort);
          resolve();
        }, 400 * (attempt + 1));
        signal.addEventListener("abort", abort, { once: true });
      });
    }
  }
}
