import { useEffect, useRef, useState } from "react";
import { updateSystemSettings, type SystemSettings } from "@/shared/api/systemSettings";
import { SettingSaveIndicator, type SettingSaveState } from "./SettingSaveIndicator";

const AUTOSAVE_DELAY_MS = 400;
const SAVING_INDICATOR_MS = 1000;

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export function DatabaseCacheSettings({
  settings,
}: {
  settings: SystemSettings;
}) {
  const [current, setCurrent] = useState(settings);
  const cache = current.database_cache!;
  const initialValue = current.database_cache_mb ?? cache.active_mb ?? cache.recommended_mb;
  const [value, setValue] = useState(String(initialValue));
  const [saveState, setSaveState] = useState<SettingSaveState>("idle");
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);
  const saveGeneration = useRef(0);
  const saveChain = useRef<Promise<void>>(Promise.resolve());
  const lastScheduled = useRef(initialValue);
  const persistedValue = useRef(initialValue);
  const parsed = Number(value);
  const valid = value.trim() !== "" && Number.isInteger(parsed) &&
    parsed >= cache.min_mb && parsed <= cache.max_mb;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    if (!cache.available) return undefined;
    if (!valid) {
      ++saveGeneration.current;
      setSaveState("error");
      return undefined;
    }
    if (parsed === lastScheduled.current) {
      // A scheduled write still owns the indicator. Otherwise, reverting an
      // invalid edit to the persisted value must clear its stale error.
      if (parsed === initialValue) {
        setSaveState((state) => state === "error" ? "idle" : state);
        setError(null);
      }
      return undefined;
    }

    const requested = parsed;
    const generation = ++saveGeneration.current;
    const savingStartedAt = Date.now();
    lastScheduled.current = requested;
    setSaveState("saving");
    setError(null);
    let requestStarted = false;
    const timer = window.setTimeout(() => {
      requestStarted = true;
      const request = saveChain.current
        .catch(() => undefined)
        .then(() => updateSystemSettings({ database_cache_mb: requested }));
      // Serialize writes, but do not make a later write wait for the one-second
      // presentation delay used by the save indicator.
      saveChain.current = request.then(() => undefined, () => undefined);
      void request.then(
        async (updated) => {
          const persisted = updated.database_cache_mb ?? requested;
          persistedValue.current = persisted;
          if (!mounted.current || generation !== saveGeneration.current) return;
          setCurrent(updated);
          const remaining = SAVING_INDICATOR_MS - (Date.now() - savingStartedAt);
          if (remaining > 0) await wait(remaining);
          if (!mounted.current || generation !== saveGeneration.current) return;
          lastScheduled.current = persisted;
          setValue(String(persisted));
          setSaveState("saved");
        },
        (reason: unknown) => {
          if (!mounted.current || generation !== saveGeneration.current) return;
          lastScheduled.current = persistedValue.current;
          setSaveState("error");
          setError(reason instanceof Error ? reason.message : "The cache setting could not be saved.");
        }
      );
    }, AUTOSAVE_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      if (!requestStarted && lastScheduled.current === requested) {
        lastScheduled.current = persistedValue.current;
        setSaveState((state) => state === "saving" ? "idle" : state);
      }
    };
  }, [cache.available, initialValue, parsed, valid]);

  return (
    <div className="mt-6 border-t border-slate-200 pt-5">
      <h2 className="m-0 text-lg font-semibold text-slate-950">Database memory</h2>
      <p className="mt-1 text-sm text-slate-600">
        RAM used to cache database pages. Image processing uses additional memory.
        Changes take effect the next time you restart QuantEM. Upgrades keep your choice.
      </p>
      <p className="mt-2 text-sm text-slate-600">
        Active cache: {cache.active_mb === null ? "Unavailable" : `${cache.active_mb.toLocaleString()} MiB`}.
        {cache.total_ram_mb !== null ? ` Detected RAM: ${cache.total_ram_mb.toLocaleString()} MiB.` : ""}
      </p>
      <div className="mt-3 flex flex-wrap items-end gap-3">
        <label className="grid gap-1.5 text-sm font-medium text-slate-800">
          Database cache (MiB)
          <span className="grid grid-cols-[10rem_1.25rem] items-center gap-2">
            <input
              className="h-10 w-40 rounded-lg border border-slate-300 px-3 text-slate-950 disabled:bg-slate-100"
              type="number"
              min={cache.min_mb}
              max={cache.max_mb}
              step={1}
              value={value}
              disabled={!cache.available}
              onChange={(event) => { setValue(event.target.value); setError(null); }}
              aria-invalid={!valid}
              aria-describedby="database-cache-help"
            />
            <SettingSaveIndicator state={saveState} label="database cache size" />
          </span>
        </label>
        <button
          type="button"
          className="h-10 rounded-lg border border-slate-300 px-3 text-sm disabled:opacity-50"
          disabled={!cache.available}
          onClick={() => { setValue(String(cache.recommended_mb)); setError(null); }}
        >
          Use suggested size
        </button>
      </div>
      <p id="database-cache-help" className="mt-2 text-xs text-slate-500">
        Suggested: {cache.recommended_mb.toLocaleString()} MiB. Allowed: {cache.min_mb}–{cache.max_mb.toLocaleString()} MiB.
      </p>
      {!valid ? <p className="mt-2 text-sm text-red-700">Enter a whole number within the allowed range.</p> : null}
      {!cache.available ? <p className="mt-2 text-sm text-slate-600">Available when using PostgreSQL.</p> : null}
      {cache.restart_required ? (
        <p className="mt-2 text-sm text-amber-800" role="status">
          Saved size: {current.database_cache_mb?.toLocaleString()} MiB. Takes effect after restarting QuantEM.
        </p>
      ) : null}
      {error ? <p className="mt-2 text-sm text-red-700" role="alert">{error}</p> : null}
    </div>
  );
}
