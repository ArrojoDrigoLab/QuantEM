export type SettingSaveState = "idle" | "saving" | "saved" | "error";

/** Fixed-width feedback keeps settings fields from shifting while autosaving. */
export function SettingSaveIndicator({
  state,
  label,
}: {
  state: SettingSaveState;
  label: string;
}) {
  if (state === "idle") return <span className="block h-5 w-5" aria-hidden="true" />;
  if (state === "saving") {
    return (
      <span className="flex h-5 w-5 items-center justify-center text-cyan-700" role="status" aria-label={`Saving ${label}`}>
        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" aria-hidden="true">
          <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeOpacity="0.25" strokeWidth="3" />
          <path d="M21 12a9 9 0 0 0-9-9" fill="none" stroke="currentColor" strokeLinecap="round" strokeWidth="3" />
        </svg>
      </span>
    );
  }
  if (state === "saved") {
    return (
      <span className="flex h-5 w-5 items-center justify-center text-emerald-600" role="status" aria-label={`${label} saved`}>
        <svg className="h-5 w-5" viewBox="0 0 24 24" aria-hidden="true">
          <path d="m5 12 4 4L19 6" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" />
        </svg>
      </span>
    );
  }
  return (
    <span className="flex h-5 w-5 items-center justify-center font-bold text-red-600" role="status" aria-label={`${label} could not be saved`}>
      !
    </span>
  );
}
