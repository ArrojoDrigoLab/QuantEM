import { ANALYSIS_FILES } from "./outputSelection";

export function AnalysisFileSelection({ value, onChange }: {
  value: Record<string, boolean>; onChange: (next: Record<string, boolean>) => void;
}) {
  return <fieldset className="flex flex-wrap gap-3">
    <legend className="font-medium">Files to generate</legend>
    {Object.entries(ANALYSIS_FILES).map(([name, label]) => <label key={name}>
      <input type="checkbox" checked={value[name] !== false}
        onChange={(event) => onChange({ ...value, [name]: event.target.checked })} /> {label} <span className="text-xs text-slate-500">({name})</span>
    </label>)}
    <p className="w-full text-sm text-slate-600">Files are generated when the selected measurements apply. Saved measurements remain available in the app.</p>
  </fieldset>;
}
