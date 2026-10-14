export function AnalysisNotes({ notes }: { notes: string[] }) {
  if (!notes.length) return null;
  return <section aria-label="Notes" className="space-y-2 text-sm text-slate-700">
    <h3 className="font-semibold text-slate-900">Notes</h3>
    <ul className="list-disc space-y-1 pl-5">{notes.map((note) => <li key={note}>{note}</li>)}</ul>
  </section>;
}
