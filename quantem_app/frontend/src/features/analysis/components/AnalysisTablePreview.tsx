import { useEffect, useState } from "react";
import { getAnalysisExportUrl, getAnalysisTable, type AnalysisTablePage } from "@/shared/api/analysis";
import type { AnalysisRun } from "@/shared/types/analysis";
import { Button } from "@/shared/ui/design";
import { extractApiErrorMessage } from "@/utils/apiErrors";

const TABLE_NAMES: Record<string, string> = {
  "objects.csv": "Objects", "area.csv": "Area", "composition.csv": "Composition", "image_summary.csv": "Image summary",
};

export function AnalysisTablePreview({ run }: { run: AnalysisRun }) {
  const files = Object.keys(TABLE_NAMES).filter((name) => run.exports.includes(name));
  const [file, setFile] = useState(files[0] ?? "");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<AnalysisTablePage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    if (!file) return;
    const controller = new AbortController();
    setLoading(true); setError(""); setPage(null);
    void getAnalysisTable(run.id, file, offset, controller.signal)
      .then((result) => { if (!controller.signal.aborted) setPage(result); })
      .catch((cause) => { if (!controller.signal.aborted) setError(extractApiErrorMessage(cause, "This table could not be loaded.")); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [run.id, file, offset, retry]);

  if (!files.length) return <p className="text-sm text-slate-600">No spreadsheet is available for this image.</p>;
  return <div className="flex min-h-0 flex-1 flex-col gap-3">
    <div className="flex flex-wrap items-center justify-between gap-3">
      {files.length > 1 ? <label className="flex items-center gap-2 text-sm">Table
        <select className="rounded-md border border-slate-300 bg-white px-3 py-2" value={file} onChange={(event) => { setFile(event.target.value); setOffset(0); }}>
          {files.map((name) => <option key={name} value={name}>{TABLE_NAMES[name]}</option>)}
        </select>
      </label> : <span className="font-medium">{TABLE_NAMES[file]}</span>}
      <a className="inline-flex h-10 items-center justify-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-500" href={getAnalysisExportUrl(run.id, file)} download>Download</a>
    </div>
    {error ? <div className="flex items-center gap-3"><p role="alert" className="text-sm text-red-700">{error}</p><Button size="sm" onClick={() => setRetry((value) => value + 1)}>Retry</Button></div> :
      loading ? <p role="status" className="text-sm">Loading table…</p> : page && <>
        <div className="min-h-0 flex-1 overflow-auto rounded-md border border-slate-200" tabIndex={0} aria-label="Scroll spreadsheet">
          <table className="w-full border-separate border-spacing-0 text-left text-sm tabular-nums" aria-label={`${TABLE_NAMES[file]} spreadsheet`}>
            <thead className="sticky top-0 z-10 bg-slate-100"><tr>
              <th scope="col" className="border-b border-r border-slate-200 px-3 py-2 text-slate-500">Row</th>
              {page.columns.map((column, index) => <th key={index} scope="col" className="whitespace-nowrap border-b border-r border-slate-200 px-3 py-2 font-medium">{column}</th>)}
            </tr></thead>
            <tbody>{page.rows.map((row, index) => <tr key={page.offset + index} className="odd:bg-white even:bg-slate-50">
              <th scope="row" className="border-b border-r border-slate-200 px-3 py-2 font-normal text-slate-500">{(page.offset + index + 1).toLocaleString()}</th>
              {page.columns.map((_, column) => <td key={column} className="whitespace-pre border-b border-r border-slate-200 px-3 py-2">{row[column] ?? ""}</td>)}
            </tr>)}</tbody>
          </table>
          {!page.rows.length && <p className="p-3 text-sm text-slate-600">No rows in this table.</p>}
        </div>
        <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <Button size="sm" disabled={!offset} onClick={() => setOffset(Math.max(0, offset - page.limit))}>Previous rows</Button>
          <span>{page.rows.length ? `Rows ${(page.offset + 1).toLocaleString()}–${(page.offset + page.rows.length).toLocaleString()}` : "0 rows"}</span>
          <Button size="sm" disabled={!page.has_more} onClick={() => setOffset(offset + page.limit)}>Next rows</Button>
        </div>
      </>}
  </div>;
}
