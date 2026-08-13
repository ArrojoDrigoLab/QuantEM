import { Link } from "react-router-dom";

import { ModelManagementSection } from "@/features/models/ModelManagementSection";

/** Legacy direct route; the primary model-management surface is Settings. */
export function ModelsScreen() {
  return (
    <main className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
      <div className="mx-auto flex max-w-5xl flex-col gap-5">
        <header className="flex flex-wrap items-start justify-between gap-4 rounded-lg border border-slate-200 bg-white px-5 py-4 shadow-sm">
          <div>
            <h1 className="m-0 text-2xl font-semibold text-slate-950">Models</h1>
            <p className="mt-2 max-w-3xl text-sm text-slate-600">
              Model weights are not included in the application. Select models here to
              download / remove.
            </p>
          </div>
          <Link
            className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2"
            to="/"
          >
            Back to Home
          </Link>
        </header>
        <ModelManagementSection />
      </div>
    </main>
  );
}
