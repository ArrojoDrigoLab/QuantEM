import { Link } from "react-router-dom";
import type { Experiment } from "@/shared/types/common";
import { pinExperiment } from "@/shared/api/workspace";
import { Button } from "@/shared/ui/design";
import { useState } from "react";

export function orderExperiments(experiments: Experiment[], sort: string) {
  return [...experiments].sort((a, b) =>
    Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) ||
    (sort === "recent" ? (b.last_opened_at ?? "").localeCompare(a.last_opened_at ?? "") : 0) ||
    a.name.localeCompare(b.name));
}
export function RecentExperiments({ experiments, onChange }: {
  experiments: Experiment[]; onChange: () => void;
}) {
  const [error, setError] = useState("");
  const recent = orderExperiments(experiments.filter(e => e.pinned || e.last_opened_at), "recent").slice(0, 6);
  if (!recent.length) return null;
  return <section className="rounded-lg border border-slate-200 bg-white p-4" aria-label="Continue working">
    <h2 className="mb-3 text-base font-semibold">Continue working</h2>
    <div className="flex flex-wrap gap-3">{recent.map(experiment => <div key={experiment.id}
      className="flex items-center gap-2 rounded border border-slate-200 px-3 py-2">
      <Link to={experiment.last_route || `/experiments/${experiment.id}`}>{experiment.name}</Link>
      <Button size="sm" aria-label={`${experiment.pinned ? "Unpin" : "Pin"} ${experiment.name}`}
        onClick={() => { void pinExperiment(experiment.id, !experiment.pinned).then(onChange)
          .catch(() => setError("The pin could not be saved. Try again.")); }}>
        {experiment.pinned ? "Unpin" : "Pin"}
      </Button>
    </div>)}</div>
    {error && <p role="alert">{error}</p>}
  </section>;
}
