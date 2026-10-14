import { useLayoutEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { Experiment } from "@/shared/types/common";

const CHIP_CLASS = "block w-max max-w-full shrink-0 truncate rounded border border-slate-200 px-3 py-2 text-sm font-medium";

export function RecentExperiments({ experiments }: { experiments: Experiment[] }) {
  const recent = useMemo(() => experiments.filter(experiment => experiment.last_opened_at)
    .sort((a, b) => Date.parse(b.last_opened_at!) - Date.parse(a.last_opened_at!) || a.name.localeCompare(b.name))
    .slice(0, 3), [experiments]);
  const measureRef = useRef<HTMLDivElement>(null);
  const [visibleCount, setVisibleCount] = useState(3);

  useLayoutEffect(() => {
    const row = measureRef.current;
    if (!row) return;
    const measure = () => {
      const available = row.getBoundingClientRect().width;
      if (available <= 0) return;
      const gap = parseFloat(getComputedStyle(row).columnGap) || 0;
      let used = 0;
      let count = 0;
      for (const item of row.children) {
        const width = item.getBoundingClientRect().width;
        const next = used + (count ? gap : 0) + width;
        if (next > available + 0.5) break;
        used = next;
        count += 1;
      }
      setVisibleCount(Math.max(1, count));
    };
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(row);
    for (const item of row.children) observer.observe(item);
    return () => observer.disconnect();
  }, [recent]);

  if (!recent.length) return null;
  return <section className="rounded-lg border border-slate-200 bg-white p-4" aria-label="Continue working">
    <h2 className="mb-3 text-base font-semibold">Continue working</h2>
    <div className="relative min-w-0">
      {/* Measure all candidates without putting hidden links in the tab order. */}
      <div ref={measureRef} aria-hidden="true" className="pointer-events-none invisible absolute inset-x-0 top-0 flex gap-3 overflow-hidden">
        {recent.map(experiment => <span key={experiment.id} className={CHIP_CLASS}>{experiment.name}</span>)}
      </div>
      <div className="flex gap-3 overflow-hidden">
        {recent.slice(0, visibleCount).map(experiment => <Link key={experiment.id}
          className={`${CHIP_CLASS} hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-cyan-500`}
          title={experiment.name} to={experiment.last_route || `/experiments/${experiment.id}`}>{experiment.name}</Link>)}
      </div>
    </div>
  </section>;
}
