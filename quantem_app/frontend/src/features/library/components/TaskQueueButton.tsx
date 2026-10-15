import { useEffect, useId, useState } from "react";
import { getJobQueueStatus } from "@/shared/api/jobs";
import { Button } from "@/shared/ui/design";

export function TaskQueueButton({ isOpen, onClick }: {
  isOpen: boolean;
  onClick: () => void;
}) {
  const tooltipId = useId();
  const [counts, setCounts] = useState<{ running: number; queued: number } | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let current = true;
    let fetching = false;
    const update = async () => {
      if (fetching) return;
      fetching = true;
      try {
        const queue = await getJobQueueStatus();
        if (!current) return;
        setCounts({
          running: queue.running.length,
          queued: queue.queues.reduce((count, row) => count + row.pending.length, 0),
        });
        setUnavailable(false);
      } catch {
        if (current) { setCounts(null); setUnavailable(true); }
      } finally {
        fetching = false;
      }
    };
    void update();
    const timer = window.setInterval(() => void update(), 10000);
    return () => { current = false; window.clearInterval(timer); };
  }, [isOpen]);

  const active = counts !== null && (counts.running > 0 || counts.queued > 0);
  const status = counts
    ? `${counts.running} running · ${counts.queued} queued`
    : unavailable ? "Status unavailable" : "Checking tasks…";

  return (
    <span className="group relative inline-flex">
      <Button
        size="icon"
        className="relative"
        aria-label="Tasks & queues"
        aria-describedby={tooltipId}
        aria-expanded={isOpen}
        onClick={onClick}
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round"
          aria-hidden="true">
          <path d="M9 5H6a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-3" />
          <rect x="9" y="2" width="6" height="5" rx="1" />
          <path d="M9 12h6M9 17h6" />
        </svg>
        {active && <span aria-hidden="true" className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-blue-500 ring-2 ring-white" />}
      </Button>
      <span id={tooltipId} role="tooltip"
        className="pointer-events-none invisible absolute right-0 top-full z-50 mt-2 w-max rounded-md bg-slate-900 px-3 py-2 text-xs font-normal text-white opacity-0 shadow-lg group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100">
        <span className="block font-semibold">Tasks &amp; queues</span>
        <span>{status}</span>
      </span>
    </span>
  );
}
