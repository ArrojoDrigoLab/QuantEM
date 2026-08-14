import { useEffect, useMemo } from "react";
import { Link } from "react-router-dom";

import {
  getHomeEntryPage,
  type AssetOrdering,
} from "@/shared/api/assets";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import type { HomeEntry } from "@/shared/types/images";
import { ImageCard } from "@/features/library/components/ImageCard";
import { isLibraryEntryUnfinished } from "@/features/library/components/imageCardUtils";
import { extractApiErrorMessage } from "@/utils/apiErrors";

const PREVIEW_LIMIT = 5;

interface ExperimentPreviewPanelProps {
  experimentId: string;
  name: string;
  search: string;
  dataset: string;
  ordering: AssetOrdering;
  refreshKey: number;
  onDelete: (image: HomeEntry) => void;
  deleting: boolean;
  highlightedIds: Set<string>;
  selecting: boolean;
  selectedIds: Set<string>;
  pinnedEntries: HomeEntry[];
  onToggleSelect: (assetId: string, selected: boolean) => void;
  onEntriesLoaded: (experimentId: string, entries: HomeEntry[]) => void;
}

/** A home-page experiment summary with only its first five image previews. */
export function ExperimentPreviewPanel({
  experimentId,
  name,
  search,
  dataset,
  ordering,
  refreshKey,
  onDelete,
  deleting,
  highlightedIds,
  selecting,
  selectedIds,
  pinnedEntries,
  onToggleSelect,
  onEntriesLoaded,
}: ExperimentPreviewPanelProps) {
  const { data, loading, error, refetch } = useApiQuery(
    () =>
      getHomeEntryPage({
        search,
        ordering,
        experiment: experimentId,
        ...(dataset ? { dataset } : {}),
        limit: PREVIEW_LIMIT,
        offset: 0,
      }),
    [dataset, experimentId, ordering, refreshKey, search]
  );
  const entries = useMemo(() => {
    const seen = new Set(pinnedEntries.map((entry) => entry.id));
    return [
      ...pinnedEntries,
      ...(data?.results ?? []).filter((entry) => !seen.has(entry.id)),
    ].slice(0, PREVIEW_LIMIT);
  }, [data, pinnedEntries]);
  const total = Math.max(data?.total ?? 0, entries.length);

  useEffect(() => {
    onEntriesLoaded(experimentId, entries);
  }, [entries, experimentId, onEntriesLoaded]);

  useEffect(() => {
    if (!entries.some(isLibraryEntryUnfinished)) return undefined;
    const intervalId = window.setInterval(() => void refetch(), 3000);
    return () => clearInterval(intervalId);
  }, [entries, refetch]);

  if (!loading && !error && total === 0) return null;

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="m-0 text-lg font-semibold text-slate-950">
          <Link
            className="text-cyan-800 hover:text-cyan-950 hover:underline"
            to={`/experiments/${experimentId}`}
          >
            {name}
          </Link>
        </h2>
        {data || entries.length > 0 ? (
          <p className="m-0 text-sm text-slate-600">
            {entries.length} of {total} {total === 1 ? "image" : "images"}
          </p>
        ) : null}
      </div>

      {loading ? <p className="m-0 text-sm text-slate-600">Loading previews…</p> : null}
      {error ? (
        <p className="m-0 text-sm text-red-700">
          {extractApiErrorMessage(error, "The experiment previews could not be loaded.")}
        </p>
      ) : null}
      {entries.length > 0 ? (
        <div className="flex gap-4 overflow-x-auto pb-1">
          {entries.map((image) => (
            <div key={image.id} className="h-[300px] w-[230px] shrink-0">
              <ImageCard
                image={image}
                onDelete={onDelete}
                deleting={deleting}
                justImported={highlightedIds.has(image.id)}
                selectable={selecting}
                selected={selectedIds.has(image.id)}
                onToggleSelect={onToggleSelect}
              />
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
