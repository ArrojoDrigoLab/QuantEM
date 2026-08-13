import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "react-router-dom";
import {
  deleteAsset,
  deleteExperiment,
  getAsset,
  getHomeEntryPage,
  recoverDeferredUploadedAssetPipelines,
  type AssetOrdering,
} from "@/shared/api/assets";
import type { Experiment } from "@/shared/types/common";
import { getSystemStatus } from "@/shared/api/jobs";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Button, PageState, Panel } from "@/shared/ui/design";
import { FineTuneMenuButton } from "@/features/finetune/FineTuneMenuButton";
import { ExperimentPreviewPanel } from "@/features/library/components/ExperimentPreviewPanel";
import {
  isLibraryEntryUnfinished,
} from "@/features/library/components/imageCardUtils";
import {
  ImageUploadPanel,
  type ImageUploadPanelHandle,
  type ImportBatchPosition,
} from "@/features/library/components/ImageUploadPanel";
import { JobQueueSidebar } from "@/features/library/components/JobQueueSidebar";
import { WorkflowGuide } from "@/features/library/components/WorkflowGuide";
import {
  hasSeenWorkflowGuide,
  rememberWorkflowGuideDismissed,
} from "@/features/library/components/workflowGuideStorage";
import { useExperiments } from "@/features/library/components/grouping/useExperiments";
import { LibrarySelectionBar } from "@/features/library/components/grouping/LibrarySelectionBar";
import { SettingsDialog } from "@/features/settings/SettingsDialog";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import { UNASSIGNED_FILTER } from "@/shared/types/images";
import type { AssetDetail, HomeEntry } from "@/shared/types/images";

/**
 * The sorts this control offers, and why "Status" is not one of them.
 *
 * It used to be. `/api/assets/` has no status ordering (`ASSET_ORDERINGS` in
 * `assets/views.py` covers name, created and updated only, and an unknown value
 * is *silently* replaced by `display_name`), so the page fetched the newest 60
 * rows by date and then arranged those 60 by `preprocess_stage` — a raw enum
 * name, sorted alphabetically, so "CANCELLED" led and "SKIPPED" trailed and
 * neither the order nor the words matched anything on the cards.
 *
 * Two independent reasons that had to go rather than be papered over:
 *
 * 1. **It sorted a window, not the library.** With 212 images the user was
 *    offered "order the library by status" and got "reorder page 1". That is
 *    the same class of defect as the one that made a new import vanish — a
 *    control whose label describes the library while its effect covers one
 *    page.
 * 2. **The order was arbitrary.** Alphabetical over `ENCODING`, `FAILED`,
 *    `NONE`, `SAM` is not an order any user asked for, and the card says
 *    "Preparing 62%", "Ready", "Failed" — words that sort differently again.
 *
 * A real status sort needs `preprocess_stage` in the server's `ASSET_ORDERINGS`
 * and a defined rank; that is a backend change and it is not this package's
 * file to make. Until then the option is gone, because a control that does
 * nothing is worse than one that is absent: the user cannot see that it did
 * nothing. Every card still states its own status, and Tasks & Queues lists
 * the work.
 *
 * A stored preference of `"status"` from before this change fails
 * `isSortField` and falls back to `created_at`, which is the default anyway.
 */
type SortField = "display_name" | "created_at" | "updated_at";
type SortDirection = "asc" | "desc";

interface StoredLibraryControls {
  search?: string;
  sortField?: SortField;
  sortDirection?: SortDirection;
}

const LIBRARY_CONTROLS_STORAGE_KEY = "quantem-library-controls-v1";
const LIBRARY_SUMMARY_LIMIT = 1;
const SORT_FIELDS: SortField[] = ["display_name", "created_at", "updated_at"];
const SORT_DIRECTIONS: SortDirection[] = ["asc", "desc"];
const NO_PINNED_ENTRIES: HomeEntry[] = [];

function isSortField(value: unknown): value is SortField {
  return typeof value === "string" && SORT_FIELDS.includes(value as SortField);
}

function isSortDirection(value: unknown): value is SortDirection {
  return (
    typeof value === "string" && SORT_DIRECTIONS.includes(value as SortDirection)
  );
}

function loadStoredLibraryControls(): StoredLibraryControls {
  if (typeof window === "undefined") return {};
  try {
    const rawValue = window.localStorage.getItem(LIBRARY_CONTROLS_STORAGE_KEY);
    if (!rawValue) return {};
    const candidate = JSON.parse(rawValue) as Record<string, unknown>;
    const stored: StoredLibraryControls = {};
    if (typeof candidate.search === "string") stored.search = candidate.search;
    if (isSortField(candidate.sortField)) stored.sortField = candidate.sortField;
    if (isSortDirection(candidate.sortDirection)) {
      stored.sortDirection = candidate.sortDirection;
    }
    return stored;
  } catch {
    return {};
  }
}

function saveStoredLibraryControls(state: StoredLibraryControls) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      LIBRARY_CONTROLS_STORAGE_KEY,
      JSON.stringify(state)
    );
  } catch {
    // Control persistence must never block rendering.
  }
}

function sortFieldLabel(field: SortField): string {
  return {
    display_name: "Name",
    created_at: "Imported",
    updated_at: "Updated",
  }[field];
}

/**
 * The `ordering` this page must send for the chosen sort.
 *
 * The library used to send none. `/api/assets/` then falls back to
 * `display_name` (`assets/views.py:73`), returns the alphabetically first 60
 * rows, and the client sorts *those* by import date — so with 62 assets a new
 * import named late in the alphabet was not on page 1 at all, while the sort
 * control read "Imported / Descending" and the footer read "Showing 60 of 62
 * images". Reproduced twice by the mapping pass. The window the server picks
 * has to be the window the user asked for.
 *
 * Every field this control offers is one the server implements — see
 * {@link SortField} for the one that was removed and why.
 */
function serverOrderingFor(
  field: SortField,
  direction: SortDirection
): AssetOrdering {
  const prefix = direction === "desc" ? "-" : "";
  return `${prefix}${field}` as AssetOrdering;
}

/**
 * The freshly created asset, as a library entry.
 *
 * `POST /api/assets/upload/` answers with the detail payload, which carries
 * every field a card reads except the two the list serializer adds. Building
 * the card from it means the import is on screen the instant the request
 * returns, with no dependency on a refetch landing, on the poll not being
 * mid-flight, and on the new row being inside whatever page the server chose.
 */
function entryFromUploadedAsset(asset: AssetDetail): HomeEntry {
  return {
    ...asset,
    // `AssetDetail.asset_type` is the wider `AvailabilityFilter` (it can be
    // "all"); a library entry is only ever one of the two real kinds, and an
    // upload is always local.
    asset_type: "local",
    metadata_summary: `${asset.width} x ${asset.height}`,
    can_open: asset.is_workable ?? false,
  };
}

/**
 * What deleting this experiment actually does, with the numbers in it.
 *
 * Composed as whole sentences rather than assembled from fragments in JSX:
 * "Its 1 image stays ... and move" is what fragment assembly
 * produces, and a reassurance that reads as broken English is not reassuring.
 *
 * The promise it makes is real. The server first gives every image its own
 * experiment named after its display name; the datasets do not survive,
 * because a dataset cannot exist outside its experiment.
 */
function describeExperimentDeletion(experiment: Experiment): string {
  const images =
    experiment.asset_count === 0
      ? "It holds no images."
      : experiment.asset_count === 1
        ? "Its 1 image stays in the library and moves to its own named experiment."
        : `Its ${experiment.asset_count} images stay in the library and each moves to its own named experiment.`;
  const datasets =
    experiment.datasets.length === 0
      ? ""
      : experiment.datasets.length === 1
        ? " Its 1 dataset goes with it, because a dataset cannot exist outside an experiment."
        : ` Its ${experiment.datasets.length} datasets go with it, because a dataset cannot exist outside an experiment.`;
  return `${images}${datasets} No image files are deleted.`;
}

export function LibraryPage() {
  const [searchParams] = useSearchParams();
  const [initialStoredControls] = useState<StoredLibraryControls>(() =>
    loadStoredLibraryControls()
  );
  const [search, setSearch] = useState(() => initialStoredControls.search ?? "");
  const [sortField, setSortField] = useState<SortField>(
    () => initialStoredControls.sortField ?? "created_at"
  );
  const [sortDirection, setSortDirection] = useState<SortDirection>(
    () => initialStoredControls.sortDirection ?? "desc"
  );
  const [entries, setEntries] = useState<HomeEntry[]>([]);
  const [entryTotal, setEntryTotal] = useState(0);
  /**
   * `entriesLoading` is the *first* load only. It used to be every load, and
   * `entriesLoading` gates the whole grid — so the 3 s poll that runs while
   * anything is preprocessing unmounted every card, blanked the page to
   * "Loading images…", and remounted them, re-requesting every thumbnail, once
   * every 3 s for the entire ~100 s of a large import. It did the same thing at
   * exactly the moment the post-upload refetch landed: the library vanished
   * while the user was looking for their new card. A background refetch now
   * says so in one line and leaves the grid alone.
   */
  const [entriesLoading, setEntriesLoading] = useState(true);
  const [entriesRefetching, setEntriesRefetching] = useState(false);
  const [entriesError, setEntriesError] = useState<Error | null>(null);
  const [deleteConfirmImage, setDeleteConfirmImage] = useState<HomeEntry | null>(null);
  const [deleteConfirmExperiment, setDeleteConfirmExperiment] =
    useState<Experiment | null>(null);
  const [isQueueSidebarOpen, setIsQueueSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [showHelp, setShowHelp] = useState(() => !hasSeenWorkflowGuide());
  const [previewRefreshKey, setPreviewRefreshKey] = useState(0);
  const [previewEntriesByExperiment, setPreviewEntriesByExperiment] = useState<
    Record<string, HomeEntry[]>
  >({});
  /**
   * The images imported in this session, pinned to the front of the grid in
   * the order they landed.
   *
   * Held locally rather than waited for in a list response: that is what makes
   * "the cards are there, first, immediately" true regardless of which page the
   * server would have put them on, whether the refetch was swallowed by an
   * in-flight poll, or how the library happens to be sorted right now.
   */
  const [justImported, setJustImported] = useState<HomeEntry[]>([]);
  /** Whether the library is still going to open the import by itself. */
  /** True while a file is being dragged anywhere over this page. */
  const [pageDragActive, setPageDragActive] = useState(false);
  /**
   * The grouping controls. All three are deliberately **not** persisted.
   *
   * The search box is, and that is already the edge of what a remembered
   * control should do: a filter restored on the next launch, pointing at an
   * experiment that has since been renamed or deleted, presents an empty
   * library with no visible cause. These reset with the session.
   */
  const [experimentFilter, setExperimentFilter] = useState(
    () => searchParams.get("experiment") ?? ""
  );
  const [datasetFilter, setDatasetFilter] = useState("");
  /** Selecting mode, and what is selected in it. Off until asked for. */
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const { experiments, reload: reloadExperiments } = useExperiments();
  const entriesRequestIdRef = useRef(0);
  const entriesInFlightKeyRef = useRef<string | null>(null);
  /** Read inside `loadEntries` without making it depend on the entry list. */
  const entriesRef = useRef<HomeEntry[]>([]);
  /**
   * A refetch that arrived while one was already running.
   *
   * `refetchEntries` used to `return` in that case and raise nothing, so the
   * post-upload refetch was silently dropped whenever the 3 s poll happened to
   * be in flight. Narrow (a ~60 ms request inside a 3 000 ms period) but real
   * and invisible. Now it is remembered and re-run.
   */
  const refetchPendingRef = useRef(false);
  const loadEntriesRef = useRef<(() => Promise<void>) | null>(null);
  const uploadPanelRef = useRef<ImageUploadPanelHandle | null>(null);
  const pageDragDepthRef = useRef(0);

  const {
    data: systemStatus,
    error: systemStatusError,
    refetch: refetchSystemStatus,
  } = useApiQuery(() => getSystemStatus(), []);

  useEffect(() => {
    saveStoredLibraryControls({ search, sortField, sortDirection });
  }, [search, sortDirection, sortField]);

  useEffect(() => {
    // A browser/app close can interrupt a multi-file import after its assets
    // exist but before the final atomic queue call. Recovery is idempotent and
    // runs only when the home page mounts, never in the middle of a live batch.
    void recoverDeferredUploadedAssetPipelines().catch(() => {
      // Nothing is lost and the durable markers remain set; the next home-page
      // load retries. The cards continue to show the queued spinner meanwhile.
    });
  }, []);

  const entryQueryParams = useMemo(
    () => ({
      search,
      availability: "local" as const,
      // Never omitted. See `serverOrderingFor`.
      ordering: serverOrderingFor(sortField, sortDirection),
      // Filtered by the server, not by the page. This summary request supplies
      // the authoritative count/empty state; the panels below fetch each
      // experiment's five visible cards.
      ...(experimentFilter ? { experiment: experimentFilter } : {}),
      ...(datasetFilter ? { dataset: datasetFilter } : {}),
    }),
    [datasetFilter, experimentFilter, search, sortDirection, sortField]
  );

  const loadEntries = useCallback(
    async () => {
      const requestKey = JSON.stringify(entryQueryParams);
      if (entriesInFlightKeyRef.current === requestKey) return;
      entriesInFlightKeyRef.current = requestKey;
      const requestId = ++entriesRequestIdRef.current;
      // The page only disappears when there is no current result: polling and
      // post-upload refreshes leave the experiment previews in place.
      if (entriesRef.current.length === 0) {
        setEntriesLoading(true);
      } else {
        setEntriesRefetching(true);
      }
      setEntriesError(null);
      try {
        const page = await getHomeEntryPage({
          ...entryQueryParams,
          limit: LIBRARY_SUMMARY_LIMIT,
          offset: 0,
        });
        if (requestId !== entriesRequestIdRef.current) return;
        setEntries(page.results);
        setEntryTotal(page.total);
      } catch (err) {
        if (requestId !== entriesRequestIdRef.current) return;
        setEntriesError(err instanceof Error ? err : new Error("Unknown error"));
        setEntries([]);
        setEntryTotal(0);
      } finally {
        if (entriesInFlightKeyRef.current === requestKey) {
          entriesInFlightKeyRef.current = null;
        }
        if (requestId === entriesRequestIdRef.current) {
          setEntriesLoading(false);
          setEntriesRefetching(false);
        }
        // A refetch that collided with this one. Run it now rather than
        // dropping it on the floor.
        if (refetchPendingRef.current) {
          refetchPendingRef.current = false;
          void loadEntriesRef.current?.();
        }
      }
    },
    [entryQueryParams]
  );

  useEffect(() => {
    entriesRef.current = entries;
  }, [entries]);

  useEffect(() => {
    loadEntriesRef.current = loadEntries;
  }, [loadEntries]);

  useEffect(() => {
    void loadEntries();
  }, [loadEntries]);

  const refetchEntries = useCallback(async () => {
    if (entriesInFlightKeyRef.current !== null) {
      refetchPendingRef.current = true;
      return;
    }
    await loadEntries();
  }, [loadEntries]);

  const { mutate: deleteImageMutation, loading: deleting } = useApiMutation(
    (assetId: string) => deleteAsset(assetId),
    {
      onSuccess: () => {
        setDeleteConfirmImage(null);
        setPreviewRefreshKey((current) => current + 1);
        void refetchEntries();
        void reloadExperiments();
      },
    }
  );

  /**
   * Keep the pinned imports' own cards current, about once a second.
   *
   * Their own requests, not the list poll: a pinned card has to keep counting
   * up even when it is not inside the page the server returned (a library
   * sorted by name, a library past 60 images), and these are the cards whose
   * progress the user is actually watching.
   *
   * One sweep at a time, sequentially, rather than an interval per card: forty
   * pinned imports on a `setInterval(1000)` each would put forty requests a
   * second on a single-process loopback server that is simultaneously building
   * forty pyramids. A chained timeout also cannot overlap itself, so a slow
   * response delays the next sweep instead of stacking on top of it.
   */
  const pinnedUnfinishedIds = useMemo(
    () =>
      justImported
        .filter(isLibraryEntryUnfinished)
        .map((entry) => entry.id),
    [justImported]
  );
  const pinnedUnfinishedIdsRef = useRef<string[]>(pinnedUnfinishedIds);
  useEffect(() => {
    pinnedUnfinishedIdsRef.current = pinnedUnfinishedIds;
  }, [pinnedUnfinishedIds]);
  // The effect restarts only when the *count* crosses zero, so a sweep that
  // updates one card does not cancel itself partway through the rest.
  const anyPinnedUnfinished = pinnedUnfinishedIds.length > 0;
  useEffect(() => {
    if (!anyPinnedUnfinished) return undefined;
    let cancelled = false;
    let timerId = 0;
    const sweep = async () => {
      for (const assetId of pinnedUnfinishedIdsRef.current) {
        if (cancelled) return;
        try {
          const asset = await getAsset(assetId);
          if (cancelled) return;
          setJustImported((current) =>
            current.map((entry) =>
              entry.id === asset.id ? entryFromUploadedAsset(asset) : entry
            )
          );
        } catch {
          // A failed status poll leaves the last known card in place; the next
          // sweep tries again.
        }
      }
      if (cancelled) return;
      timerId = window.setTimeout(() => {
        void sweep();
      }, 1000);
    };
    void sweep();
    return () => {
      cancelled = true;
      clearTimeout(timerId);
    };
  }, [anyPinnedUnfinished]);

  /**
   * The grid: the session's import first, then the library.
   *
   * Pinning keeps newly imported images visible immediately for every sort,
   * page and library size while their queued/processing state changes.
   */
  const pinnedImportIds = useMemo(
    () => new Set(justImported.map((entry) => entry.id)),
    [justImported]
  );
  const hasAnyImages = entryTotal > 0 || entries.length > 0 || justImported.length > 0;

  /**
   * The grouping controls only exist once there is something to group by. The
   * first import creates its experiment automatically, so an empty library has
   * no filter and needs no setup prompt.
   */
  const hasGroups = experiments.length > 0;
  const filtered = Boolean(experimentFilter || datasetFilter);
  /** The experiment the filter names, when it names a real one. */
  const filteredExperiment =
    experiments.find((row) => row.id === experimentFilter) ?? null;
  const datasetOptions = useMemo(() => {
    const chosen = experiments.find((row) => row.id === experimentFilter);
    return chosen
      ? chosen.datasets
      : experiments.flatMap((row) => row.datasets);
  }, [experimentFilter, experiments]);

  const shownExperiments = useMemo(
    () => {
      const headers = new Map(
        experiments.map((experiment) => [
          experiment.id,
          { id: experiment.id, name: experiment.name },
        ])
      );
      // A new experiment can exist in the upload response one request before
      // the catalogue refetch returns. Keep its first card immediate instead
      // of temporarily filing it under nowhere.
      justImported.forEach((entry) => {
        if (entry.experiment_id && entry.experiment_name) {
          headers.set(entry.experiment_id, {
            id: entry.experiment_id,
            name: entry.experiment_name,
          });
        }
      });
      const rows = [...headers.values()];
      return experimentFilter
        ? rows.filter((experiment) => experiment.id === experimentFilter)
        : rows;
    },
    [experimentFilter, experiments, justImported]
  );
  const activePreviewGroupIds = useMemo(
    () => new Set(shownExperiments.map((experiment) => experiment.id)),
    [shownExperiments]
  );
  useEffect(() => {
    setPreviewEntriesByExperiment((current) => {
      const retained = Object.fromEntries(
        Object.entries(current).filter(([groupId]) => activePreviewGroupIds.has(groupId))
      );
      return Object.keys(retained).length === Object.keys(current).length
        ? current
        : retained;
    });
  }, [activePreviewGroupIds]);
  const previewEntries = useMemo(() => {
    const byId = new Map<string, HomeEntry>();
    Object.values(previewEntriesByExperiment)
      .flat()
      .forEach((entry) => byId.set(entry.id, entry));
    return [...byId.values()];
  }, [previewEntriesByExperiment]);
  const pinnedEntriesByExperiment = useMemo(() => {
    const grouped: Record<string, HomeEntry[]> = {};
    justImported.forEach((entry) => {
      if (datasetFilter && !entry.dataset_ids?.includes(datasetFilter)) return;
      const groupId = entry.experiment_id;
      if (!groupId) return;
      if (!activePreviewGroupIds.has(groupId)) return;
      grouped[groupId] = [...(grouped[groupId] ?? []), entry];
    });
    return grouped;
  }, [activePreviewGroupIds, datasetFilter, justImported]);

  const selectedEntries = useMemo(() => {
    const byId = new Map(previewEntries.map((entry) => [entry.id, entry]));
    return [...selectedIds]
      .map((id) => byId.get(id))
      .filter((entry): entry is HomeEntry => Boolean(entry));
  }, [previewEntries, selectedIds]);

  const handlePreviewEntriesLoaded = useCallback((groupId: string, loaded: HomeEntry[]) => {
    setPreviewEntriesByExperiment((current) => ({
      ...current,
      [groupId]: loaded,
    }));
  }, []);

  const dismissWorkflowGuide = useCallback(() => {
    rememberWorkflowGuideDismissed();
    setShowHelp(false);
  }, []);

  const toggleSelect = useCallback((assetId: string, next: boolean) => {
    setSelectedIds((current) => {
      const updated = new Set(current);
      if (next) updated.add(assetId);
      else updated.delete(assetId);
      return updated;
    });
  }, []);

  const handleGroupingApplied = useCallback(() => {
    // Both sides moved: the cards carry their new experiment and dataset, and
    // the catalogue's counts (and possibly a brand new experiment) changed.
    void refetchEntries();
    void reloadExperiments();
    setPreviewRefreshKey((current) => current + 1);
    setSelectedIds(new Set());
  }, [refetchEntries, reloadExperiments]);

  const openFilePicker = useCallback(() => {
    uploadPanelRef.current?.openFilePicker();
  }, []);

  const handleUploaded = useCallback(
    (asset: AssetDetail, batch: ImportBatchPosition) => {
      const entry = entryFromUploadedAsset(asset);
      setJustImported((current) => {
        const fromThisBatch = batch.index === 1 ? [] : current;
        return [
          ...fromThisBatch.filter((pinned) => pinned.id !== entry.id),
          entry,
        ];
      });
      setPreviewRefreshKey((current) => current + 1);
      void refetchEntries();
      void reloadExperiments();
    },
    [refetchEntries, reloadExperiments]
  );

  // The whole page is a drop target, not just the panel: a user aiming at a
  // 60 px strip and missing should not have the browser navigate away from the
  // application and render their TIFF as a document.
  const handlePageDragEnter = (event: React.DragEvent) => {
    if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
    event.preventDefault();
    pageDragDepthRef.current += 1;
    setPageDragActive(true);
  };

  const handlePageDragOver = (event: React.DragEvent) => {
    if (!Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  };

  const handlePageDragLeave = () => {
    pageDragDepthRef.current = Math.max(0, pageDragDepthRef.current - 1);
    if (pageDragDepthRef.current === 0) setPageDragActive(false);
  };

  const handlePageDrop = (event: React.DragEvent) => {
    pageDragDepthRef.current = 0;
    setPageDragActive(false);
    const dropped = event.dataTransfer?.files;
    if (!dropped || dropped.length === 0) return;
    event.preventDefault();
    uploadPanelRef.current?.acceptDroppedFiles(dropped);
  };

  return (
    <div
      className="min-h-screen px-5 py-5 text-slate-900 lg:px-8"
      onDragEnter={handlePageDragEnter}
      onDragOver={handlePageDragOver}
      onDragLeave={handlePageDragLeave}
      onDrop={handlePageDrop}
    >
      <div className="mx-auto flex max-w-[1500px] flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-4 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
          <h1 className="text-2xl font-semibold tracking-normal text-slate-950">
            QuantEM
          </h1>
          <div className="flex flex-wrap items-center gap-2">
            {/* No import button here, by owner ruling: the import panel sits
                directly below this row, always visible, and takes both a click
                and a dropped file. A second entry point in the top bar competes
                with the thing it points at. The empty state keeps its button --
                there is no library to look at there, so the panel is the only
                thing on screen and pointing at it is useful. */}
            <Button
              onClick={() => {
                if (showHelp) dismissWorkflowGuide();
                else setShowHelp(true);
              }}
              aria-expanded={showHelp}
            >
              {showHelp ? "Hide guide" : "How this works"}
            </Button>
            <FineTuneMenuButton />
            <Button onClick={() => setIsQueueSidebarOpen(true)}>Tasks & Queues</Button>
            <Button
              size="icon"
              variant="secondary"
              aria-label="Settings"
              title="Settings"
              onClick={() => setSettingsOpen(true)}
            >
              <svg
                className="h-5 w-5"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.83 2.83-.06-.06A1.7 1.7 0 0 0 15 19.4a1.7 1.7 0 0 0-1 .6 1.7 1.7 0 0 0-.4 1.1V21H9.6v-.1A1.7 1.7 0 0 0 8.6 19.4a1.7 1.7 0 0 0-1.88.34l-.06.06-2.83-2.83.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-.6-1 1.7 1.7 0 0 0-1.1-.4H3V9.6h.1A1.7 1.7 0 0 0 4.6 8.6a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.83-2.83.06.06A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-.6 1.7 1.7 0 0 0 .4-1.1V3h4v.1A1.7 1.7 0 0 0 15.4 4.6a1.7 1.7 0 0 0 1.88-.34l.06-.06 2.83 2.83-.06.06A1.7 1.7 0 0 0 19.4 9c.38.27.71.62.88 1.08.16.43.21.91.12 1.37v1.1c.09.46.04.94-.12 1.37A2.7 2.7 0 0 1 19.4 15Z" />
              </svg>
            </Button>
          </div>
        </header>

        {showHelp ? <WorkflowGuide onDismiss={dismissWorkflowGuide} /> : null}

        <ImageUploadPanel
          ref={uploadPanelRef}
          pageDragActive={pageDragActive}
          compact={experiments.length > 0}
          onUploaded={handleUploaded}
        />

        <Panel className="flex flex-wrap items-end justify-between gap-4 p-4">
          <div className="min-w-[240px] flex-1">
            <label
              className="block text-xs font-semibold uppercase tracking-wide text-slate-500"
              htmlFor="library-search"
            >
              Search
            </label>
            <input
              id="library-search"
              className="mt-2 h-10 w-full rounded-md border border-slate-300 bg-white px-3 text-sm text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Name or filename"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              className="h-9 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              value={sortField}
              aria-label="Sort field"
              onChange={(event) => setSortField(event.target.value as SortField)}
            >
              {SORT_FIELDS.map((field) => (
                <option key={field} value={field}>
                  {sortFieldLabel(field)}
                </option>
              ))}
            </select>
            <select
              className="h-9 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
              value={sortDirection}
              aria-label="Sort direction"
              onChange={(event) =>
                setSortDirection(event.target.value as SortDirection)
              }
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </select>
            {/* Only once there is something to filter by. See `hasGroups`. */}
            {hasGroups ? (
              <>
                <select
                  className="h-9 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  value={experimentFilter}
                  aria-label="Experiment"
                  onChange={(event) => {
                    setExperimentFilter(event.target.value);
                    // A dataset from the experiment being filtered away would
                    // silently return nothing, which reads as an empty library.
                    setDatasetFilter("");
                  }}
                >
                  <option value="">All experiments</option>
                  {experiments.map((experiment) => (
                    <option key={experiment.id} value={experiment.id}>
                      {experiment.name} ({experiment.asset_count})
                    </option>
                  ))}
                </select>
                <select
                  className="h-9 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  value={datasetFilter}
                  aria-label="Dataset"
                  onChange={(event) => setDatasetFilter(event.target.value)}
                >
                  <option value="">All datasets</option>
                  {datasetOptions.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name} ({dataset.asset_count})
                    </option>
                  ))}
                  <option value={UNASSIGNED_FILTER}>Not in a dataset</option>
                </select>
                {/* The only way to undo a typed name, offered next to the
                    experiment it would remove rather than on a screen of its
                    own. Nothing here renames: that is not what was asked for,
                    and an unused rename box is another control to get wrong. */}
                {filteredExperiment ? (
                  <Button
                    onClick={() => setDeleteConfirmExperiment(filteredExperiment)}
                  >
                    Delete experiment
                  </Button>
                ) : null}
              </>
            ) : null}
            <Button
              aria-pressed={selecting}
              onClick={() => {
                setSelecting((current) => !current);
                setSelectedIds(new Set());
              }}
            >
              {selecting ? "Done selecting" : "Select images"}
            </Button>
          </div>
        </Panel>

        {/* Appears only once something is selected, so the ordinary library is
            unchanged. Where images are put into an experiment after the fact,
            which is every image that already exists. */}
        {selecting && selectedEntries.length > 0 ? (
          <LibrarySelectionBar
            selected={selectedEntries}
            experiments={experiments}
            onApplied={handleGroupingApplied}
            onClearSelection={() => setSelectedIds(new Set())}
            onSelectAllShown={() =>
              setSelectedIds(new Set(previewEntries.map((entry) => entry.id)))
            }
            shownCount={previewEntries.length}
          />
        ) : null}
        {selecting && selectedEntries.length === 0 ? (
          <p className="m-0 text-sm text-slate-700">
            Tick the images you want to put in an experiment or a dataset.
          </p>
        ) : null}

        {entriesLoading ? <PageState title="Loading images..." /> : null}
        {entriesError ? (
          <PageState
            title="Error loading images"
            detail={extractApiErrorMessage(
              entriesError,
              "The library could not be loaded."
            )}
            tone="error"
          />
        ) : null}

        {!entriesLoading && !entriesError ? (
          !hasAnyImages ? (
            <PageState
              title={
                search
                  ? "No images match that search"
                  : filtered
                    ? "No images in this part of the library"
                    : "No images yet"
              }
              detail={
                search ? (
                  "Clear the search box to see the whole library."
                ) : filtered ? (
                  // Not "no images yet": the library may be full, and offering
                  // an Import button here would answer a question nobody asked.
                  "Set the experiment and dataset back to All to see the whole library."
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <p className="m-0">
                      QuantEM works on images you import from this machine. TIFF
                      and PNG are accepted.
                    </p>
                    <Button variant="primary" onClick={openFilePicker}>
                      Import an image
                    </Button>
                  </div>
                )
              }
            />
          ) : (
            <div className="flex flex-col gap-5">
              {shownExperiments.map((experiment) => (
                <ExperimentPreviewPanel
                  key={experiment.id}
                  experimentId={experiment.id}
                  name={experiment.name}
                  search={search}
                  dataset={datasetFilter}
                  ordering={serverOrderingFor(sortField, sortDirection)}
                  refreshKey={previewRefreshKey}
                  onDelete={setDeleteConfirmImage}
                  deleting={deleting}
                  highlightedIds={pinnedImportIds}
                  selecting={selecting}
                  selectedIds={selectedIds}
                  pinnedEntries={
                    pinnedEntriesByExperiment[experiment.id] ?? NO_PINNED_ENTRIES
                  }
                  onToggleSelect={toggleSelect}
                  onEntriesLoaded={handlePreviewEntriesLoaded}
                />
              ))}
            </div>
          )
        ) : null}

        {!entriesLoading && !entriesError ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            {/* slate-600: slate-500 on white is 4.48:1, just under AA. */}
            <p className="text-sm text-slate-600">
              {/* `max` because the pinned import is on screen a beat before it
                  is in the server's count, and "Showing 1 of 0" is nonsense. */}
              {Math.max(entryTotal, justImported.length)} images
              {/* Said in one line instead of replacing the whole grid with a
                  loading state every three seconds. */}
              {entriesRefetching ? " · updating…" : ""}
            </p>
            <Button onClick={() => setPreviewRefreshKey((current) => current + 1)}>
              Refresh previews
            </Button>
          </div>
        ) : null}

        <ConfirmDialog
          isOpen={deleteConfirmImage !== null}
          title="Delete Image"
          message={
            deleteConfirmImage
              ? `Are you sure you want to delete "${deleteConfirmImage.display_name}"? This will permanently delete the image and all associated files (PNG, NGFF, overlays, probability maps, and segmentations). This action cannot be undone.`
              : ""
          }
          confirmText="Delete"
          cancelText="Cancel"
          onConfirm={() => {
            if (deleteConfirmImage) void deleteImageMutation(deleteConfirmImage.id);
          }}
          onCancel={() => setDeleteConfirmImage(null)}
        />
        {/* "Delete" over a group of images reads as "delete the images", and
            here it does not: each image moves into its own experiment first.
            The count makes that reassurance concrete. */}
        <ConfirmDialog
          isOpen={deleteConfirmExperiment !== null}
          title="Delete experiment"
          message={
            deleteConfirmExperiment
              ? `Delete the experiment "${deleteConfirmExperiment.name}"?`
              : ""
          }
          details={
            deleteConfirmExperiment ? (
              <p className="m-0">
                {describeExperimentDeletion(deleteConfirmExperiment)}
              </p>
            ) : null
          }
          confirmText="Delete experiment"
          cancelText="Cancel"
          onConfirm={() => {
            const doomed = deleteConfirmExperiment;
            setDeleteConfirmExperiment(null);
            if (!doomed) return;
            void deleteExperiment(doomed.id)
              .catch(() => {
                // The catalogue reload below shows whether it went; an error
                // toast over an optional organising action is not worth a
                // banner across the library.
              })
              .finally(() => {
                setExperimentFilter("");
                setDatasetFilter("");
                setPreviewRefreshKey((current) => current + 1);
                void reloadExperiments();
                void refetchEntries();
              });
          }}
          onCancel={() => setDeleteConfirmExperiment(null)}
        />
        <JobQueueSidebar
          isOpen={isQueueSidebarOpen}
          onClose={() => setIsQueueSidebarOpen(false)}
        />
        <SettingsDialog
          isOpen={settingsOpen}
          status={systemStatus}
          statusError={systemStatusError}
          onClose={() => setSettingsOpen(false)}
          onRetryStatus={() => void refetchSystemStatus()}
        />
      </div>
    </div>
  );
}
