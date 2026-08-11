import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  deleteAsset,
  deleteExperiment,
  getAsset,
  getHomeEntryPage,
  type AssetOrdering,
} from "@/shared/api/assets";
import type { Experiment } from "@/shared/types/common";
import { getSystemStatus } from "@/shared/api/jobs";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { FineTuneMenuButton } from "@/features/finetune/FineTuneMenuButton";
import { ImageCard } from "@/features/library/components/ImageCard";
import {
  isLibraryEntryUnfinished,
} from "@/features/library/components/imageCardUtils";
import {
  ImageUploadPanel,
  type ImageUploadPanelHandle,
  type ImportBatchPosition,
} from "@/features/library/components/ImageUploadPanel";
import { ImportConfirmation } from "@/features/library/components/ImportConfirmation";
import { JobQueueSidebar } from "@/features/library/components/JobQueueSidebar";
import { WorkflowGuide } from "@/features/library/components/WorkflowGuide";
import { hasSeenWorkflowGuide } from "@/features/library/components/workflowGuideStorage";
import { useExperiments } from "@/features/library/components/grouping/useExperiments";
import { groupEntriesByDataset } from "@/features/library/components/grouping/groupEntries";
import { LibrarySelectionBar } from "@/features/library/components/grouping/LibrarySelectionBar";
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
 * nothing. Nothing else is lost — every card states its own status, a failed
 * import is named in the confirmation strip, and Tasks & Queues lists the work.
 *
 * A stored preference of `"status"` from before this change fails
 * `isSortField` and falls back to `created_at`, which is the default anyway.
 */
type SortField = "display_name" | "created_at" | "updated_at";
type SortDirection = "asc" | "desc";

/**
 * Seconds between "your image is ready" and the library opening it.
 *
 * The navigation itself is what the owner asked for (ask #3). The countdown is
 * what makes it stoppable: on a 475 MP image the pyramid finishes ~100 s after
 * the import, which is long enough for the user to have started doing something
 * else on this screen, and the route change used to arrive with no warning at
 * all.
 */
const AUTO_OPEN_COUNTDOWN_SECONDS = 5;

interface StoredLibraryControls {
  search?: string;
  sortField?: SortField;
  sortDirection?: SortDirection;
}

const LIBRARY_CONTROLS_STORAGE_KEY = "quantem-library-controls-v1";
const LIBRARY_PAGE_SIZE = 60;
/**
 * Card grid geometry, in px. Row virtualisation keys off these.
 *
 * `CARD_HEIGHT` is a hard constant because absolutely-positioned virtual rows
 * need one, and the card is `overflow: hidden`, so anything the card's content
 * adds beyond it is silently cut off. That is how the status and pixel-size
 * badges -- the row that says whether an image is calibrated, which is what the
 * rest of the app depends on -- ended up sliced off the bottom of every card
 * with a two-line name.
 *
 * `ImageCard` now guarantees it fits: the text block is `shrink-0` and the
 * thumbnail is `flex-1`, so the preview absorbs the slack and the badges cannot
 * be pushed out. The height below is chosen so the preview still gets a
 * sensible share of the card at the widest text block the card can produce
 * (two-line name, one-line filename, dimensions, one badge row = 134 px):
 * 336 - 134 - 2 (border) = 200 px of preview at 258 px wide, a touch taller
 * than the 4:3 it used to be.
 */
const CARD_WIDTH = 260;
const CARD_HEIGHT = 336;
const CARD_GAP = 16;
const GRID_OVERSCAN_ROWS = 2;
const SORT_FIELDS: SortField[] = ["display_name", "created_at", "updated_at"];
const SORT_DIRECTIONS: SortDirection[] = ["asc", "desc"];

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
 * "Its 1 image stays ... and become unassigned" is what fragment assembly
 * produces, and a reassurance that reads as broken English is not reassuring.
 *
 * The promise it makes is real. `Asset.experiment` is `SET_NULL`, so the images
 * survive and become unassigned; the datasets do not, because a dataset cannot
 * exist outside an experiment.
 */
function describeExperimentDeletion(experiment: Experiment): string {
  const images =
    experiment.asset_count === 0
      ? "It holds no images."
      : experiment.asset_count === 1
        ? "Its 1 image stays in the library and becomes unassigned."
        : `Its ${experiment.asset_count} images stay in the library and become unassigned.`;
  const datasets =
    experiment.datasets.length === 0
      ? ""
      : experiment.datasets.length === 1
        ? " Its 1 dataset goes with it, because a dataset cannot exist outside an experiment."
        : ` Its ${experiment.datasets.length} datasets go with it, because a dataset cannot exist outside an experiment.`;
  return `${images}${datasets} No image files are deleted.`;
}

function compareText(left: string, right: string): number {
  return left.localeCompare(right, undefined, { sensitivity: "base" });
}

function compareEntries(
  left: HomeEntry,
  right: HomeEntry,
  sortField: SortField,
  sortDirection: SortDirection
): number {
  let comparison = 0;
  if (sortField === "created_at" || sortField === "updated_at") {
    comparison = Date.parse(left[sortField]) - Date.parse(right[sortField]);
  } else {
    comparison = compareText(left.display_name, right.display_name);
  }
  if (comparison === 0) comparison = compareText(left.display_name, right.display_name);
  return sortDirection === "asc" ? comparison : -comparison;
}

/** Fires `onLoadMore` when the sentinel scrolls near the viewport. */
function useLoadMoreOnIntersect<T extends HTMLElement>(
  enabled: boolean,
  onLoadMore: () => void
) {
  const ref = useRef<T | null>(null);
  const onLoadMoreRef = useRef(onLoadMore);

  useEffect(() => {
    onLoadMoreRef.current = onLoadMore;
  }, [onLoadMore]);

  useEffect(() => {
    if (!enabled || typeof IntersectionObserver === "undefined") return undefined;
    const node = ref.current;
    if (!node) return undefined;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry?.isIntersecting) {
          onLoadMoreRef.current();
        }
      },
      { rootMargin: "700px 0px" }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [enabled]);

  return ref;
}

/**
 * Row-virtualised card grid: only the rows intersecting the scroll viewport
 * (plus an overscan) are mounted, so a library of thousands of images still
 * renders a bounded number of thumbnails.
 */
function ImageGrid({
  images,
  onOpen,
  onDelete,
  deleting,
  highlightedIds,
  selecting = false,
  selectedIds,
  onToggleSelect,
}: {
  images: HomeEntry[];
  onOpen: (assetId: string) => void;
  onDelete: (image: HomeEntry) => void;
  deleting: boolean;
  /** The images the import confirmation is pointing at. */
  highlightedIds: Set<string>;
  /** The library is in selecting mode, so every card carries a tick box. */
  selecting?: boolean;
  selectedIds?: Set<string>;
  onToggleSelect?: (assetId: string, selected: boolean) => void;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);
  const [gridTop, setGridTop] = useState(0);

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      setContainerWidth(node.clientWidth);
    });
    observer.observe(node);
    setContainerWidth(node.clientWidth);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const readViewport = () => {
      setScrollTop(window.scrollY);
      setViewportHeight(window.innerHeight);
      const node = containerRef.current;
      setGridTop(node ? node.getBoundingClientRect().top + window.scrollY : 0);
    };
    readViewport();
    window.addEventListener("scroll", readViewport, { passive: true });
    window.addEventListener("resize", readViewport);
    return () => {
      window.removeEventListener("scroll", readViewport);
      window.removeEventListener("resize", readViewport);
    };
  }, []);

  const columns = Math.max(
    1,
    Math.floor((containerWidth + CARD_GAP) / (CARD_WIDTH + CARD_GAP)) || 1
  );
  const rowHeight = CARD_HEIGHT + CARD_GAP;
  const rowCount = Math.ceil(images.length / columns);
  const firstVisibleRow = Math.max(
    0,
    Math.floor((scrollTop - gridTop) / rowHeight) - GRID_OVERSCAN_ROWS
  );
  const lastVisibleRow = Math.min(
    rowCount,
    Math.ceil((scrollTop - gridTop + viewportHeight) / rowHeight) +
      GRID_OVERSCAN_ROWS
  );
  const visibleRows: number[] = [];
  for (let row = firstVisibleRow; row < lastVisibleRow; row += 1) {
    visibleRows.push(row);
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full"
      style={{ height: Math.max(0, rowCount * rowHeight - CARD_GAP) }}
    >
      {visibleRows.map((row) => (
        <div
          key={row}
          className="absolute left-0 right-0 flex gap-4"
          style={{ top: row * rowHeight, height: CARD_HEIGHT }}
        >
          {images
            .slice(row * columns, row * columns + columns)
            .map((image) => (
              <div key={image.id} style={{ width: CARD_WIDTH }}>
                <ImageCard
                  image={image}
                  onOpen={onOpen}
                  onDelete={onDelete}
                  deleting={deleting}
                  justImported={highlightedIds.has(image.id)}
                  selectable={selecting}
                  selected={selectedIds?.has(image.id) ?? false}
                  onToggleSelect={onToggleSelect}
                />
              </div>
            ))}
        </div>
      ))}
    </div>
  );
}

export function LibraryPage() {
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
  const [hasMoreEntries, setHasMoreEntries] = useState(false);
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
  const [entriesLoadingMore, setEntriesLoadingMore] = useState(false);
  const [entriesError, setEntriesError] = useState<Error | null>(null);
  const [deleteConfirmImage, setDeleteConfirmImage] = useState<HomeEntry | null>(null);
  const [deleteConfirmExperiment, setDeleteConfirmExperiment] =
    useState<Experiment | null>(null);
  const [isQueueSidebarOpen, setIsQueueSidebarOpen] = useState(false);
  const [showHelp, setShowHelp] = useState(() => !hasSeenWorkflowGuide());
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
  const [autoOpenImport, setAutoOpenImport] = useState(false);
  const [autoOpenCountdown, setAutoOpenCountdown] = useState<number | null>(null);
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
  const [experimentFilter, setExperimentFilter] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("");
  const [groupByDataset, setGroupByDataset] = useState(false);
  /** Selecting mode, and what is selected in it. Off until asked for. */
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const { experiments, reload: reloadExperiments } = useExperiments();
  const entriesRequestIdRef = useRef(0);
  const entriesInFlightKeyRef = useRef<string | null>(null);
  /** Read inside `loadEntryPage` without making it depend on the entry list. */
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
  const loadEntryPageRef = useRef<
    ((offset: number, mode: "replace" | "append") => Promise<void>) | null
  >(null);
  const uploadPanelRef = useRef<ImageUploadPanelHandle | null>(null);
  const pageDragDepthRef = useRef(0);
  const { setSelectedAssetId } = useSelectionStore();
  const navigate = useNavigate();

  const { data: systemStatus } = useApiQuery(() => getSystemStatus(), []);

  useEffect(() => {
    saveStoredLibraryControls({ search, sortField, sortDirection });
  }, [search, sortDirection, sortField]);

  const entryQueryParams = useMemo(
    () => ({
      search,
      availability: "local" as const,
      // Never omitted. See `serverOrderingFor`.
      ordering: serverOrderingFor(sortField, sortDirection),
      // Filtered by the server, not by the page: the page holds one window of
      // sixty rows, and narrowing a window is not narrowing a library. This is
      // the same defect the sort control had -- see `SortField`.
      ...(experimentFilter ? { experiment: experimentFilter } : {}),
      ...(datasetFilter ? { dataset: datasetFilter } : {}),
    }),
    [datasetFilter, experimentFilter, search, sortDirection, sortField]
  );

  const loadEntryPage = useCallback(
    async (offset: number, mode: "replace" | "append") => {
      const requestKey = JSON.stringify({ mode, offset, entryQueryParams });
      if (entriesInFlightKeyRef.current === requestKey) return;
      entriesInFlightKeyRef.current = requestKey;
      const requestId = ++entriesRequestIdRef.current;
      if (mode === "replace") {
        // The grid only disappears when there is no grid: a refetch over cards
        // that are already on screen refreshes them in place.
        if (entriesRef.current.length === 0) {
          setEntriesLoading(true);
        } else {
          setEntriesRefetching(true);
        }
      } else {
        setEntriesLoadingMore(true);
      }
      setEntriesError(null);
      try {
        const page = await getHomeEntryPage({
          ...entryQueryParams,
          limit: LIBRARY_PAGE_SIZE,
          offset,
        });
        if (requestId !== entriesRequestIdRef.current) return;
        setEntries((current) =>
          mode === "replace" ? page.results : [...current, ...page.results]
        );
        setEntryTotal(page.total);
        setHasMoreEntries(page.has_more);
      } catch (err) {
        if (requestId !== entriesRequestIdRef.current) return;
        setEntriesError(err instanceof Error ? err : new Error("Unknown error"));
        if (mode === "replace") {
          setEntries([]);
          setEntryTotal(0);
          setHasMoreEntries(false);
        }
      } finally {
        if (entriesInFlightKeyRef.current === requestKey) {
          entriesInFlightKeyRef.current = null;
        }
        if (requestId === entriesRequestIdRef.current) {
          if (mode === "replace") {
            setEntriesLoading(false);
            setEntriesRefetching(false);
          } else {
            setEntriesLoadingMore(false);
          }
        }
        // A refetch that collided with this one. Run it now rather than
        // dropping it on the floor.
        if (refetchPendingRef.current) {
          refetchPendingRef.current = false;
          void loadEntryPageRef.current?.(0, "replace");
        }
      }
    },
    [entryQueryParams]
  );

  useEffect(() => {
    entriesRef.current = entries;
  }, [entries]);

  useEffect(() => {
    loadEntryPageRef.current = loadEntryPage;
  }, [loadEntryPage]);

  useEffect(() => {
    void loadEntryPage(0, "replace");
  }, [loadEntryPage]);

  const refetchEntries = useCallback(async () => {
    if (entriesInFlightKeyRef.current !== null) {
      refetchPendingRef.current = true;
      return;
    }
    await loadEntryPage(0, "replace");
  }, [loadEntryPage]);

  const handleLoadMore = useCallback(() => {
    if (!hasMoreEntries || entriesLoadingMore) return;
    void loadEntryPage(entries.length, "append");
  }, [entries.length, entriesLoadingMore, hasMoreEntries, loadEntryPage]);

  const loadMoreRef = useLoadMoreOnIntersect<HTMLDivElement>(
    !entriesLoading && !entriesLoadingMore && hasMoreEntries,
    handleLoadMore
  );

  const { mutate: deleteImageMutation, loading: deleting } = useApiMutation(
    (assetId: string) => deleteAsset(assetId),
    {
      onSuccess: () => {
        setDeleteConfirmImage(null);
        void refetchEntries();
      },
    }
  );

  const openImage = useCallback(
    (assetId: string) => {
      setSelectedAssetId(assetId);
      navigate(`/assets/${assetId}/viewer`);
    },
    [navigate, setSelectedAssetId]
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
          // A failed status poll is not a reason to tear the confirmation
          // down; the next sweep tries again and the card still says what it
          // last knew.
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
   * The announced hand-off into the viewer.
   *
   * This is owner ask #3 and it is kept — but the app used to perform it
   * silently, ~100 s after the import, from an effect the user had no way to
   * see or stop. Now the strip has said "I will open it here when it is ready"
   * for the whole wait, and the last five seconds are counted out loud with a
   * "Stay in the library" button beside them.
   */
  const soleImport = justImported.length === 1 ? justImported[0] : null;
  const importIsReady = Boolean(soleImport?.ngff_ready);
  useEffect(() => {
    if (!autoOpenImport || !importIsReady) {
      setAutoOpenCountdown(null);
      return undefined;
    }
    setAutoOpenCountdown(AUTO_OPEN_COUNTDOWN_SECONDS);
    const intervalId = window.setInterval(() => {
      setAutoOpenCountdown((current) => (current === null ? null : current - 1));
    }, 1000);
    return () => clearInterval(intervalId);
  }, [autoOpenImport, importIsReady]);

  useEffect(() => {
    if (autoOpenCountdown === null || autoOpenCountdown > 0) return;
    const assetId = soleImport?.id;
    setAutoOpenCountdown(null);
    setAutoOpenImport(false);
    if (assetId) openImage(assetId);
  }, [autoOpenCountdown, openImage, soleImport]);

  const hasUnfinishedEntries = useMemo(
    () => entries.some(isLibraryEntryUnfinished),
    [entries]
  );

  // Keep per-image status fresh while anything is still being prepared.
  useEffect(() => {
    if (!hasUnfinishedEntries) return undefined;
    const intervalId = window.setInterval(() => {
      void refetchEntries();
    }, 3000);
    return () => clearInterval(intervalId);
  }, [hasUnfinishedEntries, refetchEntries]);

  /**
   * The grid: the session's import first, then the library.
   *
   * Pinning is not cosmetic. It is the only construction under which "your
   * image is the first card" is true for every sort, every page and every
   * library size — which is the guarantee the confirmation strip is making one
   * line above it.
   */
  const pinnedImportIds = useMemo(
    () => new Set(justImported.map((entry) => entry.id)),
    [justImported]
  );
  const visibleImages = useMemo(() => {
    const sorted = [...entries].sort((left, right) =>
      compareEntries(left, right, sortField, sortDirection)
    );
    if (justImported.length === 0) return sorted;
    return [
      ...justImported,
      ...sorted.filter((entry) => !pinnedImportIds.has(entry.id)),
    ];
  }, [entries, justImported, pinnedImportIds, sortDirection, sortField]);

  /**
   * The grouping controls only exist once there is something to group by.
   *
   * An unorganised library is a legitimate steady state, not an unfinished
   * setup, so it gets no filter it cannot use and no prompt to go and organise
   * something. The place to make the first experiment is the import form, where
   * the user already knows what the images are.
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

  const selectedEntries = useMemo(
    () => visibleImages.filter((entry) => selectedIds.has(entry.id)),
    [selectedIds, visibleImages]
  );

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
    setSelectedIds(new Set());
  }, [refetchEntries, reloadExperiments]);

  const groups = useMemo(
    () => (groupByDataset ? groupEntriesByDataset(visibleImages) : []),
    [groupByDataset, visibleImages]
  );

  const handleImageClick = openImage;

  const openFilePicker = useCallback(() => {
    uploadPanelRef.current?.openFilePicker();
  }, []);

  const handleUploaded = useCallback(
    (asset: AssetDetail, batch: ImportBatchPosition) => {
      const entry = entryFromUploadedAsset(asset);
      setJustImported((current) => [
        ...current.filter((pinned) => pinned.id !== entry.id),
        entry,
      ]);
      // Owner ask #3 -- land in the workspace -- is about *the* image. There is
      // no defensible answer to which of forty a batch should open, and
      // navigating away while the other thirty-nine are still uploading would
      // interrupt the very queue the user is watching. So a batch pins its
      // cards and stays put.
      setAutoOpenImport(batch.total === 1);
      setAutoOpenCountdown(null);
      void refetchEntries();
    },
    [refetchEntries]
  );

  const dismissImportConfirmation = useCallback(() => {
    setJustImported([]);
    setAutoOpenImport(false);
    setAutoOpenCountdown(null);
  }, []);

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
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">
              QuantEM
            </p>
            <h1 className="text-2xl font-semibold tracking-normal text-slate-950">
              Library
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {/* No import button here, by owner ruling: the import panel sits
                directly below this row, always visible, and takes both a click
                and a dropped file. A second entry point in the top bar competes
                with the thing it points at. The empty state keeps its button --
                there is no library to look at there, so the panel is the only
                thing on screen and pointing at it is useful. */}
            {systemStatus ? (
              // Not a warning. CPU-only is a fully supported configuration --
              // every released model runs on CPU -- and an amber badge on the
              // first screen reads as a broken install. It is a fact about
              // speed, so it is stated as one.
              <Badge
                tone={systemStatus.cuda_available ? "good" : "default"}
                title={
                  systemStatus.cuda_available
                    ? "A CUDA GPU was found; segmentation runs on it."
                    : "No CUDA GPU found. Everything works on CPU, just more slowly."
                }
              >
                {systemStatus.cuda_available ? "GPU: CUDA" : "Running on CPU"}
              </Badge>
            ) : null}
            <Button
              onClick={() => setShowHelp((current) => !current)}
              aria-expanded={showHelp}
            >
              {showHelp ? "Hide guide" : "How this works"}
            </Button>
            {/* The only route to the models screen for a user who never opens
                the fine-tuning wizard, which is where the catalogue used to be
                reachable from and nowhere else. */}
            <Link
              className="inline-flex h-10 items-center rounded-md border border-slate-300 bg-white px-4 text-sm font-medium text-slate-800 shadow-sm hover:border-slate-400 hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2"
              to="/models"
            >
              Models
            </Link>
            <FineTuneMenuButton />
            <Button onClick={() => setIsQueueSidebarOpen(true)}>Tasks & Queues</Button>
          </div>
        </header>

        {showHelp ? <WorkflowGuide onDismiss={() => setShowHelp(false)} /> : null}

        <ImageUploadPanel
          ref={uploadPanelRef}
          pageDragActive={pageDragActive}
          onUploaded={handleUploaded}
        />

        {justImported.length > 0 ? (
          <ImportConfirmation
            entries={justImported}
            autoOpen={autoOpenImport}
            countdownSeconds={autoOpenCountdown}
            onOpenNow={() => {
              if (!soleImport) return;
              if (soleImport.ngff_ready) {
                setAutoOpenCountdown(null);
                setAutoOpenImport(false);
                openImage(soleImport.id);
              } else {
                // Not openable yet — this is a promise to open it, which is
                // exactly what the auto-open flag is.
                setAutoOpenImport(true);
              }
            }}
            onStayHere={() => {
              setAutoOpenImport(false);
              setAutoOpenCountdown(null);
            }}
            onDismiss={dismissImportConfirmation}
          />
        ) : null}

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
                  {/* Not an experiment with a blank name: the images in none. */}
                  <option value={UNASSIGNED_FILTER}>Not in an experiment</option>
                </select>
                <select
                  className="h-9 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-cyan-500"
                  value={datasetFilter}
                  aria-label="Dataset"
                  disabled={experimentFilter === UNASSIGNED_FILTER}
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
                <label className="flex items-center gap-2 text-sm text-slate-800">
                  <input
                    type="checkbox"
                    checked={groupByDataset}
                    onChange={(event) => setGroupByDataset(event.target.checked)}
                  />
                  Group by dataset
                </label>
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
              setSelectedIds(new Set(visibleImages.map((entry) => entry.id)))
            }
            shownCount={visibleImages.length}
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
          visibleImages.length === 0 ? (
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
          ) : groupByDataset ? (
            // One grid per section. Each `ImageGrid` measures its own offset,
            // so the row virtualisation keeps working section by section.
            <div className="flex flex-col gap-6">
              {groups.map((group) => (
                <section key={group.key} className="flex flex-col gap-2">
                  <h2 className="m-0 text-sm font-semibold text-slate-900">
                    {group.label}
                    {group.sublabel ? (
                      <span className="font-normal text-slate-600">
                        {" "}
                        · {group.sublabel}
                      </span>
                    ) : null}
                    <span className="font-normal text-slate-600">
                      {" "}
                      · {group.entries.length}
                    </span>
                  </h2>
                  <ImageGrid
                    images={group.entries}
                    onOpen={handleImageClick}
                    onDelete={setDeleteConfirmImage}
                    deleting={deleting}
                    highlightedIds={pinnedImportIds}
                    selecting={selecting}
                    selectedIds={selectedIds}
                    onToggleSelect={toggleSelect}
                  />
                </section>
              ))}
              {/* The sections cover what has been fetched, and the footer says
                  how much that is. Said here rather than left to be inferred
                  from a count that stops short of the library total. */}
              {hasMoreEntries ? (
                <p className="m-0 text-sm text-slate-600">
                  These sections cover the images loaded so far. Load more to
                  fill them in.
                </p>
              ) : null}
            </div>
          ) : (
            <ImageGrid
              images={visibleImages}
              onOpen={handleImageClick}
              onDelete={setDeleteConfirmImage}
              deleting={deleting}
              highlightedIds={pinnedImportIds}
              selecting={selecting}
              selectedIds={selectedIds}
              onToggleSelect={toggleSelect}
            />
          )
        ) : null}

        {!entriesLoading && !entriesError ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            {/* slate-600: slate-500 on white is 4.48:1, just under AA. */}
            <p className="text-sm text-slate-600">
              {/* `max` because the pinned import is on screen a beat before it
                  is in the server's count, and "Showing 1 of 0" is nonsense. */}
              Showing {visibleImages.length} of{" "}
              {Math.max(entryTotal, visibleImages.length)} images
              {/* Said in one line instead of replacing the whole grid with a
                  loading state every three seconds. */}
              {entriesRefetching ? " · updating…" : ""}
            </p>
            <Button
              disabled={!hasMoreEntries || entriesLoadingMore}
              onClick={handleLoadMore}
            >
              {entriesLoadingMore
                ? "Loading..."
                : hasMoreEntries
                  ? "Load more"
                  : "All loaded"}
            </Button>
            <div ref={loadMoreRef} className="h-px w-full" aria-hidden="true" />
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
            here it does not: `Asset.experiment` is SET_NULL on purpose. The
            message says so with the count in it, because a number is what
            makes a reassurance believable. */}
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
      </div>
    </div>
  );
}
