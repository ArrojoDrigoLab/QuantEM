import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Link, useNavigate } from "react-router-dom";
import { deleteAsset, getAsset, getHomeEntryPage } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import { useApiMutation } from "@/shared/hooks/useApiMutation";
import { useApiQuery } from "@/shared/hooks/useApiQuery";
import { useSelectionStore } from "@/shared/stores/useSelectionStore";
import { ConfirmDialog } from "@/shared/ui/ConfirmDialog";
import { Badge, Button, PageState, Panel } from "@/shared/ui/design";
import { ImageCard } from "@/features/library/components/ImageCard";
import { isLibraryEntryProcessing } from "@/features/library/components/imageCardUtils";
import { ImageUploadPanel } from "@/features/library/components/ImageUploadPanel";
import { JobQueueSidebar } from "@/features/library/components/JobQueueSidebar";
import { WorkflowGuide } from "@/features/library/components/WorkflowGuide";
import { hasSeenWorkflowGuide } from "@/features/library/components/workflowGuideStorage";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { HomeEntry } from "@/shared/types/images";

type SortField = "display_name" | "created_at" | "updated_at" | "status";
type SortDirection = "asc" | "desc";

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
const SORT_FIELDS: SortField[] = [
  "display_name",
  "created_at",
  "updated_at",
  "status",
];
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
    status: "Status",
  }[field];
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
  } else if (sortField === "status") {
    comparison = compareText(left.preprocess_stage, right.preprocess_stage);
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
}: {
  images: HomeEntry[];
  onOpen: (assetId: string) => void;
  onDelete: (image: HomeEntry) => void;
  deleting: boolean;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(0);

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
  const gridTop = containerRef.current
    ? containerRef.current.getBoundingClientRect().top + scrollTop
    : 0;
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
  const [entriesLoading, setEntriesLoading] = useState(true);
  const [entriesLoadingMore, setEntriesLoadingMore] = useState(false);
  const [entriesError, setEntriesError] = useState<Error | null>(null);
  const [deleteConfirmImage, setDeleteConfirmImage] = useState<HomeEntry | null>(null);
  const [isQueueSidebarOpen, setIsQueueSidebarOpen] = useState(false);
  const [showHelp, setShowHelp] = useState(() => !hasSeenWorkflowGuide());
  // The import panel is collapsed by default, so "Import an image above" used
  // to point at a button. `importPanelKey` remounts the panel to force it open
  // when the empty state asks for it.
  const [importPanelOpen, setImportPanelOpen] = useState(false);
  const [importPanelKey, setImportPanelKey] = useState(0);
  const [pendingNavigateImageId, setPendingNavigateImageId] = useState<string | null>(
    null
  );
  const entriesRequestIdRef = useRef(0);
  const entriesInFlightKeyRef = useRef<string | null>(null);
  const { setSelectedAssetId } = useSelectionStore();
  const navigate = useNavigate();

  const { data: systemStatus } = useApiQuery(() => getSystemStatus(), []);

  useEffect(() => {
    saveStoredLibraryControls({ search, sortField, sortDirection });
  }, [search, sortDirection, sortField]);

  const entryQueryParams = useMemo(
    () => ({ search, availability: "local" as const }),
    [search]
  );

  const loadEntryPage = useCallback(
    async (offset: number, mode: "replace" | "append") => {
      const requestKey = JSON.stringify({ mode, offset, entryQueryParams });
      if (entriesInFlightKeyRef.current === requestKey) return;
      entriesInFlightKeyRef.current = requestKey;
      const requestId = ++entriesRequestIdRef.current;
      if (mode === "replace") {
        setEntriesLoading(true);
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
          } else {
            setEntriesLoadingMore(false);
          }
        }
      }
    },
    [entryQueryParams]
  );

  useEffect(() => {
    void loadEntryPage(0, "replace");
  }, [loadEntryPage]);

  const refetchEntries = useCallback(async () => {
    if (entriesInFlightKeyRef.current !== null) return;
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

  // A freshly imported image is only openable once its NGFF pyramid exists, so
  // poll until it is ready (or preprocessing gives up) and then jump into it.
  useEffect(() => {
    if (!pendingNavigateImageId) return undefined;
    let cancelled = false;
    const checkStatus = async () => {
      const image = await getAsset(pendingNavigateImageId);
      if (cancelled) return;
      if (image.ngff_ready) {
        setSelectedAssetId(image.id);
        setPendingNavigateImageId(null);
        navigate(`/assets/${image.id}/viewer`);
      } else if (
        image.preprocess_stage === "FAILED" ||
        image.preprocess_stage === "CANCELLED"
      ) {
        setPendingNavigateImageId(null);
      }
    };
    void checkStatus();
    const intervalId = window.setInterval(() => {
      void checkStatus();
    }, 1000);
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [pendingNavigateImageId, navigate, setSelectedAssetId]);

  const hasProcessingEntries = useMemo(
    () => entries.some(isLibraryEntryProcessing),
    [entries]
  );

  // Keep per-image status fresh while anything is still preprocessing.
  useEffect(() => {
    if (!hasProcessingEntries) return undefined;
    const intervalId = window.setInterval(() => {
      void refetchEntries();
    }, 3000);
    return () => clearInterval(intervalId);
  }, [hasProcessingEntries, refetchEntries]);

  const visibleImages = useMemo(
    () =>
      [...entries].sort((left, right) =>
        compareEntries(left, right, sortField, sortDirection)
      ),
    [entries, sortDirection, sortField]
  );

  const handleImageClick = useCallback(
    (assetId: string) => {
      setSelectedAssetId(assetId);
      navigate(`/assets/${assetId}/viewer`);
    },
    [navigate, setSelectedAssetId]
  );

  const openImportPanel = useCallback(() => {
    setImportPanelOpen(true);
    setImportPanelKey((current) => current + 1);
  }, []);

  return (
    <div className="min-h-screen px-5 py-5 text-slate-900 lg:px-8">
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
            <Button onClick={() => setIsQueueSidebarOpen(true)}>Tasks & Queues</Button>
          </div>
        </header>

        {showHelp ? <WorkflowGuide onDismiss={() => setShowHelp(false)} /> : null}

        <ImageUploadPanel
          key={importPanelKey}
          defaultExpanded={importPanelOpen}
          onUploaded={(image) => {
            void refetchEntries();
            setPendingNavigateImageId(image.id);
          }}
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
          </div>
        </Panel>

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
              title={search ? "No images match that search" : "No images yet"}
              detail={
                search ? (
                  "Clear the search box to see the whole library."
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <p className="m-0">
                      QuantEM works on images you import from this machine. TIFF
                      and PNG are accepted.
                    </p>
                    <Button variant="primary" onClick={openImportPanel}>
                      Import an image
                    </Button>
                  </div>
                )
              }
            />
          ) : (
            <ImageGrid
              images={visibleImages}
              onOpen={handleImageClick}
              onDelete={setDeleteConfirmImage}
              deleting={deleting}
            />
          )
        ) : null}

        {!entriesLoading && !entriesError ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            {/* slate-600: slate-500 on white is 4.48:1, just under AA. */}
            <p className="text-sm text-slate-600">
              Showing {visibleImages.length} of {entryTotal} images
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
        <JobQueueSidebar
          isOpen={isQueueSidebarOpen}
          onClose={() => setIsQueueSidebarOpen(false)}
        />
      </div>
    </div>
  );
}
