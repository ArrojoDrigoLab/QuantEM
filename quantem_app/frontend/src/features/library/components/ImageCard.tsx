import { useEffect, useId, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  getAssetNgffThumbnailUrl,
  getAssetPreviewThumbnailUrl,
} from "@/shared/api/assets";
import { resolveEntryPixelSize } from "@/shared/pixelSize";
import { cx } from "@/shared/ui/cx";
import { Badge, Button } from "@/shared/ui/design";
import { PixelSizeTag } from "@/shared/ui/PixelSize";
import type { HomeEntry } from "@/shared/types/images";
import { isLibraryEntryProcessing } from "@/features/library/components/imageCardUtils";

const NGFF_THUMBNAIL_CACHE_VERSION = "ngff-thumb-v2";
const PREVIEW_THUMBNAIL_CACHE_VERSION = "preview-thumb-v1";

export function ImageCard({
  image,
  onDelete,
  deleting = false,
  justImported = false,
  selectable = false,
  selected = false,
  onToggleSelect,
  useActionMenu = false,
  onEdit,
  onExport,
}: {
  image: HomeEntry;
  onDelete?: (image: HomeEntry) => void;
  deleting?: boolean;
  /** This image belongs to the latest import batch and is pinned/highlighted. */
  justImported?: boolean;
  /**
   * The library is in selecting mode, so this card carries a tick box.
   *
   * Off unless the user asked for it: a permanent checkbox on every card is a
   * bulk-editing screen, and this is a library. Nothing else about the card
   * changes -- the name still opens the image, delete is still where it was.
   */
  selectable?: boolean;
  selected?: boolean;
  onToggleSelect?: (assetId: string, selected: boolean) => void;
  /** Replace the direct trash button with an Edit/Delete overflow menu. */
  useActionMenu?: boolean;
  onEdit?: (image: HomeEntry) => void;
  onExport?: (image: HomeEntry) => void;
}) {
  const assetId = image.id;
  const [thumbnailFailed, setThumbnailFailed] = useState(false);
  const [actionsOpen, setActionsOpen] = useState(false);
  const actionsRef = useRef<HTMLDivElement | null>(null);
  const actionsMenuId = useId();
  const errorTooltipId = useId();
  useEffect(() => {
    setThumbnailFailed(false);
    setActionsOpen(false);
  }, [image.id, image.updated_at]);
  useEffect(() => {
    if (!actionsOpen) return undefined;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!actionsRef.current?.contains(event.target as Node)) {
        setActionsOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setActionsOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [actionsOpen]);
  const processing = isLibraryEntryProcessing(image);
  const failed =
    image.preprocess_stage === "FAILED" || image.preprocess_stage === "CANCELLED";
  // Prefer the dedicated PREVIEW rendition when it exists; otherwise fall back
  // to the on-the-fly NGFF thumbnail (the coarsest pyramid level, rendered on
  // demand).
  const thumbnailUrl = image.preview_thumbnail_url
    ? getAssetPreviewThumbnailUrl(
        assetId,
        `${image.updated_at}-${PREVIEW_THUMBNAIL_CACHE_VERSION}`
      )
    : getAssetNgffThumbnailUrl(
        assetId,
        `${image.updated_at}-${NGFF_THUMBNAIL_CACHE_VERSION}`
      );
  const showThumbnail = !thumbnailFailed;
  // Resolve list payloads through the same pixel-size path as detail screens;
  // the shared tag owns the formatting in both places.
  const pixelSize = resolveEntryPixelSize(image);
  return (
    <article
      className={cx(
        "relative flex h-full flex-col overflow-hidden rounded-lg border bg-white shadow-sm",
        justImported
          ? "border-cyan-500 ring-2 ring-cyan-500 ring-offset-2"
          : selected
            ? "border-cyan-600 ring-2 ring-cyan-600"
            : "border-slate-200"
      )}
    >
      {selectable ? (
        // Top left, opposite the delete button, so the two destructive-looking
        // controls are not adjacent. Labelled with the image's name because a
        // grid of sixty identical unlabelled tick boxes is unusable with a
        // screen reader.
        <label className="absolute left-2 top-2 z-10 flex h-8 w-8 items-center justify-center rounded-md border border-slate-300 bg-white/95 shadow-sm">
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={selected}
            aria-label={`Select ${image.display_name}`}
            onChange={(event) => onToggleSelect?.(assetId, event.target.checked)}
          />
        </label>
      ) : null}
      {onDelete && !useActionMenu ? (
        <Button
          className="absolute right-2 top-2 z-10 h-8 w-8 border-red-200 bg-white/95 text-red-700 shadow-sm hover:border-red-300 hover:bg-red-50"
          size="icon"
          variant="secondary"
          aria-label={`Delete ${image.display_name}`}
          title="Delete image"
          disabled={deleting}
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onDelete(image);
          }}
        >
          <svg
            aria-hidden="true"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path d="M3 6h18" />
            <path d="M8 6V4h8v2" />
            <path d="M19 6l-1 14H6L5 6" />
            <path d="M10 11v5" />
            <path d="M14 11v5" />
          </svg>
        </Button>
      ) : null}
      {useActionMenu && (onEdit || onExport || onDelete) ? (
        <div ref={actionsRef} className="absolute right-2 top-2 z-20">
          <Button
            className="h-8 w-8 bg-white/95 shadow-sm"
            size="icon"
            variant="secondary"
            aria-label={`Options for ${image.display_name}`}
            aria-expanded={actionsOpen}
            aria-haspopup="menu"
            aria-controls={actionsOpen ? actionsMenuId : undefined}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              setActionsOpen((current) => !current);
            }}
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
              <circle cx="12" cy="5" r="1.8" fill="currentColor" />
              <circle cx="12" cy="12" r="1.8" fill="currentColor" />
              <circle cx="12" cy="19" r="1.8" fill="currentColor" />
            </svg>
          </Button>
          {actionsOpen ? (
            <div
              id={actionsMenuId}
              className="absolute right-0 mt-1 w-28 overflow-hidden rounded-md border border-slate-200 bg-white py-1 shadow-lg"
              role="menu"
            >
              {onEdit ? (
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full px-3 py-2 text-left text-sm text-slate-800 hover:bg-slate-100"
                  onClick={(event) => {
                    event.stopPropagation();
                    setActionsOpen(false);
                    onEdit(image);
                  }}
                >
                  Edit
                </button>
              ) : null}
              {onExport ? (
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full px-3 py-2 text-left text-sm text-slate-800 hover:bg-slate-100"
                  onClick={(event) => {
                    event.stopPropagation();
                    setActionsOpen(false);
                    onExport(image);
                  }}
                >
                  Export
                </button>
              ) : null}
              {onDelete ? (
                <button
                  type="button"
                  role="menuitem"
                  className="block w-full px-3 py-2 text-left text-sm text-red-700 hover:bg-red-50"
                  disabled={deleting}
                  onClick={(event) => {
                    event.stopPropagation();
                    setActionsOpen(false);
                    onDelete(image);
                  }}
                >
                  Delete
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
      {processing ? (
        <span
          className="absolute bottom-3 right-3 z-10 inline-flex h-8 w-8 items-center justify-center rounded-full border border-cyan-200 bg-white/95 text-cyan-700 shadow-sm"
          role="status"
          aria-label={`Processing ${image.display_name}`}
          title="Processing image"
        >
          <svg
            className="h-4 w-4 animate-spin"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="9"
              stroke="currentColor"
              strokeWidth="3"
            />
            <path
              className="opacity-90"
              d="M21 12a9 9 0 0 0-9-9"
              stroke="currentColor"
              strokeLinecap="round"
              strokeWidth="3"
            />
          </svg>
        </span>
      ) : null}
      {failed ? (
        <span className="group absolute bottom-3 right-3 z-20">
          <button
            type="button"
            className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-red-200 bg-white/95 text-base font-bold text-red-700 shadow-sm"
            aria-label={`Import failed for ${image.display_name}`}
            aria-describedby={errorTooltipId}
          >
            !
          </button>
          <span
            id={errorTooltipId}
            role="tooltip"
            className="invisible absolute bottom-full right-0 mb-2 w-72 rounded-lg border border-red-200 bg-white p-3 text-left text-xs font-normal leading-5 text-slate-800 opacity-0 shadow-xl transition group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
          >
            <span className="block">
              {image.preprocess_error || "The image could not be processed."}
            </span>
            <span className="mt-2 block">
              Delete this image and try to re-upload it.
            </span>
          </span>
        </span>
      ) : null}
      {/* `flex-1 min-h-0`, not `aspect-[4/3]`.
          The grid is row-virtualised, so the card height is a constant the
          layout has to live inside. With an aspect-ratio thumbnail the card's
          natural height was whatever the thumbnail plus the text block happened
          to add up to, the article is `overflow-hidden`, and a two-line display
          name pushed the total past that constant -- so the row that got cut
          off was the last one, which is the status and pixel-size badges. The
          preview is the part that can afford to give: it now absorbs whatever
          the text block does not use, and the text block can never be clipped.
          `min-h-0` is required or the flex item refuses to shrink below its
          content. */}
      <Link
        to={`/assets/${assetId}/viewer`}
        className="min-h-0 flex-1 bg-slate-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-cyan-500"
        aria-label={`Open ${image.display_name}`}
      >
        {showThumbnail ? (
          <img
            className="h-full w-full object-cover"
            src={thumbnailUrl}
            alt=""
            loading="lazy"
            decoding="async"
            onError={() => setThumbnailFailed(true)}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-slate-500">
            <span className="text-xs font-semibold uppercase tracking-wide">
              Preview unavailable
            </span>
            <span className="text-sm">NGFF required</span>
          </div>
        )}
      </Link>
      <div className="shrink-0 space-y-3 p-3">
        <div>
          {/* Two lines' worth of box, always: `line-clamp-2` caps a long name at
              two lines, and reserving the second keeps every card's image/text
              seam on the same line across a row. */}
          <h3 className="line-clamp-2 min-h-10 text-sm font-semibold text-slate-950">
            {/* Keep navigation declarative. AssetRoute derives selection from
                the URL, so the card does not maintain a second route state. */}
            <Link
              to={`/assets/${assetId}/viewer`}
              className="text-left font-semibold text-cyan-800 hover:text-cyan-950 hover:underline focus:outline-none focus:ring-2 focus:ring-cyan-500"
            >
              {image.display_name}
            </Link>
          </h3>
          {image.notes?.trim() ? (
            <p
              className="mt-1 line-clamp-2 text-xs leading-4 text-slate-600"
              title={image.notes}
            >
              {image.notes}
            </p>
          ) : null}
          {/* slate-600, not slate-400: the dimensions are information, and
              slate-400 on white is 2.63:1. */}
          <p className="text-xs text-slate-600">
            {image.width} x {image.height}
            {image.stored_depth && image.stored_depth > 1
              ? ` x ${image.stored_depth}`
              : ""}
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {justImported ? <Badge tone="info">Just imported</Badge> : null}
          {pixelSize.calibrated ? (
            <PixelSizeTag valueNm={pixelSize.valueNm} />
          ) : null}
        </div>
      </div>
    </article>
  );
}
