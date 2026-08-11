import { useEffect, useState } from "react";
import {
  getAssetNgffThumbnailUrl,
  getAssetPreviewThumbnailUrl,
} from "@/shared/api/assets";
import { resolveEntryPixelSize } from "@/shared/pixelSize";
import { cx } from "@/shared/ui/cx";
import { Badge, Button } from "@/shared/ui/design";
import { PixelSizeBadge } from "@/shared/ui/PixelSize";
import { getStageDisplay } from "@/features/library/components/imageCardUtils";
import type { HomeEntry } from "@/shared/types/images";

const NGFF_THUMBNAIL_CACHE_VERSION = "ngff-thumb-v2";
const PREVIEW_THUMBNAIL_CACHE_VERSION = "preview-thumb-v1";

export function ImageCard({
  image,
  onOpen,
  onDelete,
  deleting = false,
  justImported = false,
  selectable = false,
  selected = false,
  onToggleSelect,
}: {
  image: HomeEntry;
  onOpen: (assetId: string) => void;
  onDelete?: (image: HomeEntry) => void;
  deleting?: boolean;
  /**
   * This is the image the user just imported. The confirmation strip above the
   * grid says "it is the first card below"; this is what makes that sentence
   * point at something.
   */
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
}) {
  const assetId = image.id;
  const [thumbnailFailed, setThumbnailFailed] = useState(false);
  useEffect(() => {
    setThumbnailFailed(false);
  }, [image.id, image.updated_at]);
  const ready = image.ngff_ready || image.preprocess_stage === "DONE";
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
  // The list payload carries no renditions, but it does carry
  // `file_declared_pixel_size_nm`, which is all the provenance needs. Before
  // that field existed every calibrated image here resolved to "manual" and the
  // tooltip asserted the file declared nothing -- contradicting the viewer,
  // which read the same 5 nm/px straight off the TIFF tag.
  //
  // The provenance is printed on the badge, not left to its colour: this is the
  // screen where images are compared side by side, and "read from the file"
  // versus "typed by a person" is the difference between a number a caption can
  // cite and one that needs checking. Emerald-vs-cyan plus a hover tooltip was
  // reachable by neither a keyboard nor a touch screen.
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
      {onDelete ? (
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
      <div className="min-h-0 flex-1 bg-slate-100">
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
      </div>
      <div className="shrink-0 space-y-3 p-3">
        <div>
          {/* Two lines' worth of box, always: `line-clamp-2` caps a long name at
              two lines, and reserving the second keeps every card's image/text
              seam on the same line across a row. */}
          <h3 className="line-clamp-2 min-h-10 text-sm font-semibold text-slate-950">
            {/* Hash route, not a bare path: the app runs under HashRouter with
                `base: './'`, so a middle-click or "copy link address" on
                "/assets/<id>/viewer" navigates for real and lands on a white
                screen. The onClick keeps the in-app (non-reloading) route. */}
            <a
              href={`#/assets/${assetId}/viewer`}
              className="text-left font-semibold text-cyan-800 hover:text-cyan-950 hover:underline focus:outline-none focus:ring-2 focus:ring-cyan-500"
              onClick={(event) => {
                event.preventDefault();
                onOpen(assetId);
              }}
            >
              {image.display_name}
            </a>
          </h3>
          {/* One line, ellipsised. A 60-character filename used to wrap to three
              and push the badges out of the card; the whole string is still
              readable on hover and on the viewer screen. */}
          <p
            className="mt-1 truncate text-xs text-slate-500"
            title={image.original_filename}
          >
            {image.original_filename}
          </p>
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
          {/* A failed import is not an amber "still working on it". Rendered
              as its own span rather than through `Badge`'s tone set, which has
              no danger tone -- and the reason, which is on the payload as
              `preprocess_error` and was shown nowhere at all, is on the
              tooltip instead of behind a trip to the Tasks drawer. */}
          {failed ? (
            <span
              className="inline-flex items-center rounded-full border border-red-200 bg-red-50 px-2 py-0.5 text-xs font-medium text-red-700"
              title={image.preprocess_error || undefined}
            >
              {getStageDisplay(image)}
            </span>
          ) : (
            <Badge tone={ready ? "good" : "warning"}>
              {getStageDisplay(image)}
            </Badge>
          )}
          <PixelSizeBadge resolved={pixelSize} />
        </div>
      </div>
    </article>
  );
}
