/** Pixel-size display and editor. */

import { useEffect, useState } from "react";
import { updateAsset } from "@/shared/api/assets";
import {
  formatPixelSizeNm,
  parsePixelSizeInput,
  resolvePixelSize,
} from "@/shared/pixelSize";
import { Badge, Button } from "@/shared/ui/design";
import { cx } from "@/shared/ui/cx";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { AssetDetail } from "@/shared/types/images";

export interface PixelSizeTagProps {
  valueNm: number | null;
  className?: string;
}

/**
 * The one read-only representation of image resolution across the application.
 * A calibrated tag contains only the normalized value: `X nm/px`.
 */
export function PixelSizeTag({ valueNm, className }: PixelSizeTagProps) {
  const label = formatPixelSizeNm(valueNm);
  if (!label) {
    return (
      <Badge
        tone="warning"
        className={className}
        title="This image has no pixel size, so areas and distances can only be reported in pixels. Set one to unlock µm² and nm."
      >
        Pixel size not set
      </Badge>
    );
  }
  return (
    <Badge tone="good" className={className}>
      {label}
    </Badge>
  );
}

export interface PixelSizeEditorProps {
  asset: AssetDetail;
  /** Called with the patched asset so the caller can refresh its own copy. */
  onSaved?: (asset: AssetDetail) => void;
  className?: string;
  /** Viewer chrome uses a scale-only badge and an icon-sized edit affordance. */
  compact?: boolean;
}

/**
 * Badge plus an inline edit form that PATCHes `pixel_size_nm`.
 *
 * Clearing the field sends `null`, which the backend reads as "unknown" -- the
 * only honest way back out of a wrong calibration.
 */
export function PixelSizeEditor({
  asset,
  onSaved,
  className,
  compact = false,
}: PixelSizeEditorProps) {
  const resolved = resolvePixelSize(asset);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setEditing(false);
    setError(null);
  }, [asset.id]);

  const startEditing = () => {
    setDraft(resolved.valueNm === null ? "" : String(resolved.valueNm));
    setError(null);
    setEditing(true);
  };

  const handleSave = async () => {
    const parsed = parsePixelSizeInput(draft);
    if (parsed.error) {
      setError(parsed.error);
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const updated = await updateAsset(asset.id, { pixel_size_nm: parsed.value });
      setEditing(false);
      onSaved?.(updated);
    } catch (err) {
      setError(extractApiErrorMessage(err, "The pixel size could not be saved."));
    } finally {
      setSaving(false);
    }
  };

  if (!editing) {
    return (
      <div className={cx("flex flex-wrap items-center gap-2", className)}>
        <PixelSizeTag valueNm={resolved.valueNm} />
        <Button
          size={compact ? "icon" : "sm"}
          onClick={startEditing}
          aria-label={resolved.calibrated ? "Edit pixel size" : "Set pixel size"}
          title={resolved.calibrated ? "Edit pixel size" : "Set pixel size"}
        >
          {compact ? <PencilIcon /> : resolved.calibrated ? "Edit pixel size" : "Set pixel size"}
        </Button>
      </div>
    );
  }

  return (
    <div className={cx("flex flex-col gap-1", className)}>
      <div className="flex flex-wrap items-center gap-2">
        <label
          className={
            compact
              ? "sr-only"
              : "text-xs font-semibold uppercase tracking-wide text-slate-500"
          }
          htmlFor={`pixel-size-${asset.id}`}
        >
          Pixel size
        </label>
        <input
          id={`pixel-size-${asset.id}`}
          className="h-9 w-28 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          type="number"
          min="0"
          step="any"
          inputMode="decimal"
          value={draft}
          placeholder="e.g. 4.2"
          disabled={saving}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void handleSave();
            }
            if (event.key === "Escape") setEditing(false);
          }}
        />
        <span className="text-sm text-slate-500">nm/px</span>
        <Button size="sm" variant="primary" disabled={saving} onClick={() => void handleSave()}>
          {saving ? "Saving…" : "Save"}
        </Button>
        <Button size="sm" disabled={saving} onClick={() => setEditing(false)}>
          Cancel
        </Button>
      </div>
      <p className={compact ? "hidden" : "m-0 text-xs text-slate-500"}>
        Leave blank to mark the image uncalibrated.
      </p>
      {error ? <p className="m-0 text-xs text-red-700">{error}</p> : null}
    </div>
  );
}

function PencilIcon() {
  return (
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
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}
