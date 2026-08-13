/**
 * The tool's controls, rendered into the toolbar's `extraModes` slot.
 *
 * This package renders into that slot from its own file rather than opening
 * `WorkflowModeToolbar.tsx`, which is what the slot exists for.
 */

import type { SamBoxTool } from "./useSamBoxTool";
import "./SamBoxToolControls.css";

function BoxIcon() {
  return (
    <svg
      width={18}
      height={18}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="3" y="5" width="18" height="14" rx="1" strokeDasharray="4 3" />
      <circle cx="12" cy="12" r="3.2" strokeDasharray="0" />
    </svg>
  );
}

function megabytes(bytes: number): string {
  return `${Math.round(bytes / 1e6)} MB`;
}

export interface SamBoxToolControlsProps {
  tool: SamBoxTool;
  /** Whether box-to-object is the selected correction sub-tool. */
  selected: boolean;
  onToggle: () => void;
}

export function SamBoxToolControls({
  tool,
  selected,
  onToggle,
}: SamBoxToolControlsProps) {
  const { model } = tool;
  const download = model?.download;
  const downloading = download?.status === "RUNNING";

  return (
    <div className="sam-box-tool">
      <button
        type="button"
        className={`icon-tool-button ${selected ? "active" : ""}`}
        onClick={onToggle}
        aria-pressed={selected}
        aria-label="Box to object"
        title="Box to object: drag a box around one object and it is segmented and added"
      >
        <BoxIcon />
      </button>

      {tool.isActive && (
        <div className="sam-box-tool-detail">
          {tool.isSubmitting && (
            <span className="sam-box-tool-hint" role="status">
              {tool.pendingCount === 1
                ? "Segmenting the box."
                : `Segmenting ${tool.pendingCount} boxes.`}
            </span>
          )}

          {!tool.isSubmitting && tool.modelReady && (
            <span className="sam-box-tool-hint">
              Drag a box around one object.
              {tool.lastTiming && !tool.lastTiming.cache_hit
                ? " The first box in a new area takes a moment longer."
                : ""}
            </span>
          )}

          {!tool.modelReady && !downloading && (
            <div className="sam-box-tool-gate">
              <p className="sam-box-tool-hint">
                {model
                  ? `${model.model} is not on this computer yet. It is a ${megabytes(
                      model.size_bytes
                    )} download and only needs to happen once.`
                  : "Checking whether the segmenting model is on this computer."}
              </p>
              {download?.status === "FAILED" && download.error && (
                <p className="sam-box-tool-error" role="alert">
                  {download.error}
                </p>
              )}
              {model && (
                <button
                  type="button"
                  className="sam-box-tool-download"
                  onClick={tool.downloadModel}
                >
                  {download?.status === "FAILED" ? "Try again" : "Download"}
                </button>
              )}
            </div>
          )}

          {downloading && download && (
            <div className="sam-box-tool-gate">
              <progress
                className="sam-box-tool-progress"
                max={download.bytes_total || 1}
                value={download.bytes_done}
                aria-label="Downloading the segmenting model"
              />
              <p className="sam-box-tool-hint">
                Downloading {megabytes(download.bytes_done)} of{" "}
                {megabytes(download.bytes_total)}
                {download.percent !== null ? ` (${download.percent}%)` : ""}.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
