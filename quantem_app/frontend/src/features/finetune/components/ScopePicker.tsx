/**
 * The applies-to selector: experiments, their datasets, and the images inside.
 *
 * Dataset-level by default and expandable to images, which is the owner's
 * shape and also the honest one — a dataset is the unit a user thinks in, and
 * a list of six hundred image names with tick boxes is not a selector.
 *
 * The arithmetic and the selection rules are all in `scopeTree.ts`; this file
 * is the tree drawn, and every count on it comes from there.
 */

import { useState } from "react";
import {
  isDatasetFullySelected,
  isGroupFullySelected,
  isImageSelected,
  setDatasetSelected,
  setDatasetImageSelected,
  setGroupSelected,
  setImageSelected,
  type ScopeDatasetNode,
  type ScopeGroupNode,
  type ScopeImageNode,
  type ScopeSelection,
} from "@/features/finetune/scopeTree";

function annotationText(count: number): string {
  return `${count} ${count === 1 ? "annotation" : "annotations"}`;
}

function imageText(count: number): string {
  return `${count} ${count === 1 ? "image" : "images"}`;
}

function ImageRow({
  image,
  checked,
  disabled,
  onToggle,
}: {
  image: ScopeImageNode;
  checked: boolean;
  disabled: boolean;
  onToggle: (on: boolean) => void;
}) {
  return (
    <li className="flex items-center gap-2 py-0.5 pl-6">
      <label className="flex min-w-0 flex-1 items-center gap-2 text-xs text-slate-700">
        <input
          type="checkbox"
          className="h-3.5 w-3.5"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onToggle(event.target.checked)}
        />
        <span className="truncate" title={image.name}>
          {image.name}
        </span>
      </label>
      <span className="shrink-0 text-xs tabular-nums text-slate-500">
        {annotationText(image.annotationCount)}
      </span>
    </li>
  );
}

function DatasetRow({
  dataset,
  selection,
  onChange,
}: {
  dataset: ScopeDatasetNode;
  selection: ScopeSelection;
  onChange: (next: ScopeSelection) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const selected = isDatasetFullySelected(selection, dataset);
  return (
    <li className="py-0.5">
      <div className="flex items-center gap-2">
        <button
          type="button"
          className="h-5 w-5 shrink-0 rounded text-xs text-slate-500 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          aria-expanded={expanded}
          aria-label={`${expanded ? "Collapse" : "Expand"} ${dataset.name}`}
          onClick={() => setExpanded((current) => !current)}
        >
          {expanded ? "▾" : "▸"}
        </button>
        <label className="flex min-w-0 flex-1 items-center gap-2 text-sm text-slate-800">
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={selected}
            onChange={(event) =>
              onChange(setDatasetSelected(selection, dataset, event.target.checked))
            }
          />
          <span className="truncate font-medium" title={dataset.name}>
            {dataset.name}
          </span>
        </label>
        <span className="shrink-0 text-xs tabular-nums text-slate-500">
          {imageText(dataset.imageCount)} · {annotationText(dataset.annotationCount)}
        </span>
      </div>
      {expanded ? (
        dataset.images.length > 0 ? (
          <ul className="m-0 list-none p-0">
            {dataset.images.map((image) => (
              <ImageRow
                key={image.id}
                image={image}
                checked={isImageSelected(selection, image.id, dataset.id)}
                disabled={false}
                onToggle={(on) =>
                  onChange(
                    setDatasetImageSelected(selection, dataset, image.id, on)
                  )
                }
              />
            ))}
          </ul>
        ) : (
          <p className="m-0 py-1 pl-6 text-xs text-slate-500">
            No images to list for this dataset.
          </p>
        )
      ) : null}
    </li>
  );
}

export function ScopePicker({
  groups,
  selection,
  onChange,
  emptyMessage,
}: {
  groups: ScopeGroupNode[];
  selection: ScopeSelection;
  onChange: (next: ScopeSelection) => void;
  emptyMessage: string;
}) {
  if (groups.length === 0) {
    return (
      <p className="m-0 px-3 py-4 text-sm text-slate-500" data-testid="scope-empty">
        {emptyMessage}
      </p>
    );
  }
  return (
    <div data-testid="scope-tree">
      {groups.map((group) => (
        <section key={group.key} className="border-b border-slate-100 px-3 py-2 last:border-b-0">
          <div className="flex items-center gap-2">
            <label className="flex min-w-0 flex-1 items-center gap-2 text-sm text-slate-900">
              <input
                type="checkbox"
                className="h-4 w-4"
                checked={isGroupFullySelected(selection, group)}
                onChange={(event) =>
                  onChange(setGroupSelected(selection, group, event.target.checked))
                }
              />
              <span className="truncate font-semibold" title={group.name}>
                {group.name}
              </span>
            </label>
            <span className="shrink-0 text-xs tabular-nums text-slate-500">
              {annotationText(group.annotationCount)}
            </span>
          </div>
          <ul className="m-0 mt-1 list-none p-0">
            {group.datasets.map((dataset) => (
              <DatasetRow
                key={dataset.id}
                dataset={dataset}
                selection={selection}
                onChange={onChange}
              />
            ))}
            {group.images.map((image) => (
              <ImageRow
                key={image.id}
                image={image}
                checked={isImageSelected(selection, image.id, null)}
                disabled={false}
                onToggle={(on) => onChange(setImageSelected(selection, image.id, on))}
              />
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
