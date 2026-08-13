/**
 * "Pick an existing one, or type a new name."
 *
 * One control, used by the import form and by the library's selection bar, so
 * naming an experiment means the same thing in both places and neither needs a
 * separate management screen to create one first. Creating a group from where
 * it is needed is the whole reason there is no such screen.
 *
 * The choice is a discriminated union rather than a pair of loosely coupled
 * strings, because "separate per-image experiments", "this experiment" and
 * "an experiment I am about to name" are three different intents and a blank
 * string cannot tell the first from the third.
 */

import { useId } from "react";
import type { GroupingChoice } from "@/features/library/components/grouping/groupingChoices";

/**
 * Four intents, not two strings.
 *
 * `keep` is what makes a bulk change safe: a selection of forty images can be
 * put in a dataset **without** the same click also deciding their experiment.
 * It is the same tri-state the server takes (omit the field, send null, send a
 * value), and it is only offered where a selection already has values worth
 * keeping -- the import form has nothing to keep, so it does not show it.
 */
/** The sentinel the `<select>` uses for "I want to type a name". */
const NEW_VALUE = "__new__";
/** The sentinel for "do not touch this". Only rendered when offered. */
const KEEP_VALUE = "__keep__";

export interface GroupingOption {
  id: string;
  name: string;
  /** Shown after the name when it is known. Omitted rather than guessed. */
  count?: number;
}

export function GroupingPicker({
  label,
  options,
  value,
  onChange,
  disabled = false,
  noneLabel,
  newLabel,
  newPlaceholder,
  keepLabel,
  help,
}: {
  label: string;
  options: GroupingOption[];
  value: GroupingChoice;
  onChange: (next: GroupingChoice) => void;
  disabled?: boolean;
  /** What "not in one" is called here. Never phrased as a missing value. */
  noneLabel: string;
  newLabel: string;
  newPlaceholder: string;
  /** Offer "leave this alone". Omit where there is nothing to leave alone. */
  keepLabel?: string;
  help?: string;
}) {
  const fieldId = useId();
  const selectValue =
    value.kind === "existing"
      ? value.id
      : value.kind === "new"
        ? NEW_VALUE
        : value.kind === "keep"
          ? KEEP_VALUE
          : "";

  return (
    <div className="flex min-w-[200px] flex-1 flex-col gap-1">
      <label
        className="text-sm font-medium text-slate-800"
        htmlFor={`${fieldId}-select`}
      >
        {label}
      </label>
      <select
        id={`${fieldId}-select`}
        className="h-9 w-full rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
        value={selectValue}
        disabled={disabled}
        aria-describedby={help ? `${fieldId}-help` : undefined}
        onChange={(event) => {
          const next = event.target.value;
          if (next === KEEP_VALUE) onChange({ kind: "keep" });
          else if (next === "") onChange({ kind: "none" });
          else if (next === NEW_VALUE) onChange({ kind: "new", name: "" });
          else onChange({ kind: "existing", id: next });
        }}
      >
        {keepLabel ? <option value={KEEP_VALUE}>{keepLabel}</option> : null}
        <option value="">{noneLabel}</option>
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.count === undefined
              ? option.name
              : `${option.name} (${option.count})`}
          </option>
        ))}
        <option value={NEW_VALUE}>{newLabel}</option>
      </select>
      {value.kind === "new" ? (
        <input
          className="h-9 w-full rounded-md border border-slate-300 px-3 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          type="text"
          value={value.name}
          disabled={disabled}
          placeholder={newPlaceholder}
          aria-label={newLabel}
          onChange={(event) => onChange({ kind: "new", name: event.target.value })}
        />
      ) : null}
      {help ? (
        <span className="text-xs text-slate-600" id={`${fieldId}-help`}>
          {help}
        </span>
      ) : null}
    </div>
  );
}
