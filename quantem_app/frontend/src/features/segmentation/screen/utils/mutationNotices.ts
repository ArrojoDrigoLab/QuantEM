/**
 * What a geometry mutation actually did, when that is not what was asked for.
 *
 * Every drawing and editing tool on the labeling screen posts an outline and
 * gets back a 2xx body describing what became of it. Three different things in
 * that body mean "less happened than the gesture looked like", and each one
 * used to be read in exactly one place or in none at all:
 *
 *   - `outlines.detail` — the outline separated into several objects, or was
 *     refused for spanning a pixel or less. Read only by the Draw tool.
 *   - `measurement.detail` — the geometry is committed but the morphometrics
 *     behind it are missing. Read only by the Draw tool, and not even typed on
 *     the remove-area response.
 *   - nothing created, updated or deleted — the request was well-formed, the
 *     server had no complaint, and the store is unchanged. Read nowhere.
 *
 * The last one is the backstop. It is the only signal available on the merge
 * path (`merge_overlaps`), where the server deliberately makes no per-outline
 * claim because a lobe too thin to stand alone may still survive inside the
 * object it fuses with — so it cannot say in advance what it kept, but it can
 * always say that nothing changed.
 *
 * Collecting them in one function is the point: the failure this replaces was
 * five call sites each deciding for itself how much of the response to look at,
 * and four of them deciding "none".
 */

import type {
  ConfirmBatchResponse,
  RemoveAreaResponse,
} from "@/shared/types/segmentation";

/** The subset of a mutation response these notices are derived from. */
export interface MutationOutcome {
  created?: number;
  updated?: number;
  deleted?: number;
  outlines?: ConfirmBatchResponse["outlines"];
  measurement?: ConfirmBatchResponse["measurement"];
}

export interface MutationNoticeOptions {
  /**
   * What to say when the response reports no change at all. Omit on a call
   * where "nothing changed" is a legitimate outcome the user already sees.
   */
  nothingStoredMessage?: string;
}

/**
 * Sentences to show the user about a mutation that did not fail.
 *
 * Empty for the ordinary case: one outline in, one object out, measured. Join
 * with a space and show as a notice, never as an error — the drawing was
 * committed and calling it a failure would say the opposite.
 */
export function collectMutationNotices(
  response: MutationOutcome | RemoveAreaResponse | null | undefined,
  options: MutationNoticeOptions = {}
): string[] {
  if (!response) return [];

  const notices: string[] = [];

  const outlines = (response as MutationOutcome).outlines;
  if (outlines?.detail) notices.push(outlines.detail);

  // Committed either way (207, not an error), but these objects reach
  // objects.csv with their morphometric columns missing.
  const measurement = (response as MutationOutcome).measurement;
  if (measurement?.detail) notices.push(measurement.detail);

  const changed =
    (response.created ?? 0) + (response.updated ?? 0) + (response.deleted ?? 0);
  if (changed === 0 && notices.length === 0 && options.nothingStoredMessage) {
    notices.push(options.nothingStoredMessage);
  }

  return notices;
}

/**
 * `collectMutationNotices`, joined into the one sentence a toast shows.
 *
 * `null` when there is nothing to say, so a caller can write
 * `if (message) showNoticeToast(message)` and never show an empty toast.
 */
export function mutationNoticeMessage(
  response: MutationOutcome | RemoveAreaResponse | null | undefined,
  options: MutationNoticeOptions = {}
): string | null {
  const notices = collectMutationNotices(response, options);
  return notices.length > 0 ? notices.join(" ") : null;
}
