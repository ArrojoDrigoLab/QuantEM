import { useCallback, useEffect, useRef } from "react";
import { updateSegmentLabelsBatch } from "@/shared/api/segmentations/annotations";
import { extractApiErrorMessage } from "@/utils/apiErrors";
import type { LabelState } from "@/shared/types/common";
import type { SegmentationOverlayMutationState } from "@/shared/types/segmentation";

/**
 * How long answers are gathered before one request carries them all.
 *
 * A reviewer working at speed answers several objects a second, and each answer
 * used to be its own round-trip. The window turns a burst into one request
 * without the person ever waiting for it: the optimistic overlay has already
 * recoloured the object by the time the key is released, so the only thing this
 * delays is a write nobody is watching.
 *
 * It is a *window*, not a debounce. The timer starts with the first pending
 * answer and is never pushed back by later ones, so a reviewer who never pauses
 * still gets a request every 350 ms rather than none until they stop.
 */
export const LABEL_ANSWER_COALESCE_WINDOW_MS = 350;

export interface LabelAnswer {
  segmentId: string;
  labelState: LabelState;
  /** Shown if the request fails and this is the only answer in the batch. */
  fallbackMessage: string;
}

interface LabelAnswerQueueContext {
  segmentationId: string | null;
  sourceModel: string | null;
}

interface UseLabelAnswerQueueArgs {
  segmentationId: string | null;
  activeSourceModel: string | null;
  rollbackOptimisticLabel: (segmentId: string) => void;
  stageOptimisticRevisionTargets: (
    segmentIds: string[],
    targetRevision?: number | null
  ) => void;
  getOptimisticTargetRevision: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => number | null;
  handleOverlayMutationRefresh: (
    overlay: SegmentationOverlayMutationState | null | undefined
  ) => void;
  showErrorToast: (message: string) => void;
}

/**
 * One sentence for a failed batch, without hiding how much it took with it.
 *
 * The server's own reason is the useful half and is kept verbatim -- a
 * completion lock, say, explains itself far better than any generic wording
 * here could. What coalescing adds is that a single failure can now revert
 * several answers at once, and a person who gave five and sees one toast has to
 * be told that all five went back, or they will assume four of them stuck.
 */
function batchFailureMessage(answers: LabelAnswer[], error: unknown): string {
  if (answers.length === 1) {
    return extractApiErrorMessage(error, answers[0].fallbackMessage);
  }
  const reason = extractApiErrorMessage(error, "Those answers could not be saved.");
  return `${reason} All ${answers.length} have been put back the way they were.`;
}

/**
 * Gather single kept/removed/un-marked answers into one request.
 *
 * Every answer is still sent, in the order it was given, and an answer that is
 * changed again before the window closes is sent once with its final value --
 * so confirming an object and then taking it back costs one request carrying
 * the state the reviewer actually left it in, not two that race.
 *
 * The pending answers are flushed when the window closes, when the reviewer
 * moves to another segmentation or source model, when the screen unmounts, and
 * when the page is hidden or closed. That last one matters: without it, closing
 * the window within 350 ms of a keypress would discard the answer, and this
 * package exists to stop answers being lost, not to find a new way to lose them.
 */
export function useLabelAnswerQueue({
  segmentationId,
  activeSourceModel,
  rollbackOptimisticLabel,
  stageOptimisticRevisionTargets,
  getOptimisticTargetRevision,
  handleOverlayMutationRefresh,
  showErrorToast,
}: UseLabelAnswerQueueArgs) {
  // Everything the send path reads lives in a ref: it runs from a timer, an
  // unmount and a page-hide listener, none of which can be allowed to see a
  // stale render's copy of a callback.
  const pendingRef = useRef<Map<string, LabelAnswer>>(new Map());
  const timerRef = useRef<number | null>(null);
  const inFlightRef = useRef<Promise<void>>(Promise.resolve());
  const mountedRef = useRef(true);
  const contextRef = useRef<LabelAnswerQueueContext>({
    segmentationId,
    sourceModel: activeSourceModel,
  });
  const callbacksRef = useRef({
    rollbackOptimisticLabel,
    stageOptimisticRevisionTargets,
    getOptimisticTargetRevision,
    handleOverlayMutationRefresh,
    showErrorToast,
  });
  callbacksRef.current = {
    rollbackOptimisticLabel,
    stageOptimisticRevisionTargets,
    getOptimisticTargetRevision,
    handleOverlayMutationRefresh,
    showErrorToast,
  };

  const clearWindow = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const sendBatch = useCallback(
    async (answers: LabelAnswer[], context: LabelAnswerQueueContext) => {
      if (answers.length === 0 || !context.segmentationId) return;
      const callbacks = callbacksRef.current;
      try {
        const response = await updateSegmentLabelsBatch({
          labels: answers.map((answer) => ({
            id: answer.segmentId,
            label_state: answer.labelState,
          })),
          source_model: context.sourceModel,
        });
        if (!mountedRef.current) return;
        const overlay = response.overlays?.[context.segmentationId];
        callbacks.stageOptimisticRevisionTargets(
          answers.map((answer) => answer.segmentId),
          callbacks.getOptimisticTargetRevision(overlay)
        );
        callbacks.handleOverlayMutationRefresh(overlay);
      } catch (error) {
        // The answers never reached the server, so the optimistic labels are
        // now a claim about state that does not exist. Put every one of them
        // back and say so once, rather than a toast per answer.
        for (const answer of answers) {
          callbacks.rollbackOptimisticLabel(answer.segmentId);
        }
        if (mountedRef.current) {
          callbacks.showErrorToast(batchFailureMessage(answers, error));
        }
        console.error("Failed to save segment answers:", error);
      }
    },
    []
  );

  const drainPending = useCallback(() => {
    clearWindow();
    const answers = Array.from(pendingRef.current.values());
    pendingRef.current.clear();
    if (answers.length === 0) return inFlightRef.current;
    const context = { ...contextRef.current };
    // Chained, not fired in parallel: two batches in flight at once could reach
    // the server out of order, and the later answer is the one the reviewer
    // means.
    inFlightRef.current = inFlightRef.current.then(() => sendBatch(answers, context));
    return inFlightRef.current;
  }, [clearWindow, sendBatch]);

  const flushAnswers = useCallback(async () => {
    await drainPending();
  }, [drainPending]);

  const enqueueAnswer = useCallback(
    (answer: LabelAnswer) => {
      if (!contextRef.current.segmentationId) return;
      // Keyed by segment: the last answer for an object is the one that counts,
      // and it keeps its place in the order it was first given.
      pendingRef.current.delete(answer.segmentId);
      pendingRef.current.set(answer.segmentId, answer);
      if (timerRef.current !== null) return;
      timerRef.current = window.setTimeout(() => {
        timerRef.current = null;
        void drainPending();
      }, LABEL_ANSWER_COALESCE_WINDOW_MS);
    },
    [drainPending]
  );

  // Moving to another segmentation or another source model changes what the
  // request means, so whatever is pending is sent under the context it was
  // given in before the new one takes effect.
  useEffect(() => {
    const previous = contextRef.current;
    if (
      previous.segmentationId === segmentationId &&
      previous.sourceModel === activeSourceModel
    ) {
      return;
    }
    void drainPending();
    contextRef.current = { segmentationId, sourceModel: activeSourceModel };
  }, [activeSourceModel, drainPending, segmentationId]);

  useEffect(() => {
    const flushOnLeaving = () => {
      if (pendingRef.current.size === 0) return;
      void drainPending();
    };
    const flushIfHidden = () => {
      if (document.visibilityState === "hidden") flushOnLeaving();
    };
    window.addEventListener("pagehide", flushOnLeaving);
    document.addEventListener("visibilitychange", flushIfHidden);
    return () => {
      window.removeEventListener("pagehide", flushOnLeaving);
      document.removeEventListener("visibilitychange", flushIfHidden);
    };
  }, [drainPending]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      // Send first, then stop touching component state: the answers are the
      // user's work and must leave, but the callbacks belong to a screen that
      // is going away.
      void drainPending();
      mountedRef.current = false;
    };
  }, [drainPending]);

  return { enqueueAnswer, flushAnswers };
}
