/**
 * Which interaction mode the labeling screen is in.
 *
 * Navigate is a mode in the strict sense: while it is on, the interaction
 * router drops every labeling click and drag, so anything that gives the user a
 * tool has to turn it off or the first stroke silently does nothing. Packages
 * that add a mode change this file; packages that add a message change the
 * toast slot beside it.
 *
 * **Navigate now starts off.** It used to start on, which meant the first thing
 * every user did on every image was click an object, watch nothing happen, and
 * go looking for the reason — the screen's own answer was a passive note in the
 * sidebar saying clicks pan instead of labeling. Arriving armed to label is the
 * whole point of the canvas reform: the click does what the tool says, and the
 * gestures that pan (hold space and drag, or drag with the middle button) do
 * not need a mode at all.
 */

import { useCallback, useEffect, useState } from "react";

export function useSegmentationModes({
  currentSegmentationId,
  readOnly = false,
}: {
  currentSegmentationId: string | null;
  readOnly?: boolean;
}) {
  const [leftNavigateMode, setLeftNavigateMode] = useState(readOnly);
  const [showConfirmedPanel, setShowConfirmedPanel] = useState(true);
  const [uncertainLimit, setUncertainLimit] = useState(50);
  // Opening or locking a result always starts with its saved objects visible.
  useEffect(() => {
    if (!readOnly) return;
    setLeftNavigateMode(true);
    setShowConfirmedPanel(true);
  }, [currentSegmentationId, readOnly]);

  const toggleLeftNavigateMode = useCallback(() => {
    setLeftNavigateMode((previous) => !previous);
  }, []);

  /**
   * Turn Navigate off.
   *
   * Called when the user picks a drawing tool or enters Correct: while Navigate
   * is on the interaction router drops every labeling click and drag, so the
   * first stroke after choosing a tool silently did nothing.
   */
  const exitNavigateMode = useCallback(() => {
    setLeftNavigateMode(false);
  }, []);

  /*
   * Switching between unlocked organelles preserves the chosen editing modes.
   *
   * There used to be an effect that forced Navigate back on whenever
   * `currentSegmentationId` changed. Picking mitochondria after nucleus is not
   * a request to stop labeling, and the re-arm was invisible: the canvas simply
   * stopped responding to clicks, in the middle of a proofreading rhythm, with
   * no event the user could connect it to. A mode the user chose stays chosen
   * until the user changes it.
   */

  return {
    leftNavigateMode,
    setLeftNavigateMode,
    toggleLeftNavigateMode,
    exitNavigateMode,
    showConfirmedPanel,
    setShowConfirmedPanel,
    uncertainLimit,
    setUncertainLimit,
  };
}
