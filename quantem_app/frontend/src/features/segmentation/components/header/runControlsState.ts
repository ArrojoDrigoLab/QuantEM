/**
 * Whether the run-without-a-pixel-size confirmation is open.
 *
 * Split out of `SegmentationHeader.tsx` unchanged, and kept apart from
 * `RunControls.tsx` so that file exports components only. The button and the
 * dialog render in two different places in the `<header>`, so the one boolean
 * they share cannot live in either of them.
 */

import { useCallback, useState } from "react";

export function useRunScaleConfirm() {
  const [isOpen, setIsOpen] = useState(false);
  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);
  return { isOpen, open, close };
}
