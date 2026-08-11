/**
 * The labeling screen's own UI state, composed from its two halves.
 *
 * The mode state and the toast slot were one 96-line hook and are now
 * `screen/state/modes.ts` and `screen/state/toasts.ts`. Adding a mode and
 * adding a message are different jobs done by different packages, and they were
 * colliding on the same `useEffect` that resets both when the segmentation
 * changes — that reset is now one line in each file. The return shape is
 * unchanged, so every caller and `useSegmentationScreenUiState.test.tsx` are
 * untouched.
 */

import { useSegmentationModes } from "@/features/segmentation/screen/state/modes";
import { useSegmentationToast } from "@/features/segmentation/screen/state/toasts";

interface UseSegmentationScreenUiStateArgs {
  currentSegmentationId: string | null;
}

export function useSegmentationScreenUiState({
  currentSegmentationId,
}: UseSegmentationScreenUiStateArgs) {
  const modes = useSegmentationModes({ currentSegmentationId });
  const toast = useSegmentationToast({ currentSegmentationId });

  return {
    leftNavigateMode: modes.leftNavigateMode,
    setLeftNavigateMode: modes.setLeftNavigateMode,
    toggleLeftNavigateMode: modes.toggleLeftNavigateMode,
    exitNavigateMode: modes.exitNavigateMode,
    showConfirmedPanel: modes.showConfirmedPanel,
    setShowConfirmedPanel: modes.setShowConfirmedPanel,
    uncertainLimit: modes.uncertainLimit,
    setUncertainLimit: modes.setUncertainLimit,
    toast: toast.toast,
    showErrorToast: toast.showErrorToast,
    showNoticeToast: toast.showNoticeToast,
    dismissToast: toast.dismissToast,
  };
}
