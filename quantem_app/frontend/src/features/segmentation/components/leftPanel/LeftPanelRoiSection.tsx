import type {
  LeftPanelRoiState,
  LeftPanelWorkflowState,
} from "@/features/segmentation/components/leftPanel/types";

interface LeftPanelRoiSectionProps {
  workflow: LeftPanelWorkflowState;
  roi: LeftPanelRoiState;
}

/**
 * Unreachable, pending deletion.
 *
 * This rendered `RoiAnnotationPanel`, and only ever when the workflow mode was
 * `annotate`. That mode has been removed -- its handlers had already been
 * emptied to `() => {}`, so entering it gave the user a canvas that quietly did
 * nothing -- and with it the only condition under which this panel appeared.
 *
 * It is left as a stub rather than deleted here because the panel it mounted,
 * its stylesheet and the surrounding dead props are one removal owned by the
 * delete package; splitting that across two changes would leave a half-removed
 * component in the tree.
 */
export function LeftPanelRoiSection(props: LeftPanelRoiSectionProps) {
  void props;
  return null;
}
