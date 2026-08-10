import { RoiAnnotationPanel } from "@/features/segmentation/components/RoiAnnotationPanel";
import type {
  LeftPanelRoiState,
  LeftPanelWorkflowState,
} from "@/features/segmentation/components/leftPanel/types";

interface LeftPanelRoiSectionProps {
  workflow: LeftPanelWorkflowState;
  roi: LeftPanelRoiState;
}

export function LeftPanelRoiSection({ workflow, roi }: LeftPanelRoiSectionProps) {
  if (workflow.mode !== "annotate") {
    return null;
  }

  return (
    <RoiAnnotationPanel
      activeRoi={roi.activeRoi}
      roiPoints={roi.roiPoints}
      roiPointsSubmitted={roi.roiPointsSubmitted}
      roiComplete={roi.roiComplete}
      roiLabelMode={roi.roiLabelMode}
      brushSize={roi.brushSize}
      onRoiLabelModeChange={roi.onRoiLabelModeChange}
      onBrushSizeChange={roi.onBrushSizeChange}
      onSubmitRoiLabels={roi.onSubmitRoiLabels}
      onClearRoiLabels={roi.onClearRoiLabels}
      onReselectRoi={roi.onReselectRoi}
      onMarkRoiComplete={roi.onMarkRoiComplete}
    />
  );
}
