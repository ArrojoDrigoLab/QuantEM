import type { SegmentOverlay } from "@/viewer/types";
import type { UserFeedback } from "@/shared/types/segmentation";
import { circlePoints } from "@/features/segmentation/overlays/shared";

export function generateUserFeedbackPointOverlays(
  feedbackItems: UserFeedback[]
): SegmentOverlay[] {
  return feedbackItems.flatMap((feedback) => {
    if (!feedback.point) return [];

    let fillColor = "#ffd400";
    let strokeColor = "#b59600";
    if (feedback.utilized_status === "SUCCESS") {
      fillColor = "#33cc66";
      strokeColor = "#1f8f4a";
    } else if (feedback.utilized_status === "FAILED") {
      fillColor = "#ff5d5d";
      strokeColor = "#c83737";
    }

    return [
      {
        id: `user-feedback-${feedback.id}`,
        geometry: circlePoints({ x: feedback.point.x, y: feedback.point.y }, 5, 12),
        fillColor,
        fillOpacity: 0.8,
        strokeColor,
        strokeOpacity: 0.95,
        strokeWidth: 1.5,
      },
    ];
  });
}
