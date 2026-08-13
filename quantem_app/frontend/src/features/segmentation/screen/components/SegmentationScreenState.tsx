import type { AssetDetail } from "@/shared/types";

interface SegmentationScreenStateProps {
  kind: "no-selection" | "loading" | "image-error" | "preprocess" | "empty";
  image?: AssetDetail | null;
  preprocessLabel?: string;
}

export function SegmentationScreenState({
  kind,
  image = null,
  preprocessLabel = "",
}: SegmentationScreenStateProps) {
  if (kind === "no-selection") {
    return (
      <div className="segmentation-screen">
        <div className="no-selection">
          <p>No image selected. Please select an image from your library.</p>
        </div>
      </div>
    );
  }

  if (kind === "loading") {
    return (
      <div className="segmentation-screen">
        <div className="loading">Loading...</div>
      </div>
    );
  }

  if (kind === "image-error") {
    return (
      <div className="segmentation-screen">
        <div className="error">Failed to load image</div>
      </div>
    );
  }

  if (kind === "preprocess") {
    return (
      <div className="segmentation-screen">
        <div className="loading">
          <p>Preparing segmentation assets...</p>
          <p>Preprocess status: {preprocessLabel}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="segmentation-screen">
      <header className="segmentation-header">
        <div className="header-info">
          <h2>{image?.display_name}</h2>
        </div>
      </header>
      <main className="segmentation-main">
        <section className="segmentation-prompt">
          <div className="segmentation-empty">
            No segmentations available for this image. Create one from the viewer
            sidebar.
          </div>
        </section>
      </main>
    </div>
  );
}
