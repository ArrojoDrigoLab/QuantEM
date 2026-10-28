export const ANALYSIS_FILES: Record<string, string> = {
  "region_summary.csv": "Region summaries",
  "cell_summary.csv": "Per-cell measurements",
  "organelle_objects.csv": "All organelle objects",
  "objects.csv": "Primary organelle objects",
  "image_summary.csv": "Image summary",
  "composition.csv": "Compartment composition",
  "area.csv": "Area-only measurements",
};

export const defaultOutputFiles = () => Object.fromEntries(
  Object.keys(ANALYSIS_FILES).map((name) => [name, true])
);
