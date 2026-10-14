export function metricLabel(metric: string): string {
  let label = metric;
  const units: Record<string, string> = { um2: "µm²", um: "µm", nm: "nm", px: "px", pixels: "px" };
  for (const [token, rawUnit] of Object.entries(units)) {
    if (!new RegExp(`_${token}(?:_|$)`).test(metric)) continue;
    const unit = ["px", "pixels"].includes(token) && /area|tissue|foreground|denominator/.test(metric) ? "px²" : rawUnit;
    label = `${metric.replace(new RegExp(`_${token}(?=_|$)`), "").replaceAll("_", " ")} (${unit})`;
    break;
  }
  return label.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}
