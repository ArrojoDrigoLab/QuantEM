import type { TrainingRound, TrainingTrace } from "@/shared/types/finetune";

function LossHistory({ trace, label }: { trace: TrainingTrace; label: string }) {
  const history = trace.history ?? [];
  const hasValidation = history.some((point) => point.validation_loss != null);
  const maxLoss = Math.max(0.01, ...history.flatMap((p) => [p.train_loss, p.validation_loss ?? 0]));
  const maxStep = Math.max(1, ...history.map((p) => p.step));
  const points = (key: "train_loss" | "validation_loss") => history
    .filter((p) => p[key] != null)
    .map((p) => `${10 + 380 * p.step / maxStep},${110 - 100 * (p[key] ?? 0) / maxLoss}`).join(" ");
  return <section aria-label={label} className="mt-2">
    <h4 className="text-xs font-semibold">{label}</h4>
    <p className="text-xs">{trace.steps} steps; {trace.patches_seen} / {trace.n_patches} tiles visited.</p>
    {history.length > 0 && <>
      <svg viewBox="0 0 400 125" role="img" aria-label={`${label}: loss versus steps`} className="w-full max-w-lg">
        <path d="M10 5 V110 H395" stroke="#94a3b8" fill="none" />
        <polyline points={points("train_loss")} fill="none" stroke="#2563eb" strokeWidth="2" />
        {hasValidation && <polyline points={points("validation_loss")} fill="none" stroke="#d97706" strokeWidth="2" />}
        <text x="12" y="12" fontSize="9">Loss ≤ {maxLoss.toFixed(3)}</text>
        <text x="10" y="123" fontSize="9">0</text><text x="350" y="123" fontSize="9">{maxStep} steps</text>
      </svg>
      <p className="text-xs"><span className="text-blue-700">Training loss</span>
        {hasValidation && <> · <span className="text-amber-700">Validation loss</span></>}</p>
      <div className="max-h-48 overflow-auto">
        <table aria-label={label} className="w-full text-left text-xs">
          <thead><tr><th>Step</th><th>Train loss</th>{hasValidation && <><th>Validation loss</th><th>Validation Dice</th></>}</tr></thead>
          <tbody>{history.map((p) => <tr key={p.step}><td>{p.step}</td><td>{p.train_loss.toFixed(4)}</td>
            {hasValidation && <><td>{p.validation_loss?.toFixed(4) ?? "—"}</td><td>{p.validation_dice?.toFixed(4) ?? "—"}</td></>}</tr>)}</tbody>
        </table>
      </div>
    </>}
  </section>;
}

export function TrainingHistory({ rounds }: { rounds?: TrainingRound[] }) {
  if (!rounds?.length) return null;
  return (
    <details className="rounded border border-slate-200 p-3">
      <summary className="cursor-pointer text-sm font-semibold">Training history</summary>
      {rounds.map((round) => {
        const selection = round.checkpoint_selection?.selection;
        return <div key={round.round} className="mt-3">
          {selection && <p className="text-xs">
            Validation selected step {selection.selected_step}{selection.stopped_early ? " (stopped early)" : ""}.
            All training areas were then refitted from the same starting weights, with at least one complete pass.
          </p>}
          {round.checkpoint_selection?.note && <p className="text-xs text-amber-800">{round.checkpoint_selection.note}</p>}
          {selection && <LossHistory trace={selection} label={`Round ${round.round} checkpoint selection`} />}
          <LossHistory trace={round.training} label={`Round ${round.round} ${selection ? "final refit" : "training"}`} />
        </div>;
      })}
    </details>
  );
}
