import type { TrainingRound } from "@/shared/types/finetune";

export function TrainingHistory({ rounds }: { rounds?: TrainingRound[] }) {
  if (!rounds?.length) return null;
  return (
    <details className="rounded border border-slate-200 p-3">
      <summary className="cursor-pointer text-sm font-semibold">Training history</summary>
      {rounds.map((round) => {
        const selection = round.checkpoint_selection?.selection;
        const history = selection?.history ?? round.training.history ?? [];
        const maxLoss = Math.max(0.01, ...history.flatMap((p) => [p.train_loss, p.validation_loss ?? 0]));
        const maxStep = Math.max(1, ...history.map((p) => p.step));
        const points = (key: "train_loss" | "validation_loss") => history
          .filter((p) => p[key] != null)
          .map((p) => `${10 + 380 * p.step / maxStep},${110 - 100 * (p[key] ?? 0) / maxLoss}`).join(" ");
        return <div key={round.round} className="mt-3">
          <p className="text-xs">Round {round.round}: {round.training.steps} final training steps;
            {" "}{round.training.patches_seen} / {round.training.n_patches} tiles visited.
            {selection ? ` Validation selected step ${selection.selected_step}${selection.stopped_early ? " (stopped early)" : ""}; all training areas were then refitted.` : ""}
          </p>
          <svg viewBox="0 0 400 125" role="img" aria-label={`Round ${round.round} training and validation loss versus steps`} className="w-full max-w-lg">
            <path d="M10 5 V110 H395" stroke="#94a3b8" fill="none" />
            <polyline points={points("train_loss")} fill="none" stroke="#2563eb" strokeWidth="2" />
            <polyline points={points("validation_loss")} fill="none" stroke="#d97706" strokeWidth="2" />
            <text x="10" y="123" fontSize="9">0</text><text x="350" y="123" fontSize="9">{maxStep} steps</text>
          </svg>
          <p className="text-xs"><span className="text-blue-700">Training loss</span> · <span className="text-amber-700">Validation loss</span></p>
          <div className="max-h-48 overflow-auto">
            <table className="w-full text-left text-xs"><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss</th><th>Validation Dice</th></tr></thead>
              <tbody>{history.map((p) => <tr key={p.step}><td>{p.step}</td><td>{p.train_loss.toFixed(4)}</td><td>{p.validation_loss?.toFixed(4) ?? "—"}</td><td>{p.validation_dice?.toFixed(4) ?? "—"}</td></tr>)}</tbody>
            </table>
          </div>
        </div>;
      })}
    </details>
  );
}
