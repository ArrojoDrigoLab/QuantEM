/**
 * The intended workflow, written down.
 *
 * There was previously nothing in the app that said what the five screens are
 * for or in what order to use them, so a first-time user met a library, a
 * collapsed import panel and an empty state that pointed at a button that was
 * not visible. This shows on first launch and can be reopened from the header.
 *
 * One line per step, naming the step and nothing else. Caveats that used to
 * live here -- what happens without a pixel size, which device inference picks,
 * how a confirmed area treats unconfirmed pixels -- belong in the dialog where
 * the user is making that choice, not in an overview they read once. Several of
 * them were also describing internal decisions rather than anything the reader
 * has to act on.
 */

import { Button, Panel } from "@/shared/ui/design";
import { rememberWorkflowGuideDismissed } from "@/features/library/components/workflowGuideStorage";

interface WorkflowStep {
  title: string;
  body: string;
}

const STEPS: WorkflowStep[] = [
  {
    title: "1. Import",
    body: "Add a local image.",
  },
  {
    title: "2. View and segment",
    body: "Pick an organelle and model size.",
  },
  {
    title: "3. Proofread",
    body: "Confirm objects or modify segmentation outputs.",
  },
  {
    title: "4. Analyse",
    body: "Measure composition and object statistics.",
  },
  {
    title: "5. Fine-tune (optional)",
    body: "Fine-tune a model on your own confirmed areas.",
  },
];

export function WorkflowGuide({ onDismiss }: { onDismiss: () => void }) {
  return (
    <Panel className="p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="m-0 text-base font-semibold text-slate-950">
            How QuantEM works
          </h2>
        </div>
        <Button
          size="sm"
          onClick={() => {
            rememberWorkflowGuideDismissed();
            onDismiss();
          }}
        >
          Got it
        </Button>
      </div>
      <ol className="m-0 mt-4 grid list-none grid-cols-1 gap-3 p-0 md:grid-cols-2 xl:grid-cols-5">
        {STEPS.map((step) => (
          <li
            key={step.title}
            className="rounded-md border border-slate-200 bg-slate-50 p-3"
          >
            <p className="m-0 text-sm font-semibold text-slate-900">{step.title}</p>
            <p className="m-0 mt-1 text-xs leading-relaxed text-slate-600">
              {step.body}
            </p>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
