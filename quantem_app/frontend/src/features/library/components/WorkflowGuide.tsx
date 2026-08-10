/**
 * The intended workflow, written down.
 *
 * There was previously nothing in the app that said what the five screens are
 * for or in what order to use them, so a first-time user met a library, a
 * collapsed import panel and an empty state that pointed at a button that was
 * not visible. This shows on first launch and can be reopened from the header.
 *
 * The pixel-size step is called out because it is the one that silently decides
 * whether any of the downstream numbers can carry physical units.
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
    body:
      "Add a TIFF or PNG from this machine. If the file declares a pixel size it is read automatically; if not, type one. Without a pixel size every measurement stays in pixels.",
  },
  {
    title: "2. View and segment",
    body:
      "Open the image, pick an organelle and a model, and run it. Segmentation runs locally — on the GPU if there is one, otherwise on the CPU.",
  },
  {
    title: "3. Proofread",
    body:
      "Confirm and reject objects, and draw or erase to correct them. Then mark a confirmed area: inside it, anything you have not confirmed counts as background.",
  },
  {
    title: "4. Analyse",
    body:
      "Measure composition, object statistics, point enrichment and distances. Runs that ran on an uncalibrated image are labelled as such and report no µm².",
  },
  {
    title: "5. Adapt (optional)",
    body:
      "Fine-tune a model on your own confirmed areas. The held-out score is always shown with the split mode that produced it.",
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
          <p className="m-0 mt-1 text-sm text-slate-500">
            Everything runs on this machine. Nothing is uploaded anywhere.
          </p>
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
