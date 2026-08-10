/**
 * The composition of the ground truth, next to the score it produced.
 *
 * The wizard is careful about every other way a number can mislead — split
 * mode, fitted-vs-held-out crops, the oracle as a ceiling — and said nothing
 * about the one that mattered most here: 86 of 90 "annotations" were the
 * model's own candidates that the user confirmed. A held-out Dice measured
 * against mostly-self-generated labels is close to a self-agreement score, and
 * the wizard was reporting it as if it were independent evidence.
 *
 * Stated as composition rather than as a verdict. Confirming model output is
 * the intended workflow and is not cheating; it just changes what the number
 * means, and that is the sentence the caption needs.
 */

import { Panel } from "@/shared/ui/design";
import { formatNumber } from "@/shared/ui/format";
import {
  modelDerivedFraction,
  needsSelfAgreementCaveat,
  type GroundTruthProvenance,
} from "@/features/finetune/groundTruthProvenance";

export interface GroundTruthProvenancePanelProps {
  provenance: GroundTruthProvenance | null;
  loading: boolean;
  error: string | null;
  /** Rendered inside an existing Panel when false. */
  standalone?: boolean;
}

export function GroundTruthProvenancePanel({
  provenance,
  loading,
  error,
  standalone = true,
}: GroundTruthProvenancePanelProps) {
  const body = (
    <>
      <h3 className="m-0 text-sm font-semibold text-slate-900">
        What the ground truth is made of
      </h3>

      {loading ? (
        <p className="m-0 mt-2 text-sm text-slate-600">
          Counting the annotations behind this score…
        </p>
      ) : error ? (
        <p className="m-0 mt-2 text-sm text-amber-800">
          The annotations behind this score could not be counted ({error}), so
          this panel cannot say how much of the ground truth came from the model
          itself.
        </p>
      ) : !provenance || provenance.regions === 0 ? (
        <p className="m-0 mt-2 text-sm text-slate-600">
          No completed area was found for this segmentation, so there is nothing
          to break down.
        </p>
      ) : (
        <ProvenanceBody provenance={provenance} />
      )}
    </>
  );

  return standalone ? <Panel className="p-4">{body}</Panel> : body;
}

function ProvenanceBody({ provenance }: { provenance: GroundTruthProvenance }) {
  const fraction = modelDerivedFraction(provenance);
  const caveat = needsSelfAgreementCaveat(provenance);

  return (
    <>
      <p className="m-0 mt-1 text-xs text-slate-500">
        Counted inside the {provenance.regions} completed{" "}
        {provenance.regions === 1 ? "area" : "areas"} the run trained on.
      </p>

      <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat
          label="Confirmed from model"
          value={provenance.confirmedFromModel}
          detail="the model proposed it, you accepted it"
        />
        <Stat
          label="Drawn by hand"
          value={provenance.drawnByHand}
          detail="independent of the model"
        />
        <Stat
          label="Rejected"
          value={provenance.rejected}
          detail="explicitly marked not an object"
        />
        <Stat
          label="Positives in total"
          value={provenance.totalConfirmed}
          detail="what the Dice is measured against"
        />
      </div>

      {fraction !== null ? (
        <p className="m-0 mt-3 text-sm text-slate-700">
          <span className="font-semibold tabular-nums">
            {formatNumber(fraction * 100, 0)}%
          </span>{" "}
          of the positives started as this model&apos;s own output.
        </p>
      ) : null}

      {caveat ? (
        <div className="mt-3 rounded-md border border-amber-200 bg-amber-50 p-3">
          <p className="m-0 text-xs font-semibold uppercase tracking-wide text-amber-800">
            Read before quoting the score
          </p>
          <p className="m-0 mt-2 text-sm text-amber-900">
            Almost all of the ground truth is the model&apos;s own candidates
            that you confirmed, so the held-out Dice is largely measuring the
            model against itself. It says the adaptation reproduces the
            decisions you agreed with — it does not say the model finds objects
            you never saw, because there are almost none of those in the
            reference.
          </p>
          <p className="m-0 mt-2 text-sm text-amber-900">
            To make the score independent evidence, draw a region from scratch:
            annotate every object in a completed area without starting from the
            model&apos;s proposals.
          </p>
        </div>
      ) : null}
    </>
  );
}

function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: number;
  detail: string;
}) {
  return (
    <div>
      <p className="m-0 text-xs uppercase tracking-wide text-slate-500">{label}</p>
      <p className="m-0 text-lg font-semibold tabular-nums text-slate-900">
        {value}
      </p>
      <p className="m-0 text-xs text-slate-500">{detail}</p>
    </div>
  );
}
