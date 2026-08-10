/**
 * What marking an area complete will do to the candidates inside it.
 *
 * A completed area is the one hard precondition for guided fine-tuning, and it
 * works by declaring that inside the polygon, anything not confirmed is true
 * background. That is the rule that makes a Dice mean anything -- and it is
 * also how 88 model candidates get silently turned into negative examples by a
 * single click. The dialog always stated the rule correctly and never once said
 * how many objects it was about to apply it to.
 *
 * The count is the number of CANDIDATE/INFERRED objects whose geometry
 * intersects the drawn polygon, which is exactly the set
 * `_confirmed_objects_in` will leave out of the training target.
 */

import type { PendingBackgroundCount } from "@/features/segmentation/screen/hooks/useCompletedRoiWorkflow";

export interface CompletedRoiBackgroundNoticeProps {
  pending: PendingBackgroundCount;
}

export function CompletedRoiBackgroundNotice({
  pending,
}: CompletedRoiBackgroundNoticeProps) {
  if (pending.status === "loading") {
    return <p>Counting the candidates inside this area…</p>;
  }

  if (pending.status === "error") {
    // Saying "0" here would be a lie about training data, so it says nothing
    // and admits the gap instead.
    return (
      <p>
        The candidates inside this area could not be counted, so this dialog
        cannot say how many will become background. Everything inside that you
        have not confirmed will be treated as background either way.
      </p>
    );
  }

  if (pending.status !== "ready" || pending.count === null) {
    return null;
  }

  if (pending.count === 0) {
    return (
      <p>
        <strong>No unconfirmed candidates</strong> sit inside this area, so
        nothing will be turned into background by saving it.
      </p>
    );
  }

  const plural = pending.count === 1 ? "" : "s";
  return (
    <>
      <p>
        <strong>
          {pending.count} unconfirmed candidate{plural}
        </strong>{" "}
        {pending.count === 1 ? "sits" : "sit"} inside this area. Saving it marks{" "}
        {pending.count === 1 ? "that object" : "all of them"} as background, and
        fine-tuning will train against {pending.count === 1 ? "it" : "them"} as
        negative examples.
      </p>
      <p>
        Confirm the ones that are real objects first if that is not what you
        want.
      </p>
    </>
  );
}
