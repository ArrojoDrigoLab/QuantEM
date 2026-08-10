/**
 * The sentences that say what running uncalibrated costs.
 *
 * Shared by all three gates into an inference run — the create-segmentation
 * dialog, "Run Full Segmentation", and the import form's organelle checkboxes —
 * so they cannot drift apart. The biologist praised this wording on the first
 * door while the second had no dialog at all and the third, the one everybody
 * starts on, said only that units stay in pixels. The fix is the same wording
 * everywhere, not a third wording.
 *
 * The load-bearing claim lives in {@link UncalibratedObjectSetNotice} and is
 * deliberately about *which objects exist*, not about their units. The units
 * framing is true and was already on the import form, and it is what let five
 * images get segmented uncalibrated on first contact: a reader who believes the
 * only cost is "µm² instead of px²" reasonably concludes the objects are the
 * same objects and the scale can be set afterwards. It cannot. The pack
 * resamples the image to its working resolution *before* it looks for anything,
 * so the pixel size is part of the input, and the same crop imported at no
 * pixel size, at 5 nm/px and at 10 nm/px comes back with three different object
 * sets — not three views of one.
 */

import { formatPixelSizeNm } from "@/shared/pixelSize";
import type { ScaleMismatch } from "@/features/models/scaleMismatch";

/**
 * What actually changes. One paragraph, one place, every door.
 *
 * Says "completely" rather than putting a multiplier on it: the honest summary
 * of the measured range is that the count is not a stable quantity across pixel
 * sizes, and any figure quoted here would be read as a bound it is not.
 */
export function UncalibratedObjectSetNotice() {
  return (
    <p>
      <strong>
        This changes which objects exist, not just the units they are reported
        in.
      </strong>{" "}
      The model resamples the image to its working resolution before it looks
      for anything, so the pixel size is part of what it is given. The same
      image run at the wrong scale can come back with a completely different
      number of objects — far more, far fewer, or none at all — and they are not
      the same objects measured differently. Setting the pixel size afterwards
      relabels the units; it does not recover the objects.
    </p>
  );
}

/** `"quantem:mito (8 nm/px)"`. */
function describePack(mismatch: ScaleMismatch): string {
  return `${mismatch.packId} (${formatPixelSizeNm(mismatch.canonicalNm)})`;
}

export function UncalibratedScaleWarning({
  mismatch,
}: {
  mismatch: ScaleMismatch;
}) {
  return (
    <>
      <p>
        <strong>This image has no pixel size</strong>, so {mismatch.packId} will
        run at native scale instead of the{" "}
        {formatPixelSizeNm(mismatch.canonicalNm)} it was trained at. The objects
        it finds will be whatever that mismatch produces, and no measurement
        taken from them can be reported in µm². Set the pixel size on this
        screen first if you know it.
      </p>
      <UncalibratedObjectSetNotice />
    </>
  );
}

/**
 * The same warning for a screen that queues several runs at once.
 *
 * The import form's four checkboxes are four separate inference passes, and
 * they do not all declare a working resolution — ER runs at native scale by
 * design and is filtered out upstream, so an import with only ER ticked gets no
 * warning at all. Naming the packs that *are* affected keeps the sentence true
 * when only some of the boxes matter.
 *
 * `certain` is the difference between a fact and a possibility, and it earns its
 * own wording. The import form decides an image's calibration before the image
 * exists, from a file it may not be able to read: the box may be blank because
 * the *file* carries the value, which is what the helper text tells people to
 * do. Asserting "will run uncalibrated" over that import is how the warning
 * that matters most becomes the one people click through, so an unread file gets
 * an "if" and a button that does not assert either.
 */
export function UncalibratedImportWarning({
  mismatches,
  certain = true,
  unreadableFileName = null,
}: {
  mismatches: ScaleMismatch[];
  /** True only when the file has been read and carries no pixel size. */
  certain?: boolean;
  /** The file this form tried and failed to read a pixel size out of. */
  unreadableFileName?: string | null;
}) {
  if (mismatches.length === 0) return null;
  const packs = mismatches.map(describePack).join(", ");
  const plural = mismatches.length !== 1;

  if (!certain) {
    return (
      <div className="upload-scale-warning" role="status">
        <p>
          <strong>
            If no pixel size arrives with this image, {packs} will run at the
            wrong scale.
          </strong>{" "}
          {plural ? "They resample" : "It resamples"} to a fixed working
          resolution and fall{plural ? "" : "s"} back to the image's native
          scale when there is none to resample to. A value typed above is used
          whatever the file says; left blank, the file's own value is used when
          it has one.
          {unreadableFileName
            ? ` This form could not read one out of ${unreadableFileName}, so it cannot tell you which of those this will be.`
            : ""}
        </p>
        <UncalibratedObjectSetNotice />
        <p>
          Type the pixel size above if you know it. Setting it after the import
          does not re-run anything segmented now.
        </p>
      </div>
    );
  }

  return (
    <div className="upload-scale-warning" role="status">
      <p>
        <strong>No pixel size, and {plural ? "models" : "a model"} that
        need{plural ? "" : "s"} one.</strong>{" "}
        {packs} will run at this image's native
        scale instead of the resolution {plural ? "they were" : "it was"}{" "}
        trained at.
      </p>
      <UncalibratedObjectSetNotice />
      <p>
        Type the pixel size above if you know it, or untick the organelles and
        run them from the image once it is calibrated. Nothing is lost by
        waiting.
      </p>
    </div>
  );
}
