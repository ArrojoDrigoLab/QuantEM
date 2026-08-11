/**
 * What the app says, and offers, for each class of failure it can hit.
 *
 * The backend names the class (`error_code`, catalogued in
 * `quantem/core/error_codes.py`); this file turns that name into three things
 * a red sentence on its own never had:
 *
 *  - **what happened**, in one line, in the app's voice;
 *  - **whose fault it is** — said plainly, because the alternative is a user
 *    who assumes it is theirs and stops. "The model pack is not installed"
 *    blames nobody; "could not read your image" names the file and not the
 *    person; "somebody stopped it" is not a failure at all and must not be
 *    dressed as one;
 *  - **what to do**, as a control that exists in this application. Not advice,
 *    not a command to type, not a support address. A link to a screen or a
 *    button the surface already renders.
 *
 * The server's own sentence is never thrown away. It is the only text that can
 * name *this* run's particulars — which pack, which file, which folder — so a
 * surface renders `body` (the class) and the server's sentence (the instance)
 * together. This file supplies the class.
 *
 * Two rules this file has to keep, both enforced elsewhere:
 *
 *  - **Every code in the enum has an entry, and every entry an action.**
 *    `quantem/core/tests/test_error_codes.py` parses this file and fails if a
 *    code is added on one side and not the other. A code that renders as a
 *    blank space is worse than no code.
 *  - **No string here contains a shell command, a route, an internal name or a
 *    path.** Invariant I-12. The code itself (`model_not_installed`) is a wire
 *    value and never appears in the words.
 */

/**
 * The wire values of `quantem.core.error_codes.ErrorCode`.
 *
 * Written out rather than generated: the backend enum is the authority, and the
 * source test that reads both files is what keeps this list honest. A generated
 * file would move the failure from a test to a build step nobody runs.
 */
export const FAILURE_CODES = [
  "model_not_installed",
  "out_of_memory",
  "no_pixel_size",
  "image_unreadable",
  "disk_full",
  "cancelled",
  "probability_map_missing",
  "image_too_large_for_memory",
  "duplicate_image",
  "upload_too_large",
] as const;

export type FailureCode = (typeof FAILURE_CODES)[number];

/** Where an action takes the user, or what it asks the surface to do. */
export interface FailureAction {
  /** The control's label. Imperative, and specific to this failure. */
  label: string;
  /**
   * A hash route inside this application, for actions that are "go there".
   * Absent when the action is a control the surface itself renders.
   */
  href?: string;
  /**
   * A control the *surface* owns, when the fix is here rather than elsewhere.
   * The surface decides how to render it; this file only says which one is the
   * right offer for this failure.
   */
  control?: "retry" | "run-region" | "run-inference" | "set-pixel-size" | "choose-file";
}

export interface FailureCopy {
  /** One line. What happened, as a statement of fact. */
  headline: string;
  /** Two sentences at most: whose fault, and why it happened. */
  body: string;
  /** The one thing to do next. Always present. */
  action: FailureAction;
  /**
   * True when the failure is not a fault and the copy must not apologise.
   * A cancellation is a decision somebody made; styling it as an error and
   * saying "sorry" teaches users to distrust the ones that are real.
   */
  benign?: boolean;
}

export const FAILURE_COPY: Record<FailureCode, FailureCopy> = {
  model_not_installed: {
    headline: "This model is not installed on this computer.",
    body:
      "Nothing is wrong with your image or your work. QuantEM ships without model weights, and this one has not been added yet — or it was added and cannot be built here.",
    action: { label: "Open Models", href: "#/models" },
  },
  out_of_memory: {
    headline: "This computer ran out of memory partway through.",
    body:
      "The image was larger than the memory available while it ran. Nothing already saved was lost. Closing other applications, or running over a region instead of the whole image, usually gets it through.",
    action: { label: "Run on a region instead", control: "run-region" },
  },
  no_pixel_size: {
    headline: "This image has no pixel size yet.",
    body:
      "Most models were trained at a fixed number of nanometres per pixel and have to rescale your image to match, so they cannot start without it. This is a stop before the run, not a bad result after one.",
    action: { label: "Set the pixel size", control: "set-pixel-size" },
  },
  image_unreadable: {
    headline: "This file could not be opened as an image.",
    body:
      "The bytes are truncated, or the file is not the format its name suggests. This is about the file itself, not about anything you did in QuantEM.",
    action: { label: "Choose a different file", control: "choose-file" },
  },
  disk_full: {
    headline: "There is no room left on the drive QuantEM stores data on.",
    body:
      "The write stopped partway, so this result was not saved. Your existing images and objects are untouched. Free some space on that drive and run it again.",
    action: { label: "Try again", control: "retry" },
  },
  cancelled: {
    headline: "Stopped, because it was asked to stop.",
    body:
      "Nothing failed. Work that had already been saved is still saved; anything the run had not finished was discarded.",
    action: { label: "Start it again", control: "retry" },
    benign: true,
  },
  probability_map_missing: {
    headline:
      "The confidence map this image was segmented from is no longer stored.",
    body:
      "Moving the accuracy dial re-reads that map, so without it the threshold cannot change on its own. Running inference again rebuilds it, and your confirmed and rejected objects are kept.",
    action: { label: "Run inference again", control: "run-inference" },
  },
  image_too_large_for_memory: {
    headline: "This image is too large to do all at once on this computer.",
    body:
      "QuantEM worked out the memory it would need before starting, rather than crashing partway. The image is fine; it needs to be done a region at a time here.",
    action: { label: "Run on a region instead", control: "run-region" },
  },
  duplicate_image: {
    headline: "This image is already in your library.",
    body:
      "The file you chose is byte-for-byte one that was imported before, so nothing was added. The copy already there has whatever segmentations and labels you have done on it.",
    action: { label: "Open the copy you already have", href: "#/" },
  },
  upload_too_large: {
    headline: "This file is larger than this installation will accept at once.",
    body:
      "The limit is a setting of this install, not a property of your image. Importing fewer files in one go, or raising the limit for this installation, both work.",
    action: { label: "Choose a different file", control: "choose-file" },
  },
};

/** True when `value` is a code this build knows how to talk about. */
export function isFailureCode(value: unknown): value is FailureCode {
  return (
    typeof value === "string" &&
    (FAILURE_CODES as readonly string[]).includes(value)
  );
}

/**
 * The `error_code` carried by an API payload, if it carries one this build
 * knows.
 *
 * Takes `unknown` on purpose. Codes arrive on several differently-shaped
 * bodies — an error response, a job row, a segmentation — and a surface should
 * be able to ask this question without every one of those types having to
 * declare the field first. An unknown or absent code returns `null`, and the
 * caller falls back to rendering the server's sentence alone, which is exactly
 * what every surface did before codes existed.
 */
export function readFailureCode(value: unknown): FailureCode | null {
  if (!value || typeof value !== "object") return null;
  const code = (value as { error_code?: unknown }).error_code;
  return isFailureCode(code) ? code : null;
}

/** The copy for a code, or `null` when there is no code to look up. */
export function failureCopy(code: unknown): FailureCopy | null {
  return isFailureCode(code) ? FAILURE_COPY[code] : null;
}
