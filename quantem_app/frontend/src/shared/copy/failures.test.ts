/**
 * The failure catalogue's client half: every code readable, every entry usable.
 *
 * The Python side (`quantem/core/tests/test_error_codes.py`) proves the two
 * lists of codes are the same list and that every entry has the three fields.
 * This file proves the *behaviour* the surfaces rely on, which Python cannot
 * see: that an unknown code degrades to "no copy" rather than to a crash or a
 * blank, and that the words themselves keep the promises the module docstring
 * makes about them.
 */

import { describe, expect, it } from "vitest";
import {
  FAILURE_CODES,
  FAILURE_COPY,
  failureCopy,
  isFailureCode,
  readFailureCode,
} from "./failures";

describe("reading a code off a payload", () => {
  it("finds one wherever the field is", () => {
    expect(readFailureCode({ error_code: "disk_full" })).toBe("disk_full");
    expect(
      readFailureCode({ detail: "…", error_code: "model_not_installed" })
    ).toBe("model_not_installed");
  });

  it("returns null rather than throwing on anything else", () => {
    // Every shape a surface can plausibly hand it, including the ones a
    // defensive read exists for.
    expect(readFailureCode(null)).toBeNull();
    expect(readFailureCode(undefined)).toBeNull();
    expect(readFailureCode("disk_full")).toBeNull();
    expect(readFailureCode(42)).toBeNull();
    expect(readFailureCode({})).toBeNull();
    expect(readFailureCode({ error_code: null })).toBeNull();
    expect(readFailureCode({ error_code: 7 })).toBeNull();
  });

  it("treats a code this build has never heard of as no code at all", () => {
    // The compatibility case: a newer backend naming a class this frontend
    // does not know. Rendering the server's sentence alone is right; rendering
    // an empty box where the explanation should be is not.
    expect(readFailureCode({ error_code: "quantum_flux_inversion" })).toBeNull();
    expect(failureCopy("quantum_flux_inversion")).toBeNull();
    expect(isFailureCode("quantum_flux_inversion")).toBe(false);
  });
});

describe("the copy itself", () => {
  it("offers a control that exists in this application", () => {
    for (const code of FAILURE_CODES) {
      const { action } = FAILURE_COPY[code];
      expect(action.label.length, code).toBeGreaterThan(0);
      // Exactly one kind of action, so a surface never has to choose.
      expect(Boolean(action.href) !== Boolean(action.control), code).toBe(true);
      if (action.href) expect(action.href.startsWith("#/"), code).toBe(true);
    }
  });

  it("never tells the reader to type anything", () => {
    // The register check the Python gate cannot make on prose alone: no
    // backticks, no flags, no file extensions, no capitalised class names.
    for (const code of FAILURE_CODES) {
      const { headline, body } = FAILURE_COPY[code];
      for (const text of [headline, body]) {
        expect(text, code).not.toMatch(/`|--[a-z]|\.py\b|\bAPI\b/);
      }
    }
  });

  it("does not apologise for a cancellation", () => {
    // A cancellation is a decision somebody made. Dressing it as a fault is
    // how a user learns to ignore the notices that are real.
    const cancelled = FAILURE_COPY.cancelled;
    expect(cancelled.benign).toBe(true);
    // "Nothing failed" is the opposite of an apology and is the point of the
    // entry, so the word itself is allowed; the register is what is asserted.
    expect(`${cancelled.headline} ${cancelled.body}`).not.toMatch(
      /sorry|apolog|went wrong|unable|error/i
    );
    expect(cancelled.body).toMatch(/Nothing failed/);
    expect(cancelled.body).toMatch(/still saved/i);
  });

  it("takes the blame off the user where the fault is not theirs", () => {
    // The one thing this package exists for: a self-blaming user who quits.
    expect(FAILURE_COPY.model_not_installed.body).toMatch(
      /Nothing is wrong with your image or your work/
    );
    expect(FAILURE_COPY.out_of_memory.body).toMatch(/Nothing already saved/);
    expect(FAILURE_COPY.image_unreadable.body).toMatch(
      /not about anything you did/
    );
  });

  it("keeps the user's own work out of every remedy", () => {
    // The invariant that outranks all of this copy: a model pass, a retry or a
    // re-run never destroys a manual correction. The one code whose remedy is
    // "run inference again" has to say so where the user reads it.
    expect(FAILURE_COPY.probability_map_missing.body).toMatch(
      /confirmed and rejected objects are kept/
    );
  });
});
