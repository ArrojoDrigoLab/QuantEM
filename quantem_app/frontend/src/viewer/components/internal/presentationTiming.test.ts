import { expect, it } from "vitest";
import { PresentationTiming } from "./presentationTiming";

it("times the matching displayed generation and does not restart on polling", () => {
  const timing = new PresentationTiming();
  timing.requested("old", 10);
  timing.requested("replacement", 20);
  timing.requested("replacement", 30);
  expect(timing.displayed("unknown", 40)).toBeUndefined();
  expect(timing.displayed("old", 50)).toBe(40);
  expect(timing.displayed("replacement", 80)).toBe(60);
  timing.requested("replacement", 90);
  expect(timing.displayed("replacement", 100)).toBeUndefined();
});
