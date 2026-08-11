import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  HeldoutDice,
  OracleCeiling,
  SplitModeNote,
} from "@/features/improve/components/HonestScore";

describe("HeldoutDice (honesty rule 1)", () => {
  it("never renders a score without its split mode", () => {
    render(<HeldoutDice value={0.871} mode="image-disjoint" />);
    expect(screen.getByText("0.871")).toBeInTheDocument();
    expect(screen.getByText("image-disjoint")).toBeInTheDocument();
  });

  it("labels a within-image score as within-image", () => {
    render(<HeldoutDice value={0.9} mode="within-image" />);
    expect(screen.getByText("0.900")).toBeInTheDocument();
    expect(screen.getByText("within-image")).toBeInTheDocument();
  });

  it("shows no number at all when there was no held-out data", () => {
    render(<HeldoutDice value={0.95} mode="no-heldout" />);
    expect(screen.queryByText("0.950")).not.toBeInTheDocument();
    expect(screen.getByText("—")).toBeInTheDocument();
    expect(screen.getByText("no held-out data")).toBeInTheDocument();
  });
});

describe("SplitModeNote", () => {
  it("says a within-image score does not measure generalisation", () => {
    render(<SplitModeNote mode="within-image" />);
    expect(
      screen.getByText(/does not measure generalisation to a new image/)
    ).toBeInTheDocument();
  });

  it("says an image-disjoint score does", () => {
    render(<SplitModeNote mode="image-disjoint" />);
    expect(
      screen.getByText(/measures generalisation to a new image/)
    ).toBeInTheDocument();
  });
});

describe("OracleCeiling (honesty rule 3)", () => {
  it("presents the oracle as a ceiling, not a target", () => {
    render(<OracleCeiling value={0.93} />);
    expect(screen.getByText(/not a target/i)).toBeInTheDocument();
    expect(screen.getByText("0.930")).toBeInTheDocument();
    expect(screen.getByText(/chosen using the answers/)).toBeInTheDocument();
  });

  it("renders nothing when no oracle was computed", () => {
    const { container } = render(<OracleCeiling value={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
