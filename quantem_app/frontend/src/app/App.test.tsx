/**
 * Routing, only where it is user-facing.
 *
 * The catch-all route used to be `<Navigate to="/" replace />`: a nonsense
 * address became the library with nothing said. From the reader's side that is
 * indistinguishable from the app losing their place, and it is the friendlier
 * half of the same defect that leaves a path-style nested address rendering an
 * entirely blank page.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MemoryRouter } from "react-router-dom";
import App from "@/app/App";

const NONSENSE = "/assets/8f14e45f-ceea-467a-9c1e-1b1d2f1b1a11/viewer/deeper/still";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>
  );
}

describe("an address the router does not know", () => {
  it("says so, rather than silently becoming the library", async () => {
    renderAt(NONSENSE);

    expect(
      await screen.findByRole("heading", { name: "This page does not exist" })
    ).toBeInTheDocument();
  });

  it("offers a way back", async () => {
    renderAt(NONSENSE);
    await screen.findByTestId("not-found-screen");

    expect(
      screen.getByRole("link", { name: "Back to the library" })
    ).toHaveAttribute("href", "#/");
  });

  it("says the work is safe, because a strange page is when that is doubted", async () => {
    renderAt(NONSENSE);
    const panel = await screen.findByTestId("not-found-screen");

    expect(panel.textContent).toContain("Nothing has happened to your images");
  });

  it("does not read the address back, identifiers and all", async () => {
    // The obvious version of this page quotes what you typed. On this app that
    // is `#/assets/<uuid>/…`, and a raw identifier is both useless to a person
    // and forbidden in user-facing copy.
    const { container } = renderAt(NONSENSE);
    await screen.findByTestId("not-found-screen");

    expect(container.textContent).not.toContain("8f14e45f");
    expect(container.textContent).not.toContain("/assets/");
  });
});
