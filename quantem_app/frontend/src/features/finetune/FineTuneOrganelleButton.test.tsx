import { render, screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { server } from "@/test/msw/server";
import type { AssetDetail, ImageSegmentation } from "@/shared/types";

import { FineTuneOrganelleButton } from "./FineTuneOrganelleButton";

const BASE = "http://127.0.0.1:8000";
const image = { id: "asset-1" } as AssetDetail;
const segmentation = {
  id: "seg-1",
  segmentation_type: {
    id: "type-1",
    short_name: "Mitochondria",
    long_name: "Mitochondria",
  },
} as ImageSegmentation;

describe("FineTuneOrganelleButton", () => {
  it("rechecks eligibility when a completed area or ROI changes", async () => {
    let eligible = false;
    server.use(
      http.post(`${BASE}/api/finetune/preview/`, () =>
        HttpResponse.json({ annotation_count: eligible ? 1 : 0 })
      )
    );

    const view = render(
      <FineTuneOrganelleButton
        image={image}
        currentSegmentation={segmentation}
        eligibilityRevision=""
      />
    );

    const button = await screen.findByTestId("finetune-organelle-button");
    await waitFor(() => expect(button).toBeDisabled());
    expect(button).toHaveAttribute(
      "title",
      "Mark an ROI as Done or outline a Confirmed area you have annotated to enable fine-tuning"
    );

    eligible = true;
    view.rerender(
      <FineTuneOrganelleButton
        image={image}
        currentSegmentation={segmentation}
        eligibilityRevision="roi-1"
      />
    );

    await waitFor(() => expect(button).toBeEnabled());
  });
});
