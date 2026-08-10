import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  makeErSegmentation,
  makeImage,
  makeSegmentation,
  setupSegmentationScreenTest,
} from "@/features/segmentation/SegmentationScreen.testUtils";
import { useSegmentationRouteState } from "@/features/segmentation/screen/hooks/useSegmentationRouteState";
import { getAsset, getAssetSegmentations } from "@/shared/api/assets";

function makeWrapper(initialEntry: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route
            path="/assets/:assetId/labeling/:segmentationTypeName"
            element={<>{children}</>}
          />
        </Routes>
      </MemoryRouter>
    );
  };
}

const SOURCE_MODEL_OPTIONS = [
  {
    value: "quantem:mito",
    label: "QuantEM",
    model_family: "quantem",
    variant: "",
    is_default: true,
    count: 0,
  },
  {
    value: "omniem:mito",
    label: "OmniEM",
    model_family: "omniem",
    variant: "",
    is_default: false,
    count: 34,
  },
  {
    value: "manual",
    label: "Manual",
    model_family: "manual",
    variant: "",
    is_default: false,
    count: 0,
  },
];

describe("useSegmentationRouteState", () => {
  beforeEach(() => {
    setupSegmentationScreenTest();
    window.localStorage.clear();
  });

  it("resolves the current segmentation from the route", async () => {
    vi.mocked(getAsset).mockResolvedValue(makeImage());
    vi.mocked(getAssetSegmentations).mockResolvedValue([makeSegmentation()]);

    const { result } = renderHook(() => useSegmentationRouteState(), {
      wrapper: makeWrapper("/assets/img-1/labeling/mitochondria"),
    });

    await waitFor(() => {
      expect(result.current.currentSegmentation?.id).toBe("seg-1");
    });

    expect(result.current.isErSegmentation).toBe(false);
    expect(result.current.isTissueSegmentation).toBe(false);
    expect(result.current.supportsPointFeedback).toBe(true);
    expect(result.current.preprocessReady).toBe(true);
  });

  it("reports ER flags and preprocess status from the fetched image", async () => {
    vi.mocked(getAsset).mockResolvedValue(
      makeImage({
        preprocess_stage: "ENCODING",
        preprocess_progress: 47,
      })
    );
    vi.mocked(getAssetSegmentations).mockResolvedValue([makeErSegmentation()]);

    const { result } = renderHook(() => useSegmentationRouteState(), {
      wrapper: makeWrapper("/assets/img-1/labeling/Endoplasmic%20Reticulum"),
    });

    await waitFor(() => {
      expect(result.current.currentSegmentation?.segmentation_type.long_name).toBe(
        "Endoplasmic Reticulum"
      );
    });

    expect(result.current.isErSegmentation).toBe(true);
    expect(result.current.preprocessReady).toBe(false);
    expect(result.current.preprocessLabel).toBe("ENCODING (47%)");
  });

  /**
   * uat13 #4: reopening the labeling screen reset the family toggle to the
   * app default (QuantEM is `is_default` on every organelle) even when every
   * object came from an OmniEM run — 34 fresh candidates invisible behind
   * "No objects from QuantEM yet".
   */
  describe("family toggle default", () => {
    it("defaults to the family that owns the objects, not is_default", async () => {
      vi.mocked(getAsset).mockResolvedValue(makeImage());
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        makeSegmentation({ source_models: SOURCE_MODEL_OPTIONS }),
      ]);

      const { result } = renderHook(() => useSegmentationRouteState(), {
        wrapper: makeWrapper("/assets/img-1/labeling/mitochondria"),
      });

      await waitFor(() => {
        expect(result.current.activeSourceModel).toBe("omniem:mito");
      });
    });

    it("still honours an explicit ?source_model= over the objects", async () => {
      vi.mocked(getAsset).mockResolvedValue(makeImage());
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        makeSegmentation({ source_models: SOURCE_MODEL_OPTIONS }),
      ]);

      const { result } = renderHook(() => useSegmentationRouteState(), {
        wrapper: makeWrapper(
          "/assets/img-1/labeling/mitochondria?source_model=quantem%3Amito"
        ),
      });

      await waitFor(() => {
        expect(result.current.activeSourceModel).toBe("quantem:mito");
      });
    });

    it("uses the family the user last chose for this segmentation when both own objects", async () => {
      window.localStorage.setItem(
        "quantem.labeling.source-model.seg-1",
        "omniem:mito"
      );
      vi.mocked(getAsset).mockResolvedValue(makeImage());
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        makeSegmentation({
          source_models: SOURCE_MODEL_OPTIONS.map((option) => ({
            ...option,
            count: option.value === "manual" ? 0 : 17,
          })),
        }),
      ]);

      const { result } = renderHook(() => useSegmentationRouteState(), {
        wrapper: makeWrapper("/assets/img-1/labeling/mitochondria"),
      });

      await waitFor(() => {
        expect(result.current.activeSourceModel).toBe("omniem:mito");
      });
    });

    it("remembers an explicit toggle change per segmentation", async () => {
      vi.mocked(getAsset).mockResolvedValue(makeImage());
      vi.mocked(getAssetSegmentations).mockResolvedValue([
        makeSegmentation({ source_models: SOURCE_MODEL_OPTIONS }),
      ]);

      const { result } = renderHook(() => useSegmentationRouteState(), {
        wrapper: makeWrapper("/assets/img-1/labeling/mitochondria"),
      });
      await waitFor(() => {
        expect(result.current.currentSegmentation?.id).toBe("seg-1");
      });

      act(() => {
        result.current.handleSourceModelChange("manual");
      });

      expect(
        window.localStorage.getItem("quantem.labeling.source-model.seg-1")
      ).toBe("manual");
    });
  });
});
