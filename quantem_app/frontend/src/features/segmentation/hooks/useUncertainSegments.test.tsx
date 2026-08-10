import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useUncertainSegments } from "@/features/segmentation/hooks/useUncertainSegments";
import { getUncertainSegments } from "@/shared/api/segmentations/annotations";
import type { ImageSegmentation } from "@/shared/types/images";

vi.mock("@/shared/api/segmentations/annotations", async () => {
  const actual = await vi.importActual<
    typeof import("@/shared/api/segmentations/annotations")
  >(
    "@/shared/api/segmentations/annotations"
  );
  return {
    ...actual,
    getUncertainSegments: vi.fn(),
  };
});

function makeSegmentation(): ImageSegmentation {
  return {
    id: "seg-1",
    asset: "img-1",
    segmentation_type: {
      id: "type-1",
      internal_name: "quantem_internal_er",
      short_name: "ER",
      long_name: "Endoplasmic Reticulum",
      default_color: null,
      tags: [],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    segment_counts: { CONFIRMED: 0, CANDIDATE: 0 },
    status_stage: "CANDIDATES_READY",
    status_progress: 100,
    status_error: null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    config: {
      supports_instance_params: false,
      instance_params: null,
    },
  };
}

describe("useUncertainSegments", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not fetch outside uncertain mode", async () => {
    const { result } = renderHook(() =>
      useUncertainSegments(makeSegmentation(), "review", 50)
    );

    await waitFor(() => {
      expect(result.current.uncertainSegments).toEqual([]);
    });
    expect(getUncertainSegments).not.toHaveBeenCalled();
  });

  it("fetches uncertain segments in uncertain mode", async () => {
    vi.mocked(getUncertainSegments).mockResolvedValue([
      {
        id: "segment-1",
        segmentation: "seg-1",
        label_state: "CANDIDATE",
        confidence_score: 0.51,
        geometry_coords: [],
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      },
    ]);

    const { result } = renderHook(() =>
      useUncertainSegments(makeSegmentation(), "uncertain", 25)
    );

    await waitFor(() => {
      expect(result.current.uncertainSegments).toHaveLength(1);
    });
    expect(getUncertainSegments).toHaveBeenCalledWith("seg-1", 25, undefined);
  });
});
