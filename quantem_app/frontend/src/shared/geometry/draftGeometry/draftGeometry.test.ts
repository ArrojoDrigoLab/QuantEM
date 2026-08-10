import { describe, expect, it } from "vitest";
import {
  appendRepairSection,
  applySpliceContract,
  applySpliceContractsToBaseRing,
  buildAutomaticPrefillPolygons,
  buildPatchBBoxFromPoints,
  buildConfirmPolygons,
  canCloseDraftPolygon,
  closeActiveRepairSession,
  closeDraftPolygon,
  cutDraftPolygonsWithLasso,
  deleteLastDraftSegment,
  patchBBoxWithinLimit,
  projectPointToPolyline,
  resolveRepairStartAnchor,
  resolveDraftPolygonCloseability,
  resolveRepairSessionCloseability,
  validateAutomaticPatchSelection,
  type DraftPolygon,
} from "@/shared/geometry/draftGeometry";
import { cutClosedRingWithLasso } from "@/shared/geometry/draftGeometry/cut";
import {
  buildSpatialSegmentCandidatePairs,
  resolveSingleAcceptedPolygonRing,
  signedRingArea,
} from "@/shared/geometry/draftGeometry/shared";

function sampleLine(
  start: { x: number; y: number },
  end: { x: number; y: number },
  segments: number
) {
  return Array.from({ length: segments + 1 }, (_, index) => ({
    x: start.x + (((end.x - start.x) * index) / segments),
    y: start.y + (((end.y - start.y) * index) / segments),
  }));
}

function makeDenseRepairRemainingPath() {
  const segmentsPerEdge = 1600;
  return [
    ...sampleLine({ x: 20, y: 0 }, { x: 240, y: 0 }, segmentsPerEdge),
    ...sampleLine({ x: 240, y: 0 }, { x: 240, y: 240 }, segmentsPerEdge).slice(1),
    ...sampleLine({ x: 240, y: 240 }, { x: 0, y: 240 }, segmentsPerEdge).slice(1),
    ...sampleLine({ x: 0, y: 240 }, { x: 0, y: 0 }, segmentsPerEdge).slice(1),
    ...sampleLine({ x: 0, y: 0 }, { x: 10, y: 0 }, segmentsPerEdge).slice(1),
  ];
}

describe("draftGeometry", () => {
  it("closes an open polygon into a confirmable ring", () => {
    const polygons: DraftPolygon[] = [
      {
        id: "poly-1",
        closed: false,
        segments: [
          {
            id: "seg-1",
            kind: "section",
            endedOutsidePatch: false,
            points: [
              { x: 120, y: 220 },
              { x: 170, y: 260 },
              { x: 140, y: 300 },
            ],
          },
        ],
      },
    ];

    const closed = closeDraftPolygon(polygons, () => "closing-1");
    expect(closed[0]?.closed).toBe(true);
    const [ring] = buildConfirmPolygons(closed);
    expect(ring).toBeDefined();
    expect(ring?.[0]).toEqual([120, 220]);
    expect(ring?.at(-1)).toEqual([120, 220]);
    expect(new Set(ring?.map(([x, y]) => `${x}:${y}`))).toEqual(
      new Set(["120:220", "170:260", "140:300"])
    );
  });

  it("preserves outside-patch points in the raw confirm polygon", () => {
    const polygons: DraftPolygon[] = [
      {
        id: "poly-2",
        closed: true,
        segments: [
          {
            id: "seg-1",
            kind: "section",
            endedOutsidePatch: true,
            points: [
              { x: 160, y: 240 },
              { x: 70, y: 240 },
            ],
          },
          {
            id: "seg-2",
            kind: "section",
            endedOutsidePatch: false,
            points: [
              { x: 70, y: 340 },
              { x: 180, y: 340 },
            ],
          },
          {
            id: "closing-2",
            kind: "closing",
            endedOutsidePatch: false,
            points: [
              { x: 180, y: 340 },
              { x: 160, y: 240 },
            ],
          },
        ],
      },
    ];

    const [ring] = buildConfirmPolygons(polygons);
    expect(ring).toBeDefined();
    expect(ring).toEqual([
      [160, 240],
      [70, 240],
      [70, 340],
      [180, 340],
      [160, 240],
    ]);
  });

  it("reopens the last polygon when deleting its closing segment", () => {
    const polygons: DraftPolygon[] = [
      {
        id: "poly-3",
        closed: true,
        segments: [
          {
            id: "seg-1",
            kind: "section",
            endedOutsidePatch: false,
            points: [
              { x: 120, y: 220 },
              { x: 170, y: 260 },
              { x: 140, y: 300 },
            ],
          },
          {
            id: "closing-3",
            kind: "closing",
            endedOutsidePatch: false,
            points: [
              { x: 140, y: 300 },
              { x: 120, y: 220 },
            ],
          },
        ],
      },
    ];

    const updated = deleteLastDraftSegment(polygons);
    expect(updated[0]?.closed).toBe(false);
    expect(updated[0]?.segments).toHaveLength(1);
  });

  it("converts an automatic prefill ring into a closed confirmable polygon", () => {
    const polygons = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [120, 220],
            [170, 220],
            [170, 260],
            [120, 260],
            [120, 220],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );

    expect(polygons).toHaveLength(1);
    expect(polygons[0]?.closed).toBe(true);
    expect(polygons[0]?.segments).toHaveLength(2);
    expect(buildConfirmPolygons(polygons)).toEqual([
      [
        [120, 220],
        [170, 220],
        [170, 260],
        [120, 260],
        [120, 220],
      ],
    ]);
  });

  it("repairs a self-touching automatic prefill ring to one simple cycle", () => {
    const polygons = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [10, 10],
            [30, 10],
            [30, 20],
            [40, 20],
            [30, 20],
            [30, 30],
            [30, 40],
            [10, 40],
            [10, 10],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );

    const [ring] = buildConfirmPolygons(polygons);
    expect(ring).toBeDefined();
    expect(ring?.[0]).toEqual([10, 10]);
    expect(ring?.at(-1)).toEqual([10, 10]);
    expect(ring).toContainEqual([30, 20]);
    expect(ring).toContainEqual([30, 30]);
    expect(ring).not.toContainEqual([40, 20]);
  });

  it("does not explode candidate tracking for many disjoint segments in one spatial bucket", () => {
    const segments = Array.from({ length: 4000 }, (_, index) => ({
      start: { x: 12, y: index * 2 },
      end: { x: 12, y: (index * 2) + 1 },
    }));

    expect(
      buildSpatialSegmentCandidatePairs(segments, { cellSize: 100_000 })
    ).toEqual([]);
  });

  it("applies splice sections without changing points outside the selected bbox", () => {
    const result = applySpliceContract(
      [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
        { x: 0, y: 0 },
      ],
      {
        bbox: {
          x0: 20,
          y0: -30,
          x1: 80,
          y1: 20,
        },
        sections: [
          {
            source_path: [
              [20, 0],
              [80, 0],
            ],
            replacement_path: [
              [20, 0],
              [20, -20],
              [80, -20],
              [80, 0],
            ],
            start_anchor: {
              edge: "left",
              point: [20, 0],
              perimeter_offset: 50,
            },
            end_anchor: {
              edge: "right",
              point: [80, 0],
              perimeter_offset: 110,
            },
          },
        ],
      }
    );

    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") {
      return;
    }
    expect(result.ring).toContainEqual({ x: 100, y: 0 });
    expect(result.ring).toContainEqual({ x: 100, y: 100 });
    expect(result.ring).toContainEqual({ x: 20, y: -20 });
    expect(result.ring).toContainEqual({ x: 80, y: -20 });
    expect(result.ring).not.toContainEqual({ x: 105, y: 100 });
  });

  it("applies multiple splice sections in source-ring order", () => {
    const result = applySpliceContract(
      [
        { x: 40, y: 60 },
        { x: 160, y: 60 },
        { x: 160, y: 140 },
        { x: 40, y: 140 },
        { x: 40, y: 60 },
      ],
      {
        bbox: {
          x0: 80,
          y0: 40,
          x1: 120,
          y1: 160,
        },
        sections: [
          {
            source_path: [
              [80, 60],
              [120, 60],
            ],
            replacement_path: [
              [80, 60],
              [80, 50],
              [120, 50],
              [120, 60],
            ],
            start_anchor: {
              edge: "left",
              point: [80, 60],
              perimeter_offset: 20,
            },
            end_anchor: {
              edge: "right",
              point: [120, 60],
              perimeter_offset: 60,
            },
          },
          {
            source_path: [
              [120, 140],
              [80, 140],
            ],
            replacement_path: [
              [120, 140],
              [120, 150],
              [80, 150],
              [80, 140],
            ],
            start_anchor: {
              edge: "right",
              point: [120, 140],
              perimeter_offset: 140,
            },
            end_anchor: {
              edge: "left",
              point: [80, 140],
              perimeter_offset: 220,
            },
          },
        ],
      }
    );

    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") {
      return;
    }
    expect(result.ring).toContainEqual({ x: 80, y: 50 });
    expect(result.ring).toContainEqual({ x: 120, y: 50 });
    expect(result.ring).toContainEqual({ x: 120, y: 150 });
    expect(result.ring).toContainEqual({ x: 80, y: 150 });
    expect(result.ring).toContainEqual({ x: 160, y: 60 });
    expect(result.ring).toContainEqual({ x: 40, y: 140 });
  });

  it("applies full-border batch splice contracts against the same base ring", () => {
    const result = applySpliceContractsToBaseRing(
      [
        { x: 0, y: 0 },
        { x: 100, y: 0 },
        { x: 100, y: 100 },
        { x: 0, y: 100 },
        { x: 0, y: 0 },
      ],
      [
        {
          bbox: { x0: 20, y0: -20, x1: 80, y1: 20 },
          sections: [
            {
              source_path: [
                [20, 0],
                [80, 0],
              ],
              replacement_path: [
                [20, 0],
                [20, -10],
                [80, -10],
                [80, 0],
              ],
              start_anchor: null,
              end_anchor: null,
            },
          ],
        },
        {
          bbox: { x0: 20, y0: 80, x1: 80, y1: 120 },
          sections: [
            {
              source_path: [
                [80, 100],
                [20, 100],
              ],
              replacement_path: [
                [80, 100],
                [80, 110],
                [20, 110],
                [20, 100],
              ],
              start_anchor: null,
              end_anchor: null,
            },
          ],
        },
      ]
    );

    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") {
      return;
    }
    expect(result.ring).toContainEqual({ x: 20, y: -10 });
    expect(result.ring).toContainEqual({ x: 80, y: -10 });
    expect(result.ring).toContainEqual({ x: 80, y: 110 });
    expect(result.ring).toContainEqual({ x: 20, y: 110 });
    expect(result.ring).toContainEqual({ x: 100, y: 0 });
    expect(result.ring).toContainEqual({ x: 0, y: 100 });
  });

  it("keeps lasso cutting valid after applying a splice", () => {
    const splice = applySpliceContract(
      [
        { x: 40, y: 60 },
        { x: 160, y: 60 },
        { x: 160, y: 140 },
        { x: 40, y: 140 },
        { x: 40, y: 60 },
      ],
      {
        bbox: {
          x0: 80,
          y0: 40,
          x1: 120,
          y1: 160,
        },
        sections: [
          {
            source_path: [
              [80, 60],
              [120, 60],
            ],
            replacement_path: [
              [80, 60],
              [80, 50],
              [120, 50],
              [120, 60],
            ],
            start_anchor: {
              edge: "left",
              point: [80, 60],
              perimeter_offset: 20,
            },
            end_anchor: {
              edge: "right",
              point: [120, 60],
              perimeter_offset: 60,
            },
          },
        ],
      }
    );

    expect(splice.kind).toBe("ok");
    if (splice.kind !== "ok") {
      return;
    }

    const cutResult = cutClosedRingWithLasso(splice.ring, [
      { x: 75, y: 45 },
      { x: 125, y: 45 },
      { x: 125, y: 65 },
      { x: 75, y: 65 },
      { x: 75, y: 45 },
    ]);

    expect(cutResult.kind).toBe("ok");
  });

  it("resolves a self-intersecting ring to the largest accepted polygon", () => {
    const resolved = resolveSingleAcceptedPolygonRing([
      { x: 10, y: 10 },
      { x: 30, y: 10 },
      { x: 30, y: 20 },
      { x: 40, y: 20 },
      { x: 30, y: 20 },
      { x: 30, y: 30 },
      { x: 30, y: 40 },
      { x: 10, y: 40 },
      { x: 10, y: 10 },
    ]);

    expect(resolved).not.toBeNull();
    if (!resolved) {
      return;
    }

    expect(resolved[0]).toEqual(resolved.at(-1));
    expect(Math.abs(signedRingArea(resolved))).toBeCloseTo(600);
    expect(resolved).toContainEqual({ x: 30, y: 20 });
    expect(resolved).toContainEqual({ x: 30, y: 30 });
    expect(resolved).not.toContainEqual({ x: 40, y: 20 });
  });

  it("cuts a closed polygon with a lasso into an active repair session", () => {
    const [polygon] = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [10, 10],
            [30, 10],
            [30, 30],
            [10, 30],
            [10, 10],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );

    const result = cutDraftPolygonsWithLasso(
      [polygon],
      [
        { x: 14, y: 8 },
        { x: 26, y: 8 },
        { x: 26, y: 12 },
        { x: 14, y: 12 },
      ],
      { x: 14, y: 8 },
      (prefix) => `${prefix}-id`
    );

    expect(result.kind).toBe("ok");
    if (result.kind !== "ok") {
      return;
    }
    expect(result.polygons).toHaveLength(0);
    expect(result.repairSessions).toHaveLength(1);
    expect(result.repairSessions[0]?.active).toBe(true);
    expect(result.repairSessions[0]?.remainingPath.at(0)?.y).toBe(10);
    expect(result.repairSessions[0]?.remainingPath.at(-1)?.y).toBe(10);
  });

  it("rejects lasso cuts that would create multiple repair gaps", () => {
    const [polygon] = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [10, 10],
            [30, 10],
            [30, 30],
            [10, 30],
            [10, 10],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );

    const result = cutDraftPolygonsWithLasso(
      [polygon],
      [
        { x: 18, y: 5 },
        { x: 22, y: 5 },
        { x: 22, y: 35 },
        { x: 18, y: 35 },
      ],
      { x: 20, y: 5 },
      (prefix) => `${prefix}-id`
    );

    expect(result).toEqual({
      kind: "error",
      message: "That cut would create multiple repair gaps.",
    });
  });

  it("projects repair clicks to the nearest point on a remaining boundary edge", () => {
    const projection = projectPointToPolyline(
      { x: 6, y: 4 },
      [
        { x: 0, y: 0 },
        { x: 10, y: 0 },
        { x: 10, y: 10 },
      ]
    );

    expect(projection?.point).toEqual({ x: 6, y: 0 });
    expect(projection?.segmentIndex).toBe(0);
  });

  it("validates a boundary-touching automatic patch selection", () => {
    const [polygon] = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [10, 10],
            [30, 10],
            [30, 30],
            [10, 30],
            [10, 10],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );
    const patchBBox = buildPatchBBoxFromPoints(
      { x: 14, y: 8 },
      { x: 26, y: 12 }
    );

    expect(patchBBox).not.toBeNull();
    expect(validateAutomaticPatchSelection(polygon, patchBBox!)).toBeNull();
  });

  it("rejects an automatic patch selection that does not touch the boundary", () => {
    const [polygon] = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [10, 10],
            [30, 10],
            [30, 30],
            [10, 30],
            [10, 10],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );

    expect(
      validateAutomaticPatchSelection(polygon, {
        x0: 14,
        y0: 14,
        x1: 20,
        y1: 20,
      })
    ).toBe("The selected patch must intersect the current polygon boundary.");
    expect(
      patchBBoxWithinLimit(
        {
          x0: 0,
          y0: 0,
          x1: 513,
          y1: 20,
        },
        512
      )
    ).toBe(false);
  });

  it("closes an active repair session into a confirmable polygon", () => {
    const [polygon] = buildAutomaticPrefillPolygons(
      {
        geometry_type: "Polygon",
        polygons: [
          [
            [10, 10],
            [30, 10],
            [30, 30],
            [10, 30],
            [10, 10],
          ],
        ],
      },
      (prefix) => `${prefix}-id`
    );

    const cut = cutDraftPolygonsWithLasso(
      [polygon],
      [
        { x: 14, y: 8 },
        { x: 26, y: 8 },
        { x: 26, y: 12 },
        { x: 14, y: 12 },
      ],
      { x: 14, y: 8 },
      (prefix) => `${prefix}-id`
    );
    expect(cut.kind).toBe("ok");
    if (cut.kind !== "ok") {
      return;
    }

    const repaired = appendRepairSection(cut.repairSessions, {
      id: "repair-segment-1",
      kind: "section",
      endedOutsidePatch: false,
      points: [
        { x: 14, y: 10 },
        { x: 18, y: 6 },
        { x: 22, y: 6 },
      ],
    });
    const closed = closeActiveRepairSession(repaired, (prefix) => `${prefix}-id`);

    expect(closed).not.toBeNull();
    expect(closed?.repairSessions).toHaveLength(0);
    expect(closed?.polygon.closed).toBe(true);
    const [ring] = buildConfirmPolygons(closed ? [closed.polygon] : []);
    expect(ring).toBeDefined();
    expect(ring).toContainEqual([14, 10]);
    expect(ring).toContainEqual([18, 6]);
    expect(ring).toContainEqual([22, 6]);
    expect(ring).toContainEqual([26, 10]);
    expect(ring).toContainEqual([30, 10]);
    expect(ring).toContainEqual([30, 30]);
    expect(ring).toContainEqual([10, 30]);
    expect(ring).toContainEqual([10, 10]);
  });

  it("starts a repair from the closer cut endpoint", () => {
    const session = {
      id: "repair-anchor",
      sourcePolygonId: "poly-anchor",
      remainingPath: [
        { x: 26, y: 10 },
        { x: 30, y: 10 },
        { x: 30, y: 30 },
        { x: 10, y: 30 },
        { x: 10, y: 10 },
        { x: 14, y: 10 },
      ],
      repairSegments: [],
      startAnchor: null,
      endAnchor: null,
      active: true,
    };

    expect(resolveRepairStartAnchor(session, { x: 13, y: 12 })).toEqual({ x: 14, y: 10 });
    expect(resolveRepairStartAnchor(session, { x: 27, y: 12 })).toEqual({ x: 26, y: 10 });
  });

  it("closes a dense repair outline by stitching directly to the remaining dangling end", () => {
    const closeability = resolveRepairSessionCloseability({
      id: "repair-dense",
      sourcePolygonId: "poly-dense",
      remainingPath: makeDenseRepairRemainingPath(),
      repairSegments: [
        {
          id: "repair-segment-dense",
          kind: "section",
          endedOutsidePatch: false,
          points: [
            { x: 10, y: 0 },
            { x: 14, y: -12 },
            { x: 20, y: -8 },
          ],
        },
      ],
      startAnchor: { x: 10, y: 0 },
      endAnchor: null,
      active: true,
    });

    expect(closeability.kind).toBe("ok");
    if (closeability.kind !== "ok") {
      return;
    }
    expect(closeability.ring[0]).toEqual({ x: 10, y: 0 });
    expect(closeability.ring).toContainEqual({ x: 20, y: 0 });
  });

  it("ignores sub-threshold slivers when close polygon resolves one real polygon", () => {
    const polygon: DraftPolygon = {
      id: "poly-sliver",
      closed: false,
      segments: [
        {
          id: "seg-sliver",
          kind: "section",
          endedOutsidePatch: false,
          points: [
            { x: 0, y: 0 },
            { x: 60, y: 0 },
            { x: 60, y: 60 },
            { x: 0, y: 60 },
            { x: 0, y: 0 },
            { x: 8, y: 0 },
            { x: 8, y: 40 },
            { x: 0, y: 40 },
          ],
        },
      ],
    };

    const closeability = resolveDraftPolygonCloseability(polygon);
    expect(closeability.kind).toBe("ok");
    if (closeability.kind !== "ok") {
      return;
    }
    expect(closeability.ring).toContainEqual({ x: 60, y: 0 });
    expect(closeability.ring).toContainEqual({ x: 60, y: 60 });
    expect(closeability.ring).toContainEqual({ x: 0, y: 60 });
    expect(closeability.ring).not.toContainEqual({ x: 8, y: 40 });

    const closed = closeDraftPolygon([polygon], () => "closing-sliver");
    const [ring] = buildConfirmPolygons(closed);
    expect(ring).toBeDefined();
    expect(ring).toContainEqual([60, 0]);
    expect(ring).toContainEqual([60, 60]);
    expect(ring).toContainEqual([0, 60]);
    expect(ring).not.toContainEqual([8, 40]);
  });

  it("keeps the larger polygon when closing creates multiple visible polygons", () => {
    const polygon: DraftPolygon = {
      id: "poly-largest",
      closed: false,
      segments: [
        {
          id: "seg-largest",
          kind: "section",
          endedOutsidePatch: false,
          points: [
            { x: 0, y: 0 },
            { x: 100, y: 0 },
            { x: 100, y: 100 },
            { x: 0, y: 100 },
            { x: 0, y: 0 },
            { x: 30, y: 0 },
            { x: 30, y: 80 },
            { x: 0, y: 80 },
          ],
        },
      ],
    };

    expect(canCloseDraftPolygon(polygon)).toBe(true);
    const closeability = resolveDraftPolygonCloseability(polygon);
    expect(closeability.kind).toBe("ok");
    if (closeability.kind !== "ok") {
      return;
    }
    expect(closeability.ring).toContainEqual({ x: 100, y: 0 });
    expect(closeability.ring).toContainEqual({ x: 100, y: 100 });
    expect(closeability.ring).toContainEqual({ x: 0, y: 100 });
    expect(closeability.ring).not.toContainEqual({ x: 30, y: 80 });

    const closed = closeDraftPolygon([polygon], () => "closing-largest");
    const [ring] = buildConfirmPolygons(closed);
    expect(ring).toBeDefined();
    expect(ring).toContainEqual([100, 0]);
    expect(ring).toContainEqual([100, 100]);
    expect(ring).toContainEqual([0, 100]);
    expect(ring).not.toContainEqual([30, 80]);
  });

  it("falls back to a direct end-to-start close when the nearest prior intersection does not resolve", () => {
    const polygon: DraftPolygon = {
      id: "poly-invalid",
      closed: false,
      segments: [
        {
          id: "seg-1",
          kind: "section",
          endedOutsidePatch: false,
          points: [
            { x: 10, y: 10 },
            { x: 30, y: 10 },
            { x: 30, y: 30 },
            { x: 10, y: 30 },
          ],
        },
        {
          id: "seg-2",
          kind: "section",
          endedOutsidePatch: false,
          points: [
            { x: 10, y: 30 },
            { x: 20, y: 0 },
            { x: 20, y: 40 },
          ],
        },
      ],
    };

    expect(canCloseDraftPolygon(polygon)).toBe(true);
    const closeability = resolveDraftPolygonCloseability(polygon);
    expect(closeability.kind).toBe("ok");
    if (closeability.kind !== "ok") {
      return;
    }

    expect(Math.abs(signedRingArea(closeability.ring))).toBeGreaterThan(150);

    const closed = closeDraftPolygon([polygon], () => "closing-fallback");
    expect(closed[0]?.closed).toBe(true);
    expect(buildConfirmPolygons(closed)).toHaveLength(1);
  });

  it("returns a stitched repair ring even when the direct close self-crosses", () => {
    const closeability = resolveRepairSessionCloseability({
      id: "repair-1",
      sourcePolygonId: "poly-1",
      remainingPath: [
        { x: 26, y: 10 },
        { x: 30, y: 10 },
        { x: 30, y: 30 },
        { x: 10, y: 30 },
        { x: 10, y: 10 },
        { x: 14, y: 10 },
      ],
      repairSegments: [
        {
          id: "repair-segment-1",
          kind: "section",
          endedOutsidePatch: false,
          points: [
            { x: 14, y: 10 },
            { x: 20, y: 4 },
            { x: 20, y: 34 },
          ],
        },
      ],
      startAnchor: { x: 14, y: 10 },
      endAnchor: null,
      active: true,
    });

    expect(closeability.kind).toBe("ok");
    if (closeability.kind !== "ok") {
      return;
    }

    expect(closeability.ring).toContainEqual({ x: 26, y: 10 });

    const closed = closeActiveRepairSession(
      [
        {
          id: "repair-1",
          sourcePolygonId: "poly-1",
          remainingPath: [
            { x: 26, y: 10 },
            { x: 30, y: 10 },
            { x: 30, y: 30 },
            { x: 10, y: 30 },
            { x: 10, y: 10 },
            { x: 14, y: 10 },
          ],
          repairSegments: [
            {
              id: "repair-segment-1",
              kind: "section",
              endedOutsidePatch: false,
              points: [
                { x: 14, y: 10 },
                { x: 20, y: 4 },
                { x: 20, y: 34 },
              ],
            },
          ],
          startAnchor: { x: 14, y: 10 },
          endAnchor: null,
          active: true,
        },
      ],
      (prefix) => `${prefix}-id`,
      closeability.ring
    );

    expect(closed).not.toBeNull();
    expect(closed?.polygon.closed).toBe(true);
  });
});
