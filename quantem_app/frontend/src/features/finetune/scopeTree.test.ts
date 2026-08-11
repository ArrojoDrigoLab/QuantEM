/**
 * The two sums this dialog can get wrong, pinned.
 *
 * The fixture is the owner's own example, verbatim: a ten-image dataset in
 * which two images carry three annotations each and a third carries one. It
 * must read **7** — not 3 (annotated images), not 10 (images), and not the tile
 * count, which is a different and larger number.
 */

import { describe, expect, it } from "vitest";
import {
  UNASSIGNED_GROUP_KEY,
  buildScopeTree,
  emptySelection,
  filterScopeTree,
  isGroupFullySelected,
  isImageSelected,
  selectionKey,
  selectionTotals,
  setDatasetSelected,
  setGroupSelected,
  setImageSelected,
  toSelectionPayload,
} from "@/features/finetune/scopeTree";
import type { FineTuneScopeResponse } from "@/shared/types/finetune";

function image(id: string, name: string, confirmed: number, done: number) {
  return {
    id,
    name,
    confirmed_areas: confirmed,
    done_rois: done,
    annotation_count: confirmed + done,
  };
}

/** The owner's example, plus a second experiment to test the tree's shape. */
const SCOPE: FineTuneScopeResponse = {
  experiments: [
    {
      id: "exp-fasted",
      name: "Fasted cohort",
      datasets: [
        {
          id: "ds-liver",
          name: "Liver 24h",
          image_count: 10,
          annotated_image_count: 3,
          annotation_count: 7,
          images: [
            image("img-1", "liver_01.tif", 3, 0),
            image("img-2", "liver_02.tif", 2, 1),
            image("img-3", "liver_03.tif", 0, 1),
          ],
        },
      ],
      ungrouped_images: [image("img-loose", "liver_stray.tif", 1, 0)],
    },
    {
      id: "exp-fed",
      name: "Fed cohort",
      datasets: [],
      ungrouped_images: [image("img-fed", "fed_01.tif", 2, 0)],
    },
  ],
  unassigned_images: [image("img-orphan", "scratch.tif", 4, 0)],
};

describe("features/finetune/scopeTree", () => {
  it("puts the unassigned images in their own group, not in an experiment", () => {
    const tree = buildScopeTree(SCOPE);
    expect(tree.map((group) => group.key)).toEqual([
      "exp-fasted",
      "exp-fed",
      UNASSIGNED_GROUP_KEY,
    ]);
    expect(tree[2].kind).toBe("unassigned");
    expect(tree[2].datasets).toHaveLength(0);
    expect(tree[2].images.map((i) => i.id)).toEqual(["img-orphan"]);
  });

  it("shows 7 for the owner's ten-image dataset", () => {
    const tree = buildScopeTree(SCOPE);
    const dataset = tree[0].datasets[0];
    const selection = setDatasetSelected(emptySelection(), dataset, true);

    const totals = selectionTotals(tree, selection);
    expect(totals.annotationCount).toBe(7);
    // Ten, because the whole dataset is in scope -- including the seven images
    // with nothing on them yet, which the run will simply find nothing in.
    expect(totals.imageCount).toBe(10);
  });

  it("counts individually picked images on their own", () => {
    const tree = buildScopeTree(SCOPE);
    let selection = emptySelection();
    selection = setImageSelected(selection, "img-1", true);
    selection = setImageSelected(selection, "img-3", true);

    expect(selectionTotals(tree, selection)).toEqual({
      annotationCount: 4,
      imageCount: 2,
    });
  });

  it("never counts an image twice when its dataset is also picked", () => {
    const tree = buildScopeTree(SCOPE);
    const dataset = tree[0].datasets[0];
    let selection = setImageSelected(emptySelection(), "img-1", true);
    selection = setDatasetSelected(selection, dataset, true);

    // The individual pick was absorbed, so the payload the server unions
    // carries the dataset alone.
    expect(toSelectionPayload("type-1", selection)).toEqual({
      segmentation_type: "type-1",
      asset_ids: [],
      dataset_ids: ["ds-liver"],
    });
    expect(selectionTotals(tree, selection).annotationCount).toBe(7);
    // And the image still reads as selected, because it is.
    expect(isImageSelected(selection, "img-1", "ds-liver")).toBe(true);
  });

  it("selects and clears a whole experiment", () => {
    const tree = buildScopeTree(SCOPE);
    const group = tree[0];
    const selection = setGroupSelected(emptySelection(), group, true);

    expect(isGroupFullySelected(selection, group)).toBe(true);
    expect(selectionTotals(tree, selection).annotationCount).toBe(8);
    expect(
      isGroupFullySelected(setGroupSelected(selection, group, false), group)
    ).toBe(false);
  });

  it("keeps a dataset's children when the dataset name matches, and only the matches when it does not", () => {
    const tree = buildScopeTree(SCOPE);

    const byDataset = filterScopeTree(tree, "liver 24");
    expect(byDataset).toHaveLength(1);
    expect(byDataset[0].datasets[0].images).toHaveLength(3);

    const byImage = filterScopeTree(tree, "liver_02");
    expect(byImage[0].datasets[0].images.map((i) => i.id)).toEqual(["img-2"]);
    // The loose image in the same experiment did not match, so it is gone.
    expect(byImage[0].images).toHaveLength(0);
  });

  it("drops a group that matches nothing", () => {
    expect(filterScopeTree(buildScopeTree(SCOPE), "kidney")).toHaveLength(0);
  });

  it("keys a selection stably, whatever order it was built in", () => {
    const forwards = setImageSelected(
      setImageSelected(emptySelection(), "img-1", true),
      "img-2",
      true
    );
    const backwards = setImageSelected(
      setImageSelected(emptySelection(), "img-2", true),
      "img-1",
      true
    );
    expect(selectionKey("t", forwards)).toBe(selectionKey("t", backwards));
  });
});
