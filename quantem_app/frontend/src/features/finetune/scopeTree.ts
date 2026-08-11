/**
 * The applies-to tree, and what a selection in it adds up to.
 *
 * Pure, and separate from the component, because two of the three things this
 * dialog can get wrong are arithmetic:
 *
 * * **the count.** The owner's example is a ten-image dataset in which two
 *   images carry three annotations each and a third carries one, and the dialog
 *   must show **7** — the number of annotated regions, not the number of images
 *   and not the number of training tiles they are cut into.
 * * **double counting.** A dataset and one of its own images are two ways of
 *   selecting the same asset, and the server unions them; a client that added
 *   them would show a total the run could never match. Selecting a dataset
 *   therefore *replaces* any individual picks inside it, and the two id sets
 *   stay disjoint by construction.
 *
 * The third — whether the selection is legal at all — is deliberately *not*
 * here. `POST /preview/` decides that, and the dialog renders its verdict; a
 * second implementation of the same-experiment rule on this side would be a
 * rule that can drift from the one the 400 enforces.
 */

import type {
  FineTuneScopeDataset,
  FineTuneScopeImage,
  FineTuneScopeResponse,
  FineTuneScopeSelectionPayload,
} from "@/shared/types/finetune";

/**
 * The group holding images that belong to no experiment.
 *
 * A sibling of the experiments rather than an experiment with an empty id,
 * because that is what it is on the wire and because the same-experiment rule
 * treats it as one group of its own: an unassigned image cannot be trained
 * alongside an experiment-bound one.
 */
export const UNASSIGNED_GROUP_KEY = "unassigned";
export const UNASSIGNED_GROUP_NAME = "Not in an experiment";

export interface ScopeImageNode {
  id: string;
  name: string;
  confirmedAreas: number;
  doneRois: number;
  annotationCount: number;
}

export interface ScopeDatasetNode {
  id: string;
  name: string;
  imageCount: number;
  annotatedImageCount: number;
  /**
   * The server's own total for the dataset. Used whole rather than re-derived
   * from `images`, which the contract does not promise is exhaustive.
   */
  annotationCount: number;
  images: ScopeImageNode[];
}

export interface ScopeGroupNode {
  key: string;
  kind: "experiment" | "unassigned";
  name: string;
  datasets: ScopeDatasetNode[];
  /** In the group, in no dataset. */
  images: ScopeImageNode[];
  annotationCount: number;
  imageCount: number;
}

export interface ScopeSelection {
  datasetIds: ReadonlySet<string>;
  assetIds: ReadonlySet<string>;
}

export function emptySelection(): ScopeSelection {
  return { datasetIds: new Set(), assetIds: new Set() };
}

function imageNode(image: FineTuneScopeImage): ScopeImageNode {
  return {
    id: image.id,
    name: image.name,
    confirmedAreas: image.confirmed_areas ?? 0,
    doneRois: image.done_rois ?? 0,
    annotationCount: image.annotation_count ?? 0,
  };
}

function datasetNode(dataset: FineTuneScopeDataset): ScopeDatasetNode {
  return {
    id: dataset.id,
    name: dataset.name,
    imageCount: dataset.image_count ?? 0,
    annotatedImageCount: dataset.annotated_image_count ?? 0,
    annotationCount: dataset.annotation_count ?? 0,
    images: (dataset.images ?? []).map(imageNode),
  };
}

/** One flat list of groups: the experiments, then the unassigned images. */
export function buildScopeTree(
  response: FineTuneScopeResponse | null
): ScopeGroupNode[] {
  if (!response) return [];
  const groups: ScopeGroupNode[] = (response.experiments ?? []).map((experiment) => {
    const datasets = (experiment.datasets ?? []).map(datasetNode);
    const images = (experiment.ungrouped_images ?? []).map(imageNode);
    return {
      key: experiment.id,
      kind: "experiment" as const,
      name: experiment.name,
      datasets,
      images,
      annotationCount: totalOf(datasets, images),
      imageCount: imageCountOf(datasets, images),
    };
  });
  const unassigned = (response.unassigned_images ?? []).map(imageNode);
  if (unassigned.length > 0) {
    groups.push({
      key: UNASSIGNED_GROUP_KEY,
      kind: "unassigned",
      name: UNASSIGNED_GROUP_NAME,
      datasets: [],
      images: unassigned,
      annotationCount: totalOf([], unassigned),
      imageCount: unassigned.length,
    });
  }
  return groups;
}

function totalOf(datasets: ScopeDatasetNode[], images: ScopeImageNode[]): number {
  return (
    datasets.reduce((sum, dataset) => sum + dataset.annotationCount, 0) +
    images.reduce((sum, image) => sum + image.annotationCount, 0)
  );
}

function imageCountOf(datasets: ScopeDatasetNode[], images: ScopeImageNode[]): number {
  return (
    datasets.reduce((sum, dataset) => sum + dataset.imageCount, 0) + images.length
  );
}

function matches(text: string, needle: string): boolean {
  return text.toLowerCase().includes(needle);
}

/**
 * The tree narrowed to what the search box says.
 *
 * A group or dataset whose own name matches keeps all of its children — you
 * searched for the dataset, so you want the dataset — while one that survives
 * only because an image inside it matched keeps just the matching images.
 */
export function filterScopeTree(
  groups: ScopeGroupNode[],
  query: string
): ScopeGroupNode[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return groups;
  const filtered: ScopeGroupNode[] = [];
  for (const group of groups) {
    const groupMatches = matches(group.name, needle);
    if (groupMatches) {
      filtered.push(group);
      continue;
    }
    const datasets = group.datasets
      .map((dataset) =>
        matches(dataset.name, needle)
          ? dataset
          : { ...dataset, images: dataset.images.filter((i) => matches(i.name, needle)) }
      )
      .filter(
        (dataset) => matches(dataset.name, needle) || dataset.images.length > 0
      );
    const images = group.images.filter((image) => matches(image.name, needle));
    if (datasets.length > 0 || images.length > 0) {
      filtered.push({ ...group, datasets, images });
    }
  }
  return filtered;
}

export function isDatasetSelected(
  selection: ScopeSelection,
  datasetId: string
): boolean {
  return selection.datasetIds.has(datasetId);
}

/** An image is in scope on its own account, or because its dataset is. */
export function isImageSelected(
  selection: ScopeSelection,
  imageId: string,
  datasetId?: string | null
): boolean {
  if (datasetId && selection.datasetIds.has(datasetId)) return true;
  return selection.assetIds.has(imageId);
}

export function setDatasetSelected(
  selection: ScopeSelection,
  dataset: ScopeDatasetNode,
  on: boolean
): ScopeSelection {
  const datasetIds = new Set(selection.datasetIds);
  const assetIds = new Set(selection.assetIds);
  if (on) {
    datasetIds.add(dataset.id);
    // The dataset now stands for every image in it. Leaving the individual
    // picks behind would double the dataset's images in the count while the
    // server, which unions the two lists, counted them once.
    for (const image of dataset.images) assetIds.delete(image.id);
  } else {
    datasetIds.delete(dataset.id);
  }
  return { datasetIds, assetIds };
}

export function setImageSelected(
  selection: ScopeSelection,
  imageId: string,
  on: boolean
): ScopeSelection {
  const assetIds = new Set(selection.assetIds);
  if (on) assetIds.add(imageId);
  else assetIds.delete(imageId);
  return { datasetIds: new Set(selection.datasetIds), assetIds };
}

/** Every dataset and loose image in one group, on or off together. */
export function setGroupSelected(
  selection: ScopeSelection,
  group: ScopeGroupNode,
  on: boolean
): ScopeSelection {
  let next = selection;
  for (const dataset of group.datasets) next = setDatasetSelected(next, dataset, on);
  for (const image of group.images) next = setImageSelected(next, image.id, on);
  return next;
}

export function isGroupFullySelected(
  selection: ScopeSelection,
  group: ScopeGroupNode
): boolean {
  if (group.datasets.length === 0 && group.images.length === 0) return false;
  return (
    group.datasets.every((dataset) => selection.datasetIds.has(dataset.id)) &&
    group.images.every((image) => selection.assetIds.has(image.id))
  );
}

export function isSelectionEmpty(selection: ScopeSelection): boolean {
  return selection.datasetIds.size === 0 && selection.assetIds.size === 0;
}

export interface ScopeTotals {
  annotationCount: number;
  imageCount: number;
}

/**
 * The live count, while the preview request for this selection is still out.
 *
 * A whole dataset contributes the dataset's own totals; individually picked
 * images contribute their own. The server's `annotation_count` supersedes this
 * the moment it lands — the two are the same sum over the same records, and
 * where they could differ the server is right.
 */
export function selectionTotals(
  groups: ScopeGroupNode[],
  selection: ScopeSelection
): ScopeTotals {
  let annotationCount = 0;
  let imageCount = 0;
  for (const group of groups) {
    for (const dataset of group.datasets) {
      if (selection.datasetIds.has(dataset.id)) {
        annotationCount += dataset.annotationCount;
        imageCount += dataset.imageCount;
        continue;
      }
      for (const image of dataset.images) {
        if (!selection.assetIds.has(image.id)) continue;
        annotationCount += image.annotationCount;
        imageCount += 1;
      }
    }
    for (const image of group.images) {
      if (!selection.assetIds.has(image.id)) continue;
      annotationCount += image.annotationCount;
      imageCount += 1;
    }
  }
  return { annotationCount, imageCount };
}

/** The selection as the two id lists §4.2 and §4.3 take, in a stable order. */
export function toSelectionPayload(
  segmentationTypeId: string,
  selection: ScopeSelection
): FineTuneScopeSelectionPayload {
  return {
    segmentation_type: segmentationTypeId,
    asset_ids: [...selection.assetIds].sort(),
    dataset_ids: [...selection.datasetIds].sort(),
  };
}

/** A stable string for "is the preview on screen the preview for this?". */
export function selectionKey(
  segmentationTypeId: string,
  selection: ScopeSelection
): string {
  const payload = toSelectionPayload(segmentationTypeId, selection);
  return [
    payload.segmentation_type,
    payload.dataset_ids.join(","),
    payload.asset_ids.join(","),
  ].join("|");
}
