/**
 * Filter state lives in the query string, so any view can be linked or cited.
 *
 * Values are encoded as *labels*, not dictionary indices. Indices are an
 * implementation detail of one export and would silently point at a different
 * species after the next one; a URL printed in a paper has to keep meaning what
 * it meant. It also makes the link legible:
 *
 *   ?organ=Liver,Pancreas&species=Mus+musculus&res=4+%E2%80%93+8+nm/px
 */
import { ALL_FACETS, NO_VALUE, TREE_FACETS, emptySelection } from './engine.js';

/** Query-string key for each selectable rank or flat facet. */
const PARAMS = {
  kingdom: { facet: 'taxonomy', rank: 'kingdom' },
  species: { facet: 'taxonomy', rank: 'species' },
  organ: { facet: 'anatomy', rank: 'organ' },
  tissue: { facet: 'anatomy', rank: 'Tissue Region' },
  modality: { facet: 'modality', dictionary: 'modality' },
  res: { facet: 'resolution', dictionary: 'resolution' },
  dim: { facet: 'dim', dictionary: 'dimensionality' },
  repo: { facet: 'repository', dictionary: 'repository' },
};

/** Label shown, and accepted in a URL, for assets missing a facet entirely. */
export const NO_VALUE_TOKEN = 'none';

export function buildLookups(facetsJson) {
  const toIndex = {};
  const toLabel = {};
  for (const [name, values] of Object.entries(facetsJson.dictionaries)) {
    toIndex[name] = new Map(values.map((label, i) => [label, i]));
    toLabel[name] = values;
  }
  return { toIndex, toLabel };
}

export function selectionFromSearch(search, lookups) {
  const params = new URLSearchParams(search);
  const selection = emptySelection();
  selection.q = params.get('q') || '';

  for (const [param, spec] of Object.entries(PARAMS)) {
    const raw = params.get(param);
    if (!raw) continue;
    for (const label of raw.split(',').map((s) => s.trim()).filter(Boolean)) {
      if (spec.rank) {
        const index = lookups.toIndex[spec.rank]?.get(label);
        if (index !== undefined) selection[spec.facet].add(`${spec.rank}:${index}`);
      } else if (label === NO_VALUE_TOKEN) {
        selection[spec.facet].add(NO_VALUE);
      } else {
        const index = lookups.toIndex[spec.dictionary]?.get(label);
        if (index !== undefined) selection[spec.facet].add(index);
      }
    }
  }
  return selection;
}

export function searchFromSelection(selection, lookups, extras = {}) {
  const params = new URLSearchParams();
  if (selection.q) params.set('q', selection.q);

  for (const [param, spec] of Object.entries(PARAMS)) {
    const labels = [];
    if (spec.rank) {
      for (const key of selection[spec.facet]) {
        const [rank, index] = String(key).split(':');
        if (rank === spec.rank) labels.push(lookups.toLabel[rank][Number(index)]);
      }
    } else {
      for (const index of selection[spec.facet]) {
        labels.push(index === NO_VALUE ? NO_VALUE_TOKEN : lookups.toLabel[spec.dictionary][index]);
      }
    }
    if (labels.length) params.set(param, labels.sort().join(','));
  }

  for (const [key, value] of Object.entries(extras)) {
    if (value) params.set(key, value);
  }
  const query = params.toString();
  return query ? `?${query}` : '';
}

/** Count how many individual values are selected, for the "clear all" affordance. */
export function selectionSize(selection) {
  return ALL_FACETS.reduce((total, key) => total + selection[key].size, 0) + (selection.q ? 1 : 0);
}

export function toggle(selection, facet, value) {
  const next = { ...selection };
  for (const key of ALL_FACETS) next[key] = new Set(selection[key]);
  if (next[facet].has(value)) next[facet].delete(value);
  else next[facet].add(value);
  return next;
}

/**
 * Selecting a parent in a tree clears its children from the selection.
 * Without this, "Liver" plus "Hepatocyte" reads as a redundant pair and the
 * checkbox states contradict each other.
 */
export function toggleTreeParent(selection, facetKey, parentIndex, childIndicesUnderParent) {
  const { parent, child } = TREE_FACETS[facetKey];
  const next = toggle(selection, facetKey, `${parent}:${parentIndex}`);
  if (next[facetKey].has(`${parent}:${parentIndex}`)) {
    for (const childIndex of childIndicesUnderParent) next[facetKey].delete(`${child}:${childIndex}`);
  }
  return next;
}
