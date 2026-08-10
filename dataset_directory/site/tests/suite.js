/**
 * The filter engine test suite.
 *
 * Kept free of the DOM and of fetch so the same assertions run two ways: in a
 * browser against the live dev server (open ../tests.html), and headlessly in
 * CI. A suite that only runs in one of those tends to stop running in the other.
 *
 * Every check runs against the real exported corpus, so a failure means either
 * the engine or the export is wrong — both are worth knowing.
 */
import {
  ALL_FACETS, NO_VALUE, TREE_FACETS,
  emptySelection, evaluate, hydrate, isEmptySelection, rankDatasets, formatNm,
} from '../assets/engine.js';
import { buildLookups, searchFromSelection, selectionFromSearch, toggle, toggleTreeParent } from '../assets/state.js';

export function runSuite({ manifest, facetsJson, datasetsJson, assetsJson }) {
  const results = [];
  let passed = 0;
  let failed = 0;

  function check(name, fn) {
    try {
      const detail = fn();
      results.push({ name, ok: true, detail: detail == null ? '' : String(detail) });
      passed += 1;
    } catch (error) {
      results.push({ name, ok: false, detail: error.message });
      failed += 1;
    }
  }

  function assert(condition, message) {
    if (!condition) throw new Error(message);
  }
  function equal(actual, expected, message) {
    if (actual !== expected) throw new Error(`${message}: expected ${expected}, got ${actual}`);
  }

  const datasets = datasetsJson.rows;
  const data = hydrate(assetsJson, datasets);
  const lookups = buildLookups(facetsJson);
  const facets = Object.fromEntries(facetsJson.facets.map((f) => [f.key, f]));

  const idOf = (dictionary, label) => {
    const index = lookups.toIndex[dictionary].get(label);
    if (index === undefined) throw new Error(`no ${dictionary} named ${label}`);
    return index;
  };

  // ---------------------------------------------------------------- the data

  check('the export matches its own manifest', () => {
    equal(data.n, manifest.counts.assets, 'asset count');
    equal(datasets.length, manifest.counts.datasets, 'dataset count');
    return `${data.n.toLocaleString()} assets, ${datasets.length} datasets`;
  });

  check('every asset belongs to exactly one dataset in range', () => {
    for (let i = 0; i < data.n; i += 1) {
      assert(data.dataset[i] >= 0 && data.dataset[i] < datasets.length, `asset ${i} has a bad dataset index`);
    }
  });

  check('dataset asset counts sum to the corpus', () => {
    equal(datasets.reduce((a, d) => a + d.n, 0), data.n, 'sum of dataset.n');
  });

  check('every multi-valued facet is a well-formed CSR', () => {
    for (const [name, csr] of Object.entries(data.multi)) {
      equal(csr.offsets.length, data.n + 1, `${name} offsets length`);
      equal(csr.offsets[csr.offsets.length - 1], csr.values.length, `${name} final offset`);
    }
  });

  check('kingdom, organ and species cover every asset', () => {
    for (const rank of ['kingdom', 'organ', 'species']) {
      const csr = data.multi[rank];
      let missing = 0;
      for (let i = 0; i < data.n; i += 1) if (csr.offsets[i] === csr.offsets[i + 1]) missing += 1;
      equal(missing, 0, `${rank} assets with no value`);
    }
  });

  check('resolution has no holes — every asset lands in a band', () => {
    const bands = lookups.toLabel.resolution.length;
    for (let i = 0; i < data.n; i += 1) {
      assert(data.single.resolution[i] >= 0 && data.single.resolution[i] < bands, `asset ${i} has no band`);
    }
    return `${lookups.toLabel.resolution.length} bands including Unknown`;
  });

  // ------------------------------------------------------------ the engine

  const empty = emptySelection();
  const base = evaluate(data, datasets, empty);

  check('an empty selection matches everything', () => {
    assert(isEmptySelection(empty), 'emptySelection should read as empty');
    equal(base.assetIndices.length, data.n, 'matched assets');
    equal(rankDatasets(datasets, base.datasetCounts, 'assets').length, datasets.length, 'visible datasets');
  });

  check('unfiltered dataset counts equal each dataset total', () => {
    for (let i = 0; i < datasets.length; i += 1) {
      equal(base.datasetCounts[i], datasets[i].n, `dataset ${datasets[i].name}`);
    }
  });

  check('unfiltered facet counts equal the published facet counts', () => {
    for (const facet of facetsJson.facets) {
      if (facet.kind === 'tree') {
        const { parent, child } = TREE_FACETS[facet.key];
        for (const root of facet.roots) {
          equal(base.counts[facet.key].get(`${parent}:${root.id}`) || 0, root.n, `${facet.key} ${root.label}`);
          for (const node of root.children) {
            const seen = base.counts[facet.key].get(`${child}:${node.id}`) || 0;
            assert(seen >= node.n, `${facet.key} ${root.label}/${node.label}: ${seen} < ${node.n}`);
          }
        }
      } else {
        for (const value of facet.values) {
          equal(base.counts[facet.key].get(value.id) || 0, value.n, `${facet.key} ${value.label}`);
        }
      }
    }
  });

  check('selecting one value narrows to exactly that value', () => {
    const lung = idOf('organ', 'Lung');
    const selection = toggle(empty, 'anatomy', `organ:${lung}`);
    const result = evaluate(data, datasets, selection);
    const csr = data.multi.organ;
    for (const i of result.assetIndices) {
      let hit = false;
      for (let k = csr.offsets[i]; k < csr.offsets[i + 1]; k += 1) if (csr.values[k] === lung) hit = true;
      assert(hit, `asset ${i} survived without the selected organ`);
    }
    return `${result.assetIndices.length.toLocaleString()} lung assets`;
  });

  check('several values in one facet are OR, not AND', () => {
    const lung = idOf('organ', 'Lung');
    const brain = idOf('organ', 'Brain');
    const only = (id) => evaluate(data, datasets, toggle(empty, 'anatomy', `organ:${id}`)).assetIndices.length;
    const both = evaluate(
      data, datasets,
      toggle(toggle(empty, 'anatomy', `organ:${lung}`), 'anatomy', `organ:${brain}`),
    ).assetIndices.length;
    assert(both >= Math.max(only(lung), only(brain)), 'union should be at least as large as either part');
    assert(both <= only(lung) + only(brain), 'union should not exceed the sum');
    return `${only(lung)} ∪ ${only(brain)} = ${both}`;
  });

  check('different facets are AND, not OR', () => {
    const lung = idOf('organ', 'Lung');
    const tem = idOf('modality', 'TEM');
    const organOnly = evaluate(data, datasets, toggle(empty, 'anatomy', `organ:${lung}`));
    const both = evaluate(
      data, datasets,
      toggle(toggle(empty, 'anatomy', `organ:${lung}`), 'modality', tem),
    );
    assert(both.assetIndices.length <= organOnly.assetIndices.length, 'adding a facet must not widen the result');
    for (const i of both.assetIndices) equal(data.single.modality[i], tem, `asset ${i} modality`);
    return `${organOnly.assetIndices.length} → ${both.assetIndices.length}`;
  });

  check('a tree facet ORs across both of its ranks', () => {
    // Selecting an organ and a tissue context under a *different* organ must give
    // the union. Intersecting them would produce an empty, confusing result.
    const lung = idOf('organ', 'Lung');
    const anatomy = facets.anatomy;
    const otherRoot = anatomy.roots.find((r) => r.id !== lung && r.children.length > 0);
    const child = otherRoot.children[0];
    const selection = toggle(toggle(empty, 'anatomy', `organ:${lung}`), 'anatomy', `Tissue Region:${child.id}`);
    const union = evaluate(data, datasets, selection).assetIndices.length;
    const lungOnly = evaluate(data, datasets, toggle(empty, 'anatomy', `organ:${lung}`)).assetIndices.length;
    assert(union > lungOnly, `union (${union}) should exceed lung alone (${lungOnly})`);
    return `Lung ∪ ${child.label} = ${union}`;
  });

  check('dataset counts under a filter are the matching assets of that dataset', () => {
    const tem = idOf('modality', 'TEM');
    const result = evaluate(data, datasets, toggle(empty, 'modality', tem));
    const expected = new Int32Array(datasets.length);
    for (let i = 0; i < data.n; i += 1) if (data.single.modality[i] === tem) expected[data.dataset[i]] += 1;
    for (let i = 0; i < datasets.length; i += 1) {
      equal(result.datasetCounts[i], expected[i], `dataset ${datasets[i].name}`);
    }
  });

  check('a dataset with no matching asset disappears', () => {
    const et = idOf('modality', 'Electron tomography');
    const result = evaluate(data, datasets, toggle(empty, 'modality', et));
    const visible = rankDatasets(datasets, result.datasetCounts, 'assets');
    assert(visible.length < datasets.length, 'some datasets should drop out');
    for (const entry of visible) assert(entry.matched > 0, `${entry.row.name} shown with zero matches`);
    return `${datasets.length} → ${visible.length} datasets`;
  });

  check('a partially matching dataset keeps its total for the "of N" display', () => {
    const three = idOf('dimensionality', '3D acquisition');
    const result = evaluate(data, datasets, toggle(empty, 'dim', 1));
    const partial = rankDatasets(datasets, result.datasetCounts, 'assets')
      .find((entry) => entry.matched < entry.row.n);
    assert(partial, 'expected at least one partially matching dataset');
    assert(partial.matched < partial.row.n, 'matched should be below the total');
    void three;
    return `${partial.row.name}: ${partial.matched} of ${partial.row.n}`;
  });

  check('facet counts are computed under the other filters, not under all of them', () => {
    // With a modality selected, the modality facet must still show what the other
    // modalities would give — otherwise the unselected ones all read zero and the
    // filter becomes a one-way door.
    const tem = idOf('modality', 'TEM');
    const result = evaluate(data, datasets, toggle(empty, 'modality', tem));
    const others = facets.modality.values.filter((v) => v.id !== tem);
    const nonZero = others.filter((v) => (result.counts.modality.get(v.id) || 0) > 0);
    assert(nonZero.length > 0, 'every other modality read zero — counts were scoped to the full filter');
    return `${nonZero.length} of ${others.length} other modalities still selectable`;
  });

  check('a facet count predicts the result of selecting that value', () => {
    const lung = idOf('organ', 'Lung');
    const withLung = evaluate(data, datasets, toggle(empty, 'anatomy', `organ:${lung}`));
    const sem = idOf('modality', 'SEM');
    const predicted = withLung.counts.modality.get(sem) || 0;
    const actual = evaluate(
      data, datasets,
      toggle(toggle(empty, 'anatomy', `organ:${lung}`), 'modality', sem),
    ).assetIndices.length;
    equal(actual, predicted, 'predicted vs actual');
    return `${predicted} lung SEM assets, predicted and actual`;
  });

  check('the "not recorded" bucket is selectable and finds the untagged assets', () => {
    const result = evaluate(data, datasets, toggle(empty, 'modality', NO_VALUE));
    assert(result.assetIndices.length > 0, 'no assets found for the missing-modality bucket');
    for (const i of result.assetIndices) equal(data.single.modality[i], NO_VALUE, `asset ${i}`);
    return `${result.assetIndices.length} assets with no modality`;
  });

  check('unknown resolution is reachable', () => {
    const unknown = idOf('resolution', 'Unknown');
    const result = evaluate(data, datasets, toggle(empty, 'resolution', unknown));
    assert(result.assetIndices.length > 0, 'no assets in the unknown band');
    const visible = rankDatasets(datasets, result.datasetCounts, 'assets');
    return `${result.assetIndices.length.toLocaleString()} assets across ${visible.length} datasets`;
  });

  check('free-text search matches names and narrows datasets', () => {
    const selection = { ...emptySelection(), q: 'mitochondri' };
    const result = evaluate(data, datasets, selection);
    assert(result.assetIndices.length > 0, 'no matches for a term known to be present');
    assert(result.assetIndices.length < data.n, 'search matched everything');
    return `${result.assetIndices.length.toLocaleString()} assets`;
  });

  check('search combines with facets as AND', () => {
    const tem = idOf('modality', 'TEM');
    const withBoth = evaluate(data, datasets, { ...toggle(empty, 'modality', tem), q: 'mouse' });
    const searchOnly = evaluate(data, datasets, { ...emptySelection(), q: 'mouse' });
    assert(withBoth.assetIndices.length <= searchOnly.assetIndices.length, 'adding a facet must narrow');
    for (const i of withBoth.assetIndices) equal(data.single.modality[i], tem, `asset ${i}`);
  });

  check('checking a parent clears its selected children', () => {
    const anatomy = facets.anatomy;
    const root = anatomy.roots.find((r) => r.children.length > 1);
    const childIds = root.children.map((c) => c.id);
    let selection = toggle(empty, 'anatomy', `Tissue Region:${childIds[0]}`);
    assert(selection.anatomy.has(`Tissue Region:${childIds[0]}`), 'child should be selected');
    selection = toggleTreeParent(selection, 'anatomy', root.id, childIds);
    assert(selection.anatomy.has(`organ:${root.id}`), 'parent should now be selected');
    assert(!selection.anatomy.has(`Tissue Region:${childIds[0]}`), 'child should have been cleared');
    return root.label;
  });

  check('toggling is immutable — the previous selection is untouched', () => {
    const before = toggle(empty, 'modality', 0);
    const after = toggle(before, 'modality', 1);
    equal(before.modality.size, 1, 'original selection size');
    equal(after.modality.size, 2, 'derived selection size');
  });

  // ---------------------------------------------------------------- the URL

  check('a selection round-trips through the query string', () => {
    const lung = idOf('organ', 'Lung');
    const mus = idOf('species', 'Mus musculus');
    const tem = idOf('modality', 'TEM');
    let selection = toggle(empty, 'anatomy', `organ:${lung}`);
    selection = toggle(selection, 'taxonomy', `species:${mus}`);
    selection = toggle(selection, 'modality', tem);
    selection = toggle(selection, 'resolution', idOf('resolution', 'Unknown'));
    selection.q = 'liver';

    const search = searchFromSelection(selection, lookups);
    const restored = selectionFromSearch(search, lookups);
    for (const key of ALL_FACETS) {
      equal([...restored[key]].sort().join('|'), [...selection[key]].sort().join('|'), `facet ${key}`);
    }
    equal(restored.q, selection.q, 'query text');
    return decodeURIComponent(search);
  });

  check('the URL carries labels, not indices, so links survive a re-export', () => {
    const lung = idOf('organ', 'Lung');
    const search = searchFromSelection(toggle(empty, 'anatomy', `organ:${lung}`), lookups);
    assert(search.includes('organ=Lung'), `expected a readable label, got ${search}`);
  });

  check('the missing-value bucket survives the round trip', () => {
    const selection = toggle(empty, 'modality', NO_VALUE);
    const restored = selectionFromSearch(searchFromSelection(selection, lookups), lookups);
    assert(restored.modality.has(NO_VALUE), 'the "not recorded" selection was lost');
  });

  check('an unknown label in a URL is ignored rather than throwing', () => {
    const restored = selectionFromSearch('?organ=NotAnOrgan&modality=TEM', lookups);
    equal(restored.anatomy.size, 0, 'bad organ should be dropped');
    equal(restored.modality.size, 1, 'good modality should survive');
  });

  // -------------------------------------------------------------- formatting

  check('resolutions format without misleading precision', () => {
    equal(formatNm(1.5875), '1.59', '1.5875');
    equal(formatNm(12.5), '12.5', '12.5');
    equal(formatNm(316.5), '317', '316.5');
  });

  // ------------------------------------------------------------- performance

  check('a full filter and recount stays interactive', () => {
    const lung = idOf('organ', 'Lung');
    const selection = toggle(toggle(empty, 'anatomy', `organ:${lung}`), 'modality', idOf('modality', 'TEM'));
    const started = performance.now();
    const runs = 10;
    for (let i = 0; i < runs; i += 1) evaluate(data, datasets, selection);
    const each = (performance.now() - started) / runs;
    assert(each < 50, `${each.toFixed(1)} ms per pass is too slow to feel immediate`);
    return `${each.toFixed(1)} ms per full pass over ${data.n.toLocaleString()} assets`;
  });

  return { passed, failed, results };
}
