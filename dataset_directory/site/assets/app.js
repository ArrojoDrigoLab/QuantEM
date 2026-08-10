/**
 * The QuantEM dataset directory.
 *
 * Everything is static: four JSON files and a folder of thumbnails. There is no
 * API, no database, and no server-side rendering. The whole corpus is filtered
 * in the browser, which is why the counts beside every dataset can respond to
 * each keystroke.
 */
import {
  ALL_FACETS,
  describeAsset,
  emptySelection,
  evaluate,
  hydrate,
  isEmptySelection,
  rankDatasets,
} from './engine.js';
import { renderFacetRail } from './facets.js';
import {
  buildLookups,
  searchFromSelection,
  selectionFromSearch,
  selectionSize,
  toggle,
  toggleTreeParent,
} from './state.js';

const PAGE = 120;
const SCHEMA_MAJOR = 1;
// What ?sort= accepts. Anything else falls back, so a stale or hand-edited
// link cannot leave the control showing no selection at all.
const SORTS = new Set(['assets', 'name']);

const dom = {
  search: document.getElementById('search'),
  rail: document.getElementById('rail'),
  results: document.getElementById('results'),
  resultHead: document.getElementById('result-head'),
  viewToggle: document.getElementById('view-toggle'),
  sort: document.getElementById('sort'),
  clear: document.getElementById('clear'),
  detail: document.getElementById('detail'),
};

const app = {
  data: null,
  datasets: [],
  facets: [],
  lookups: null,
  manifest: null,
  selection: emptySelection(),
  view: 'datasets',
  sort: 'assets',
  expanded: new Set(),
  openDataset: null,
  shown: PAGE,
  result: null,
};

// ---------------------------------------------------------------- utilities

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function thumbUrl(hexId) {
  return `thumbs/${hexId.slice(0, 2)}/${hexId}.webp`;
}

/**
 * Link to a dataset while keeping the filters that led you to it, so closing
 * the panel returns you to the view you were in and the panel itself can say
 * how many of the dataset's assets match.
 */
function datasetHref(id) {
  const params = new URLSearchParams(location.search);
  params.set('dataset', id);
  return `?${params.toString()}`;
}

function thumbnail(hexId, alt, hasThumb) {
  if (!hasThumb) {
    const placeholder = element('div', 'thumb thumb-missing');
    placeholder.setAttribute('role', 'img');
    placeholder.setAttribute('aria-label', `${alt} — no preview available`);
    return placeholder;
  }
  const image = element('img', 'thumb');
  image.src = thumbUrl(hexId);
  image.alt = alt;
  image.loading = 'lazy';
  image.decoding = 'async';
  image.addEventListener('error', () => image.replaceWith(thumbnail(hexId, alt, false)), { once: true });
  return image;
}

function repositoryLabel(index) {
  return app.lookups.toLabel.repository[index];
}

// ------------------------------------------------------------------- render

function datasetCard({ row, matched }) {
  const card = element('article', 'card');
  const strip = element('div', 'card-thumbs');
  const heroes = row.hero.length ? row.hero : [];
  for (let i = 0; i < 4; i += 1) {
    strip.append(thumbnail(heroes[i], row.name, Boolean(heroes[i])));
  }
  card.append(strip);

  const body = element('div', 'card-body');
  if (row.experiment) body.append(element('p', 'card-eyebrow', row.experiment));

  const heading = element('h3', 'card-title');
  const link = element('a', null, row.name);
  link.href = datasetHref(row.id);
  link.title = row.name;
  heading.append(link);
  body.append(heading);

  const filtered = matched !== row.n;
  const count = element('p', 'card-count');
  count.append(element('strong', null, `${matched.toLocaleString()} ${matched === 1 ? 'asset' : 'assets'}`));
  if (filtered) count.append(element('span', 'card-count-total', ` of ${row.n.toLocaleString()}`));
  body.append(count);

  const meta = element('p', 'card-meta');
  const bits = [];
  if (row.n2d) bits.push(`${row.n2d.toLocaleString()} 2D`);
  if (row.n3d) bits.push(`${row.n3d.toLocaleString()} 3D`);
  bits.push(repositoryLabel(row.repository));
  meta.textContent = bits.join(' · ');
  body.append(meta);

  card.append(body);
  return card;
}

function assetCard(index) {
  const data = app.data;
  const hexId = data.ids[index];
  const row = app.datasets[data.dataset[index]];
  const card = element('article', 'tile');
  card.append(thumbnail(hexId, data.names[index] || row.name, data.single.thumb[index] === 1));
  const caption = element('div', 'tile-body');
  caption.append(element('p', 'tile-title', data.names[index] || '(untitled)'));
  const description = describeAsset(data, index);
  if (description) caption.append(element('p', 'tile-meta', description));
  const parent = element('a', 'tile-dataset', row.name);
  parent.href = datasetHref(row.id);
  caption.append(parent);
  card.append(caption);
  return card;
}

function renderResults() {
  dom.results.replaceChildren();
  const { datasetCounts, assetIndices } = app.result;

  if (app.view === 'datasets') {
    const rows = rankDatasets(app.datasets, datasetCounts, app.sort);
    dom.resultHead.textContent =
      `${rows.length.toLocaleString()} of ${app.datasets.length.toLocaleString()} datasets` +
      (isEmptySelection(app.selection) ? '' : ` · ${assetIndices.length.toLocaleString()} matching assets`);
    if (!rows.length) {
      dom.results.append(emptyState());
      return;
    }
    const grid = element('div', 'card-grid');
    rows.slice(0, app.shown).forEach((entry) => grid.append(datasetCard(entry)));
    dom.results.append(grid);
    if (rows.length > app.shown) dom.results.append(moreButton(rows.length));
    return;
  }

  dom.resultHead.textContent = `${assetIndices.length.toLocaleString()} of ${app.data.n.toLocaleString()} images and volumes`;
  if (!assetIndices.length) {
    dom.results.append(emptyState());
    return;
  }
  const grid = element('div', 'tile-grid');
  assetIndices.slice(0, app.shown).forEach((index) => grid.append(assetCard(index)));
  dom.results.append(grid);
  if (assetIndices.length > app.shown) dom.results.append(moreButton(assetIndices.length));
}

function emptyState() {
  const box = element('div', 'empty');
  box.append(element('p', null, 'Nothing matches every filter you have selected.'));
  const reset = element('button', 'link-button', 'Clear all filters');
  reset.type = 'button';
  reset.addEventListener('click', clearAll);
  box.append(reset);
  return box;
}

function moreButton(total) {
  const wrapper = element('div', 'more');
  const button = element(
    'button',
    'button',
    `Show more (${(total - app.shown).toLocaleString()} remaining)`,
  );
  button.type = 'button';
  button.addEventListener('click', () => {
    app.shown += PAGE * 2;
    renderResults();
  });
  wrapper.append(button);
  return wrapper;
}

function renderRail() {
  dom.rail.replaceChildren(
    renderFacetRail({
      facets: app.facets,
      counts: app.result.counts,
      selection: app.selection,
      expandedKeys: app.expanded,
      actions: {
        toggle(facetKey, value) {
          app.selection = toggle(app.selection, facetKey, value);
          commit();
        },
        toggleTreeParent(facetKey, parentIndex, childIds) {
          app.selection = toggleTreeParent(app.selection, facetKey, parentIndex, childIds);
          commit();
        },
        toggleExpanded(key) {
          if (app.expanded.has(key)) app.expanded.delete(key);
          else app.expanded.add(key);
          renderRail();
        },
      },
    }),
  );
  const active = selectionSize(app.selection);
  dom.clear.hidden = active === 0;
  dom.clear.textContent = `Clear ${active} filter${active === 1 ? '' : 's'}`;
}

// -------------------------------------------------------------- detail view

function renderDetail() {
  const id = app.openDataset;
  if (!id) {
    dom.detail.hidden = true;
    dom.detail.replaceChildren();
    document.body.classList.remove('has-detail');
    return;
  }
  const rowIndex = app.datasets.findIndex((d) => d.id === id);
  if (rowIndex < 0) {
    app.openDataset = null;
    return renderDetail();
  }
  const row = app.datasets[rowIndex];

  const panel = element('div', 'detail-panel');
  const close = element('button', 'detail-close', '×');
  close.type = 'button';
  close.setAttribute('aria-label', 'Close dataset');
  close.addEventListener('click', () => {
    app.openDataset = null;
    commit();
  });
  panel.append(close);

  if (row.experiment) panel.append(element('p', 'detail-eyebrow', row.experiment));
  panel.append(element('h2', null, row.name));

  const facts = element('dl', 'detail-facts');
  const addFact = (term, value) => {
    facts.append(element('dt', null, term));
    facts.append(value instanceof Node ? wrapDd(value) : element('dd', null, value));
  };
  addFact('2D images', row.n2d.toLocaleString());
  addFact('3D acquisitions', row.n3d.toLocaleString());
  addFact('Repository', repositoryLabel(row.repository));
  if (row.url) {
    const link = element('a', null, row.url);
    link.href = row.url;
    link.rel = 'noopener';
    addFact('Source', link);
  } else {
    addFact('Source', 'Deposition pending');
  }
  panel.append(facts);

  const note = element('p', 'detail-note');
  note.textContent =
    'QuantEM does not host or redistribute the source imaging data. Reuse terms are set by ' +
    'each depositor — check the source repository.';
  panel.append(note);

  const own = [];
  for (let i = 0; i < app.data.n; i += 1) {
    if (app.data.dataset[i] === rowIndex && app.result.matched[i]) own.push(i);
  }
  panel.append(
    element(
      'h3',
      'detail-subhead',
      own.length === row.n
        ? `${row.n.toLocaleString()} assets`
        : `${own.length.toLocaleString()} of ${row.n.toLocaleString()} assets match the current filters`,
    ),
  );
  const grid = element('div', 'tile-grid');
  own.slice(0, 240).forEach((index) => grid.append(assetCard(index)));
  panel.append(grid);
  if (own.length > 240) {
    panel.append(element('p', 'detail-note', `Showing the first 240 of ${own.length.toLocaleString()}.`));
  }

  dom.detail.replaceChildren(panel);
  dom.detail.hidden = false;
  document.body.classList.add('has-detail');
  panel.scrollTop = 0;
}

function wrapDd(node) {
  const dd = element('dd');
  dd.append(node);
  return dd;
}

// ---------------------------------------------------------------- lifecycle

function recompute() {
  app.result = evaluate(app.data, app.datasets, app.selection);
}

function render() {
  recompute();
  renderRail();
  renderResults();
  renderDetail();
}

function commit({ replace = false } = {}) {
  app.shown = PAGE;
  const extras = {};
  if (app.view !== 'datasets') extras.view = app.view;
  if (app.sort !== 'assets') extras.sort = app.sort;
  if (app.openDataset) extras.dataset = app.openDataset;
  const search = searchFromSelection(app.selection, app.lookups, extras);
  const url = `${location.pathname}${search}${location.hash}`;
  if (replace) history.replaceState(null, '', url);
  else history.pushState(null, '', url);
  render();
}

function readUrl() {
  const params = new URLSearchParams(location.search);
  app.selection = selectionFromSearch(location.search, app.lookups);
  app.view = params.get('view') === 'images' ? 'images' : 'datasets';
  app.sort = SORTS.has(params.get('sort')) ? params.get('sort') : 'assets';
  app.openDataset = params.get('dataset');
  app.shown = PAGE;
  dom.search.value = app.selection.q;
  for (const button of dom.viewToggle.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(button.dataset.view === app.view));
  }
  dom.sort.value = app.sort;
}

function clearAll() {
  app.selection = emptySelection();
  dom.search.value = '';
  commit();
}

function wireControls() {
  let timer = null;
  dom.search.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      app.selection = { ...app.selection, q: dom.search.value };
      for (const key of ALL_FACETS) app.selection[key] = new Set(app.selection[key]);
      commit({ replace: true });
    }, 160);
  });

  for (const button of dom.viewToggle.querySelectorAll('button')) {
    button.addEventListener('click', () => {
      app.view = button.dataset.view;
      commit();
    });
  }

  dom.sort.addEventListener('change', () => {
    app.sort = dom.sort.value;
    commit();
  });

  dom.clear.addEventListener('click', clearAll);

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && app.openDataset) {
      app.openDataset = null;
      commit();
    }
  });

  // Intercept in-page links so navigation stays a single history entry model.
  document.addEventListener('click', (event) => {
    const anchor = event.target.closest('a[href^="?"]');
    if (!anchor || event.metaKey || event.ctrlKey || event.shiftKey) return;
    event.preventDefault();
    history.pushState(null, '', anchor.getAttribute('href'));
    readUrl();
    render();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  window.addEventListener('popstate', () => {
    readUrl();
    render();
  });
}

async function boot() {
  const load = (name) => fetch(`data/${name}`).then((response) => {
    if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
    return response.json();
  });

  try {
    const [manifest, facetsJson, datasetsJson, assetsJson] = await Promise.all([
      load('manifest.json'),
      load('facets.json'),
      load('datasets.json'),
      load('assets.json'),
    ]);

    const major = Number(String(manifest.schema_version).split('.')[0]);
    if (major !== SCHEMA_MAJOR) {
      throw new Error(
        `This page reads schema ${SCHEMA_MAJOR}.x but the data is ${manifest.schema_version}.`,
      );
    }

    app.manifest = manifest;
    app.facets = facetsJson.facets;
    app.datasets = datasetsJson.rows;
    app.lookups = buildLookups(facetsJson);
    app.data = hydrate(assetsJson, app.datasets);

    readUrl();
    wireControls();
    render();
    document.body.classList.remove('is-loading');
  } catch (error) {
    document.body.classList.remove('is-loading');
    dom.results.replaceChildren(
      element('div', 'empty', `Could not load the directory data. ${error.message}`),
    );
  }
}

boot();
