/**
 * Headless runner for the engine test suite.
 *
 *   node site/tests/run.mjs [path-to-data-dir]
 *
 * Reads the published artifacts from disk and runs the same assertions the
 * browser page runs, so continuous integration exercises the real suite rather
 * than a reduced copy of it. Exits non-zero if anything fails.
 */
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { runSuite } from './suite.js';

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = resolve(process.argv[2] || resolve(here, '..', '..', 'data'));

const load = async (name) => JSON.parse(await readFile(resolve(dataDir, name), 'utf8'));

const [manifest, facetsJson, datasetsJson, assetsJson] = await Promise.all(
  ['manifest.json', 'facets.json', 'datasets.json', 'assets.json'].map(load),
);

const { passed, failed, results } = runSuite({ manifest, facetsJson, datasetsJson, assetsJson });

for (const result of results) {
  const detail = result.detail ? ` — ${result.detail}` : '';
  console.log(`${result.ok ? 'ok  ' : 'FAIL'}  ${result.name}${detail}`);
}
console.log(`\n${failed ? `${failed} failed, ${passed} passed` : `all ${passed} tests passed`}`);

process.exit(failed ? 1 : 0);
