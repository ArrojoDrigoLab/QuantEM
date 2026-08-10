# `exporter/`

Turns a read-only extract of the corpus database into the static files the directory site reads.

The site has no server and no database. This is the only thing that touches the corpus, it runs
offline, and its output is four JSON files and a CSV — small enough to commit, so what the site
publishes is reviewable in a diff.

---

## Usage

```bash
export PYTHONPATH=src

# 1. Build the published data
python -m quantem_directory build \
    --extract /path/to/corpus-extract \
    --urls    /path/to/dataset_urls.csv \
    --out     ../data \
    --snapshot 2026-08-05

# 2. Render thumbnails (hours; reads over the network, so use plenty of workers)
python -m quantem_directory thumbs \
    --tile-index /path/to/canonical_tiles.csv \
    --tile-root  /path/to/tile-export-root \
    --extract    /path/to/corpus-extract \
    --out ../thumbs --px 256 --workers 24

# 3. Rebuild so the data records which assets have a thumbnail
python -m quantem_directory build --extract ... --urls ... --out ../data --thumbs ../thumbs

# 4. Gate it
python -m quantem_directory verify --data ../data --thumbs ../thumbs

# 5. Package the thumbnails for a release, then attach the archive to that tag
python -m quantem_directory pack --out ../thumbs \
    --archive ../thumbs-256-v1.tar.gz --release-tag directory-thumbs-v1
python -m quantem_directory build --extract ... --urls ... --out ../data --thumbs ../thumbs
```

Every path is an argument. Nothing about a particular machine is baked in.

## Contents

| File | What it is |
|---|---|
| `src/quantem_directory/extract.py` | Reads the extract CSVs. The only module that knows their shape. |
| `src/quantem_directory/derive.py` | The rules that decide published numbers — dimensionality, links, repository, resolution bands |
| `src/quantem_directory/build.py` | Writes the published artifacts |
| `src/quantem_directory/thumbs.py` | Renders and packages thumbnails |
| `src/quantem_directory/verify.py` | The gate: structure, privacy, exclusions |
| `src/quantem_directory/allowlist.py` | What may be published, as an allow-list |
| `excluded.json` | Corpus entries deliberately kept out |
| `vocabulary_overrides.json` | Facet-value corrections |
| `dataset_links.json` | Dataset links, applied as depositions complete |
| `tools/thumbs_from_folder.py` | Renders thumbnails from a folder downloaded by hand |

## What the exporter is, and is not, responsible for

It publishes the extract as it receives it. Datasets and images arrive under the names they are
deposited and catalogued under, and nothing here rewrites them — an exported name is the name, and
the site, the data files and the source repository all agree by construction rather than by
reconciliation.

It also does not pin the corpus to a fixed size. The corpus grows; `verify` checks that the export
is internally consistent and that nothing withheld reached it, not that it still matches a count
someone wrote down. An expected-count file can be passed to `verify` for a one-off check, but none
is committed.

## The four rules that decide published numbers

These are in `derive.py` with tests pinning each, because changing one changes what the site
reports.

**Dimensionality.** An asset is a 3D acquisition if it is tagged `3D`, *or* its resolution string
names three or more axis extents in nanometres, *or* it has more than one plane. The tag alone does
not reproduce the published 2D/3D split — a large number of assets are tagged `Mixed` or not at all.
Note that the axis test only sees numbers immediately followed by `nm`, so a resolution written
`27x27x80nm` reads as one component and those volumes are caught by their depth instead. That is the
behaviour of the rule that produced the published counts, and the tests pin it so a well-meaning
regex "fix" cannot silently move assets between the totals.

**Dataset link.** A dataset's own DOI, then its source URL, then its experiment's DOI. A dataset
with none of these has not been deposited, and the site says so rather than showing a dead link.

**Repository.** Derived from that link, not from the corpus's internal source key. The internal key
records how a dataset was originally *discovered*, so it is empty for everything that was not
machine-scraped — which would have shown contributed data as having no repository when in fact it
lives at the BioImage Archive. One wrinkle: EBI issues the DOI prefix `10.6019` for both EMPIAR and
the BioImage Archive, so the accession has to be inspected rather than the prefix trusted.

**Resolution bands.** Log-spaced, matching the scale the corpus composition figures use, with
`Unknown` as a real selectable band. A substantial share of assets have no parsable in-plane
resolution, including every asset of the single largest dataset; without a band they would vanish
the moment anyone touched the facet.

## The privacy gate

`allowlist.py` is an allow-list, not a filter. The corpus tags assets with a free-text group name —
there is no enum constraining it — so a deny-list would fail open the first time a new group
appeared. Instead:

- A small, explicit set of tag groups is eligible for publication: `kingdom`, `species`, `organ`,
  `Tissue Region`, `modality`.
- Every other group is withheld, either by name or by shape. A group that is neither published nor
  classified raises at load time and stops the build, so a new one gets a decision rather than a
  default. Matching by shape as well as by name is what keeps the classification complete as the
  corpus grows, without this repository having to enumerate the variables a group records.
- Facet vocabularies are held to a stricter bar than dataset names, and a value unfit to be a public
  label is dropped by shape rather than listed individually — so suppressing one never requires
  writing it down here.
- Before deployment, every published byte is scanned for email addresses, ORCIDs and leaked
  contributor records.

Dataset and image names are the depositors' own published titles and are reproduced verbatim, so a
record here matches the record at the source repository. They are checked for the patterns that are
unacceptable anywhere, but not rewritten.

Licence tags are withheld deliberately rather than for privacy: reuse terms belong to the depositor
and should be read from the source repository, not from a copy here that can go stale.

## Thumbnails, and why they are crops

An electron micrograph does not survive being shrunk whole — a 20,000-pixel montage reduced to
256 px is a uniform grey rectangle. So each thumbnail is the asset's highest-tissue-content
2048 px tile, taken from the tile corpus already cut and scored for pretraining. At 256 px that
still resolves membranes and organelles.

Sources are tried in order: the ranked canonical tiles, then any tile still present in the asset's
tile directory (the index is a point-in-time build and some assets were re-tiled or cleared since),
then a preview PNG, then the asset's own raster.

Only the fallback sources get a histogram stretch. Canonical tiles were already normalized by the
tiling pipeline using percentiles estimated *per source*, so tiles of one asset are consistent with
each other; re-stretching each tile independently would undo that. Raw rasters have had no such
treatment, and many are stored across a narrow slice of the range — genuine images whose pixel
values run from 17 to 40 — which is flat grey until stretched.

Where the tile corpus had no coverage, the image was fetched from its publishing repository and
matched by filename, so no asset ever shows another's picture. The stragglers came from the tiling
run manifests, which record the exact file each asset was read from under the tiler's own id —
including the test holdout, which the tiler skipped deliberately and which therefore has no tile
despite being on disk. `tools/thumbs_from_folder.py` remains for anything that ever has to be
downloaded by hand.

Every rendered thumbnail is then measured *as viewed* — contrast, dynamic range and tone after
encoding — because a tile that looks fine at 2048 px can still be a flat square at 256. Anything
without usable range is re-rendered with a stretch, or from a different plane.

## Testing

```bash
PYTHONPATH=src python -m pytest tests -q
```

The tests build a miniature corpus from scratch, shaped around the cases that actually caused
trouble — an asset with two kingdoms, an asset with no modality, a dataset whose link comes from its
experiment, a dataset with no link at all — so they run without any of the real data.

## Setup

```bash
conda env create -f environment.yml
```
