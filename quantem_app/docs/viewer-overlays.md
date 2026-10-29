# Viewer overlays

QuantEM uses integer label pyramids for persistent segmentation displays and
probability images for interactive threshold previews. Object geometry and
global masks remain authoritative; the viewer stores are derived data.

## Persistent display pipeline

`segmentation/overlay_ngff` builds two Zarr v2 pyramids: `labels` (uint32) and
`border` (uint8). Level zero is rasterized from current, non-superseded objects,
or from a global binary mask. Polygon holes preserve underlying objects. Paint
priority resolves overlaps; the label pyramid uses mode pooling and the border
pyramid uses max pooling. Chunk size is 256 × 256. Full object builds use bounded
2048 × 2048 macro-tile work with a border halo and bounded worker submission.
Global masks also convert labels and borders one macro tile at a time.

Each segmentation has an aggregate bundle for accepted annotations and optional
model bundles for that model's objects plus manual annotations. The aggregate
contains all current objects, with visibility selected through its palette.
The requested model's build status, error, and job remain visible while an
aggregate fallback is available. The labeling viewer draws accepted annotations
from the aggregate, without presenting another model's fallback candidates as
the selected model's predictions.
Binary masks do not request an object-picking map or participate in object picking.

The frontend combines label and border loaders into stable Viv selections.
`IdMapLabelLayer` packs each decoded uint32 tile losslessly into RGBA8 once.
`PaletteBitmapLayer` uses integer shader decoding, a shared palette texture, and
the baked border texture. Opacity and highlight updates change uniforms; local
color changes replace the palette. Neither operation refetches raster tiles.
The CPU renderer remains a fallback for palettes exceeding GPU texture limits.
Picking follows the original bitmap through Viv's composite boundaries and maps
visible labels to object UUIDs from the same snapshot.
Fully transparent pixels are excluded from picking in both rendering paths.

## What changes the display

| Change | Work |
| --- | --- |
| Color, opacity, borders, highlight | Local palette or uniforms; no server rebuild |
| Threshold preview slider | GPU threshold uniform; no image recoloring or model rerun |
| Confirm/reject an object without overlaps | Palette revision only |
| Confirm/reject an overlapping object | Immediate palette update and queued priority repaint |
| Completed ROI visibility | Palette revision; candidates inside completed regions are suppressed |
| Small geometry edit on a settled bundle | Synchronous partial snapshot when within the policy limits |
| Larger edits or edits during a build | Dirty regions coalesce for queued partial/full work |
| Initial display, invalid format, explicit rebuild, global-mask replacement | Full snapshot |
| Preparing an annotation ROI or restoring objects | Geometry invalidation for affected bundles |

Partial rebuilds union dirty chunk coordinates instead of filling the rectangle
between distant edits. Raster buffers are at most 260 × 260 pixels, including
the two-pixel halo on each side. Each affected ancestor chunk is regenerated
once per array and level. At execution time, coalesced work is promoted to a full
build if it exceeds 512 level-zero chunks or 20% of the image's chunks. Existing
label rows for retained objects survive partial builds; obsolete rows are removed
and only newly assigned identities are inserted.
Job completion queues any remaining edits without requiring an open viewer.

## Snapshot and revision contract

Format 7 stores immutable raster generations. Full and partial writers serialize
per bundle, build a staging tree, then publish its database pointer and label
assignments transactionally. Partial staging trees hard-link unchanged files
when supported and copy otherwise. Changed chunks are explicitly detached before
writing, so correctness does not depend on Zarr's overwrite implementation.

Raster URLs carry `rev=<bundle_version>-<applied_revision>`. Requests resolve
exactly that generation; missing generations return an error rather than current
pixels under an old URL. A small `snapshot.json` records the token. A separate
`labels.json` stores the immutable label-to-object mapping and is read only when
needed by palettes, picking, or analysis. Tile requests do not parse that map.

Palette requests use the raster token but resolve current object visibility.
They are not HTTP-cacheable. The client rejects incompatible generations and
retries transient palette reads. One failed overview palette does not clear
other overlays. Metadata and tile failures are surfaced in the viewer.
The server shares JSON visibility rows and binary palettes across requests using
a revision-keyed, 32 MiB bounded cache. A changed revision invalidates both the
object scan and completed-ROI suppression result. Uncached reads verify that the
visibility revision stayed stable while building; they do not hold a write lock.

The confirmed bundle's geometry and palette revisions are acknowledged after a
loaded viewport draws. Model counters cannot retire confirmed optimistic
outlines, and loading Zarr metadata alone is not a displayed revision. Optimistic
outlines remain until both required confirmed revisions have been shown.
The previous pixels, palette and picking map stay together on screen while a
replacement loads. Its tiles warm in a transparent, non-pickable layer; after
the viewport is ready, the complete replacement becomes visible. A failed
replacement leaves the previous display intact. Changing organelles or hiding
an overlay clears that display rather than retaining unrelated content.
Opacity, border and highlight controls still work during replacement loading.

Publication retains the newest three snapshots and every snapshot younger than
one hour. Older snapshots are pruned on later publications. Idle viewers whose
snapshot expires must reload; data is never silently substituted. Polling does
not rewrite debug manifests, and stale polls cannot overwrite newer build state.
Successful store validation is cached for 30 seconds, with inexpensive directory
identity checks on each poll. Validation checks both arrays at every zoom level;
sparse tile shapes come from the requested array metadata, including after an
asset replacement. Snapshot reclamation runs only after the enclosing database
transaction commits. Partial publication removes deleted and superseded
objects from the live label index and new identity sidecar. A high-water mark
prevents label reuse in unchanged chunks; old snapshots keep their original maps.
Existing format 6 stores rebuild lazily on first use.

## Probability and comparison overlays

Threshold previews upload the grayscale probability texture once per source or
mask edit. The shader applies the threshold and confidence opacity. Completed
regions and accepted polygons are erased from the prepared image so predictions
do not bleed through accepted work.

Saved-result comparisons offer two modes. **Quick preview** uses a cached,
reduced-resolution thresholded PNG. **Detailed result** builds an immutable
native-resolution label pyramid in a cancellable CPU job, using the same stored
probability replay, morphology, minimum-area filtering, polygon repair and
completed-annotation protection as confirmation. The canonical result uses its
current corrected objects or global mask. Alternatives show the result of applying
that retained prediction with current completed annotations; they are not archival
snapshots of past manual work. Comparison builds do not modify accepted geometry.

Detailed generations are keyed by the saved prediction and current annotation
identity, and checked again before publication. Missing model settings and failed
builds have explicit errors and retry controls. Quick previews use ETags; detailed
tiles use immutable URLs. Old detailed generations follow the three-version,
one-hour retention policy, and both caches are reclaimed with the last retained
reference to their probability map. Opacity changes do not rebuild either mode.
Bulk annotation edits invalidate comparisons through the registered display
revisions even when individual object timestamps are not rewritten. Comparison
and confirmation share the global-mask postprocessing and protection helpers.

The unused ER preview control/store/API island, obsolete CPU probability
colorizer, no-op tile invalidator, and unused mutable label allocation helpers
have been removed. Empty generic-NGFF plumbing has also been removed from the
segmentation panels. The generic viewer API remains available for RGB and other
image overlays; SVG overlays remain useful for editing and optimistic feedback.

## Validation and remaining scaling costs

Run the frontend unit suite and build with the scripts in `frontend/package.json`.
Backend regressions live in `segmentation/tests/test_overlay_snapshots.py` and
`segmentation/tests/test_result_overlays.py`, alongside the existing overlay, ROI,
saved-result, analysis, and job tests. Detailed comparison tests cover parity with
actual confirmation, protected manual areas, global morphology and native pixels
beyond the quick preview's resolution limit.

For real GPU verification, start Vite from `frontend` and open
`/tests/browser/overlay-gpu.html`. The harness reports PASS only after checking
label byte decoding, colors, hidden labels, picking, and repeated display edits.
On Windows Chromium/WebGL2 with NVIDIA hardware, 50 opacity updates caused zero
additional tile reads and zero texture uploads. This establishes cache behavior,
not an end-to-end latency claim. Other GPU drivers and macOS still need platform
validation.
`/tests/browser/overlay-handoff.html` additionally verifies real rendered pixels
and picking while palettes and replacement tiles are delayed, including opacity
changes during loading, after the switch,
after a failed replacement, and when hiding the overlay.

Remaining costs include directory traversal when cloning a snapshot, writing its
complete identity sidecar, fetching full palettes for server state changes, and
loading the authoritative global boolean mask. A one-hour retention window can
retain many versions during rapid geometry editing. The tile cache is bounded by
count because Viv does not expose usable decoded-byte accounting on its content.

If production profiling identifies those costs as dominant, the next steps are
versioned chunk manifests that share references without directory cloning,
palette deltas, explicit tile byte accounting, and chunked global-mask access.
On-demand tile rendering is another option, but requires snapshot-pinned geometry
and label allocation to preserve consistency between pyramid levels and picking.
It should replace the generation contract deliberately rather than introduce a
second path that can serve mixed revisions.
