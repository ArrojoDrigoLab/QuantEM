"""Box-prompted object adding: draw a rectangle, get a segmented object.

The user drags a box on the labeling view and one object is stored. There is no
queue and no polling -- :mod:`quantem.sam.views` does the whole thing in the
request, because the expensive half (the image encoder) is cached per crop
window, so the second box in a neighbourhood costs a decoder pass of a few tens
of milliseconds.

Nothing here imports torch at module scope. Django starts this app on every
launch, including launches that never prompt a box, and the backend is built
lazily on the first prompt.
"""
