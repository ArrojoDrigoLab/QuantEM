"""Organising the image library: experiments and the datasets inside them.

Every active image belongs to exactly one experiment. Datasets are optional
named subsets inside that experiment. A simple one-image import automatically
creates an experiment named after the image, so this invariant adds no setup
step for the user.

The levels exist because a fine-tune has to be scoped to something. Training a
model on "every mitochondria annotation in the library" pools images from
unrelated preparations, and the resulting model belongs to none of them. An
experiment is the boundary the user already thinks in, so it is the boundary a
fine-tune is not allowed to cross.
"""
