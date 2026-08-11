"""Organising the image library: experiments and the datasets inside them.

Two levels, both optional. An unorganised library works exactly as it always
did -- every image sits in no dataset and no experiment, and every screen
treats that as its own bucket rather than as a problem to nag about.

The levels exist because a fine-tune has to be scoped to something. Training a
model on "every mitochondria annotation in the library" pools images from
unrelated preparations, and the resulting model belongs to none of them. An
experiment is the boundary the user already thinks in, so it is the boundary a
fine-tune is not allowed to cross.
"""
