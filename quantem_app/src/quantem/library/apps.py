from django.apps import AppConfig


class LibraryConfig(AppConfig):
    """Experiments and datasets -- the grouping layer over the image library."""

    name = "quantem.library"
    label = "library"
    verbose_name = "Library organisation"
