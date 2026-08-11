from django.apps import AppConfig


class SamConfig(AppConfig):
    """Box-prompted object adding.

    Holds no models and needs no migration, so it does not have to be in
    ``INSTALLED_APPS`` for the feature to work -- the one wiring line that
    matters is the ``include`` of :mod:`quantem.sam.urls` from
    ``quantem.core.urls``.
    """

    name = "quantem.sam"
    label = "sam"
    verbose_name = "Box-prompted object adding"
