"""Django app config for the quantitative analysis suite.

Everything in :mod:`compartments`, :mod:`distances`, :mod:`montecarlo`,
:mod:`morphometrics` and :mod:`rollup` is pure numpy and stays that way -- those
modules import no Django and are tested without a database. This app config
exists for one reason: :class:`~quantem.analysis.models.AnalysisRun` has to be a
row somewhere, because a run is started by an HTTP request, executed later by
the job queue, and read back long after both have finished.
"""

from django.apps import AppConfig


class AnalysisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "quantem.analysis"
