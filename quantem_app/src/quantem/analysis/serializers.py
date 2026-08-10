"""Serializers for the analysis API.

The response shape is fixed by ``API_CONTRACT.md`` §Analysis and is deliberately
*flat*: the frontend reads ``composition``/``objects``/``points``/``distances``/
``monte_carlo`` at the top level, not nested under ``results``. Flattening here
rather than in the frontend keeps the contract in one language.
"""

from __future__ import annotations

from typing import Any

from rest_framework import serializers

from .distances import DEFAULT_BAND_EDGES_NM
from .loaders import POINT_SOURCES, AnalysisInputError, normalise_params
from .models import AnalysisRun
from .montecarlo import DEFAULT_REPLICATES, DEFAULT_SEED


class AnalysisRunCreateSerializer(serializers.Serializer):
    """Validates ``POST /api/segmentations/<seg_id>/analysis/``.

    Field shapes are checked here; everything that needs the database (do these
    segmentation ids exist, are they on the same image, is the distance target
    one of the compartments) is checked by
    :func:`quantem.analysis.loaders.normalise_params`, so the API and the job
    cannot disagree about what was asked for.

    Requires ``context["segmentation"]``.
    """

    compartments = serializers.DictField(
        child=serializers.CharField(), required=False, default=dict
    )
    tissue_segmentation_id = serializers.CharField(
        required=False, allow_null=True, allow_blank=True, default=None
    )
    points_source = serializers.ChoiceField(
        choices=POINT_SOURCES, required=False, allow_null=True, default=None
    )
    points_csv = serializers.CharField(
        required=False, allow_blank=True, default="", trim_whitespace=False
    )
    distance_target = serializers.CharField(
        required=False, allow_null=True, allow_blank=True, default=None
    )
    band_edges_nm = serializers.ListField(
        child=serializers.FloatField(),
        required=False,
        default=lambda: list(DEFAULT_BAND_EDGES_NM),
    )
    replicates = serializers.IntegerField(required=False, default=DEFAULT_REPLICATES)
    seed = serializers.IntegerField(required=False, default=DEFAULT_SEED)
    group = serializers.CharField(required=False, allow_blank=True, default="")

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        segmentation = self.context.get("segmentation")
        if segmentation is None:  # pragma: no cover - wiring error, not user input
            raise RuntimeError("AnalysisRunCreateSerializer needs a segmentation.")
        try:
            return normalise_params(attrs, segmentation=segmentation)
        except AnalysisInputError as exc:
            raise serializers.ValidationError(str(exc)) from exc


class AnalysisRunSummarySerializer(serializers.ModelSerializer):
    """One row of the run list. Enough to pick a run, not enough to plot one."""

    id = serializers.CharField(read_only=True)
    n_objects = serializers.SerializerMethodField()
    calibrated = serializers.SerializerMethodField()
    n_caveats = serializers.SerializerMethodField()
    segmentation_deleted = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisRun
        fields = [
            "id",
            "status",
            "group",
            "created_at",
            "started_at",
            "finished_at",
            "export_dir",
            "error",
            "n_objects",
            "calibrated",
            "n_caveats",
            "segmentation_deleted",
        ]
        read_only_fields = fields

    def get_segmentation_deleted(self, obj: AnalysisRun) -> bool:
        """See :meth:`AnalysisRunSerializer.get_segmentation_deleted`."""
        return obj.segmentation_id is None

    def get_n_objects(self, obj: AnalysisRun) -> int | None:
        return ((obj.results or {}).get("objects") or {}).get("n")

    def get_calibrated(self, obj: AnalysisRun) -> bool | None:
        return (obj.results or {}).get("calibrated")

    def get_n_caveats(self, obj: AnalysisRun) -> int:
        return len((obj.results or {}).get("caveats") or [])


class AnalysisRunSerializer(serializers.ModelSerializer):
    """The full run, with ``results`` flattened to the top level.

    Sections absent from a run (no points imported, no distance target,
    uncalibrated so no nanometres) are returned as ``null`` rather than omitted:
    a missing key and a computed-but-empty section are different states and the
    UI has to be able to tell them apart.
    """

    id = serializers.CharField(read_only=True)
    segmentation_id = serializers.CharField(read_only=True)
    segmentation_deleted = serializers.SerializerMethodField()
    pixel_size_nm = serializers.SerializerMethodField()
    calibrated = serializers.SerializerMethodField()
    composition = serializers.SerializerMethodField()
    objects = serializers.SerializerMethodField()
    points = serializers.SerializerMethodField()
    distances = serializers.SerializerMethodField()
    monte_carlo = serializers.SerializerMethodField()
    monte_carlo_self_check = serializers.SerializerMethodField()
    caveats = serializers.SerializerMethodField()
    exports = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisRun
        fields = [
            "id",
            "segmentation_id",
            "segmentation_deleted",
            "status",
            "group",
            "created_at",
            "started_at",
            "finished_at",
            "params",
            "pixel_size_nm",
            "calibrated",
            "composition",
            "objects",
            "points",
            "distances",
            "monte_carlo",
            "monte_carlo_self_check",
            "caveats",
            "export_dir",
            "exports",
            "error",
        ]
        read_only_fields = fields

    def _section(self, obj: AnalysisRun, key: str) -> Any:
        return (obj.results or {}).get(key)

    def get_segmentation_deleted(self, obj: AnalysisRun) -> bool:
        """True when the segmentation this run measured has since been deleted.

        Every run is created with a segmentation and
        ``DELETE /api/segmentations/<id>/`` is the only thing that nulls the
        reference (``on_delete=SET_NULL``), so a null id has exactly one
        meaning. The run's numbers and export bundle remain the record of an
        analysis that happened; this flag says its objects can no longer be
        revisited, so the run cannot be reproduced from the app.
        """
        return obj.segmentation_id is None

    def get_pixel_size_nm(self, obj: AnalysisRun) -> float | None:
        return self._section(obj, "pixel_size_nm")

    def get_calibrated(self, obj: AnalysisRun) -> bool | None:
        return self._section(obj, "calibrated")

    def get_composition(self, obj: AnalysisRun) -> Any:
        return self._section(obj, "composition")

    def get_objects(self, obj: AnalysisRun) -> Any:
        return self._section(obj, "objects")

    def get_points(self, obj: AnalysisRun) -> Any:
        return self._section(obj, "points")

    def get_distances(self, obj: AnalysisRun) -> Any:
        return self._section(obj, "distances")

    def get_monte_carlo(self, obj: AnalysisRun) -> Any:
        return self._section(obj, "monte_carlo")

    def get_monte_carlo_self_check(self, obj: AnalysisRun) -> Any:
        return self._section(obj, "monte_carlo_self_check")

    def get_caveats(self, obj: AnalysisRun) -> list[str]:
        return self._section(obj, "caveats") or []

    def get_exports(self, obj: AnalysisRun) -> list[str]:
        """Bundle files that actually exist on disk, newest run state first."""
        path = obj.export_path
        if path is None or not path.is_dir():
            return []
        return sorted(entry.name for entry in path.iterdir() if entry.is_file())
