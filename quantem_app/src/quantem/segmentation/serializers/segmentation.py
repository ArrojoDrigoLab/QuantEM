"""Segmentation-level serializers."""

from django.db.models import Count, Q
from django.db.models.fields.json import KeyTransform
from rest_framework import serializers

from quantem.jobs.constants import ACTIVE_SEGMENTATION_JOB_TYPES
from quantem.jobs.models import Job
from quantem.segmentation.instance_params import (
    INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD,
    INSTANCE_PARAM_CENTER_MIN_DISTANCE,
    INSTANCE_PARAM_DOWNSAMPLING_FACTOR,
    INSTANCE_PARAM_SEGMENTATION_THRESHOLD,
    supports_instance_params,
)
from quantem.segmentation.models import (
    ImageSegmentation,
    ProbabilityMap,
    SegmentationConfig,
    SegmentationType,
    SegmentObject,
)
from quantem.segmentation.organelle_tasks import (
    _SUPPRESSING_LABEL_STATES,
    zero_object_notice,
)
from quantem.segmentation.run_identity import (
    RUN_FEATURE_KEY,
    calibrated_after_the_fact,
    read_run_identity,
)
from quantem.segmentation.source_models import (
    SOURCE_MODEL_MANUAL,
    get_source_model_definition,
    source_model_payload,
    source_models_for_organelle,
    unique_source_models,
)
from quantem.segmentation.type_definitions import MANUAL_ONLY_INTERNAL_NAMES

#: One short line per ``run_notice`` kind, for the chip that sits beside the
#: stage. The long form is ``message`` plus ``next_steps``; a chip cannot carry
#: those, and the one it used to compose for itself -- *"Ran and found no
#: objects"* -- is false over a proofread segmentation holding twelve confirmed
#: objects, which is exactly where the second kind fires. Written here so the
#: chip and the box beneath it cannot say different things.
RUN_NOTICE_SUMMARIES = {
    "no_objects": "Ran and found no objects",
    "no_new_objects": "Ran and added no new objects",
}


def _pixel_size_sort_key(value: object) -> tuple[int, float, str]:
    """Numbers ascending, then anything unsortable, then ``null`` last.

    A stamp is JSON out of the database and can hold whatever was written into
    it, so this cannot be ``sorted(values)`` -- one damaged row would raise a
    ``TypeError`` out of a read endpoint. ``None`` sorts last because it is the
    finding, not the smallest scale: a reader scanning ``[5.0, null]`` should
    reach the number first and the missing one at the end.
    """
    if value is None:
        return (2, 0.0, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value), "")
    return (1, 0.0, repr(value))


def _last_run_added_nothing(segmentation: ImageSegmentation) -> bool:
    """True when the most recent finished run over this segmentation made none.

    The only record of it. ``status_stage`` is ``CANDIDATES_READY`` either way,
    the segmentation stores no per-run object count, and the objects that are
    here are the *previous* run's -- so without the job row there is nothing to
    distinguish "ran and added twelve" from "ran and added nothing" once the run
    is over.

    Read from ``result_json`` rather than recomputed, because ``found_objects``
    is what the run itself reported at the moment it finished
    (:func:`quantem.jobs.handlers._segmentation_run_outcome`); counting objects
    now would answer a different question. Most recent wins, so a later run that
    does produce something clears the notice on the next read without anything
    having to remember to.
    """
    result = (
        Job.objects.filter(
            type__in=ACTIVE_SEGMENTATION_JOB_TYPES,
            status="SUCCESS",
            payload_json__segmentation_id=str(segmentation.id),
        )
        .order_by("-finished_at", "-created_at")
        .values_list("result_json", flat=True)
        .first()
    )
    if not isinstance(result, dict):
        # No run of this segmentation has ever finished here -- a CLI run, a
        # pruned queue, objects imported some other way. Nothing to report.
        return False
    return result.get("found_objects") is False


def _notice(segmentation: ImageSegmentation, *, kind: str) -> dict:
    """``zero_object_notice`` plus the two things a chip needs to render it."""
    notice = dict(zero_object_notice(segmentation))
    notice["kind"] = kind
    notice["summary"] = RUN_NOTICE_SUMMARIES[kind]
    return notice


class SegmentationTypeSerializer(serializers.ModelSerializer):
    """Read-only view of a segmentation type.

    QuantEM does not ship a corpus-catalog surface, so this is the only
    serializer exposed from it.
    """

    class Meta:
        model = SegmentationType
        fields = [
            "id",
            "internal_name",
            "short_name",
            "long_name",
            "default_color",
            "created_at",
            "updated_at",
        ]
        read_only_fields = fields


class ImageSegmentationSerializer(serializers.ModelSerializer):
    segmentation_type = SegmentationTypeSerializer(read_only=True)
    segment_counts = serializers.SerializerMethodField()
    source_models = serializers.SerializerMethodField()
    segment_counts_by_source_model = serializers.SerializerMethodField()
    config = serializers.SerializerMethodField()
    is_complete = serializers.SerializerMethodField()
    run_notice = serializers.SerializerMethodField()
    objects_pixel_size = serializers.SerializerMethodField()
    segmentation_type_id = serializers.PrimaryKeyRelatedField(
        queryset=SegmentationType.objects.all(),
        source="segmentation_type",
        write_only=True,
        required=False,
    )
    status_stage_display = serializers.CharField(
        source="get_status_stage_display",
        read_only=True,
    )

    class Meta:
        model = ImageSegmentation
        fields = [
            "id",
            "asset",
            "segmentation_type",
            "segmentation_type_id",
            "segment_counts",
            "source_models",
            "segment_counts_by_source_model",
            "config",
            "is_complete",
            "run_notice",
            "objects_pixel_size",
            # The dial position the objects on screen were found at, or ``null``
            # when nobody has moved it. Read-only here: it is written by the
            # re-extract that acts on it, never by a PATCH, because a level with
            # no matching object set is a claim about objects that do not exist.
            "include_level",
            "status_stage",
            "status_stage_display",
            "status_progress",
            "status_error",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "asset",
            "include_level",
            "status_progress",
            "status_error",
            "created_at",
            "updated_at",
        ]

    def get_segment_counts(self, obj):
        counts = {choice[0]: 0 for choice in SegmentObject.LABEL_STATE_CHOICES}
        qs = (
            SegmentObject.objects.filter(segmentation=obj)
            .values("label_state")
            .annotate(count=Count("id"))
        )
        for row in qs:
            counts[row["label_state"]] = row["count"]
        return counts

    def _source_counts(self, obj) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        rows = (
            SegmentObject.objects.filter(segmentation=obj)
            .values("source_model", "label_state")
            .annotate(count=Count("id"))
        )
        for row in rows:
            source_model = str(row["source_model"] or "")
            if not source_model:
                continue
            bucket = counts.setdefault(
                source_model,
                {choice[0]: 0 for choice in SegmentObject.LABEL_STATE_CHOICES},
            )
            bucket[str(row["label_state"])] = int(row["count"])
        return counts

    def get_source_models(self, obj):
        source_counts = self._source_counts(obj)
        catalog = source_models_for_organelle(obj.segmentation_type.internal_name)
        known_values = [definition.value for definition in catalog]
        actual_values = unique_source_models(source_counts.keys())
        values = unique_source_models([*known_values, *actual_values, SOURCE_MODEL_MANUAL])
        payloads = []
        for value in values:
            definition = get_source_model_definition(value)
            total = sum(source_counts.get(value, {}).values())
            if definition is not None:
                payloads.append(source_model_payload(definition, count=total))
            elif value == SOURCE_MODEL_MANUAL:
                payloads.append(
                    {
                        "value": SOURCE_MODEL_MANUAL,
                        "label": "Manual",
                        "model_family": "manual",
                        "variant": "",
                        "is_default": False,
                        "count": total,
                    }
                )
            else:
                payloads.append(
                    {
                        "value": value,
                        "label": value,
                        "model_family": value.split(":", 1)[0],
                        "variant": "",
                        "is_default": False,
                        "count": total,
                    }
                )
        return payloads

    def get_segment_counts_by_source_model(self, obj):
        source_counts = self._source_counts(obj)
        for definition in source_models_for_organelle(obj.segmentation_type.internal_name):
            source_counts.setdefault(
                definition.value,
                {choice[0]: 0 for choice in SegmentObject.LABEL_STATE_CHOICES},
            )
        source_counts.setdefault(
            SOURCE_MODEL_MANUAL,
            {choice[0]: 0 for choice in SegmentObject.LABEL_STATE_CHOICES},
        )
        confirmed_total = int(
            SegmentObject.objects.filter(
                segmentation=obj,
                label_state="CONFIRMED",
            ).count()
        )
        for bucket in source_counts.values():
            bucket["CONFIRMED"] = confirmed_total
        return source_counts

    def get_config(self, obj):
        try:
            config = obj.config
        except SegmentationConfig.DoesNotExist:
            return None
        supports_params = supports_instance_params(obj.segmentation_type.internal_name)
        return {
            "supports_instance_params": supports_params,
            "instance_params": config.get_instance_params() if supports_params else None,
        }

    def get_is_complete(self, obj):
        return obj.status_stage == "COMPLETED"

    def get_objects_pixel_size(self, obj):
        """What the image's pixel size was **when these objects were made**.

        The labeling header says ``5 nm/px · entered by hand`` and an ordinary
        objects chip over a set produced before that number existed. Nothing on
        the screen where a user decides the work is finished said so, and
        neither did the Analysis screen before a run was spent: it surfaced in
        the finished bundle, as blank micron columns and ``calibrated: false``.

        It is read off the objects' own run stamps rather than inferred from the
        asset. Inferring it from the asset's value alone fires on every
        calibrated image -- the crying-wolf failure a previous round already
        fixed on the import form -- and the stamp is the only record of what the
        run actually saw.

            ``produced_nm``
                Every distinct ``native_pixel_size_nm`` the stamps recorded,
                numbers first and ``null`` last. ``null`` is a real member: it
                is a run that had no pixel size to resample with, not a gap in
                the record. Two entries means two runs at different scales, and
                the objects are not one population.
            ``predates_calibration``
                :func:`~quantem.segmentation.run_identity.calibrated_after_the_fact`,
                the same predicate ``run_analysis`` blanks its physical units
                on. Imported rather than restated: a screen that says the
                objects are fine over a bundle that refuses to convert them is a
                disagreement nobody sees until they compare the two.
            ``unstamped_count``
                Objects carrying no run at all -- hand-drawn outlines, or
                objects made before stamping existed. Counted, never folded into
                ``produced_nm``: "not produced by a model" is not "produced at
                an unknown scale", and treating it as the latter is what would
                tell someone to discard their own polygons.

        ``null`` when the segmentation holds no objects, because every field
        above would then be a statement about nothing.

        One query, grouped in the database on the ``run`` sub-object, so the
        cost does not grow with the object count -- this runs per segmentation
        in the list the labeling screen polls, and pulling every ``features``
        blob into Python to find two distinct numbers would not survive an image
        with three thousand objects.
        """
        rows = (
            SegmentObject.objects.filter(segmentation=obj)
            .values(run=KeyTransform(RUN_FEATURE_KEY, "features"))
            .annotate(count=Count("id"))
            .order_by()
        )

        # One row per distinct ``run`` sub-object -- one per inference run, not
        # one per object -- so this loop is over runs however many objects there
        # are. ``list`` and ``in`` rather than a set: a damaged stamp can hold an
        # unhashable value, and that is not a reason to fail a read.
        produced: list[object] = []
        unstamped = 0
        total = 0
        for row in rows:
            count = int(row["count"])
            total += count
            # The same reader the analysis manifest uses, so "stamped" means the
            # same thing on both sides: a ``run`` dict carrying a real id.
            stamp = read_run_identity({RUN_FEATURE_KEY: row["run"]})
            if stamp is None:
                unstamped += count
                continue
            value = stamp.get("native_pixel_size_nm")
            if isinstance(value, float):
                value = round(value, 12)
            if value not in produced:
                produced.append(value)

        if total == 0:
            return None

        produced.sort(key=_pixel_size_sort_key)
        return {
            "produced_nm": produced,
            "predates_calibration": calibrated_after_the_fact(
                produced_pixel_size_nm=produced,
                recorded_pixel_size_nm=getattr(obj.asset, "pixel_size_nm", None),
            ),
            "unstamped_count": unstamped,
        }

    def get_run_notice(self, obj):
        """What the stage does not say, when the stage alone would mislead.

        ``CANDIDATES_READY`` is the stage a finished run leaves behind whether
        it produced two hundred objects or none, so "Candidates ready" was the
        whole of what a user with zero objects was told; the explanation and the
        three things to check went to the job log, which no screen renders.
        ``null`` in every other case, so a client can render this whenever it is
        present without deciding when it applies.

        Derived, not stored: it answers "what is true of this segmentation
        *now*", so confirming one object makes it disappear on the next read
        rather than lingering as a record of a run that has since been fixed.

        **There are two empty runs, and only one of them used to get here.**
        Suppressing the notice the moment any ``SegmentObject`` existed made
        :func:`~quantem.segmentation.organelle_tasks._zero_object_advice`'s
        proofread branch unreachable by construction -- that branch needs
        ``labelled > 0``, and one labelled object was enough to return ``None``
        above it. So a user with 12 confirmed objects clicked Run Full
        Segmentation, got SUCCESS in four seconds with nothing new, and read
        "Candidates ready" while the sentences that explained it sat in
        ``job.result_json.next_steps``, which nothing renders. They polled for
        two and a half minutes to be sure. That is this application's own
        recommended remedy -- *"Set the image's pixel size and re-run
        inference"* -- completing successfully, doing nothing, and withholding
        both the reason and the route that does work.

        The two cases are told apart by what is labelled here, not by the object
        count, because they need opposite wording:

        * **Nothing labelled.** Zero objects and the run really did find
          nothing: the empty-run advice, unchanged.
        * **Something labelled, and the last finished run added none of it.**
          Extraction drops a candidate that lands on a confirmed or excluded
          object, so this outcome is *expected*; the notice says so, and names
          ``labels/clear`` when the labelled objects predate the image's pixel
          size and no re-run can lift that.

        A segmentation holding only unlabelled candidates gets ``null``: nothing
        there suppresses anything, so a run that added nothing has no benign
        explanation to offer, and the empty-run wording would be false over the
        candidates on screen.

        The two original guards still hold. A manual-only type
        (``MANUAL_ONLY_INTERNAL_NAMES`` -- tissue) is set to
        ``CANDIDATES_READY`` at creation with no run behind it and no model that
        could have found anything, and would otherwise be told to lower a
        threshold it does not have.
        """
        if obj.status_stage != "CANDIDATES_READY":
            return None
        if obj.segmentation_type.internal_name in MANUAL_ONLY_INTERNAL_NAMES:
            return None

        counts = SegmentObject.objects.filter(segmentation=obj).aggregate(
            total=Count("id"),
            # Deliberately the tuple `_zero_object_advice` branches on,
            # imported rather than repeated: this method chooses which of that
            # function's two branches to publish, and a private copy here would
            # eventually ask for one and be handed the other.
            labelled=Count("id", filter=Q(label_state__in=_SUPPRESSING_LABEL_STATES)),
        )
        if counts["labelled"]:
            if not _last_run_added_nothing(obj):
                return None
            return _notice(obj, kind="no_new_objects")
        if counts["total"]:
            return None
        return _notice(obj, kind="no_objects")


class ImageSegmentationCreateSerializer(serializers.Serializer):
    segmentation_type_id = serializers.UUIDField(required=False)
    segmentation_type_name = serializers.CharField(max_length=100, required=False)
    source_model = serializers.CharField(max_length=128, required=False, allow_blank=True)

    def validate(self, attrs):
        if not attrs.get("segmentation_type_id") and not attrs.get("segmentation_type_name"):
            raise serializers.ValidationError(
                "Either 'segmentation_type_id' or 'segmentation_type_name' must be provided."
            )
        if attrs.get("segmentation_type_id") and attrs.get("segmentation_type_name"):
            raise serializers.ValidationError(
                "Provide either 'segmentation_type_id' or 'segmentation_type_name', not both."
            )
        return attrs


class SegmentationInstanceParamsSerializer(serializers.Serializer):
    center_min_distance = serializers.IntegerField(min_value=1, max_value=512)
    center_confidence_threshold = serializers.FloatField(min_value=0.0, max_value=1.0)
    segmentation_threshold = serializers.FloatField(min_value=0.0, max_value=1.0)
    downsampling_factor = serializers.IntegerField(min_value=1, max_value=16, allow_null=True)


class SegmentationInstanceParamsPatchSerializer(serializers.Serializer):
    center_min_distance = serializers.IntegerField(min_value=1, max_value=512, required=False)
    center_confidence_threshold = serializers.FloatField(
        min_value=0.0, max_value=1.0, required=False
    )
    segmentation_threshold = serializers.FloatField(min_value=0.0, max_value=1.0, required=False)
    downsampling_factor = serializers.IntegerField(
        min_value=1,
        max_value=16,
        allow_null=True,
        required=False,
    )

    def validate(self, attrs):
        if not attrs:
            raise serializers.ValidationError("Provide at least one instance parameter field.")
        return attrs

    def to_instance_params_update(self) -> dict[str, int | float | None]:
        mapping = {
            "center_min_distance": INSTANCE_PARAM_CENTER_MIN_DISTANCE,
            "center_confidence_threshold": INSTANCE_PARAM_CENTER_CONFIDENCE_THRESHOLD,
            "segmentation_threshold": INSTANCE_PARAM_SEGMENTATION_THRESHOLD,
            "downsampling_factor": INSTANCE_PARAM_DOWNSAMPLING_FACTOR,
        }
        return {mapping[key]: value for key, value in self.validated_data.items()}


class SegmentationConfigResponseSerializer(serializers.Serializer):
    supports_instance_params = serializers.BooleanField()
    instance_params = SegmentationInstanceParamsSerializer(allow_null=True)


class ProbabilityMapSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProbabilityMap
        fields = [
            "id",
            "name",
            "file_path",
            "channel_index",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "file_path", "created_at", "updated_at"]


class SegmentationOverlayRebuildSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=["partial", "full"], default="full")
    source_model = serializers.CharField(max_length=128, required=False, allow_blank=True)
