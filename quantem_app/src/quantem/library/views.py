"""HTTP surface for experiments, datasets, and putting images in them.

Plain :class:`~rest_framework.views.APIView` classes and explicit ``path()``
entries, which is how every other app in this tree is written -- no router, no
viewset, no new dependency. Error bodies are ``{"detail": "..."}`` in the
application's own voice.

Nothing here requires advance setup. A one-image import creates its own
experiment automatically, and every list answers with an empty list before the
first import.
"""

from __future__ import annotations

from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from django.db.models import Prefetch
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from quantem.assets.asset_mutations import update_asset
from quantem.assets.models import Asset
from quantem.library.grouping import (
    UNSET,
    apply_grouping,
    resolve_dataset,
    resolve_experiment,
)
from quantem.library.models import Dataset, Experiment, create_image_experiment
from quantem.library.serializers import (
    DatasetWriteSerializer,
    ExperimentWriteSerializer,
    serialize_dataset,
    serialize_experiment,
)

#: What a client is told when it names something that has since been deleted.
#: One sentence, no identifier: the row it asked for is gone and the only
#: useful next step is to look at the list again.
EXPERIMENT_GONE = "That experiment is no longer in the library. Refresh to see what is there now."
DATASET_GONE = "That dataset is no longer in the library. Refresh to see what is there now."

#: The largest number of images one assignment may touch. Far past a plate of
#: forty and past any library this application is built for, and it exists only
#: so a malformed request cannot ask the server to load a million rows.
MAX_ASSETS_PER_ASSIGNMENT = 5000


def _experiment_queryset():
    """Experiments with everything :func:`serialize_experiment` reads."""
    return Experiment.objects.prefetch_related(
        Prefetch("datasets", queryset=Dataset.objects.order_by("name", "created_at"))
    )


def _detail(message: str, code: int) -> Response:
    return Response({"detail": message}, status=code)


def _flatten(detail) -> str | None:
    """The first sentence out of a nested DRF or Django error structure."""
    if isinstance(detail, dict):
        for value in detail.values():
            found = _flatten(value)
            if found:
                return found
        return None
    if isinstance(detail, (list, tuple)):
        for value in detail:
            found = _flatten(value)
            if found:
                return found
        return None
    text = str(detail).strip() if detail is not None else ""
    return text or None


def _first_error(exc) -> str:
    """One sentence out of a Django ``ValidationError``."""
    detail = getattr(exc, "messages", None) or getattr(exc, "detail", None)
    return _flatten(detail) or str(exc)


def _first_serializer_error(serializer) -> str:
    """One sentence out of ``serializer.errors``.

    A dict of field to list of messages is the wrong shape for a screen that
    shows one line, and the field names are the serializer's, not the form's.
    The first message is the one the user can act on.
    """
    return _flatten(serializer.errors) or "That could not be saved. Check the name and try again."


class ExperimentListCreateView(APIView):
    """List every experiment, or make one.

    The list carries each experiment's datasets and its image counts, because
    every caller wants them together: the library's filter, the import form's
    two pickers, and the tree a fine-tune's scope is chosen from.
    """

    def get(self, request):
        del request
        return Response([serialize_experiment(row) for row in _experiment_queryset()])

    def post(self, request):
        serializer = ExperimentWriteSerializer(data=request.data or {})
        if not serializer.is_valid():
            return _detail(_first_serializer_error(serializer), status.HTTP_400_BAD_REQUEST)
        experiment = Experiment.objects.create(
            name=serializer.validated_data["name"],
            notes=serializer.validated_data.get("notes", ""),
        )
        return Response(serialize_experiment(experiment), status=status.HTTP_201_CREATED)


class ExperimentDetailView(APIView):
    """Read, rename or delete one experiment.

    Deleting one **keeps every image**. Each active image moves into a new
    experiment named after its display name before the old experiment and its
    datasets are removed. Tombstoned rows are detached without creating
    visible empty experiments.
    """

    def _get(self, experiment_id) -> Experiment | None:
        return _experiment_queryset().filter(id=experiment_id).first()

    def get(self, request, experiment_id):
        del request
        experiment = self._get(experiment_id)
        if experiment is None:
            return _detail(EXPERIMENT_GONE, status.HTTP_404_NOT_FOUND)
        return Response(serialize_experiment(experiment))

    def patch(self, request, experiment_id):
        experiment = self._get(experiment_id)
        if experiment is None:
            return _detail(EXPERIMENT_GONE, status.HTTP_404_NOT_FOUND)
        serializer = ExperimentWriteSerializer(
            instance=experiment, data=request.data or {}, partial=True
        )
        if not serializer.is_valid():
            return _detail(_first_serializer_error(serializer), status.HTTP_400_BAD_REQUEST)
        updated = []
        for field in ("name", "notes"):
            if field in serializer.validated_data:
                setattr(experiment, field, serializer.validated_data[field])
                updated.append(field)
        if updated:
            experiment.save(update_fields=[*updated, "updated_at"])
        return Response(serialize_experiment(experiment))

    def delete(self, request, experiment_id):
        del request
        experiment = self._get(experiment_id)
        if experiment is None:
            return _detail(EXPERIMENT_GONE, status.HTTP_404_NOT_FOUND)
        with transaction.atomic():
            assets = list(
                Asset.objects.select_for_update()
                .filter(experiment_id=experiment.id)
                .prefetch_related("datasets")
            )
            for asset in assets:
                asset.datasets.clear()
                if asset.lifecycle_status == Asset.LIFECYCLE_ACTIVE:
                    replacement = create_image_experiment(asset.display_name)
                    Asset.objects.filter(id=asset.id).update(experiment=replacement)
                else:
                    Asset.objects.filter(id=asset.id).update(experiment=None)
            experiment.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class DatasetListCreateView(APIView):
    """List datasets, or make one inside an experiment.

    ``?experiment=`` narrows the list. Without it every dataset is returned,
    which is what a flat picker wants.
    """

    def get(self, request):
        datasets = Dataset.objects.select_related("experiment")
        experiment_id = request.query_params.get("experiment")
        if experiment_id:
            datasets = datasets.filter(experiment_id=experiment_id)
        return Response([serialize_dataset(row) for row in datasets])

    def post(self, request):
        serializer = DatasetWriteSerializer(data=request.data or {})
        if not serializer.is_valid():
            return _detail(_first_serializer_error(serializer), status.HTTP_400_BAD_REQUEST)
        dataset = Dataset.objects.create(
            experiment_id=serializer.validated_data["experiment"],
            name=serializer.validated_data["name"],
            notes=serializer.validated_data.get("notes", ""),
        )
        return Response(serialize_dataset(dataset), status=status.HTTP_201_CREATED)


class DatasetDetailView(APIView):
    """Read, rename or delete one dataset.

    Deleting a dataset keeps its images too. They stay in the experiment and
    simply stop being in this subset of it.
    """

    def _get(self, dataset_id) -> Dataset | None:
        return Dataset.objects.select_related("experiment").filter(id=dataset_id).first()

    def get(self, request, dataset_id):
        del request
        dataset = self._get(dataset_id)
        if dataset is None:
            return _detail(DATASET_GONE, status.HTTP_404_NOT_FOUND)
        return Response(serialize_dataset(dataset))

    def patch(self, request, dataset_id):
        dataset = self._get(dataset_id)
        if dataset is None:
            return _detail(DATASET_GONE, status.HTTP_404_NOT_FOUND)
        serializer = DatasetWriteSerializer(instance=dataset, data=request.data or {}, partial=True)
        if not serializer.is_valid():
            return _detail(_first_serializer_error(serializer), status.HTTP_400_BAD_REQUEST)
        updated = []
        for field in ("name", "notes"):
            if field in serializer.validated_data:
                setattr(dataset, field, serializer.validated_data[field])
                updated.append(field)
        if updated:
            dataset.save(update_fields=[*updated, "updated_at"])
        return Response(serialize_dataset(dataset))

    def delete(self, request, dataset_id):
        del request
        dataset = self._get(dataset_id)
        if dataset is None:
            return _detail(DATASET_GONE, status.HTTP_404_NOT_FOUND)
        dataset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class AssetGroupingView(APIView):
    """Put images into an experiment and a dataset, or take them out.

    One endpoint for one image and for forty, because the library selects
    images and then acts on the selection, and a single-image path would be the
    same code with a different arity.

    The three fields are independent and each is optional:

    * omit ``experiment`` entirely and the images keep the one they have;
    * send it as ``null`` and each image gets its own named experiment;
    * send ``experiment_name`` instead of an id to create one on the way past,
      which is what "type a new name" in the picker does.

    ``datasets`` behaves the same way, inside whichever experiment the images
    end up in. An image moved to another experiment leaves any dataset the new
    experiment does not contain -- see :mod:`quantem.library.grouping` -- and
    the response says how many memberships that cost so the screen can report
    it rather than the user discovering it later.
    """

    def post(self, request):
        payload = request.data or {}

        asset_ids = payload.get("asset_ids") or []
        if not isinstance(asset_ids, (list, tuple)) or not asset_ids:
            return _detail(
                "Choose at least one image to organise.",
                status.HTTP_400_BAD_REQUEST,
            )
        if len(asset_ids) > MAX_ASSETS_PER_ASSIGNMENT:
            return _detail(
                "That is more images than one change can cover. Select fewer and try again.",
                status.HTTP_400_BAD_REQUEST,
            )

        assets = list(
            Asset.objects.filter(
                id__in=[str(value) for value in asset_ids],
                lifecycle_status=Asset.LIFECYCLE_ACTIVE,
            ).prefetch_related("datasets")
        )
        if not assets:
            return _detail(
                "None of those images are in the library any more. Refresh and try again.",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            experiment, datasets = self._resolve(payload)
        except DjangoValidationError as exc:
            return _detail(_first_error(exc), status.HTTP_400_BAD_REQUEST)

        mode = str(payload.get("datasets_mode") or "replace")
        if mode not in {"replace", "add"}:
            return _detail(
                "That is not a way to change dataset membership.",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            outcome = apply_grouping(
                assets,
                experiment=experiment,
                datasets=datasets,
                datasets_mode=mode,
            )
        except DjangoValidationError as exc:
            return _detail(_first_error(exc), status.HTTP_400_BAD_REQUEST)

        resolved_experiment = None if experiment is UNSET else experiment
        return Response(
            {
                "assets_changed": outcome.assets_changed,
                "dataset_links_dropped": outcome.dataset_links_dropped,
                "assets_moved_out_of_datasets": (outcome.assets_moved_out_of_datasets),
                "datasets_left": outcome.datasets_left,
                "experiment": (
                    serialize_experiment(resolved_experiment)
                    if resolved_experiment is not None
                    else None
                ),
                "datasets": [
                    serialize_dataset(dataset)
                    for dataset in (datasets if isinstance(datasets, list) else [])
                ],
            }
        )

    def _resolve(self, payload: dict):
        """Turn the request's ids and typed names into rows, or leave them be.

        The tri-state is what makes "move these into an experiment without
        touching their datasets" expressible; see
        :mod:`quantem.library.grouping`.
        """
        mentions_experiment = "experiment" in payload or "experiment_name" in payload
        experiment = UNSET
        if mentions_experiment:
            experiment = resolve_experiment(
                experiment_id=payload.get("experiment"),
                experiment_name=payload.get("experiment_name"),
            )

        mentions_datasets = "datasets" in payload or "dataset_name" in payload
        if not mentions_datasets:
            return experiment, UNSET

        # Datasets are resolved inside whichever experiment the images will end
        # up in. When the request does not mention one, that is the experiment
        # the dataset itself belongs to, and `resolve_dataset` accepts it.
        scope = None if experiment is UNSET else experiment
        resolved: list[Dataset] = []
        for dataset_id in payload.get("datasets") or []:
            dataset = resolve_dataset(experiment=scope, dataset_id=dataset_id)
            if dataset is not None:
                resolved.append(dataset)
        typed = resolve_dataset(experiment=scope, dataset_name=payload.get("dataset_name"))
        if typed is not None and typed not in resolved:
            resolved.append(typed)
        return experiment, resolved


class AssetLibraryEditView(APIView):
    """Atomically edit one image's details and dataset membership."""

    def patch(self, request, asset_id):
        payload = request.data or {}
        try:
            with transaction.atomic():
                asset = (
                    Asset.objects.select_for_update()
                    .filter(id=asset_id, lifecycle_status=Asset.LIFECYCLE_ACTIVE)
                    .first()
                )
                if asset is None:
                    return _detail(
                        "That image is no longer in the library.",
                        status.HTTP_404_NOT_FOUND,
                    )
                dataset_ids = payload.get("datasets", UNSET)
                datasets = UNSET
                if dataset_ids is not UNSET:
                    if not isinstance(dataset_ids, (list, tuple)):
                        raise DjangoValidationError("Datasets must be a list.")
                    datasets = []
                    for dataset_id in dataset_ids:
                        dataset = resolve_dataset(
                            experiment=asset.experiment,
                            dataset_id=dataset_id,
                        )
                        if dataset is not None:
                            datasets.append(dataset)

                detail_payload = {
                    key: payload[key]
                    for key in ("display_name", "pixel_size_nm", "notes")
                    if key in payload
                }
                update_asset(asset, detail_payload, inside_transaction=True)
                apply_grouping(
                    [asset],
                    experiment=UNSET,
                    datasets=datasets,
                    datasets_mode="replace",
                )
        except (DjangoValidationError, ValueError) as exc:
            return _detail(_first_error(exc), status.HTTP_400_BAD_REQUEST)

        asset.refresh_from_db()
        from quantem.assets.serializers import serialize_asset_detail

        return Response(serialize_asset_detail(asset))
