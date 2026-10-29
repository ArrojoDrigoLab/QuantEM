"""Either feature schema upgrades to the common schema without losing rows."""

from copy import deepcopy
from uuid import uuid4

import pytest
from django.db import connections
from django.db.migrations.executor import MigrationExecutor
from django.db.utils import ConnectionHandler


@pytest.fixture(params=["sqlite", "postgresql"])
def feature_database(request, tmp_path, django_db_blocker):
    """Exercise each history from a fresh database, without reversing siblings."""
    original = connections["default"]
    database_name = None
    if request.param == "postgresql":
        if original.vendor != "postgresql":
            pytest.skip("Requires the PostgreSQL integration lane")
        config = deepcopy(original.settings_dict)
        database_name = f"test_quantem_features_{uuid4().hex}"
        config["NAME"] = database_name
        with django_db_blocker.unblock(), original.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {original.ops.quote_name(database_name)}")
    else:
        config = {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(tmp_path / "upgrade.sqlite3"),
        }
    isolated = ConnectionHandler({"default": config})["default"]
    try:
        connections["default"] = isolated
        yield isolated
    finally:
        isolated.close()
        connections["default"] = original
        if database_name:
            with django_db_blocker.unblock(), original.cursor() as cursor:
                cursor.execute(f"DROP DATABASE {original.ops.quote_name(database_name)}")


@pytest.mark.parametrize(
    "start",
    [
        "0010_segmentation_definitions_and_area_instances",
        "0011_model_application_batch",
    ],
)
def test_upgrade_from_either_feature_history_preserves_data(
    start, feature_database, django_db_blocker
):
    isolated = feature_database
    final = ("segmentation", "0012_reconcile_definitions_and_model_results")
    with django_db_blocker.unblock():
        executor = MigrationExecutor(isolated)
        targets = [node for node in executor.loader.graph.leaf_nodes() if node[0] != "segmentation"]
        targets.append(("segmentation", start))
        executor.migrate(targets)
        apps = executor.loader.project_state(targets).apps
        experiment = apps.get_model("library", "Experiment").objects.create(name="Schema upgrade")
        asset = apps.get_model("assets", "Asset").objects.create(
            display_name="Schema upgrade",
            experiment=experiment,
            logical_width=32,
            logical_height=32,
        )
        definition = apps.get_model("segmentation", "SegmentationType").objects.create(
            internal_name="upgrade_custom",
            short_name="Upgrade",
            long_name="Upgrade organelle",
        )
        segmentation = apps.get_model("segmentation", "ImageSegmentation").objects.create(
            asset_id=asset.id,
            segmentation_type=definition,
        )
        result_id = None
        if start == "0011_model_application_batch":
            result = apps.get_model("segmentation", "SavedSegmentationResult").objects.create(
                segmentation=segmentation,
                name="Retained result",
                full_image=True,
                is_canonical=True,
            )
            result_id = result.id
        else:
            definition.category = "area"
            definition.representation = "semantic"
            definition.save()
        executor = MigrationExecutor(isolated)
        executor.migrate([final])
        apps = executor.loader.project_state([final]).apps
        saved = apps.get_model("segmentation", "ImageSegmentation").objects.get(pk=segmentation.pk)
        assert saved.asset_id == asset.id
        assert saved.segmentation_type_id == definition.pk
        if result_id:
            assert (
                apps.get_model("segmentation", "SavedSegmentationResult")
                .objects.get(pk=result_id)
                .is_canonical
            )
        else:
            assert (
                apps.get_model("segmentation", "SegmentationType")
                .objects.get(pk=definition.pk)
                .category
                == "area"
            )
        leaves = executor.loader.graph.leaf_nodes("segmentation")
        assert len(leaves) == 1
        assert final in executor.loader.graph.forwards_plan(leaves[0])
