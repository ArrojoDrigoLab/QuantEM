import re
import uuid

import django.db.models.deletion
from django.db import migrations, models

TISSUE = "quantem_internal_tissue"
CELLS = "quantem_internal_cells"
LEGACY_ANALYSIS_MASK = "quantem_internal_analysis_mask"
SAMPLE_ALIASES = {"tissue", "tissue mask", "sample area"}
CELL_NAME = re.compile(r"^Cell\s+(\d+)$", re.IGNORECASE)
OBJECT_NAME = re.compile(r"^Object\s+(\d+)$", re.IGNORECASE)


def _free_builtin_names(SegmentationType, *, using, short_name, long_name, internal_name):
    manager = SegmentationType.objects.using(using)

    def replacement(row, field):
        original = getattr(row, field)
        base = f"{original} (custom)"
        candidate = base
        suffix = 2
        lookup = f"{field}__iexact"
        while manager.exclude(id=row.id).filter(**{lookup: candidate}).exists():
            candidate = f"{base} {suffix}"
            suffix += 1
        return candidate

    for row in manager.exclude(internal_name=internal_name).filter(short_name__iexact=short_name):
        row.short_name = replacement(row, "short_name")
        row.save(using=using, update_fields=["short_name"])
    for row in manager.exclude(internal_name=internal_name).filter(long_name__iexact=long_name):
        row.long_name = replacement(row, "long_name")
        row.save(using=using, update_fields=["long_name"])


def _classify_types(SegmentationType, *, using):
    manager = SegmentationType.objects.using(using)
    builtins = {
        "quantem_internal_mito": ("organelle", "instances", "non_overlapping"),
        "quantem_internal_er": ("organelle", "semantic", "non_overlapping"),
        "quantem_internal_nucleus": ("organelle", "instances", "non_overlapping"),
        "quantem_internal_ld": ("organelle", "instances", "non_overlapping"),
        TISSUE: ("area", "semantic", "non_overlapping"),
        CELLS: ("area", "instances", "non_overlapping"),
        LEGACY_ANALYSIS_MASK: ("area", "instances", "overlapping"),
    }
    for internal_name, (category, representation, topology) in builtins.items():
        manager.filter(internal_name=internal_name).update(
            category=category,
            representation=representation,
            topology_policy=topology,
        )
    manager.exclude(internal_name__in=builtins).filter(measurement_mode="global").update(
        representation="semantic"
    )


def _persistent_cell_names(AnalysisMaskObject, segmentation_id, *, using):
    rows = list(
        AnalysisMaskObject.objects.using(using)
        .filter(segmentation_id=segmentation_id)
        .order_by("sort_order", "created_at", "id")
    )
    used = set()
    for row in rows:
        match = CELL_NAME.fullmatch((row.name or "").strip())
        if match:
            used.add(int(match.group(1)))
    highest = max(used, default=0)
    for row in rows:
        name = (row.name or "").strip()
        cell_match = CELL_NAME.fullmatch(name)
        if cell_match:
            continue
        object_match = OBJECT_NAME.fullmatch(name)
        if object_match and int(object_match.group(1)) not in used:
            number = int(object_match.group(1))
        elif not name or object_match:
            highest += 1
            while highest in used:
                highest += 1
            number = highest
        else:
            # A deliberate custom name stays a deliberate custom name.
            continue
        used.add(number)
        highest = max(highest, number)
        row.name = f"Cell {number}"
        row.save(using=using, update_fields=["name"])


def migrate_definitions_and_masks(apps, schema_editor):
    alias = schema_editor.connection.alias
    SegmentationType = apps.get_model("segmentation", "SegmentationType")
    ImageSegmentation = apps.get_model("segmentation", "ImageSegmentation")
    AnalysisMaskObject = apps.get_model("segmentation", "AnalysisMaskObject")
    GlobalMask = apps.get_model("segmentation", "GlobalMask")
    type_objects = SegmentationType.objects.using(alias)
    segmentation_objects = ImageSegmentation.objects.using(alias)
    mask_objects = AnalysisMaskObject.objects.using(alias)
    global_masks = GlobalMask.objects.using(alias)

    _free_builtin_names(
        SegmentationType,
        using=alias,
        short_name="Sample Area",
        long_name="Sample Area",
        internal_name=TISSUE,
    )
    tissue, _ = type_objects.get_or_create(
        internal_name=TISSUE,
        defaults={
            "short_name": "Sample Area",
            "long_name": "Sample Area",
            "measurement_mode": "global",
            "category": "area",
            "representation": "semantic",
            "topology_policy": "non_overlapping",
        },
    )
    tissue.short_name = "Sample Area"
    tissue.long_name = "Sample Area"
    tissue.description = (
        "Select all area containing the sample. Use this to remove background, "
        "empty spaces, or other areas you wish to exclude from analysis."
    )
    tissue.category = "area"
    tissue.representation = "semantic"
    tissue.topology_policy = "non_overlapping"
    tissue.measurement_mode = "global"
    tissue.save(using=alias)

    _free_builtin_names(
        SegmentationType,
        using=alias,
        short_name="Cells",
        long_name="Cells",
        internal_name=CELLS,
    )
    cells, _ = type_objects.get_or_create(
        internal_name=CELLS,
        defaults={
            "short_name": "Cells",
            "long_name": "Cells",
            "measurement_mode": "global",
            "category": "area",
            "representation": "instances",
            "topology_policy": "non_overlapping",
        },
    )
    cells.short_name = "Cells"
    cells.long_name = "Cells"
    cells.description = (
        "Segment individual cells. Non-overlapping segmented cells can be "
        "tagged and categorized, and analyzed individually."
    )
    cells.category = "area"
    cells.representation = "instances"
    cells.topology_policy = "non_overlapping"
    cells.measurement_mode = "global"
    cells.save(using=alias)

    _classify_types(SegmentationType, using=alias)
    legacy = type_objects.filter(internal_name=LEGACY_ANALYSIS_MASK).first()
    if legacy is None:
        return

    target_cell_ids = set()
    rows = segmentation_objects.filter(
        segmentation_type=legacy, superseded_by__isnull=True
    ).order_by("asset_id", "created_at", "id")
    for source in rows:
        name = (source.display_name or "").strip().casefold()
        target_type = tissue if name in SAMPLE_ALIASES else cells
        target = (
            segmentation_objects.filter(
                asset_id=source.asset_id,
                segmentation_type=target_type,
                superseded_by__isnull=True,
            )
            .exclude(id=source.id)
            .order_by("created_at", "id")
            .first()
        )
        if target is None:
            source.segmentation_type = target_type
            source.display_name = ""
            source.save(using=alias, update_fields=["segmentation_type", "display_name"])
            target = source
        else:
            source.superseded_by = target
            source.save(using=alias, update_fields=["superseded_by"])
            last_order = (
                mask_objects.filter(segmentation=target)
                .order_by("-sort_order")
                .values_list("sort_order", flat=True)
                .first()
                or 0
            )
            moved = list(
                mask_objects.filter(segmentation=source).order_by("sort_order", "created_at", "id")
            )
            for offset, obj in enumerate(moved, start=1):
                obj.segmentation = target
                obj.sort_order = last_order + offset
                obj.save(using=alias, update_fields=["segmentation", "sort_order"])
            if moved:
                source_mask = global_masks.filter(segmentation=source).first()
                if source_mask is not None:
                    metadata = dict(source_mask.metadata or {})
                    metadata["instances_migrated_to"] = str(target.id)
                    source_mask.metadata = metadata
                    source_mask.save(using=alias, update_fields=["metadata"])
            # If vector instances already define the target, keep a raster-only
            # source on its superseded row. Runtime materialisation will add
            # its connected components without mistaking that raster for the
            # already-moved vector objects. Moving it onto the target here
            # would make it invisible as soon as any vector object existed.
            if (
                not global_masks.filter(segmentation=target).exists()
                and not mask_objects.filter(segmentation=target).exists()
            ):
                global_masks.filter(segmentation=source).update(segmentation=target)
        if target_type.id == cells.id:
            target_cell_ids.add(target.id)

    for segmentation_id in target_cell_ids:
        _persistent_cell_names(AnalysisMaskObject, segmentation_id, using=alias)


class Migration(migrations.Migration):
    # PostgreSQL defers the self-referential foreign-key index until the
    # schema editor exits.  The data conversion below updates that same table,
    # leaving pending trigger events that prevent the deferred CREATE INDEX
    # inside one transaction.  Commit schema operations before converting data.
    atomic = False
    dependencies = [("segmentation", "0009_correctionrevision")]

    operations = [
        migrations.AddField(
            model_name="segmentationtype",
            name="category",
            field=models.CharField(
                choices=[("organelle", "Organelle or structure"), ("area", "Analysis area")],
                default="organelle",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="segmentationtype",
            name="description",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddField(
            model_name="segmentationtype",
            name="representation",
            field=models.CharField(
                choices=[("instances", "Individual instances"), ("semantic", "One semantic mask")],
                default="instances",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="segmentationtype",
            name="topology_policy",
            field=models.CharField(
                choices=[
                    ("non_overlapping", "Non-overlapping instances"),
                    ("overlapping", "Overlapping instances"),
                ],
                default="non_overlapping",
                max_length=24,
            ),
        ),
        migrations.AddField(
            model_name="imagesegmentation",
            name="superseded_by",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.PROTECT,
                related_name="superseded_segmentations",
                to="segmentation.imagesegmentation",
            ),
        ),
        migrations.AlterField(
            model_name="analysismaskobject",
            name="name",
            field=models.CharField(blank=True, default="", max_length=100),
        ),
        migrations.CreateModel(
            name="AnalysisMaskObjectAnnotation",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("tags", models.JSONField(blank=True, default=list)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "object",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="annotation",
                        to="segmentation.analysismaskobject",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="SegmentObjectAnnotation",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4, editable=False, primary_key=True, serialize=False
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("display_name", models.CharField(blank=True, default="", max_length=100)),
                ("tags", models.JSONField(blank=True, default=list)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "object",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="annotation",
                        to="segmentation.segmentobject",
                    ),
                ),
            ],
        ),
        migrations.RunPython(migrate_definitions_and_masks, migrations.RunPython.noop),
    ]
