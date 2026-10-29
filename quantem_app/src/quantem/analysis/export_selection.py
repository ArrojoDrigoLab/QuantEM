"""Optional files in a saved analysis; measurements and identity stay intact."""

import json

from rest_framework.exceptions import ValidationError

FILES = (
    "objects.csv",
    "image_summary.csv",
    "composition.csv",
    "area.csv",
    "region_summary.csv",
    "cell_summary.csv",
    "organelle_objects.csv",
)


def output_files(raw):
    if raw is None:
        return dict.fromkeys(FILES, True)
    if (
        not isinstance(raw, dict)
        or set(raw) - set(FILES)
        or any(type(value) is not bool for value in raw.values())
    ):
        raise ValidationError("Output files must be named checkboxes with true or false values.")
    return {name: raw.get(name, True) for name in FILES}


def select_export_files(run):
    selection = output_files(run.params.get("output_files"))
    if run.params.get("outputs", {}).get("individual_objects") is False:
        selection["objects.csv"] = selection["organelle_objects.csv"] = False
    directory = run.export_path
    if directory is None:
        return
    for name, enabled in selection.items():
        if not enabled:
            (directory / name).unlink(missing_ok=True)
    path = directory / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["output_selection"] = selection
        if isinstance(manifest.get("outputs"), dict):
            manifest["outputs"] = {
                name: value
                for name, value in manifest["outputs"].items()
                if selection.get(name, True)
            }
        files = (
            manifest.get("outputs", {}).get("files")
            if isinstance(manifest.get("outputs"), dict)
            else None
        )
        if isinstance(files, list):
            manifest["outputs"]["files"] = [
                entry for entry in files if selection.get(entry.get("filename"), True)
            ]
        path.write_text(
            json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )
