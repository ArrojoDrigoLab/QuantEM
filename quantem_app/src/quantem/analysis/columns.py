"""Units on spreadsheet headers; internal metric keys stay stable."""

import re


def column_title(name: str) -> str:
    if "(" in name:
        return name
    if name == "pixel_size_nm":
        return "resolution (nm/px)"
    if name == "values_in_pixel_size_nm":
        return "measurement resolution (nm/px)"
    if name == "objects_per_um2":
        return "object density (objects/µm²)"
    for token, unit in (("um2", "µm²"), ("um", "µm"), ("nm", "nm"), ("px", "px"), ("pixels", "px")):
        if re.search(rf"_{token}(?:_|$)", name):
            if token in {"px", "pixels"} and any(
                key in name for key in ("area", "tissue", "foreground", "denominator")
            ):
                unit = "px²"
            title = re.sub(rf"_{token}(?=_|$)", "", name).replace("_", " ")
            return f"{title} ({unit})"
    return name
