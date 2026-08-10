"""Per-object morphometrics and per-image summaries.

Lives in the core, not the plugin, so the napari plugin and the desktop application report
identical numbers for the same mask.

Units: physical (nm, nm^2) when a pixel size is supplied, pixels otherwise. Column names say which,
always — silently mixing the two is how measurement tables become uninterpretable.
"""

from __future__ import annotations

import numpy as np

#: regionprops properties requested for every object.
_PROPS = (
    "label",
    "area",
    "equivalent_diameter_area",
    "perimeter",
    "major_axis_length",
    "minor_axis_length",
    "eccentricity",
    "solidity",
    "orientation",
    "centroid",
    "bbox",
)
_INTENSITY_PROPS = ("intensity_mean", "intensity_min", "intensity_max", "intensity_std")


def _std_intensity(region, intensities):  # extra property for regionprops_table
    """Std over the OBJECT's pixels, not over its bounding box.

    skimage hands a two-argument extra property ``(regionmask, intensity_image_of_bbox)``, and the
    background inside that bbox is zero-filled. Taking np.std of the whole box therefore measures
    the object's shape as much as its intensity: a perfectly uniform circle of value 200 reports a
    std of 72.4 instead of 0. Worst for elongated and curved objects -- mitochondria and ER.
    """
    vals = np.asarray(intensities)[np.asarray(region, dtype=bool)]
    return float(np.std(vals)) if vals.size else 0.0


def measure_objects(
    labels: np.ndarray,
    intensity_image: np.ndarray | None = None,
    *,
    pixel_size_nm: float | tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Per-object table as a dict of columns (directly usable as a napari ``layer.features``).

    Length columns are in nm and area in nm^2 when ``pixel_size_nm`` is given; otherwise px / px^2.
    """
    from skimage.measure import regionprops_table

    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("measure_objects expects a 2-D label image")

    props = list(_PROPS)
    extra = {}
    img = None
    if intensity_image is not None:
        img = np.asarray(intensity_image)
        if img.shape != labels.shape:
            raise ValueError("intensity image and labels must have the same shape")
        props += ["intensity_mean", "intensity_min", "intensity_max"]
        extra["intensity_std"] = _std_intensity

    if labels.max() == 0:
        return {"label": np.zeros(0, dtype=np.int32)}

    table = regionprops_table(
        labels,
        intensity_image=img,
        properties=props,
        extra_properties=tuple(extra.values()) if extra else None,
    )

    scale_r, scale_c = _scales(pixel_size_nm)
    unit = "nm" if pixel_size_nm is not None else "px"
    iso = float(np.sqrt(scale_r * scale_c))

    out: dict[str, np.ndarray] = {"label": np.asarray(table["label"], dtype=np.int32)}
    out[f"area_{unit}2"] = np.asarray(table["area"], dtype=np.float64) * scale_r * scale_c
    for src, dst in (
        ("equivalent_diameter_area", "equivalent_diameter"),
        ("perimeter", "perimeter"),
        ("major_axis_length", "major_axis"),
        ("minor_axis_length", "minor_axis"),
    ):
        if src in table:
            out[f"{dst}_{unit}"] = np.asarray(table[src], dtype=np.float64) * iso
    for k in ("eccentricity", "solidity", "orientation"):
        if k in table:
            out[k] = np.asarray(table[k], dtype=np.float64)
    if "centroid-0" in table:
        out[f"centroid_row_{unit}"] = np.asarray(table["centroid-0"], dtype=np.float64) * scale_r
        out[f"centroid_col_{unit}"] = np.asarray(table["centroid-1"], dtype=np.float64) * scale_c
    for i, name in enumerate(("bbox_min_row", "bbox_min_col", "bbox_max_row", "bbox_max_col")):
        key = f"bbox-{i}"
        if key in table:
            out[f"{name}_px"] = np.asarray(table[key], dtype=np.int64)
    for k in _INTENSITY_PROPS:
        if k in table:
            out[k] = np.asarray(table[k], dtype=np.float64)
        elif k == "intensity_std" and "_std_intensity" in table:
            out[k] = np.asarray(table["_std_intensity"], dtype=np.float64)
    return out


def summarize(
    labels: np.ndarray,
    *,
    pixel_size_nm: float | tuple[float, float] | None = None,
    tissue_mask: np.ndarray | None = None,
) -> dict:
    """Per-image summary: counts, total area, and the area fraction.

    ``tissue_mask`` restricts the denominator of the area fraction to real tissue — the quantity
    Fig. 4C reports. Without it the denominator is the whole image.
    """
    labels = np.asarray(labels)
    scale_r, scale_c = _scales(pixel_size_nm)
    unit = "nm" if pixel_size_nm is not None else "px"
    px_area = scale_r * scale_c

    fg = labels > 0
    obj_px = int(fg.sum())
    denom_px = (
        int(np.asarray(tissue_mask, dtype=bool).sum()) if tissue_mask is not None else fg.size
    )

    sizes = np.bincount(labels.ravel())[1:] if fg.any() else np.zeros(0)
    sizes = sizes[sizes > 0]
    # Count distinct ids, not the largest id. Anything that leaves gaps in the numbering -- a user
    # deleting an object in napari, a proofreading filter -- makes labels.max() an overcount:
    # ids {1, 5} is two objects, not five.
    n = int(sizes.size)

    return {
        "n_objects": n,
        f"total_object_area_{unit}2": obj_px * px_area,
        f"image_area_{unit}2": denom_px * px_area,
        "area_fraction": (obj_px / denom_px) if denom_px else 0.0,
        "area_fraction_denominator": "tissue_mask" if tissue_mask is not None else "whole_image",
        f"mean_object_area_{unit}2": float(sizes.mean() * px_area) if sizes.size else 0.0,
        f"median_object_area_{unit}2": float(np.median(sizes) * px_area) if sizes.size else 0.0,
        "units": unit,
    }


def _scales(pixel_size_nm) -> tuple[float, float]:
    if pixel_size_nm is None:
        return 1.0, 1.0
    if np.isscalar(pixel_size_nm):
        return float(pixel_size_nm), float(pixel_size_nm)
    r, c = pixel_size_nm
    return float(r), float(c)


def to_csv(columns: dict[str, np.ndarray], path) -> None:
    """Write a column dict to CSV without requiring pandas."""
    import csv

    keys = list(columns)
    n = len(next(iter(columns.values()))) if keys else 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for i in range(n):
            w.writerow([columns[k][i] for k in keys])
