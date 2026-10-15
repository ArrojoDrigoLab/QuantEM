"""Use image names until the user assigns a shared sample ID."""


def sample_identity(
    image_name: str | None,
    asset_id: object,
    sample_id: str | None = "",
    *,
    explicit: bool | None = None,
) -> dict[str, str | bool]:
    sample = (sample_id or "").strip()
    is_explicit = bool(sample) if explicit is None else bool(explicit and sample)
    return {
        "sample_id": sample if is_explicit else (image_name or "").strip() or f"Image {asset_id}",
        "sample_is_explicit": is_explicit,
    }
