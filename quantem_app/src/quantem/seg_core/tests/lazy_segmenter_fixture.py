import numpy as np

from quantem.seg_core.base_segmenter import BaseSegmenter


class LazyTestSegmenter(BaseSegmenter):
    """Minimal BaseSegmenter implementation used by the registry laziness test."""

    @property
    def name(self) -> str:
        return "lazy"

    @property
    def generated_flag(self) -> str:
        return "lazy_generated"

    @property
    def prob_map_prefix(self) -> str:
        return "lazy"

    def load_models(self) -> None:
        return None

    def run_dl_inference(
        self,
        image: np.ndarray,
        cached_prob_maps: dict[str, np.ndarray | None],
        on_progress=None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        _ = (image, cached_prob_maps, on_progress, kwargs)
        return {}

    def combine_prob_maps(self, prob_maps: dict[str, np.ndarray]) -> np.ndarray:
        _ = prob_maps
        return np.zeros((1, 1), dtype=np.float32)

    def extract_instances(
        self,
        prob: np.ndarray,
        image: np.ndarray,
        prob_maps: dict[str, np.ndarray],
        *,
        min_area: int,
        coordinate_offset: tuple[float, float] | None,
        on_progress=None,
    ) -> list:
        _ = (
            prob,
            image,
            prob_maps,
            min_area,
            coordinate_offset,
            on_progress,
        )
        return []

    def get_dl_model_names(self) -> list[str]:
        return []
