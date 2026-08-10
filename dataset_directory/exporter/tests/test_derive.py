"""The derivation rules decide published numbers, so they are pinned here."""
from __future__ import annotations

import pytest

from quantem_directory import derive


class TestDimensionality:
    def test_explicit_tag_wins(self):
        assert derive.is_three_dimensional(
            dimensionality_tags=["3D"], resolution_field="", depth=None
        )

    def test_three_nanometre_axes_imply_a_volume(self):
        # Three axis extents rescue assets whose dimensionality tag says
        # "Mixed" or is missing entirely.
        assert derive.is_three_dimensional(
            dimensionality_tags=["Mixed"], resolution_field="4nm x 4nm x 8nm", depth=None
        )

    @pytest.mark.parametrize(
        "resolution_field",
        [
            "27x27x80nm",  # three axes, only the last suffixed -> reads as one
            "148.183 x 148.183 nm/pixel; 30 nm z-step",  # reads as two
        ],
    )
    def test_only_nm_suffixed_components_count(self, resolution_field):
        # The corpus writes resolutions a dozen different ways, and the rule
        # that produced the published counts only sees numbers immediately
        # followed by "nm". Volumes written this way are caught by their depth
        # instead. Pinned so a well-meaning regex "fix" cannot silently move
        # assets between the 2D and 3D totals.
        assert not derive.is_three_dimensional(
            dimensionality_tags=[], resolution_field=resolution_field, depth=None
        )
        assert derive.is_three_dimensional(
            dimensionality_tags=[], resolution_field=resolution_field, depth=200
        )

    def test_two_axes_do_not(self):
        assert not derive.is_three_dimensional(
            dimensionality_tags=[], resolution_field="1.59nm x 1.59nm", depth=1
        )

    def test_depth_greater_than_one_implies_a_volume(self):
        assert derive.is_three_dimensional(
            dimensionality_tags=[], resolution_field="", depth=310
        )

    def test_single_plane_is_not_a_volume(self):
        assert not derive.is_three_dimensional(
            dimensionality_tags=["2D"], resolution_field="", depth=1
        )

    def test_tag_case_and_padding_are_tolerated(self):
        assert derive.is_three_dimensional(
            dimensionality_tags=[" 3d "], resolution_field="", depth=None
        )


class TestReferenceUrl:
    def test_dataset_doi_wins(self):
        assert (
            derive.reference_url(
                dataset_doi="10.5281/zenodo.1", source_url="https://example.org", experiment_doi="10.1/x"
            )
            == "https://doi.org/10.5281/zenodo.1"
        )

    def test_source_url_is_second(self):
        assert (
            derive.reference_url(source_url="https://webknossos.org/x", experiment_doi="10.1/x")
            == "https://webknossos.org/x"
        )

    def test_experiment_doi_is_last(self):
        assert derive.reference_url(experiment_doi="10.1/x") == "https://doi.org/10.1/x"

    def test_nothing_is_none_not_empty_string(self):
        # The site distinguishes "not deposited" from "empty link", so this must
        # not collapse to a falsy string that renders as a dead anchor.
        assert derive.reference_url() is None
        assert derive.reference_url(dataset_doi="  ", source_url="") is None


class TestRepository:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://doi.org/10.6019/EMPIAR-12585", "EMPIAR"),
            ("https://doi.org/10.6019/S-BIAD3712", "BioImage Archive"),
            ("https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD370", "BioImage Archive"),
            ("https://doi.org/10.5281/zenodo.11196419", "Zenodo"),
            ("https://doi.org/10.25378/janelia.22670176", "OpenOrganelle"),
            ("https://openorganelle.janelia.org/datasets/aic_desmosome-1", "OpenOrganelle"),
            ("https://doi.org/10.60533/boss-2025-gcyr", "BossDB"),
            ("https://webknossos.org/datasets/x", "WEBKNOSSOS"),
            ("https://doi.org/10.5061/dryad.bh78sn5", "Dryad"),
            ("https://doi.org/10.6084/m9.figshare.30043093.v1", "figshare"),
            ("https://idr.openmicroscopy.org/webclient/?show=project-1", "IDR"),
        ],
    )
    def test_known_repositories(self, url, expected):
        assert derive.repository_of(url) == expected

    def test_one_doi_prefix_serves_two_repositories(self):
        # EBI issues 10.6019 for both EMPIAR and the BioImage Archive, so the
        # accession shape has to be inspected rather than the prefix trusted.
        assert derive.repository_of("https://doi.org/10.6019/EMPIAR-1") != derive.repository_of(
            "https://doi.org/10.6019/S-BIAD1"
        )

    def test_missing_link_is_not_deposited(self):
        assert derive.repository_of(None) == derive.NOT_DEPOSITED
        assert derive.repository_of("   ") == derive.NOT_DEPOSITED

    def test_unrecognised_host_is_not_invented(self):
        assert derive.repository_of("https://example.org/data/1") == derive.OTHER_REPOSITORY


class TestResolutionBands:
    @pytest.mark.parametrize(
        "nm,expected",
        [
            (0.1112, "< 1 nm/px"),
            (0.999, "< 1 nm/px"),
            (1.0, "1 – 4 nm/px"),
            (3.48, "1 – 4 nm/px"),
            (4.0, "4 – 8 nm/px"),
            (8.0, "8 – 32 nm/px"),
            (31.9, "8 – 32 nm/px"),
            (32.0, "≥ 32 nm/px"),
            (316.5, "≥ 32 nm/px"),
        ],
    )
    def test_boundaries_are_half_open(self, nm, expected):
        assert derive.resolution_band(nm) == expected

    def test_unknown_is_a_band_not_a_hole(self):
        # A third of the corpus has no parsable resolution. If unknown were not
        # selectable, choosing any band would silently hide those assets.
        assert derive.resolution_band(None) == derive.RESOLUTION_UNKNOWN
        assert derive.RESOLUTION_UNKNOWN in derive.resolution_band_labels()

    def test_unknown_sorts_last(self):
        assert derive.resolution_band_labels()[-1] == derive.RESOLUTION_UNKNOWN


class TestFormatting:
    def test_two_dimensional(self):
        assert derive.format_dimensions(2048, 2119, None) == "2048×2119"
        assert derive.format_dimensions(2048, 2119, 1) == "2048×2119"

    def test_three_dimensional(self):
        assert derive.format_dimensions(2048, 2119, 310) == "2048×2119×310"

    def test_incomplete(self):
        assert derive.format_dimensions(None, 2119, None) is None

    def test_numeric_parsing_tolerates_extract_quirks(self):
        assert derive.parse_int("310") == 310
        assert derive.parse_int("310.0") == 310
        assert derive.parse_int("") is None
        assert derive.parse_int(None) is None
        assert derive.parse_int("not a number") is None
        assert derive.parse_float("1.59") == pytest.approx(1.59)
        assert derive.parse_float("") is None
