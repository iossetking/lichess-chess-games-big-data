import numpy as np
import pytest

from lichess_pipeline.chess_vision import (
    board_destination,
    marker_centers_by_corner,
    square_regions,
)


def test_board_destination_uses_full_image_extent():
    np.testing.assert_array_equal(board_destination(800), [[0, 0], [799, 0], [799, 799], [0, 799]])


def test_square_regions_cover_all_64_squares():
    regions = square_regions(800)
    assert len(regions) == 64
    assert regions["a8"] == (0, 0, 100, 100)
    assert regions["h1"] == (700, 700, 800, 800)


def test_square_regions_rejects_non_divisible_size():
    with pytest.raises(ValueError, match="divisible"):
        square_regions(801)


def test_marker_centers_follow_explicit_marker_order():
    markers = {
        10: np.array([[[0, 0], [2, 0], [2, 2], [0, 2]]]),
        11: np.array([[[8, 0], [10, 0], [10, 2], [8, 2]]]),
        12: np.array([[[8, 8], [10, 8], [10, 10], [8, 10]]]),
        13: np.array([[[0, 8], [2, 8], [2, 10], [0, 10]]]),
    }
    np.testing.assert_array_equal(
        marker_centers_by_corner(markers, (10, 11, 12, 13)),
        [[1, 1], [9, 1], [9, 9], [1, 9]],
    )
    assert marker_centers_by_corner({10: markers[10]}, (10, 11, 12, 13)) is None
