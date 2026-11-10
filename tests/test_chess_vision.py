import numpy as np
import pytest

from lichess_pipeline.chess_vision import (
    FenStabilizer,
    PieceDetection,
    board_destination,
    detections_to_fen,
    extrapolate_board_corners,
    marker_centers_by_corner,
    piece_symbol_from_label,
    square_regions,
    validate_ordered_board_corners,
)


def test_board_destination_uses_full_image_extent():
    np.testing.assert_array_equal(board_destination(800), [[0, 0], [799, 0], [799, 799], [0, 799]])


def test_extrapolate_board_corners_expands_a_regular_inner_grid():
    inner = np.array([[x, y] for y in range(10, 80, 10) for x in range(10, 80, 10)])
    np.testing.assert_array_equal(extrapolate_board_corners(inner), [[5, 5], [75, 5], [75, 75], [5, 75]])


def test_piece_labels_accept_common_yolo_conventions():
    assert piece_symbol_from_label("w-pawn") == "P"
    assert piece_symbol_from_label("white_knight") == "N"
    assert piece_symbol_from_label("black king") == "k"
    assert piece_symbol_from_label("bishop") is None


def test_detections_to_fen_maps_centres_and_keeps_highest_confidence_per_square():
    detections = [
        PieceDetection("black_king", 0.9, (401, 1, 498, 98)),  # e8
        PieceDetection("white_pawn", 0.7, (1, 601, 98, 698)),  # a2
        PieceDetection("white_queen", 0.6, (101, 601, 198, 698)),  # b2
        PieceDetection("black_queen", 0.8, (102, 602, 199, 699)),  # replaces b2
        PieceDetection("white_rook", 0.99, (-100, 0, -20, 50)),  # outside board
    ]
    assert detections_to_fen(detections, turn="b", castling="Kq") == "4k3/8/8/8/8/8/Pq6/8 b Kq - 0 1"


def test_fen_stabilizer_publishes_once_per_stable_position():
    stabilizer = FenStabilizer(required_frames=3)
    assert stabilizer.update("first") is None
    assert stabilizer.update("other") is None
    assert stabilizer.update("first") is None
    assert stabilizer.update("first") is None
    assert stabilizer.update("first") == "first"
    assert stabilizer.update("first") is None


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


def test_validate_ordered_board_corners_accepts_a_convex_board():
    corners = validate_ordered_board_corners(np.array([[10, 10], [90, 20], [80, 90], [15, 80]]))
    np.testing.assert_array_equal(corners, [[10, 10], [90, 20], [80, 90], [15, 80]])


def test_validate_ordered_board_corners_rejects_crossed_click_order():
    with pytest.raises(ValueError, match="convex"):
        validate_ordered_board_corners(np.array([[10, 10], [80, 90], [90, 20], [15, 80]]))
