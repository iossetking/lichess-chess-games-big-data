import numpy as np
import pytest

from lichess_pipeline.chess_vision import (
    FenStabilizer,
    PieceDetection,
    PieceLabelSmoother,
    box_iou,
    board_destination,
    crop_piece,
    detect_any_screen_board,
    detect_green_screen_board,
    detect_screen_board_by_grid,
    detections_to_fen,
    extrapolate_board_corners,
    fuse_piece_probabilities,
    marker_centers_by_corner,
    piece_description_from_label,
    piece_symbol_from_label,
    square_regions,
    validate_fen_position,
    validate_ordered_board_corners,
)


def test_board_destination_uses_full_image_extent():
    np.testing.assert_array_equal(board_destination(800), [[0, 0], [799, 0], [799, 799], [0, 799]])


def test_extrapolate_board_corners_expands_a_regular_inner_grid():
    inner = np.array([[x, y] for y in range(10, 80, 10) for x in range(10, 80, 10)])
    np.testing.assert_array_equal(extrapolate_board_corners(inner), [[5, 5], [75, 5], [75, 75], [5, 75]])


def test_green_screen_board_finds_grid_without_pieces():
    image = np.empty((400, 400, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            image[row * 50:(row + 1) * 50, column * 50:(column + 1) * 50] = (
                (208, 236, 235) if (row + column) % 2 == 0 else (86, 149, 119)
            )
    np.testing.assert_allclose(detect_green_screen_board(image),
                               [[0, 0], [399, 0], [399, 399], [0, 399]], atol=2)
    assert detect_green_screen_board(np.zeros_like(image)) is None


def test_piece_labels_accept_common_yolo_conventions():
    assert piece_symbol_from_label("w-pawn") == "P"
    assert piece_symbol_from_label("white_knight") == "N"
    assert piece_symbol_from_label("black king") == "k"
    assert piece_symbol_from_label("bishop") is None
    assert piece_description_from_label("w-pawn") == "peón blanco"
    assert piece_description_from_label("black king") == "rey negro"


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


def _piece(label, confidence, box=(10, 10, 50, 90)):
    return PieceDetection(label, confidence, box)


def test_box_iou_basics():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_label_smoother_ignores_a_single_misclassified_frame():
    smoother = PieceLabelSmoother()
    labels = [smoother.update([_piece(label, 0.8)])[0].label
              for label in ("knight", "knight", "knight", "pawn", "knight")]
    assert labels == ["knight"] * 5


def test_label_smoother_follows_a_real_change_within_a_few_frames():
    smoother = PieceLabelSmoother()
    for _ in range(3):
        smoother.update([_piece("knight", 0.8)])
    labels = [smoother.update([_piece("rook", 0.9)])[0].label for _ in range(4)]
    assert labels[0] == "knight" and labels[-1] == "rook"


def test_label_smoother_tracks_pieces_independently_and_forgets_old_tracks():
    smoother = PieceLabelSmoother(max_missing=2)
    left, right = (0, 0, 40, 80), (200, 0, 240, 80)
    for _ in range(3):
        smoother.update([_piece("king", 0.9, left), _piece("pawn", 0.9, right)])
    out = smoother.update([_piece("queen", 0.4, left), _piece("pawn", 0.9, right)])
    assert [d.label for d in out] == ["king", "pawn"]
    for _ in range(4):
        smoother.update([])
    # The old king track expired, so a new piece starts fresh.
    assert smoother.update([_piece("queen", 0.4, left)])[0].label == "queen"


def test_label_smoother_tolerates_small_camera_jitter_on_slender_piece():
    smoother = PieceLabelSmoother(min_iou=0.4)
    # Narrow bishop box shifts horizontally by 12px (IoU ~0.25 < 0.4, but same piece).
    box_a = (100, 100, 120, 200)
    box_b = (112, 100, 132, 200)
    for _ in range(3):
        smoother.update([_piece("bishop", 0.9, box_a)])
    out = smoother.update([_piece("pawn", 0.5, box_b)])
    assert out[0].label == "bishop"


def test_fuse_piece_probabilities_combines_detector_and_classifier():
    # Detector says queen (0.55), classifier strongly says king (0.90).
    probs = [0.02, 0.90, 0.01, 0.01, 0.05, 0.01]  # bishop, king, knight, pawn, queen, rook
    label, conf = fuse_piece_probabilities("queen", 0.55, probs, classifier_weight=1.25)
    assert label == "king"
    assert 0.5 < conf <= 1.0


def test_crop_piece_returns_requested_square_size_even_near_borders():
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    crop = crop_piece(image, (-5, 0, 25, 40), size=160)
    assert crop.shape == (160, 160, 3)


def test_detect_screen_board_by_grid_finds_lichess_brown_board_in_browser_frame():
    frame = np.full((720, 1280, 3), (45, 42, 40), dtype=np.uint8)
    x0, y0, side = 360, 120, 60
    for r in range(8):
        for c in range(8):
            colour = (181, 217, 240) if (r + c) % 2 == 0 else (99, 136, 181)  # Lichess brown (BGR)
            frame[y0 + r * side:y0 + (r + 1) * side, x0 + c * side:x0 + (c + 1) * side] = colour
            # Simulate pieces in the centre of several squares so corners must still work.
            if r in (0, 1, 3, 4, 6, 7) and c % 2 == 0:
                cy, cx = y0 + r * side + side // 2, x0 + c * side + side // 2
                frame[cy - 14:cy + 14, cx - 14:cx + 14] = (30, 30, 30)
    corners = detect_screen_board_by_grid(frame)
    assert corners is not None
    expected = [[x0, y0], [x0 + 8 * side - 1, y0], [x0 + 8 * side - 1, y0 + 8 * side - 1], [x0, y0 + 8 * side - 1]]
    np.testing.assert_allclose(corners, expected, atol=8)
    assert detect_any_screen_board(np.zeros_like(frame)) is None


def test_validate_fen_position_reports_clear_spanish_diagnostics():
    ok, reason = validate_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert ok and reason == "OK"

    ok, reason = validate_fen_position("8/8/8/8/8/8/8/4K3 w - - 0 1")
    assert not ok and "rey negro" in reason

    # Black king in check while it is White's turn -> wrong turn hint
    ok, reason = validate_fen_position("4k3/4Q3/8/8/8/8/8/4K3 w - - 0 1")
    assert not ok and "[t]" in reason
