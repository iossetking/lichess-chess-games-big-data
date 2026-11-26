"""Headless checks for live rendering and camera-frame scheduling."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lichess_pipeline.chess_vision import PieceDetection

spec = importlib.util.spec_from_file_location(
    "live_chess_fen", Path(__file__).resolve().parents[1] / "scripts/live_chess_fen.py"
)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def test_boxes_show_shape_percentage_without_grid_or_mask(monkeypatch):
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    texts = []
    monkeypatch.setattr(
        live.cv2, "putText", lambda img, text, *args: texts.append(text)
    )
    mask = np.array([[40, 40], [100, 40], [70, 100]])
    live.draw_detections(
        image, [PieceDetection("white_pawn", 0.923, (40, 40, 100, 120), mask)]
    )
    assert texts == ["peon 92%"]
    assert image[80, 70].sum() == 0  # No mask fill.
    assert image[150].sum() == 0  # No grid across the image.


@pytest.mark.parametrize("scene, expected_row", [("2d", "8"), ("3d", "7")])
def test_native_detection_projects_centre_or_base(monkeypatch, scene, expected_row):
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(
        size=800,
        turn="w",
        flip=False,
        stable_frames=1,
        redetect_every=0,
        confidence=0.55,
        imgsz=640,
        scene=scene,
        castling="-",
        en_passant="-",
        screen_board=False,
        smooth=0.8,
    )
    seen = []

    def detect(image, *args):
        seen.append(image.shape)
        worker._stop.set()
        return [PieceDetection("white_pawn", 0.92, (10, 10, 90, 110))]

    worker = live.LiveInference(
        camera, SimpleNamespace(detect=detect), None, None, args
    )
    worker.corners = live.board_destination(800)
    worker._run()
    assert seen == [(800, 800, 3)]
    ranks = worker.fen.split()[0].split("/")
    assert ranks[0 if expected_row == "8" else 1] == "P7"
    assert worker.board_image is not None


def test_pieces_are_displayed_without_finding_board(monkeypatch):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(
        size=800,
        turn="w",
        flip=False,
        stable_frames=1,
        redetect_every=0,
        confidence=0.55,
        imgsz=640,
        scene="2d",
        castling="-",
        en_passant="-",
        screen_board=False,
        smooth=0.8,
    )
    monkeypatch.setattr(live, "detect_board_corners", lambda frame: None)

    def detect(image, *args):
        worker._stop.set()
        return [PieceDetection("pawn", 0.9, (10, 30, 60, 90))]

    worker = live.LiveInference(
        camera, SimpleNamespace(detect=detect), None, None, args
    )
    worker._run()
    assert worker.corners is None
    assert worker.board_image.sum() > 0
    assert worker.latency_ms > 0


class FakePolicy:
    """Minimal predictor returning the first legal move of whatever FEN it gets."""

    def __init__(self, probability: float = 0.4):
        self.probability = probability

    def predict_top_moves(self, fen: str, top_k: int = 3):
        import chess

        moves = [move.uci() for move in list(chess.Board(fen).legal_moves)[:top_k]]
        return [(move, self.probability) for move in moves]


def test_live_inference_runs_every_policy_and_reports_results(monkeypatch):
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(
        size=800,
        turn="w",
        flip=False,
        stable_frames=1,
        redetect_every=0,
        confidence=0.55,
        imgsz=640,
        scene="2d",
        castling="-",
        en_passant="-",
        screen_board=False,
        smooth=0.8,
    )
    policies = [
        live.PolicyStyle("Scratch", (255, 255, 0), FakePolicy(0.40)),
        live.PolicyStyle("Maia-SF", (235, 80, 235), FakePolicy(0.55)),
    ]

    def detect(image, *args):
        worker._stop.set()
        return [
            PieceDetection("white_king", 0.99, (300, 700, 380, 790)),
            PieceDetection("white_pawn", 0.97, (290, 590, 370, 690)),
            PieceDetection("black_king", 0.99, (300, 20, 380, 110)),
        ]

    worker = live.LiveInference(
        camera, SimpleNamespace(detect=detect), None, policies, args
    )
    worker.corners = live.board_destination(800)
    worker._run()

    labels = {result["label"] for result in worker.results}
    assert labels == {"Scratch", "Maia-SF"}
    assert all(result["san"] for result in worker.results)
    assert "Scratch" in worker.move and "Maia-SF" in worker.move


def test_arrow_polyline_uses_an_l_path_only_for_knights():
    straight = live._arrow_polyline("e2", "e4", 800, False)
    knight = live._arrow_polyline("g8", "f6", 800, False)

    assert len(straight) == 2
    assert len(knight) == 3
    assert knight[1] == live.square_center("f8", 800, False)


def test_square_center_flips_with_the_board_orientation():
    upright = live.square_center("a8", 800, False)
    flipped = live.square_center("a8", 800, True)

    assert upright == (50.0, 50.0)
    assert flipped == (799 - 50.0, 799 - 50.0)


def test_draw_ar_overlay_projects_arrows_badges_and_hud():
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    corners = np.array([[100, 50], [700, 50], [700, 550], [100, 550]], dtype=np.float32)
    results = [
        {
            "label": "Scratch",
            "colour": (255, 255, 0),
            "uci": "e2e4",
            "san": "e4",
            "probability": 0.40,
        },
        {
            "label": "Maia-SF",
            "colour": (235, 80, 235),
            "uci": "e2e4",
            "san": "e4",
            "probability": 0.55,
        },
    ]

    live.draw_ar_overlay(
        frame,
        corners,
        800,
        results,
        "w",
        False,
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )

    assert frame.sum() > 0


def test_parse_camera_source_handles_indices_and_stream_urls():
    assert live.parse_camera_source("0") == 0
    assert live.parse_camera_source(" 1 ") == 1
    assert live.parse_camera_source(2) == 2
    assert live.parse_camera_source("http://192.168.1.25:8080/video") == "http://192.168.1.25:8080/video"
    assert live.parse_camera_source("rtsp://phone/live") == "rtsp://phone/live"


def test_parse_region_validates_four_positive_dimensions():
    assert live.parse_region("10, 20, 800, 800") == {
        "left": 10,
        "top": 20,
        "width": 800,
        "height": 800,
    }
    with pytest.raises(ValueError, match="four"):
        live.parse_region("10,20,800")
    with pytest.raises(ValueError, match="positive"):
        live.parse_region("10,20,0,800")


def test_screen_board_warps_board_before_yolo_and_maps_2d():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(
        size=800,
        turn="w",
        flip=False,
        stable_frames=1,
        redetect_every=0,
        confidence=0.25,
        imgsz=640,
        scene="auto",
        castling="-",
        en_passant="-",
        screen_board=True,
        screen_capture=False,
        region_is_board=False,
        hold_missing=3,
        smooth=0.8,
    )
    seen = []

    def detect(image, *args):
        seen.append(image.shape)
        worker._stop.set()
        return [PieceDetection("white_pawn", 0.92, (10, 10, 90, 90))]

    worker = live.LiveInference(
        camera, SimpleNamespace(detect=detect), None, None, args
    )
    worker.corners = np.array([[200, 100], [600, 100], [600, 500], [200, 500]], dtype=np.float32)
    worker._run()
    assert seen == [(800, 800, 3)]
    assert worker.fen.split()[0].split("/")[0] == "P7"


def test_screen_capture_region_locks_non_green_board_to_frame(monkeypatch):
    frame = np.full((600, 600, 3), 100, dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(
        size=800,
        turn="w",
        flip=False,
        stable_frames=1,
        redetect_every=0,
        confidence=0.25,
        imgsz=640,
        scene="auto",
        castling="-",
        en_passant="-",
        screen_board=True,
        screen_capture=True,
        region_is_board=True,
        hold_missing=0,
        smooth=0.0,
    )
    monkeypatch.setattr(live, "detect_green_screen_board", lambda img: None)
    seen = []

    def detect(image, *args):
        seen.append(image.shape)
        worker._stop.set()
        return [PieceDetection("black_king", 0.95, (410, 10, 490, 90))]

    worker = live.LiveInference(
        camera, SimpleNamespace(detect=detect), None, None, args
    )
    worker._run()
    assert worker.corners is not None
    np.testing.assert_array_equal(worker.corners, [[0, 0], [599, 0], [599, 599], [0, 599]])
    assert seen == [(800, 800, 3)]
    assert worker.fen.split()[0].split("/")[0] == "4k3"


def test_macos_screen_capture_access_returns_none_off_darwin(monkeypatch):
    monkeypatch.setattr(live.sys, "platform", "linux")
    assert live._macos_screen_capture_access() is None


def test_live_inference_cleans_impossible_castling_and_repairs_duplicate_king():
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(
        size=800,
        turn="w",
        flip=False,
        stable_frames=1,
        redetect_every=0,
        confidence=0.25,
        imgsz=640,
        scene="2d",
        castling="KQkq",
        en_passant="-",
        screen_board=True,
        screen_capture=True,
        region_is_board=True,
        hold_missing=0,
        smooth=0.0,
    )
    policies = [live.PolicyStyle("Maia-SF", (235, 80, 235), FakePolicy(0.65))]

    def detect(image, *args):
        worker._stop.set()
        # Kings moved from e1/e8 (e.g. castled to g1/g8) and white queen mislabelled as a second white king.
        return [
            PieceDetection("white-king", 0.95, (610, 710, 690, 790)),   # g1
            PieceDetection("white-king", 0.60, (310, 710, 390, 790)),   # d1 -> demoted to Q
            PieceDetection("black-king", 0.96, (610, 10, 690, 90)),     # g8
        ]

    worker = live.LiveInference(
        camera, SimpleNamespace(detect=detect), None, policies, args
    )
    worker.corners = live.board_destination(800)
    worker._run()

    assert worker.fen.split()[2] == "-"  # impossible KQkq stripped
    assert "Q" in worker.fen.split()[0]  # extra king repaired to queen
    assert len(worker.results) == 1
    assert worker.results[0]["label"] == "Maia-SF"
