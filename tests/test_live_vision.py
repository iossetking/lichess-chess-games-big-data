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
