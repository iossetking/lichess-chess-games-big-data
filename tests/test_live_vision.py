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
    monkeypatch.setattr(live.cv2, "putText", lambda img, text, *args: texts.append(text))
    mask = np.array([[40, 40], [100, 40], [70, 100]])
    live.draw_detections(image, [PieceDetection("white_pawn", .923, (40, 40, 100, 120), mask)])
    assert texts == ["peon 92%"]
    assert image[80, 70].sum() == 0  # No mask fill.
    assert image[150].sum() == 0  # No grid across the image.


@pytest.mark.parametrize("scene, expected_row", [("2d", "8"), ("3d", "7")])
def test_native_detection_projects_centre_or_base(monkeypatch, scene, expected_row):
    frame = np.zeros((800, 800, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(size=800, turn="w", flip=False, stable_frames=1,
                           redetect_every=0, confidence=.55, imgsz=640,
                           scene=scene, castling="-", en_passant="-", screen_board=False)
    seen = []

    def detect(image, *args):
        seen.append(image.shape)
        worker._stop.set()
        return [PieceDetection("white_pawn", .92, (10, 10, 90, 110))]

    worker = live.LiveInference(camera, SimpleNamespace(detect=detect), None, None, args)
    worker.corners = live.board_destination(800)
    worker._run()
    assert seen == [(800, 800, 3)]
    ranks = worker.fen.split()[0].split("/")
    assert ranks[0 if expected_row == "8" else 1] == "P7"
    assert worker.board_image is not None


def test_pieces_are_displayed_without_finding_board(monkeypatch):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    camera = SimpleNamespace(sample=lambda: (frame.copy(), 1, live.perf_counter()))
    args = SimpleNamespace(size=800, turn="w", flip=False, stable_frames=1,
                           redetect_every=0, confidence=.55, imgsz=640,
                           scene="2d", castling="-", en_passant="-", screen_board=False)
    monkeypatch.setattr(live, "detect_board_corners", lambda frame: None)

    def detect(image, *args):
        worker._stop.set()
        return [PieceDetection("pawn", .9, (10, 30, 60, 90))]

    worker = live.LiveInference(camera, SimpleNamespace(detect=detect), None, None, args)
    worker._run()
    assert worker.corners is None
    assert worker.board_image.sum() > 0
    assert worker.latency_ms > 0
