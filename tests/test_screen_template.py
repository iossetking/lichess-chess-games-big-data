"""Verify the generated 2D screen template works with the YOLO-seg trainer and board detector."""

import importlib.util
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from lichess_pipeline.chess_vision import detect_green_screen_board

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator = _load_script("generate_screen_template", "scripts/generate_screen_template.py")
trainer = _load_script("train_screen_chess_yoloseg", "scripts/train_screen_chess_yoloseg.py")


def test_generate_screen_template_produces_valid_board_and_extractable_sprites(tmp_path):
    output = tmp_path / "screen-template.png"
    result = generator.generate_screen_template(output, size=800)
    assert result == output

    bgr = cv2.imread(str(output))
    assert bgr is not None
    assert bgr.shape == (800, 800, 3)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Empty middle squares (d5 light, e5 dark) must have exact flat background colours.
    np.testing.assert_array_equal(rgb[350, 350], generator.LIGHT_SQUARE)
    np.testing.assert_array_equal(rgb[350, 450], generator.DARK_SQUARE)

    sprites = trainer.extract_sprites(output)
    assert set(sprites) == set(trainer.SPRITE_SQUARES)
    for symbol, (_tile, mask) in sprites.items():
        assert cv2.countNonZero(mask) > 500, f"Sprite {symbol} mask is too small"
        # Pieces should stay inside their square without touching the outer 1 px tile edge.
        assert np.all(mask[0, :] == 0) and np.all(mask[-1, :] == 0)
        assert np.all(mask[:, 0] == 0) and np.all(mask[:, -1] == 0)

    canvas, labels = trainer.render_position(chess.Board(), sprites)
    assert canvas.shape == (800, 800, 3)
    assert len(labels) == 32

    corners = detect_green_screen_board(bgr)
    assert corners is not None
    np.testing.assert_allclose(corners, [[0, 0], [799, 0], [799, 799], [0, 799]], atol=5)


def test_generate_screen_template_rejects_invalid_size(tmp_path):
    with pytest.raises(ValueError, match="divisible by 8"):
        generator.generate_screen_template(tmp_path / "bad.png", size=801)
    with pytest.raises(ValueError, match="at least 8"):
        generator.generate_screen_template(tmp_path / "zero.png", size=0)
