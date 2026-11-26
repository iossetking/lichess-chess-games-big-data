"""Train a multi-style, camera-robust YOLO-seg model for 2D chessboards on screens.

Generates synthetic boards combining multiple 2D piece sets (the supplied template
such as Chess.com Neo *and* Lichess/cburnett SVGs from ``chess.svg.PIECES``),
popular board colour themes, last-move/check highlights, coordinates, slight
rectification jitter and realistic phone/monitor camera artifacts.

Example:
    uv run python scripts/train_screen_chess_yoloseg.py --epochs 50 --device 0
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import chess
import cv2
import numpy as np
from PIL import Image

BOARD_SIZE = 800
SQUARE_SIZE = BOARD_SIZE // 8
LIGHT_SQUARE = np.array((235, 236, 208), dtype=np.uint8)
DARK_SQUARE = np.array((119, 149, 86), dtype=np.uint8)

CLASS_IDS = {
    "b": 0, "k": 1, "n": 2, "p": 3, "q": 4, "r": 5,
    "B": 6, "K": 7, "N": 8, "P": 9, "Q": 10, "R": 11,
}
CLASS_NAMES = (
    "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook",
    "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook",
)
SPRITE_SQUARES = {
    "r": "a8", "n": "b8", "b": "c8", "q": "d8", "k": "e8", "p": "a7",
    "R": "a1", "N": "b1", "B": "c1", "Q": "d1", "K": "e1", "P": "a2",
}
# Pieces that appear on BOTH a light and a dark square in the starting position
# allow exact closed-form two-background alpha matting (zero colour halo).
DUAL_BG_SQUARES: dict[str, tuple[str, str]] = {
    # symbol: (light_square, dark_square)
    "r": ("a8", "h8"),
    "n": ("g8", "b8"),
    "b": ("c8", "f8"),
    "p": ("b7", "a7"),
    "R": ("h1", "a1"),
    "N": ("b1", "g1"),
    "B": ("f1", "c1"),
    "P": ("a2", "b2"),
}

# Popular online chessboard themes: (light_rgb, dark_rgb, highlight_light, highlight_dark, weight)
BOARD_THEMES: tuple[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], float], ...] = (
    ((235, 236, 208), (119, 149, 86), (245, 246, 130), (185, 202, 67), 0.32),   # Chess.com Green
    ((240, 217, 181), (181, 136, 99), (205, 210, 106), (170, 162, 58), 0.22),   # Lichess Brown
    ((234, 233, 210), (75, 115, 153), (244, 246, 128), (140, 172, 120), 0.12),  # Chess.com Blue
    ((222, 227, 230), (140, 162, 173), (205, 210, 106), (155, 172, 110), 0.08), # Lichess Blue
    ((238, 238, 210), (79, 119, 45), (246, 246, 105), (186, 202, 43), 0.08),    # Tournament Green
    ((230, 204, 160), (156, 111, 68), (235, 215, 110), (186, 165, 62), 0.07),   # Walnut Wood
    ((216, 217, 216), (120, 130, 140), (220, 224, 130), (158, 168, 100), 0.06), # Slate Gray
    ((239, 239, 239), (136, 119, 183), (242, 240, 140), (168, 162, 120), 0.05), # Lichess Purple
)

SPLITS = {"train": 1600, "valid": 200, "test": 200}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "data" / "vision" / "screen-template.png")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "vision" / "screen-chess-seg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-model", default="yolo11s-seg.pt")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "data" / "runs" / "vision")
    parser.add_argument("--name", default="screen-chess-yolo11s-seg")
    parser.add_argument("--train-count", type=int, default=SPLITS["train"])
    parser.add_argument("--valid-count", type=int, default=SPLITS["valid"])
    parser.add_argument("--test-count", type=int, default=SPLITS["test"])
    parser.add_argument("--no-cburnett", action="store_true", help="Only use --template sprites (skip chess.svg cburnett)")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate the derived synthetic dataset")
    return parser.parse_args()


def _square_bounds(square_name: str) -> tuple[int, int, int, int]:
    file_index = ord(square_name[0]) - ord("a")
    row_index = 8 - int(square_name[1])
    x, y = file_index * SQUARE_SIZE, row_index * SQUARE_SIZE
    return x, y, x + SQUARE_SIZE, y + SQUARE_SIZE


def _is_light_square(square_name: str) -> bool:
    file_index = ord(square_name[0]) - ord("a")
    row_index = 8 - int(square_name[1])
    return (file_index + row_index) % 2 == 0


def _detect_template_square_colours(template: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Measure the exact light and dark square colours from the empty middle ranks."""
    light_samples, dark_samples = [], []
    for row in range(2, 6):
        for col in range(8):
            y1, x1 = row * SQUARE_SIZE + 25, col * SQUARE_SIZE + 25
            patch = template[y1:y1 + 50, x1:x1 + 50].reshape(-1, 3)
            if (row + col) % 2 == 0:
                light_samples.append(patch)
            else:
                dark_samples.append(patch)
    light = np.median(np.vstack(light_samples), axis=0).astype(np.float32)
    dark = np.median(np.vstack(dark_samples), axis=0).astype(np.float32)
    return light, dark


def _square_background(square_name: str, light: np.ndarray = LIGHT_SQUARE, dark: np.ndarray = DARK_SQUARE) -> np.ndarray:
    return light if _is_light_square(square_name) else dark


def extract_sprites(template_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract RGB sprite pixels and binary masks from a starting-position template."""
    rgba_sprites = extract_rgba_sprites_from_template(template_path)
    sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, rgba in rgba_sprites.items():
        mask = (rgba[:, :, 3] >= 48).astype(np.uint8) * 255
        sprites[symbol] = rgba[:, :, :3].copy(), mask
    return sprites


def extract_rgba_sprites_from_template(template_path: Path) -> dict[str, np.ndarray]:
    """Extract halo-free RGBA (100x100x4) sprites from a starting-position template.

    Uses closed-form two-background alpha matting for every piece type that
    appears on both light and dark squares (rook, knight, bishop, pawn) and
    soft background-decontaminated matting for queens and kings.
    """
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")
    image = Image.open(template_path).convert("RGB").resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.LANCZOS)
    template = np.asarray(image)
    light_bg, dark_bg = _detect_template_square_colours(template)
    bg_diff = light_bg - dark_bg
    bg_diff_norm2 = float(np.dot(bg_diff, bg_diff))

    sprites: dict[str, np.ndarray] = {}
    for symbol, square_name in SPRITE_SQUARES.items():
        if symbol in DUAL_BG_SQUARES and bg_diff_norm2 > 100.0:
            sq_l, sq_d = DUAL_BG_SQUARES[symbol]
            xl1, yl1, xl2, yl2 = _square_bounds(sq_l)
            xd1, yd1, xd2, yd2 = _square_bounds(sq_d)
            tile_l = template[yl1:yl2, xl1:xl2].astype(np.float32)
            tile_d = template[yd1:yd2, xd1:xd2].astype(np.float32)

            # Two-background matting: I_L - I_D = (1 - alpha) * (B_L - B_D)
            delta = tile_l - tile_d
            one_minus_alpha = np.sum(delta * bg_diff[None, None, :], axis=2) / bg_diff_norm2
            alpha = np.clip(1.0 - one_minus_alpha, 0.0, 1.0)
            # Suppress subtle background noise in empty regions.
            dist_l = np.abs(tile_l - light_bg).max(axis=2)
            dist_d = np.abs(tile_d - dark_bg).max(axis=2)
            bg_mask = (dist_l < 6.0) & (dist_d < 6.0)
            alpha[bg_mask] = 0.0

            # Recover true foreground RGB from the higher-contrast background.
            safe_alpha = np.maximum(alpha, 0.08)[:, :, None]
            fg_l = (tile_l - (1.0 - alpha[:, :, None]) * light_bg[None, None, :]) / safe_alpha
            fg_d = (tile_d - (1.0 - alpha[:, :, None]) * dark_bg[None, None, :]) / safe_alpha
            fg = np.clip(0.5 * (fg_l + fg_d), 0.0, 255.0)
            # Clean very low-alpha pixels so borders remain crisp.
            fg[alpha < 0.06] = 0.0
            alpha[alpha < 0.06] = 0.0
        else:
            x1, y1, x2, y2 = _square_bounds(square_name)
            tile = template[y1:y2, x1:x2].astype(np.float32)
            bg = _square_background(square_name, light_bg, dark_bg)
            diff = np.abs(tile - bg[None, None, :]).max(axis=2)
            core = (diff > 16.0).astype(np.uint8) * 255
            core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # Keep the main connected component (drop stray noise) while preserving
            # genuine transparent cutouts (such as the openings inside the Neo king).
            contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outer_mask = np.zeros_like(core)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                cv2.drawContours(outer_mask, [largest], -1, 255, thickness=cv2.FILLED)
            filled = cv2.bitwise_and(outer_mask, (diff > 12.0).astype(np.uint8) * 255)
            filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # Soft ramp along the 1-2px anti-aliased boundary (outer + cutout edges).
            eroded = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=1)
            edge_band = (filled > 0) & (eroded == 0)
            alpha = np.where(eroded > 0, 1.0, 0.0).astype(np.float32)
            alpha[edge_band] = np.clip((diff[edge_band] - 4.0) / 32.0, 0.0, 1.0)
            # Decontaminate edge pixels toward interior foreground.
            safe_alpha = np.maximum(alpha, 0.15)[:, :, None]
            fg = np.where(
                edge_band[:, :, None],
                np.clip((tile - (1.0 - alpha[:, :, None]) * bg[None, None, :]) / safe_alpha, 0.0, 255.0),
                tile,
            )
            fg[alpha < 0.06] = 0.0
            alpha[alpha < 0.06] = 0.0

        # Ensure outer 1px border is empty so pieces stay cleanly inside their tile.
        alpha[0, :] = 0.0
        alpha[-1, :] = 0.0
        alpha[:, 0] = 0.0
        alpha[:, -1] = 0.0

        if int((alpha >= 0.2).sum()) < 80:
            raise ValueError(f"Could not isolate sprite {symbol!r} from {template_path}")

        rgba = np.empty((SQUARE_SIZE, SQUARE_SIZE, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.rint(fg).astype(np.uint8)
        rgba[:, :, 3] = np.rint(alpha * 255.0).astype(np.uint8)
        sprites[symbol] = rgba
    return sprites


def extract_cburnett_rgba_sprites() -> dict[str, np.ndarray]:
    """Render clean RGBA sprites for Lichess/Wikipedia cburnett via ``generate_screen_template``."""
    spec = importlib.util.spec_from_file_location(
        "generate_screen_template", PROJECT_ROOT / "scripts" / "generate_screen_template.py"
    )
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    sprites: dict[str, np.ndarray] = {}
    for symbol in SPRITE_SQUARES:
        rgba = gen.render_piece_sprite(symbol, SQUARE_SIZE)
        # Keep outer 1px border transparent.
        rgba[0, :, 3] = 0
        rgba[-1, :, 3] = 0
        rgba[:, 0, 3] = 0
        rgba[:, -1, 3] = 0
        sprites[symbol] = rgba
    return sprites


def empty_board(
    light: np.ndarray = LIGHT_SQUARE,
    dark: np.ndarray = DARK_SQUARE,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    canvas = np.empty((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            colour = light if (row + column) % 2 == 0 else dark
            canvas[row * SQUARE_SIZE:(row + 1) * SQUARE_SIZE, column * SQUARE_SIZE:(column + 1) * SQUARE_SIZE] = colour
    if rng is not None and rng.random() < 0.25:
        # Subtle board texture / wood grain.
        grain = rng.normal(0.0, rng.uniform(1.5, 4.5), (BOARD_SIZE, BOARD_SIZE, 1))
        if rng.random() < 0.5:
            grain = cv2.GaussianBlur(grain.astype(np.float32), (15, 3), 0)[:, :, None]
        canvas = np.clip(canvas.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return canvas


def _add_board_overlays(
    canvas: np.ndarray,
    board: chess.Board,
    light: np.ndarray,
    dark: np.ndarray,
    hl_light: np.ndarray,
    hl_dark: np.ndarray,
    flipped: bool,
    rng: np.random.Generator,
) -> None:
    """Paint realistic Chess.com / Lichess UI overlays onto the empty board."""
    # 1. Last-move highlight squares (very common in real online play).
    if rng.random() < 0.65:
        for sq in rng.choice(64, size=2, replace=False):
            file_idx, rank_idx = chess.square_file(int(sq)), chess.square_rank(int(sq))
            col = 7 - file_idx if flipped else file_idx
            row = rank_idx if flipped else 7 - rank_idx
            is_light = (row + col) % 2 == 0
            hl = hl_light if is_light else hl_dark
            x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
            canvas[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE] = hl

    # 2. Check highlight (red radial tint on a king square).
    if rng.random() < 0.15:
        kings = [sq for sq, p in board.piece_map().items() if p.piece_type == chess.KING]
        if kings:
            sq = int(rng.choice(kings))
            file_idx, rank_idx = chess.square_file(sq), chess.square_rank(sq)
            col = 7 - file_idx if flipped else file_idx
            row = rank_idx if flipped else 7 - rank_idx
            x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
            yy, xx = np.ogrid[:SQUARE_SIZE, :SQUARE_SIZE]
            dist = np.sqrt((xx - SQUARE_SIZE / 2) ** 2 + (yy - SQUARE_SIZE / 2) ** 2) / (SQUARE_SIZE * 0.55)
            strength = np.clip(1.0 - dist, 0.0, 1.0)[:, :, None] * 0.75
            patch = canvas[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE].astype(np.float32)
            red = np.array([235.0, 45.0, 45.0], dtype=np.float32)
            canvas[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE] = np.clip(
                patch * (1.0 - strength) + red * strength, 0, 255
            ).astype(np.uint8)

    # 3. Coordinate labels (1-8 on left/right edge, a-h on bottom edge).
    if rng.random() < 0.55:
        ranks = "12345678" if flipped else "87654321"
        files = "hgfedcba" if flipped else "abcdefgh"
        right_side = rng.random() < 0.35
        for row in range(8):
            col = 7 if right_side else 0
            is_light = (row + col) % 2 == 0
            txt_colour = tuple(int(v) for v in (dark if is_light else light))
            tx = col * SQUARE_SIZE + (SQUARE_SIZE - 16 if right_side else 5)
            ty = row * SQUARE_SIZE + 19
            cv2.putText(canvas, ranks[row], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_colour, 2, cv2.LINE_AA)
        for col in range(8):
            row = 7
            is_light = (row + col) % 2 == 0
            txt_colour = tuple(int(v) for v in (dark if is_light else light))
            tx = col * SQUARE_SIZE + SQUARE_SIZE - 16
            ty = row * SQUARE_SIZE + SQUARE_SIZE - 7
            cv2.putText(canvas, files[col], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_colour, 2, cv2.LINE_AA)

    # 4. Legal-move hint dots on empty squares.
    if rng.random() < 0.25:
        occupied = set(board.piece_map().keys())
        empty_sqs = [sq for sq in range(64) if sq not in occupied]
        if empty_sqs:
            n_dots = min(len(empty_sqs), int(rng.integers(2, 7)))
            for sq in rng.choice(empty_sqs, size=n_dots, replace=False):
                file_idx, rank_idx = chess.square_file(int(sq)), chess.square_rank(int(sq))
                col = 7 - file_idx if flipped else file_idx
                row = rank_idx if flipped else 7 - rank_idx
                cx, cy = col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2
                overlay = canvas.copy()
                cv2.circle(overlay, (cx, cy), int(SQUARE_SIZE * 0.16), (40, 40, 40), -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, canvas)


def random_legal_board(rng: np.random.Generator) -> chess.Board:
    """Sample diverse openings, middlegames, endgames, and promoted positions."""
    mode = float(rng.random())
    board = chess.Board()
    if mode < 0.15:
        # Starting position or very early opening (0-6 plies)
        plies = int(rng.integers(0, 7))
    elif mode < 0.65:
        # Middlegame (7-65 plies)
        plies = int(rng.integers(7, 66))
    elif mode < 0.90:
        # Endgame / deeper play (66-140 plies)
        plies = int(rng.integers(66, 141))
    else:
        # Extra pieces / promotion-heavy positions to boost king/queen/rook/bishop/knight recall
        board.clear_board()
        squares = rng.choice(64, size=int(rng.integers(8, 28)), replace=False).tolist()
        # Always include both kings
        board.set_piece_at(squares[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(squares[1], chess.Piece(chess.KING, chess.BLACK))
        pool = [
            chess.Piece(chess.QUEEN, chess.WHITE), chess.Piece(chess.QUEEN, chess.BLACK),
            chess.Piece(chess.ROOK, chess.WHITE), chess.Piece(chess.ROOK, chess.BLACK),
            chess.Piece(chess.BISHOP, chess.WHITE), chess.Piece(chess.BISHOP, chess.BLACK),
            chess.Piece(chess.KNIGHT, chess.WHITE), chess.Piece(chess.KNIGHT, chess.BLACK),
            chess.Piece(chess.PAWN, chess.WHITE), chess.Piece(chess.PAWN, chess.BLACK),
        ]
        for sq in squares[2:]:
            piece = pool[int(rng.integers(0, len(pool)))]
            if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                piece = chess.Piece(chess.QUEEN, piece.color)
            board.set_piece_at(sq, piece)
        return board

    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[int(rng.integers(0, len(moves)))])
    return board


def _jitter_sprite(rgba: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply subtle scale, offset, and tone variations to an RGBA piece sprite."""
    out = rgba.copy()
    scale = float(rng.uniform(0.91, 1.04))
    dx = float(rng.uniform(-3.0, 3.0))
    dy = float(rng.uniform(-3.0, 3.0))
    center = SQUARE_SIZE / 2.0
    matrix = np.array(
        [[scale, 0.0, center * (1.0 - scale) + dx],
         [0.0, scale, center * (1.0 - scale) + dy]],
        dtype=np.float32,
    )
    out = cv2.warpAffine(out, matrix, (SQUARE_SIZE, SQUARE_SIZE), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    # Subtle piece colour/contrast shift (e.g., cream vs stark white, dark charcoal vs jet black).
    if rng.random() < 0.5:
        gain = float(rng.uniform(0.92, 1.06))
        bias = float(rng.uniform(-10.0, 10.0))
        rgb = np.clip(out[:, :, :3].astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        out[:, :, :3] = rgb
    return out


def render_position(
    board: chess.Board,
    sprites: dict[str, tuple[np.ndarray, np.ndarray]] | dict[str, np.ndarray],
    *,
    rng: np.random.Generator | None = None,
    augment_board: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Render ``board`` and return ``(RGB image, YOLO-seg polygon label lines)``.

    Accepts either legacy ``(rgb, mask)`` tuples (from :func:`extract_sprites`)
    or RGBA ``(H, W, 4)`` arrays (from :func:`extract_rgba_sprites_from_template`
    or :func:`extract_cburnett_rgba_sprites`).
    """
    flipped = False
    if augment_board and rng is not None:
        weights = np.array([t[4] for t in BOARD_THEMES], dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(len(BOARD_THEMES), p=weights))
        light, dark, hl_light, hl_dark, _ = BOARD_THEMES[idx]
        light_arr = np.array(light, dtype=np.float32)
        dark_arr = np.array(dark, dtype=np.float32)
        if rng.random() < 0.35:
            # Mild random tint around the chosen theme
            shift = rng.uniform(-12, 12, size=3)
            light_arr = np.clip(light_arr + shift, 140, 255)
            dark_arr = np.clip(dark_arr + shift, 35, 210)
        light_u8 = light_arr.astype(np.uint8)
        dark_u8 = dark_arr.astype(np.uint8)
        canvas = empty_board(light_u8, dark_u8, rng=rng)
        flipped = bool(rng.random() < 0.30)
        _add_board_overlays(
            canvas, board, light_u8, dark_u8,
            np.array(hl_light, dtype=np.uint8), np.array(hl_dark, dtype=np.uint8),
            flipped, rng,
        )
    else:
        canvas = empty_board()

    polygons_with_cls: list[tuple[int, np.ndarray]] = []
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        entry = sprites[symbol]
        file_idx, rank_idx = chess.square_file(square), chess.square_rank(square)
        col = 7 - file_idx if flipped else file_idx
        row = rank_idx if flipped else 7 - rank_idx
        x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
        target = canvas[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]

        if isinstance(entry, tuple):
            sprite_rgb, mask = entry
            target[mask > 0] = sprite_rgb[mask > 0]
            binary = mask
        else:
            rgba = _jitter_sprite(entry, rng) if (augment_board and rng is not None) else entry
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            fg = rgba[:, :, :3].astype(np.float32)
            blended = fg * alpha + target.astype(np.float32) * (1.0 - alpha)
            canvas[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
            binary = (rgba[:, :, 3] >= 48).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 60:
            continue
        # Mildly simplify dense contours so label files stay compact without losing silhouette accuracy.
        approx = cv2.approxPolyDP(contour, 0.8, True).reshape(-1, 2)
        if len(approx) < 3:
            approx = contour.reshape(-1, 2)
        if len(approx) < 3:
            continue
        polygons_with_cls.append((CLASS_IDS[symbol], (approx + np.array((x, y))).astype(np.float32)))

    # Optional slight rectification/crop jitter (simulates imperfect 4-corner detection or slight bezel margin).
    if augment_board and rng is not None and rng.random() < 0.55:
        margin = BOARD_SIZE * 0.04
        src = np.array([[0, 0], [BOARD_SIZE - 1, 0], [BOARD_SIZE - 1, BOARD_SIZE - 1], [0, BOARD_SIZE - 1]], dtype=np.float32)
        dst = src + rng.uniform(-margin, margin, size=(4, 2)).astype(np.float32)
        homography = cv2.getPerspectiveTransform(src, dst)
        border_col = tuple(int(v) for v in rng.integers(15, 70, size=3))
        canvas = cv2.warpPerspective(
            canvas, homography, (BOARD_SIZE, BOARD_SIZE),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_col,
        )
        warped_polys: list[tuple[int, np.ndarray]] = []
        for cls_id, poly in polygons_with_cls:
            pts = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), homography).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0.0, BOARD_SIZE - 1.0)
            pts[:, 1] = np.clip(pts[:, 1], 0.0, BOARD_SIZE - 1.0)
            if (pts[:, 0].max() - pts[:, 0].min()) >= 8 and (pts[:, 1].max() - pts[:, 1].min()) >= 8:
                warped_polys.append((cls_id, pts))
        polygons_with_cls = warped_polys

    labels: list[str] = []
    for cls_id, poly in polygons_with_cls:
        norm = np.clip(poly / float(BOARD_SIZE), 0.0, 1.0)
        coordinates = " ".join(f"{value:.6f}" for point in norm for value in point)
        labels.append(f"{cls_id} {coordinates}")
    return canvas, labels


def phone_like_appearance(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate webcam-on-screen degradations while preserving digital clarity on ~25% of images."""
    if rng.random() < 0.25:
        # Clean digital screenshot path (direct screen capture).
        return image

    result = image.astype(np.float32)

    # Colour temperature / white-balance shift (cool LED / warm Night Shift).
    if rng.random() < 0.65:
        gains = rng.uniform(0.86, 1.14, size=3).astype(np.float32)
        result *= gains[None, None, :]

    # Brightness, contrast, and gamma.
    alpha = float(rng.uniform(0.75, 1.22))
    beta = float(rng.uniform(-24.0, 24.0))
    result = np.clip(result * alpha + beta, 0.0, 255.0)
    if rng.random() < 0.45:
        gamma = float(rng.uniform(0.80, 1.28))
        result = 255.0 * np.power(result / 255.0, gamma)

    # Diagonal or radial screen glare / ambient reflection.
    if rng.random() < 0.45:
        h, w = result.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = float(rng.uniform(0.1 * w, 0.9 * w)), float(rng.uniform(0.1 * h, 0.9 * h))
        sigma = float(rng.uniform(0.25 * w, 0.65 * w))
        glare = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        intensity = float(rng.uniform(18.0, 68.0))
        result = np.clip(result + glare[:, :, None] * intensity, 0.0, 255.0)

    # Subtle screen moiré / scanlines.
    if rng.random() < 0.30:
        h, w = result.shape[:2]
        freq = float(rng.uniform(0.15, 0.7))
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(2.5, 9.0))
        if rng.random() < 0.5:
            wave = np.sin(np.arange(h, dtype=np.float32) * freq + phase)[:, None, None]
        else:
            wave = np.sin(np.arange(w, dtype=np.float32) * freq + phase)[None, :, None]
        result = np.clip(result + wave * amp, 0.0, 255.0)

    out = result.astype(np.uint8)

    # Mild defocus or resolution loss (as when webcam sees a phone from 40-70 cm).
    if rng.random() < 0.45:
        scale = float(rng.uniform(0.45, 0.85))
        small = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    elif rng.random() < 0.35:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), float(rng.uniform(0.3, 1.2)))

    # Camera sensor noise.
    if rng.random() < 0.45:
        noise = rng.normal(0.0, float(rng.uniform(1.5, 7.0)), out.shape).astype(np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Occasional stronger JPEG compression artifacts.
    if rng.random() < 0.35:
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=int(rng.integers(55, 86)))
        buf.seek(0)
        out = np.asarray(Image.open(buf).convert("RGB"))

    return out


def generate_dataset(
    template_path: Path,
    destination: Path,
    force: bool,
    *,
    splits: dict[str, int] | None = None,
    include_cburnett: bool = True,
) -> Path:
    data_yaml = destination / "data.yaml"
    if data_yaml.exists() and not force:
        return data_yaml
    if force and destination.exists():
        shutil.rmtree(destination)

    sprite_sets: list[tuple[str, dict[str, np.ndarray]]] = []
    if template_path.is_file():
        sprite_sets.append(("template", extract_rgba_sprites_from_template(template_path)))
    if include_cburnett:
        sprite_sets.append(("cburnett", extract_cburnett_rgba_sprites()))
    if not sprite_sets:
        raise FileNotFoundError(f"No sprite templates available (missing {template_path})")

    active_splits = splits or SPLITS
    for split_index, (split, count) in enumerate(active_splits.items()):
        image_dir = destination / split / "images"
        label_dir = destination / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(2026 + split_index)
        for index in range(count):
            _, sprites = sprite_sets[index % len(sprite_sets)]
            image, labels = render_position(random_legal_board(rng), sprites, rng=rng, augment_board=True)
            image = phone_like_appearance(image, rng)
            Image.fromarray(image).save(image_dir / f"{index:05}.jpg", quality=92)
            (label_dir / f"{index:05}.txt").write_text("\n".join(labels), encoding="utf-8")
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data_yaml.write_text(
        f"path: {destination}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return data_yaml


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        print("--epochs, --imgsz and --batch must be positive", file=sys.stderr)
        return 2
    splits = {"train": args.train_count, "valid": args.valid_count, "test": args.test_count}
    data_yaml = generate_dataset(
        args.template,
        args.data_dir,
        args.force,
        splits=splits,
        include_cburnett=not args.no_cburnett,
    )
    print(f"Synthetic multi-style screen dataset ready: {data_yaml}")
    if args.generate_only:
        return 0

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=max(12, args.epochs // 3),
        seed=2026,
        workers=2 if str(args.device) != "mps" else 0,
        cls=1.0,
        cos_lr=True,
        hsv_h=0.12,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=6.0,
        translate=0.08,
        scale=0.35,
        perspective=0.0005,
        fliplr=0.5,
        mosaic=0.2,
        close_mosaic=5,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Ultralytics did not produce {best}")
    output = PROJECT_ROOT / "data" / "models" / "screen-chess-pieces-seg.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"Screen-specific YOLO-seg weights copied to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
