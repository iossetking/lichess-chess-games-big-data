"""Train YOLO-seg for the exact 2D chessboard style shown on a phone screen.

The template is one standard starting-position board.  Its twelve sprites are
extracted automatically, then recomposed into legal random chess positions
with exact segmentation polygons.  This is intentionally different from the
physical-board trainer: it learns the rendered pieces visible on the screen.

Example:
    uv run python scripts/train_screen_chess_yoloseg.py --epochs 35 --device mps
"""

from __future__ import annotations

import argparse
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
SPLITS = {"train": 250, "valid": 50, "test": 50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=PROJECT_ROOT / "data" / "vision" / "screen-template.png")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "vision" / "screen-chess-seg")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-model", default=str(PROJECT_ROOT / "data" / "models" / "yolo11n-seg.pt"))
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "data" / "runs" / "vision")
    parser.add_argument("--name", default="screen-chess-yolo11n-seg")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate the derived synthetic dataset")
    return parser.parse_args()


def _square_bounds(square_name: str) -> tuple[int, int, int, int]:
    file_index = ord(square_name[0]) - ord("a")
    row_index = 8 - int(square_name[1])
    x, y = file_index * SQUARE_SIZE, row_index * SQUARE_SIZE
    return x, y, x + SQUARE_SIZE, y + SQUARE_SIZE


def _square_background(square_name: str) -> np.ndarray:
    file_index = ord(square_name[0]) - ord("a")
    row_index = 8 - int(square_name[1])
    return LIGHT_SQUARE if (file_index + row_index) % 2 == 0 else DARK_SQUARE


def extract_sprites(template_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract RGB sprite pixels and alpha masks from the standard-position template."""
    if not template_path.is_file():
        raise FileNotFoundError(f"Template not found: {template_path}")
    image = Image.open(template_path).convert("RGB").resize((BOARD_SIZE, BOARD_SIZE), Image.Resampling.LANCZOS)
    template = np.asarray(image)
    sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, square_name in SPRITE_SQUARES.items():
        x1, y1, x2, y2 = _square_bounds(square_name)
        tile = template[y1:y2, x1:x2].copy()
        background = _square_background(square_name).astype(np.int16)
        difference = np.abs(tile.astype(np.int16) - background).max(axis=2)
        # The board uses exactly two flat square colours, while anti-aliased
        # piece pixels differ by more than this small tolerance.
        mask = (difference > 12).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(mask) < 80:
            raise ValueError(f"Could not isolate sprite {symbol!r} from {template_path}")
        sprites[symbol] = tile, mask
    return sprites


def empty_board() -> np.ndarray:
    canvas = np.empty((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            colour = LIGHT_SQUARE if (row + column) % 2 == 0 else DARK_SQUARE
            canvas[row * SQUARE_SIZE:(row + 1) * SQUARE_SIZE, column * SQUARE_SIZE:(column + 1) * SQUARE_SIZE] = colour
    return canvas


def random_legal_board(rng: np.random.Generator) -> chess.Board:
    board = chess.Board()
    for _ in range(int(rng.integers(0, 100))):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[int(rng.integers(0, len(moves)))])
    return board


def render_position(board: chess.Board, sprites: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    canvas = empty_board()
    labels: list[str] = []
    for square, piece in board.piece_map().items():
        symbol = piece.symbol()
        sprite, mask = sprites[symbol]
        column, rank = chess.square_file(square), chess.square_rank(square)
        x, y = column * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE
        target = canvas[y:y + SQUARE_SIZE, x:x + SQUARE_SIZE]
        target[mask > 0] = sprite[mask > 0]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        # Skip degenerate points but retain the true silhouette rather than a box.
        if len(contour) < 3:
            continue
        polygon = (contour + np.array((x, y))).astype(np.float32) / BOARD_SIZE
        coordinates = " ".join(f"{value:.6f}" for point in polygon for value in point)
        labels.append(f"{CLASS_IDS[symbol]} {coordinates}")
    return canvas, labels


def phone_like_appearance(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add mild screen/camera variation without changing segmentation geometry."""
    alpha = float(rng.uniform(0.82, 1.15))
    beta = float(rng.uniform(-18, 18))
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if rng.random() < 0.45:
        result = cv2.GaussianBlur(result, (3, 3), float(rng.uniform(0.2, 0.9)))
    if rng.random() < 0.35:
        noise = rng.normal(0, rng.uniform(1, 5), result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result


def generate_dataset(template_path: Path, destination: Path, force: bool) -> Path:
    data_yaml = destination / "data.yaml"
    if data_yaml.exists() and not force:
        return data_yaml
    if force and destination.exists():
        shutil.rmtree(destination)
    sprites = extract_sprites(template_path)
    for split_index, (split, count) in enumerate(SPLITS.items()):
        image_dir = destination / split / "images"
        label_dir = destination / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(2026 + split_index)
        for index in range(count):
            image, labels = render_position(random_legal_board(rng), sprites)
            image = phone_like_appearance(image, rng)
            Image.fromarray(image).save(image_dir / f"{index:05}.jpg", quality=94)
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
    data_yaml = generate_dataset(args.template, args.data_dir, args.force)
    print(f"Synthetic screen dataset ready: {data_yaml}")
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
        patience=max(10, args.epochs // 4),
        seed=2026,
        workers=0,
        perspective=0.001,
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
