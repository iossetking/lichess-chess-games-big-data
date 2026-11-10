"""Rectify and segment a chessboard from a static phone photo.

Run with:
    uv run python scripts/chessboard_photo_demo.py --image path/to/photo.jpg

Click the INNER 8x8 grid corners in this exact order: a8, h8, h1, a1.  The
script writes a rectified board image with the 64 square labels overlaid.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support direct execution before an editable package installation is available.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2
import numpy as np

from lichess_pipeline.chess_vision import (
    DEFAULT_BOARD_SIZE,
    board_destination,
    square_regions,
    validate_ordered_board_corners,
)

CORNER_NAMES = ("a8", "h8", "h1", "a1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Path to a photo of a chessboard")
    parser.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE, help="Rectified board width and height")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Default: <image name>_rectified.png next to the photo",
    )
    return parser.parse_args()


def resize_for_selection(image: np.ndarray, max_edge: int = 1600) -> np.ndarray:
    """Limit very large phone photos while retaining enough detail for an 800 px board."""
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def draw_selected_corners(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    preview = image.copy()
    for index, point in enumerate(points):
        cv2.circle(preview, point, 8, (0, 255, 0), -1)
        cv2.putText(
            preview,
            f"{index + 1}: {CORNER_NAMES[index]}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
    if len(points) > 1:
        cv2.polylines(preview, [np.asarray(points, dtype=np.int32)], False, (0, 255, 0), 2)
    return preview


def select_board_corners(image: np.ndarray) -> np.ndarray | None:
    """Ask the user for a8, h8, h1 and a1 with an OpenCV click interface."""
    points: list[tuple[int, int]] = []
    window_name = "Select board corners"

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Click the INNER board grid corners in order: 1=a8, 2=h8, 3=h1, 4=a1.")
    print("Press Enter after four clicks; r resets; q or Esc cancels.")
    while True:
        preview = draw_selected_corners(image, points)
        remaining = "Select " + CORNER_NAMES[len(points)] if len(points) < 4 else "Press Enter to rectify"
        cv2.putText(preview, remaining, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            return None
        if key == ord("r"):
            points.clear()
        if key in (10, 13) and len(points) == 4:
            try:
                return validate_ordered_board_corners(np.asarray(points, dtype=np.float32))
            except ValueError as error:
                print(f"Invalid corner selection: {error}. Press r and try again.", file=sys.stderr)


def draw_grid(image: np.ndarray, regions: dict[str, tuple[int, int, int, int]]) -> None:
    for name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(image, name, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)


def main() -> int:
    args = parse_args()
    if args.size % 8:
        print("--size must be divisible by 8", file=sys.stderr)
        return 2
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1

    image = resize_for_selection(image)
    corners = select_board_corners(image)
    if corners is None:
        cv2.destroyAllWindows()
        return 0

    board = cv2.warpPerspective(image, cv2.getPerspectiveTransform(corners, board_destination(args.size)), (args.size, args.size))
    draw_grid(board, square_regions(args.size))
    output = args.output or args.image.with_name(f"{args.image.stem}_rectified.png")
    if not cv2.imwrite(str(output), board):
        print(f"Could not write output image: {output}", file=sys.stderr)
        return 1
    print(f"Saved rectified board with 64 regions to: {output}")

    selected = draw_selected_corners(image, [tuple(point.astype(int)) for point in corners])
    cv2.polylines(selected, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    cv2.imshow("Source photo", selected)
    cv2.imshow("Rectified board", board)
    print("Press q, Esc or Enter to close.")
    while True:
        if cv2.waitKey(20) & 0xFF in (27, ord("q"), 10, 13):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
