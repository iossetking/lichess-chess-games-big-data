"""Live ArUco-assisted chessboard rectification demo.

Print four ArUco markers (IDs 10--13) and arrange them just outside the
physical board's top-left, top-right, bottom-right and bottom-left corners.
Run with: uv run python scripts/chessboard_camera_demo.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

# Support direct execution (`python scripts/chessboard_camera_demo.py`) even in
# environments where the project has not been installed in editable mode yet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2
import numpy as np

from lichess_pipeline.chess_vision import (
    DEFAULT_BOARD_SIZE,
    board_destination,
    classify_square,
    marker_centers_by_corner,
    square_regions,
)

MARKER_IDS = (10, 11, 12, 13)  # top-left, top-right, bottom-right, bottom-left


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE, help="Rectified board width/height")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name")
    return parser.parse_args()


def get_dictionary(name: str) -> cv2.aruco.Dictionary:
    if not hasattr(cv2.aruco, name):
        choices = ", ".join(item for item in dir(cv2.aruco) if item.startswith("DICT_"))
        raise ValueError(f"Unknown ArUco dictionary {name!r}. Available: {choices}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detected_marker_map(corners: Iterable[np.ndarray], ids: np.ndarray | None) -> dict[int, np.ndarray]:
    if ids is None:
        return {}
    return {int(marker_id): corner for marker_id, corner in zip(ids.flatten(), corners, strict=True)}


def draw_grid(image: np.ndarray, regions: dict[str, tuple[int, int, int, int]]) -> None:
    for name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(image, name, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)


def main() -> int:
    args = parse_args()
    if args.size % 8:
        print("--size must be divisible by 8", file=sys.stderr)
        return 2

    dictionary = get_dictionary(args.dictionary)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    destination = board_destination(args.size)
    regions = square_regions(args.size)
    print("Press q or Esc to stop. Show all four markers to rectify the board.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera", file=sys.stderr)
                return 1

            corners, ids, _ = detector.detectMarkers(frame)
            markers = detected_marker_map(corners, ids)
            ordered_corners = marker_centers_by_corner(markers, MARKER_IDS)
            preview = frame.copy()
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(preview, corners, ids)

            if ordered_corners is None:
                cv2.putText(preview, "Need ArUco IDs 10, 11, 12, 13", (20, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Camera", preview)
            else:
                transform = cv2.getPerspectiveTransform(ordered_corners, destination)
                board = cv2.warpPerspective(frame, transform, (args.size, args.size))
                draw_grid(board, regions)
                # This loop is intentionally the model integration point.  It is
                # not displayed yet to avoid pretending the placeholder predicts pieces.
                _ = {name: classify_square(board[y1:y2, x1:x2])
                     for name, (x1, y1, x2, y2) in regions.items()}
                cv2.polylines(preview, [ordered_corners.astype(np.int32)], True, (0, 255, 0), 3)
                cv2.imshow("Camera", preview)
                cv2.imshow("Rectified board", board)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
