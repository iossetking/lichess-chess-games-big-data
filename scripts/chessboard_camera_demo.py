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
    validate_ordered_board_corners,
)

MARKER_IDS = (10, 11, 12, 13)  # top-left, top-right, bottom-right, bottom-left
MANUAL_CORNER_NAMES = ("a8", "h8", "h1", "a1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE, help="Rectified board width/height")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name")
    parser.add_argument(
        "--manual-corners",
        action="store_true",
        help="Calibrate a marker-free board by clicking its four corners after pressing c",
    )
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


def draw_manual_corners(image: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    preview = image.copy()
    for index, point in enumerate(points):
        cv2.circle(preview, point, 8, (0, 255, 0), -1)
        cv2.putText(
            preview,
            f"{index + 1}: {MANUAL_CORNER_NAMES[index]}",
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
    if len(points) > 1:
        cv2.polylines(preview, [np.asarray(points, dtype=np.int32)], False, (0, 255, 0), 2)
    return preview


def select_manual_corners(frame: np.ndarray) -> np.ndarray | None:
    """Capture a marker-free board by clicking a8, h8, h1 and a1."""
    points: list[tuple[int, int]] = []
    window_name = "Manual board calibration"

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Click the inner grid corners: 1=a8, 2=h8, 3=h1, 4=a1. Enter confirms; r resets.")
    while True:
        preview = draw_manual_corners(frame, points)
        instruction = (
            f"Select {MANUAL_CORNER_NAMES[len(points)]}"
            if len(points) < 4
            else "Press Enter to use these corners"
        )
        cv2.putText(preview, instruction, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            return None
        if key == ord("r"):
            points.clear()
        if key in (10, 13) and len(points) == 4:
            try:
                corners = validate_ordered_board_corners(np.asarray(points, dtype=np.float32))
                cv2.destroyWindow(window_name)
                return corners
            except ValueError as error:
                print(f"Invalid corners: {error}. Press r and try again.", file=sys.stderr)


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
    manual_corners: np.ndarray | None = None
    if args.manual_corners:
        print("Manual mode: point the camera at the board image and press c to calibrate it.")
    else:
        print("Press q or Esc to stop. Show all four markers to rectify the board.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera", file=sys.stderr)
                return 1

            preview = frame.copy()
            if args.manual_corners:
                ordered_corners = manual_corners
                if ordered_corners is None:
                    cv2.putText(preview, "Press c to select board corners", (20, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                corners, ids, _ = detector.detectMarkers(frame)
                markers = detected_marker_map(corners, ids)
                ordered_corners = marker_centers_by_corner(markers, MARKER_IDS)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(preview, corners, ids)

            if ordered_corners is not None:
                transform = cv2.getPerspectiveTransform(ordered_corners, destination)
                board = cv2.warpPerspective(frame, transform, (args.size, args.size))
                draw_grid(board, regions)
                # This loop is intentionally the model integration point.  It is
                # not displayed yet to avoid pretending the placeholder predicts pieces.
                _ = {name: classify_square(board[y1:y2, x1:x2])
                     for name, (x1, y1, x2, y2) in regions.items()}
                cv2.polylines(preview, [ordered_corners.astype(np.int32)], True, (0, 255, 0), 3)
                cv2.imshow("Rectified board", board)
            elif not args.manual_corners:
                cv2.putText(preview, "Need ArUco IDs 10, 11, 12, 13", (20, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Camera", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if args.manual_corners and key == ord("c"):
                manual_corners = select_manual_corners(frame)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
