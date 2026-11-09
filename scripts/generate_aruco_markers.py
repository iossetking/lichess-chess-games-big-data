"""Generate the four printable markers required by chessboard_camera_demo.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

MARKER_IDS = (10, 11, 12, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/aruco_markers"))
    parser.add_argument("--pixels", type=int, default=600, help="Width and height of each PNG")
    parser.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pixels < 20:
        raise ValueError("--pixels must be at least 20")
    if not hasattr(cv2.aruco, args.dictionary):
        raise ValueError(f"Unknown ArUco dictionary: {args.dictionary}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dictionary))
    args.output.mkdir(parents=True, exist_ok=True)
    for marker_id in MARKER_IDS:
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, args.pixels)
        output = args.output / f"aruco_{marker_id}.png"
        cv2.imwrite(str(output), image)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
