"""Geometry helpers for recognizing a physical chessboard from a camera frame.

The runtime demo detects four ArUco markers placed immediately outside the
board corners.  From them it creates a bird's-eye image and the 64 square
rectangles.  Keeping the geometry independent from OpenCV makes it testable
and leaves ``classify_square`` as the integration point for a future model.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

BOARD_FILES = "abcdefgh"
BOARD_RANKS = "87654321"
DEFAULT_BOARD_SIZE = 800


def board_destination(size: int = DEFAULT_BOARD_SIZE) -> np.ndarray:
    """Return the four destination points for a square, top-down board."""
    if size < 8:
        raise ValueError("Board size must be at least 8 pixels")
    edge = size - 1
    return np.array([[0, 0], [edge, 0], [edge, edge], [0, edge]], dtype=np.float32)


def square_regions(size: int = DEFAULT_BOARD_SIZE) -> dict[str, tuple[int, int, int, int]]:
    """Map algebraic names (a8 through h1) to ``(x1, y1, x2, y2)`` pixels."""
    if size % 8:
        raise ValueError("Board size must be divisible by 8")
    side = size // 8
    return {
        f"{file}{rank}": (column * side, row * side, (column + 1) * side, (row + 1) * side)
        for row, rank in enumerate(BOARD_RANKS)
        for column, file in enumerate(BOARD_FILES)
    }


def marker_centers_by_corner(
    detected_markers: Mapping[int, np.ndarray], marker_ids: tuple[int, int, int, int]
) -> np.ndarray | None:
    """Return marker centers ordered as TL, TR, BR, BL, or ``None`` if incomplete.

    Each marker is an array with the four pixel coordinates reported by OpenCV.
    The order of ``marker_ids`` is deliberately explicit, so a board can be
    identified unambiguously even when the camera is rotated.
    """
    if any(marker_id not in detected_markers for marker_id in marker_ids):
        return None
    centers = [np.asarray(detected_markers[marker_id], dtype=np.float32).reshape(-1, 2).mean(axis=0)
               for marker_id in marker_ids]
    return np.asarray(centers, dtype=np.float32)


def validate_ordered_board_corners(corners: np.ndarray) -> np.ndarray:
    """Validate four convex board corners ordered TL, TR, BR, BL.

    This is used by the still-image workflow, where a person clicks the board
    corners instead of providing ArUco markers.  The explicit order preserves
    the a8-to-h1 orientation after perspective correction.
    """
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Expected four finite 2D corner points")

    edge_vectors = np.roll(points, -1, axis=0) - points
    next_edge_vectors = np.roll(edge_vectors, -1, axis=0)
    # 2D scalar cross product.  Writing it out avoids NumPy's 3D-only
    # ``cross`` behavior in current releases.
    turns = edge_vectors[:, 0] * next_edge_vectors[:, 1] - edge_vectors[:, 1] * next_edge_vectors[:, 0]
    if np.any(np.isclose(turns, 0)) or not (np.all(turns > 0) or np.all(turns < 0)):
        raise ValueError("Corners must form a convex quadrilateral in perimeter order")

    signed_area = 0.5 * np.abs(
        np.dot(points[:, 0], np.roll(points[:, 1], -1))
        - np.dot(points[:, 1], np.roll(points[:, 0], -1))
    )
    if signed_area < 1:
        raise ValueError("Board corner area is too small")
    return points


def classify_square(square_image: np.ndarray) -> tuple[str, float]:
    """Placeholder for the fine-tuned 13-class piece classifier.

    Expected output labels: ``empty``, ``white_pawn`` ... ``black_king``.
    Replacing this function is enough to connect the vision demo to a model.
    """
    del square_image
    return "empty", 1.0
