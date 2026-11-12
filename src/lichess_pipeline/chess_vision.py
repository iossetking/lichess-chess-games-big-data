"""Vision primitives for ``camera -> board -> pieces -> FEN``.

The live application uses OpenCV to find the 7 x 7 internal intersections of
the board, rectifies that quadrilateral, and runs a chess-piece YOLO checkpoint
on the resulting bird's-eye image.  Keeping the mapping from YOLO detections
to FEN here makes it testable without a camera, GPU, or the optional
``ultralytics`` package.

ArUco helpers remain for backwards compatibility with the original printable
marker demo, but markers are no longer required by the live pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

BOARD_FILES = "abcdefgh"
BOARD_RANKS = "87654321"
DEFAULT_BOARD_SIZE = 800


@dataclass(frozen=True)
class PieceDetection:
    """One piece predicted by an object detector in rectified-board pixels."""

    label: str
    confidence: float
    # ``xyxy`` is deliberately model-neutral: (left, top, right, bottom).
    xyxy: tuple[float, float, float, float]
    # A YOLO-seg checkpoint supplies the pixel contour; a detection checkpoint
    # leaves it as ``None`` and is still rendered with a labelled box.
    mask: np.ndarray | None = None


_PIECE_SYMBOLS = {
    "whitepawn": "P", "wpawn": "P", "wp": "P", "pawnwhite": "P",
    "whiteknight": "N", "wknight": "N", "wn": "N", "knightwhite": "N",
    "whitebishop": "B", "wbishop": "B", "wb": "B", "bishopwhite": "B",
    "whiterook": "R", "wrook": "R", "wr": "R", "rookwhite": "R",
    "whitequeen": "Q", "wqueen": "Q", "wq": "Q", "queenwhite": "Q",
    "whiteking": "K", "wking": "K", "wk": "K", "kingwhite": "K",
    "blackpawn": "p", "bpawn": "p", "bp": "p", "pawnblack": "p",
    "blackknight": "n", "bknight": "n", "bn": "n", "knightblack": "n",
    "blackbishop": "b", "bbishop": "b", "bb": "b", "bishopblack": "b",
    "blackrook": "r", "brook": "r", "br": "r", "rookblack": "r",
    "blackqueen": "q", "bqueen": "q", "bq": "q", "queenblack": "q",
    "blackking": "k", "bking": "k", "bk": "k", "kingblack": "k",
}


def board_destination(size: int = DEFAULT_BOARD_SIZE) -> np.ndarray:
    """Return the four destination points for a square, top-down board."""
    if size < 8:
        raise ValueError("Board size must be at least 8 pixels")
    edge = size - 1
    return np.array([[0, 0], [edge, 0], [edge, edge], [0, edge]], dtype=np.float32)


def extrapolate_board_corners(inner_corners: np.ndarray) -> np.ndarray:
    """Turn the 7 x 7 internal chessboard intersections into outer corners.

    ``cv2.findChessboardCornersSB`` returns only the intersections *between*
    the 64 squares.  Extrapolating half a cell around that grid gives the four
    physical board corners in TL, TR, BR, BL image order.
    """
    points = np.asarray(inner_corners, dtype=np.float32).reshape(-1, 2)
    if points.shape != (49, 2):
        raise ValueError("Expected the 49 internal corners of a 7x7 grid")
    grid = points.reshape(7, 7, 2)
    horizontal_top = (grid[0, 6] - grid[0, 0]) / 6
    horizontal_bottom = (grid[6, 6] - grid[6, 0]) / 6
    vertical_left = (grid[6, 0] - grid[0, 0]) / 6
    vertical_right = (grid[6, 6] - grid[0, 6]) / 6
    return np.asarray(
        [
            grid[0, 0] - (horizontal_top + vertical_left) / 2,
            grid[0, 6] + (horizontal_top - vertical_right) / 2,
            grid[6, 6] + (horizontal_bottom + vertical_right) / 2,
            grid[6, 0] + (-horizontal_bottom + vertical_left) / 2,
        ],
        dtype=np.float32,
    )


def detect_board_corners(frame: np.ndarray) -> np.ndarray | None:
    """Locate a normal 8 x 8 board automatically using its internal grid.

    This is the no-marker fast path.  It works best when the camera sees the
    whole board, the squares have contrast, and the grid is not heavily hidden
    by hands.  A YOLO segmentation model can be used by the application as a
    fallback for more difficult boards.
    """
    import cv2

    if frame.ndim not in {2, 3}:
        raise ValueError("frame must be a grayscale or colour image")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    scale = min(1.0, 640 / max(gray.shape[:2]))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCornersSB(gray, (7, 7), flags)
    if not found or corners is None:
        return None
    return extrapolate_board_corners(corners / scale)


def detect_green_screen_board(frame: np.ndarray) -> np.ndarray | None:
    """Find the green/cream 2D board by its green square regions, quickly.

    The convex hull of green squares reaches all four outer corners of the
    alternating grid. This path is intended for the supplied screen template.
    """
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, 480 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (32, 35, 30), (95, 255, 240))
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eligible = [contour for contour in contours if cv2.contourArea(contour) > small.shape[0] * small.shape[1] * 0.001]
    if not eligible or sum(cv2.contourArea(contour) for contour in eligible) < small.shape[0] * small.shape[1] * 0.08:
        return None
    hull = cv2.convexHull(np.concatenate(eligible))
    rectangle = cv2.minAreaRect(hull)
    (_, _), (rw, rh), _ = rectangle
    if min(rw, rh) < min(small.shape[:2]) * 0.25 or max(rw, rh) / min(rw, rh) > 1.8:
        return None
    points = cv2.boxPoints(rectangle)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
    ordered = points[[np.argmin(sums), np.argmin(differences), np.argmax(sums), np.argmax(differences)]]
    if len(np.unique(ordered, axis=0)) != 4:
        return None
    projected = cv2.warpPerspective(green, cv2.getPerspectiveTransform(
        ordered.astype(np.float32), board_destination(320)), (320, 320))
    matches = 0
    for row in range(2, 6):
        for col in range(8):
            patch = projected[row * 40 + 16:row * 40 + 24, col * 40 + 16:col * 40 + 24]
            is_green = np.mean(patch) > 127
            matches += is_green == ((row + col) % 2 == 1)
    if matches < 26:
        return None
    return (ordered / scale).astype(np.float32)


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


def piece_symbol_from_label(label: str) -> str | None:
    """Convert common YOLO class names to one FEN piece symbol.

    Checkpoints published by different projects use labels such as
    ``w-pawn``, ``white_pawn`` or ``black king``.  This normalizes those
    harmless formatting differences while rejecting labels without a colour.
    """
    normalized = "".join(character for character in label.lower() if character.isalnum())
    return _PIECE_SYMBOLS.get(normalized)


def piece_description_from_label(label: str) -> str | None:
    """Return a concise Spanish description such as ``"peón negro"``."""
    symbol = piece_symbol_from_label(label)
    if symbol is None:
        shape_names = {"pawn": "peón", "knight": "caballo", "bishop": "alfil",
                       "rook": "torre", "queen": "dama", "king": "rey"}
        return shape_names.get(label.lower().strip())
    names = {"p": "peón", "n": "caballo", "b": "alfil", "r": "torre", "q": "dama", "k": "rey"}
    colour = "blanco" if symbol.isupper() else "negro"
    return f"{names[symbol.lower()]} {colour}"


def detections_to_fen(
    detections: Sequence[PieceDetection],
    board_size: int = DEFAULT_BOARD_SIZE,
    *,
    turn: str = "w",
    castling: str = "-",
    en_passant: str = "-",
) -> str:
    """Map piece boxes on a rectified board to a complete FEN string.

    A detector can emit overlapping boxes.  The highest-confidence recognised
    box wins per square, which avoids producing impossible multiple occupants.
    Move counters and historical rights cannot be seen in a photo; callers
    must preserve them from the game record through ``turn``, ``castling`` and
    ``en_passant``.
    """
    if board_size < 8 or board_size % 8:
        raise ValueError("board_size must be divisible by 8 and at least 8")
    if turn not in {"w", "b"}:
        raise ValueError("turn must be 'w' or 'b'")
    if castling != "-" and any(symbol not in "KQkq" for symbol in castling):
        raise ValueError("castling must be '-' or a combination of KQkq")

    side = board_size / 8
    occupied: dict[tuple[int, int], tuple[float, str]] = {}
    for detection in detections:
        symbol = piece_symbol_from_label(detection.label)
        if symbol is None:
            continue
        x1, y1, x2, y2 = detection.xyxy
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        column, row = int(center_x // side), int(center_y // side)
        if not (0 <= column < 8 and 0 <= row < 8):
            continue
        key = (row, column)
        existing = occupied.get(key)
        if existing is None or detection.confidence > existing[0]:
            occupied[key] = (detection.confidence, symbol)

    ranks: list[str] = []
    for row in range(8):
        empty = 0
        rank: list[str] = []
        for column in range(8):
            piece = occupied.get((row, column))
            if piece is None:
                empty += 1
            else:
                if empty:
                    rank.append(str(empty))
                    empty = 0
                rank.append(piece[1])
        if empty:
            rank.append(str(empty))
        ranks.append("".join(rank))
    return f"{'/'.join(ranks)} {turn} {castling} {en_passant} 0 1"


class FenStabilizer:
    """Publish a FEN only after it is unchanged for several video frames."""

    def __init__(self, required_frames: int = 4) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        self.required_frames = required_frames
        self._candidate: str | None = None
        self._count = 0
        self._published: str | None = None

    def update(self, fen: str) -> str | None:
        if fen == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = fen, 1
        if self._count >= self.required_frames and fen != self._published:
            self._published = fen
            return fen
        return None


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
