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

    Clusters green chessboard squares, fits the board perimeter, and verifies
    the alternating 8x8 pattern using square corners so pieces in square
    centres and bright screen backlights do not break detection.
    """
    import cv2

    height, width = frame.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Exclude low-saturation cream squares (S~30) while keeping backlit screens (V up to 255).
    green = cv2.inRange(hsv, (34, 40, 25), (98, 255, 255))
    frame_area = small.shape[0] * small.shape[1]
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eligible = [contour for contour in contours if cv2.contourArea(contour) > frame_area * 0.0003]
    if not eligible:
        return None

    median_area = float(np.median([cv2.contourArea(contour) for contour in eligible]))
    side = max(3, round(float(np.sqrt(median_area))))
    ksize = max(5, min(31, int(side * 0.6) | 1))
    dilated = cv2.dilate(green, np.ones((ksize, ksize), np.uint8), iterations=1)
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(dilated)
    if num_labels <= 1:
        return None
    counts = np.bincount(labels[green > 0], minlength=num_labels)
    counts[0] = 0
    best_label = int(np.argmax(counts))
    if counts[best_label] == 0:
        return None

    cluster_mask = np.where(labels == best_label, np.uint8(255), np.uint8(0))
    green_cluster = cv2.bitwise_and(green, cluster_mask)
    if cv2.countNonZero(green_cluster) < frame_area * 0.015:
        return None

    cluster_contours, _ = cv2.findContours(green_cluster, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cluster_contours:
        return None
    hull = cv2.convexHull(np.concatenate(cluster_contours))
    rectangle = cv2.minAreaRect(hull)
    (_, _), (rw, rh), _ = rectangle
    if min(rw, rh) < min(small.shape[:2]) * 0.16 or max(rw, rh) / min(rw, rh) > 1.8:
        return None

    points = cv2.boxPoints(rectangle)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
    ordered = points[[np.argmin(sums), np.argmin(differences), np.argmax(sums), np.argmax(differences)]]
    if len(np.unique(ordered, axis=0)) != 4:
        return None

    projected = cv2.warpPerspective(
        green_cluster,
        cv2.getPerspectiveTransform(ordered.astype(np.float32), board_destination(320)),
        (320, 320),
    )
    matches = 0
    for row in range(8):
        r = row * 40
        for col in range(8):
            c = col * 40
            corners_mean = (
                float(np.mean(projected[r + 3:r + 9, c + 3:c + 9]))
                + float(np.mean(projected[r + 3:r + 9, c + 31:c + 37]))
                + float(np.mean(projected[r + 31:r + 37, c + 3:c + 9]))
                + float(np.mean(projected[r + 31:r + 37, c + 31:c + 37]))
            ) / 4.0
            is_green = corners_mean > 100
            matches += is_green == ((row + col) % 2 == 1)
    if matches < 46:
        return None
    return (ordered / scale).astype(np.float32)


def _find_best_grid_run(profile: np.ndarray, min_step: float, max_step: float) -> tuple[float, float, float] | None:
    """Locate 7 evenly-spaced internal grid line peaks along a 1D edge profile.

    Returns ``(origin, step, score)`` where ``origin`` is the outer board start
    coordinate (one cell before the first internal line) and ``step`` is the
    square side in pixels, or ``None`` if no strong 8x8 periodicity is found.
    """
    length = len(profile)
    if length < 64:
        return None
    norm = profile.astype(np.float32)
    mean, std = float(norm.mean()), float(norm.std())
    if std < 1e-3:
        return None
    norm = np.clip((norm - mean) / std, 0.0, 6.0)
    # Small smoothing tolerance for 1-2px alignment jitter.
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25], dtype=np.float32)
    smooth = np.convolve(norm, kernel, mode="same")

    best: tuple[float, float, float] | None = None
    step_lo = max(8, int(np.floor(min_step)))
    step_hi = min(length // 8, int(np.ceil(max_step)))
    if step_lo > step_hi:
        return None

    for step_int in range(step_lo, step_hi + 1):
        step = float(step_int)
        span = 8.0 * step
        max_start = int(length - 1 - span)
        if max_start < 0:
            continue
        # Score the 7 internal grid lines plus both outer borders (9 lines total)
        # so the board cannot shift by one square onto uniform background.
        inner_offsets = np.arange(1, 8, dtype=np.float32) * step
        outer_offsets = np.array([0.0, span], dtype=np.float32)
        starts = np.arange(0, max_start + 1, dtype=np.float32)
        inner_vals = smooth[np.rint(starts[:, None] + inner_offsets[None, :]).astype(np.int32)]
        outer_vals = smooth[np.rint(starts[:, None] + outer_offsets[None, :]).astype(np.int32)]
        scores = inner_vals.mean(axis=1) * 0.5 + inner_vals.min(axis=1) * 0.35 + outer_vals.min(axis=1) * 0.15
        best_idx = int(np.argmax(scores))
        score = float(scores[best_idx])
        if best is None or score > best[2]:
            best = (float(starts[best_idx]), step, score)
    return best


def _verify_checkerboard_parity(gray_board: np.ndarray) -> float:
    """Return fraction of 64 squares agreeing with an alternating light/dark grid."""
    side = gray_board.shape[0] // 8
    margin = max(2, side // 10)
    patch = max(3, side // 6)
    even_vals, odd_vals = [], []
    cell_vals = np.empty((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            y0, x0 = r * side, c * side
            samples = [
                float(np.median(gray_board[y0 + margin:y0 + margin + patch, x0 + margin:x0 + margin + patch])),
                float(np.median(gray_board[y0 + margin:y0 + margin + patch, x0 + side - margin - patch:x0 + side - margin])),
                float(np.median(gray_board[y0 + side - margin - patch:y0 + side - margin, x0 + margin:x0 + margin + patch])),
                float(np.median(gray_board[y0 + side - margin - patch:y0 + side - margin, x0 + side - margin - patch:x0 + side - margin])),
            ]
            # Use the median of the 4 square corners (robust to piece corners or coordinates).
            val = float(np.median(samples))
            cell_vals[r, c] = val
            if (r + c) % 2 == 0:
                even_vals.append(val)
            else:
                odd_vals.append(val)
    med_even, med_odd = float(np.median(even_vals)), float(np.median(odd_vals))
    if abs(med_even - med_odd) < 12.0:
        return 0.0
    threshold = 0.5 * (med_even + med_odd)
    even_is_lighter = med_even > med_odd
    matches = 0
    for r in range(8):
        for c in range(8):
            is_light = cell_vals[r, c] > threshold
            expected = even_is_lighter if (r + c) % 2 == 0 else (not even_is_lighter)
            matches += int(is_light == expected)
    return matches / 64.0


def detect_screen_board_by_grid(frame: np.ndarray) -> np.ndarray | None:
    """Detect an upright 2D chessboard of *any* colour theme (Lichess, Chess.com, etc.).

    Works on direct screenshots, browser windows, and level webcam views of a
    screen by locating the 7x7 internal grid edges and confirming alternating
    square-corner brightness across the 64 squares.
    """
    import cv2

    if frame.ndim not in {2, 3}:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    sh, sw = small.shape[:2]

    gx = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3))
    # Suppress piece-interior edges slightly by preferring straight continuous lines.
    x_profile = gx.mean(axis=0)
    y_profile = gy.mean(axis=1)

    min_step = min(sh, sw) * 0.22 / 8.0
    max_step = min(sh, sw) / 8.0
    x_run = _find_best_grid_run(x_profile, min_step, max_step)
    y_run = _find_best_grid_run(y_profile, min_step, max_step)
    if x_run is None or y_run is None:
        return None

    x0, sx, _ = x_run
    y0, sy, _ = y_run
    # Refine each axis inside the other axis's band so browser sidebars don't confuse it.
    ry1, ry2 = max(0, int(y0)), min(sh, int(np.ceil(y0 + 8 * sy)))
    rx1, rx2 = max(0, int(x0)), min(sw, int(np.ceil(x0 + 8 * sx)))
    if ry2 - ry1 >= 64 and rx2 - rx1 >= 64:
        x_refined = _find_best_grid_run(gx[ry1:ry2, :].mean(axis=0), min_step, max_step)
        y_refined = _find_best_grid_run(gy[:, rx1:rx2].mean(axis=1), min_step, max_step)
        if x_refined is not None:
            x0, sx, _ = x_refined
        if y_refined is not None:
            y0, sy, _ = y_refined

    if sx <= 0 or sy <= 0 or max(sx, sy) / min(sx, sy) > 1.25:
        return None
    x1, y1 = x0 + 8.0 * sx, y0 + 8.0 * sy
    if x1 > sw + 2 or y1 > sh + 2:
        return None

    corners_small = np.array(
        [[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]],
        dtype=np.float32,
    )
    warped = cv2.warpPerspective(
        small, cv2.getPerspectiveTransform(corners_small, board_destination(320)), (320, 320)
    )
    if _verify_checkerboard_parity(warped) < 0.78:
        return None
    return (corners_small / scale).astype(np.float32)


def detect_any_screen_board(frame: np.ndarray) -> np.ndarray | None:
    """Locate a 2D chessboard on a screen across colour themes and mild tilt."""
    corners = detect_green_screen_board(frame)
    if corners is not None:
        return corners
    return detect_screen_board_by_grid(frame)


def sanitize_fen_castling(fen: str) -> str:
    """Drop castling flags that are physically impossible for the detected king/rook squares."""
    parts = fen.split()
    if len(parts) < 3 or parts[2] == "-":
        return fen
    import chess

    try:
        board = chess.Board(fen)
    except ValueError:
        return fen
    valid_flags = ""
    if "K" in parts[2] and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
        valid_flags += "K"
    if "Q" in parts[2] and board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
        valid_flags += "Q"
    if "k" in parts[2] and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
        valid_flags += "k"
    if "q" in parts[2] and board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
        valid_flags += "q"
    parts[2] = valid_flags or "-"
    return " ".join(parts)


def validate_fen_position(fen: str) -> tuple[bool, str]:
    """Validate a FEN with ``python-chess`` and return ``(is_valid, spanish_reason)``."""
    import chess

    try:
        board = chess.Board(fen)
    except ValueError as error:
        return False, f"FEN mal formado: {error}"
    if board.is_valid():
        return True, "OK"

    status = board.status()
    reasons: list[str] = []
    if status & chess.STATUS_NO_WHITE_KING:
        reasons.append("falta rey blanco")
    if status & chess.STATUS_NO_BLACK_KING:
        reasons.append("falta rey negro")
    if status & chess.STATUS_TOO_MANY_KINGS:
        reasons.append("mas de un rey del mismo color")
    if status & chess.STATUS_PAWNS_ON_BACKRANK:
        reasons.append("peon en fila 1 u 8")
    if status & chess.STATUS_OPPOSITE_CHECK:
        reasons.append("rival en jaque (pulsa [t] para cambiar turno)")
    if status & chess.STATUS_TOO_MANY_WHITE_PAWNS:
        reasons.append("mas de 8 peones blancos")
    if status & chess.STATUS_TOO_MANY_BLACK_PAWNS:
        reasons.append("mas de 8 peones negros")
    if status & chess.STATUS_TOO_MANY_WHITE_PIECES:
        reasons.append("demasiadas piezas blancas")
    if status & chess.STATUS_TOO_MANY_BLACK_PIECES:
        reasons.append("demasiadas piezas negras")
    if not reasons:
        reasons.append("posicion ilegal")
    return False, "; ".join(reasons)


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


PIECE_CLASSES = ("bishop", "king", "knight", "pawn", "queen", "rook")


def repair_king_queen_detections(detections: Sequence[PieceDetection]) -> list[PieceDetection]:
    """Enforce at most one king per colour by resolving king/queen swaps.

    Digital and physical detectors occasionally mistake a queen for a second
    king (or vice versa), which makes ``chess.Board.is_valid()`` reject the
    entire FEN.  For each colour:

    - If >= 2 kings are detected, keep the highest-confidence king and demote
      the extra king(s) to queens.
    - If 0 kings and >= 2 queens are detected, promote the lowest-confidence
      queen to a king.
    """
    repaired = list(detections)
    for king_sym, queen_sym, king_label, queen_label in (
        ("K", "Q", "white-king", "white-queen"),
        ("k", "q", "black-king", "black-queen"),
    ):
        kings = [i for i, d in enumerate(repaired) if piece_symbol_from_label(d.label) == king_sym]
        queens = [i for i, d in enumerate(repaired) if piece_symbol_from_label(d.label) == queen_sym]
        if len(kings) >= 2:
            best_king = max(kings, key=lambda idx: repaired[idx].confidence)
            for idx in kings:
                if idx != best_king:
                    det = repaired[idx]
                    repaired[idx] = PieceDetection(queen_label, det.confidence, det.xyxy, det.mask)
        elif len(kings) == 0 and len(queens) >= 2:
            least_certain_queen = min(queens, key=lambda idx: repaired[idx].confidence)
            det = repaired[least_certain_queen]
            repaired[least_certain_queen] = PieceDetection(king_label, det.confidence, det.xyxy, det.mask)
    return repaired


def crop_piece(image: np.ndarray, xyxy: Sequence[float], size: int = 128, margin: float = 1.25) -> np.ndarray:
    """Square, margin-padded crop around a detected piece, resized to ``size``.

    Shared by training and inference so the classifier sees identical framing.
    Pixels outside the frame are replicated from the border.
    """
    import cv2

    x1, y1, x2, y2 = (float(v) for v in xyxy)
    side = max(x2 - x1, y2 - y1, 4.0) * margin
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    # Map the square window onto the output with an affine warp: one call, handles borders.
    scale = size / side
    matrix = np.array([[scale, 0, size / 2 - cx * scale], [0, scale, size / 2 - cy * scale]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def fuse_piece_probabilities(detector_label: str, detector_confidence: float, classifier_probs: Sequence[float],
                             classifier_weight: float = 1.0) -> tuple[str, float]:
    """Combine the detector's label with the crop classifier's distribution.

    The detector contributes a vote of ``detector_confidence`` for its class;
    the classifier contributes its full probability vector scaled by
    ``classifier_weight``.  Returns the winning class and its normalised score.
    """
    scores = {name: classifier_weight * float(p) for name, p in zip(PIECE_CLASSES, classifier_probs)}
    if detector_label in scores:
        scores[detector_label] += detector_confidence
    label = max(scores, key=scores.get)
    return label, scores[label] / sum(scores.values())


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _box_match_score(a: Sequence[float], b: Sequence[float], min_iou: float) -> float:
    """Match score combining IoU with a center-proximity fallback for slender pieces."""
    overlap = box_iou(a, b)
    if overlap >= min_iou:
        return overlap
    # Slender pieces (bishops/pawns) can shift a few pixels with camera jitter
    # and drop below min_iou even when clearly the same physical piece.
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    ref_size = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1.0)
    dist = float(np.hypot(ax - bx, ay - by)) / ref_size
    if dist <= 0.35 and overlap > 0.0:
        return max(overlap, min_iou * (1.0 - dist))
    return 0.0


class PieceLabelSmoother:
    """Stabilise the *type* of each piece across video frames.

    A single frame can call a knight a pawn when the view is blurred or partly
    hidden.  Each physical piece is followed from frame to frame (greedy IoU +
    center-proximity matching) and the label shown is the confidence-weighted
    vote of its recent frames, so one bad frame cannot flip the label while a
    real change (a piece replaced on the same square) still wins within a few
    frames.

    ``decay`` multiplies old votes every frame (0.8 means the last ~5 frames
    dominate).  Tracks unseen for ``max_missing`` frames are dropped.  When
    ``hold_missing > 0``, established tracks (seen at least ``min_hold_hits``
    times) are carried forward for up to ``hold_missing`` missed frames to
    prevent detection blinking.
    """

    def __init__(
        self,
        decay: float = 0.8,
        min_iou: float = 0.3,
        max_missing: int = 5,
        hold_missing: int = 0,
        min_hold_hits: int = 2,
    ) -> None:
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if hold_missing < 0:
            raise ValueError("hold_missing must be zero or positive")
        if min_hold_hits < 1:
            raise ValueError("min_hold_hits must be at least 1")
        self.decay, self.min_iou, self.max_missing = decay, min_iou, max_missing
        self.hold_missing, self.min_hold_hits = hold_missing, min_hold_hits
        self._tracks: list[dict] = []

    def update(self, detections: Sequence[PieceDetection]) -> list[PieceDetection]:
        for track in self._tracks:
            track["missing"] += 1
            track["votes"] = {label: weight * self.decay for label, weight in track["votes"].items()}
        pairs = sorted(
            ((_box_match_score(track["box"], detection.xyxy, self.min_iou), t, d)
             for t, track in enumerate(self._tracks) for d, detection in enumerate(detections)),
            reverse=True,
        )
        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for score, t, d in pairs:
            if score <= 0.0:
                break
            if t in used_tracks or d in assigned:
                continue
            used_tracks.add(t)
            assigned[d] = t
        smoothed: list[PieceDetection] = []
        for d, detection in enumerate(detections):
            if d in assigned:
                track = self._tracks[assigned[d]]
            else:
                track = {"votes": {}, "box": detection.xyxy, "missing": 0, "hits": 0, "last_conf": 0.0}
                self._tracks.append(track)
            track["box"], track["missing"] = detection.xyxy, 0
            track["hits"] += 1
            track["votes"][detection.label] = track["votes"].get(detection.label, 0.0) + detection.confidence
            label = max(track["votes"], key=track["votes"].get)
            share = track["votes"][label] / sum(track["votes"].values())
            # Report the winning label with a confidence that reflects agreement.
            confidence = detection.confidence if label == detection.label else detection.confidence * share
            track["last_conf"] = confidence
            smoothed.append(PieceDetection(label, confidence, detection.xyxy, detection.mask))

        if self.hold_missing > 0:
            for t, track in enumerate(self._tracks):
                if t in used_tracks:
                    continue
                if not (1 <= track["missing"] <= self.hold_missing and track["hits"] >= self.min_hold_hits):
                    continue
                if not track["votes"]:
                    continue
                if any(box_iou(track["box"], det.xyxy) >= self.min_iou for det in detections):
                    continue
                label = max(track["votes"], key=track["votes"].get)
                held_conf = max(0.01, track["last_conf"] * (self.decay ** track["missing"]))
                smoothed.append(PieceDetection(label, held_conf, track["box"], None))

        self._tracks = [track for track in self._tracks if track["missing"] <= self.max_missing]
        return smoothed


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
