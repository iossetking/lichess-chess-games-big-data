"""Small pure helpers shared by the exploration, cleaning and label pipeline.

FEN convention used everywhere in this project: a "normalized FEN" keeps only
the first four fields (placement, side to move, castling, en passant) and drops
the move counters. Positions are always exchanged as complete FEN strings, so
the side to move, castling rights and en passant travel with the board.

Lichess writes the en passant square after every double pawn push, whereas
python-chess only keeps it when a capture is legal. Positions are therefore
normalized through ``chess.Board(...).fen(en_passant="legal")`` when they come
from a replay; ``normalize_fen`` only truncates and is used for cheap keys.
"""

from __future__ import annotations

import chess

# Phase thresholds on the total number of pieces (kings and pawns included).
OPENING_MIN_PIECES = 29  # >= 29 pieces: opening (at most 3 pieces traded)
MIDDLEGAME_MIN_PIECES = 15  # 15..28 pieces: middlegame; <= 14: endgame


def normalize_fen(fen: str) -> str:
    """Keep the first four FEN fields; no validation, no rewriting."""
    return " ".join(fen.split()[:4])


def side_to_move(fen: str) -> str:
    """Return the FEN side-to-move field (``"w"`` or ``"b"``)."""
    return fen.split()[1]


def piece_count_from_fen(fen: str) -> int:
    """Number of pieces on the board, using only the placement field."""
    placement = fen.split()[0]
    return sum(1 for char in placement if char.isalpha())


def phase_from_piece_count(count: int) -> str:
    """Game phase label from the piece count (see module thresholds)."""
    if count >= OPENING_MIN_PIECES:
        return "opening"
    if count >= MIDDLEGAME_MIN_PIECES:
        return "middlegame"
    return "endgame"


def phase_from_fen(fen: str) -> str:
    """Game phase label of a FEN, from its piece count."""
    return phase_from_piece_count(piece_count_from_fen(fen))


def canonical_legal_uci(fen: str, move_uci: str) -> str | None:
    """Return ``move_uci`` in standard UCI form when legal, else ``None``.

    The official evaluation files write the principal variation in
    ``UCI_Chess960``, so castling can appear as ``e1h1``. This normalizes it to
    the standard ``e1g1`` that the 64x64 move encoding and legal-move mask use.
    Null moves (``0000``) and moves that are not legal are rejected.
    """
    if not fen or not move_uci:
        return None
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return None
        move = board.parse_uci(move_uci)
        if move not in board.legal_moves:
            return None
        return board.uci(move)
    except ValueError:
        return None


def first_move_is_legal(fen: str, move_uci: str) -> bool:
    """True when ``fen`` is a valid position and ``move_uci`` is legal in it."""
    return canonical_legal_uci(fen, move_uci) is not None
