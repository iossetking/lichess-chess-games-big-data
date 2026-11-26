"""Board encoding, move indexing and legal-move masking.

The model contract is a single FEN string in and a single UCI move string out.
Between the two, a position becomes an ``(18, 8, 8)`` float tensor and a move
becomes one of 4096 ``from_square * 64 + to_square`` classes. Masking that
classification with the position's legal moves means the model can never
return an illegal move, only a badly ranked one.

**Side-to-move canonicalisation.** The board is always encoded from the
perspective of the player who has to move: when Black is to move the position
is mirrored (ranks flipped, colours swapped) and so is the move, exactly like
Maia does. The same patterns therefore look identical for both colours, which
doubles the effective amount of training signal.

Board orientation follows the natural tensor layout: row 0 is rank 8 and row 7
is rank 1, matching how a FEN string is read left to right, top to bottom.

The 18 planes are:

* ``0..5``: the side to move's pawn, knight, bishop, rook, queen and king.
* ``6..11``: the same six piece types for the opponent.
* ``12..15``: castling availability (side to move K, Q; opponent K, Q).
* ``16``: the en-passant target square, if any.
* ``17``: which side the original position had to move (1 = White).

``encode_fen_legacy`` reproduces the original 13-plane, un-mirrored encoding so
checkpoints trained before the upgrade keep working.
"""

from __future__ import annotations

import chess
import numpy as np

BOARD_SIZE = 8
NUM_SQUARES = 64
NUM_MOVE_CLASSES = NUM_SQUARES * NUM_SQUARES  # 4096

# Planes 0..5 are the side to move, planes 6..11 the opponent.
PIECE_PLANES: dict[str, int] = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}
CASTLING_PLANES = {"K": 12, "Q": 13, "k": 14, "q": 15}
EN_PASSANT_PLANE = 16
SIDE_TO_MOVE_PLANE = 17
NUM_PLANES = 18

# The pre-upgrade layout, kept for legacy checkpoints.
LEGACY_PIECE_PLANES: dict[str, int] = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}
LEGACY_SIDE_TO_MOVE_PLANE = 12
LEGACY_NUM_PLANES = 13

DEFAULT_PROMOTION = "q"


def mirror_square(square: int) -> int:
    """Flip a square vertically (``a1 <-> a8``), used to canonicalise Black."""
    return square ^ 56


def fullmove_rating_square(square: int) -> tuple[int, int]:
    """Return the ``(row, column)`` tensor position of a chess square."""
    return BOARD_SIZE - 1 - chess.square_rank(square), chess.square_file(square)


def encode_fen(fen: str) -> np.ndarray:
    """Encode a FEN as a side-to-move canonical ``(18, 8, 8)`` tensor."""
    board = chess.Board(fen)
    white_to_move = board.turn == chess.WHITE
    canonical = board if white_to_move else board.mirror()
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square, piece in canonical.piece_map().items():
        row, column = fullmove_rating_square(square)
        tensor[PIECE_PLANES[piece.symbol()], row, column] = 1.0
    for plane, right in (
        (CASTLING_PLANES["K"], chess.BB_H1),
        (CASTLING_PLANES["Q"], chess.BB_A1),
        (CASTLING_PLANES["k"], chess.BB_H8),
        (CASTLING_PLANES["q"], chess.BB_A8),
    ):
        if canonical.castling_rights & right:
            tensor[plane, :, :] = 1.0
    if canonical.ep_square is not None:
        row, column = fullmove_rating_square(canonical.ep_square)
        tensor[EN_PASSANT_PLANE, row, column] = 1.0
    if white_to_move:
        tensor[SIDE_TO_MOVE_PLANE, :, :] = 1.0
    return tensor


def encode_fen_legacy(fen: str) -> np.ndarray:
    """Encode a FEN with the original un-mirrored ``(13, 8, 8)`` layout."""
    board = chess.Board(fen)
    tensor = np.zeros((LEGACY_NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, column = fullmove_rating_square(square)
        tensor[LEGACY_PIECE_PLANES[piece.symbol()], row, column] = 1.0
    if board.turn == chess.WHITE:
        tensor[LEGACY_SIDE_TO_MOVE_PLANE, :, :] = 1.0
    return tensor


def canonical_move(fen: str, move_uci: str) -> str:
    """Return ``move_uci`` in the mirrored board of Black-to-move positions."""
    board = chess.Board(fen)
    if board.turn == chess.WHITE:
        return move_uci
    move = chess.Move.from_uci(move_uci)
    return chess.Move(
        mirror_square(move.from_square),
        mirror_square(move.to_square),
        move.promotion,
    ).uci()


def move_to_index(move_uci: str) -> int:
    """Map a UCI move to its 0..4095 class (promotion piece ignored)."""
    from_square = chess.parse_square(move_uci[:2])
    to_square = chess.parse_square(move_uci[2:4])
    return from_square * NUM_SQUARES + to_square


def index_to_move(
    index: int, board: chess.Board | None = None, *, canonical: bool = True
) -> str:
    """Map a class back to UCI, undoing the side-to-move mirroring.

    The 64x64 encoding cannot tell a promotion piece apart, so when a board is
    given and a pawn moves onto the first or last rank the move is completed
    with the queen promotion, which is what the notebooks default to.
    """
    if not 0 <= index < NUM_MOVE_CLASSES:
        raise ValueError(f"move index out of range: {index}")
    from_square = index // NUM_SQUARES
    to_square = index % NUM_SQUARES
    if canonical and board is not None and board.turn == chess.BLACK:
        from_square, to_square = mirror_square(from_square), mirror_square(to_square)
    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
    if board is None:
        return move_uci
    piece = board.piece_at(from_square)
    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and chess.square_rank(to_square) in (0, 7)
    ):
        return move_uci + DEFAULT_PROMOTION
    return move_uci


def legal_move_mask(fen: str, *, canonical: bool = True) -> np.ndarray:
    """Boolean ``(4096,)`` mask of the moves that are legal in ``fen``.

    With ``canonical`` the indices follow the mirrored board, so they line up
    with :func:`encode_fen`; pass ``canonical=False`` for legacy checkpoints.
    """
    board = chess.Board(fen)
    black_to_move = canonical and board.turn == chess.BLACK
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=bool)
    for move in board.legal_moves:
        from_square, to_square = move.from_square, move.to_square
        if black_to_move:
            from_square, to_square = (
                mirror_square(from_square),
                mirror_square(to_square),
            )
        mask[from_square * NUM_SQUARES + to_square] = True
    return mask


def legal_move_indices(fen: str, *, canonical: bool = True) -> np.ndarray:
    """Sorted integer indices of the legal moves in ``fen``."""
    return np.flatnonzero(legal_move_mask(fen, canonical=canonical))


def encode_position(fen: str, move_uci: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Encode ``(fen, move_uci)`` in a single ``chess.Board`` parse.

    Returns ``(features, target_index, packed_legal_mask)`` matching
    ``encode_fen(fen).astype(np.uint8)``,
    ``move_to_index(canonical_move(fen, move_uci))``, and
    ``np.packbits(legal_move_mask(fen))``.
    """
    board = chess.Board(fen)
    white_to_move = board.turn == chess.WHITE
    canonical = board if white_to_move else board.mirror()

    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    for square, piece in canonical.piece_map().items():
        row, column = fullmove_rating_square(square)
        tensor[PIECE_PLANES[piece.symbol()], row, column] = 1
    for plane, right in (
        (CASTLING_PLANES["K"], chess.BB_H1),
        (CASTLING_PLANES["Q"], chess.BB_A1),
        (CASTLING_PLANES["k"], chess.BB_H8),
        (CASTLING_PLANES["q"], chess.BB_A8),
    ):
        if canonical.castling_rights & right:
            tensor[plane, :, :] = 1
    if canonical.ep_square is not None:
        row, column = fullmove_rating_square(canonical.ep_square)
        tensor[EN_PASSANT_PLANE, row, column] = 1
    if white_to_move:
        tensor[SIDE_TO_MOVE_PLANE, :, :] = 1

    move = chess.Move.from_uci(move_uci)
    from_square, to_square = move.from_square, move.to_square
    if not white_to_move:
        from_square, to_square = (
            mirror_square(from_square),
            mirror_square(to_square),
        )
    target = from_square * NUM_SQUARES + to_square

    mask = np.zeros(NUM_MOVE_CLASSES, dtype=bool)
    for legal_move in board.legal_moves:
        legal_from, legal_to = legal_move.from_square, legal_move.to_square
        if not white_to_move:
            legal_from, legal_to = (
                mirror_square(legal_from),
                mirror_square(legal_to),
            )
        mask[legal_from * NUM_SQUARES + legal_to] = True

    return tensor, target, np.packbits(mask)
