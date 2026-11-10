"""Board encoding, move indexing and legal-move masking.

The model contract is a single FEN string in and a single UCI move string out.
Between the two, a position becomes a ``(13, 8, 8)`` float tensor and a move
becomes one of 4096 ``from_square * 64 + to_square`` classes. Masking that
classification with the position's legal moves means the model can never
return an illegal move, only a badly ranked one.

Board orientation follows the natural tensor layout: row 0 is rank 8 and row 7
is rank 1, matching how a FEN string is read left to right, top to bottom.
"""

from __future__ import annotations

import chess
import numpy as np

BOARD_SIZE = 8
NUM_SQUARES = 64
NUM_MOVE_CLASSES = NUM_SQUARES * NUM_SQUARES  # 4096

# Planes 0..5 are white pieces, planes 6..11 the same pieces in black,
# plane 12 marks the side to move.
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
SIDE_TO_MOVE_PLANE = 12
NUM_PLANES = 13
PROMOTION_RANKS = (0, 7)  # tensor rows: rank 8 and rank 1
DEFAULT_PROMOTION = "q"


def encode_fen(fen: str) -> np.ndarray:
    """Encode a FEN as a ``(13, 8, 8)`` float32 tensor."""
    board = chess.Board(fen)
    tensor = np.zeros((NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = BOARD_SIZE - 1 - chess.square_rank(square)
        column = chess.square_file(square)
        tensor[PIECE_PLANES[piece.symbol()], row, column] = 1.0
    if board.turn == chess.WHITE:
        tensor[SIDE_TO_MOVE_PLANE, :, :] = 1.0
    return tensor


def move_to_index(move_uci: str) -> int:
    """Map a UCI move to its 0..4095 class (promotion piece ignored)."""
    from_square = chess.parse_square(move_uci[:2])
    to_square = chess.parse_square(move_uci[2:4])
    return from_square * NUM_SQUARES + to_square


def index_to_move(index: int, board: chess.Board | None = None) -> str:
    """Map a class back to UCI, appending ``q`` for back-rank pawn pushes.

    The 64x64 encoding cannot tell a promotion piece apart, so when a board is
    given and a pawn moves onto the first or last rank the move is completed
    with the queen promotion, which is what the notebooks default to.
    """
    if not 0 <= index < NUM_MOVE_CLASSES:
        raise ValueError(f"move index out of range: {index}")
    from_square = index // NUM_SQUARES
    to_square = index % NUM_SQUARES
    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
    if board is None:
        return move_uci
    piece = board.piece_at(from_square)
    target_row = BOARD_SIZE - 1 - chess.square_rank(to_square)
    if piece is not None and piece.piece_type == chess.PAWN and target_row in PROMOTION_RANKS:
        return move_uci + DEFAULT_PROMOTION
    return move_uci


def legal_move_mask(fen: str) -> np.ndarray:
    """Boolean ``(4096,)`` mask of the moves that are legal in ``fen``."""
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=bool)
    for move in chess.Board(fen).legal_moves:
        mask[move_to_index(move.uci())] = True
    return mask


def legal_move_indices(fen: str) -> np.ndarray:
    """Sorted integer indices of the legal moves in ``fen``."""
    return np.flatnonzero(legal_move_mask(fen))
