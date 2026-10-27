"""Small pure helpers shared by the exploration and the label pipeline.

FEN convention used everywhere in this project: a "normalized FEN" keeps only
the first four fields (placement, side to move, castling, en passant) and drops
the move counters. The en passant field is taken from python-chess
(`board.epd()` style, i.e. the square is only kept when an en passant capture
is legal), whereas Lichess writes the square after every double pawn push.
`normalize_fen` does NOT rewrite the string it receives (it only truncates it);
`puzzle_position` returns python-chess output. To compare both sources
consistently, `canonical_fen` runs a FEN through python-chess.

To MATCH positions across datasets (dedup, removing puzzle positions from
training) use `match_key` / `match_key_expr`: they keep placement, side to move
and castling and drop en passant, so Lichess FENs and python-chess FENs give the
same key without running python-chess on hundreds of millions of rows. Dropping
en passant can only create extra matches, which is the safe side for leakage.
"""

import chess
import polars as pl

# Phase thresholds on the total number of pieces (kings and pawns included).
OPENING_MIN_PIECES = 29  # >= 29 pieces: opening (at most 3 pieces traded)
MIDDLEGAME_MIN_PIECES = 15  # 15..28 pieces: middlegame; <= 14: endgame


def normalize_fen(fen: str) -> str:
    """Keep the first four FEN fields; no validation, no rewriting."""
    return " ".join(fen.split()[:4])


def canonical_fen(fen: str) -> str:
    """Normalized FEN as printed by python-chess (en passant only if legal)."""
    return normalize_fen(chess.Board(fen).fen(en_passant="legal"))


def match_key(fen: str) -> str:
    """Position key for joins: placement, side to move and castling only."""
    return " ".join(fen.split()[:3])


def match_key_expr(fen_col: str | pl.Expr) -> pl.Expr:
    """Polars version of `match_key` (string operations only)."""
    col = pl.col(fen_col) if isinstance(fen_col, str) else fen_col
    return col.str.split(" ").list.slice(0, 3).list.join(" ")


def piece_count_expr(fen_col: str | pl.Expr) -> pl.Expr:
    """Number of pieces in the placement field, using only string operations."""
    col = pl.col(fen_col) if isinstance(fen_col, str) else fen_col
    placement = col.str.split(" ").list.first()
    return placement.str.count_matches(r"[pnbrqkPNBRQK]").cast(pl.Int8)


def phase_expr(fen_col: str | pl.Expr) -> pl.Expr:
    """Game phase label from the piece count (see module thresholds)."""
    count = piece_count_expr(fen_col)
    return (
        pl.when(count >= OPENING_MIN_PIECES)
        .then(pl.lit("opening"))
        .when(count >= MIDDLEGAME_MIN_PIECES)
        .then(pl.lit("middlegame"))
        .otherwise(pl.lit("endgame"))
    )


def has_eval_expr(movetext_col: str | pl.Expr) -> pl.Expr:
    """True when the movetext carries at least one `[%eval ...]` comment."""
    col = pl.col(movetext_col) if isinstance(movetext_col, str) else movetext_col
    return col.str.contains("[%eval", literal=True)


def has_clock_expr(movetext_col: str | pl.Expr) -> pl.Expr:
    """True when the movetext carries at least one `[%clk ...]` comment."""
    col = pl.col(movetext_col) if isinstance(movetext_col, str) else movetext_col
    return col.str.contains("[%clk", literal=True)


def puzzle_position(fen: str, moves: str | None) -> str:
    """Normalized FEN of the position the solver sees.

    Lichess puzzle FENs are before the opponent's move `Moves[0]`; this applies
    that move with python-chess and returns the canonical FEN. Raises
    ValueError on an empty move list or an illegal move.
    """
    if not moves or not moves.split():
        raise ValueError("empty move list")
    board = chess.Board(fen)
    board.push_uci(moves.split()[0])
    return normalize_fen(board.fen(en_passant="legal"))


def first_move_is_legal(fen: str | None, line: str | None) -> bool:
    """True when `fen` is a valid position and the first move of `line` is legal.

    The move must be written in standard UCI (castling as e1g1, not e1h1).
    """
    if not fen or not line or not line.split():
        return False
    try:
        board = chess.Board(fen)
        if not board.is_valid():
            return False
        first = line.split()[0]
        move = board.parse_uci(first)
        return board.uci(move) == first
    except ValueError:
        return False
