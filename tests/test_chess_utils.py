import chess

from lichess_pipeline.chess_utils import (
    canonical_legal_uci,
    first_move_is_legal,
    normalize_fen,
    phase_from_piece_count,
    piece_count_from_fen,
)


def test_normalize_fen_keeps_first_four_fields():
    assert normalize_fen(chess.STARTING_FEN) == (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    )
    assert normalize_fen("  extra   spaces  here  ") == "extra spaces here"


def test_piece_count_from_fen():
    assert piece_count_from_fen(chess.STARTING_FEN) == 32
    assert piece_count_from_fen("8/8/8/8/8/8/4k3/4K3 w - - 0 1") == 2


def test_phase_from_piece_count_thresholds():
    assert phase_from_piece_count(32) == "opening"
    assert phase_from_piece_count(29) == "opening"
    assert phase_from_piece_count(28) == "middlegame"
    assert phase_from_piece_count(15) == "middlegame"
    assert phase_from_piece_count(14) == "endgame"


def test_first_move_is_legal_accepts_legal_and_rejects_illegal():
    assert first_move_is_legal(chess.STARTING_FEN, "e2e4") is True
    assert first_move_is_legal(chess.STARTING_FEN, "e2e5") is False
    assert first_move_is_legal("not a fen", "e2e4") is False
    assert first_move_is_legal(chess.STARTING_FEN, "") is False


def test_first_move_is_legal_accepts_standard_and_chess960_castling():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    assert first_move_is_legal(fen, "e1g1") is True
    assert first_move_is_legal(fen, "e1h1") is True
    assert canonical_legal_uci(fen, "e1h1") == "e1g1"
    assert canonical_legal_uci(fen, "e1g1") == "e1g1"


def test_first_move_is_legal_rejects_null_and_illegal_moves():
    assert first_move_is_legal(chess.STARTING_FEN, "0000") is False
    assert canonical_legal_uci(chess.STARTING_FEN, "0000") is None
    assert canonical_legal_uci("not a fen", "e2e4") is None
