"""Network-free tests for chess_utils and the generic dataset scan."""

import polars as pl
import pytest

from lichess_pipeline.chess_utils import (
    canonical_fen,
    first_move_is_legal,
    has_clock_expr,
    has_eval_expr,
    match_key,
    match_key_expr,
    normalize_fen,
    phase_expr,
    piece_count_expr,
    puzzle_position,
)
from lichess_pipeline.data_ingestion import (
    EVALUATIONS_SCHEMA,
    scan_hf_dataset,
    validate_schema,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_normalize_fen_keeps_four_fields() -> None:
    assert (
        normalize_fen(START) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"
    )
    assert normalize_fen("8/8/8/8/8/8/8/K6k w - -") == "8/8/8/8/8/8/8/K6k w - -"


def test_canonical_fen_drops_illegal_en_passant() -> None:
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3"
    assert canonical_fen(fen).endswith("KQkq -")


def test_piece_count_and_phase() -> None:
    frame = pl.DataFrame(
        {"fen": [START, "8/8/8/8/8/8/8/K6k w - -", "8/8/8/8/8/8/PPPP4/KQ4kq w - -"]}
    )
    out = frame.select(piece_count_expr("fen").alias("n"), phase_expr("fen").alias("p"))
    assert out["n"].to_list() == [32, 2, 8]
    assert out["p"].to_list() == ["opening", "endgame", "endgame"]


def test_phase_opening_and_middlegame() -> None:
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq -"
    assert (
        pl.DataFrame({"f": [fen]}).select(phase_expr("f").alias("p"))["p"][0]
        == "opening"
    )
    fen2 = "r4rk1/pp3ppp/2n5/8/8/2N5/PP3PPP/R4RK1 w - -"
    assert (
        pl.DataFrame({"f": [fen2]}).select(phase_expr("f").alias("p"))["p"][0]
        == "middlegame"
    )


def test_has_eval_and_clock() -> None:
    frame = pl.DataFrame(
        {"m": ["1. e4 { [%eval 0.3] [%clk 0:03:00] }", "1. e4 e5", None]}
    )
    out = frame.select(has_eval_expr("m").alias("e"), has_clock_expr("m").alias("c"))
    assert out["e"].to_list() == [True, False, None]
    assert out["c"].to_list() == [True, False, None]


def test_puzzle_position_applies_first_move() -> None:
    assert puzzle_position(START, "e2e4 e7e5") == (
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -"
    )


def test_puzzle_position_rejects_illegal_move() -> None:
    with pytest.raises(ValueError):
        puzzle_position(START, "e2e5 e7e5")


def test_first_move_is_legal() -> None:
    assert first_move_is_legal(START, "e2e4 e7e5")
    assert not first_move_is_legal(START, "e2e5")
    assert not first_move_is_legal("not a fen", "e2e4")
    assert not first_move_is_legal(START, "")


def test_validate_schema_with_custom_expected() -> None:
    schema = pl.Schema({"fen": pl.String(), "depth": pl.Int64()})
    with pytest.raises(ValueError, match="dtype mismatches"):
        validate_schema(schema, ["fen", "depth"], EVALUATIONS_SCHEMA)
    validate_schema(schema, ["fen"], EVALUATIONS_SCHEMA)


def test_scan_hf_dataset_rejects_unknown_columns_and_empty_files() -> None:
    with pytest.raises(ValueError, match="Unknown columns"):
        scan_hf_dataset("a/b", ["x.parquet"], "rev", ["nope"], EVALUATIONS_SCHEMA)
    with pytest.raises(ValueError, match="No files"):
        scan_hf_dataset("a/b", [], "rev", ["fen"], EVALUATIONS_SCHEMA)


def test_match_key_ignores_en_passant_on_both_sides() -> None:
    # Lichess writes e3 after 1.e4; python-chess drops it (no legal capture).
    lichess_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3"
    puzzle_fen = puzzle_position(START, "e2e4 e7e5")
    assert match_key(lichess_fen) == match_key(puzzle_fen)
    out = pl.DataFrame({"f": [lichess_fen, puzzle_fen]}).select(match_key_expr("f"))
    assert out["f"].to_list() == [match_key(lichess_fen)] * 2


def test_phase_boundaries() -> None:
    # 29 pieces -> opening, 28 -> middlegame, 15 -> middlegame, 14 -> endgame.
    def fen_with(n: int) -> str:
        pieces = "K" + "k" + "P" * (n - 2)
        rows = [pieces[i : i + 8] for i in range(0, len(pieces), 8)]
        rows = [r + (str(8 - len(r)) if len(r) < 8 else "") for r in rows]
        rows += ["8"] * (8 - len(rows))
        return "/".join(rows) + " w - -"

    frame = pl.DataFrame({"f": [fen_with(n) for n in (29, 28, 15, 14)]})
    out = frame.select(piece_count_expr("f").alias("n"), phase_expr("f").alias("p"))
    assert out["n"].to_list() == [29, 28, 15, 14]
    assert out["p"].to_list() == ["opening", "middlegame", "middlegame", "endgame"]


def test_first_move_is_legal_edge_cases() -> None:
    castle = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq -"
    assert first_move_is_legal(castle, "e1g1")
    assert not first_move_is_legal(castle, "e1h1")  # non-standard UCI castling
    promo = "8/P7/8/8/8/8/8/K6k w - -"
    assert first_move_is_legal(promo, "a7a8q")
    assert not first_move_is_legal(promo, "a7a8")
    assert not first_move_is_legal(START, None)
    assert not first_move_is_legal(None, "e2e4")
    assert not first_move_is_legal("8/8/8/8/8/8/8/8 w - -", "a1a2")  # no kings


def test_puzzle_position_rejects_empty_moves() -> None:
    with pytest.raises(ValueError, match="empty"):
        puzzle_position(START, "")
    with pytest.raises(ValueError, match="empty"):
        puzzle_position(START, None)
