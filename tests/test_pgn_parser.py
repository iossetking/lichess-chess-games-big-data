import json

import polars as pl

from lichess_pipeline.pgn_parser import (
    clean_games_df,
    extract_positions_df,
    parse_eval_line,
    parse_pgn_game,
)

RAW_GAME_WITH_COMMENTS = """[Event "Rated Blitz game"]
[Site "https://lichess.org/xyz98765"]
[White "playerA"]
[Black "playerB"]
[Result "1-0"]
[UTCDate "2025.01.01"]
[UTCTime "12:00:00"]
[WhiteElo "1950"]
[BlackElo "1880"]
[WhiteRatingDiff "+5"]
[BlackRatingDiff "-5"]
[ECO "C50"]
[Opening "Italian Game"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 { [%eval 0.18] [%clk 0:03:00] } 1... e5 { [%eval 0.22] [%clk 0:03:00] } 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 1-0
"""


def test_parse_pgn_game_strips_comments_and_counts_plies():
    row = parse_pgn_game(RAW_GAME_WITH_COMMENTS)
    assert row is not None
    assert row["game_id"] == "xyz98765"
    assert row["WhiteElo"] == 1950
    assert row["BlackElo"] == 1880
    assert row["num_plies"] == 12
    assert "[%clk" not in str(row["movetext"])


def test_clean_games_df_filters_elo_bullet_and_termination():
    good = parse_pgn_game(RAW_GAME_WITH_COMMENTS)
    bullet = {**good, "game_id": "b1", "Event": "Rated Bullet game"}
    low_elo = {**good, "game_id": "l1", "WhiteElo": 1400}
    forfeit = {**good, "game_id": "f1", "Termination": "Time forfeit"}
    df = pl.DataFrame([good, bullet, low_elo, forfeit])
    cleaned = clean_games_df(df, min_elo=1800, min_plies=10)
    assert cleaned["game_id"].to_list() == ["xyz98765"]


def test_extract_positions_df_replays_legal_moves():
    df = pl.DataFrame([parse_pgn_game(RAW_GAME_WITH_COMMENTS)])
    positions = extract_positions_df(df, max_positions=5)
    assert len(positions) == 5
    assert positions["move_uci"].to_list()[:2] == ["e2e4", "e7e5"]
    assert positions["turn"].to_list()[:2] == ["w", "b"]


def test_parse_eval_line_selects_max_depth_and_handles_missing_pvs():
    valid_json = json.dumps(
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "evals": [
                {"knodes": 100, "depth": 18, "pvs": [{"cp": 25, "line": "c7c5 g1f3"}]},
                {"knodes": 500, "depth": 24, "pvs": [{"cp": 20, "line": "e7e5 g1f3"}]},
            ],
        }
    )
    parsed = parse_eval_line(valid_json, min_depth=20)
    assert parsed is not None
    assert parsed["depth"] == 24
    assert parsed["best_move"] == "e7e5"
    assert parsed["cp"] == 20
    assert parse_eval_line('{"fen": "invalid", "evals": []}', min_depth=20) is None
    assert parse_eval_line("not json", min_depth=20) is None


def test_parse_eval_line_canonicalizes_chess960_castling():
    castling = json.dumps(
        {
            "fen": "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "evals": [{"knodes": 10, "depth": 30, "pvs": [{"cp": 10, "line": "e8h8"}]}],
        }
    )
    parsed = parse_eval_line(castling, min_depth=20)
    assert parsed is not None
    assert parsed["best_move"] == "e8g8"


def test_parse_eval_line_returns_none_for_structurally_malformed_entries():
    def payload(evals):
        return json.dumps({"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "evals": evals})

    assert parse_eval_line(payload([None]), min_depth=20) is None
    assert parse_eval_line(payload([{"depth": 30, "pvs": []}]), min_depth=20) is None
    assert parse_eval_line(payload([{"depth": 30}]), min_depth=20) is None
    assert parse_eval_line(payload([{"depth": 30, "pvs": [None]}]), min_depth=20) is None
    assert parse_eval_line(payload([{"depth": 30, "pvs": [{"cp": 5}]}]), min_depth=20) is None
    assert parse_eval_line(payload([{"depth": 30, "pvs": [{"line": "0000"}]}]), min_depth=20) is None
