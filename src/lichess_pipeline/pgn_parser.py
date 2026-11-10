"""Stage 2: parse official Lichess PGN games and Stockfish evaluations.

Two independent readers live here, both built on the streaming helpers of
``data_ingestion.py``:

* ``parse_pgn_sample_df`` reads ``.pgn.zst`` games, keeps the columns that
  matter for modelling, strips clock/eval comments and replays every move with
  python-chess. ``extract_positions_df`` then turns those games into
  ``(fen, move_uci)`` rows, the Stage 1 training signal (human moves).
* ``load_clean_evals_df`` reads ``lichess_db_eval.jsonl.zst``, keeps the
  highest-depth evaluation of each position and returns ``(fen, best_move)``
  rows, the Stage 2 fine-tuning signal (Stockfish moves).

Nothing here writes files: notebooks decide what to persist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import chess
import polars as pl

from lichess_pipeline.chess_utils import (
    first_move_is_legal,
    normalize_fen,
    phase_from_piece_count,
    piece_count_from_fen,
    side_to_move,
)
from lichess_pipeline.data_ingestion import (
    OFFICIAL_EVALS_URL,
    games_month_url,
    stream_pgn_blocks,
    stream_zst_lines,
)

VALID_RESULTS = ("1-0", "0-1", "1/2-1/2")
DEFAULT_MIN_ELO = 1800
DEFAULT_MIN_PLIES = 10
DEFAULT_MIN_DEPTH = 20
DEFAULT_MAX_GAMES = 10_000
DEFAULT_MAX_EVALS = 50_000
DEFAULT_MAX_POSITIONS = 50_000

GAME_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "White": pl.String(),
    "Black": pl.String(),
    "Result": pl.String(),
    "WhiteElo": pl.Int16(),
    "BlackElo": pl.Int16(),
    "WhiteRatingDiff": pl.Int16(),
    "BlackRatingDiff": pl.Int16(),
    "UTCDate": pl.String(),
    "UTCTime": pl.String(),
    "ECO": pl.String(),
    "Opening": pl.String(),
    "Event": pl.String(),
    "TimeControl": pl.String(),
    "Termination": pl.String(),
    "movetext": pl.String(),
    "num_plies": pl.Int16(),
}

POSITION_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.String(),
    "fen": pl.String(),
    "move_uci": pl.String(),
    "move_ply": pl.Int16(),
    "turn": pl.String(),
    "piece_count": pl.Int16(),
    "game_phase": pl.String(),
    "avg_elo": pl.Float32(),
}

EVAL_SCHEMA: dict[str, pl.DataType] = {
    "fen": pl.String(),
    "norm_fen": pl.String(),
    "turn": pl.String(),
    "piece_count": pl.Int16(),
    "game_phase": pl.String(),
    "depth": pl.Int16(),
    "knodes": pl.Int32(),
    "cp": pl.Int16(),
    "mate": pl.Int8(),
    "best_move": pl.String(),
}

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]$')
_COMMENT_RE = re.compile(r"\{[^}]*\}")
_NAG_RE = re.compile(r"\$\d+")
_MOVE_NUMBER_RE = re.compile(r"\b\d+\.(?:\.\.)?")
_RESULT_RE = re.compile(r"\b(?:1-0|0-1|1/2-1/2)\b")


def games_sample_url(year: int, month: int) -> str:
    """URL of one month of official standard rated games."""
    return games_month_url(year, month)


def evals_url() -> str:
    """URL of the official Stockfish evaluations file."""
    return OFFICIAL_EVALS_URL


def _to_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _game_id(site: str) -> str:
    return site.rstrip("/").rsplit("/", 1)[-1]


def _split_pgn(raw_pgn: str) -> tuple[dict[str, str], str]:
    """Split a PGN block into its header tags and its movetext."""
    headers: dict[str, str] = {}
    movetext_lines: list[str] = []
    in_movetext = False
    for line in raw_pgn.splitlines():
        stripped = line.strip()
        if not in_movetext:
            match = _HEADER_RE.match(stripped)
            if match:
                headers[match.group(1)] = match.group(2)
                continue
            if not stripped:
                continue
            in_movetext = True
        movetext_lines.append(stripped)
    return headers, " ".join(movetext_lines)


def _san_tokens(movetext: str) -> list[str]:
    """Strip comments, NAGs, move numbers and the result token from movetext."""
    text = _COMMENT_RE.sub(" ", movetext)
    text = _NAG_RE.sub(" ", text)
    text = _MOVE_NUMBER_RE.sub(" ", text)
    text = _RESULT_RE.sub(" ", text)
    return text.split()


def _replay_moves(movetext: str) -> list[str]:
    """Return the legally replayed SAN tokens; stops at the first illegal one."""
    board = chess.Board()
    played: list[str] = []
    for token in _san_tokens(movetext):
        try:
            board.push_san(token)
        except ValueError:
            break
        played.append(token)
    return played


def parse_pgn_game(raw_pgn: str) -> dict[str, object] | None:
    """Parse one PGN block into a flat row, or ``None`` when unusable."""
    headers, movetext_raw = _split_pgn(raw_pgn)
    site = headers.get("Site")
    result = headers.get("Result", "")
    if not site or result not in VALID_RESULTS:
        return None
    played = _replay_moves(movetext_raw)
    if not played:
        return None
    return {
        "game_id": _game_id(site),
        "White": headers.get("White"),
        "Black": headers.get("Black"),
        "Result": result,
        "WhiteElo": _to_int(headers.get("WhiteElo")),
        "BlackElo": _to_int(headers.get("BlackElo")),
        "WhiteRatingDiff": _to_int(headers.get("WhiteRatingDiff")),
        "BlackRatingDiff": _to_int(headers.get("BlackRatingDiff")),
        "UTCDate": headers.get("UTCDate"),
        "UTCTime": headers.get("UTCTime"),
        "ECO": headers.get("ECO"),
        "Opening": headers.get("Opening"),
        "Event": headers.get("Event"),
        "TimeControl": headers.get("TimeControl"),
        "Termination": headers.get("Termination"),
        "movetext": " ".join(played),
        "num_plies": len(played),
    }


def _to_frame(rows: list[dict[str, object]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).select(
        [pl.col(name).cast(dtype) for name, dtype in schema.items()]
    )


def parse_pgn_sample_df(
    source: str | Path, max_games: int = DEFAULT_MAX_GAMES
) -> pl.DataFrame:
    """Read up to ``max_games`` PGN blocks and return the parsed game rows."""
    rows = []
    for block in stream_pgn_blocks(source, max_games):
        parsed = parse_pgn_game(block)
        if parsed is not None:
            rows.append(parsed)
    return _to_frame(rows, GAME_SCHEMA)


def clean_games_df(
    df: pl.DataFrame,
    min_elo: int = DEFAULT_MIN_ELO,
    min_plies: int = DEFAULT_MIN_PLIES,
) -> pl.DataFrame:
    """Apply the agreed game filters: both ratings, no bullet, normal end."""
    return df.filter(
        pl.col("Result").is_in(VALID_RESULTS)
        & pl.col("movetext").is_not_null()
        & (pl.col("WhiteElo") >= min_elo)
        & (pl.col("BlackElo") >= min_elo)
        & (pl.col("Termination") == "Normal")
        & (~pl.col("Event").str.contains("Bullet"))
        & (pl.col("num_plies") >= min_plies)
    )


def extract_positions_df(
    clean_df: pl.DataFrame, max_positions: int = DEFAULT_MAX_POSITIONS
) -> pl.DataFrame:
    """Turn cleaned games into ``(fen, move_uci)`` training rows.

    Each row records the position *before* the move, so ``fen`` plus
    ``move_uci`` is exactly what a policy model must learn.
    """
    rows: list[dict[str, object]] = []
    for game in clean_df.iter_rows(named=True):
        if len(rows) >= max_positions:
            break
        board = chess.Board()
        white_elo = game["WhiteElo"] or 0
        black_elo = game["BlackElo"] or 0
        avg_elo = (white_elo + black_elo) / 2
        for ply, token in enumerate(str(game["movetext"]).split(), start=1):
            if len(rows) >= max_positions:
                break
            try:
                move = board.parse_san(token)
            except ValueError:
                break
            fen = board.fen(en_passant="legal")
            count = piece_count_from_fen(fen)
            rows.append(
                {
                    "game_id": game["game_id"],
                    "fen": fen,
                    "move_uci": move.uci(),
                    "move_ply": ply,
                    "turn": side_to_move(fen),
                    "piece_count": count,
                    "game_phase": phase_from_piece_count(count),
                    "avg_elo": avg_elo,
                }
            )
            board.push(move)
    return _to_frame(rows, POSITION_SCHEMA)


def parse_eval_line(line: str, min_depth: int = DEFAULT_MIN_DEPTH) -> dict[str, object] | None:
    """Parse one JSONL evaluation line, keeping only its deepest entry.

    Returns ``None`` for malformed lines, empty evaluations, depths below
    ``min_depth`` and positions whose best move is not legal.
    """
    try:
        payload = json.loads(line)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    fen = payload.get("fen")
    evals = payload.get("evals") or []
    if not fen or not evals:
        return None
    best = max(evals, key=lambda entry: entry.get("depth", 0))
    depth = best.get("depth", 0)
    if depth < min_depth:
        return None
    pvs = best.get("pvs") or []
    if not pvs:
        return None
    first_pv = pvs[0] or {}
    tokens = str(first_pv.get("line") or "").split()
    if not tokens:
        return None
    best_move = tokens[0]
    if not first_move_is_legal(fen, best_move):
        return None
    norm = normalize_fen(fen)
    count = piece_count_from_fen(norm)
    return {
        "fen": fen,
        "norm_fen": norm,
        "turn": side_to_move(norm),
        "piece_count": count,
        "game_phase": phase_from_piece_count(count),
        "depth": depth,
        "knodes": best.get("knodes"),
        "cp": first_pv.get("cp"),
        "mate": first_pv.get("mate"),
        "best_move": best_move,
    }


def load_clean_evals_df(
    source: str | Path,
    max_evals: int = DEFAULT_MAX_EVALS,
    min_depth: int = DEFAULT_MIN_DEPTH,
) -> pl.DataFrame:
    """Read up to ``max_evals`` unique, legal, deep-enough evaluation rows."""
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for line in stream_zst_lines(source):
        if not line.strip():
            continue
        parsed = parse_eval_line(line, min_depth)
        if parsed is None or parsed["norm_fen"] in seen:
            continue
        seen.add(str(parsed["norm_fen"]))
        rows.append(parsed)
        if len(rows) >= max_evals:
            break
    return _to_frame(rows, EVAL_SCHEMA)
