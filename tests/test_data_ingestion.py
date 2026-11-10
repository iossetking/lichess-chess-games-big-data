from pathlib import Path

import zstandard as zstd

from lichess_pipeline.data_ingestion import (
    games_month_url,
    stream_pgn_blocks,
    stream_zst_lines,
)

SAMPLE_PGN = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/abcd1234"]\n'
    '[WhiteElo "1950"]\n[BlackElo "1920"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 1-0\n\n"
    '[Event "Rated Rapid game"]\n[Site "https://lichess.org/efgh5678"]\n'
    '[WhiteElo "2100"]\n[BlackElo "2050"]\n[Result "0-1"]\n\n'
    "1. d4 d5 0-1\n\n"
)


def test_games_month_url():
    assert games_month_url(2025, 1) == (
        "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    )


def test_stream_zst_lines_handles_early_stop(tmp_path: Path):
    zst_path = tmp_path / "sample.jsonl.zst"
    cctx = zstd.ZstdCompressor()
    zst_path.write_bytes(cctx.compress(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'))
    lines = list(stream_zst_lines(zst_path, max_lines=2))
    assert lines == ['{"a": 1}', '{"a": 2}']


def test_stream_pgn_blocks_splits_games_and_respects_max_games(tmp_path: Path):
    zst_path = tmp_path / "sample.pgn.zst"
    cctx = zstd.ZstdCompressor()
    zst_path.write_bytes(cctx.compress(SAMPLE_PGN.encode("utf-8")))
    blocks = list(stream_pgn_blocks(zst_path, max_games=1))
    assert len(blocks) == 1
    assert "https://lichess.org/abcd1234" in blocks[0]
    assert "1. e4 e5" in blocks[0]
