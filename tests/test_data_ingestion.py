import io
import socket
import urllib.error
from datetime import date
from pathlib import Path

import pytest
import zstandard as zstd

from lichess_pipeline import data_ingestion
from lichess_pipeline.data_ingestion import (
    LichessConnectionError,
    assumed_game_months,
    available_game_months,
    download_sample_to_disk,
    games_month_url,
    months_in_range,
    stream_pgn_blocks,
    stream_zst_lines,
)

INDEX_HTML = """
<a href="lichess_db_standard_rated_2025-01.pgn.zst">x</a>
<a href="lichess_db_standard_rated_2025-01.pgn.zst.torrent">x</a>
<a href="lichess_db_standard_rated_2013-12.pgn.zst">x</a>
<a href="counts.txt">counts.txt</a>
"""

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


def test_open_binary_retries_transient_dns_errors(monkeypatch: pytest.MonkeyPatch):
    failures = [socket.gaierror(8, "nodename nor servname provided")] * 2
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: int = 60) -> io.BytesIO:
        calls.append(url)
        if failures:
            raise urllib.error.URLError(failures.pop(0))
        return io.BytesIO(b"payload")

    monkeypatch.setattr(data_ingestion.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_ingestion.time, "sleep", lambda _seconds: None)

    response = data_ingestion._open_binary(
        "https://database.lichess.org/lichess_db_standard_rated_2025-01.pgn.zst"
    )

    assert response.read() == b"payload"
    assert len(calls) == 3


def test_open_binary_reports_dns_failure_with_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_urlopen(url: str, timeout: int = 60) -> io.BytesIO:
        raise urllib.error.URLError(
            socket.gaierror(8, "nodename nor servname provided")
        )

    monkeypatch.setattr(data_ingestion.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_ingestion.time, "sleep", lambda _seconds: None)

    with pytest.raises(LichessConnectionError) as excinfo:
        data_ingestion._open_binary(
            "https://database.lichess.org/lichess_db_standard_rated_2025-01.pgn.zst"
        )

    message = str(excinfo.value)
    assert "database.lichess.org" in message
    assert "DNS" in message


def test_open_binary_does_not_retry_http_errors(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_urlopen(url: str, timeout: int = 60) -> io.BytesIO:
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data_ingestion.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(data_ingestion.time, "sleep", lambda _seconds: None)

    with pytest.raises(LichessConnectionError) as excinfo:
        data_ingestion._open_binary("https://database.lichess.org/missing.pgn.zst")

    assert "404" in str(excinfo.value)
    assert len(calls) == 1


def test_download_to_disk_converts_http_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def fake_urlopen(url: str, timeout: int = 60) -> io.BytesIO:
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data_ingestion.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(LichessConnectionError) as excinfo:
        data_ingestion.download_to_disk(
            "https://database.lichess.org/missing.pgn.zst",
            tmp_path / "missing.pgn.zst",
            max_bytes=1024,
        )

    assert "404" in str(excinfo.value)
    assert not (tmp_path / "missing.pgn.zst").exists()


def test_download_sample_to_disk_writes_a_bounded_prefix(tmp_path: Path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    destination = tmp_path / "prefix.bin"
    result = download_sample_to_disk(source.as_uri(), destination, max_bytes=4)
    assert result == destination
    assert destination.read_bytes() == b"0123"


def test_available_game_months_parses_index_and_ignores_other_files(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        data_ingestion.urllib.request,
        "urlopen",
        lambda url, timeout=60: io.BytesIO(INDEX_HTML.encode("utf-8")),
    )

    assert available_game_months() == [(2013, 12), (2025, 1)]


def test_available_game_months_raises_when_index_has_no_files(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        data_ingestion.urllib.request,
        "urlopen",
        lambda url, timeout=60: io.BytesIO(b"<html>empty</html>"),
    )

    with pytest.raises(LichessConnectionError):
        available_game_months()


def test_available_game_months_wraps_http_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_urlopen(url: str, timeout: int = 60) -> io.BytesIO:
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    monkeypatch.setattr(data_ingestion.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(LichessConnectionError):
        available_game_months()


def test_months_in_range_returns_selected_and_missing():
    available = [(2024, 1), (2024, 2), (2024, 3), (2025, 1), (2025, 2)]

    selected, missing = months_in_range((2024, 2025), (1, 2), available)

    assert selected == [(2024, 1), (2024, 2), (2025, 1), (2025, 2)]
    assert missing == []


def test_months_in_range_reports_unavailable_months_and_sorts_ranges():
    available = [(2024, 1), (2024, 3)]

    selected, missing = months_in_range((2024, 2024), (3, 1), available)

    assert selected == [(2024, 1), (2024, 3)]
    assert missing == [(2024, 2)]


def test_assumed_game_months_stops_at_last_closed_month():
    months = assumed_game_months(today=date(2026, 11, 18))

    assert months[0] == (2013, 1)
    assert months[-1] == (2026, 10)
    assert len(months) == 166
