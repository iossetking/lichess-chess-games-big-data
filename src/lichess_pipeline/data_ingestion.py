"""
Streams `.pgn.zst` (standard games) and `.jsonl.zst` (Stockfish evaluations)
either from ``https://database.lichess.org/`` or from a local file,
decompressing on the fly with ``zstandard``. Nothing is parsed here beyond
splitting PGN games into blocks: parsing and cleaning live in
``pgn_parser.py``.

Only the first N lines/games are read, so the notebooks never download a
whole multi-gigabyte month: the HTTP stream is closed as soon as enough data
has been consumed.
"""

from __future__ import annotations

import io
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path

import zstandard as zstd

OFFICIAL_GAMES_URL_TEMPLATE = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"
)
OFFICIAL_GAMES_INDEX_URL = "https://database.lichess.org/standard/"
OFFICIAL_EVALS_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
FIRST_GAMES_YEAR = 2013
_GAMES_INDEX_RE = re.compile(
    r"lichess_db_standard_rated_(\d{4})-(\d{2})\.pgn\.zst(?!\.torrent)"
)
PGN_RESULT_TOKENS = ("1-0", "0-1", "1/2-1/2")
DOWNLOAD_CHUNK_BYTES = 1 << 20
OPEN_ATTEMPTS = 3
OPEN_BACKOFF_SECONDS = 0.75


class LichessConnectionError(RuntimeError):
    """Raised when the official Lichess files cannot be reached."""


def games_month_url(year: int, month: int) -> str:
    """URL of one month of official standard rated games (``.pgn.zst``)."""
    return OFFICIAL_GAMES_URL_TEMPLATE.format(year=year, month=month)


def available_game_months(timeout: int = 60) -> list[tuple[int, int]]:
    """Months with an official standard games archive, oldest first.

    Reads the directory index of ``database.lichess.org/standard/`` (a few
    dozen KB) instead of probing every month, so discovering availability
    costs one request.
    """
    try:
        with _urlopen(OFFICIAL_GAMES_INDEX_URL, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise LichessConnectionError(
            f"El índice de «{OFFICIAL_GAMES_INDEX_URL}» respondió "
            f"{exc.code} {exc.reason}. Vuelve a intentarlo en unos minutos."
        ) from exc
    except OSError as exc:
        raise LichessConnectionError(
            f"No se pudo leer el índice de «{OFFICIAL_GAMES_INDEX_URL}»: {exc}."
        ) from exc
    months = sorted(
        {(int(year), int(month)) for year, month in _GAMES_INDEX_RE.findall(html)}
    )
    if not months:
        raise LichessConnectionError(
            f"El índice de «{OFFICIAL_GAMES_INDEX_URL}» no listó ningún "
            "archivo .pgn.zst."
        )
    return months


def months_in_range(
    year_range: Sequence[int],
    month_range: Sequence[int],
    available: Sequence[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Cross a year range with a month range, splitting available from missing.

    ``year_range=(2020, 2022)`` and ``month_range=(3, 9)`` yield every month
    from March to September of 2020, 2021 and 2022. Months without an archive
    are returned separately so the caller can report them instead of failing.
    """
    first_year, last_year = sorted(year_range)
    first_month, last_month = sorted(month_range)
    present = set(available)
    selected: list[tuple[int, int]] = []
    missing: list[tuple[int, int]] = []
    for year in range(first_year, last_year + 1):
        for month in range(first_month, last_month + 1):
            (selected if (year, month) in present else missing).append((year, month))
    return selected, missing


def last_closed_month(today: date | None = None) -> tuple[int, int]:
    """Year and month of the most recent month that has already ended."""
    reference = today or date.today()
    previous = reference.replace(day=1) - timedelta(days=1)
    return previous.year, previous.month


def assumed_game_months(today: date | None = None) -> list[tuple[int, int]]:
    """Continuous 2013-01 to the last closed month.

    Fallback for when the index cannot be reached: Lichess has published every
    month since 2013, so assuming a continuous range is right for all but the
    newest month.
    """
    last_year, last_month = last_closed_month(today)
    return [
        (year, month)
        for year in range(FIRST_GAMES_YEAR, last_year + 1)
        for month in range(1, 13)
        if (year, month) <= (last_year, last_month)
    ]


def is_remote(source: str | Path) -> bool:
    """True when ``source`` is an HTTP(S) URL instead of a local path."""
    return str(source).startswith(("http://", "https://"))


def _is_transient(reason: object) -> bool:
    """True for network errors worth retrying (DNS, timeouts, resets)."""
    return isinstance(reason, (socket.gaierror, TimeoutError, ConnectionError))


def _connection_message(url: str, reason: object) -> str:
    host = urllib.parse.urlsplit(url).hostname or url
    if isinstance(reason, socket.gaierror):
        return (
            f"No se pudo resolver el host «{host}»: {reason}. "
            "Revisa la conexión a internet, el DNS o la VPN y vuelve a intentarlo."
        )
    return f"No se pudo conectar con «{url}»: {reason}. Vuelve a intentarlo."


def _urlopen(url: str, timeout: int = 60) -> io.IOBase:
    """Open ``url``, retrying transient DNS/connection failures.

    A failed ``getaddrinfo`` is common right after a sleep/wake or a network
    change; retrying briefly avoids surfacing it as a crash in the notebooks.
    """
    for attempt in range(1, OPEN_ATTEMPTS + 1):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            if not _is_transient(exc.reason):
                raise
            if attempt == OPEN_ATTEMPTS:
                raise LichessConnectionError(
                    _connection_message(url, exc.reason)
                ) from exc
            time.sleep(OPEN_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")


def _open_binary(source: str | Path) -> io.IOBase:
    if is_remote(source):
        return _urlopen(str(source))
    return open(source, "rb")


def stream_zst_lines(source: str | Path, max_lines: int | None = None) -> Iterator[str]:
    """Yield decompressed UTF-8 lines from a local ``.zst`` file or an URL.

    Stops after ``max_lines`` lines and closes the underlying stream, which is
    what keeps a sample read from downloading the whole file. A truncated
    stream (or one closed on purpose mid-frame) ends the iteration quietly.
    """
    if max_lines is not None and max_lines <= 0:
        raise ValueError("max_lines must be positive or None")
    raw = _open_binary(source)
    reader = zstd.ZstdDecompressor().stream_reader(raw, closefd=True)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    try:
        for count, line in enumerate(text, start=1):
            yield line.rstrip("\r\n")
            if max_lines is not None and count >= max_lines:
                return
    finally:
        text.close()


def stream_pgn_blocks(source: str | Path, max_games: int) -> Iterator[str]:
    """Yield complete PGN game blocks (headers plus movetext) up to ``max_games``.

    A game ends at a non-header line finishing with one of the three PGN result
    tokens. Blank lines are dropped. The trailing token lets black's ``1...``
    move numbers and ``{ [%clk ...] }`` comments pass through untouched; the
    parser strips them.
    """
    if max_games <= 0:
        raise ValueError("max_games must be positive")
    buffer: list[str] = []
    emitted = 0
    for line in stream_zst_lines(source):
        stripped = line.strip()
        if not stripped:
            continue
        buffer.append(stripped)
        if stripped.startswith("[") or not stripped.endswith(PGN_RESULT_TOKENS):
            continue
        yield "\n".join(buffer)
        buffer = []
        emitted += 1
        if emitted >= max_games:
            return


def download_sample_to_disk(
    url: str, dest_path: Path, max_bytes: int = 50_000_000
) -> Path:
    """Copy a bounded prefix of ``url`` to ``dest_path`` (50 MB by default).

    The default cap means asking for a local copy never silently pulls a
    multi-gigabyte monthly archive. Pass ``max_bytes=None`` for the full file.
    """
    return download_to_disk(url, dest_path, max_bytes=max_bytes)


def download_to_disk(url: str, dest_path: Path, max_bytes: int | None = None) -> Path:
    """Copy ``url`` to ``dest_path``, optionally stopping after ``max_bytes``.

    Used when the notebook is asked to keep a local sample in ``data/raw/``;
    ``max_bytes`` keeps the download bounded even for a monthly PGN archive.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with _urlopen(url) as response, dest_path.open("wb") as out:
        while max_bytes is None or written < max_bytes:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            if max_bytes is not None:
                chunk = chunk[: max_bytes - written]
            out.write(chunk)
            written += len(chunk)
    return dest_path
