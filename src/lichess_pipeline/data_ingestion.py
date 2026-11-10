"""Stage 1: read-only access to the official Lichess database files.

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
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import zstandard as zstd

OFFICIAL_GAMES_URL_TEMPLATE = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"
)
OFFICIAL_EVALS_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
PGN_RESULT_TOKENS = ("1-0", "0-1", "1/2-1/2")
DOWNLOAD_CHUNK_BYTES = 1 << 20


def games_month_url(year: int, month: int) -> str:
    """URL of one month of official standard rated games (``.pgn.zst``)."""
    return OFFICIAL_GAMES_URL_TEMPLATE.format(year=year, month=month)


def is_remote(source: str | Path) -> bool:
    """True when ``source`` is an HTTP(S) URL instead of a local path."""
    return str(source).startswith(("http://", "https://"))


def _open_binary(source: str | Path) -> io.IOBase:
    if is_remote(source):
        return urllib.request.urlopen(str(source), timeout=60)
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
    except (zstd.ZstdError, EOFError):
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


def download_to_disk(url: str, dest_path: Path, max_bytes: int | None = None) -> Path:
    """Copy ``url`` to ``dest_path``, optionally stopping after ``max_bytes``.

    Used when the notebook is asked to keep a local sample in ``data/raw/``;
    ``max_bytes`` keeps the download bounded even for a monthly PGN archive.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with urllib.request.urlopen(url, timeout=60) as response, dest_path.open("wb") as out:
        while max_bytes is None or written < max_bytes:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            if max_bytes is not None:
                chunk = chunk[: max_bytes - written]
            out.write(chunk)
            written += len(chunk)
    return dest_path
