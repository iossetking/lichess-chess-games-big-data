"""Stage 1: read-only access to the Lichess standard-chess-games dataset.

Lists months and shard files, reads a month remotely with Polars (hf:// URLs)
or downloads it, selects columns and validates the schema. It never filters rows, cleans data or writes processed output.
"""

import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

REPOSITORY_ID = "Lichess/standard-chess-games"
REPOSITORY_REVISION = "de4e636eddf568a9394cc01fb0b9e1da04a6babf"
REPOSITORY_REVISION_DATE = "2025-10-16"

EXPECTED_SCHEMA: dict[str, pl.DataType] = {
    "Event": pl.String(),
    "Site": pl.String(),
    "White": pl.String(),
    "Black": pl.String(),
    "Result": pl.String(),
    "WhiteTitle": pl.String(),
    "BlackTitle": pl.String(),
    "WhiteElo": pl.Int16(),
    "BlackElo": pl.Int16(),
    "WhiteRatingDiff": pl.Int16(),
    "BlackRatingDiff": pl.Int16(),
    "UTCDate": pl.Date(),
    "UTCTime": pl.Time(),
    "ECO": pl.String(),
    "Opening": pl.String(),
    "Termination": pl.String(),
    "TimeControl": pl.String(),
    "movetext": pl.String(),
}

DEFAULT_COLUMNS: list[str] = [
    "Site",
    "Result",
    "WhiteElo",
    "BlackElo",
    "WhiteTitle",
    "BlackTitle",
    "TimeControl",
    "Termination",
    "Event",
    "ECO",
    "Opening",
    "UTCDate",
    "UTCTime",
    "movetext",
]

_PARTITION_RE = re.compile(r"^data/year=(\d{4})/month=(\d{2})/[^/]+\.parquet$")


def parse_month_partitions(files: Sequence[str]) -> list[str]:
    """Return the sorted unique "YYYY-MM" months found in repo file paths."""
    months = {
        f"{m.group(1)}-{m.group(2)}" for f in files if (m := _PARTITION_RE.match(f))
    }
    return sorted(months)


def month_shard_files(files: Sequence[str], year: int, month: int) -> list[str]:
    """Return the sorted parquet shard paths of one month (any shard count)."""
    prefix = f"data/year={year:04d}/month={month:02d}/"
    return sorted(f for f in files if f.startswith(prefix) and _PARTITION_RE.match(f))


def list_repo_parquet_files(
    revision: str = REPOSITORY_REVISION, api: HfApi | None = None
) -> list[str]:
    """List parquet files of the dataset at a pinned revision (one API call)."""
    api = api or HfApi()
    files = api.list_repo_files(REPOSITORY_ID, repo_type="dataset", revision=revision)
    return [f for f in files if f.endswith(".parquet")]


def remote_month_urls(
    year: int,
    month: int,
    revision: str = REPOSITORY_REVISION,
    files: Sequence[str] | None = None,
) -> list[str]:
    """hf:// URLs of a month's shards, pinned to `revision`, for remote Polars reads."""
    if files is None:
        files = list_repo_parquet_files(revision)
    shards = month_shard_files(files, year, month)
    if not shards:
        raise ValueError(f"No shards found for {year:04d}-{month:02d}.")
    return [f"hf://datasets/{REPOSITORY_ID}@{revision}/{shard}" for shard in shards]


def month_download_size(
    year: int,
    month: int,
    revision: str = REPOSITORY_REVISION,
    files: Sequence[str] | None = None,
    api: HfApi | None = None,
) -> int:
    """Total bytes of a month's shards. Pass `files` to avoid re-listing."""
    api = api or HfApi()
    if files is None:
        files = list_repo_parquet_files(revision, api)
    shards = month_shard_files(files, year, month)
    if not shards:
        raise ValueError(f"No shards found for {year:04d}-{month:02d}.")
    infos = api.get_paths_info(
        REPOSITORY_ID, shards, repo_type="dataset", revision=revision
    )
    return sum(info.size for info in infos)


def _rate_limit_delay(error: Exception) -> float | None:
    """Return the Retry-After seconds for an HTTP 429 (0.0 if absent), else None."""
    if not isinstance(error, HfHubHTTPError) or error.response is None:
        return None
    if error.response.status_code != 429:
        return None
    try:
        return float(error.response.headers.get("Retry-After", 0))
    except ValueError:
        return 0.0


def _download_with_retry(
    filename: str,
    cache_dir: Path,
    revision: str,
    max_retries: int,
    base_delay: float,
    downloader: Callable[..., str],
    sleep: Callable[[float], None],
) -> str:
    for attempt in range(max_retries + 1):
        try:
            return downloader(
                repo_id=REPOSITORY_ID,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
            )
        except Exception as error:
            retry_after = _rate_limit_delay(error)
            if retry_after is None or attempt == max_retries:
                raise
            sleep(max(base_delay * 2**attempt, retry_after))
    raise AssertionError("unreachable")


def download_month(
    year: int,
    month: int,
    cache_dir: Path,
    revision: str = REPOSITORY_REVISION,
    files: Sequence[str] | None = None,
    max_retries: int = 5,
    base_delay: float = 2.0,
    downloader: Callable[..., str] = hf_hub_download,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    """Download all shards of a month into `cache_dir`.

    Retries with exponential backoff only on HTTP 429 (honouring Retry-After);
    other errors propagate. The auth token comes from the local Hugging Face
    login, never from the repo.
    """
    if files is None:
        files = list_repo_parquet_files(revision)
    shards = month_shard_files(files, year, month)
    if not shards:
        raise ValueError(f"No shards found for {year:04d}-{month:02d}.")
    return [
        Path(
            _download_with_retry(
                filename,
                cache_dir,
                revision,
                max_retries,
                base_delay,
                downloader,
                sleep,
            )
        )
        for filename in shards
    ]


def validate_schema(schema: pl.Schema, columns: Sequence[str]) -> None:
    """Raise ValueError listing missing columns and dtype mismatches."""
    missing: list[str] = []
    mismatched: list[str] = []
    for name in columns:
        if name not in schema:
            missing.append(name)
        elif name in EXPECTED_SCHEMA and schema[name] != EXPECTED_SCHEMA[name]:
            mismatched.append(
                f"{name}: expected {EXPECTED_SCHEMA[name]}, got {schema[name]}"
            )
    if missing or mismatched:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if mismatched:
            parts.append(f"dtype mismatches: {mismatched}")
        raise ValueError("Schema validation failed; " + "; ".join(parts))


def scan_month(
    paths: Sequence[Path | str], columns: Sequence[str] = DEFAULT_COLUMNS
) -> pl.LazyFrame:
    """Lazily scan local paths or hf:// URLs, validate and select columns.

    Remote reads only fetch what a query needs (footers, chosen columns), but
    every collect goes over the network again.
    """
    unknown = [c for c in columns if c not in EXPECTED_SCHEMA]
    if unknown:
        raise ValueError(f"Unknown columns requested: {unknown}")
    lazy = pl.scan_parquet([str(p) for p in paths], hive_partitioning=False)
    validate_schema(lazy.collect_schema(), columns)
    return lazy.select(list(columns))
