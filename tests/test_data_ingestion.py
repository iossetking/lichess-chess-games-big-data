"""Network-free tests for the Stage 1 ingestion module."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from huggingface_hub.errors import HfHubHTTPError

from lichess_pipeline.data_ingestion import (
    DEFAULT_COLUMNS,
    EXPECTED_SCHEMA,
    download_month,
    month_download_size,
    month_shard_files,
    parse_month_partitions,
    remote_month_urls,
    scan_month,
    validate_schema,
)

FILES = [
    "README.md",
    "data/year=2013/month=01/train-00000-of-00001.parquet",
    "data/year=2024/month=02/train-00001-of-00002.parquet",
    "data/year=2024/month=02/train-00000-of-00002.parquet",
    "data/year=2024/month=11/train-00000-of-00001.parquet",
]


def _http_error(status: int, headers: dict[str, str] | None = None) -> HfHubHTTPError:
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=None)
    return HfHubHTTPError(f"HTTP {status}", response=response)  # type: ignore[arg-type]


def test_parse_month_partitions_sorted_unique_ignores_non_data() -> None:
    assert parse_month_partitions(FILES) == ["2013-01", "2024-02", "2024-11"]


def test_month_shard_files_variable_shard_count_sorted() -> None:
    assert month_shard_files(FILES, 2024, 2) == [
        "data/year=2024/month=02/train-00000-of-00002.parquet",
        "data/year=2024/month=02/train-00001-of-00002.parquet",
    ]
    assert len(month_shard_files(FILES, 2013, 1)) == 1
    assert month_shard_files(FILES, 2020, 5) == []


def test_month_download_size_sums_sizes_with_fake_api() -> None:
    class Info:
        def __init__(self, size: int) -> None:
            self.size = size

    class FakeApi:
        def get_paths_info(self, repo_id, paths, **kwargs):
            return [Info(10) for _ in paths]

    assert month_download_size(2024, 2, files=FILES, api=FakeApi()) == 20  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No shards"):
        month_download_size(2020, 5, files=FILES, api=FakeApi())  # type: ignore[arg-type]


def test_download_month_retries_on_429_with_backoff(tmp_path: Path) -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def downloader(**kwargs) -> str:
        calls.append(kwargs["filename"])
        if len(calls) <= 2:
            raise _http_error(429)
        return str(tmp_path / Path(kwargs["filename"]).name)

    paths = download_month(
        2013, 1, tmp_path, files=FILES, downloader=downloader, sleep=sleeps.append
    )
    assert len(calls) == 3
    assert sleeps == [2.0, 4.0]
    assert paths == [tmp_path / "train-00000-of-00001.parquet"]


def test_download_month_does_not_retry_other_errors(tmp_path: Path) -> None:
    sleeps: list[float] = []

    def downloader(**kwargs) -> str:
        raise _http_error(404)

    with pytest.raises(HfHubHTTPError):
        download_month(
            2013, 1, tmp_path, files=FILES, downloader=downloader, sleep=sleeps.append
        )
    assert sleeps == []


def test_download_month_gives_up_after_max_retries(tmp_path: Path) -> None:
    def downloader(**kwargs) -> str:
        raise _http_error(429)

    with pytest.raises(HfHubHTTPError):
        download_month(
            2013,
            1,
            tmp_path,
            files=FILES,
            max_retries=2,
            downloader=downloader,
            sleep=lambda s: None,
        )


def test_validate_schema_reports_missing_and_mismatched() -> None:
    schema = pl.Schema({"WhiteElo": pl.String(), "Site": pl.String()})
    with pytest.raises(ValueError) as exc:
        validate_schema(schema, ["WhiteElo", "Site", "Result"])
    message = str(exc.value)
    assert "Result" in message and "WhiteElo" in message


def test_validate_schema_ok() -> None:
    schema = pl.Schema(EXPECTED_SCHEMA)
    validate_schema(schema, DEFAULT_COLUMNS)


def test_scan_month_selects_columns_and_rejects_bad_schema(tmp_path: Path) -> None:
    good = tmp_path / "good.parquet"
    pl.DataFrame(
        {"Site": ["a"], "WhiteElo": pl.Series([1500], dtype=pl.Int16)}
    ).write_parquet(good)
    out = scan_month([good], ["WhiteElo"]).collect()
    assert out.columns == ["WhiteElo"] and out.height == 1

    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"WhiteElo": [1500]}).write_parquet(bad)  # Int64, not Int16
    with pytest.raises(ValueError, match="dtype mismatches"):
        scan_month([bad], ["WhiteElo"])
    with pytest.raises(ValueError, match="Unknown columns"):
        scan_month([good], ["Nope"])


def test_download_month_honours_retry_after(tmp_path: Path) -> None:
    sleeps: list[float] = []
    calls: list[str] = []

    def downloader(**kwargs) -> str:
        calls.append(kwargs["filename"])
        if len(calls) == 1:
            raise _http_error(429, {"Retry-After": "30"})
        return str(tmp_path / "x.parquet")

    download_month(
        2013, 1, tmp_path, files=FILES, downloader=downloader, sleep=sleeps.append
    )
    assert sleeps == [30.0]


def test_download_month_multi_shard_order_and_per_shard_retries(tmp_path: Path) -> None:
    failures = {"train-00001-of-00002.parquet": 1}
    sleeps: list[float] = []

    def downloader(**kwargs) -> str:
        name = Path(kwargs["filename"]).name
        if failures.get(name, 0) > 0:
            failures[name] -= 1
            raise _http_error(429)
        return str(tmp_path / name)

    paths = download_month(
        2024, 2, tmp_path, files=FILES, downloader=downloader, sleep=sleeps.append
    )
    assert [p.name for p in paths] == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]
    assert sleeps == [2.0]


def test_remote_month_urls_pinned_and_sorted() -> None:
    urls = remote_month_urls(2024, 2, revision="abc123", files=FILES)
    assert urls == [
        "hf://datasets/Lichess/standard-chess-games@abc123/data/year=2024/month=02/train-00000-of-00002.parquet",
        "hf://datasets/Lichess/standard-chess-games@abc123/data/year=2024/month=02/train-00001-of-00002.parquet",
    ]
    with pytest.raises(ValueError, match="No shards"):
        remote_month_urls(2020, 5, files=FILES)
