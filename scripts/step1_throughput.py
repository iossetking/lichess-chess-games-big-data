"""Time a full remote read of `movetext` for one shard of the latest month.

Run: uv run python scripts/step1_throughput.py
Prints seconds, MB/s and extrapolations for one month and for the 4.94 TB set.
"""

import time

from huggingface_hub import HfApi

from lichess_pipeline.data_ingestion import (
    EXPECTED_SCHEMA,
    REPOSITORY_ID,
    REPOSITORY_REVISION,
    list_repo_parquet_files,
    month_shard_files,
    parse_month_partitions,
    scan_hf_dataset,
)

TOTAL_BYTES = 4.94e12


def main() -> None:
    files = list_repo_parquet_files()
    year, month = (int(p) for p in parse_month_partitions(files)[-1].split("-"))
    shards = month_shard_files(files, year, month)
    shard_mb = (
        HfApi()
        .get_paths_info(
            REPOSITORY_ID, shards[:1], repo_type="dataset", revision=REPOSITORY_REVISION
        )[0]
        .size
        / 1e6
    )
    start = time.perf_counter()
    frame = scan_hf_dataset(
        REPOSITORY_ID, shards[:1], REPOSITORY_REVISION, ["movetext"], EXPECTED_SCHEMA
    ).collect()
    seconds = time.perf_counter() - start
    raw_mb = frame["movetext"].estimated_size("mb")
    print(f"shard {shards[0]}: rows={len(frame):,} shards_in_month={len(shards)}")
    print(
        f"seconds={seconds:.1f} shard_file_mb={shard_mb:.0f} movetext_raw_mb={raw_mb:.0f}"
    )
    print(f"raw MB/s={raw_mb / seconds:.1f} file MB/s={shard_mb / seconds:.1f}")
    print(f"month ~ {seconds * len(shards) / 60:.1f} min sequential")
    print(f"4.94 TB ~ {TOTAL_BYTES / 1e6 / (shard_mb / seconds) / 3600:.1f} h")


if __name__ == "__main__":
    main()
