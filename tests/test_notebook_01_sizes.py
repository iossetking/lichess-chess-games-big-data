import importlib.util
import os
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1] / "notebooks" / "01_eda_and_cleaning.py"
)


@pytest.fixture(scope="module")
def size_helpers(tmp_path_factory: pytest.TempPathFactory):
    previous_mpl = os.environ.get("MPLCONFIGDIR")
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_path_factory.mktemp("mplconfig")))
    try:
        spec = importlib.util.spec_from_file_location("nb01_eda_and_cleaning", NOTEBOOK_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        first_cell = next(iter(module.app._cell_manager.cell_data())).cell
        _output, definitions = first_cell.run()
        return definitions["dataset_sizes_df"], definitions["display_dataset_sizes"]
    finally:
        if previous_mpl is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_mpl


def _write_sized_file(path: Path, size_bytes: int) -> Path:
    path.write_bytes(b"\0" * size_bytes)
    return path


def test_dataset_sizes_df_single_raw_file(size_helpers, tmp_path: Path):
    dataset_sizes_df, _ = size_helpers
    raw_file = _write_sized_file(tmp_path / "lichess_2025-01.pgn.zst", 1024**2)

    df = dataset_sizes_df([raw_file], show_individual=False)

    assert df.to_dicts() == [
        {
            "Archivo": "lichess_2025-01.pgn.zst",
            "MB": "1.00",
            "GB": "0.000977",
            "TB": "0.000000954",
        }
    ]


def test_dataset_sizes_df_multiple_raw_files_sums_only(size_helpers, tmp_path: Path):
    dataset_sizes_df, _ = size_helpers
    first = _write_sized_file(tmp_path / "lichess_2025-01.pgn.zst", 1024**2)
    second = _write_sized_file(tmp_path / "lichess_2025-02.pgn.zst", 3 * 1024**2)

    df = dataset_sizes_df([first, second], show_individual=False)

    assert df.to_dicts() == [
        {
            "Archivo": "Total (2 archivos)",
            "MB": "4.00",
            "GB": "0.003906",
            "TB": "0.000003815",
        }
    ]


def test_dataset_sizes_df_three_cleaned_parquets_and_total(size_helpers, tmp_path: Path):
    dataset_sizes_df, _ = size_helpers
    games = _write_sized_file(tmp_path / "games_clean_sample.parquet", 1024**2)
    positions = _write_sized_file(tmp_path / "positions_games_sample.parquet", 2 * 1024**2)
    evals = _write_sized_file(tmp_path / "evals_clean_sample.parquet", 5 * 1024**2)

    df = dataset_sizes_df([games, positions, evals], show_individual=True)

    assert df["Archivo"].to_list() == [
        "games_clean_sample.parquet",
        "positions_games_sample.parquet",
        "evals_clean_sample.parquet",
        "Total",
    ]
    assert df.to_dicts() == [
        {
            "Archivo": "games_clean_sample.parquet",
            "MB": "1.00",
            "GB": "0.000977",
            "TB": "0.000000954",
        },
        {
            "Archivo": "positions_games_sample.parquet",
            "MB": "2.00",
            "GB": "0.001953",
            "TB": "0.000001907",
        },
        {
            "Archivo": "evals_clean_sample.parquet",
            "MB": "5.00",
            "GB": "0.004883",
            "TB": "0.000004768",
        },
        {
            "Archivo": "Total",
            "MB": "8.00",
            "GB": "0.007812",
            "TB": "0.000007629",
        },
    ]


def test_dataset_sizes_df_single_path_and_empty_sequence(size_helpers, tmp_path: Path):
    dataset_sizes_df, display_dataset_sizes = size_helpers
    single = _write_sized_file(tmp_path / "single.parquet", 1024**2)

    single_df = dataset_sizes_df(single)
    assert single_df.to_dicts() == dataset_sizes_df([single]).to_dicts()

    empty_df = dataset_sizes_df([])
    assert empty_df.is_empty()
    assert empty_df.columns == ["Archivo", "MB", "GB", "TB"]
    assert display_dataset_sizes([]) is not None


def test_display_dataset_sizes_smoke(size_helpers, tmp_path: Path):
    _, display_dataset_sizes = size_helpers
    first = _write_sized_file(tmp_path / "a.parquet", 1024**2)
    second = _write_sized_file(tmp_path / "b.parquet", 2 * 1024**2)

    assert display_dataset_sizes([first, second], title="Raw", show_individual=False) is not None
    assert display_dataset_sizes([first, second], title="Clean", show_individual=True) is not None
