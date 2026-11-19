"""Grouped holdout and k-fold partitions."""

import polars as pl
import pytest

from lichess_pipeline.model.splits import kfold, split_holdout


def make_frame() -> pl.DataFrame:
    # 10 games with 3 positions each.
    return pl.DataFrame(
        {
            "game_id": [f"g{game:02d}" for game in range(10) for _ in range(3)],
            "position": list(range(30)),
        }
    )


def test_holdout_by_group_keeps_games_whole_and_covers_every_row():
    frame = make_frame()
    held, remaining = split_holdout(frame, fraction=0.2, group_col="game_id")

    assert set(held["game_id"]).isdisjoint(remaining["game_id"])
    assert held.height + remaining.height == frame.height
    assert held.height == 3 * 2  # 20% of 10 games
    assert set(held["position"]).isdisjoint(remaining["position"])


def test_holdout_by_rows_is_disjoint_and_complete():
    frame = make_frame()
    held, remaining = split_holdout(frame, fraction=0.2, seed=1)

    assert held.height == 6
    assert set(held["position"]).isdisjoint(remaining["position"])
    assert set(held["position"]) | set(remaining["position"]) == set(
        frame["position"]
    )


@pytest.mark.parametrize("group_col", ["game_id", None])
def test_kfold_covers_each_row_once_and_never_shares_a_group(group_col):
    frame = make_frame()
    folds = kfold(frame, n_folds=5, group_col=group_col)

    assert len(folds) == 5
    validated: list[int] = []
    for train, validation in folds:
        assert validation.height > 0
        assert set(train["position"]).isdisjoint(validation["position"])
        assert train.height + validation.height == frame.height
        if group_col is not None:
            assert set(train[group_col]).isdisjoint(validation[group_col])
        validated.extend(validation["position"].to_list())

    assert sorted(validated) == list(range(frame.height))


def test_kfold_balances_the_validation_slices():
    frame = make_frame()
    sizes = [validation.height for _, validation in kfold(frame, n_folds=4)]

    assert max(sizes) - min(sizes) <= 1


def test_split_rejects_invalid_arguments():
    frame = make_frame()
    with pytest.raises(ValueError, match="fraction"):
        split_holdout(frame, fraction=0)
    with pytest.raises(ValueError, match="fraction"):
        split_holdout(frame, fraction=1)
    with pytest.raises(ValueError, match="at least 2"):
        kfold(frame, n_folds=1)
    with pytest.raises(ValueError, match="exceeds the number of distinct"):
        kfold(frame, n_folds=11, group_col="game_id")
