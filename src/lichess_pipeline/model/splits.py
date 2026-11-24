"""Grouped holdout splits and k-fold partitions.

Two facts shape these helpers:

* Positions from the same game are near-copies of each other, so a row split
  would leak information across train and validation. When a ``group_col`` is
  given, whole groups go to a single side.
* Cross-validation needs the same guarantee in every fold, and every row must
  land in exactly one validation fold.
"""

from __future__ import annotations

import polars as pl


def _chunks(series: pl.Series, n_chunks: int) -> list[pl.Series]:
    """Split ``series`` into ``n_chunks`` contiguous, near-equal chunks."""
    size, remainder = divmod(len(series), n_chunks)
    chunks: list[pl.Series] = []
    start = 0
    for index in range(n_chunks):
        length = size + (1 if index < remainder else 0)
        chunks.append(series.slice(start, length))
        start += length
    return chunks


def _shuffled_groups(frame: pl.DataFrame, group_col: str, seed: int) -> pl.Series:
    return (
        frame[group_col]
        .unique(maintain_order=True)
        .sample(fraction=1.0, shuffle=True, seed=seed)
    )


def split_holdout(
    frame: pl.DataFrame,
    fraction: float = 0.2,
    group_col: str | None = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(held_out, remaining)`` with no group on both sides.

    ``fraction`` is the share of rows (or groups, when ``group_col`` is given)
    that goes to the held-out split.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    if group_col is not None:
        groups = _shuffled_groups(frame, group_col, seed)
        n_held = max(1, int(fraction * len(groups)))
        held_groups = groups.head(n_held)
        mask = pl.col(group_col).is_in(held_groups.implode())
        return frame.filter(mask), frame.filter(~mask)
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    n_held = max(1, int(fraction * shuffled.height))
    return shuffled.head(n_held), shuffled.tail(shuffled.height - n_held)


def kfold_indices(
    frame: pl.DataFrame,
    n_folds: int = 5,
    group_col: str | None = None,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Return ``[(train_indices, validation_indices), ...]`` covering ``frame``.

    Matches :func:`kfold` but returns row positions so callers can slice an
    already-encoded :class:`torch.utils.data.Dataset` via ``Subset`` without
    duplicating tensors in RAM.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if group_col is not None:
        groups = _shuffled_groups(frame, group_col, seed)
        if len(groups) < n_folds:
            raise ValueError(
                f"n_folds ({n_folds}) exceeds the number of distinct "
                f"'{group_col}' values ({len(groups)})"
            )
        rows_by_group: dict[object, list[int]] = {}
        for row_index, value in enumerate(frame[group_col].to_list()):
            rows_by_group.setdefault(value, []).append(row_index)
        chunks = [chunk.to_list() for chunk in _chunks(groups, n_folds)]
        folds: list[tuple[list[int], list[int]]] = []
        for fold_index, val_groups in enumerate(chunks):
            val_indices = [
                row_index for group in val_groups for row_index in rows_by_group[group]
            ]
            train_indices = [
                row_index
                for other_index, other_groups in enumerate(chunks)
                if other_index != fold_index
                for group in other_groups
                for row_index in rows_by_group[group]
            ]
            folds.append((train_indices, val_indices))
        return folds
    if frame.height < n_folds:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds the number of rows ({frame.height})"
        )
    shuffled = pl.Series(range(frame.height)).sample(
        fraction=1.0, shuffle=True, seed=seed
    )
    chunks = [chunk.to_list() for chunk in _chunks(shuffled, n_folds)]
    return [
        (
            [
                row_index
                for other_index, other_chunk in enumerate(chunks)
                if other_index != fold_index
                for row_index in other_chunk
            ],
            val_chunk,
        )
        for fold_index, val_chunk in enumerate(chunks)
    ]


def kfold(
    frame: pl.DataFrame,
    n_folds: int = 5,
    group_col: str | None = None,
    seed: int = 42,
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    """Return ``[(train, validation), ...]`` covering ``frame`` exactly once.

    Every fold's validation slice is disjoint from its training slice and from
    the other folds' validation slices. With ``group_col``, a whole group stays
    in one slice.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if group_col is not None:
        groups = _shuffled_groups(frame, group_col, seed)
        if len(groups) < n_folds:
            raise ValueError(
                f"n_folds ({n_folds}) exceeds the number of distinct "
                f"'{group_col}' values ({len(groups)})"
            )
        folds = []
        for validation_groups in _chunks(groups, n_folds):
            mask = pl.col(group_col).is_in(validation_groups.implode())
            folds.append((frame.filter(~mask), frame.filter(mask)))
        return folds
    if frame.height < n_folds:
        raise ValueError(
            f"n_folds ({n_folds}) exceeds the number of rows ({frame.height})"
        )
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
    folds = []
    for validation_rows in _chunks(pl.Series(range(shuffled.height)), n_folds):
        n_validation = len(validation_rows)
        index = int(validation_rows[0]) if n_validation else 0
        validation = shuffled.slice(index, n_validation)
        train = pl.concat(
            [shuffled.head(index), shuffled.tail(shuffled.height - index - n_validation)]
        )
        folds.append((train, validation))
    return folds
