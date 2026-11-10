"""A small policy network trained from scratch on human games, then fine-tuned.

One model, two stages:

1. Learn ``(fen -> played move)`` from high-Elo human games (``.pgn.zst``).
2. Fine-tune the same weights on ``(fen -> Stockfish move)`` from
   ``lichess_db_eval.jsonl.zst``.

The network always scores all 4096 ``from/to`` classes. Illegal moves are
masked just before ``softmax``, so predictions are legal by construction and
the validation metrics measure ranking among legal moves, not legality.
"""

from __future__ import annotations

import chess
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lichess_pipeline.features import (
    NUM_MOVE_CLASSES,
    encode_fen,
    index_to_move,
    legal_move_mask,
    move_to_index,
)

MASK_VALUE = -1e9
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 256
DEFAULT_CHANNELS = (64, 128, 128)


class ChessPolicyNet(nn.Module):
    """Three convolutional blocks followed by a 4096-class linear head."""

    def __init__(self, channels: tuple[int, ...] = DEFAULT_CHANNELS) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 13
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(in_channels * 8 * 8, NUM_MOVE_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.flatten(self.features(x), start_dim=1))


def make_tensor_dataset(
    df: pl.DataFrame, fen_col: str = "fen", move_col: str = "move_uci"
) -> TensorDataset:
    """Encode rows into ``(board tensor, move class, legal mask)`` triples.

    The legal mask travels with each row so validation can rank moves the same
    way inference does. It costs 4096 booleans per position, which is small
    next to the board tensor for the notebook-sized samples used here.
    """
    if df.height == 0:
        raise ValueError("Cannot build a tensor dataset from an empty frame")
    fens = df[fen_col].to_list()
    features = torch.from_numpy(np.stack([encode_fen(fen) for fen in fens]))
    targets = torch.tensor(
        [move_to_index(move) for move in df[move_col].to_list()], dtype=torch.long
    )
    masks = torch.from_numpy(np.stack([legal_move_mask(fen) for fen in fens]))
    return TensorDataset(features, targets, masks)


def _loader(dataset: TensorDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _masked_logits(model: nn.Module, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    logits = model(features)
    return logits.masked_fill(~masks.to(torch.bool), MASK_VALUE)


def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    with torch.no_grad():
        for features, targets, masks in loader:
            logits = _masked_logits(model, features, masks)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            top_k = logits.topk(3, dim=1).indices
            hits = top_k == targets.unsqueeze(1)
            correct_top1 += int(hits[:, 0].sum())
            correct_top3 += int(hits.any(dim=1).sum())
            total += targets.size(0)
    return total_loss / total, correct_top1 / total, correct_top3 / total


def evaluate_model(
    model: nn.Module,
    dataset: TensorDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, float]:
    """Score ``model`` on a dataset: legal-move loss plus Top-1/Top-3 accuracy."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate on an empty dataset")
    loss, top1, top3 = _evaluate(
        model, _loader(dataset, batch_size, shuffle=False), nn.CrossEntropyLoss()
    )
    return {"loss": loss, "top1": top1, "top3": top3}


def train_model(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict[str, float]]:
    """Train ``model`` and return one metrics dict per epoch.

    The same function serves both stages: call it once with the human-game
    dataset, then again with the Stockfish dataset and a smaller ``lr``.
    """
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation datasets must not be empty")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(train_ds, batch_size, shuffle=True)
    val_loader = _loader(val_ds, batch_size, shuffle=False)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, targets, masks in train_loader:
            optimizer.zero_grad()
            logits = _masked_logits(model, features, masks)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
        val_loss, val_top1, val_top3 = _evaluate(model, val_loader, criterion)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / seen,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top3": val_top3,
            }
        )
    return history


def predict_top_moves(
    model: nn.Module, fen: str, top_k: int = 3
) -> list[tuple[str, float]]:
    """Return up to ``top_k`` ``(uci, probability)`` legal moves for ``fen``.

    A terminal position has no legal moves and returns an empty list.
    """
    board = chess.Board(fen)
    mask = legal_move_mask(fen)
    available = int(mask.sum())
    if available == 0:
        return []
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(encode_fen(fen)).unsqueeze(0)
        logits = model(features)[0].masked_fill(
            torch.from_numpy(~mask), MASK_VALUE
        )
        probabilities = torch.softmax(logits, dim=0)
    values, indices = probabilities.topk(min(top_k, available))
    return [
        (index_to_move(int(index), board), float(value))
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]
