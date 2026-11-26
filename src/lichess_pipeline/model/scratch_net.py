"""Policy network trained from scratch on human games, then fine-tuned.

One model, two stages:

1. Learn ``(fen -> played move)`` from high-Elo human games (``.pgn.zst``).
2. Fine-tune the same weights on ``(fen -> Stockfish move)`` from
   ``lichess_db_eval.jsonl.zst``.

The network always scores all 4096 ``from/to`` classes. Illegal moves are
masked just before ``softmax``, so predictions are legal by construction and
the validation metrics measure ranking among legal moves, not legality.

Architecture (v3): an 18-plane side-to-move canonical board feeds a residual
convolutional tower with a full-board receptive field. The move head is
spatial and factorised: each square embedding is projected into an origin and
a destination vector, and a move is scored with their dot product plus a
learned 64x64 geometry bias. That shares learning across all squares (a rare
move still trains its origin and destination) and keeps the model around 1.2M
parameters instead of 34M.

``LegacyChessPolicyNet`` reproduces the original 13-plane flat-head network so
checkpoints saved before the upgrade keep loading.
"""

from __future__ import annotations

import chess
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lichess_pipeline.features import (
    BOARD_SIZE,
    LEGACY_NUM_PLANES,
    NUM_MOVE_CLASSES,
    NUM_PLANES,
    encode_fen,
    encode_fen_legacy,
    encode_position,
    index_to_move,
    legal_move_mask,
)

MASK_VALUE = -1e9
PACKED_MASK_BYTES = NUM_MOVE_CLASSES // 8  # 512 bytes for 4096 legal-move bits
_UNPACK_BITS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 256
DEFAULT_CHANNELS = 128
DEFAULT_DEPTH = 4
HEAD_HIDDEN = 64

# Legacy architecture (13 planes, three conv blocks, flat 4096-way head).
LEGACY_CHANNELS = (64, 128, 128)


def pick_device(preferred: str | torch.device | None = None) -> torch.device:
    """Return the fastest available device: CUDA, then Apple MPS, then CPU."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with batch norm and a residual connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + x)


class SpatialMoveHead(nn.Module):
    """Score moves as ``proj_from(from) . proj_to(to)`` plus a geometry bias.

    Unlike a flat ``Linear(8192, 4096)``, every square shares the same two
    projections, so the head learns what a movable origin and a good target
    look like instead of memorising 4096 independent move rows.
    """

    def __init__(
        self, channels: int = DEFAULT_CHANNELS, hidden: int = HEAD_HIDDEN
    ) -> None:
        super().__init__()
        self.proj_from = nn.Linear(channels, hidden)
        self.proj_to = nn.Linear(channels, hidden)
        # 64x64 learned bias encoding "this from->to pair is plausible".
        self.spatial_bias = nn.Parameter(torch.zeros(64, 64))
        self.scale = hidden**-0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        squares = features.flatten(2).transpose(1, 2)  # (B, 64, C)
        origins = self.proj_from(squares)
        destinations = self.proj_to(squares)
        logits = self.scale * (origins @ destinations.transpose(1, 2))
        # While LoRA is training, ``spatial_bias`` is a module returning the
        # frozen bias plus its low-rank update; otherwise it is the raw tensor.
        bias = self.spatial_bias() if isinstance(self.spatial_bias, nn.Module) else self.spatial_bias
        return (logits + bias).reshape(logits.shape[0], NUM_MOVE_CLASSES)


class ChessPolicyNet(nn.Module):
    """Residual conv tower plus a spatial factorised move head."""

    canonical = True
    num_input_planes = NUM_PLANES

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        depth: int = DEFAULT_DEPTH,
        head_hidden: int = HEAD_HIDDEN,
    ) -> None:
        super().__init__()
        if channels < 1 or depth < 0:
            raise ValueError("channels must be positive and depth non-negative")
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(depth)])
        self.head = SpatialMoveHead(channels, head_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class LegacyChessPolicyNet(nn.Module):
    """Original v1/v2 network: 13 planes, three conv blocks, flat 4096 head.

    It is kept only to load old checkpoints; new training should use
    :class:`ChessPolicyNet`.
    """

    canonical = False
    num_input_planes = LEGACY_NUM_PLANES

    def __init__(self, channels: tuple[int, ...] = LEGACY_CHANNELS) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = LEGACY_NUM_PLANES
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


def unpack_legal_masks(masks: torch.Tensor) -> torch.Tensor:
    """Return a ``(B, 4096)`` boolean mask from either bool or bit-packed input."""
    if masks.shape[-1] == NUM_MOVE_CLASSES:
        return masks.to(torch.bool)
    if masks.shape[-1] != PACKED_MASK_BYTES:
        raise ValueError(
            f"Expected mask width {NUM_MOVE_CLASSES} or {PACKED_MASK_BYTES}, "
            f"got {masks.shape[-1]}"
        )
    bits = _UNPACK_BITS.to(masks.device)
    return ((masks.unsqueeze(-1) & bits) != 0).reshape(
        masks.shape[0], NUM_MOVE_CLASSES
    )


def make_tensor_dataset(
    df: pl.DataFrame, fen_col: str = "fen", move_col: str = "move_uci"
) -> TensorDataset:
    """Encode rows into compact ``(board, move class, packed legal mask)`` triples.

    Board and move are side-to-move canonical, and the legal mask uses the
    same mirrored indices, so validation ranks moves exactly as inference
    does. Binary board planes are stored as ``uint8`` (1,152 B/row) and the
    4096-bit legal mask is bit-packed into 512 ``uint8`` bytes (~1.6 KB/row
    total instead of ~8.7 KB/row), then unpacked on the fly per mini-batch.
    """
    count = df.height
    if count == 0:
        raise ValueError("Cannot build a tensor dataset from an empty frame")
    fens = df[fen_col].to_list()
    moves = df[move_col].to_list()
    features = np.empty((count, NUM_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    targets = np.empty(count, dtype=np.int64)
    masks = np.empty((count, PACKED_MASK_BYTES), dtype=np.uint8)
    for index, (fen, move) in enumerate(zip(fens, moves, strict=True)):
        features[index], targets[index], masks[index] = encode_position(fen, move)
    return TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(targets),
        torch.from_numpy(masks),
    )


def _loader(
    dataset: Dataset, batch_size: int, shuffle: bool, pin_memory: bool = False
) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )


def _masked_logits(
    model: nn.Module, features: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    with torch.autocast(
        device_type=features.device.type, enabled=features.device.type == "cuda"
    ):
        logits = model(features.to(dtype=torch.float32))
    return logits.float().masked_fill(~unpack_legal_masks(masks), MASK_VALUE)


def _evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    with torch.no_grad():
        for features, targets, masks in loader:
            features, targets, masks = (
                features.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
                masks.to(device, non_blocking=True),
            )
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
    dataset: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Score ``model`` on a dataset: legal-move loss plus Top-1/Top-3 accuracy."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate on an empty dataset")
    target_device = (
        pick_device(device) if device is not None else next(model.parameters()).device
    )
    model.to(target_device)
    loss, top1, top3 = _evaluate(
        model,
        _loader(
            dataset,
            batch_size,
            shuffle=False,
            pin_memory=target_device.type == "cuda",
        ),
        nn.CrossEntropyLoss(),
        target_device,
    )
    return {"loss": loss, "top1": top1, "top3": top3}


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to("cpu").clone()
        for name, tensor in model.state_dict().items()
    }


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | torch.device | None = None,
    keep_best: bool = True,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train ``model`` and return one metrics dict per epoch.

    The same function serves both stages: call it once with the human-game
    dataset, then again with the Stockfish dataset and a smaller ``lr``.

    The learning rate follows a cosine schedule over ``epochs``, which sharpens
    the final predictions, and with ``keep_best`` the weights of the epoch with
    the lowest validation loss are restored at the end (``model.best_epoch``;
    the last epoch is kept when ``keep_best`` is false).
    """
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation datasets must not be empty")
    target_device = pick_device(device)
    model.to(target_device)
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    scaler = torch.amp.GradScaler(
        target_device.type, enabled=target_device.type == "cuda"
    )
    criterion = nn.CrossEntropyLoss()
    use_pin_memory = target_device.type == "cuda"
    train_loader = _loader(train_ds, batch_size, shuffle=True, pin_memory=use_pin_memory)
    val_loader = _loader(val_ds, batch_size, shuffle=False, pin_memory=use_pin_memory)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, targets, masks in train_loader:
            features, targets, masks = (
                features.to(target_device, non_blocking=True),
                targets.to(target_device, non_blocking=True),
                masks.to(target_device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = _masked_logits(model, features, masks)
            loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
        scheduler.step()
        val_loss, val_top1, val_top3 = _evaluate(
            model, val_loader, criterion, target_device
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / seen,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top3": val_top3,
            }
        )
        if keep_best and val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, _cpu_state(model), epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch = best_epoch if keep_best else epochs
    model.best_val_loss = best_loss if keep_best else history[-1]["val_loss"]
    model.eval()
    return history


def predict_top_moves(
    model: nn.Module, fen: str, top_k: int = 3
) -> list[tuple[str, float]]:
    """Return up to ``top_k`` ``(uci, probability)`` legal moves for ``fen``.

    A terminal position has no legal moves and returns an empty list. Models
    that wrap another predictor (``load_policy_checkpoint`` returns a Maia
    adapter this way) are delegated to their own ``predict_top_moves``.
    """
    if hasattr(model, "predict_top_moves"):
        return model.predict_top_moves(fen, top_k=top_k)
    board = chess.Board(fen)
    canonical = getattr(model, "canonical", True)
    mask = legal_move_mask(fen, canonical=canonical)
    available = int(mask.sum())
    if available == 0:
        return []
    device = next(model.parameters()).device
    encoded = encode_fen(fen) if canonical else encode_fen_legacy(fen)
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(encoded).unsqueeze(0).to(device)
        logits = model(features)[0].masked_fill(
            torch.from_numpy(~mask).to(device), MASK_VALUE
        )
        probabilities = torch.softmax(logits, dim=0)
    values, indices = probabilities.topk(min(top_k, available))
    return [
        (index_to_move(int(index), board, canonical=canonical), float(value))
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]


__all__ = [
    "ChessPolicyNet",
    "LegacyChessPolicyNet",
    "SpatialMoveHead",
    "evaluate_model",
    "make_tensor_dataset",
    "pick_device",
    "predict_top_moves",
    "train_model",
    "unpack_legal_masks",
]
