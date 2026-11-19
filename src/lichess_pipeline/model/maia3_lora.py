"""Approach 3: fine-tune a pretrained Maia-3 with LoRA adapters.

Maia-3 (CSSLab, "Chessformer") is a transformer over the 64 squares that
predicts the move a human of a given Elo would play. This module keeps its
pretrained weights frozen and trains small low-rank adapters (LoRA) on top, so
the same base model can be specialised twice with very few trainable weights:
once on human moves and once on Stockfish best moves.

Three conventions come from Maia-3 itself and are easy to get wrong:

* **Mirroring.** The model always sees the side to move as White. When Black is
  to move the board is mirrored (ranks flipped, colours swapped) and so is the
  move: ``e7e5`` becomes ``e2e4``. Every helper here takes the original FEN and
  UCI move and mirrors internally.
* **Move vocabulary.** 4352 classes: the 4096 ``from -> to`` square pairs plus
  256 promotions (``from file``, ``to file``, piece in ``qrbn``), so
  underpromotions are real classes instead of always becoming a queen.
* **History.** The network takes the last eight positions. The cleaned datasets
  only keep the current FEN, so the position is repeated eight times, which is
  exactly what Maia-3's own inference does without a move history.

LoRA targets the MLP of every transformer block and the two policy-head
projections. The attention Q/K/V weights are fused inside
``nn.MultiheadAttention`` and are left frozen.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import polars as pl
import torch
from maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
from maia3.models import MAIA3Model
from maia3.utils import get_all_possible_moves, mirror_move
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ALL_MOVES: list[str] = get_all_possible_moves()
MOVE_TO_INDEX: dict[str, int] = {move: index for index, move in enumerate(ALL_MOVES)}
NUM_MOVE_CLASSES = len(ALL_MOVES)  # 4352
PIECE_PLANES = 12

DEFAULT_MODEL_NAME = "maia3-5m"
DEFAULT_ELO = 2500.0
DEFAULT_RANK = 8
DEFAULT_ALPHA = 16.0
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGETS = ("linear1", "linear2", "proj_sq_from", "proj_sq_to")
DEFAULT_EPOCHS = 3
DEFAULT_LR = 5e-4
DEFAULT_BATCH_SIZE = 256
MASK_VALUE = -1e9


# --------------------------------------------------------------------------
# LoRA
# --------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank update ``B @ A``.

    ``B`` starts at zero, so a freshly wrapped layer returns exactly what the
    pretrained layer returned; training only moves the low-rank part.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        dropout: float = DEFAULT_LORA_DROPOUT,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return self.base(x) + self.scaling * update


def apply_lora(
    model: nn.Module,
    rank: int = DEFAULT_RANK,
    alpha: float = DEFAULT_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
) -> list[str]:
    """Freeze ``model`` and wrap its target linear layers with LoRA, in place.

    Returns the dotted names of the wrapped layers. Only the LoRA matrices
    require gradients afterwards.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                lora.to(child.weight.device)
                setattr(parent, child_name, lora)
                wrapped.append(f"{parent_name}.{child_name}".lstrip("."))
    if not wrapped:
        raise ValueError(f"No linear layers named {targets} were found to adapt")
    model.lora_config = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "targets": list(targets),
    }
    return wrapped


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter weights: a few megabytes instead of the whole model."""
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a" in name or ".lora_b" in name
    }


def save_adapter(model: nn.Module, path: Path, base_model: str, **extra: Any) -> Path:
    """Write the LoRA weights and the settings needed to rebuild them."""
    config = getattr(model, "lora_config", None)
    if config is None:
        raise ValueError("The model has no LoRA adapter; call apply_lora first")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model": base_model,
        "lora_config": config,
        "extra": extra,
        "state": lora_state_dict(model),
    }
    torch.save(payload, path)
    return path


def load_adapter(model: nn.Module, path: Path) -> dict[str, Any]:
    """Apply a saved adapter to a freshly loaded base model, in place.

    Returns the saved metadata (``base_model``, ``lora_config``, ``extra``).
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["lora_config"]
    apply_lora(
        model,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config["dropout"]),
        targets=tuple(config["targets"]),
    )
    missing, unexpected = model.load_state_dict(payload["state"], strict=False)
    not_loaded = [name for name in missing if ".lora_" in name]
    if not_loaded or unexpected:
        raise ValueError(
            f"Adapter does not match the model: missing {not_loaded}, "
            f"unexpected {list(unexpected)}"
        )
    model.eval()
    return {key: payload[key] for key in ("base_model", "lora_config", "extra")}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


# --------------------------------------------------------------------------
# Loading the pretrained model
# --------------------------------------------------------------------------


def build_maia3(config: dict[str, Any]) -> MAIA3Model:
    """Build an untrained Maia-3 from an architecture config dict."""
    return MAIA3Model(SimpleNamespace(**config))


def pick_device(device: str | torch.device | None = None) -> torch.device:
    """Use the requested device, or CUDA when it is available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_maia3(
    name: str = DEFAULT_MODEL_NAME,
    device: str | torch.device | None = None,
    cache_dir: Path | None = None,
) -> MAIA3Model:
    """Download (once) and load an official pretrained Maia-3 in eval mode."""
    spec = resolve_model_spec(name)
    model = build_maia3(spec.config)
    checkpoint_path = resolve_checkpoint_path(
        spec, cache_dir=str(cache_dir) if cache_dir is not None else None
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    # Older checkpoints call the attention-bias generator "smolgen".
    state = {key.replace("smolgen", "gab"): value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} lacks weights: {missing[:5]}")
    del unexpected  # extra training-only tensors are harmless
    return model.to(pick_device(device)).eval()


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


# --------------------------------------------------------------------------
# Board and move encoding
# --------------------------------------------------------------------------


def encode_fen(fen: str) -> np.ndarray:
    """One-hot ``(64, 12)`` board seen from the side to move.

    Rows are python-chess squares (``a1`` = 0 ... ``h8`` = 63) of the mirrored
    board; columns are the six piece types of the side to move followed by the
    opponent's.
    """
    board = chess.Board(fen)
    if board.turn == chess.BLACK:
        board = board.mirror()
    planes = np.zeros((chess.SQUARES[-1] + 1, PIECE_PLANES), dtype=np.uint8)
    for square, piece in board.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        planes[square, piece.piece_type - 1 + offset] = 1
    return planes


def move_to_index(fen: str, move_uci: str) -> int | None:
    """Class of ``move_uci`` in ``fen``, or ``None`` if outside the vocabulary."""
    if chess.Board(fen).turn == chess.BLACK:
        move_uci = mirror_move(move_uci)
    return MOVE_TO_INDEX.get(move_uci)


def index_to_move(fen: str, index: int) -> str:
    """UCI move of class ``index`` in the real (un-mirrored) position."""
    if not 0 <= index < NUM_MOVE_CLASSES:
        raise ValueError(f"move index out of range: {index}")
    move_uci = ALL_MOVES[index]
    if chess.Board(fen).turn == chess.BLACK:
        move_uci = mirror_move(move_uci)
    return move_uci


def legal_move_mask(fen: str) -> np.ndarray:
    """Boolean ``(4352,)`` mask of the legal moves of ``fen``."""
    board = chess.Board(fen)
    black_to_move = board.turn == chess.BLACK
    mask = np.zeros(NUM_MOVE_CLASSES, dtype=bool)
    for move in board.legal_moves:
        move_uci = mirror_move(move.uci()) if black_to_move else move.uci()
        index = MOVE_TO_INDEX.get(move_uci)
        if index is not None:
            mask[index] = True
    return mask


def make_maia_dataset(
    df: pl.DataFrame,
    fen_col: str = "fen",
    move_col: str = "move_uci",
    elo_col: str | None = None,
    default_elo: float = DEFAULT_ELO,
) -> tuple[TensorDataset, int]:
    """Encode rows into ``(board, elo, move class, legal mask)`` tensors.

    ``elo_col`` gives a per-row rating (the human games carry ``avg_elo``);
    without it every row uses ``default_elo``, which is how the Stockfish
    positions are presented to the model. Rows whose target move is not a legal
    move of the position are dropped; the second return value counts them.
    """
    fens = df[fen_col].to_list()
    moves = df[move_col].to_list()
    elos = df[elo_col].to_list() if elo_col is not None else [None] * len(fens)
    # Preallocated: the legal masks dominate memory (about 4 KB per position).
    boards = np.zeros((len(fens), 64, PIECE_PLANES), dtype=np.uint8)
    masks = np.zeros((len(fens), NUM_MOVE_CLASSES), dtype=bool)
    targets: list[int] = []
    ratings: list[float] = []
    for fen, move, elo in zip(fens, moves, elos, strict=True):
        index = move_to_index(fen, move)
        mask = legal_move_mask(fen)
        if index is None or not mask[index]:
            continue
        boards[len(targets)] = encode_fen(fen)
        masks[len(targets)] = mask
        targets.append(index)
        ratings.append(float(elo) if elo is not None and elo > 0 else default_elo)
    kept = len(targets)
    if kept == 0:
        raise ValueError("Cannot build a Maia dataset without any usable row")
    dataset = TensorDataset(
        torch.from_numpy(boards[:kept]),
        torch.tensor(ratings, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
        torch.from_numpy(masks[:kept]),
    )
    return dataset, len(fens) - kept


# --------------------------------------------------------------------------
# Training, evaluation and inference
# --------------------------------------------------------------------------


def _masked_logits(
    model: nn.Module, boards: torch.Tensor, elos: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    """Run the model on a batch and hide the illegal moves.

    The stored ``(B, 64, 12)`` boards are expanded here to the eight-position
    history the network expects, by repeating the current position.
    """
    device = _model_device(model)
    boards = boards.to(device, non_blocking=True)
    elos = elos.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)
    tokens = boards.float().repeat(1, 1, model.cfg.history)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits, _, _ = model(tokens, elos, elos)
    return logits.float().masked_fill(~masks, MASK_VALUE)


def evaluate(
    model: nn.Module, dataset: TensorDataset, batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, float]:
    """Legal-move cross-entropy plus Top-1 / Top-3 accuracy on ``dataset``."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate on an empty dataset")
    device = _model_device(model)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    top1 = 0
    top3 = 0
    with torch.no_grad():
        for boards, elos, targets, masks in DataLoader(dataset, batch_size=batch_size):
            logits = _masked_logits(model, boards, elos, masks)
            targets = targets.to(device)
            total_loss += criterion(logits, targets).item()
            hits = logits.topk(3, dim=1).indices == targets.unsqueeze(1)
            top1 += int(hits[:, 0].sum())
            top3 += int(hits.any(dim=1).sum())
    model.train(was_training)
    total = len(dataset)
    return {"loss": total_loss / total, "top1": top1 / total, "top3": top3 / total}


def train_lora(
    model: nn.Module,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    weight_decay: float = 0.0,
    keep_best: bool = True,
) -> list[dict[str, float]]:
    """Train the LoRA weights of ``model`` and return one metrics dict per epoch.

    With ``keep_best`` the adapter is restored to the epoch with the lowest
    validation loss (``model.lora_best_epoch``; 0 means the pretrained model was
    never beaten), so report final numbers on a separate test split.

    The history rows use the same keys as ``scratch_net.train_model`` so both
    approaches can be plotted and compared with the same code.
    """
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation datasets must not be empty")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Nothing to train: call apply_lora on the model first")
    device = _model_device(model)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    history: list[dict[str, float]] = []
    # Epoch 0 is the untouched adapter; it stays the answer if training never
    # beats it on validation, which can happen with a small sample.
    best_loss = evaluate(model, val_ds, batch_size)["loss"] if keep_best else math.inf
    best_state = lora_state_dict(model) if keep_best else None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for boards, elos, targets, masks in loader:
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(_masked_logits(model, boards, elos, masks), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
        metrics = evaluate(model, val_ds, batch_size)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / seen,
                "val_loss": metrics["loss"],
                "val_top1": metrics["top1"],
                "val_top3": metrics["top3"],
            }
        )
        if keep_best and metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_state = lora_state_dict(model)
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.lora_best_epoch = best_epoch if keep_best else epochs
    model.eval()
    return history


def predict_top_moves(
    model: nn.Module, fen: str, top_k: int = 3, elo: float = DEFAULT_ELO
) -> list[tuple[str, float]]:
    """Return up to ``top_k`` ``(uci, probability)`` legal moves for ``fen``.

    A terminal position has no legal moves and returns an empty list.
    """
    mask = legal_move_mask(fen)
    available = int(mask.sum())
    if available == 0:
        return []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = _masked_logits(
            model,
            torch.from_numpy(encode_fen(fen)).unsqueeze(0),
            torch.tensor([elo], dtype=torch.float32),
            torch.from_numpy(mask).unsqueeze(0),
        )[0]
        probabilities = torch.softmax(logits, dim=0)
    model.train(was_training)
    values, indices = probabilities.topk(min(top_k, available))
    return [
        (index_to_move(fen, int(index)), float(value))
        for value, index in zip(values.tolist(), indices.tolist(), strict=True)
    ]
