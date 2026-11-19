"""Head-only LoRA fine-tuning for the from-scratch policy network.

Stage 1 trains :class:`~lichess_pipeline.model.scratch_net.ChessPolicyNet` from
scratch on human moves. Stage 2 then specialises it on Stockfish best moves.
Plain full-weight fine-tuning updates every parameter, which on a small
Stockfish sample can pull the network away from what Stage 1 learned. This
module offers the constrained alternative:

* the convolutions and the move head keep their Stage 1 weights, frozen;
* a low-rank update ``B @ A`` is added to the head output, and only ``A`` and
  ``B`` are trained;
* ``B`` starts at zero, so before training the adapted network is numerically
  identical to Stage 1;
* :func:`merge_head_lora` folds the adapter into a plain
  :class:`ChessPolicyNet`, which is what gets saved as a standalone checkpoint.

The head holds ``8192 x 4096`` weights, so adapting it alone already covers most
of the network's capacity while touching a tiny fraction of it (``rank`` 8 is
about 100k trainable parameters out of roughly 34M).
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lichess_pipeline.model.scratch_net import ChessPolicyNet

DEFAULT_RANK = 8
DEFAULT_ALPHA = 16.0
DEFAULT_DROPOUT = 0.05
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_WEIGHT_DECAY = 0.0


class LoRAHead(nn.Module):
    """A frozen move head plus a trainable low-rank correction.

    ``forward`` returns ``head(x) + scaling * dropout(x) @ A.T @ B.T``. ``B`` is
    initialised to zero so the adapter starts as an exact identity.
    """

    def __init__(
        self,
        head: nn.Linear,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.head = head
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, head.in_features))
        self.lora_b = nn.Parameter(torch.zeros(head.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return self.head(x) + self.scaling * update


class LoRAPolicyNet(nn.Module):
    """``ChessPolicyNet`` with its head replaced by a :class:`LoRAHead`.

    The wrapper exists only while training. Its output has the same shape and
    meaning as the base network's (raw logits over the 4096 move classes), so
    the training and evaluation helpers in ``scratch_net`` accept it unchanged.
    """

    def __init__(self, base: ChessPolicyNet, adapter: LoRAHead) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.lora_config = {
            "rank": int(adapter.lora_a.shape[0]),
            "alpha": float(adapter.scaling * adapter.lora_a.shape[0]),
            "dropout": float(adapter.dropout.p),
            "target": "head",
        }
        self.best_epoch = 0
        self.best_val_loss = math.inf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = torch.flatten(self.base.features(x), start_dim=1)
        return self.adapter(features)

    def train(self, mode: bool = True) -> LoRAPolicyNet:
        """Keep the frozen base in eval mode, including its BatchNorm buffers.

        ``requires_grad_(False)`` stops the optimiser but not BatchNorm, which
        keeps updating ``running_mean``/``running_var`` in training mode. Left
        alone, a "frozen" network would still change its outputs. Only the
        adapter follows the requested mode (for its dropout).
        """
        super().train(mode)
        self.base.eval()
        self.adapter.train(mode)
        return self


def apply_head_lora(
    model: ChessPolicyNet,
    rank: int = DEFAULT_RANK,
    alpha: float = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
) -> LoRAPolicyNet:
    """Freeze ``model`` and wrap it in a trainable head adapter, in place."""
    if rank < 1:
        raise ValueError("LoRA rank must be at least 1")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = LoRAHead(model.head, rank=rank, alpha=alpha, dropout=dropout)
    adapter.to(model.head.weight.device)
    return LoRAPolicyNet(model, adapter)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def train_head_lora(
    model: LoRAPolicyNet,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    keep_best: bool = True,
) -> list[dict[str, float]]:
    """Train only the adapter, optionally keeping the best validation epoch.

    The history rows use the same keys as ``scratch_net.train_model`` so the
    three approaches can share plotting code. With ``keep_best`` the adapter is
    restored to the epoch with the lowest validation loss (``model.best_epoch``;
    0 means no epoch beat the untouched Stage 1 head). Set ``keep_best=False``
    for the final refit, where the epoch count already comes from
    cross-validation and the validation set is only used for monitoring.
    """
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation datasets must not be empty")
    if not isinstance(model, LoRAPolicyNet):
        raise TypeError("Expected a LoRAPolicyNet: call apply_head_lora first")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Nothing to train: call apply_head_lora first")

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    history: list[dict[str, float]] = []

    best_loss = (
        _validation_metrics(model, val_loader, criterion)["loss"]
        if keep_best
        else math.inf
    )
    best_state = _adapter_state(model) if keep_best else None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, targets, masks in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _masked_logits(model, features, masks)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
        metrics = _validation_metrics(model, val_loader, criterion)
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
            best_state = _adapter_state(model)
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.best_epoch = best_epoch if keep_best else epochs
    model.best_val_loss = best_loss if keep_best else history[-1]["val_loss"]
    model.eval()
    return history


def merge_head_lora(model: LoRAPolicyNet) -> ChessPolicyNet:
    """Fold the adapter into a standalone :class:`ChessPolicyNet`.

    Returns a deep copy of the frozen base whose head weight includes
    ``scaling * B @ A``. The result has plain ``ChessPolicyNet`` keys, so it can
    be saved and reloaded without any LoRA bookkeeping.
    """
    merged = copy.deepcopy(model.base)
    adapter = model.adapter
    with torch.no_grad():
        merged.head.weight.add_(adapter.scaling * (adapter.lora_b @ adapter.lora_a))
    for parameter in merged.parameters():
        parameter.requires_grad_(True)
    merged.eval()
    return merged


# --------------------------------------------------------------------------
# Batch helpers
# --------------------------------------------------------------------------

MASK_VALUE = -1e9


def _masked_logits(
    model: nn.Module, features: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    logits = model(features)
    return logits.masked_fill(~masks.to(torch.bool), MASK_VALUE)


def _validation_metrics(
    model: LoRAPolicyNet, loader: DataLoader, criterion: nn.Module
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    with torch.no_grad():
        for features, targets, masks in loader:
            logits = _masked_logits(model, features, masks)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            top_k = logits.topk(3, dim=1).indices
            hits = top_k == targets.unsqueeze(1)
            correct_top1 += int(hits[:, 0].sum())
            correct_top3 += int(hits.any(dim=1).sum())
            total += targets.size(0)
    model.train(was_training)
    return {
        "loss": total_loss / total,
        "top1": correct_top1 / total,
        "top3": correct_top3 / total,
    }


def _adapter_state(model: LoRAPolicyNet) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".lora_" in name
    }
