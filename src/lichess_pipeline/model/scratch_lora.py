"""LoRA fine-tuning of the whole from-scratch policy network.

Stage 1 trains :class:`~lichess_pipeline.model.scratch_net.ChessPolicyNet` from
scratch on human moves. Stage 2 then specialises it on Stockfish best moves.
Plain full-weight fine-tuning updates every parameter, which on a small
Stockfish sample can pull the network away from what Stage 1 learned. This
module offers the constrained alternative:

* every Stage 1 weight is frozen;
* a low-rank update ``B @ A`` is added to each residual convolution, to the two
  head projections and to the head's ``64 x 64`` geometry bias, and only those
  adapters are trained;
* the adapters start at zero, so before training the wrapped network is
  numerically identical to Stage 1;
* :func:`merge_head_lora` folds every adapter into plain ``nn.Conv2d`` /
  ``nn.Linear`` layers and a plain bias parameter, which is what gets saved as
  a standalone checkpoint.

The spatial head only holds ~1.7 % of the parameters, so adapting it alone
would freeze the residual tower that extracts the tactical patterns. Adapting
the conv blocks and the head gives ~86k trainable parameters at rank 8 (about
7 % of the network), the same scale Maia-3 uses while staying cheap.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    pick_device,
    unpack_legal_masks,
)

DEFAULT_RANK = 8
DEFAULT_ALPHA = 16.0
DEFAULT_DROPOUT = 0.05
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_WEIGHT_DECAY = 0.0
HEAD_TARGETS = ("proj_from", "proj_to", "spatial_bias")


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable low-rank correction."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        dropout: float = DEFAULT_DROPOUT,
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

    def merged(self) -> nn.Linear:
        """Fold the adapter into a plain ``nn.Linear``."""
        base = self.base
        plain = nn.Linear(
            base.in_features, base.out_features, bias=base.bias is not None
        ).to(base.weight.device)
        with torch.no_grad():
            plain.weight.copy_(base.weight + self.scaling * (self.lora_b @ self.lora_a))
            if base.bias is not None:
                plain.bias.copy_(base.bias)
        return plain


class LoRAConv2d(nn.Module):
    """A frozen ``nn.Conv2d`` plus a low-rank update on its flattened kernel."""

    def __init__(
        self,
        base: nn.Conv2d,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.scaling = alpha / rank
        fan_in = base.in_channels * base.kernel_size[0] * base.kernel_size[1]
        self.lora_a = nn.Parameter(torch.empty(rank, fan_in))
        self.lora_b = nn.Parameter(torch.zeros(base.out_channels, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def _delta_kernel(self) -> torch.Tensor:
        kernel = self.base.kernel_size
        return (self.lora_b @ self.lora_a).reshape(
            self.base.out_channels, self.base.in_channels, *kernel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = functional.conv2d(
            self.dropout(x),
            self.scaling * self._delta_kernel(),
            stride=self.base.stride,
            padding=self.base.padding,
            dilation=self.base.dilation,
            groups=self.base.groups,
        )
        return self.base(x) + update

    def merged(self) -> nn.Conv2d:
        """Fold the adapter into a plain ``nn.Conv2d``."""
        base = self.base
        plain = nn.Conv2d(
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            groups=base.groups,
            bias=base.bias is not None,
        ).to(base.weight.device)
        with torch.no_grad():
            plain.weight.copy_(base.weight + self.scaling * self._delta_kernel())
            if base.bias is not None:
                plain.bias.copy_(base.bias)
        return plain


class LoRABias(nn.Module):
    """A frozen ``64 x 64`` geometry bias plus a trainable low-rank update."""

    def __init__(
        self,
        base: torch.Tensor,
        rank: int = DEFAULT_RANK,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be at least 1")
        self.scaling = alpha / rank
        self.base = nn.Parameter(base.detach().clone(), requires_grad=False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.shape[0]))
        self.lora_b = nn.Parameter(torch.zeros(base.shape[1], rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self) -> torch.Tensor:
        return self.base + self.scaling * (self.lora_b @ self.lora_a)

    def merged(self) -> nn.Parameter:
        """Fold the adapter into a plain bias parameter."""
        return nn.Parameter(self().detach().clone(), requires_grad=True)


ADAPTER_TYPES = (LoRALinear, LoRAConv2d, LoRABias)


class LoRAPolicyNet(nn.Module):
    """``ChessPolicyNet`` with trainable adapters across its tower and head.

    The wrapper exists only while training. Its output has the same shape and
    meaning as the base network's (raw logits over the 4096 move classes), so
    the training and evaluation helpers in ``scratch_net`` accept it unchanged.
    """

    def __init__(self, base: ChessPolicyNet, targets: tuple[str, ...]) -> None:
        super().__init__()
        self.base = base
        self.targets = tuple(targets)
        adapter = _get_module(base, self.targets[0])
        self.lora_config = {
            "rank": int(adapter.lora_a.shape[0]),
            "alpha": float(adapter.scaling * adapter.lora_a.shape[0]),
            "dropout": float(getattr(getattr(adapter, "dropout", None), "p", 0.0)),
            "targets": list(self.targets),
        }
        self.best_epoch = 0
        self.best_val_loss = math.inf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def train(self, mode: bool = True) -> LoRAPolicyNet:
        """Keep the frozen network in eval mode, including BatchNorm buffers.

        ``requires_grad_(False)`` stops the optimiser but not BatchNorm, which
        keeps updating ``running_mean``/``running_var`` in training mode. Left
        alone, a "frozen" network would still change its outputs. Only the
        adapters follow the requested mode (for their dropout).
        """
        super().train(mode)
        self.base.eval()
        for module in self.base.modules():
            if isinstance(module, ADAPTER_TYPES):
                module.train(mode)
        return self


def _get_module(root: nn.Module, dotted: str) -> nn.Module:
    module = root
    for part in dotted.split("."):
        module = getattr(module, part)
    return module


def _set_module(root: nn.Module, dotted: str, replacement: nn.Module | nn.Parameter) -> None:
    parts = dotted.split(".")
    parent = _get_module(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], replacement)


def apply_head_lora(
    model: ChessPolicyNet,
    rank: int = DEFAULT_RANK,
    alpha: float = DEFAULT_ALPHA,
    dropout: float = DEFAULT_DROPOUT,
) -> LoRAPolicyNet:
    """Freeze ``model`` and wrap its convolutions and head with LoRA, in place.

    Kept name-compatible with the previous head-only version: the adapter now
    covers ``blocks.*.conv1``, ``blocks.*.conv2``, ``head.proj_from``,
    ``head.proj_to`` and ``head.spatial_bias``.
    """
    if rank < 1:
        raise ValueError("LoRA rank must be at least 1")
    if not isinstance(model, ChessPolicyNet):
        raise TypeError("Expected a ChessPolicyNet with a spatial move head")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets: list[str] = []
    for index, block in enumerate(model.blocks):
        for name in ("conv1", "conv2"):
            convolution = getattr(block, name)
            if not isinstance(convolution, nn.Conv2d):
                continue
            adapter = LoRAConv2d(convolution, rank=rank, alpha=alpha, dropout=dropout)
            setattr(block, name, adapter.to(convolution.weight.device))
            targets.append(f"blocks.{index}.{name}")
    for name in ("proj_from", "proj_to"):
        linear = getattr(model.head, name)
        if not isinstance(linear, nn.Linear):
            continue
        adapter = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
        setattr(model.head, name, adapter.to(linear.weight.device))
        targets.append(f"head.{name}")
    if isinstance(model.head.spatial_bias, torch.Tensor):
        bias = model.head.spatial_bias
        del model.head.spatial_bias  # a Module cannot replace a registered Parameter in place
        model.head.spatial_bias = LoRABias(bias, rank=rank, alpha=alpha).to(bias.device)
        targets.append("head.spatial_bias")
    if not targets:
        raise ValueError("The network has no layers to adapt")
    return LoRAPolicyNet(model, tuple(targets))


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


def train_head_lora(
    model: LoRAPolicyNet,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    keep_best: bool = True,
    device: str | torch.device | None = None,
) -> list[dict[str, float]]:
    """Train only the adapters, optionally keeping the best validation epoch.

    The history rows use the same keys as ``scratch_net.train_model`` so the
    three approaches can share plotting code. With ``keep_best`` the adapters
    are restored to the epoch with the lowest validation loss
    (``model.best_epoch``; 0 means no epoch beat the untouched Stage 1 network).
    Set ``keep_best=False`` for the final refit, where the epoch count already
    comes from cross-validation and validation is only used for monitoring.
    """
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train and validation datasets must not be empty")
    if not isinstance(model, LoRAPolicyNet):
        raise TypeError("Expected a LoRAPolicyNet: call apply_head_lora first")
    target_device = pick_device(device)
    model.to(target_device)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Nothing to train: call apply_head_lora first")

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(
        target_device.type, enabled=target_device.type == "cuda"
    )
    criterion = nn.CrossEntropyLoss()
    use_pin_memory = target_device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=use_pin_memory
    )
    history: list[dict[str, float]] = []

    best_loss = (
        _validation_metrics(model, val_loader, criterion, target_device)["loss"]
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
        metrics = _validation_metrics(model, val_loader, criterion, target_device)
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
    """Fold every adapter into a standalone, plain :class:`ChessPolicyNet`.

    Returns a deep copy of the frozen base whose convolutions, head projections
    and geometry bias include ``scaling * B @ A``. The result has plain
    ``ChessPolicyNet`` keys, so it can be saved and reloaded without any LoRA
    bookkeeping.
    """
    merged = copy.deepcopy(model.base)
    for target in model.targets:
        adapter = _get_module(merged, target)
        if not isinstance(adapter, ADAPTER_TYPES):
            continue
        _set_module(merged, target, adapter.merged())
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
    with torch.autocast(
        device_type=features.device.type, enabled=features.device.type == "cuda"
    ):
        logits = model(features.to(dtype=torch.float32))
    return logits.float().masked_fill(~unpack_legal_masks(masks), MASK_VALUE)


def _validation_metrics(
    model: LoRAPolicyNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
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


__all__ = [
    "LoRABias",
    "LoRAConv2d",
    "LoRALinear",
    "LoRAPolicyNet",
    "apply_head_lora",
    "count_trainable_parameters",
    "merge_head_lora",
    "train_head_lora",
]
