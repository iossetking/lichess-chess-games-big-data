"""Locate, save and load every move-prediction checkpoint the project produces.

Three checkpoint families are recognised under ``data/models/``:

* **Spatial v3** ``ChessPolicyNet`` (18 planes, residual tower, factorised
  head) — what notebooks 02 and 04 train from now on;
* **Legacy v1/v2** ``LegacyChessPolicyNet`` (13 planes, flat 4096-way head) —
  kept so earlier checkpoints keep loading;
* **Maia-3 LoRA adapters** saved by notebook 03 (``maia3_lora_human.pt``,
  ``maia3_lora_stockfish.pt``), wrapped in :class:`Maia3Policy` so they expose
  the same ``predict_top_moves(fen, top_k)`` contract.

New checkpoints are written in half precision (``float16``) by
:func:`save_policy_checkpoint`: the same network at half the file size.
Integer buffers such as BatchNorm's ``num_batches_tracked`` keep their dtype,
and ``load_state_dict`` copies half tensors back into ``float32`` parameters,
so predictions are practically unchanged.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    LegacyChessPolicyNet,
    pick_device,
)

CHECKPOINT_SUFFIXES = {".pt", ".pth"}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_BLOCK_CONV = re.compile(r"^blocks\.(\d+)\.conv1\.weight$")
_LEGACY_CONV = re.compile(r"^features\.(\d+)\.weight$")
_STEM_CONV = re.compile(r"^stem\.\d+\.weight$")


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def compact_state_dict(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return a half-precision copy of ``state`` for storage.

    Only floating-point tensors are halved; integer buffers (BatchNorm's
    ``num_batches_tracked``) must keep their dtype or the checkpoint would no
    longer load. The original tensors are left untouched.
    """
    compact: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        detached = tensor.detach().cpu()
        compact[name] = (
            detached.half() if detached.is_floating_point() else detached.clone()
        )
    return compact


def save_policy_checkpoint(model: nn.Module, path: Path) -> Path:
    """Write ``model`` as a compact half-precision state dict.

    The parent directory is created when missing. Use this instead of
    ``torch.save(model.state_dict(), path)`` so every new checkpoint stays
    small enough for Git and fast to load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact_state_dict(model.state_dict()), path)
    return path


# --------------------------------------------------------------------------
# Discovering and loading
# --------------------------------------------------------------------------


def discover_policy_checkpoints(models_dir: Path) -> dict[str, Path]:
    """Return ``label -> path`` for every checkpoint under ``models_dir``.

    Labels are the relative path without the suffix, using ``/`` as separator,
    so ``runs/full-data/lora_v2.pt`` becomes ``runs/full-data/lora_v2``.
    Hidden directories are skipped and the result is sorted by label.
    """
    if not models_dir.is_dir():
        return {}
    found: dict[str, Path] = {}
    for path in models_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
            continue
        relative = path.relative_to(models_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        found[relative.with_suffix("").as_posix()] = path
    return dict(sorted(found.items()))


def is_lfs_pointer(path: Path) -> bool:
    """True when ``path`` is a Git LFS pointer instead of the real checkpoint.

    Clones without Git LFS leave a small text file in place of the weights;
    detecting it produces a clear message instead of an unpickling traceback.
    """
    if path.stat().st_size > 1024:
        return False
    with path.open("rb") as stream:
        return stream.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX


def is_maia_adapter(payload: Any) -> bool:
    """True when a loaded ``.pt`` payload is a Maia-3 LoRA adapter bundle."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("state"), dict)
        and "lora_config" in payload
    )


def infer_policy_config(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Describe the architecture that ``state`` belongs to.

    Returns ``{"kind": "spatial", "channels": int, "depth": int}`` for the
    current network or ``{"kind": "legacy", "channels": tuple}`` for the
    original three-block version. Used to rebuild the exact same model before
    ``load_state_dict``.
    """
    stem_key = next((name for name in state if _STEM_CONV.match(name)), None)
    if stem_key is not None:
        depth = 0
        for name in state:
            match = _BLOCK_CONV.match(name)
            if match is not None:
                depth = max(depth, int(match.group(1)) + 1)
        return {
            "kind": "spatial",
            "channels": int(state[stem_key].shape[0]),
            "depth": depth,
        }
    widths: list[tuple[int, int]] = []
    for name, tensor in state.items():
        match = _LEGACY_CONV.match(name)
        if match is not None and tensor.dim() == 4:
            widths.append((int(match.group(1)), int(tensor.shape[0])))
    if widths:
        return {
            "kind": "legacy",
            "channels": tuple(width for _, width in sorted(widths)),
        }
    raise ValueError("the file has no ChessPolicyNet weights")


def build_policy_net(config: dict[str, Any]) -> nn.Module:
    """Build the empty network described by :func:`infer_policy_config`."""
    if config["kind"] == "spatial":
        return ChessPolicyNet(channels=config["channels"], depth=config["depth"])
    return LegacyChessPolicyNet(channels=tuple(config["channels"]))


class Maia3Policy:
    """Give a Maia-3 LoRA checkpoint the ``ChessPolicyNet`` inference interface."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device

    def eval(self) -> Maia3Policy:
        self.model.eval()
        return self

    def predict_top_moves(self, fen: str, top_k: int = 3) -> list[tuple[str, float]]:
        from lichess_pipeline.model.maia3_lora import (
            predict_top_moves as maia_predict,
        )

        return maia_predict(self.model, fen, top_k=top_k)


def load_maia3_policy(path: Path, device: str | None = None) -> Maia3Policy:
    """Rebuild a pretrained Maia-3 and apply the saved LoRA adapter."""
    from lichess_pipeline.model.maia3_lora import (
        DEFAULT_MODEL_NAME,
        load_adapter,
        load_maia3,
    )
    from lichess_pipeline.model.maia3_lora import pick_device as maia_device

    payload = torch.load(path, map_location="cpu", weights_only=True)
    base_name = str(payload.get("base_model") or DEFAULT_MODEL_NAME)
    target = maia_device(device)
    model = load_maia3(base_name, device=target)
    load_adapter(model, path)
    return Maia3Policy(model.eval(), target)


def load_policy_checkpoint(path: Path, device: str | None = None) -> Any:
    """Load ``path`` into a ready-to-use predictor.

    The architecture (spatial v3, legacy, or Maia-3 LoRA) comes from the file
    itself. A corrupt, incompatible or un-downloaded (Git LFS) file raises
    ``RuntimeError`` naming the file, so the notebook can report which
    checkpoint failed instead of showing PyTorch's raw unpickling traceback.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"Checkpoint inválido: {path.name}: es un puntero de Git LFS sin "
            "descargar; ejecuta `git lfs pull`"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if is_maia_adapter(payload):
            return load_maia3_policy(path, device=device)
        model = build_policy_net(infer_policy_config(payload))
        model.load_state_dict(payload)
        model.to(pick_device(device))
    except Exception as error:
        detail = str(error).splitlines()[0]
        if isinstance(error, (pickle.UnpicklingError, EOFError)):
            detail = "no contiene un state_dict de PyTorch válido"
        raise RuntimeError(f"Checkpoint inválido: {path.name}: {detail}") from error
    return model.eval()
