"""Locate, save and load the move-prediction checkpoints for notebooks 02 and 04.

Every ``.pt``/``.pth`` file under ``data/models/`` is a candidate, including
files in manually named run folders. The notebook lists them by their path
relative to ``data/models/`` and lets the user choose which ones to compare.
The files are PyTorch state dicts for
:class:`~lichess_pipeline.model.scratch_net.ChessPolicyNet`; the LoRA adapter is
merged before saving, so those checkpoints load through the same path too.

Checkpoints are written in half precision (``float16``): the same network with
half the file size. Integer buffers such as BatchNorm's
``num_batches_tracked`` keep their dtype, and ``load_state_dict`` copies the
half tensors back into the network's ``float32`` parameters, so predictions are
practically unchanged. :func:`load_policy_checkpoint` reads the convolution
widths from the file, so checkpoints trained with a different ``channels``
tuple keep loading without retraining.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import torch
from torch import nn

from lichess_pipeline.model.scratch_net import ChessPolicyNet

CHECKPOINT_SUFFIXES = {".pt", ".pth"}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_CONV_WEIGHT = re.compile(r"^features\.(\d+)\.weight$")


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


def infer_channels(state: dict[str, torch.Tensor]) -> tuple[int, ...]:
    """Read the ``ChessPolicyNet`` convolution widths from a state dict.

    Sorted by the block index in ``features.{i}.weight``, so a checkpoint saved
    with any custom ``channels`` tuple can rebuild the exact same network.
    """
    widths: list[tuple[int, int]] = []
    for name, tensor in state.items():
        match = _CONV_WEIGHT.match(name)
        if match is not None and tensor.dim() == 4:
            widths.append((int(match.group(1)), int(tensor.shape[0])))
    if not widths:
        raise ValueError("the file has no ChessPolicyNet convolutions")
    return tuple(width for _, width in sorted(widths))


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


def load_policy_checkpoint(path: Path) -> ChessPolicyNet:
    """Load ``path`` into a fresh ``ChessPolicyNet`` ready for inference.

    The convolution widths come from the checkpoint, so networks trained with
    a different ``channels`` tuple load too. A corrupt, incompatible or
    un-downloaded (Git LFS) file raises ``RuntimeError`` naming the file, so
    the notebook can report which checkpoint failed instead of showing
    PyTorch's raw unpickling traceback.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"Checkpoint inválido: {path.name}: es un puntero de Git LFS sin "
            "descargar; ejecuta `git lfs pull`"
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = ChessPolicyNet(channels=infer_channels(state))
        model.load_state_dict(state)
    except Exception as error:
        detail = str(error).splitlines()[0]
        if isinstance(error, (pickle.UnpicklingError, EOFError)):
            detail = "no contiene un state_dict de PyTorch válido"
        raise RuntimeError(f"Checkpoint inválido: {path.name}: {detail}") from error
    return model.eval()
