"""Locate and load the move-prediction checkpoints saved for notebook 04.

Every ``.pt``/``.pth`` file under ``data/models/`` is a candidate, including
files in manually named run folders. The notebook lists them by their path
relative to ``data/models/`` and lets the user choose which ones to compare.
The files are plain PyTorch state dicts for
:class:`~lichess_pipeline.model.scratch_net.ChessPolicyNet`; the LoRA adapter is
merged before saving, so those checkpoints load through the same path too.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import torch

from lichess_pipeline.model.scratch_net import ChessPolicyNet

CHECKPOINT_SUFFIXES = {".pt", ".pth"}


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


def load_policy_checkpoint(path: Path) -> ChessPolicyNet:
    """Load ``path`` into a default ``ChessPolicyNet`` ready for inference.

    A corrupt or incompatible file raises ``RuntimeError`` naming the file, so
    the notebook can report which checkpoint failed instead of showing PyTorch's
    raw unpickling traceback.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = ChessPolicyNet()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except Exception as error:
        detail = str(error).splitlines()[0]
        if isinstance(error, (pickle.UnpicklingError, EOFError)):
            detail = "no contiene un state_dict de PyTorch válido"
        raise RuntimeError(f"Checkpoint inválido: {path.name}: {detail}") from error
    return model.eval()
