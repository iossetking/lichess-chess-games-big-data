"""Locate and load the move-prediction checkpoints saved by notebook 02.

The three ``approach1_*.pt`` files are plain PyTorch state dicts for
:class:`~lichess_pipeline.model.scratch_net.ChessPolicyNet`; the LoRA adapter is
merged before saving, so all three load through the same path. Checkpoints may
live in ``data/models/`` or in ``data/models/backups/`` after a retrain.
"""

from __future__ import annotations

from pathlib import Path

import torch

from lichess_pipeline.model.scratch_net import ChessPolicyNet

POLICY_CHECKPOINTS: dict[str, str] = {
    "Desde cero": "approach1_scratch.pt",
    "Ajuste de pesos": "approach1_weight_ftuning.pt",
    "Ajuste LoRA": "approach1_lora_ftuning.pt",
}


def discover_policy_checkpoints(models_dir: Path) -> dict[str, Path]:
    """Return ``label -> path`` for the policy checkpoints that exist.

    The main ``models_dir`` wins over ``models_dir/backups``. Only the known
    filenames are considered, so YOLO and other ``.pt`` files never appear.
    """
    found: dict[str, Path] = {}
    for label, filename in POLICY_CHECKPOINTS.items():
        for directory in (models_dir, models_dir / "backups"):
            candidate = directory / filename
            if candidate.is_file():
                found[label] = candidate
                break
    return found


def load_policy_checkpoint(path: Path) -> ChessPolicyNet:
    """Load ``path`` into a default ``ChessPolicyNet`` ready for inference."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = ChessPolicyNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()
