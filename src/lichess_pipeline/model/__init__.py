"""Move-prediction models that share one contract: FEN in, legal UCI move out."""

from lichess_pipeline.model.maia_adapter import MaiaResidualPolicyNet
from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    evaluate_model,
    make_tensor_dataset,
    predict_top_moves,
    train_model,
)

__all__ = [
    "ChessPolicyNet",
    "MaiaResidualPolicyNet",
    "evaluate_model",
    "make_tensor_dataset",
    "predict_top_moves",
    "train_model",
]
