import chess
import pytest
import torch

from lichess_pipeline.model.checkpoints import (
    POLICY_CHECKPOINTS,
    discover_policy_checkpoints,
    load_policy_checkpoint,
)
from lichess_pipeline.model.scratch_net import ChessPolicyNet, predict_top_moves


def test_discover_prefers_main_dir_and_ignores_other_checkpoints(tmp_path):
    for name in POLICY_CHECKPOINTS.values():
        (tmp_path / "backups").mkdir(exist_ok=True)
        (tmp_path / "backups" / name).write_bytes(b"backup")
    main = tmp_path / POLICY_CHECKPOINTS["Desde cero"]
    main.write_bytes(b"main")
    (tmp_path / "chess-pieces.pt").write_bytes(b"vision")

    found = discover_policy_checkpoints(tmp_path)

    assert found == {
        "Desde cero": main,
        "Ajuste de pesos": tmp_path / "backups" / POLICY_CHECKPOINTS["Ajuste de pesos"],
        "Ajuste LoRA": tmp_path / "backups" / POLICY_CHECKPOINTS["Ajuste LoRA"],
    }


def test_discover_returns_empty_mapping_for_missing_directory(tmp_path):
    assert discover_policy_checkpoints(tmp_path / "absent") == {}


def test_load_round_trips_predictions_and_switches_to_eval(tmp_path):
    model = ChessPolicyNet()
    torch.manual_seed(0)
    path = tmp_path / "approach1_scratch.pt"
    torch.save(model.state_dict(), path)

    loaded = load_policy_checkpoint(path)

    assert loaded.training is False
    assert predict_top_moves(loaded, chess.STARTING_FEN, top_k=3) == predict_top_moves(
        model, chess.STARTING_FEN, top_k=3
    )


def test_load_missing_checkpoint_names_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="approach1_scratch.pt"):
        load_policy_checkpoint(tmp_path / "approach1_scratch.pt")
