import chess
import pytest
import torch

from lichess_pipeline.model.checkpoints import (
    discover_policy_checkpoints,
    load_policy_checkpoint,
)
from lichess_pipeline.model.scratch_net import ChessPolicyNet, predict_top_moves


def test_discover_labels_nested_checkpoints_by_folder_and_filename(tmp_path):
    (tmp_path / "runs" / "full-data").mkdir(parents=True)
    (tmp_path / "runs" / "full-data" / "approach1_lora_v2.pt").write_bytes(b"a")
    (tmp_path / "runs" / "small-data").mkdir(parents=True)
    (tmp_path / "runs" / "small-data" / "scratch_v1.pth").write_bytes(b"b")
    (tmp_path / "legacy.pt").write_bytes(b"c")
    (tmp_path / "notes.txt").write_text("not a checkpoint")

    found = discover_policy_checkpoints(tmp_path)

    assert found == {
        "legacy": tmp_path / "legacy.pt",
        "runs/full-data/approach1_lora_v2": (
            tmp_path / "runs" / "full-data" / "approach1_lora_v2.pt"
        ),
        "runs/small-data/scratch_v1": (
            tmp_path / "runs" / "small-data" / "scratch_v1.pth"
        ),
    }


def test_discover_keeps_checkpoints_from_several_folders(tmp_path):
    for folder, name in (
        ("runs/a", "scratch_v1.pt"),
        ("runs/b", "scratch_v2.pt"),
        ("backups", "scratch_v0.pt"),
    ):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / name).write_bytes(b"weights")

    found = discover_policy_checkpoints(tmp_path)

    assert list(found) == [
        "backups/scratch_v0",
        "runs/a/scratch_v1",
        "runs/b/scratch_v2",
    ]


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


def test_load_corrupt_checkpoint_names_the_file(tmp_path):
    path = tmp_path / "approach1_scratch.pt"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(RuntimeError, match="approach1_scratch.pt"):
        load_policy_checkpoint(path)


def test_load_incompatible_state_dict_names_the_file(tmp_path):
    path = tmp_path / "approach1_weight_ftuning.pt"
    torch.save({"unexpected": torch.zeros(3)}, path)

    with pytest.raises(RuntimeError, match="approach1_weight_ftuning.pt"):
        load_policy_checkpoint(path)
