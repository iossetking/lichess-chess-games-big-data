import chess
import pytest
import torch

from lichess_pipeline.model.checkpoints import (
    compact_state_dict,
    discover_policy_checkpoints,
    infer_policy_config,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    LegacyChessPolicyNet,
    predict_top_moves,
)

TINY_CHANNELS = 16
TINY_DEPTH = 1


def tiny_model() -> ChessPolicyNet:
    return ChessPolicyNet(channels=TINY_CHANNELS, depth=TINY_DEPTH)


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
    torch.manual_seed(0)
    model = ChessPolicyNet().eval()
    path = tmp_path / "approach1_scratch.pt"
    torch.save(model.state_dict(), path)

    loaded = load_policy_checkpoint(path, device="cpu")

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


def test_compact_state_dict_halves_floats_and_keeps_integer_buffers():
    state = {"weight": torch.ones(2), "steps": torch.tensor(5)}

    compact = compact_state_dict(state)

    assert compact["weight"].dtype == torch.float16
    assert compact["steps"].dtype == torch.int64
    assert state["weight"].dtype == torch.float32


def test_save_policy_checkpoint_is_smaller_than_float32_and_still_loads(tmp_path):
    model = tiny_model()
    half_path = save_policy_checkpoint(model, tmp_path / "nested" / "scratch.pt")
    float_path = tmp_path / "scratch-fp32.pt"
    torch.save(model.state_dict(), float_path)

    state = torch.load(half_path, map_location="cpu", weights_only=True)

    assert half_path.is_file()
    assert state["stem.0.weight"].dtype == torch.float16
    assert half_path.stat().st_size < float_path.stat().st_size
    half_bytes = sum(t.numel() * t.element_size() for t in state.values())
    float_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    assert half_bytes < float_bytes * 0.55
    assert load_policy_checkpoint(half_path, device="cpu").training is False


def test_saved_checkpoint_keeps_predictions_close(tmp_path):
    torch.manual_seed(0)
    model = tiny_model().eval()
    loaded = load_policy_checkpoint(
        save_policy_checkpoint(model, tmp_path / "scratch.pt"), device="cpu"
    )
    features = torch.rand(3, 18, 8, 8)

    with torch.no_grad():
        assert torch.allclose(model(features), loaded(features), atol=1e-2)


def test_load_rebuilds_the_trained_width_and_depth(tmp_path):
    model = ChessPolicyNet(channels=16, depth=2)
    path = save_policy_checkpoint(model, tmp_path / "tiny.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)

    loaded = load_policy_checkpoint(path)

    assert infer_policy_config(state) == {
        "kind": "spatial",
        "channels": 16,
        "depth": 2,
    }
    assert loaded.stem[0].out_channels == 16
    assert len(loaded.blocks) == 2
    assert predict_top_moves(loaded, chess.STARTING_FEN, top_k=1)


def test_load_rebuilds_a_legacy_checkpoint_with_its_own_encoding(tmp_path):
    torch.manual_seed(0)
    legacy = LegacyChessPolicyNet(channels=(8, 8)).eval()
    path = tmp_path / "legacy_v2.pt"
    torch.save(legacy.state_dict(), path)

    loaded = load_policy_checkpoint(path)

    assert isinstance(loaded, LegacyChessPolicyNet)
    assert infer_policy_config(legacy.state_dict())["kind"] == "legacy"
    assert predict_top_moves(loaded, chess.STARTING_FEN, top_k=3) == predict_top_moves(
        legacy, chess.STARTING_FEN, top_k=3
    )


def test_load_reports_an_unpulled_git_lfs_pointer(tmp_path):
    path = tmp_path / "approach1_scratch.pt"
    path.write_bytes(
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        b"size 123\n"
    )

    with pytest.raises(RuntimeError, match="git lfs pull"):
        load_policy_checkpoint(path)


def test_maia_adapter_payloads_are_detected_separately():
    from lichess_pipeline.model.checkpoints import is_maia_adapter

    payload = {
        "base_model": "maia3-5m",
        "lora_config": {"rank": 8, "targets": ["linear1"]},
        "extra": {},
        "state": {"blocks.0.linear1.lora_a": torch.zeros(8, 4)},
    }

    assert is_maia_adapter(payload)
    assert not is_maia_adapter({"stem.0.weight": torch.zeros(2)})
    assert not is_maia_adapter([1, 2, 3])
