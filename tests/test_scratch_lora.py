"""LoRA fine-tuning across the residual tower and the spatial head.

The adapters are low-rank updates on every residual convolution, the two head
projections and the 64x64 geometry bias. The Stage 1 weights stay exactly as
they were, so a fresh adapter reproduces Stage 1 outputs.
"""

import chess
import polars as pl
import pytest
import torch
from torch import nn

from lichess_pipeline.model.scratch_lora import (
    LoRABias,
    LoRAConv2d,
    LoRALinear,
    LoRAPolicyNet,
    apply_head_lora,
    count_trainable_parameters,
    merge_head_lora,
    train_head_lora,
)
from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    make_tensor_dataset,
    pick_device,
    predict_top_moves,
)

TINY_CHANNELS = 8
TINY_DEPTH = 1


def tiny_model() -> ChessPolicyNet:
    return ChessPolicyNet(channels=TINY_CHANNELS, depth=TINY_DEPTH)


def tiny_dataset(size: int = 4):
    frame = pl.DataFrame(
        {
            "fen": [chess.STARTING_FEN] * size,
            "move_uci": ["e2e4", "d2d4", "g1f3", "c2c4"] * (size // 4),
        }
    )
    return make_tensor_dataset(frame, "fen", "move_uci")


def test_fresh_lora_keeps_stage_one_outputs_and_freezes_the_base():
    model = tiny_model()
    features = torch.rand(3, 18, 8, 8)
    before = model(features)

    wrapped = apply_head_lora(model, rank=4, alpha=8.0)
    after = wrapped(features)
    trainable, total = count_trainable_parameters(wrapped)

    assert isinstance(wrapped, LoRAPolicyNet)
    assert torch.allclose(before, after, atol=1e-6)
    assert 0 < trainable < total
    assert all(
        ".lora_" in name
        for name, parameter in wrapped.named_parameters()
        if parameter.requires_grad
    )
    # The tower keeps learning capacity: every residual conv has an adapter.
    assert isinstance(wrapped.base.blocks[0].conv1, LoRAConv2d)
    assert isinstance(wrapped.base.blocks[0].conv2, LoRAConv2d)
    assert isinstance(wrapped.base.head.proj_from, LoRALinear)
    assert isinstance(wrapped.base.head.spatial_bias, LoRABias)


def test_lora_adapts_the_whole_network_not_only_the_head():
    wrapped = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    trainable, total = count_trainable_parameters(wrapped)

    # A head-only adapter would be a tiny fraction of the 1.2M-parameter model.
    assert trainable > total * 0.05


def test_head_lora_rejects_a_non_positive_rank():
    with pytest.raises(ValueError, match="rank"):
        apply_head_lora(tiny_model(), rank=0)


def test_head_lora_rejects_a_network_without_the_spatial_head():
    from lichess_pipeline.model.maia_adapter import MaiaResidualPolicyNet

    with pytest.raises(TypeError, match="ChessPolicyNet"):
        apply_head_lora(MaiaResidualPolicyNet(channels=8, num_blocks=1))


def test_train_head_lora_returns_scratch_style_history_and_a_best_epoch():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    dataset = tiny_dataset()

    history = train_head_lora(model, dataset, dataset, epochs=2, lr=1e-2, batch_size=4)

    assert len(history) == 2
    assert {"epoch", "train_loss", "val_loss", "val_top1", "val_top3"} <= set(
        history[-1]
    )
    assert 0 <= model.best_epoch <= 2
    # The restored adapter must reproduce the metrics recorded for its epoch.
    if model.best_epoch:
        best = history[model.best_epoch - 1]
        assert best["val_loss"] == pytest.approx(model.best_val_loss)


def test_train_head_lora_moves_the_adapters_and_never_the_base():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    dataset = tiny_dataset()
    base_convs = model.base.blocks[0].conv1.base.weight.detach().clone()
    base_head = model.base.head.proj_from.base.weight.detach().clone()
    base_bias = model.base.head.spatial_bias.base.detach().clone()
    buffers_before = {
        name: buffer.detach().clone() for name, buffer in model.base.named_buffers()
    }
    adapter_before = model.base.blocks[0].conv1.lora_b.detach().clone()

    train_head_lora(model, dataset, dataset, epochs=2, lr=1e-2, batch_size=4)

    # The frozen Stage 1 weights survive; merging is what writes the update back.
    assert torch.allclose(base_convs, model.base.blocks[0].conv1.base.weight.cpu())
    assert torch.allclose(base_head, model.base.head.proj_from.base.weight.cpu())
    assert torch.allclose(base_bias, model.base.head.spatial_bias.base.cpu())
    assert not model.base.blocks[0].conv1.base.weight.requires_grad
    assert not torch.allclose(adapter_before, model.base.blocks[0].conv1.lora_b.cpu())
    assert not torch.allclose(
        torch.zeros_like(model.base.head.spatial_bias.lora_b),
        model.base.head.spatial_bias.lora_b,
    )
    # BatchNorm statistics must not drift either: the base stays in eval mode.
    for name, buffer in model.base.named_buffers():
        assert torch.equal(buffers_before[name], buffer.cpu())


def test_train_head_lora_can_skip_best_epoch_selection():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    dataset = tiny_dataset()

    history = train_head_lora(
        model, dataset, dataset, epochs=2, lr=1e-2, batch_size=4, keep_best=False
    )

    assert model.best_epoch == 2
    assert model.best_val_loss == pytest.approx(history[-1]["val_loss"])


def test_train_head_lora_rejects_a_plain_network():
    with pytest.raises(TypeError, match="apply_head_lora"):
        train_head_lora(tiny_model(), tiny_dataset(), tiny_dataset(), epochs=1)


def test_merged_model_reproduces_the_adapter_network():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    dataset = tiny_dataset()
    train_head_lora(model, dataset, dataset, epochs=2, lr=1e-2, batch_size=4)

    merged = merge_head_lora(model)
    device = next(model.parameters()).device
    features = torch.rand(5, 18, 8, 8, device=device)
    with torch.no_grad():
        expected = model(features)
        actual = merged(features)

    assert isinstance(merged, ChessPolicyNet)
    assert isinstance(merged.blocks[0].conv1, nn.Conv2d)
    assert isinstance(merged.head.proj_from, nn.Linear)
    assert isinstance(merged.head.spatial_bias, nn.Parameter)
    assert torch.allclose(expected, actual, atol=1e-4)


def test_merged_model_saves_and_loads_as_a_plain_chess_policy_net():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    merged = merge_head_lora(model)

    reloaded = ChessPolicyNet(channels=TINY_CHANNELS, depth=TINY_DEPTH)
    missing = reloaded.load_state_dict(merged.state_dict())

    assert not missing.missing_keys
    assert not missing.unexpected_keys


def test_merged_model_predicts_legal_moves():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    merged = merge_head_lora(model)
    fen = chess.STARTING_FEN
    legal = {move.uci() for move in chess.Board(fen).legal_moves}

    predictions = predict_top_moves(merged, fen, top_k=3)

    assert len(predictions) == 3
    assert all(move in legal for move, _ in predictions)


def test_adapters_use_the_requested_rank_on_convs_and_projections():
    model = apply_head_lora(tiny_model(), rank=16, alpha=32.0)
    convolution = model.base.blocks[0].conv1
    projection = model.base.head.proj_to

    assert convolution.lora_a.shape == (16, TINY_CHANNELS * 3 * 3)
    assert convolution.lora_b.shape == (TINY_CHANNELS, 16)
    assert projection.lora_a.shape == (16, TINY_CHANNELS)
    assert model.base.head.spatial_bias.lora_a.shape == (16, 64)


def test_train_head_lora_moves_cpu_model_to_selected_device():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    dataset = tiny_dataset()

    train_head_lora(model, dataset, dataset, epochs=1, lr=1e-2, batch_size=4)
    assert next(model.parameters()).device.type == pick_device().type

    train_head_lora(
        model, dataset, dataset, epochs=1, lr=1e-2, batch_size=4, device="cpu"
    )
    assert next(model.parameters()).device.type == "cpu"
