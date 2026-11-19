"""Head-only LoRA on the from-scratch policy network.

The adapter is only the low-rank update to the 4096-class move head, because
that head is where nearly all of the parameters live. The base convolutions stay
exactly as Stage 1 left them, so a fresh adapter reproduces Stage 1 outputs.
"""

import chess
import polars as pl
import pytest
import torch
from torch import nn

from lichess_pipeline.model.scratch_lora import (
    LoRAPolicyNet,
    apply_head_lora,
    count_trainable_parameters,
    merge_head_lora,
    train_head_lora,
)
from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    make_tensor_dataset,
    predict_top_moves,
)

TINY_CHANNELS = (8, 8)


def tiny_model() -> ChessPolicyNet:
    return ChessPolicyNet(channels=TINY_CHANNELS)


def tiny_dataset(size: int = 4):
    frame = pl.DataFrame(
        {
            "fen": [chess.STARTING_FEN] * size,
            "move_uci": ["e2e4", "d2d4", "g1f3", "c2c4"] * (size // 4),
        }
    )
    return make_tensor_dataset(frame, "fen", "move_uci")


def test_fresh_head_lora_keeps_stage_one_outputs_and_freezes_the_base():
    model = tiny_model()
    features = torch.rand(3, 13, 8, 8)
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


def test_head_lora_rejects_a_non_positive_rank():
    with pytest.raises(ValueError, match="rank"):
        apply_head_lora(tiny_model(), rank=0)


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


def test_train_head_lora_moves_the_adapter_and_never_the_base():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    dataset = tiny_dataset()
    base_before = model.base.head.weight.detach().clone()
    buffers_before = {
        name: buffer.detach().clone() for name, buffer in model.base.named_buffers()
    }
    adapter_before = model.adapter.lora_b.detach().clone()

    train_head_lora(model, dataset, dataset, epochs=2, lr=1e-2, batch_size=4)

    # The frozen Stage 1 weights survive; merging is what writes the update back.
    assert torch.allclose(base_before, model.base.head.weight)
    assert not model.base.head.weight.requires_grad
    assert not torch.allclose(adapter_before, model.adapter.lora_b)
    # BatchNorm statistics must not drift either: the base stays in eval mode.
    for name, buffer in model.base.named_buffers():
        assert torch.equal(buffers_before[name], buffer)


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
    features = torch.rand(5, 13, 8, 8)
    with torch.no_grad():
        expected = model(features)
        actual = merged(features)

    assert isinstance(merged, ChessPolicyNet)
    assert torch.allclose(expected, actual, atol=1e-4)


def test_merged_model_saves_and_loads_as_a_plain_chess_policy_net():
    model = apply_head_lora(tiny_model(), rank=4, alpha=8.0)
    merged = merge_head_lora(model)

    reloaded = ChessPolicyNet(channels=TINY_CHANNELS)
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


def test_head_lora_adapter_uses_the_requested_rank():
    model = apply_head_lora(tiny_model(), rank=16, alpha=32.0)
    head = model.base.head

    assert isinstance(head, nn.Linear)
    assert model.adapter.lora_a.shape == (16, head.in_features)
    assert model.adapter.lora_b.shape == (head.out_features, 16)
