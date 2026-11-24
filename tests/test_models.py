import chess
import polars as pl
import pytest
import torch

from lichess_pipeline.features import legal_move_mask
from lichess_pipeline.model.maia_adapter import MaiaResidualPolicyNet
from lichess_pipeline.model.scratch_net import (
    PACKED_MASK_BYTES,
    ChessPolicyNet,
    evaluate_model,
    make_tensor_dataset,
    predict_top_moves,
    train_model,
    unpack_legal_masks,
)


def test_predict_top_moves_always_returns_legal_moves():
    model = ChessPolicyNet()
    fen = chess.STARTING_FEN
    legal_ucis = {move.uci() for move in chess.Board(fen).legal_moves}

    preds = predict_top_moves(model, fen, top_k=3)
    assert len(preds) == 3
    for move_uci, probability in preds:
        assert move_uci in legal_ucis
        assert 0.0 <= probability <= 1.0


def test_predict_top_moves_on_checkmate_returns_empty_list():
    model = ChessPolicyNet()
    fools_mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert predict_top_moves(model, fools_mate, top_k=3) == []


def test_two_stage_training_and_maia_adapter_run_on_mini_sample():
    sample_df = pl.DataFrame(
        {
            "fen": [
                chess.STARTING_FEN,
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ]
            * 8,
            "move_uci": ["e2e4", "e7e5"] * 8,
        }
    )
    dataset = make_tensor_dataset(sample_df, "fen", "move_uci")

    model = ChessPolicyNet()
    history_stage1 = train_model(
        model, dataset, dataset, epochs=2, lr=1e-2, batch_size=8
    )
    assert len(history_stage1) == 2
    assert {"epoch", "train_loss", "val_loss", "val_top1", "val_top3"} <= set(
        history_stage1[-1]
    )

    history_stage2 = train_model(
        model, dataset, dataset, epochs=1, lr=1e-4, batch_size=8
    )
    assert len(history_stage2) == 1

    maia_model = MaiaResidualPolicyNet(channels=32, num_blocks=2)
    maia_history = train_model(
        maia_model, dataset, dataset, epochs=1, lr=1e-3, batch_size=8
    )
    assert len(maia_history) == 1
    assert len(predict_top_moves(maia_model, chess.STARTING_FEN, top_k=3)) == 3


def test_evaluate_model_reports_loss_and_top_k_accuracy():
    sample_df = pl.DataFrame(
        {
            "fen": [chess.STARTING_FEN] * 4,
            "move_uci": ["e2e4", "d2d4", "g1f3", "c2c4"],
        }
    )
    dataset = make_tensor_dataset(sample_df, "fen", "move_uci")
    model = ChessPolicyNet()
    metrics = evaluate_model(model, dataset, batch_size=4)
    assert set(metrics) == {"loss", "top1", "top3"}
    assert metrics["top3"] >= metrics["top1"]
    assert 0.0 <= metrics["top1"] <= 1.0


def test_make_tensor_dataset_uses_compact_uint8_and_bitpacked_masks():
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    ]
    frame = pl.DataFrame({"fen": fens, "move_uci": ["e2e4", "e7e5"]})
    dataset = make_tensor_dataset(frame)
    features, targets, packed_masks = dataset.tensors

    assert features.dtype == torch.uint8
    assert features.shape == (2, 18, 8, 8)
    assert targets.dtype == torch.long
    assert packed_masks.dtype == torch.uint8
    assert packed_masks.shape == (2, PACKED_MASK_BYTES)

    unpacked = unpack_legal_masks(packed_masks)
    assert unpacked.dtype == torch.bool
    for idx, fen in enumerate(fens):
        expected = torch.from_numpy(legal_move_mask(fen))
        assert torch.equal(unpacked[idx], expected)


def test_train_and_evaluate_accept_subset_without_copying_tensors():
    sample_df = pl.DataFrame(
        {
            "fen": [
                chess.STARTING_FEN,
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            ]
            * 6,
            "move_uci": ["e2e4", "e7e5"] * 6,
        }
    )
    dataset = make_tensor_dataset(sample_df)
    train_sub = torch.utils.data.Subset(dataset, list(range(8)))
    val_sub = torch.utils.data.Subset(dataset, list(range(8, 12)))

    model = ChessPolicyNet(channels=8, depth=1)
    history = train_model(model, train_sub, val_sub, epochs=1, batch_size=4)
    metrics = evaluate_model(model, val_sub, batch_size=4)

    assert len(history) == 1
    assert set(metrics) == {"loss", "top1", "top3"}


def test_make_tensor_dataset_rejects_an_empty_frame():
    empty = pl.DataFrame(schema={"fen": pl.String, "move_uci": pl.String})
    with pytest.raises(ValueError, match="empty"):
        make_tensor_dataset(empty, "fen", "move_uci")


def test_train_model_rejects_an_empty_dataset():
    empty = torch.utils.data.TensorDataset(
        torch.zeros((0, 18, 8, 8)),
        torch.zeros((0,), dtype=torch.long),
        torch.zeros((0, 4096), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="empty"):
        train_model(ChessPolicyNet(), empty, empty, epochs=1, batch_size=8)
