import chess
import polars as pl

from lichess_pipeline.model.maia_adapter import MaiaResidualPolicyNet
from lichess_pipeline.model.scratch_net import (
    ChessPolicyNet,
    make_tensor_dataset,
    predict_top_moves,
    train_model,
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
    history_stage1 = train_model(model, dataset, dataset, epochs=2, lr=1e-2, batch_size=8)
    assert len(history_stage1) == 2
    assert {"epoch", "train_loss", "val_loss", "val_top1", "val_top3"} <= set(
        history_stage1[-1]
    )

    history_stage2 = train_model(model, dataset, dataset, epochs=1, lr=1e-4, batch_size=8)
    assert len(history_stage2) == 1

    maia_model = MaiaResidualPolicyNet(channels=32, num_blocks=2)
    maia_history = train_model(maia_model, dataset, dataset, epochs=1, lr=1e-3, batch_size=8)
    assert len(maia_history) == 1
    assert len(predict_top_moves(maia_model, chess.STARTING_FEN, top_k=3)) == 3
