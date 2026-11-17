"""Maia-3 + LoRA helpers, tested on a tiny random model (no weights download)."""

from pathlib import Path

import chess
import polars as pl
import pytest
import torch

from lichess_pipeline.model.maia3_lora import (
    NUM_MOVE_CLASSES,
    LoRALinear,
    apply_lora,
    build_maia3,
    count_parameters,
    encode_fen,
    evaluate,
    index_to_move,
    legal_move_mask,
    load_adapter,
    make_maia_dataset,
    move_to_index,
    predict_top_moves,
    save_adapter,
    train_lora,
)

TINY_CONFIG = {
    "history": 8,
    "use_padding": True,
    "include_time_info": False,
    "dim_emb": 8,
    "num_blocks": 2,
    "mlp_ratio": 2.0,
    "dropout": 0.0,
    "use_gab": True,
    "use_relative_bias": False,
    "use_absolute_pe": False,
    "use_rms_norm": True,
    "omit_qkv_biases": True,
    "activation": "gelu",
    "dim_vit": 32,
    "head_hid_dim": 32,
    "num_heads": 4,
    "gab_gen_size": 8,
    "gab_per_square_dim": 0,
    "gab_intermediate_dim": 8,
}

AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
WHITE_PROMOTES = "1n5k/P7/8/8/8/8/8/K7 w - - 0 1"
BLACK_PROMOTES = "k7/8/8/8/8/8/p7/1N5K b - - 0 1"
WHITE_CAN_CASTLE = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
FOOLS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


def tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    return build_maia3(TINY_CONFIG).eval()


def random_positions(count: int, seed: int = 0) -> pl.DataFrame:
    """Positions from random playouts, labelled with the first legal move."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    board = chess.Board()
    while len(rows) < count:
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            board = chess.Board()
            continue
        rows.append({"fen": board.fen(), "move_uci": moves[0].uci(), "elo": 2000.0})
        board.push(moves[int(torch.randint(len(moves), (1,), generator=generator))])
    return pl.DataFrame(rows)


def test_black_moves_are_mirrored_and_round_trip():
    index = move_to_index(AFTER_E4, "e7e5")

    assert index == move_to_index(chess.STARTING_FEN, "e2e4")
    assert index_to_move(AFTER_E4, index) == "e7e5"


def test_encode_fen_shows_the_side_to_move_as_white():
    white_view = encode_fen(chess.STARTING_FEN)
    black_view = encode_fen(chess.Board(chess.STARTING_FEN).mirror().fen())

    assert white_view.shape == (64, 12)
    assert white_view.sum() == 32
    assert (white_view == black_view).all()
    # a1 holds a rook of the side to move: piece type 4 -> plane 3.
    assert white_view[chess.A1, 3] == 1


@pytest.mark.parametrize(
    ("fen", "moves"),
    [
        (WHITE_PROMOTES, ["a7a8q", "a7a8n", "a7b8r", "a7b8b"]),
        (BLACK_PROMOTES, ["a2a1q", "a2a1n", "a2b1r", "a2b1b"]),
        (WHITE_CAN_CASTLE, ["e1g1", "e1c1"]),
    ],
)
def test_promotions_and_castling_have_their_own_legal_classes(fen, moves):
    mask = legal_move_mask(fen)
    indices = [move_to_index(fen, move) for move in moves]

    assert len(set(indices)) == len(moves)
    assert all(mask[index] for index in indices)
    assert [index_to_move(fen, index) for index in indices] == moves


def test_legal_mask_matches_python_chess_on_random_positions():
    for fen in random_positions(60, seed=3)["fen"].to_list():
        mask = legal_move_mask(fen)
        decoded = {index_to_move(fen, int(index)) for index in mask.nonzero()[0]}

        assert mask.shape == (NUM_MOVE_CLASSES,)
        assert decoded == {move.uci() for move in chess.Board(fen).legal_moves}


def test_fresh_lora_keeps_the_output_and_trains_only_the_adapter():
    model = tiny_model()
    tokens = torch.rand(3, 64, 97)
    elos = torch.tensor([1500.0, 2000.0, 2500.0])
    before = model(tokens, elos, elos)[0]

    wrapped = apply_lora(model, rank=4, alpha=8)
    after = model(tokens, elos, elos)[0]
    trainable, total = count_parameters(model)

    # Two MLP layers per block plus the two policy projections.
    assert len(wrapped) == 2 * TINY_CONFIG["num_blocks"] + 2
    assert torch.allclose(before, after, atol=1e-6)
    assert 0 < trainable < total
    assert all(
        ".lora_" in name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_lora_linear_rejects_a_non_positive_rank():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(torch.nn.Linear(4, 4), rank=0)


def test_make_maia_dataset_drops_rows_with_illegal_targets():
    frame = pl.DataFrame(
        {
            "fen": [chess.STARTING_FEN, chess.STARTING_FEN, AFTER_E4],
            "move_uci": ["e2e4", "e2e5", "e7e5"],
            "elo": [1900.0, 1900.0, None],
        }
    )

    dataset, skipped = make_maia_dataset(frame, elo_col="elo", default_elo=2500.0)
    boards, elos, targets, masks = dataset.tensors

    assert skipped == 1
    assert boards.shape == (2, 64, 12)
    assert masks.shape == (2, NUM_MOVE_CLASSES)
    assert elos.tolist() == [1900.0, 2500.0]
    assert targets[0] == targets[1]  # e2e4 and the mirrored e7e5


def test_make_maia_dataset_rejects_a_frame_without_usable_rows():
    frame = pl.DataFrame({"fen": [chess.STARTING_FEN], "move_uci": ["e2e5"]})

    with pytest.raises(ValueError, match="usable row"):
        make_maia_dataset(frame)


def test_predict_top_moves_returns_legal_moves_and_handles_mate():
    model = tiny_model()

    for fen in (chess.STARTING_FEN, AFTER_E4, WHITE_PROMOTES, BLACK_PROMOTES):
        predictions = predict_top_moves(model, fen, top_k=3)
        legal = {move.uci() for move in chess.Board(fen).legal_moves}

        assert len(predictions) == 3
        assert all(move in legal for move, _ in predictions)
        assert all(0.0 <= probability <= 1.0 for _, probability in predictions)
    assert predict_top_moves(model, FOOLS_MATE) == []


def test_training_lowers_the_loss_and_the_adapter_round_trips(tmp_path: Path):
    frame = random_positions(48)
    train_ds, _ = make_maia_dataset(frame.head(40), elo_col="elo")
    val_ds, _ = make_maia_dataset(frame.tail(8), elo_col="elo")
    model = tiny_model()
    apply_lora(model, rank=4, alpha=8, dropout=0.0)
    before = evaluate(model, train_ds)

    history = train_lora(model, train_ds, val_ds, epochs=8, lr=2e-2, batch_size=16)
    after = evaluate(model, train_ds)
    path = save_adapter(model, tmp_path / "adapter.pt", base_model="tiny", note="t")

    restored = tiny_model()
    metadata = load_adapter(restored, path)

    assert len(history) == 8
    assert set(history[0]) == {
        "epoch",
        "train_loss",
        "val_loss",
        "val_top1",
        "val_top3",
    }
    assert after["loss"] < before["loss"]
    assert metadata["base_model"] == "tiny"
    assert metadata["lora_config"]["rank"] == 4
    assert metadata["extra"] == {"note": "t"}
    assert evaluate(restored, train_ds)["loss"] == pytest.approx(
        after["loss"], abs=1e-5
    )


def test_training_without_an_adapter_is_rejected():
    frame = random_positions(8)
    dataset, _ = make_maia_dataset(frame)
    model = tiny_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(ValueError, match="apply_lora"):
        train_lora(model, dataset, dataset, epochs=1)
