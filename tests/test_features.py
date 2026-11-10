import chess
import numpy as np

from lichess_pipeline.features import (
    NUM_MOVE_CLASSES,
    encode_fen,
    index_to_move,
    legal_move_mask,
    move_to_index,
)


def test_encode_fen_starting_board():
    tensor = encode_fen(chess.STARTING_FEN)
    assert tensor.shape == (13, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor[:6].sum() == 16.0
    assert tensor[6:12].sum() == 16.0
    assert np.all(tensor[12] == 1.0)


def test_encode_fen_places_white_king_on_e1_and_black_side_plane():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    tensor = encode_fen(fen)
    assert tensor[5, 7, 4] == 1.0  # white king -> e1 -> row 7, column 4
    assert tensor[11, 0, 4] == 1.0  # black king -> e8 -> row 0, column 4
    assert np.all(tensor[12] == 0.0)


def test_move_index_roundtrip_and_pawn_promotion():
    idx = move_to_index("e2e4")
    assert 0 <= idx < NUM_MOVE_CLASSES
    assert index_to_move(idx) == "e2e4"

    promo_fen = "8/4P3/8/8/8/8/8/k6K w - - 0 1"
    board = chess.Board(promo_fen)
    promo_idx = move_to_index("e7e8q")
    assert index_to_move(promo_idx, board) == "e7e8q"
    assert index_to_move(promo_idx) == "e7e8"


def test_legal_move_mask_starting_position_has_20_moves():
    mask = legal_move_mask(chess.STARTING_FEN)
    assert mask.shape == (NUM_MOVE_CLASSES,)
    assert int(mask.sum()) == 20
    assert mask[move_to_index("e2e4")]
    assert not mask[move_to_index("e2e5")]


def test_legal_move_mask_is_empty_on_checkmate():
    fools_mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert int(legal_move_mask(fools_mate).sum()) == 0
