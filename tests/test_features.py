import chess
import numpy as np

from lichess_pipeline.features import (
    NUM_MOVE_CLASSES,
    NUM_PLANES,
    canonical_move,
    encode_fen,
    encode_fen_legacy,
    encode_position,
    index_to_move,
    legal_move_mask,
    mirror_square,
    move_to_index,
)


def test_encode_fen_starting_board_uses_eighteen_planes():
    tensor = encode_fen(chess.STARTING_FEN)
    assert tensor.shape == (NUM_PLANES, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor[:6].sum() == 16.0
    assert tensor[6:12].sum() == 16.0
    # White is to move: all four castling rights are available.
    assert np.all(tensor[12:16] == 1.0)
    assert tensor[16].sum() == 0.0  # no en passant target
    assert np.all(tensor[17] == 1.0)


def test_encode_fen_places_white_king_on_e1():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    tensor = encode_fen(fen)
    # Black to move: the board is mirrored, so the black king is the "own" king.
    assert tensor[5, 7, 4] == 1.0  # black king -> e1 after mirroring
    assert tensor[11, 0, 4] == 1.0  # white king -> e8 after mirroring
    assert np.all(tensor[12:16] == 1.0)
    assert np.all(tensor[17] == 0.0)


def test_encoding_is_invariant_under_colour_mirroring():
    board = chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )
    mirrored = board.mirror()

    assert np.array_equal(encode_fen(board.fen())[:17], encode_fen(mirrored.fen())[:17])


def test_black_king_on_e8_is_e1_on_the_canonical_board():
    fen = "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    tensor = encode_fen(fen)
    assert tensor[5, 7, 4] == 1.0  # black (side to move) king -> e1
    assert tensor[11, 0, 4] == 1.0  # white king -> e8


def test_legacy_encoding_keeps_the_thirteen_plane_layout():
    tensor = encode_fen_legacy(chess.STARTING_FEN)
    assert tensor.shape == (13, 8, 8)
    assert tensor[5, 7, 4] == 1.0  # white king on e1, un-mirrored
    assert np.all(tensor[12] == 1.0)


def test_en_passant_and_castling_planes_are_filled():
    tensor = encode_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert tensor[16, 2, 3] == 1.0  # d6 -> row 2, column 3
    assert tensor[12].sum() == 0.0 and tensor[13].sum() == 0.0


def test_canonical_move_mirrors_black_moves():
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    assert canonical_move(chess.STARTING_FEN, "e2e4") == "e2e4"
    assert canonical_move(fen, "e7e5") == "e2e4"
    assert mirror_square(chess.E8) == chess.E1


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


def test_legal_move_mask_mirrors_black_moves():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    canonical = legal_move_mask(fen)
    legacy = legal_move_mask(fen, canonical=False)

    # Black's e7e5 is class e2e4 in canonical space but e7e5 in absolute space.
    assert canonical[move_to_index("e2e4")]
    assert not canonical[move_to_index("e7e5")]
    assert legacy[move_to_index("e7e5")]
    assert int(canonical.sum()) == int(legacy.sum()) == 20


def test_legal_move_mask_is_empty_on_checkmate():
    fools_mate = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert int(legal_move_mask(fools_mate).sum()) == 0


def test_encode_position_matches_individual_helpers():
    cases = [
        (chess.STARTING_FEN, "e2e4"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e7e5"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("8/4P3/8/8/8/8/8/k6K w - - 0 1", "e7e8q"),
    ]
    for fen, move_uci in cases:
        features, target, packed_mask = encode_position(fen, move_uci)
        expected_features = encode_fen(fen).astype(np.uint8)
        expected_target = move_to_index(canonical_move(fen, move_uci))
        expected_mask = np.packbits(legal_move_mask(fen))

        assert features.dtype == np.uint8
        assert np.array_equal(features, expected_features)
        assert target == expected_target
        assert packed_mask.dtype == np.uint8
        assert np.array_equal(packed_mask, expected_mask)
