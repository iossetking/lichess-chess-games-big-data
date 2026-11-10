# Official Lichess Pipeline & Move-Prediction Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh, end-to-end Lichess data ingestion, EDA, cleaning, and FEN-to-best-move prediction pipeline in Marimo notebooks (with reusable Python modules) using the official `database.lichess.org` `.pgn.zst` and `lichess_db_eval.jsonl.zst` datasets.

**Architecture:** Stream or download `.pgn.zst` and `lichess_db_eval.jsonl.zst` directly from `database.lichess.org` using `zstandard` + `urllib`, parse and clean records into local Parquet samples in `data/processed/` with Polars and `python-chess`, and train/compare two FEN-based move prediction models: (1) a primary compact PyTorch CNN trained from scratch on high-Elo human games and fine-tuned on Stockfish evaluations, and (2) a secondary notebook adapting pretrained Maia chess weights. All predictions apply a `python-chess` legal-move mask over the 4096 (`64x64`) move logits so 100% of predicted moves are legal.

**Tech Stack:** Python 3.14 (`uv`), `polars`, `chess` (`python-chess`), `zstandard`, `torch`, `numpy`, `matplotlib`, `marimo`.

**Spec:** Approved inline design specification from the 2026-11-09 brainstorming conversation.

## Global Constraints

- Do NOT modify or delete existing vision files: `src/lichess_pipeline/chess_vision.py`, `scripts/chessboard_camera_demo.py`, `scripts/generate_aruco_markers.py`, `tests/test_chess_vision.py`, `docs/vision_tecnica_tablero.md`, `docs/Documento_Tecnico_Vision_Tablero_Ajedrez.docx`.
- Remove obsolete Hugging Face notebooks and helpers (`notebooks/eda.py`, `notebooks/eda_evaluaciones.py`, `notebooks/eda_helpers.py`, `notebooks/lectura.py`, `notebooks/test.py`, `notebooks/data_ingestion.py`, `scripts/step1_throughput.py`).
- Prediction model input is strictly a **FEN string** (`fen: str`); output is a **legal UCI move string** (`str`, e.g. `"e2e4"`, `"e7e8q"`).
- Default cleaning filters for `.pgn.zst` games: `WhiteElo >= 1800`, `BlackElo >= 1800`, `Termination == "Normal"`, `Result` in `{"1-0", "0-1", "1/2-1/2"}`, `num_plies >= 10`, and `Event` not containing `"Bullet"` or `"UltraBullet"`.
- Default cleaning filters for `lichess_db_eval.jsonl.zst`: keep the highest-depth entry in `evals` per unique normalized FEN (first 4 FEN fields), require `depth >= 20`, and require `first_move_is_legal(fen, best_move) == True`.

## Review Focus

- **Pawn promotions in `64x64` move encoding:** A pawn moving to rank 1 or rank 8 has a 5-character UCI string (`e7e8q`), whereas `from_sq * 64 + to_sq` encodes only squares. Decoding an index back to UCI on a board where a pawn lands on the back rank must append `"q"` so `python-chess` recognizes it as legal.
- **Interrupted HTTP stream on `.zst` files:** Streaming only the first N games/evaluations from `database.lichess.org` closes the HTTP stream mid-frame; the generator must cleanly close the stream inside a `finally` block without raising `zstandard.ZstdError` on partial trailing chunks.
- **PGN movetext with clock/eval annotations and move numbers:** Official Lichess PGNs contain `{ [%eval 0.23] [%clk 0:03:00] }` comments and `1...` black move numbers; the parser must strip comments cleanly before replaying moves.
- **Positions with no legal moves (checkmate/stalemate FEN):** Calling `predict_top_moves` on a terminal FEN must return an empty list `[]` rather than crashing or returning an illegal move.
- **Malformed or missing `pvs` in `lichess_db_eval.jsonl.zst`:** Lines with empty `evals` or missing `line` must return `None` from the parser rather than raising `KeyError`/`IndexError`.

---

### Task 1: Project Cleanup, Dependencies & Official Lichess `.zst` Ingestion

**Files:**
- Modify: `pyproject.toml`
- Delete: `notebooks/eda.py`, `notebooks/eda_evaluaciones.py`, `notebooks/eda_helpers.py`, `notebooks/lectura.py`, `notebooks/test.py`, `notebooks/data_ingestion.py`, `scripts/step1_throughput.py`, `tests/test_eda_helpers.py`
- Replace: `src/lichess_pipeline/data_ingestion.py`
- Replace: `tests/test_data_ingestion.py`

**Interfaces:**
- Produces in `src/lichess_pipeline/data_ingestion.py`:
  - `OFFICIAL_GAMES_URL_TEMPLATE = "https://database.lichess.org/standard/lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"`
  - `OFFICIAL_EVALS_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"`
  - `games_month_url(year: int, month: int) -> str`
  - `stream_zst_lines(source: str | Path, max_lines: int | None = None) -> Iterator[str]`
  - `stream_pgn_blocks(source: str | Path, max_games: int) -> Iterator[str]`
  - `download_sample_to_disk(url: str, dest_path: Path, max_bytes: int = 50_000_000) -> Path`

- [ ] **Step 1: Update dependencies in `pyproject.toml` and remove old Hugging Face notebooks**

Add `"zstandard>=0.23.0"` and `"torch>=2.6.0"` to `dependencies` in `pyproject.toml` (removing unused `"datasets"` and `"huggingface-hub"`), run `uv sync`, and delete the old Hugging Face notebooks (`notebooks/eda.py`, `notebooks/eda_evaluaciones.py`, `notebooks/eda_helpers.py`, `notebooks/lectura.py`, `notebooks/test.py`, `notebooks/data_ingestion.py`, `scripts/step1_throughput.py`, `tests/test_eda_helpers.py`).

- [ ] **Step 2: Write failing tests in `tests/test_data_ingestion.py`**

```python
from pathlib import Path
import zstandard as zstd
from lichess_pipeline.data_ingestion import (
    games_month_url,
    stream_pgn_blocks,
    stream_zst_lines,
)

SAMPLE_PGN = (
    '[Event "Rated Blitz game"]\n[Site "https://lichess.org/abcd1234"]\n'
    '[WhiteElo "1950"]\n[BlackElo "1920"]\n[Result "1-0"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 1-0\n\n"
    '[Event "Rated Rapid game"]\n[Site "https://lichess.org/efgh5678"]\n'
    '[WhiteElo "2100"]\n[BlackElo "2050"]\n[Result "0-1"]\n\n'
    "1. d4 d5 0-1\n\n"
)


def test_games_month_url():
    assert games_month_url(2025, 1) == (
        "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
    )


def test_stream_zst_lines_handles_early_stop(tmp_path: Path):
    zst_path = tmp_path / "sample.jsonl.zst"
    cctx = zstd.ZstdCompressor()
    zst_path.write_bytes(cctx.compress(b'{"a": 1}\n{"a": 2}\n{"a": 3}\n'))
    lines = list(stream_zst_lines(zst_path, max_lines=2))
    assert lines == ['{"a": 1}', '{"a": 2}']


def test_stream_pgn_blocks_splits_games_and_respects_max_games(tmp_path: Path):
    zst_path = tmp_path / "sample.pgn.zst"
    cctx = zstd.ZstdCompressor()
    zst_path.write_bytes(cctx.compress(SAMPLE_PGN.encode("utf-8")))
    blocks = list(stream_pgn_blocks(zst_path, max_games=1))
    assert len(blocks) == 1
    assert "https://lichess.org/abcd1234" in blocks[0]
    assert "1. e4 e5" in blocks[0]
```

- [ ] **Step 3: Run pytest to verify failure**

Run: `uv run pytest tests/test_data_ingestion.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Implement `src/lichess_pipeline/data_ingestion.py`**

Implement `games_month_url`, `stream_zst_lines`, `stream_pgn_blocks`, and `download_sample_to_disk` using `urllib.request.urlopen` (for `http://` / `https://` sources) or `open(path, "rb")` (for local `Path`), piped through `zstandard.ZstdDecompressor().stream_reader(raw_stream)` wrapped in `io.TextIOWrapper(..., encoding="utf-8", errors="replace")`. Catch `(zstd.ZstdError, EOFError)` if the stream was intentionally truncated or stopped early. `stream_pgn_blocks` accumulates non-blank lines and yields a block whenever it sees a line that starts with a result token (`1-0`, `0-1`, `1/2-1/2`) at the end of movetext.

- [ ] **Step 5: Run pytest to verify pass**

Run: `uv run pytest tests/test_data_ingestion.py tests/test_chess_vision.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/lichess_pipeline/data_ingestion.py tests/test_data_ingestion.py
git commit -m "feat: replace HF ingestion with official Lichess .zst streaming"
```

---

### Task 2: PGN & Evaluation Parsing, Cleaning, and Position Extraction

**Files:**
- Replace: `src/lichess_pipeline/chess_utils.py`
- Populate: `src/lichess_pipeline/pgn_parser.py`
- Replace: `tests/test_chess_utils.py`
- Populate: `tests/test_pgn_parser.py`

**Interfaces:**
- Consumes: `stream_pgn_blocks`, `stream_zst_lines` from `src/lichess_pipeline/data_ingestion.py`
- Produces in `src/lichess_pipeline/chess_utils.py`:
  - `normalize_fen(fen: str) -> str`
  - `piece_count_from_fen(fen: str) -> int`
  - `phase_from_piece_count(count: int) -> str` (`>= 29` `"opening"`, `15..28` `"middlegame"`, `<= 14` `"endgame"`)
  - `first_move_is_legal(fen: str, move_uci: str) -> bool`
- Produces in `src/lichess_pipeline/pgn_parser.py`:
  - `parse_pgn_game(raw_pgn: str) -> dict[str, object] | None`
  - `parse_pgn_sample_df(source: str | Path, max_games: int = 10_000) -> pl.DataFrame`
  - `clean_games_df(df: pl.DataFrame, min_elo: int = 1800, min_plies: int = 10) -> pl.DataFrame`
  - `extract_positions_df(clean_df: pl.DataFrame, max_positions: int = 50_000) -> pl.DataFrame` (columns: `game_id`, `fen`, `move_uci`, `move_ply`, `turn`, `piece_count`, `game_phase`, `avg_elo`)
  - `parse_eval_line(line: str, min_depth: int = 20) -> dict[str, object] | None`
  - `load_clean_evals_df(source: str | Path, max_evals: int = 50_000, min_depth: int = 20) -> pl.DataFrame` (columns: `fen`, `norm_fen`, `turn`, `piece_count`, `game_phase`, `depth`, `knodes`, `cp`, `mate`, `best_move`)

- [ ] **Step 1: Write failing tests in `tests/test_chess_utils.py` and `tests/test_pgn_parser.py`**

```python
# tests/test_pgn_parser.py
import json
import polars as pl
from lichess_pipeline.pgn_parser import (
    clean_games_df,
    extract_positions_df,
    parse_eval_line,
    parse_pgn_game,
)

RAW_GAME_WITH_COMMENTS = """[Event "Rated Blitz game"]
[Site "https://lichess.org/xyz98765"]
[White "playerA"]
[Black "playerB"]
[Result "1-0"]
[UTCDate "2025.01.01"]
[UTCTime "12:00:00"]
[WhiteElo "1950"]
[BlackElo "1880"]
[WhiteRatingDiff "+5"]
[BlackRatingDiff "-5"]
[ECO "C50"]
[Opening "Italian Game"]
[TimeControl "180+0"]
[Termination "Normal"]

1. e4 { [%eval 0.18] [%clk 0:03:00] } 1... e5 { [%eval 0.22] [%clk 0:03:00] } 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 1-0
"""


def test_parse_pgn_game_strips_comments_and_counts_plies():
    row = parse_pgn_game(RAW_GAME_WITH_COMMENTS)
    assert row is not None
    assert row["game_id"] == "xyz98765"
    assert row["WhiteElo"] == 1950
    assert row["BlackElo"] == 1880
    assert row["num_plies"] == 12
    assert "[%clk" not in str(row["movetext"])


def test_clean_games_df_filters_elo_bullet_and_termination():
    good = parse_pgn_game(RAW_GAME_WITH_COMMENTS)
    bullet = {**good, "game_id": "b1", "Event": "Rated Bullet game"}
    low_elo = {**good, "game_id": "l1", "WhiteElo": 1400}
    forfeit = {**good, "game_id": "f1", "Termination": "Time forfeit"}
    df = pl.DataFrame([good, bullet, low_elo, forfeit])
    cleaned = clean_games_df(df, min_elo=1800, min_plies=10)
    assert cleaned["game_id"].to_list() == ["xyz98765"]


def test_extract_positions_df_replays_legal_moves():
    df = pl.DataFrame([parse_pgn_game(RAW_GAME_WITH_COMMENTS)])
    positions = extract_positions_df(df, max_positions=5)
    assert len(positions) == 5
    assert positions["move_uci"].to_list()[:2] == ["e2e4", "e7e5"]
    assert positions["turn"].to_list()[:2] == ["w", "b"]


def test_parse_eval_line_selects_max_depth_and_handles_missing_pvs():
    valid_json = json.dumps(
        {
            "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "evals": [
                {"knodes": 100, "depth": 18, "pvs": [{"cp": 25, "line": "c7c5 g1f3"}]},
                {"knodes": 500, "depth": 24, "pvs": [{"cp": 20, "line": "e7e5 g1f3"}]},
            ],
        }
    )
    parsed = parse_eval_line(valid_json, min_depth=20)
    assert parsed is not None
    assert parsed["depth"] == 24
    assert parsed["best_move"] == "e7e5"
    assert parsed["cp"] == 20
    assert parse_eval_line('{"fen": "invalid", "evals": []}', min_depth=20) is None
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `uv run pytest tests/test_pgn_parser.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/lichess_pipeline/chess_utils.py` and `src/lichess_pipeline/pgn_parser.py`**

In `pgn_parser.py`, use `chess.pgn.read_game(io.StringIO(raw_pgn))` for rock-solid SAN replay; strip comments and NAGs with regex for the stored `movetext`. In `parse_eval_line`, pick `max(data["evals"], key=lambda e: e.get("depth", 0))`, verify `depth >= min_depth` and `first_move_is_legal(fen, best_move)`. In `load_clean_evals_df`, deduplicate on `norm_fen`.

- [ ] **Step 4: Run pytest to verify pass**

Run: `uv run pytest tests/test_chess_utils.py tests/test_pgn_parser.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lichess_pipeline/chess_utils.py src/lichess_pipeline/pgn_parser.py tests/test_chess_utils.py tests/test_pgn_parser.py
git commit -m "feat: add PGN and Stockfish eval parsers, cleaners, and position extractor"
```

---

### Task 3: FEN Tensor Encoding, `64x64` Move Indexing & Legal-Move Masking

**Files:**
- Create: `src/lichess_pipeline/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Produces in `src/lichess_pipeline/features.py`:
  - `NUM_MOVE_CLASSES = 4096`
  - `encode_fen(fen: str) -> np.ndarray` (shape `(13, 8, 8)`, dtype `np.float32`; planes `0..5` White `P,N,B,R,Q,K`, planes `6..11` Black `p,n,b,r,q,k`, plane `12` all `1.0` if White to move else `0.0`)
  - `move_to_index(move_uci: str) -> int`
  - `index_to_move(index: int, board: chess.Board | None = None) -> str`
  - `legal_move_mask(fen: str) -> np.ndarray` (shape `(4096,)`, dtype `bool`)

- [ ] **Step 1: Write failing tests in `tests/test_features.py`**

```python
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


def test_move_index_roundtrip_and_pawn_promotion():
    idx = move_to_index("e2e4")
    assert 0 <= idx < NUM_MOVE_CLASSES
    assert index_to_move(idx) == "e2e4"

    promo_fen = "8/4P3/8/8/8/8/8/k6K w - - 0 1"
    board = chess.Board(promo_fen)
    promo_idx = move_to_index("e7e8q")
    assert index_to_move(promo_idx, board) == "e7e8q"


def test_legal_move_mask_starting_position_has_20_squares():
    mask = legal_move_mask(chess.STARTING_FEN)
    assert mask.shape == (NUM_MOVE_CLASSES,)
    assert int(mask.sum()) == 20
    assert mask[move_to_index("e2e4")]
    assert not mask[move_to_index("e2e5")]
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/lichess_pipeline/features.py`**

Implement `encode_fen`, `move_to_index`, `index_to_move`, and `legal_move_mask`.

- [ ] **Step 4: Run pytest to verify pass**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lichess_pipeline/features.py tests/test_features.py
git commit -m "feat: add 13x8x8 FEN encoder and 4096 legal-move masking"
```

---

### Task 4: Approach 1 (From-Scratch + Fine-Tuning PyTorch Policy Network) & Approach 3 (Maia Adapter)

**Files:**
- Create: `src/lichess_pipeline/model/__init__.py`
- Create: `src/lichess_pipeline/model/scratch_net.py`
- Create: `src/lichess_pipeline/model/maia_adapter.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `encode_fen`, `move_to_index`, `index_to_move`, `legal_move_mask`, `NUM_MOVE_CLASSES` from `src/lichess_pipeline/features.py`
- Produces in `src/lichess_pipeline/model/scratch_net.py`:
  - `ChessPolicyNet(nn.Module)`
  - `make_tensor_dataset(df: pl.DataFrame, fen_col: str = "fen", move_col: str = "move_uci") -> TensorDataset`
  - `train_model(model: nn.Module, train_ds: TensorDataset, val_ds: TensorDataset, epochs: int = 5, lr: float = 1e-3, batch_size: int = 256) -> list[dict[str, float]]`
  - `predict_top_moves(model: nn.Module, fen: str, top_k: int = 3) -> list[tuple[str, float]]`
- Produces in `src/lichess_pipeline/model/maia_adapter.py`:
  - `MaiaResidualPolicyNet(channels: int = 64, num_blocks: int = 4)` (residual conv tower + squeeze-excitation block + 4096 policy head)

- [ ] **Step 1: Write failing tests in `tests/test_models.py`**

```python
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
    board = chess.Board(fen)
    legal_ucis = {m.uci() for m in board.legal_moves}

    preds = predict_top_moves(model, fen, top_k=3)
    assert len(preds) == 3
    for move_uci, prob in preds:
        assert move_uci in legal_ucis
        assert 0.0 <= prob <= 1.0


def test_predict_top_moves_on_checkmate_returns_empty_list():
    model = ChessPolicyNet()
    fools_mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert predict_top_moves(model, fools_mate_fen, top_k=3) == []


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
    ds = make_tensor_dataset(sample_df, "fen", "move_uci")
    model = ChessPolicyNet()
    history_stage1 = train_model(model, ds, ds, epochs=2, lr=1e-2, batch_size=8)
    assert len(history_stage1) == 2
    assert "val_top1" in history_stage1[-1]

    history_stage2 = train_model(model, ds, ds, epochs=1, lr=1e-4, batch_size=8)
    assert len(history_stage2) == 1

    maia_model = MaiaResidualPolicyNet(channels=32, num_blocks=2)
    maia_hist = train_model(maia_model, ds, ds, epochs=1, lr=1e-3, batch_size=8)
    assert len(maia_hist) == 1
    assert len(predict_top_moves(maia_model, chess.STARTING_FEN, top_k=3)) == 3
```

- [ ] **Step 2: Run pytest to verify failure**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `src/lichess_pipeline/model/scratch_net.py` and `src/lichess_pipeline/model/maia_adapter.py`**

Implement `ChessPolicyNet`, `make_tensor_dataset`, `train_model`, `predict_top_moves`, and `MaiaResidualPolicyNet`. Ensure `predict_top_moves` checks `board.legal_moves` first and returns `[]` if empty. `train_model` computes the legal-move mask per batch for the Top-1/Top-3 validation metrics.

- [ ] **Step 4: Run pytest to verify pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lichess_pipeline/model/ tests/test_models.py
git commit -m "feat: implement scratch+finetune ChessPolicyNet and MaiaResidualPolicyNet"
```

---

### Task 5: Build the Three Marimo Notebooks for Tomorrow's Delivery

**Files:**
- Create: `notebooks/01_eda_and_cleaning.py`
- Create: `notebooks/02_model_scratch_and_finetune.py`
- Create: `notebooks/03_model_maia_pretrained.py`
- Modify: `README.md`

**Interfaces:**
- Consumes all functions from Tasks 1-4.
- Produces cleaned Parquet files in `data/processed/`:
  - `data/processed/games_clean_sample.parquet`
  - `data/processed/positions_games_sample.parquet`
  - `data/processed/evals_clean_sample.parquet`
- Produces saved model checkpoints in `data/models/`:
  - `data/models/approach1_stage1_scratch.pt`
  - `data/models/approach1_stage2_finetuned.pt`
  - `data/models/approach3_maia_finetuned.pt`

- [ ] **Step 1: Create `notebooks/01_eda_and_cleaning.py`**

Create a valid Marimo notebook (`marimo check` clean) with:
1. Interactive UI controls for Year/Month of `.pgn.zst`, sample size (`max_games`, `max_evals`), minimum Elo slider (`1500`-`2200`, default `1800`), and an optional checkbox/button to save raw `.zst` samples in `data/raw/`.
2. **Part 1 - Standard Games (`.pgn.zst`) EDA & Cleaning:** Stream sample from `database.lichess.org`, display schema/nulls/duplicates table, plot Elo distributions, time controls, terminations, game length (plies), and top openings; run `clean_games_df`, show before vs. after row retention table, run `extract_positions_df` with `python-chess`, and save `games_clean_sample.parquet` and `positions_games_sample.parquet`.
3. **Part 2 - Stockfish Evaluations (`lichess_db_eval.jsonl.zst`) EDA & Cleaning:** Stream sample from `database.lichess.org`, plot search depth distribution, centipawn (`cp`) distribution, and game phase breakdown (`opening` / `middlegame` / `endgame`); keep `depth >= 20` legal positions and save `evals_clean_sample.parquet`.

- [ ] **Step 2: Create `notebooks/02_model_scratch_and_finetune.py` (Primary - Approach 1)**

Create a valid Marimo notebook with:
1. Automatic loading of `data/processed/positions_games_sample.parquet` and `data/processed/evals_clean_sample.parquet` (if not yet generated, automatically bootstraps a small sample using `pgn_parser` so the notebook always works standalone).
2. **Stage 1 (Train from Scratch on High-Elo `.pgn.zst` Games):** Train `ChessPolicyNet`, record and plot loss + Top-1/Top-3 validation accuracy, save `approach1_stage1_scratch.pt`.
3. **Stage 2 (Fine-Tune on Stockfish `lichess_db_eval.jsonl.zst`):** Fine-tune the Stage 1 model on Stockfish best moves with lower learning rate, plot Stage 1 vs. Stage 2 accuracy improvement on the Stockfish validation set, save `approach1_stage2_finetuned.pt`.
4. **Interactive Board Predictor (`FEN -> Next Best Move`):** Marimo text input / preset selector for any FEN string, renders the SVG chessboard with `chess.svg.board` reflecting the top predicted move arrow, and displays Top-3 legal moves with probabilities comparing Stage 1 vs. Stage 2.

- [ ] **Step 3: Create `notebooks/03_model_maia_pretrained.py` (Secondary - Approach 3)**

Create a valid Marimo notebook with:
1. Loads `MaiaResidualPolicyNet` and fine-tunes/adapts it on the cleaned `evals_clean_sample.parquet` dataset.
2. Side-by-side comparison table & bar chart against `approach1_stage2_finetuned.pt` (Top-1 accuracy, Top-3 accuracy, inference latency per FEN).
3. Interactive FEN comparison widget showing the board and the move suggested by **Approach 1 (Scratch + Fine-Tune)** next to **Approach 3 (Maia Architecture)**.

- [ ] **Step 4: Verify all notebooks and run the full test suite**

Run:
```bash
uv run marimo check notebooks/01_eda_and_cleaning.py notebooks/02_model_scratch_and_finetune.py notebooks/03_model_maia_pretrained.py
uv run pytest -v
uv run ruff check src/ tests/ notebooks/
```
Expected: 0 errors in `marimo check`, all `pytest` tests PASS, `ruff check` clean.

- [ ] **Step 5: Update `README.md` with clear instructions to run the 3 Marimo notebooks (keeping the Vision section intact) and commit**

```bash
git add notebooks/ README.md
git commit -m "feat: add Marimo notebooks for official Lichess EDA/cleaning and move prediction models"
```
