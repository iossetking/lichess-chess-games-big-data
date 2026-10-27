# From Photo to Best Move — Project Proposal

Shared project for **Big Data** and **Computer Vision**. It replaces the earlier
objective in `docs/Proyecto Final.pdf` (predicting the game result with an
FT-Transformer).

## 1. Goal

Build a system that receives a **photo of a chessboard**, recognizes the
position, and predicts the **best move** for the side to move.

```
Photo ──(Computer Vision)──► FEN ──(our move model)──► best move + evaluation
                                          ▲
   Big Data: ~1.1 B Stockfish evaluations (labels) + 4.94 TB of human games
```

"Best move" means the move chosen by **Stockfish at high search depth**.
Chess is not solved: only endgames with up to 7 pieces have exact answers
(Syzygy tablebases). Our model learns to **imitate the engine without
calculating variations**. It is not expected to beat Stockfish.

## 2. Data and what each dataset is for

All Lichess datasets are CC0. They are **read remotely with Polars** (`hf://`)
at a pinned revision, so nothing large is downloaded and results are
reproducible. Reading is done by `src/lichess_pipeline/data_ingestion.py`.

| Dataset | Size | What it gives us | Why we need it |
|---|---|---|---|
| `Lichess/chess-position-evaluations` | 44 GB, ~1.1 B rows | Positions (`fen`) with Stockfish's best lines (`line`), `depth`, `cp`, `mate` | It is the **answer key**: the best move for each position is already computed, so we do not have to run Stockfish ourselves |
| `Lichess/chess-puzzles` | 0.9 GB, ~6 M puzzles | Positions with a known winning solution and a difficulty rating | An **independent exam** for the model, graded by difficulty |
| `Lichess/standard-chess-games` | 4.94 TB, 153 monthly partitions (2013-01 to 2025-09) | Billions of real human games with ratings, clocks and, in part of them, engine evaluations per move | Measures **how humans play compared with the engine** at scale, and supplies **real positions** for testing and for rendering images |
| ChessReD (Masouris & van Gemert, VISAPP 2024) | 10,800 phone photos | Real board photos with the piece on every square | Tests the vision system on **real photos**, not only synthetic ones. License to be confirmed |

## 3. How we use the 4.94 TB games dataset

The games dataset is too large to download or to replay move by move on a
laptop. We use it in three ways, from cheapest to most expensive:

1. **Full scan with column selection (all or many months).** Polars reads only
   the columns we need (`movetext`, ratings, `TimeControl`, `Termination`,
   `UTCDate`) and extracts the engine evaluation (`[%eval]`) and clock
   (`[%clk]`) comments from `movetext` with vectorized string operations, with
   no Python loops. From consecutive evaluations we get the **centipawn loss of
   each human move**, aggregated by rating, time control, game phase and year.
   *Why:* this is the Big Data part. It processes the whole history with a
   streaming pipeline, and each month becomes a small summary table.
2. **Sampled months for joins (e.g. one recent month).** For a sample of
   games, we replay the moves with python-chess, get the FEN of every position
   and join it with the evaluations dataset. *Why:* it tells us whether the
   human move matched Stockfish's best move. Replaying is slow, so it is done
   on a sample only.
3. **Real-game test set and image source.** Positions taken from these games
   (never seen in training) are used to test the move model and to render
   synthetic board images. *Why:* real games contain positions that look like
   what a user would photograph.

Practical limits: reading the full 4.94 TB remotely depends on bandwidth (at
about 500 Mbps it takes roughly a day of transfer). We start with one month per
year (13 months), process month by month, and only scale up if time allows.
Every month is saved as a compact summary, so the work can be resumed.

## 4. Models

| Role | Model | Why |
|---|---|---|
| Answer key | Stockfish labels already in the evaluations dataset | Ground truth without running an engine |
| Strong pretrained reference | **Lc0 BT4** (Leela Chess Zero network), run through ONNX in a separate environment | Strongest public network that picks moves without search; shows how close our model gets |
| Human-move contrast (optional) | **Maia-3** (23 M), successor of Maia-2 | Predicts what a *human* of a given rating would play; contrasts "best" vs "human". AGPL-3.0, to verify in the official repo |
| Simple baselines | Random legal move; small MLP/CNN on the board | Show that the main model learns something non-trivial |
| **Our model** | **Transformer encoder trained from scratch** | The main contribution of the project |

Discarded: DeepMind `searchless_chess` (JAX, 270 M parameters; used only as a
literature reference), Allie (uses search), Ultralytics YOLO (AGPL-3.0).

### Our model's architecture

- **Input**: 64 square tokens (13 classes: 6 white pieces, 6 black pieces,
  empty) with learned position embeddings, plus tokens for side to move,
  castling rights and en passant square.
- **Body**: Transformer encoder. Start with **6 layers, d_model 256, 8 heads
  (~5 M parameters)**; scale to **8 layers, d_model 512, 16 heads (~25 M)** if
  time allows. Fits in 8 GB of VRAM with mixed precision.
- **Output**: a score for every from-square/to-square pair plus promotions,
  with **illegal moves masked**, so the model always returns a legal move.
- **Target**: not only the single best move. The alternative lines in the
  evaluations dataset are converted to win probabilities and used as soft
  targets, which carries more information than one answer.
- **Training**: mixed precision, batch 512–1024, fixed seed, saved config and
  checkpoints.
- **Expected results**: DeepMind's 9 M-parameter model matches Stockfish's move
  about 64 % of the time and solves about 89 % of puzzles. That is a realistic
  ceiling for a model of our size.

## 5. Computer Vision: photo → FEN

Chosen approach, the best-documented one: **find the board, straighten it,
classify each square**. A November 2025 study (CVChess) reports 98.9 %
correct squares with this approach; reading the whole photo in one step gets
only 15 % perfect boards on ChessReD.

1. **Synthetic images**: render positions with python-chess (SVG → PNG) using
   permissively licensed Lichess piece sets (CC0: rhosgfx; CC BY 4.0:
   kiwen-suwi, Firi; MIT: fantasy, spatial, celtic; Apache 2.0: chessnut), with
   perspective, lighting, noise and blur. *Why:* unlimited images with exact
   labels at no cost.
2. **Board detection**: find the four corners (edges + Hough lines, or a
   keypoint model) and apply a homography to get a top-down 8×8 grid.
3. **Square classification**: small pretrained CNN (ResNet-18 or MobileNet)
   fine-tuned on the 13 classes.
4. **FEN assembly and correction**: detect board orientation and apply chess
   rules (one king per side, no pawns on ranks 1 and 8) to fix doubtful squares.
5. **Real photos**: fine-tune and test on ChessReD; final test on our own phone
   photos.
- **Metrics**: correct squares (%), boards with zero errors (%), mean errors
  per board.

## 6. Project steps

Each step has its own plan, tests and code review before moving on.

| # | Step | Data | Output | Why |
|---|---|---|---|---|
| 0 | **Reading** (done) | Games dataset | Remote reader with month selection and schema checks | Read any month without downloading terabytes |
| 1 | **Exploration (EDA)** of evaluations, puzzles and games | All three | Short report: depths, score ranges, puzzle difficulty, how many games carry `[%eval]` and `[%clk]` per month | Know the data before building on it; decide sample sizes |
| 2 | **Label pipeline** | Evaluations + puzzles | Clean parquet: FEN, best move, alternatives with scores; train/val/test split (90/5/5) by FEN hash; **puzzle positions removed from training** | The model learns from correct answers and is never tested on positions it has seen |
| 3 | **Games pipeline** (Big Data core) | 4.94 TB games, month by month | Per-month summaries: centipawn loss per move by rating, time control, phase and year; real-game test positions | Use the full dataset at scale with a streaming, resumable pipeline |
| 4 | **Baselines** | Step 2 output | Scores for random move and MLP/CNN | Set the minimum bar |
| 5 | **Train our Transformer** | Step 2 sample (20–50 M positions) | Trained model, metrics, config | Main model of the project |
| 6 | **Model evaluation** | Test split, puzzles, real-game positions | Top-1/top-3 agreement with Stockfish, centipawn loss, puzzle accuracy by difficulty; comparison with Lc0 BT4 (and Maia-3) | Show how good the model is and where it fails |
| 7 | **Human vs. engine analysis** | Steps 3 and 6 | Charts: how often humans of each rating find the best move, over time and by time control | Main Big Data finding |
| 8 | **Vision: synthetic data and square classifier** | Positions from steps 2–3 rendered as images | Trained square classifier | Core of the Computer Vision project |
| 9 | **Vision: board detection and real photos** | ChessReD + our photos | Photo → FEN system with accuracy on real photos | Make it work outside synthetic images |
| 10 | **Integrated demo** | All models | marimo app: photo → FEN → top-3 moves vs Stockfish | One system for both courses |
| 11 | **Reports** | Results of all steps | One report per course | Deliverables |

Steps 2 and 3 are independent and can run in parallel; step 8 can start once
step 2 produces positions.

## 7. Safeguards

- **No leakage**: split by FEN hash (same position never in two sets); remove
  puzzle positions from training; real-game test positions come from months not
  used for training.
- **Validation**: every FEN and every labeled move is checked as legal with
  python-chess; minimum analysis depth (e.g. 20).
- **Reproducibility**: pinned dataset revisions, fixed seeds, saved configs.
- **Resumable processing**: work is done file by file / month by month, with
  compact outputs saved after each one.

## 8. Dependencies and environment

New packages, added only when each step needs them: `chess` (rules, FEN,
rendering), `torch` + `torchvision` (models), `cairosvg` (SVG → PNG),
`opencv-python-headless` (homography). All resolve for Python 3.14.
Lc0 BT4 (via `onnxruntime`) and Maia-3 run in separate environments, only for
comparison. Hardware: RTX 4060 Laptop, 8 GB VRAM.

**Licenses**: python-chess is GPL-3.0, so a published repository should be
GPL-3.0. Maia-3 is AGPL-3.0. Ultralytics YOLO is avoided.

## 9. Open questions

- Confirm the ChessReD dataset license on 4TU.ResearchData.
- Confirm Maia-3 details and the Lc0 network license in their official repositories.
- Measure real training throughput on the RTX 4060 (current estimate: 1–2 k
  positions/s for the 25 M model, i.e. 7–15 h for 20–50 M positions).
- Measure remote read speed for one full month before planning the full-history scan.
