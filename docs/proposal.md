# From Photo to Best Move — Project Proposal

Shared project for **Big Data** and **Computer Vision**, replacing the earlier
game-result prediction objective in `docs/Proyecto Final.pdf`.

## 1. Goal

Build a system that takes a chessboard photo, recognizes its position, and
suggests the best move for the side to move:

```text
Photo → computer vision → FEN/position → move model → top moves + evaluation
```

This uses **two separate models**. The vision model recognizes the board; the
move model predicts a move from the recognized position. The move model learns
to imitate Stockfish from its existing evaluations; it does not calculate
variations or aim to beat Stockfish.

## 2. Data

Lichess datasets are CC0 and read remotely with Polars (`hf://`) at pinned
revisions. The reader is `src/lichess_pipeline/data_ingestion.py`.

| Dataset | What it provides | How we use it |
|---|---|---|
| `Lichess/chess-position-evaluations` (~44 GB, 1.1B rows) | FENs, Stockfish lines, depth and scores | Labels for training and evaluating the move model |
| `Lichess/chess-puzzles` (~0.9 GB, 6M puzzles) | Positions, solutions and difficulty | Independent move-model test set |
| `Lichess/standard-chess-games` (4.94 TB, 153 monthly partitions) | Human games, ratings, clocks and some engine evaluations | Big Data analysis, sampled real positions and image sources |
| ChessReD (10,800 phone photos) | Real board photos with piece labels | Real-photo vision evaluation; license must be confirmed |

## 3. Games and position images

The games collection is too large to download or replay in full on a laptop.
We will:

1. Scan selected months with Polars, reading only needed columns. Extract
   `[%eval]` and `[%clk]` comments from `movetext` with vectorized operations,
   then summarize centipawn loss by rating, time control, game phase and year.
2. Replay a sample of games with python-chess to get each position's FEN and
   compare human moves with Stockfish evaluations.
3. Use held-out real-game positions to test the move model and render a
   **selected sample** as labeled board images for vision training.

Start with one month per year (13 months), process month by month, and save
compact summaries so work can resume. A full scan depends on bandwidth; at
500 Mbps, transferring 4.94 TB would take roughly a day.

## 4. Best-move model

| Role | Model |
|---|---|
| Training labels | Stockfish evaluations already in the Lichess dataset |
| Main model | Transformer encoder trained from scratch on Stockfish-labeled positions |
| Baselines | Random legal move and a small MLP/CNN |
| Pretrained comparison | Lc0 BT4 via ONNX; optional Maia-3 for human-move comparison |

The Transformer receives a board position—not an image—as 64 square tokens
(12 piece types plus empty), with side-to-move, castling and en-passant
information. Start with 6 layers, width 256 and 8 attention heads (~5M
parameters); scale up only if time and hardware allow. Its output scores legal
moves, including promotions. Stockfish's best line and alternatives provide
the training targets. Use mixed precision, fixed seeds, saved configs and
checkpoints.

Evaluate top-1/top-3 agreement with Stockfish, centipawn loss and puzzle
accuracy by difficulty. A published ~9M-parameter model's reported 64% move
agreement and 89% puzzle success are reference points, not guaranteed results.
Lc0 BT4 (and, optionally, Maia-3) is for comparison, not our training model.

## 5. Computer vision: photo → position

The core approach is to detect and straighten the board, then classify its
squares:

1. **Make training images:** render selected FENs as labeled board images with
   licensed piece sets; add perspective, lighting, blur and noise.
2. **Find the board:** detect its corners and use a homography to create a
   top-down 8×8 grid.
3. **Recognize pieces:** train a small CNN from scratch to classify each square
   as one of 12 piece types or empty.
4. **Build the position:** determine board orientation, assemble the FEN and
   check chess rules (for example, one king per side and no pawns on the first
   or eighth rank).
5. **Test on real photos:** evaluate on ChessReD and our own phone photos.

Measure square accuracy, the percentage of boards with no errors, and mean
errors per board.

### Optional side plan: fine-tuning a pretrained vision model

If time allows, compare the from-scratch CNN with a pretrained ResNet-18 or
MobileNet. **Fine-tuning means starting with a model that already learned**
general visual patterns, replacing its output layer for the chess-square
classes, and training it further on our labeled chess images. If labeled
real-photo squares are available, use them for a final small adaptation and
compare both models on photos they have not seen. This is an optional vision
experiment; it does not change how the best-move Transformer is trained.

## 6. Project steps

| # | Step | Main output |
|---|---|---|
| 0 | Remote games reader (done) | Month selection and schema checks |
| 1 | Explore evaluations, puzzles and games | Data-quality and sample-size report |
| 2 | Prepare move labels | Valid FENs, moves, scores and 90/5/5 FEN-hash split; remove puzzle positions from training |
| 3 | Process games by month | Human-move summaries and held-out real positions |
| 4 | Build simple baselines | Random and MLP/CNN scores |
| 5 | Train the Transformer from scratch | Checkpoint, config and metrics |
| 6 | Evaluate move predictions | Test, puzzles, real positions and pretrained comparisons |
| 7 | Analyze humans vs. engine | Results by rating, time control and year |
| 8 | Train the vision classifier | CNN trained on rendered board images |
| 9 | Test vision on real photos | ChessReD and our photos; fine-tuning is optional |
| 10 | Build the demo | One photo → FEN → top-3 move suggestions |
| 11 | Write course reports | Big Data and Computer Vision reports |

Steps 2 and 3 can run in parallel; step 8 starts after step 2 provides FENs.
The demo processes a photo; continuous video or camera tracking is out of
scope.

## 7. Safeguards and environment

- Prevent leakage: split by FEN hash, remove puzzle positions from training,
  and hold out game months for real-position tests.
- Validate FENs and moves with python-chess; require a minimum evaluation depth
  (for example, 20).
- Pin dataset revisions, fix seeds, and save configs and checkpoints. Save
  monthly outputs for resumable processing.
- Add dependencies as needed: `chess`, `torch`, `torchvision`, `cairosvg` and
  `opencv-python-headless`. The project targets Python 3.14 and an RTX 4060
  Laptop with 8 GB VRAM. Run Lc0/Maia comparisons in separate environments.
