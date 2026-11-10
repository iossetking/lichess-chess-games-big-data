"""Live camera pipeline: automatic board detection -> YOLO pieces -> FEN.

Example:
    uv run python scripts/live_chess_fen.py \
      --weights data/models/chess-pieces.pt --move-model data/models/approach1_stage2_finetuned.pt

The piece checkpoint must have one class for each coloured chess piece (for
example ``w-pawn`` and ``b-king``).  An ordinary COCO YOLO checkpoint cannot
recognise chess pieces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2
import numpy as np

from lichess_pipeline.chess_vision import (
    DEFAULT_BOARD_SIZE,
    FenStabilizer,
    PieceDetection,
    board_destination,
    detect_board_corners,
    detections_to_fen,
    square_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="YOLO .pt/.onnx trained on the 12 chess pieces")
    parser.add_argument("--board-weights", type=Path, help="Optional YOLO-seg model with a chessboard/board class")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE, help="Rectified board side in pixels")
    parser.add_argument("--confidence", type=float, default=0.55, help="Minimum YOLO confidence")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--stable-frames", type=int, default=4, help="Equal frames required before publishing a FEN")
    parser.add_argument("--flip", action="store_true", help="Rotate the board 180° when the camera is on Black's side")
    parser.add_argument("--turn", choices=("w", "b"), default="w", help="Side to move; press t in the window to toggle")
    parser.add_argument("--castling", default="-", help="Known castling rights, e.g. KQkq (vision cannot infer this)")
    parser.add_argument("--en-passant", default="-", help="Known en-passant target (vision cannot infer this)")
    parser.add_argument("--move-model", type=Path, help="Optional trained policy checkpoint from this repository")
    parser.add_argument("--architecture", choices=("scratch", "maia"), default="scratch", help="Architecture of --move-model")
    return parser.parse_args()


class YoloPieceDetector:
    """Small adapter that avoids tying core vision code to Ultralytics types."""

    def __init__(self, weights: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Falta Ultralytics. Ejecuta: uv add ultralytics") from error
        self.model = YOLO(weights)

    def detect(self, image: np.ndarray, confidence: float, imgsz: int) -> list[PieceDetection]:
        result = self.model.predict(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
        if result.boxes is None:
            return []
        names: dict[int, str] = result.names
        detections: list[PieceDetection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(PieceDetection(names[class_id], float(box.conf.item()), xyxy))
        return detections


class YoloBoardSegmenter:
    """Optional board-segmentation fallback for grids obscured by pieces/hands."""

    def __init__(self, weights: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Falta Ultralytics. Ejecuta: uv add ultralytics") from error
        self.model = YOLO(weights)

    def detect(self, frame: np.ndarray, confidence: float, imgsz: int) -> np.ndarray | None:
        result = self.model.predict(frame, conf=confidence, imgsz=imgsz, verbose=False)[0]
        if result.masks is None or result.boxes is None:
            return None
        names: dict[int, str] = result.names
        candidates: list[tuple[float, np.ndarray]] = []
        for index, polygon in enumerate(result.masks.xy):
            label = str(names[int(result.boxes[index].cls.item())]).lower().replace("_", "").replace("-", "")
            if label not in {"board", "chessboard"}:
                continue
            contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.025 * perimeter, True).reshape(-1, 2)
            if len(approximation) != 4:
                approximation = cv2.boxPoints(cv2.minAreaRect(contour))
            candidates.append((float(cv2.contourArea(contour)), approximation))
        if not candidates:
            return None
        corners = max(candidates, key=lambda candidate: candidate[0])[1]
        return order_corners(corners)


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return a quadrilateral in TL, TR, BR, BL image order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums, differences = points.sum(axis=1), np.diff(points, axis=1).reshape(-1)
    return np.asarray(
        [points[np.argmin(sums)], points[np.argmin(differences)], points[np.argmax(sums)], points[np.argmax(differences)]],
        dtype=np.float32,
    )


def draw_grid(image: np.ndarray, regions: dict[str, tuple[int, int, int, int]]) -> None:
    for name, (x1, y1, x2, y2) in regions.items():
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(image, name, (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)


def draw_detections(image: np.ndarray, detections: list[PieceDetection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 110, 0), 2)
        cv2.putText(image, f"{detection.label} {detection.confidence:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 110, 0), 1)


def load_policy(path: Path, architecture: str) -> Any:
    import torch

    if architecture == "scratch":
        from lichess_pipeline.model.scratch_net import ChessPolicyNet
        model = ChessPolicyNet()
    else:
        from lichess_pipeline.model.maia_adapter import MaiaResidualPolicyNet
        model = MaiaResidualPolicyNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def main() -> int:
    args = parse_args()
    if args.size < 8 or args.size % 8:
        print("--size must be divisible by 8 and at least 8", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"YOLO piece weights not found: {args.weights}", file=sys.stderr)
        return 2
    if args.board_weights and not args.board_weights.is_file():
        print(f"YOLO board weights not found: {args.board_weights}", file=sys.stderr)
        return 2

    try:
        pieces = YoloPieceDetector(args.weights)
        board_segmenter = YoloBoardSegmenter(args.board_weights) if args.board_weights else None
        policy = load_policy(args.move_model, args.architecture) if args.move_model else None
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1

    destination, regions = board_destination(args.size), square_regions(args.size)
    stabilizer, turn = FenStabilizer(args.stable_frames), args.turn
    visible_fen, visible_move = "Waiting for board", ""
    print("Automatic board mode. q/Esc exits; t toggles side to move; f flips orientation.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the camera", file=sys.stderr)
                return 1
            corners = detect_board_corners(frame)
            if corners is None and board_segmenter is not None:
                corners = board_segmenter.detect(frame, args.confidence, args.imgsz)
            preview = frame.copy()
            if corners is None:
                cv2.putText(preview, "Looking for 8x8 board grid", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                board = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(corners, destination), (args.size, args.size))
                if args.flip:
                    board = cv2.rotate(board, cv2.ROTATE_180)
                detections = pieces.detect(board, args.confidence, args.imgsz)
                fen = detections_to_fen(detections, args.size, turn=turn, castling=args.castling, en_passant=args.en_passant)
                published = stabilizer.update(fen)
                if published is not None:
                    visible_fen = published
                    if policy is not None:
                        import chess

                        from lichess_pipeline.model import predict_top_moves
                        position = chess.Board(published)
                        if not position.is_valid():
                            visible_move = "FEN invalid: verify detections/context"
                        else:
                            moves = predict_top_moves(policy, published, top_k=1)
                            visible_move = f"Best move: {moves[0][0]} ({moves[0][1]:.1%})" if moves else "No legal move"
                draw_grid(board, regions)
                draw_detections(board, detections)
                cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3)
                cv2.imshow("Board + YOLO pieces", board)

            cv2.putText(preview, visible_fen, (20, max(36, preview.shape[0] - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            if visible_move:
                cv2.putText(preview, visible_move, (20, max(58, preview.shape[0] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord("t"):
                turn = "b" if turn == "w" else "w"
                stabilizer = FenStabilizer(args.stable_frames)
            if key == ord("f"):
                args.flip = not args.flip
                stabilizer = FenStabilizer(args.stable_frames)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
