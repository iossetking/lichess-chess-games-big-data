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
from threading import Event, Lock, Thread
from time import perf_counter, sleep
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
    detect_green_screen_board,
    detections_to_fen,
    piece_description_from_label,
    piece_symbol_from_label,
    square_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="YOLO .pt/.onnx trained on the 12 chess pieces")
    parser.add_argument("--board-weights", type=Path, help="Optional YOLO-seg model with a chessboard/board class")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE, help="Rectified board side in pixels")
    parser.add_argument("--confidence", type=float, default=0.55, help="Minimum YOLO confidence")
    parser.add_argument(
        "--require-segmentation",
        action="store_true",
        help="Require a YOLO-seg checkpoint with one pixel mask per piece",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--screen-board", action="store_true", help="Fast green/cream 2D screen board detection")
    parser.add_argument(
        "--redetect-every",
        type=int,
        default=0,
        help="Re-detect the board every N processed frames; 0 locks it until r is pressed",
    )
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
        self.is_segmentation = self.model.task == "segment"

    def detect(self, image: np.ndarray, confidence: float, imgsz: int) -> list[PieceDetection]:
        result = self.model.predict(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
        if result.boxes is None:
            return []
        names: dict[int, str] = result.names
        polygons = result.masks.xy if result.masks is not None else []
        detections: list[PieceDetection] = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls.item())
            xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            mask = np.asarray(polygons[index], dtype=np.int32) if index < len(polygons) else None
            detections.append(PieceDetection(names[class_id], float(box.conf.item()), xyxy, mask))
        return detections


class LatestFrameCamera:
    """Continuously capture camera frames and expose only the newest one.

    Inference can be slower than the camera.  Reading from ``VideoCapture`` in
    the inference loop then accumulates old frames and feels like a screenshot.
    This small producer thread discards stale frames so the displayed board is
    always the most recent view of the real board.
    """

    def __init__(self, camera_index: int) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = Lock()
        self._frame: np.ndarray | None = None
        self._stopped = Event()
        self._thread: Thread | None = None

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def start(self) -> None:
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stopped.is_set():
            ok, frame = self.capture.read()
            if ok:
                with self._lock:
                    self._frame = frame
            else:
                sleep(0.01)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.capture.release()


class LiveInference:
    """Run board localization and YOLO independently of the display loop."""

    def __init__(self, camera: LatestFrameCamera, pieces: YoloPieceDetector,
                 board_segmenter: YoloBoardSegmenter | None, policy: Any, args: argparse.Namespace) -> None:
        self.camera, self.pieces, self.board_segmenter, self.policy, self.args = camera, pieces, board_segmenter, policy, args
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self.corners: np.ndarray | None = None
        self.board_image: np.ndarray | None = None
        self.fen = "Waiting for board"
        self.move = ""
        self.inference_fps = 0.0
        self.last_result_at = 0.0
        self.turn, self.flip = args.turn, args.flip
        self.generation = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def control(self, key: int) -> None:
        with self._lock:
            if key == ord("r"):
                self.corners = None
                self.board_image = None
                self.fen, self.move = "Waiting for board", ""
            elif key == ord("f"):
                self.flip = not self.flip
            elif key == ord("t"):
                self.turn = "b" if self.turn == "w" else "w"
            self.generation += 1

    def snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, str, str, float, float]:
        with self._lock:
            return (None if self.corners is None else self.corners.copy(),
                    None if self.board_image is None else self.board_image.copy(),
                    self.fen, self.move, self.inference_fps, self.last_result_at)

    def _run(self) -> None:
        destination = board_destination(self.args.size)
        regions = square_regions(self.args.size)
        stabilizer = FenStabilizer(self.args.stable_frames)
        processed = 0
        while not self._stop.is_set():
            frame = self.camera.latest()
            if frame is None:
                sleep(0.01)
                continue
            started = perf_counter()
            with self._lock:
                corners = None if self.corners is None else self.corners.copy()
                generation, turn, flip = self.generation, self.turn, self.flip
            redetect = corners is None or (self.args.redetect_every > 0 and processed % self.args.redetect_every == 0)
            if redetect:
                found = detect_green_screen_board(frame) if self.args.screen_board else None
                if found is None:
                    found = detect_board_corners(frame)
                if found is None and self.board_segmenter is not None:
                    found = self.board_segmenter.detect(frame, self.args.confidence, self.args.imgsz)
                if found is not None:
                    corners = found
            if corners is None:
                sleep(0.01)
                continue
            board = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(corners, destination),
                                        (self.args.size, self.args.size))
            if flip:
                board = cv2.rotate(board, cv2.ROTATE_180)
            detections = self.pieces.detect(board, self.args.confidence, self.args.imgsz)
            fen = detections_to_fen(detections, self.args.size, turn=turn,
                                    castling=self.args.castling, en_passant=self.args.en_passant)
            published = stabilizer.update(fen)
            move = None
            if published is not None and self.policy is not None:
                import chess

                from lichess_pipeline.model import predict_top_moves
                position = chess.Board(published)
                if position.is_valid():
                    moves = predict_top_moves(self.policy, published, top_k=1)
                    move = f"Best move: {moves[0][0]} ({moves[0][1]:.1%})" if moves else "No legal move"
                else:
                    move = "FEN invalid: verify detections/context"
            draw_grid(board, regions)
            draw_detections(board, detections)
            elapsed = max(perf_counter() - started, 1e-6)
            with self._lock:
                if generation != self.generation:
                    stabilizer = FenStabilizer(self.args.stable_frames)
                    continue
                self.corners, self.board_image = corners, board
                if published is not None:
                    self.fen = published
                    if move is not None:
                        self.move = move
                self.inference_fps = 1 / elapsed
                self.last_result_at = perf_counter()
            processed += 1


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
    """Draw type/colour-labelled boxes and real masks from a YOLO-seg model."""
    mask_overlay = image.copy()
    has_masks = False
    for detection in detections:
        symbol = piece_symbol_from_label(detection.label)
        colour = (255, 220, 0) if symbol is not None and symbol.isupper() else (0, 70, 255)
        if detection.mask is not None and len(detection.mask) >= 3:
            cv2.fillPoly(mask_overlay, [detection.mask.reshape(-1, 1, 2)], colour)
            has_masks = True
    if has_masks:
        cv2.addWeighted(mask_overlay, 0.35, image, 0.65, 0, dst=image)
    for detection in detections:
        symbol = piece_symbol_from_label(detection.label)
        colour = (255, 220, 0) if symbol is not None and symbol.isupper() else (0, 70, 255)
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        description = piece_description_from_label(detection.label) or detection.label
        cv2.putText(image, f"{description} {detection.confidence:.2f}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)


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
    if args.redetect_every < 0:
        print("--redetect-every must be zero or positive", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"YOLO piece weights not found: {args.weights}", file=sys.stderr)
        return 2
    if args.board_weights and not args.board_weights.is_file():
        print(f"YOLO board weights not found: {args.board_weights}", file=sys.stderr)
        return 2

    try:
        pieces = YoloPieceDetector(args.weights)
        if args.require_segmentation and not pieces.is_segmentation:
            raise RuntimeError("--require-segmentation needs YOLO-seg weights, not a detection checkpoint")
        board_segmenter = YoloBoardSegmenter(args.board_weights) if args.board_weights else None
        policy = load_policy(args.move_model, args.architecture) if args.move_model else None
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    camera = LatestFrameCamera(args.camera)
    if not camera.is_opened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    camera.start()

    inference = LiveInference(camera, pieces, board_segmenter, policy, args)
    inference.start()
    print("Live camera + background YOLO. r recalibrates; q/Esc exits; t toggles turn; f flips orientation.")
    try:
        while True:
            frame = camera.latest()
            if frame is None:
                cv2.waitKey(1)
                continue
            corners, board, visible_fen, visible_move, inference_fps, updated_at = inference.snapshot()
            preview = frame.copy()
            if corners is None:
                cv2.putText(preview, "Looking for 8x8 board grid", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3)
            if board is not None:
                cv2.imshow("Board + YOLO pieces", board)
            age = perf_counter() - updated_at if updated_at else 0.0
            geometry = "locked" if corners is not None else "searching"
            cv2.putText(preview, f"CAM LIVE | YOLO {inference_fps:.1f} FPS | result {age:.1f}s old | board {geometry}", (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(preview, visible_fen, (20, max(36, preview.shape[0] - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            if visible_move:
                cv2.putText(preview, visible_move, (20, max(58, preview.shape[0] - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key in (ord("t"), ord("f"), ord("r")):
                inference.control(key)
    finally:
        inference.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
