"""Live camera pipeline: automatic board detection -> YOLO pieces -> FEN -> 5 models.

Example:
    uv run python scripts/live_chess_fen.py \
      --weights data/models/chess-pieces.pt --turn w --castling KQkq

Without ``--move-models`` the five project checkpoints that exist in
``data/models/`` are loaded (scratch, weight fine-tune, LoRA fine-tune, and the
two Maia-3 LoRA adapters). Each model contributes a coloured perspective arrow
with the move in SAN and its probability, SeeMate-style.

The piece checkpoint may use six shape classes (pawn, knight, bishop, rook,
queen, king) or twelve coloured classes. FEN requires coloured classes.  An ordinary COCO YOLO checkpoint cannot
recognise chess pieces.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, sleep
from typing import Any, NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import cv2
import numpy as np

from lichess_pipeline.chess_vision import (
    DEFAULT_BOARD_SIZE,
    PIECE_CLASSES,
    FenStabilizer,
    PieceDetection,
    PieceLabelSmoother,
    board_destination,
    crop_piece,
    detect_board_corners,
    detect_green_screen_board,
    detections_to_fen,
    fuse_piece_probabilities,
    piece_description_from_label,
    piece_symbol_from_label,
)

# The five models the project compares on the live screen, in this order.
DEFAULT_MOVE_MODELS = (
    "approach1_scratch.pt",
    "approach1_weight_ftuning.pt",
    "approach1_lora_ftuning.pt",
    "maia3_lora_human.pt",
    "maia3_lora_stockfish.pt",
)
# Display style per checkpoint stem: short label and BGR arrow colour.
POLICY_STYLES: dict[str, tuple[str, tuple[int, int, int]]] = {
    "approach1_scratch": ("Scratch", (255, 255, 0)),
    "approach1_weight_ftuning": ("Weight-FT", (0, 165, 255)),
    "approach1_lora_ftuning": ("LoRA-FT", (80, 255, 80)),
    "maia3_lora_human": ("Maia-Human", (255, 144, 30)),
    "maia3_lora_stockfish": ("Maia-SF", (235, 80, 235)),
}
FALLBACK_COLOURS = ((200, 200, 200), (0, 255, 200), (200, 120, 255))


class PolicyStyle(NamedTuple):
    """One loaded move predictor with the label and colour of its overlay."""

    label: str
    colour: tuple[int, int, int]
    model: Any


def style_for(path: Path, index: int) -> tuple[str, tuple[int, int, int]]:
    """Return the short label and BGR colour for a checkpoint."""
    return POLICY_STYLES.get(
        path.stem,
        (path.stem.replace("_", " ")[:16], FALLBACK_COLOURS[index % len(FALLBACK_COLOURS)]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="YOLO .pt/.onnx trained on six piece shapes or twelve coloured pieces")
    parser.add_argument("--board-weights", type=Path, help="Optional YOLO-seg model with a chessboard/board class")
    parser.add_argument(
        "--classifier",
        type=Path,
        default=Path("data/models/chess-piece-classifier.pt"),
        help="Optional secondary piece crop classifier (YOLO-cls) for refined recognition",
    )
    parser.add_argument("--no-classifier", action="store_true", help="Disable the stage-2 crop classifier")
    parser.add_argument("--classifier-weight", type=float, default=1.25,
                        help="Weight of the stage-2 classifier when fusing with the detector")
    parser.add_argument("--classifier-tta", action="store_true",
                        help="Enable horizontal-flip TTA on ambiguous classifier crops")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--size", type=int, default=DEFAULT_BOARD_SIZE, help="Rectified board side in pixels")
    parser.add_argument("--confidence", type=float, default=0.30, help="Minimum YOLO confidence")
    parser.add_argument(
        "--require-segmentation",
        action="store_true",
        help="Require a YOLO-seg checkpoint with one pixel mask per piece",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (train size of the v2 model)")
    parser.add_argument("--smooth", type=float, default=0.8,
                        help="Per-piece label voting decay across frames (1 = keep all history, 0 disables)")
    parser.add_argument("--device", default=None, help="Inference device: cpu, mps or 0 (automatic if omitted)")
    parser.add_argument("--scene", choices=("2d", "3d", "auto"), default="auto",
                        help="2d: box centre; 3d: box base (physical pieces); auto: pick per board (green screen board -> 2d, otherwise 3d)")
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
    parser.add_argument("--move-model", type=Path, help="Optional single trained policy checkpoint (legacy alias)")
    parser.add_argument(
        "--move-models",
        type=Path,
        nargs="+",
        help="Move checkpoints to compare on the live screen (default: the five project models that exist)",
    )
    parser.add_argument(
        "--architecture",
        choices=("auto", "scratch", "maia"),
        default="auto",
        help="Deprecated: the checkpoint file now decides its architecture",
    )
    return parser.parse_args()


class YoloPieceDetector:
    """Small adapter that avoids tying core vision code to Ultralytics types."""

    def __init__(
        self,
        weights: Path,
        classifier_weights: Path | None = None,
        device: str | None = None,
        *,
        classifier_weight: float = 1.25,
        classifier_tta: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("Falta Ultralytics. Ejecuta: uv add ultralytics") from error
        self.model = YOLO(weights)
        self.classifier = YOLO(classifier_weights) if classifier_weights and classifier_weights.is_file() else None
        self.classifier_imgsz = (
            int(getattr(self.classifier, "overrides", {}).get("imgsz", 128))
            if self.classifier is not None
            else 128
        )
        self.classifier_weight = classifier_weight
        self.classifier_tta = classifier_tta
        self.device = device
        self.is_segmentation = self.model.task == "segment"

    def warmup(self, imgsz: int, shape: tuple[int, int] = (480, 640)) -> None:
        """Compile/allocate the model before the first camera frame (removes the slow first prediction)."""
        blank = np.zeros((*shape, 3), dtype=np.uint8)
        for _ in range(2):
            self.model.predict(blank, imgsz=imgsz, device=self.device, verbose=False)
        if self.classifier is not None:
            crop_blank = [np.zeros((self.classifier_imgsz, self.classifier_imgsz, 3), dtype=np.uint8)]
            for _ in range(2):
                self.classifier.predict(crop_blank, imgsz=self.classifier_imgsz, device=self.device, verbose=False)

    def _classify_crops(self, crops: list[np.ndarray]) -> list[list[float]]:
        assert self.classifier is not None
        cls_results = self.classifier.predict(crops, imgsz=self.classifier_imgsz, device=self.device, verbose=False)
        all_probs: list[list[float]] = []
        for c_res in cls_results:
            probs = [0.0] * len(PIECE_CLASSES)
            for cid, cname in c_res.names.items():
                if cname in PIECE_CLASSES:
                    probs[PIECE_CLASSES.index(cname)] = float(c_res.probs.data[cid])
            all_probs.append(probs)
        return all_probs

    def detect(self, image: np.ndarray, confidence: float, imgsz: int) -> list[PieceDetection]:
        # agnostic NMS keeps one box per physical piece, so the same piece can
        # no longer appear as both "knight" and "pawn" at once.
        result = self.model.predict(image, conf=confidence, imgsz=imgsz, device=self.device, verbose=False,
                                    agnostic_nms=True, iou=0.5, max_det=64)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names: dict[int, str] = result.names
        polygons = result.masks.xy if result.masks is not None else []
        boxes_xyxy = [tuple(float(value) for value in box.xyxy[0].tolist()) for box in result.boxes]
        det_labels = [names[int(box.cls.item())] for box in result.boxes]
        det_confs = [float(box.conf.item()) for box in result.boxes]

        # Stage 2 crop classifier fusion (only for six-shape labels so 12-class colour labels keep their side).
        shape_indices = [i for i, lbl in enumerate(det_labels) if lbl in PIECE_CLASSES]
        if self.classifier is not None and shape_indices:
            crops = [crop_piece(image, boxes_xyxy[i], self.classifier_imgsz) for i in shape_indices]
            all_probs = self._classify_crops(crops)

            # Smart horizontal-flip TTA: only re-check ambiguous or disagreeing crops (~1-3 per frame).
            if self.classifier_tta:
                ambiguous = [
                    k for k, (idx, probs) in enumerate(zip(shape_indices, all_probs))
                    if max(probs) < 0.90 or PIECE_CLASSES[int(np.argmax(probs))] != det_labels[idx]
                ]
                if ambiguous:
                    flipped_crops = [cv2.flip(crops[k], 1) for k in ambiguous]
                    flipped_probs = self._classify_crops(flipped_crops)
                    for k, fprobs in zip(ambiguous, flipped_probs):
                        all_probs[k] = [0.5 * (a + b) for a, b in zip(all_probs[k], fprobs)]

            for idx, probs in zip(shape_indices, all_probs):
                f_label, f_conf = fuse_piece_probabilities(
                    det_labels[idx], det_confs[idx], probs, classifier_weight=self.classifier_weight
                )
                det_labels[idx] = f_label
                det_confs[idx] = f_conf

        detections: list[PieceDetection] = []
        for index in range(len(boxes_xyxy)):
            mask = np.asarray(polygons[index], dtype=np.int32) if index < len(polygons) else None
            detections.append(PieceDetection(det_labels[index], det_confs[index], boxes_xyxy[index], mask))
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
        # 720p keeps small pieces legible; the model only sees imgsz pixels anyway.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._lock = Lock()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._captured_at = 0.0
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
                    self._sequence += 1
                    self._captured_at = perf_counter()
            else:
                sleep(0.01)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def sample(self) -> tuple[np.ndarray | None, int, float]:
        with self._lock:
            return (None if self._frame is None else self._frame.copy(),
                    self._sequence, self._captured_at)

    def release(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.capture.release()


class LiveInference:
    """Run board localization and YOLO independently of the display loop."""

    def __init__(self, camera: LatestFrameCamera, pieces: YoloPieceDetector,
                 board_segmenter: YoloBoardSegmenter | None, policies: Any, args: argparse.Namespace) -> None:
        self.camera, self.pieces, self.board_segmenter, self.args = camera, pieces, board_segmenter, args
        if policies is None:
            self.policies: list[PolicyStyle] = []
        elif isinstance(policies, list):
            self.policies = policies
        else:  # a single model keeps the old constructor working
            self.policies = [PolicyStyle("Model", FALLBACK_COLOURS[0], policies)]
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)
        self.corners: np.ndarray | None = None
        self.board_image: np.ndarray | None = None
        self.fen = "Waiting for board"
        self.move = ""
        self.results: list[dict[str, Any]] = []
        self.inference_fps = 0.0
        self.last_result_at = 0.0
        self.latency_ms = 0.0
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
                self.fen, self.move, self.results = "Waiting for board", "", []
            elif key == ord("f"):
                self.flip = not self.flip
            elif key == ord("t"):
                self.turn = "b" if self.turn == "w" else "w"
            self.generation += 1

    def snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, str, str, float, float, float, list[dict[str, Any]]]:
        with self._lock:
            return (None if self.corners is None else self.corners.copy(),
                    None if self.board_image is None else self.board_image.copy(),
                    self.fen, self.move, self.inference_fps, self.last_result_at, self.latency_ms,
                    [dict(result) for result in self.results])


    def _run(self) -> None:
        destination = board_destination(self.args.size)
        stabilizer = FenStabilizer(self.args.stable_frames)
        smoother = PieceLabelSmoother(decay=self.args.smooth) if self.args.smooth > 0 else None
        processed = 0
        last_sequence = -1
        next_board_search = 0.0
        scene = self.args.scene if self.args.scene != "auto" else "3d"
        while not self._stop.is_set():
            frame, sequence, captured_at = self.camera.sample()
            if frame is None or sequence == last_sequence:
                sleep(0.01)
                continue
            last_sequence = sequence
            started = perf_counter()
            detections = self.pieces.detect(frame, self.args.confidence, self.args.imgsz)
            if smoother is not None:
                detections = smoother.update(detections)
            annotated = frame.copy()
            draw_detections(annotated, detections)
            with self._lock:
                corners = None if self.corners is None else self.corners.copy()
                generation, turn, flip = self.generation, self.turn, self.flip
            redetect = corners is None or (self.args.redetect_every > 0 and processed % self.args.redetect_every == 0)
            if redetect and started >= next_board_search:
                next_board_search = started + 1.0
                auto = self.args.scene == "auto"
                found = detect_green_screen_board(frame) if (self.args.screen_board or auto) else None
                if found is not None and auto:
                    scene = "2d"
                if found is None:
                    found = detect_board_corners(frame)
                    if found is not None and auto:
                        scene = "3d"
                if found is None and self.board_segmenter is not None:
                    found = self.board_segmenter.detect(frame, self.args.confidence, self.args.imgsz)
                if found is not None:
                    corners = found
            mapped = []
            if corners is not None:
                transform = cv2.getPerspectiveTransform(corners, destination)
                for detection in detections:
                    x1, y1, x2, y2 = detection.xyxy
                    anchor = np.array([[[(x1 + x2) / 2,
                                         y2 if scene == "3d" else (y1 + y2) / 2]]], dtype=np.float32)
                    x, y = cv2.perspectiveTransform(anchor, transform)[0, 0]
                    if flip:
                        x, y = self.args.size - 1 - x, self.args.size - 1 - y
                    mapped.append(PieceDetection(detection.label, detection.confidence,
                                                 (float(x), float(y), float(x), float(y))))
            # Six-class shape models cannot identify allegiance for FEN.
            can_publish = corners is not None and bool(detections) and all(piece_symbol_from_label(d.label) is not None for d in detections)
            fen = detections_to_fen(mapped, self.args.size, turn=turn,
                                    castling=self.args.castling, en_passant=self.args.en_passant)
            published = stabilizer.update(fen) if can_publish else None
            results: list[dict[str, Any]] = []
            message = ""
            if published is not None and self.policies:
                import chess

                from lichess_pipeline.model import predict_top_moves
                position = chess.Board(published)
                if position.is_valid():
                    for style in self.policies:
                        moves = predict_top_moves(style.model, published, top_k=1)
                        if not moves:
                            continue
                        uci, probability = moves[0]
                        try:
                            san = position.san(chess.Move.from_uci(uci))
                        except (ValueError, AssertionError):
                            san = uci
                        results.append(
                            {
                                "label": style.label,
                                "colour": style.colour,
                                "uci": uci,
                                "san": san,
                                "probability": probability,
                            }
                        )
                    message = ", ".join(
                        f"{item['label']}: {item['san']} {item['probability']:.0%}"
                        for item in results
                    )
                    if not message:
                        message = "No legal move"
                else:
                    message = "FEN invalid: verify detections/context"
            elapsed = max(perf_counter() - started, 1e-6)
            with self._lock:
                if generation != self.generation:
                    stabilizer = FenStabilizer(self.args.stable_frames)
                    continue
                self.corners, self.board_image = corners, annotated
                self.latency_ms = (perf_counter() - captured_at) * 1000
                if not can_publish:
                    self.fen = "FEN needs board + coloured piece labels"
                    self.move, self.results = "", []
                    stabilizer = FenStabilizer(self.args.stable_frames)
                if published is not None:
                    self.fen = published
                    if message:
                        self.move, self.results = message, results
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


def draw_detections(image: np.ndarray, detections: list[PieceDetection]) -> None:
    """Draw shape labels and model confidence, without grids or mask fills."""
    height, width = image.shape[:2]
    for detection in detections:
        colour = (0, 220, 80)
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        description = piece_description_from_label(detection.label) or detection.label
        description = description.removesuffix(" blanco").removesuffix(" negro")
        # OpenCV's built-in font does not support accented characters.
        description = description.replace("ó", "o")
        text = f"{description} {detection.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        left = max(0, min(x1, width - tw - 6))
        bottom = min(height - baseline - 1, max(th + 6, y1))
        cv2.rectangle(image, (left, bottom - th - 6), (min(width - 1, left + tw + 6), bottom + baseline), colour, -1)
        cv2.putText(image, text, (left + 3, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def load_policies(paths: list[Path], device: str | None = None) -> list[PolicyStyle]:
    """Load every existing checkpoint, skipping (and reporting) the missing ones."""
    from lichess_pipeline.model.checkpoints import load_policy_checkpoint

    policies: list[PolicyStyle] = []
    for index, path in enumerate(paths):
        if not path.is_file():
            print(f"Skip {path} (not found)", file=sys.stderr)
            continue
        try:
            model = load_policy_checkpoint(path, device=device)
        except Exception as error:  # noqa: BLE001 - one broken file must not stop the demo
            print(f"Skip {path.name}: {error}", file=sys.stderr)
            continue
        label, colour = style_for(path, index)
        policies.append(PolicyStyle(label, colour, model))
        print(f"Loaded {label}: {path}")
    return policies


def requested_policy_paths(args: argparse.Namespace) -> list[Path]:
    """CLI paths, or the five project models that exist under data/models."""
    if args.move_models:
        return list(args.move_models)
    if args.move_model:
        return [args.move_model]
    models_dir = Path("data/models")
    return [path for path in (models_dir / name for name in DEFAULT_MOVE_MODELS) if path.is_file()]


# --------------------------------------------------------------------------
# SeeMate-style AR overlay
# --------------------------------------------------------------------------

def square_center(square: str, board_size: int, flip: bool) -> tuple[float, float]:
    """Centre of ``square`` in rectified-board pixels, honouring ``flip``."""
    file_index = ord(square[0]) - ord("a")
    rank_index = 8 - int(square[1])
    x = (file_index + 0.5) * board_size / 8
    y = (rank_index + 0.5) * board_size / 8
    if flip:
        x, y = board_size - 1 - x, board_size - 1 - y
    return x, y


def _projector(corners: np.ndarray, board_size: int):
    matrix = cv2.getPerspectiveTransform(
        board_destination(board_size), np.asarray(corners, dtype=np.float32)
    )

    def project(point: tuple[float, float]) -> tuple[int, int]:
        projected = cv2.perspectiveTransform(
            np.array([[point]], dtype=np.float32), matrix
        )[0, 0]
        return int(np.rint(projected[0])), int(np.rint(projected[1]))

    return project


def _is_knight_move(from_square: str, to_square: str) -> bool:
    file_delta = abs(ord(from_square[0]) - ord(to_square[0]))
    rank_delta = abs(int(from_square[1]) - int(to_square[1]))
    return sorted((file_delta, rank_delta)) == [1, 2]


def _arrow_polyline(
    from_square: str, to_square: str, board_size: int, flip: bool
) -> list[tuple[float, float]]:
    """Rectified-board points for a straight or L-shaped (knight) arrow."""
    start = square_center(from_square, board_size, flip)
    end = square_center(to_square, board_size, flip)
    if _is_knight_move(from_square, to_square):
        knee = square_center(f"{to_square[0]}{from_square[1]}", board_size, flip)
        return [start, knee, end]
    return [start, end]


def _shift_perpendicular(
    points: list[tuple[int, int]], offset: float
) -> list[tuple[int, int]]:
    """Fan out arrows that share the same move so every colour stays visible."""
    if not offset or len(points) < 2:
        return points
    (x1, y1), (x2, y2) = points[0], points[-1]
    dx, dy = x2 - x1, y2 - y1
    length = float(np.hypot(dx, dy)) or 1.0
    ox, oy = -dy / length * offset, dx / length * offset
    return [(int(x + ox), int(y + oy)) for x, y in points]


def _translucent_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int],
    alpha: float = 0.75,
) -> None:
    height, width = image.shape[:2]
    x1, y1 = max(0, top_left[0]), max(0, top_left[1])
    x2, y2 = min(width - 1, bottom_right[0]), min(height - 1, bottom_right[1])
    if x2 <= x1 or y2 <= y1:
        return
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def draw_ar_overlay(
    image: np.ndarray,
    corners: np.ndarray,
    board_size: int,
    results: list[dict[str, Any]],
    turn: str,
    flip: bool,
    fen: str = "",
) -> None:
    """Project the board and each model's move onto the live camera frame."""
    height, width = image.shape[:2]
    project = _projector(corners, board_size)
    outer = [
        project((0, 0)),
        project((board_size - 1, 0)),
        project((board_size - 1, board_size - 1)),
        project((0, board_size - 1)),
    ]
    cv2.polylines(image, [np.array(outer, np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)

    groups: dict[str, list[int]] = {}
    for index, item in enumerate(results):
        groups.setdefault(str(item["uci"]), []).append(index)

    anchors: dict[int, tuple[int, int]] = {}
    for indices in groups.values():
        for order, index in enumerate(indices):
            item = results[index]
            uci = str(item["uci"])
            points = _arrow_polyline(uci[:2], uci[2:4], board_size, flip)
            camera_points = [project(point) for point in points]
            offset = 0.0 if len(indices) == 1 else (order - (len(indices) - 1) / 2) * 12
            camera_points = _shift_perpendicular(camera_points, offset)
            segments = list(itertools.pairwise(camera_points))
            for start, end in segments[:-1]:
                cv2.line(image, start, end, item["colour"], 3, cv2.LINE_AA)
            cv2.arrowedLine(
                image,
                segments[-1][0],
                segments[-1][1],
                item["colour"],
                3,
                cv2.LINE_AA,
                tipLength=0.2,
            )
            anchors[index] = (camera_points[-1][0], camera_points[-1][1] - 6 - order * 22)

    for index, item in enumerate(results):
        anchor = anchors[index]
        text = f"{item['san']} {item['probability']:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x = min(max(4, anchor[0] + 8), max(4, width - tw - 20))
        y = min(max(th + 8, anchor[1]), max(th + 8, height - baseline - 8))
        _translucent_rect(image, (x - 5, y - th - 5), (x + tw + 5, y + baseline), item["colour"], 0.85)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    white = sum(1 for char in fen.split(" ")[0] if char.isupper()) if fen else 0
    black = sum(1 for char in fen.split(" ")[0] if char.islower()) if fen else 0
    hud_lines = [
        f"TURN: {'WHITE' if turn == 'w' else 'BLACK'}  [t]",
        f"PIECES: white {white} / black {black}",
        f"FEN: {fen[:32]}",
    ]
    top_left = (10, 8)
    panel_height = 24 * len(hud_lines) + 8
    _translucent_rect(image, top_left, (430, top_left[1] + panel_height), (24, 24, 24), 0.6)
    for row, line in enumerate(hud_lines):
        cv2.putText(image, line, (18, 30 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    if results:
        legend_top = height - 26 * len(results) - 14
        _translucent_rect(image, (10, legend_top - 6), (360, height - 10), (24, 24, 24), 0.6)
        for row, item in enumerate(results):
            y = legend_top + row * 26 + 14
            cv2.circle(image, (22, y - 5), 6, item["colour"], -1, cv2.LINE_AA)
            text = f"{item['label']}: {item['san']} {item['probability']:.0%}"
            cv2.putText(image, text, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)



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
        classifier_path = None if args.no_classifier else args.classifier
        pieces = YoloPieceDetector(
            args.weights,
            classifier_path,
            args.device,
            classifier_weight=args.classifier_weight,
            classifier_tta=args.classifier_tta,
        )
        pieces.warmup(args.imgsz)
        if args.require_segmentation and not pieces.is_segmentation:
            raise RuntimeError("--require-segmentation needs YOLO-seg weights, not a detection checkpoint")
        board_segmenter = YoloBoardSegmenter(args.board_weights) if args.board_weights else None
        policies = load_policies(requested_policy_paths(args), args.device)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    if not policies:
        print(
            "Sin checkpoints de jugadas: solo se mostrará la visión. Pásalos con --move-models.",
            file=sys.stderr,
        )

    camera = LatestFrameCamera(args.camera)
    if not camera.is_opened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    camera.start()

    inference = LiveInference(camera, pieces, board_segmenter, policies, args)
    inference.start()
    print("Live camera + background YOLO. r recalibrates; q/Esc exits; t toggles turn; f flips orientation.")
    try:
        while True:
            frame = camera.latest()
            if frame is None:
                cv2.waitKey(1)
                continue
            (
                corners,
                board,
                visible_fen,
                visible_move,
                inference_fps,
                updated_at,
                latency_ms,
                results,
            ) = inference.snapshot()
            preview = frame.copy()
            if corners is None:
                cv2.putText(preview, "Looking for 8x8 board grid", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 3)
                if results:
                    draw_ar_overlay(
                        preview,
                        corners,
                        args.size,
                        results,
                        inference.turn,
                        inference.flip,
                        visible_fen,
                    )
            if board is not None:
                cv2.imshow("Pieces | boxes + confidence", board)
            age = perf_counter() - updated_at if updated_at else 0.0
            geometry = "locked" if corners is not None else "searching"
            cv2.putText(preview, f"CAM LIVE | YOLO {inference_fps:.1f} FPS | {latency_ms:.0f} ms | result {age:.1f}s old | board {geometry}", (20, 36),
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
