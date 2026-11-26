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
    repair_king_queen_detections,
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
    parser.add_argument("--weights", type=Path, required=False,
                        help="YOLO .pt/.onnx trained on six piece shapes or twelve coloured pieces")
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
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera index (e.g. 0, 1 for Continuity/virtual cam) or stream URL (e.g. http://<ip>:8080/video)",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List available local camera indices and exit")
    parser.add_argument(
        "--screen-capture",
        action="store_true",
        help="Capture the live computer screen instead of a camera (implies --screen-board)",
    )
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index for --screen-capture (1 = primary)")
    parser.add_argument(
        "--select-region",
        action="store_true",
        help="Drag a rectangle around the board on screen when using --screen-capture",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Fixed screen region 'left,top,width,height' for --screen-capture",
    )
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
    parser.add_argument(
        "--hold-missing",
        type=int,
        default=3,
        help="Frames to hold established pieces across brief misses to reduce blinking (0 disables)",
    )
    parser.add_argument(
        "--single-window",
        action="store_true",
        help="Show a single combined window instead of separate camera and detection windows",
    )
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


def parse_camera_source(raw: str | int) -> int | str:
    """Parse ``--camera`` as a numeric device index or a video stream URL."""
    if isinstance(raw, int):
        return raw
    text = raw.strip()
    if text.isdigit():
        return int(text)
    return text


def parse_region(spec: str) -> dict[str, int]:
    """Parse ``left,top,width,height`` into an ``mss`` capture region dict."""
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("--region must be four comma-separated integers: left,top,width,height")
    left, top, width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("--region width and height must be positive")
    return {"left": left, "top": top, "width": width, "height": height}


def list_available_cameras(max_index: int = 8) -> list[tuple[int, int, int]]:
    """Probe local camera indices and return ``(index, width, height)`` for open ones."""
    available: list[tuple[int, int, int]] = []
    for index in range(max_index):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append((index, width, height))
        finally:
            capture.release()
    return available


MACOS_SCREEN_PERMISSION_MSG = (
    "WARNING: macOS Screen Recording permission is not enabled for this terminal/IDE. "
    "Without it, macOS hides app/browser windows and returns a blank gray/wallpaper image.\n"
    "Enable it in: System Settings -> Privacy & Security -> Screen & System Audio Recording, "
    "then fully quit (Cmd+Q) and reopen your terminal/IDE."
)


def _macos_screen_capture_access(*, request: bool = False) -> bool | None:
    """Return macOS Screen Recording permission status, optionally requesting access."""
    if sys.platform != "darwin":
        return None
    import ctypes

    try:
        core_graphics = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/Versions/Current/CoreGraphics"
        )
        core_graphics.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        allowed = bool(core_graphics.CGPreflightScreenCaptureAccess())
        if not allowed and request:
            core_graphics.CGRequestScreenCaptureAccess.restype = ctypes.c_bool
            core_graphics.CGRequestScreenCaptureAccess()
        return allowed
    except (OSError, AttributeError):
        return None


def select_screen_region(monitor_index: int = 1) -> dict[str, int] | None:
    """Capture a monitor snapshot and let the user drag a board region with the mouse."""
    try:
        import mss
    except ImportError as error:
        raise RuntimeError("Falta mss. Ejecuta: uv add mss") from error

    if _macos_screen_capture_access(request=True) is False:
        print(MACOS_SCREEN_PERMISSION_MSG, file=sys.stderr)

    with mss.mss() as sct:
        if monitor_index < 0 or monitor_index >= len(sct.monitors):
            raise RuntimeError(f"Monitor {monitor_index} not found (available: 0..{len(sct.monitors) - 1})")
        mon = sct.monitors[monitor_index]
        shot = sct.grab(mon)

    image = np.asarray(shot)[:, :, :3].copy()
    img_h, img_w = image.shape[:2]
    scale_x = mon["width"] / img_w
    scale_y = mon["height"] / img_h
    disp_scale = min(1.0, 1280.0 / img_w, 800.0 / img_h)
    display = cv2.resize(image, None, fx=disp_scale, fy=disp_scale) if disp_scale < 1.0 else image
    window_name = "Select chessboard region (Enter=confirm, c=cancel)"
    rx, ry, rw, rh = cv2.selectROI(window_name, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
    if rw <= 0 or rh <= 0:
        return None
    return {
        "left": mon["left"] + round((rx / disp_scale) * scale_x),
        "top": mon["top"] + round((ry / disp_scale) * scale_y),
        "width": max(8, round((rw / disp_scale) * scale_x)),
        "height": max(8, round((rh / disp_scale) * scale_y)),
    }


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

    def __init__(self, camera_source: int | str) -> None:
        self.capture = cv2.VideoCapture(camera_source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(camera_source, int):
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


class ScreenCaptureSource:
    """Continuously capture a live monitor or region using ``mss``."""

    def __init__(
        self,
        monitor_index: int = 1,
        region: dict[str, int] | None = None,
        fps_limit: float = 30.0,
    ) -> None:
        try:
            import mss
        except ImportError as error:
            raise RuntimeError("Falta mss. Ejecuta: uv add mss") from error
        self.monitor_index = monitor_index
        self.fps_limit = fps_limit
        self._lock = Lock()
        self._region: dict[str, int] | None = dict(region) if region is not None else None
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._captured_at = 0.0
        self._stopped = Event()
        self._thread: Thread | None = None
        self._opened = False
        self.permission_denied = _macos_screen_capture_access(request=True) is False
        if self.permission_denied:
            print(MACOS_SCREEN_PERMISSION_MSG, file=sys.stderr)

        with mss.mss() as sct:
            for idx, mon in enumerate(sct.monitors):
                if idx == 0:
                    continue
                print(f"Screen monitor {idx}: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})")
            if monitor_index < 0 or monitor_index >= len(sct.monitors):
                raise RuntimeError(
                    f"Monitor {monitor_index} not found (available: 0..{len(sct.monitors) - 1})"
                )
            target = self._region or sct.monitors[monitor_index]
            try:
                shot = sct.grab(target)
                _ = np.asarray(shot)[:, :, :3].copy()
                self._opened = True
            except Exception as error:
                raise RuntimeError(
                    "Could not capture screen. On macOS, enable Screen Recording permission "
                    "for your terminal/IDE in System Settings -> Privacy & Security -> Screen Recording."
                ) from error

    def is_opened(self) -> bool:
        return self._opened

    def set_region(self, region: dict[str, int] | None) -> None:
        with self._lock:
            self._region = dict(region) if region is not None else None

    def start(self) -> None:
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        import mss

        min_interval = 1.0 / max(self.fps_limit, 1.0)
        with mss.mss() as sct:
            while not self._stopped.is_set():
                started = perf_counter()
                with self._lock:
                    target = dict(self._region) if self._region is not None else sct.monitors[self.monitor_index]
                try:
                    shot = sct.grab(target)
                    frame = np.asarray(shot)[:, :, :3].copy()
                except Exception:  # noqa: BLE001 - transient screen capture error must not crash worker
                    sleep(0.05)
                    continue
                with self._lock:
                    self._frame = frame
                    self._sequence += 1
                    self._captured_at = perf_counter()
                elapsed = perf_counter() - started
                remaining = min_interval - elapsed
                if remaining > 0:
                    sleep(remaining)

    def latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def sample(self) -> tuple[np.ndarray | None, int, float]:
        with self._lock:
            return (
                None if self._frame is None else self._frame.copy(),
                self._sequence,
                self._captured_at,
            )

    def release(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1)


class LiveInference:
    """Run board localization and YOLO independently of the display loop."""

    def __init__(self, camera: Any, pieces: YoloPieceDetector,
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
        screen_capture = bool(getattr(self.args, "screen_capture", False))
        is_screen_mode = bool(getattr(self.args, "screen_board", False) or screen_capture)
        hold_missing = max(0, int(getattr(self.args, "hold_missing", 0)))

        def make_smoother() -> PieceLabelSmoother | None:
            if self.args.smooth <= 0:
                return None
            return PieceLabelSmoother(decay=self.args.smooth, hold_missing=hold_missing)

        smoother = make_smoother()
        in_rectified_space = False
        processed = 0
        last_sequence = -1
        next_board_search = 0.0
        scene = "2d" if is_screen_mode else (self.args.scene if self.args.scene != "auto" else "3d")
        while not self._stop.is_set():
            frame, sequence, captured_at = self.camera.sample()
            if frame is None or sequence == last_sequence:
                sleep(0.01)
                continue
            last_sequence = sequence
            started = perf_counter()
            with self._lock:
                corners = None if self.corners is None else self.corners.copy()
                generation, turn, flip = self.generation, self.turn, self.flip
            region_is_board = bool(getattr(self.args, "region_is_board", False))
            redetect = corners is None or (self.args.redetect_every > 0 and processed % self.args.redetect_every == 0)
            if redetect and started >= next_board_search:
                next_board_search = started + 1.0
                auto = self.args.scene == "auto"
                found = detect_green_screen_board(frame) if (is_screen_mode or auto) else None
                if found is not None and auto:
                    scene = "2d"
                if found is None and not region_is_board:
                    found = detect_board_corners(frame)
                    if found is not None and auto and not is_screen_mode:
                        scene = "3d"
                if found is None and region_is_board:
                    fh, fw = frame.shape[:2]
                    found = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype=np.float32)
                if found is None and self.board_segmenter is not None:
                    found = self.board_segmenter.detect(frame, self.args.confidence, self.args.imgsz)
                if found is not None:
                    corners = found

            use_rectified = is_screen_mode and corners is not None
            if use_rectified != in_rectified_space:
                in_rectified_space = use_rectified
                smoother = make_smoother()

            mapped: list[PieceDetection] = []
            if use_rectified:
                assert corners is not None
                transform = cv2.getPerspectiveTransform(corners, destination)
                rectified = cv2.warpPerspective(frame, transform, (self.args.size, self.args.size))
                detections = self.pieces.detect(rectified, self.args.confidence, self.args.imgsz)
                if smoother is not None:
                    detections = smoother.update(detections)
                detections = repair_king_queen_detections(detections)
                annotated = rectified.copy()
                draw_detections(annotated, detections)
                for detection in detections:
                    x1, y1, x2, y2 = detection.xyxy
                    x, y = (x1 + x2) / 2, (y1 + y2) / 2
                    if flip:
                        x, y = self.args.size - 1 - x, self.args.size - 1 - y
                    mapped.append(PieceDetection(detection.label, detection.confidence,
                                                 (float(x), float(y), float(x), float(y))))
            else:
                detections = self.pieces.detect(frame, self.args.confidence, self.args.imgsz)
                if smoother is not None:
                    detections = smoother.update(detections)
                if corners is not None:
                    detections = repair_king_queen_detections(detections)
                annotated = frame.copy()
                draw_detections(annotated, detections)
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
            valid_position = False
            display_fen = published or ""
            if published is not None:
                import chess

                position = chess.Board(published)
                position.castling_rights = position.clean_castling_rights()
                cleaned_fen = position.fen()
                valid_position = position.is_valid()
                if valid_position:
                    display_fen = cleaned_fen
                    if self.policies:
                        from lichess_pipeline.model import predict_top_moves
                        for style in self.policies:
                            moves = predict_top_moves(style.model, cleaned_fen, top_k=1)
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
                    status = position.status()
                    flags = [flag.name for flag in chess.Status if flag and (status & flag)]
                    if status & chess.Status.OPPOSITE_CHECK:
                        message = "FEN invalid: OPPOSITE_CHECK (press t to switch turn)"
                    elif flags:
                        message = f"FEN invalid: {', '.join(flags)}"
                    else:
                        message = "FEN invalid: verify detections/context"
            elapsed = max(perf_counter() - started, 1e-6)
            with self._lock:
                if generation != self.generation:
                    stabilizer = FenStabilizer(self.args.stable_frames)
                    smoother = make_smoother()
                    continue
                self.corners, self.board_image = corners, annotated
                self.latency_ms = (perf_counter() - captured_at) * 1000
                if not can_publish:
                    self.fen = "FEN needs board + coloured piece labels"
                    self.move, self.results = "", []
                    stabilizer = FenStabilizer(self.args.stable_frames)
                elif published is not None:
                    self.fen = display_fen
                    if valid_position:
                        if message:
                            self.move, self.results = message, results
                    else:
                        self.move, self.results = message, []
                elif self.fen.startswith(("Waiting", "FEN ", "Stabilizing")):
                    self.fen = f"Stabilizing FEN ({stabilizer._count}/{self.args.stable_frames})"
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
    """Draw high-contrast piece boxes and dark-pill labels (cyan=white, orange=black)."""
    height, width = image.shape[:2]
    for detection in detections:
        symbol = piece_symbol_from_label(detection.label)
        if symbol is None:
            colour = (220, 100, 240)
        elif symbol.isupper():
            colour = (255, 210, 40)
        else:
            colour = (50, 145, 255)
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        description = piece_description_from_label(detection.label) or detection.label
        description = description.removesuffix(" blanco").removesuffix(" negro")
        # OpenCV's built-in font does not support accented characters.
        description = description.replace("ó", "o")
        text = f"{description} {detection.confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        left = max(0, min(x1, width - tw - 6))
        bottom = min(height - baseline - 1, max(th + 6, y1))
        right = min(width - 1, left + tw + 6)
        top = max(0, bottom - th - 6)
        cv2.rectangle(image, (left, top), (right, bottom + baseline), (24, 26, 32), -1)
        cv2.rectangle(image, (left, top), (right, bottom + baseline), colour, 1)
        cv2.putText(image, text, (left + 3, bottom - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)


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
    cv2.polylines(image, [np.array(outer, np.int32)], True, (255, 220, 40), 2, cv2.LINE_AA)

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
    _translucent_rect(image, top_left, (430, top_left[1] + panel_height), (18, 20, 24), 0.78)
    for row, line in enumerate(hud_lines):
        cv2.putText(image, line, (18, 30 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)

    if results:
        legend_top = height - 26 * len(results) - 14
        _translucent_rect(image, (10, legend_top - 6), (360, height - 10), (18, 20, 24), 0.78)
        for row, item in enumerate(results):
            y = legend_top + row * 26 + 14
            cv2.circle(image, (22, y - 5), 6, item["colour"], -1, cv2.LINE_AA)
            text = f"{item['label']}: {item['san']} {item['probability']:.0%}"
            cv2.putText(image, text, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)



def main() -> int:
    args = parse_args()
    if args.list_cameras:
        cameras = list_available_cameras()
        if not cameras:
            print("No local cameras found.")
        else:
            for index, width, height in cameras:
                print(f"Camera {index}: {width}x{height}")
        print(
            "Tip: iPhone Continuity Camera / virtual webcams appear as an index (e.g. --camera 1); "
            "Samsung/Android can stream via IP Webcam or DroidCam URL (e.g. --camera http://<ip>:8080/video)."
        )
        return 0
    if args.weights is None:
        print("--weights is required unless --list-cameras is specified", file=sys.stderr)
        return 2
    if args.size < 8 or args.size % 8:
        print("--size must be divisible by 8 and at least 8", file=sys.stderr)
        return 2
    if args.redetect_every < 0:
        print("--redetect-every must be zero or positive", file=sys.stderr)
        return 2
    if args.hold_missing < 0:
        print("--hold-missing must be zero or positive", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"YOLO piece weights not found: {args.weights}", file=sys.stderr)
        return 2
    if args.board_weights and not args.board_weights.is_file():
        print(f"YOLO board weights not found: {args.board_weights}", file=sys.stderr)
        return 2

    region: dict[str, int] | None = None
    if args.screen_capture:
        args.screen_board = True
        args.single_window = True
        if args.region:
            try:
                region = parse_region(args.region)
            except ValueError as error:
                print(error, file=sys.stderr)
                return 2
        elif args.select_region:
            try:
                region = select_screen_region(args.monitor)
            except RuntimeError as error:
                print(error, file=sys.stderr)
                return 2
    args.region_is_board = region is not None

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

    if args.screen_capture:
        try:
            camera: LatestFrameCamera | ScreenCaptureSource = ScreenCaptureSource(
                monitor_index=args.monitor,
                region=region,
            )
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
    else:
        camera_source = parse_camera_source(args.camera)
        camera = LatestFrameCamera(camera_source)
        if not camera.is_opened():
            print(f"Could not open camera {args.camera}", file=sys.stderr)
            return 1
    camera.start()

    inference = LiveInference(camera, pieces, board_segmenter, policies, args)
    inference.start()
    if args.screen_capture:
        print("Live screen + background YOLO. s selects region; r recalibrates; q/Esc exits; t toggles turn; f flips.")
    else:
        print("Live camera + background YOLO. r recalibrates; q/Esc exits; t toggles turn; f flips orientation.")
    window_title = "Screen Chess Live" if args.screen_capture else "Camera"
    source_tag = "SCREEN LIVE" if args.screen_capture else "CAM LIVE"
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
            if args.screen_capture:
                if corners is not None and board is not None:
                    preview = board.copy()
                    if results:
                        draw_ar_overlay(
                            preview,
                            board_destination(args.size),
                            args.size,
                            results,
                            inference.turn,
                            inference.flip,
                            visible_fen,
                        )
                else:
                    disp_scale = min(1.0, 960.0 / frame.shape[1], 640.0 / frame.shape[0])
                    preview = cv2.resize(frame, None, fx=disp_scale, fy=disp_scale) if disp_scale < 1.0 else frame.copy()
                    banner = (
                        "Enable macOS Screen Recording for terminal & restart (see terminal)"
                        if getattr(camera, "permission_denied", False)
                        else "Looking for 8x8 board grid (press s to select region)"
                    )
                    _translucent_rect(preview, (12, 10), (min(preview.shape[1] - 12, 820), 48), (18, 20, 24), 0.8)
                    cv2.putText(
                        preview,
                        banner,
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (100, 120, 255),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                preview = frame.copy()
                if corners is None:
                    if args.single_window and board is not None:
                        preview = board.copy()
                    _translucent_rect(preview, (12, 10), (min(preview.shape[1] - 12, 460), 48), (18, 20, 24), 0.8)
                    cv2.putText(preview, "Looking for 8x8 board grid", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (100, 120, 255), 2, cv2.LINE_AA)
                else:
                    cv2.polylines(preview, [corners.astype(np.int32)], True, (255, 220, 40), 3)
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
                if board is not None and not args.single_window:
                    cv2.imshow("Pieces | boxes + confidence", board)

            age = perf_counter() - updated_at if updated_at else 0.0
            geometry = "locked" if corners is not None else "searching"
            base_h = preview.shape[0]
            footer_h = 88
            preview = cv2.copyMakeBorder(preview, 0, footer_h, 0, 0, cv2.BORDER_CONSTANT, value=(22, 24, 28))
            status_colour = (255, 220, 110) if corners is not None else (80, 200, 255)
            cv2.putText(
                preview,
                f"{source_tag} | YOLO {inference_fps:.1f} FPS | {latency_ms:.0f} ms | result {age:.1f}s old | board {geometry}",
                (16, base_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_colour,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"FEN: {visible_fen}",
                (16, base_h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            if visible_move:
                move_colour = (90, 105, 255) if visible_move.startswith("FEN invalid") else (90, 230, 255)
                cv2.putText(
                    preview,
                    visible_move,
                    (16, base_h + 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    move_colour,
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow(window_title, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord("s") and isinstance(camera, ScreenCaptureSource):
                try:
                    new_region = select_screen_region(args.monitor)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                    new_region = None
                camera.set_region(new_region)
                args.region_is_board = new_region is not None
                inference.control(ord("r"))
            if key in (ord("t"), ord("f"), ord("r")):
                inference.control(key)
    finally:
        inference.stop()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
