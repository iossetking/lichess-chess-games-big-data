"""Live & screen chess pipeline: board detection -> rectified YOLO-seg -> FEN -> 5 move models.

Examples:
    # Default: webcam pointed at a phone/monitor screen (auto-loads screen YOLO-seg + 5 move models)
    uv run python scripts/live_chess_fen.py

    # Direct capture of a chessboard on your own screen (no webcam needed!)
    uv run python scripts/live_chess_fen.py --source screen

    # Alternative continuous screen capture with mss (--screen-capture / --select-region)
    uv run python scripts/live_chess_fen.py --screen-capture --select-region

    # Analyze a single screenshot/photo immediately
    uv run python scripts/live_chess_fen.py --image path/to/screenshot.png

    # Physical 3D chessboard with webcam (or iPhone/Android IP stream URL)
    uv run python scripts/live_chess_fen.py --mode physical --camera 1

Hotkeys inside the live window:
    c        Calibrate 4 board corners manually by clicking (a8 -> h8 -> h1 -> a1) or re-pick screen region
    r        Reset & auto-detect board corners
    t        Toggle side to move (White / Black)
    f        Flip board 180° (when viewing from Black's side)
    k        Cycle castling rights (KQkq -> KQ -> kq -> -)
    m        Toggle segmentation mask overlay
    0 / 1-5  Show all move models (0) or focus one model (1..5)
    v        Toggle compact / wide layout (in --source screen)
    s        Save annotated board snapshot + FEN to data/snapshots/ (or re-select region in --screen-capture)
    q / Esc  Quit
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
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
    detect_any_screen_board,
    detect_board_corners,
    detect_green_screen_board,
    detect_screen_board_by_grid,
    detections_to_fen,
    fuse_piece_probabilities,
    piece_description_from_label,
    piece_symbol_from_label,
    repair_king_queen_detections,
    sanitize_fen_castling,
    validate_fen_position,
    validate_ordered_board_corners,
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
CASTLING_PRESETS = ("KQkq", "KQ", "kq", "-")
CALIBRATION_LABELS = ("1/4 Sup-Izq (a8)", "2/4 Sup-Der (h8)", "3/4 Inf-Der (h1)", "4/4 Inf-Izq (a1)")

MACOS_SCREEN_PERMISSION_MSG = (
    "WARNING: macOS Screen Recording permission is not enabled for this terminal/IDE. "
    "Without it, macOS hides app/browser windows and returns a blank gray/wallpaper image.\n"
    "Enable it in: System Settings -> Privacy & Security -> Screen & System Audio Recording, "
    "then fully quit (Cmd+Q) and reopen your terminal/IDE."
)


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


def default_vision_weights(mode: str) -> Path:
    """Pick the best existing local YOLO checkpoint for ``mode``."""
    models_dir = PROJECT_ROOT / "data" / "models"
    if mode == "physical":
        candidates = ("chess-pieces.pt", "chess-pieces-seg.pt", "chess-piece-shapes.pt")
    else:
        candidates = ("screen-chess-pieces-seg.pt", "chess-pieces-seg.pt", "chess-pieces.pt")
    for name in candidates:
        path = models_dir / name
        if path.is_file():
            return path
    return models_dir / candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=("screen", "physical", "auto"),
        default="auto",
        help="Preset: 'screen' (2D board rectified before YOLO-seg), 'physical' (3D board), 'auto' (default)",
    )
    parser.add_argument(
        "--source",
        choices=("camera", "screen"),
        default="camera",
        help="Input source: 'camera' (webcam) or 'screen' (direct capture of your computer monitor)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Analyze a single screenshot or board photo instead of opening a live feed",
    )
    parser.add_argument(
        "--save-output",
        type=Path,
        help="Save the annotated analysis image (with --image, exits without opening a window)",
    )
    parser.add_argument(
        "--screen-region",
        default=None,
        help="Optional screen crop for --source screen as x,y,w,h (e.g. 100,120,800,800)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="YOLO .pt/.onnx weights (default: auto-selects screen-chess-pieces-seg.pt or chess-pieces.pt)",
    )
    parser.add_argument("--board-weights", type=Path, help="Optional YOLO-seg model with a chessboard/board class")
    parser.add_argument(
        "--classifier",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "chess-piece-classifier.pt",
        help="Optional secondary piece crop classifier (only used for 6-class shape detectors)",
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
        help="Capture the live computer screen via mss instead of a camera (implies --screen-board)",
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
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum YOLO confidence (default: 0.25)")
    parser.add_argument(
        "--require-segmentation",
        action="store_true",
        help="Require a YOLO-seg checkpoint with one pixel mask per piece",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size (default: 640)")
    parser.add_argument("--smooth", type=float, default=0.75,
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
                        help="2d: rectified board + centre; 3d: box base (physical pieces); auto: infer from mode/weights")
    parser.add_argument("--screen-board", action="store_true", help="Alias for --mode screen (2D screen board detection)")
    parser.add_argument("--native-frame", action="store_true", help="Run YOLO on raw unrectified frame even in 2D mode")
    parser.add_argument(
        "--redetect-every",
        type=int,
        default=0,
        help="Re-detect the board every N processed frames; 0 locks it until r/c is pressed",
    )
    parser.add_argument("--stable-frames", type=int, default=2, help="Equal frames required before updating move predictions")
    parser.add_argument("--flip", action="store_true", help="Rotate the board 180° when viewing from Black's side")
    parser.add_argument("--turn", choices=("w", "b"), default="w", help="Side to move; press t in the window to toggle")
    parser.add_argument("--castling", default="KQkq", help="Known castling rights (default: KQkq; press k to cycle)")
    parser.add_argument("--en-passant", default="-", help="Known en-passant target (vision cannot infer this)")
    parser.add_argument("--move-model", type=Path, help="Optional single trained policy checkpoint (legacy alias)")
    parser.add_argument(
        "--move-models",
        type=Path,
        nargs="+",
        help="Move checkpoints to compare on the live screen (default: all 5 project models in data/models/)",
    )
    parser.add_argument("--no-move-models", action="store_true", help="Run vision only without loading move models")
    parser.add_argument(
        "--architecture",
        choices=("auto", "scratch", "maia"),
        default="auto",
        help="Deprecated: the checkpoint file now decides its architecture",
    )
    return parser.parse_args()


def _resolve_effective_mode(args: argparse.Namespace) -> str:
    """Resolve ``'screen'`` vs ``'physical'`` from explicit flags and weights."""
    if getattr(args, "mode", "auto") != "auto":
        return args.mode
    if (
        getattr(args, "screen_board", False)
        or getattr(args, "screen_capture", False)
        or getattr(args, "source", "camera") == "screen"
    ):
        return "screen"
    if getattr(args, "scene", "auto") == "3d":
        return "physical"
    weights = getattr(args, "weights", None)
    if weights is not None and "screen" in Path(weights).stem.lower():
        return "screen"
    if getattr(args, "scene", "auto") == "2d":
        return "screen"
    return "screen"


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
        names = getattr(self.model, "names", {}) or {}
        self.has_coloured_classes = bool(names) and all(
            piece_symbol_from_label(str(label)) is not None for label in names.values()
        )
        # Only load the 6-class shape crop classifier when the detector actually emits shape-only labels.
        should_load_classifier = (
            not self.has_coloured_classes
            and classifier_weights is not None
            and classifier_weights.is_file()
        )
        self.classifier = YOLO(classifier_weights) if should_load_classifier else None
        self.classifier_imgsz = (
            int(getattr(self.classifier, "overrides", {}).get("imgsz", 128))
            if self.classifier is not None
            else 128
        )
        self.classifier_weight = classifier_weight
        self.classifier_tta = classifier_tta
        self.device = device
        self.is_segmentation = self.model.task == "segment"

    def warmup(self, imgsz: int, shape: tuple[int, int] = (640, 640)) -> None:
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
        result = self.model.predict(
            image,
            conf=confidence,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            agnostic_nms=True,
            iou=0.5,
            max_det=64,
        )[0]
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
    """Continuously capture camera frames and expose only the newest one."""

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
            return (
                None if self._frame is None else self._frame.copy(),
                self._sequence,
                self._captured_at,
            )

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


def _detect_screen_scale() -> float:
    """Measure pixel-to-logical-point scale on macOS Retina displays (2.0 on Retina, 1.0 otherwise)."""
    if sys.platform != "darwin":
        return 1.0
    probe = _grab_screen_bgr((0, 0, 100, 100))
    if probe is None or probe.shape[1] <= 0:
        return 1.0
    scale = round(probe.shape[1] / 100.0, 2)
    return scale if scale > 0 else 1.0


def _grab_screen_bgr(region: tuple[int, int, int, int] | None = None) -> np.ndarray | None:
    """Capture the computer screen (or ``(x, y, w, h)`` logical region) as a BGR NumPy array."""
    if sys.platform == "darwin":
        # Direct screencapture to JPEG is significantly faster than Pillow's PNG path
        # and avoids writing multi-MB PNGs every frame on Retina displays.
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            cmd = ["screencapture", "-x", "-t", "jpg"]
            if region is not None:
                cmd.extend(["-R", f"{region[0]},{region[1]},{region[2]},{region[3]}"])
            cmd.append(tmp.name)
            proc = subprocess.run(cmd, check=False, capture_output=True)
            if proc.returncode == 0:
                img = cv2.imread(tmp.name)
                if img is not None:
                    return img
    try:
        from PIL import ImageGrab

        bbox = None if region is None else (region[0], region[1], region[0] + region[2], region[1] + region[3])
        shot = ImageGrab.grab(bbox=bbox)
        rgb = np.asarray(shot.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:  # noqa: BLE001
        return None


def corners_to_logical_region(
    corners: np.ndarray,
    screen_scale: float = 1.0,
) -> tuple[int, int, int, int]:
    """Convert pixel corners ``(4, 2)`` on a full-screen grab to logical ``(x, y, w, h)``."""
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
    scale = max(float(screen_scale), 1e-3)
    lx = max(0, round(x1 / scale))
    ly = max(0, round(y1 / scale))
    lw = max(32, round((x2 - x1) / scale))
    lh = max(32, round((y2 - y1) / scale))
    return lx, ly, lw, lh


def _close_window(name: str) -> None:
    """Close an OpenCV window and pump macOS Cocoa events so it actually disappears."""
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass
    for _ in range(10):
        cv2.waitKey(1)
    sleep(0.25)


def compute_assistant_window_pos(
    board_region: tuple[int, int, int, int],
    screen_logical_size: tuple[int, int],
    window_size: tuple[int, int],
) -> tuple[int, int]:
    """Choose ``(x, y)`` for the assistant window so it does NOT overlap the chessboard."""
    bx, by, bw, _bh = board_region
    sw, sh = screen_logical_size
    ww, wh = window_size
    pad = 16
    # Prefer right of the board, then left of the board, else snap to the farthest screen edge.
    if bx + bw + pad + ww <= sw:
        wx = bx + bw + pad
    elif bx - pad - ww >= 0:
        wx = bx - pad - ww
    else:
        right_space = sw - (bx + bw)
        left_space = bx
        wx = max(0, sw - ww - 8) if right_space >= left_space else 8
    wy = max(36, min(by, max(36, sh - wh - 24)))
    return int(wx), int(wy)


def select_screen_board_region(
    screen_scale: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    """Interactive one-drag board selector for ``--source screen``.

    Returns ``((x, y, w, h), (screen_logical_w, screen_logical_h))`` or ``None``.
    """
    picker_win = "Paso 1: Selecciona el tablero (Enter confirmar | Arrastra cuadro | r refrescar)"
    cv2.namedWindow(picker_win, cv2.WINDOW_NORMAL)

    def grab_fresh() -> tuple[np.ndarray | None, float, tuple[int, int, int, int] | None]:
        full = _grab_screen_bgr(None)
        if full is None:
            return None, 1.0, None
        fh, fw = full.shape[:2]
        disp_scale = min(1.0, 1120.0 / max(fw, 1), 660.0 / max(fh, 1))
        auto_corners = detect_any_screen_board(full)
        auto_box = None
        if auto_corners is not None:
            x1, y1 = int(auto_corners[:, 0].min()), int(auto_corners[:, 1].min())
            x2, y2 = int(auto_corners[:, 0].max()), int(auto_corners[:, 1].max())
            auto_box = (x1, y1, x2, y2)
        return full, disp_scale, auto_box

    full, disp_scale, selected_px = grab_fresh()
    if full is None:
        _close_window(picker_win)
        return None

    dragging = False
    drag_start = (0, 0)
    drag_now = (0, 0)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal dragging, drag_start, drag_now, selected_px
        px = round(x / disp_scale)
        py = round(y / disp_scale)
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_start = (px, py)
            drag_now = (px, py)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            drag_now = (px, py)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            drag_now = (px, py)
            x1, x2 = sorted((drag_start[0], drag_now[0]))
            y1, y2 = sorted((drag_start[1], drag_now[1]))
            if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                selected_px = (x1, y1, x2, y2)

    cv2.setMouseCallback(picker_win, on_mouse)
    cv2.moveWindow(picker_win, 40, 40)
    try:
        while True:
            dh, dw = round(full.shape[0] * disp_scale), round(full.shape[1] * disp_scale)
            view = cv2.resize(full, (dw, dh), interpolation=cv2.INTER_AREA)

            active_box = None
            if dragging:
                x1, x2 = sorted((drag_start[0], drag_now[0]))
                y1, y2 = sorted((drag_start[1], drag_now[1]))
                active_box = (x1, y1, x2, y2)
            elif selected_px is not None:
                active_box = selected_px

            if active_box is not None:
                dx1, dy1 = round(active_box[0] * disp_scale), round(active_box[1] * disp_scale)
                dx2, dy2 = round(active_box[2] * disp_scale), round(active_box[3] * disp_scale)
                dimmed = (view.astype(np.float32) * 0.45).astype(np.uint8)
                dimmed[max(0, dy1):min(dh, dy2), max(0, dx1):min(dw, dx2)] = view[
                    max(0, dy1):min(dh, dy2), max(0, dx1):min(dw, dx2)
                ]
                view = dimmed
                colour = (0, 255, 255) if dragging else (60, 255, 110)
                cv2.rectangle(view, (dx1, dy1), (dx2, dy2), colour, 3)

            _translucent_rect(view, (0, 0), (dw, 52), (20, 24, 30), 0.85)
            if selected_px is not None and not dragging:
                msg = "Tablero listo (verde) -> Pulsa [ENTER] para empezar, o ARRASTRA otro cuadro  |  [r] refrescar  [Esc] cancelar"
                col = (90, 255, 150)
            else:
                msg = "ARRASTRA con el mouse un rectangulo sobre el tablero (de a8 a h1) y pulsa [ENTER]  |  [r] refrescar"
                col = (255, 230, 110)
            cv2.putText(view, msg, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, col, 1, cv2.LINE_AA)
            cv2.imshow(picker_win, view)
            cv2.resizeWindow(picker_win, dw, dh)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 3, 32) and selected_px is not None:  # Return, LF, Numpad Enter, or Space
                x1, y1, x2, y2 = selected_px
                corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
                logical_screen = (
                    max(640, round(full.shape[1] / max(screen_scale, 1e-3))),
                    max(480, round(full.shape[0] / max(screen_scale, 1e-3))),
                )
                return corners_to_logical_region(corners, screen_scale), logical_screen
            if key == ord("r"):
                _close_window(picker_win)
                cv2.namedWindow(picker_win, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(picker_win, on_mouse)
                cv2.moveWindow(picker_win, 40, 40)
                fresh, fresh_scale, auto_box = grab_fresh()
                if fresh is not None:
                    full, disp_scale, selected_px = fresh, fresh_scale, auto_box
            if key in (27, ord("q")):
                return None
    finally:
        _close_window(picker_win)


class LatestScreenFrame:
    """Continuously capture a fixed screen region (or monitor) and expose the latest BGR frame."""

    def __init__(self, region: tuple[int, int, int, int] | None = None, max_fps: float = 6.0) -> None:
        self.region = region
        self._interval = 1.0 / max(1.0, max_fps)
        self._lock = Lock()
        self._frame: np.ndarray | None = _grab_screen_bgr(region)
        self._sequence = 1 if self._frame is not None else 0
        self._captured_at = perf_counter() if self._frame is not None else 0.0
        self._stopped = Event()
        self._thread: Thread | None = None

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        with self._lock:
            self.region = region

    def is_opened(self) -> bool:
        return self._frame is not None

    def start(self) -> None:
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stopped.is_set():
            started = perf_counter()
            with self._lock:
                region = self.region
            frame = _grab_screen_bgr(region)
            if frame is not None:
                with self._lock:
                    self._frame = frame
                    self._sequence += 1
                    self._captured_at = perf_counter()
            remaining = self._interval - (perf_counter() - started)
            if remaining > 0:
                self._stopped.wait(remaining)

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


def predict_moves_for_fen(
    fen: str,
    policies: list[PolicyStyle],
    *,
    top_k: int = 3,
) -> tuple[list[dict[str, Any]], str, bool, str]:
    """Validate ``fen`` (auto-pruning impossible castling flags) and run every loaded policy model.

    Returns ``(results, summary_text, is_valid, status_reason)``.
    Each entry in ``results`` includes ``label``, ``colour``, ``uci``, ``san``,
    ``probability``, and ``top_moves`` (list of ``(uci, san, probability)``).
    """
    fen = sanitize_fen_castling(fen)
    is_valid, reason = validate_fen_position(fen)
    if not is_valid:
        return [], f"FEN invalido: {reason}", False, reason
    if not policies:
        return [], "Posicion valida (sin modelos de jugadas cargados)", True, "OK"

    import chess

    from lichess_pipeline.model import predict_top_moves

    position = chess.Board(fen)
    if not any(position.legal_moves):
        return [], "Posicion terminal (mate o ahogado)", True, "Sin jugadas legales"

    results: list[dict[str, Any]] = []
    for style in policies:
        moves = predict_top_moves(style.model, fen, top_k=top_k)
        if not moves:
            continue
        formatted: list[tuple[str, str, float]] = []
        for uci, probability in moves:
            try:
                san = position.san(chess.Move.from_uci(uci))
            except (ValueError, AssertionError):
                san = uci
            formatted.append((uci, san, float(probability)))
        best_uci, best_san, best_prob = formatted[0]
        results.append(
            {
                "label": style.label,
                "colour": style.colour,
                "uci": best_uci,
                "san": best_san,
                "probability": best_prob,
                "top_moves": formatted,
            }
        )
    message = ", ".join(f"{item['label']}: {item['san']} {item['probability']:.0%}" for item in results)
    if not message:
        message = "Sin jugada legal"
    return results, message, True, "OK"


class LiveInference:
    """Run board localization, rectified YOLO-seg, and 5-model move inference."""

    def __init__(
        self,
        camera: Any,
        pieces: YoloPieceDetector,
        board_segmenter: YoloBoardSegmenter | None,
        policies: Any,
        args: argparse.Namespace,
    ) -> None:
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
        self.rectified_detections: list[PieceDetection] = []
        self.fen = "Esperando tablero... (pulsa [c] para calibrar 4 esquinas)"
        self.fen_status = "Buscando tablero"
        self.fen_valid = False
        self.move = ""
        self.results: list[dict[str, Any]] = []
        self.inference_fps = 0.0
        self.last_result_at = 0.0
        self.latency_ms = 0.0
        self.turn = getattr(args, "turn", "w")
        self.flip = getattr(args, "flip", False)
        self.castling = getattr(args, "castling", "-")
        self.show_masks = True
        self.active_model_idx: int | None = None  # None = show all models
        self.manual_locked = False
        self.full_frame_is_board = bool(getattr(args, "full_frame_is_board", False))
        self.generation = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def set_manual_corners(self, corners: np.ndarray) -> None:
        """Lock user-clicked board corners (TL, TR, BR, BL) and reset stabilization."""
        validated = validate_ordered_board_corners(corners)
        with self._lock:
            self.corners = validated
            self.manual_locked = True
            self.generation += 1

    def control(self, key: int) -> None:
        with self._lock:
            if key == ord("r"):
                self.corners = None
                self.manual_locked = False
                self.board_image = None
                self.fen = "Buscando tablero..."
                self.fen_status = "Buscando tablero"
                self.fen_valid = False
                self.move, self.results = "", []
            elif key == ord("f"):
                self.flip = not self.flip
            elif key == ord("t"):
                self.turn = "b" if self.turn == "w" else "w"
            elif key == ord("k"):
                try:
                    idx = CASTLING_PRESETS.index(self.castling)
                    self.castling = CASTLING_PRESETS[(idx + 1) % len(CASTLING_PRESETS)]
                except ValueError:
                    self.castling = CASTLING_PRESETS[0]
            elif key == ord("m"):
                self.show_masks = not self.show_masks
                return  # display-only toggle; keep current FEN & predictions
            elif key == ord("0"):
                self.active_model_idx = None
                return
            elif ord("1") <= key <= ord("9"):
                chosen = key - ord("1")
                if chosen < len(self.policies):
                    self.active_model_idx = None if self.active_model_idx == chosen else chosen
                return
            self.generation += 1

    def state(self) -> dict[str, Any]:
        """Thread-safe snapshot of all live inference state for rendering."""
        with self._lock:
            return {
                "corners": None if self.corners is None else self.corners.copy(),
                "board_image": None if self.board_image is None else self.board_image.copy(),
                "fen": self.fen,
                "fen_status": self.fen_status,
                "fen_valid": self.fen_valid,
                "move": self.move,
                "inference_fps": self.inference_fps,
                "last_result_at": self.last_result_at,
                "latency_ms": self.latency_ms,
                "results": [dict(result) for result in self.results],
                "turn": self.turn,
                "flip": self.flip,
                "castling": self.castling,
                "show_masks": self.show_masks,
                "active_model_idx": self.active_model_idx,
                "manual_locked": self.manual_locked,
            }

    def snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, str, str, float, float, float, list[dict[str, Any]]]:
        """Backwards-compatible tuple snapshot."""
        snap = self.state()
        return (
            snap["corners"],
            snap["board_image"],
            snap["fen"],
            snap["move"],
            snap["inference_fps"],
            snap["last_result_at"],
            snap["latency_ms"],
            snap["results"],
        )

    def _should_rectify_before_detect(self, scene: str) -> bool:
        if getattr(self.args, "native_frame", False):
            return False
        if scene == "3d":
            return False
        if (
            getattr(self.args, "screen_board", False)
            or getattr(self.args, "screen_capture", False)
            or getattr(self.args, "mode", "auto") == "screen"
            or getattr(self.args, "source", "camera") == "screen"
        ):
            return True
        return scene == "2d" and getattr(self.pieces, "is_segmentation", False)

    def _detect_board(self, frame: np.ndarray, scene: str) -> tuple[np.ndarray | None, str]:
        auto = getattr(self.args, "scene", "auto") == "auto"
        region_is_board = bool(getattr(self.args, "region_is_board", False))
        screen_hint = (
            getattr(self.args, "screen_board", False)
            or getattr(self.args, "screen_capture", False)
            or getattr(self.args, "mode", "auto") == "screen"
            or getattr(self.args, "source", "camera") == "screen"
        )
        found: np.ndarray | None = None
        if screen_hint or auto:
            found = detect_green_screen_board(frame)
            if found is None:
                found = detect_screen_board_by_grid(frame)
            if found is not None and auto:
                scene = "2d"
        if found is None and not region_is_board:
            found = detect_board_corners(frame)
            if found is not None and auto and not screen_hint:
                scene = "3d"
        if found is None and region_is_board:
            fh, fw = frame.shape[:2]
            found = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype=np.float32)
        if found is None and self.board_segmenter is not None:
            found = self.board_segmenter.detect(frame, self.args.confidence, self.args.imgsz)
        return found, scene

    def _run(self) -> None:
        size = self.args.size
        destination = board_destination(size)
        stabilizer = FenStabilizer(self.args.stable_frames)
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
        arg_scene = getattr(self.args, "scene", "auto")
        if arg_scene != "auto":
            scene = arg_scene
        elif (
            getattr(self.args, "mode", "auto") == "screen"
            or getattr(self.args, "screen_board", False)
            or getattr(self.args, "screen_capture", False)
            or getattr(self.args, "source", "camera") == "screen"
        ):
            scene = "2d"
        else:
            scene = "3d"

        while not self._stop.is_set():
            frame, sequence, captured_at = self.camera.sample()
            if frame is None or sequence == last_sequence:
                sleep(0.01)
                continue
            last_sequence = sequence
            started = perf_counter()

            with self._lock:
                corners = None if self.corners is None else self.corners.copy()
                generation = self.generation
                turn = self.turn
                flip = self.flip
                castling = self.castling
                show_masks = self.show_masks
                manual_locked = self.manual_locked
                active_model_idx = self.active_model_idx
                prev_results = [dict(r) for r in self.results]
                prev_fen = self.fen
                prev_status = self.fen_status
                prev_valid = self.fen_valid

            if self.full_frame_is_board:
                fh, fw = frame.shape[:2]
                corners = np.array([[0, 0], [fw - 1, 0], [fw - 1, fh - 1], [0, fh - 1]], dtype=np.float32)
            else:
                redetect = (
                    not manual_locked
                    and (corners is None or (self.args.redetect_every > 0 and processed % self.args.redetect_every == 0))
                )
                if redetect and started >= next_board_search:
                    next_board_search = started + 0.8
                    found, scene = self._detect_board(frame, scene)
                    if found is not None:
                        corners = found

            use_rectified = corners is not None and self._should_rectify_before_detect(scene)
            if use_rectified != in_rectified_space:
                in_rectified_space = use_rectified
                smoother = make_smoother()

            mapped: list[PieceDetection] = []
            rectified_view: np.ndarray | None = None

            if use_rectified and corners is not None:
                transform = cv2.getPerspectiveTransform(corners, destination)
                warped = cv2.warpPerspective(frame, transform, (size, size))
                if flip:
                    warped = cv2.rotate(warped, cv2.ROTATE_180)
                detections = self.pieces.detect(warped, self.args.confidence, self.args.imgsz)
                if smoother is not None:
                    detections = smoother.update(detections)
                detections = repair_king_queen_detections(detections)
                mapped = detections
                rectified_view = warped
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
                        anchor = np.array(
                            [[[(x1 + x2) / 2, y2 if scene == "3d" else (y1 + y2) / 2]]],
                            dtype=np.float32,
                        )
                        x, y = cv2.perspectiveTransform(anchor, transform)[0, 0]
                        if flip:
                            x, y = size - 1 - x, size - 1 - y
                        mapped.append(
                            PieceDetection(detection.label, detection.confidence, (float(x), float(y), float(x), float(y)))
                        )

            can_publish = (
                corners is not None
                and bool(detections)
                and all(piece_symbol_from_label(d.label) is not None for d in detections)
            )
            fen = detections_to_fen(
                mapped,
                size,
                turn=turn,
                castling=castling,
                en_passant=self.args.en_passant,
            )
            published = stabilizer.update(fen) if can_publish else None

            results: list[dict[str, Any]] = prev_results
            message = ""
            is_valid = prev_valid
            status_reason = prev_status
            if published is not None:
                published = sanitize_fen_castling(published)
                results, message, is_valid, status_reason = predict_moves_for_fen(published, self.policies, top_k=3)

            if rectified_view is not None:
                display_fen = published if published is not None else (sanitize_fen_castling(fen) if can_publish else prev_fen)
                display_results = results if (published is not None or prev_valid) else []
                filtered_results = _filter_results(display_results, active_model_idx)
                annotated = draw_board_analysis(
                    rectified_view,
                    detections,
                    filtered_results,
                    all_results=display_results,
                    fen=display_fen,
                    fen_valid=is_valid if published is not None else prev_valid,
                    fen_status=status_reason if published is not None else prev_status,
                    turn=turn,
                    castling=castling,
                    flip=flip,
                    show_masks=show_masks,
                    active_model_idx=active_model_idx,
                    manual_locked=manual_locked,
                )

            elapsed = max(perf_counter() - started, 1e-6)
            with self._lock:
                if generation != self.generation:
                    stabilizer = FenStabilizer(self.args.stable_frames)
                    smoother = make_smoother()
                    continue
                self.corners, self.board_image = corners, annotated
                self.rectified_detections = list(detections)
                self.latency_ms = (perf_counter() - captured_at) * 1000
                if not can_publish:
                    if corners is None:
                        self.fen = "Esperando tablero (pulsa [c] para marcar 4 esquinas)"
                        self.fen_status = "Tablero no detectado"
                    else:
                        self.fen = "FEN necesita etiquetas con color (12 clases)"
                        self.fen_status = "Sin piezas detectadas"
                    self.fen_valid = False
                    self.move, self.results = "", []
                    stabilizer = FenStabilizer(self.args.stable_frames)
                if published is not None:
                    self.fen = published
                    self.fen_valid = is_valid
                    self.fen_status = status_reason
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


def _filter_results(results: list[dict[str, Any]], active_model_idx: int | None) -> list[dict[str, Any]]:
    if active_model_idx is None or not (0 <= active_model_idx < len(results)):
        return results
    return [results[active_model_idx]]


def _draw_arrows_on_rectified(
    canvas: np.ndarray,
    results: list[dict[str, Any]],
    board_size: int,
    flip: bool,
) -> None:
    """Draw SeeMate-style move arrows directly on a ``board_size x board_size`` board."""
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(results):
        groups.setdefault(str(item["uci"]), []).append(index)

    anchors: dict[int, tuple[int, int]] = {}
    for indices in groups.values():
        for order, index in enumerate(indices):
            item = results[index]
            uci = str(item["uci"])
            points = [(round(px), round(py)) for px, py in _arrow_polyline(uci[:2], uci[2:4], board_size, flip)]
            offset = 0.0 if len(indices) == 1 else (order - (len(indices) - 1) / 2) * 10
            points = _shift_perpendicular(points, offset)
            segments = list(itertools.pairwise(points))
            for start, end in segments[:-1]:
                cv2.line(canvas, start, end, item["colour"], 4, cv2.LINE_AA)
            cv2.arrowedLine(canvas, segments[-1][0], segments[-1][1], item["colour"], 4, cv2.LINE_AA, tipLength=0.22)
            anchors[index] = (points[-1][0], points[-1][1] - 8 - order * 22)

    for index, item in enumerate(results):
        ax, ay = anchors[index]
        text = f"{item['san']} {item['probability']:.0%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        x = min(max(6, ax + 6), max(6, board_size - tw - 12))
        y = min(max(th + 8, ay), max(th + 8, board_size - baseline - 8))
        _translucent_rect(canvas, (x - 4, y - th - 4), (x + tw + 4, y + baseline), item["colour"], 0.88)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)


def draw_board_analysis(
    board_bgr: np.ndarray,
    detections: list[PieceDetection],
    visible_results: list[dict[str, Any]],
    *,
    all_results: list[dict[str, Any]],
    fen: str,
    fen_valid: bool,
    fen_status: str,
    turn: str,
    castling: str,
    flip: bool,
    show_masks: bool = True,
    active_model_idx: int | None = None,
    manual_locked: bool = False,
) -> np.ndarray:
    """Compose the rectified board with colour-coded masks, arrows, and a live telemetry sidebar."""
    board_size = board_bgr.shape[0]
    board_vis = board_bgr.copy()
    side = board_size // 8

    # 1. Translucent segmentation masks (Cyan for White pieces, Coral/Orange for Black pieces).
    if show_masks:
        mask_layer = board_vis.copy()
        for det in detections:
            sym = piece_symbol_from_label(det.label)
            is_white = sym is not None and sym.isupper()
            colour = (255, 220, 80) if is_white else (60, 120, 255)
            if det.mask is not None and len(det.mask) >= 3:
                poly = det.mask.reshape(-1, 1, 2)
                cv2.fillPoly(mask_layer, [poly], colour)
        cv2.addWeighted(mask_layer, 0.30, board_vis, 0.70, 0, board_vis)

    # 2. Subtle 8x8 square grid & algebraic coordinates.
    for i in range(1, 8):
        pos = i * side
        cv2.line(board_vis, (pos, 0), (pos, board_size), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(board_vis, (0, pos), (board_size, pos), (255, 255, 255), 1, cv2.LINE_AA)
    ranks = "12345678" if flip else "87654321"
    files = "hgfedcba" if flip else "abcdefgh"
    for idx in range(8):
        cv2.putText(board_vis, ranks[idx], (4, idx * side + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(board_vis, ranks[idx], (4, idx * side + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(board_vis, files[idx], (idx * side + side - 15, board_size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(board_vis, files[idx], (idx * side + side - 15, board_size - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    # 3. Bounding box + mask contour + readable Spanish piece name & percentage on every piece.
    for det in detections:
        sym = piece_symbol_from_label(det.label)
        is_white = sym is not None and sym.isupper()
        colour = (255, 230, 90) if is_white else (50, 130, 255)
        x1, y1, x2, y2 = (max(0, min(board_size - 1, round(v))) for v in det.xyxy)
        if show_masks and det.mask is not None and len(det.mask) >= 3:
            cv2.polylines(board_vis, [det.mask.reshape(-1, 1, 2)], True, colour, 1, cv2.LINE_AA)
        cv2.rectangle(board_vis, (x1, y1), (x2, y2), colour, 2)
        desc = piece_description_from_label(det.label) or det.label
        desc = desc.removesuffix(" blanco").removesuffix(" negro").replace("ó", "o")
        tag = f"{desc} {det.confidence:.0%}"
        (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx = max(2, min(x1, board_size - tw - 6))
        ly = max(th + 5, min(y1 + th + 3, board_size - bl - 2))
        cv2.rectangle(board_vis, (lx - 2, ly - th - 4), (lx + tw + 3, ly + bl), colour, -1)
        cv2.putText(board_vis, tag, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (10, 10, 10), 1, cv2.LINE_AA)

    # 4. Move arrows projected directly on the rectified board.
    if visible_results:
        _draw_arrows_on_rectified(board_vis, visible_results, board_size, flip=flip)

    # 5. Clean right-hand telemetry sidebar (never collides with the board!).
    panel_w = 370
    panel = np.full((board_size, panel_w, 3), (26, 28, 32), dtype=np.uint8)
    y = 32
    cv2.putText(panel, "AJEDREZ EN VIVO · VISION + IA", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
    y += 14
    cv2.line(panel, (16, y), (panel_w - 16, y), (70, 75, 85), 1)
    y += 28

    # Game context
    turn_txt = "BLANCAS" if turn == "w" else "NEGRAS"
    turn_col = (245, 245, 245) if turn == "w" else (120, 180, 255)
    cv2.putText(panel, f"Turno: {turn_txt} [t]   Enroque: {castling} [k]", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, turn_col, 1, cv2.LINE_AA)
    y += 24
    orient = "Negras abajo" if flip else "Blancas abajo"
    lock_txt = "Manual [c]" if manual_locked else "Auto [r]"
    cv2.putText(panel, f"Vista: {orient} [f]   Tablero: {lock_txt}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (195, 200, 210), 1, cv2.LINE_AA)
    y += 26

    # Piece counts
    board_part = fen.split(" ")[0] if fen and "/" in fen else ""
    white_cnt = sum(1 for ch in board_part if ch.isupper())
    black_cnt = sum(1 for ch in board_part if ch.islower())
    cv2.putText(panel, f"Piezas detectadas: {white_cnt} B  /  {black_cnt} N  ({len(detections)} total)", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (215, 220, 225), 1, cv2.LINE_AA)
    y += 26

    # FEN validation badge
    status_bg = (40, 130, 60) if fen_valid else (45, 50, 165)
    status_txt = "POSICION VALIDA" if fen_valid else f"REVISAR: {fen_status}"
    cv2.rectangle(panel, (14, y - 16), (panel_w - 14, y + 10), status_bg, -1)
    cv2.putText(panel, status_txt[:42], (20, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y += 30

    # Consensus banner across the loaded move models
    if all_results:
        votes = Counter(item["san"] for item in all_results)
        top_move, top_count = votes.most_common(1)[0]
        cv2.rectangle(panel, (14, y - 16), (panel_w - 14, y + 16), (52, 62, 78), -1)
        cv2.putText(
            panel,
            f"MEJOR JUGADA ({top_count}/{len(all_results)}): {top_move}",
            (22, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (90, 255, 160),
            2,
            cv2.LINE_AA,
        )
        y += 40
    else:
        y += 8

    # Per-model predictions table
    cv2.putText(panel, "MODELOS DE JUGADA  ([0] todos, [1-5] uno)", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (165, 175, 190), 1, cv2.LINE_AA)
    y += 22
    for idx, item in enumerate(all_results):
        selected = active_model_idx is None or active_model_idx == idx
        row_bg = (44, 48, 56) if selected else (30, 32, 36)
        cv2.rectangle(panel, (14, y - 16), (panel_w - 14, y + 32), row_bg, -1)
        cv2.circle(panel, (28, y - 2), 6, item["colour"], -1, cv2.LINE_AA)
        txt_col = (245, 245, 245) if selected else (120, 125, 135)
        cv2.putText(
            panel,
            f"[{idx + 1}] {item['label']:<11} {item['san']:<6} {item['probability']:>4.0%}",
            (42, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            txt_col,
            1,
            cv2.LINE_AA,
        )
        top_moves = item.get("top_moves", [])
        if len(top_moves) > 1:
            alts = "  |  ".join(f"{san} {prob:.0%}" for _, san, prob in top_moves[1:3])
            cv2.putText(panel, f"alt: {alts}", (42, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 170, 180), 1, cv2.LINE_AA)
        y += 54

    # Compact FEN string
    y = max(y + 10, board_size - 165)
    cv2.line(panel, (16, y - 12), (panel_w - 16, y - 12), (70, 75, 85), 1)
    cv2.putText(panel, "FEN:", (16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
    fen_clean = fen.replace("ó", "o")
    for line_idx, chunk in enumerate((fen_clean[:40], fen_clean[40:80])):
        if chunk:
            cv2.putText(panel, chunk, (16, y + 22 + line_idx * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (210, 215, 220), 1, cv2.LINE_AA)

    # Hotkey cheat-sheet at bottom
    help_y = board_size - 78
    cv2.line(panel, (16, help_y - 10), (panel_w - 16, help_y - 10), (70, 75, 85), 1)
    hotkeys = (
        "[c] Marcar 4 esquinas   [r] Auto-tablero",
        "[t] Turno   [f] Girar   [k] Enroque",
        "[m] Mascaras  [0/1-5] Modelos  [s] Guardar",
        "[q / Esc] Salir",
    )
    for row, line in enumerate(hotkeys):
        cv2.putText(panel, line, (16, help_y + 8 + row * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (175, 185, 200), 1, cv2.LINE_AA)

    return np.hstack([board_vis, panel])


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
    if getattr(args, "no_move_models", False):
        return []
    if args.move_models:
        return list(args.move_models)
    if args.move_model:
        return [args.move_model]
    models_dir = PROJECT_ROOT / "data" / "models"
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
    # Place below the top status bar so text never collides.
    top_left = (10, 48)
    panel_height = 24 * len(hud_lines) + 8
    _translucent_rect(image, top_left, (430, top_left[1] + panel_height), (24, 24, 24), 0.65)
    for row, line in enumerate(hud_lines):
        cv2.putText(image, line, (18, top_left[1] + 22 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    if results:
        legend_top = height - 26 * len(results) - 44
        _translucent_rect(image, (10, legend_top - 6), (360, height - 38), (24, 24, 24), 0.65)
        for row, item in enumerate(results):
            y = legend_top + row * 26 + 14
            cv2.circle(image, (22, y - 5), 6, item["colour"], -1, cv2.LINE_AA)
            text = f"{item['label']}: {item['san']} {item['probability']:.0%}"
            cv2.putText(image, text, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)


def _parse_region(spec: str | None) -> tuple[int, int, int, int] | None:
    if not spec:
        return None
    parts = [int(v.strip()) for v in spec.split(",")]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        raise ValueError("--screen-region must be x,y,w,h with positive width and height")
    return parts[0], parts[1], parts[2], parts[3]


def _run_single_image(
    args: argparse.Namespace,
    pieces: YoloPieceDetector,
    board_segmenter: YoloBoardSegmenter | None,
    policies: list[PolicyStyle],
) -> int:
    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 2

    corners = detect_any_screen_board(frame)
    if corners is None:
        corners = detect_board_corners(frame)
    if corners is None and board_segmenter is not None:
        corners = board_segmenter.detect(frame, args.confidence, args.imgsz)
    if corners is None:
        h, w = frame.shape[:2]
        if abs(h - w) <= max(h, w) * 0.08:
            # Already a cropped square board image (e.g., screen-template.png).
            corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        else:
            print("No se pudo localizar el tablero automaticamente en la imagen.", file=sys.stderr)
            return 1

    transform = cv2.getPerspectiveTransform(corners, board_destination(args.size))
    warped = cv2.warpPerspective(frame, transform, (args.size, args.size))
    if args.flip:
        warped = cv2.rotate(warped, cv2.ROTATE_180)
    detections = repair_king_queen_detections(pieces.detect(warped, args.confidence, args.imgsz))
    fen = sanitize_fen_castling(
        detections_to_fen(
            detections,
            args.size,
            turn=args.turn,
            castling=args.castling,
            en_passant=args.en_passant,
        )
    )
    results, _message, is_valid, reason = predict_moves_for_fen(fen, policies, top_k=3)

    print(f"Piezas detectadas: {len(detections)}")
    print(f"FEN: {fen}")
    print(f"Estado: {'VALIDO' if is_valid else f'INVALIDO ({reason})'}")
    for item in results:
        alts = ", ".join(f"{san} ({prob:.1%})" for _, san, prob in item["top_moves"])
        print(f"  - {item['label']:<12}: {alts}")

    vis = draw_board_analysis(
        warped,
        detections,
        results,
        all_results=results,
        fen=fen,
        fen_valid=is_valid,
        fen_status=reason,
        turn=args.turn,
        castling=args.castling,
        flip=args.flip,
        show_masks=True,
    )
    if args.save_output:
        args.save_output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_output), vis)
        print(f"Imagen guardada en: {args.save_output}")
        return 0

    cv2.imshow("Chess Analysis (press any key to close)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def _run_mss_screen_capture(
    args: argparse.Namespace,
    pieces: YoloPieceDetector,
    board_segmenter: YoloBoardSegmenter | None,
    policies: list[PolicyStyle],
) -> int:
    region_dict: dict[str, int] | None = None
    if args.region:
        try:
            region_dict = parse_region(args.region)
        except ValueError as error:
            print(error, file=sys.stderr)
            return 2
    elif args.select_region:
        try:
            region_dict = select_screen_region(args.monitor)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 2

    args.screen_board = True
    args.single_window = True
    args.region_is_board = region_dict is not None
    try:
        source_feed = ScreenCaptureSource(monitor_index=args.monitor, region=region_dict)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    source_feed.start()
    inference = LiveInference(source_feed, pieces, board_segmenter, policies, args)
    inference.start()
    print(
        "Live screen (mss) + background YOLO. [s] select region | [r] recalibrate | "
        "[t] turn | [f] flip | [k] castling | [m] masks | [0/1-5] models | [q/Esc] quit."
    )
    window_title = "Screen Chess Live"
    try:
        while True:
            frame = source_feed.latest()
            if frame is None:
                cv2.waitKey(1)
                continue
            snap = inference.state()
            corners = snap["corners"]
            board_vis = snap["board_image"]
            if corners is not None and board_vis is not None:
                preview = board_vis.copy()
            else:
                disp_scale = min(1.0, 960.0 / frame.shape[1], 640.0 / frame.shape[0])
                preview = cv2.resize(frame, None, fx=disp_scale, fy=disp_scale) if disp_scale < 1.0 else frame.copy()
                banner = (
                    "Enable macOS Screen Recording for terminal & restart (see terminal)"
                    if getattr(source_feed, "permission_denied", False)
                    else "Looking for 8x8 board grid (press s to select region)"
                )
                _translucent_rect(preview, (12, 10), (min(preview.shape[1] - 12, 820), 48), (18, 20, 24), 0.8)
                cv2.putText(preview, banner, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (100, 120, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_title, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord("s"):
                try:
                    new_region = select_screen_region(args.monitor)
                except RuntimeError as error:
                    print(error, file=sys.stderr)
                    new_region = None
                source_feed.set_region(new_region)
                args.region_is_board = new_region is not None
                inference.control(ord("r"))
                continue
            if key in (ord("t"), ord("f"), ord("r"), ord("k"), ord("m"), ord("0")) or (ord("1") <= key <= ord("9")):
                inference.control(key)
    finally:
        inference.stop()
        source_feed.release()
        cv2.destroyAllWindows()


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
    if args.size < 8 or args.size % 8:
        print("--size must be divisible by 8 and at least 8", file=sys.stderr)
        return 2
    if args.redetect_every < 0:
        print("--redetect-every must be zero or positive", file=sys.stderr)
        return 2
    if args.hold_missing < 0:
        print("--hold-missing must be zero or positive", file=sys.stderr)
        return 2

    effective_mode = _resolve_effective_mode(args)
    args.mode = effective_mode
    if args.weights is None:
        args.weights = default_vision_weights(effective_mode)
    if not args.weights.is_file():
        print(f"YOLO piece weights not found: {args.weights}", file=sys.stderr)
        return 2
    if args.board_weights and not args.board_weights.is_file():
        print(f"YOLO board weights not found: {args.board_weights}", file=sys.stderr)
        return 2

    try:
        region = _parse_region(args.screen_region)
        classifier_path = None if args.no_classifier else args.classifier
        print(f"Loading vision model ({effective_mode} mode): {args.weights}")
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
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    if not policies and not args.no_move_models:
        print(
            "Sin checkpoints de jugadas: solo se mostrara la vision. Ejecuta `git lfs pull` o pasalos con --move-models.",
            file=sys.stderr,
        )

    if args.image is not None:
        return _run_single_image(args, pieces, board_segmenter, policies)

    if args.screen_capture and args.source != "screen":
        return _run_mss_screen_capture(args, pieces, board_segmenter, policies)

    if args.source == "screen":
        if _macos_screen_capture_access(request=True) is False:
            print(MACOS_SCREEN_PERMISSION_MSG, file=sys.stderr)
        screen_scale = _detect_screen_scale()
        logical_screen = (1440, 900)
        if region is None:
            print("Deja visible tu tablero (Chess.com / Lichess), pulsa Enter si sale en verde o arrastra un cuadro...")
            picked = select_screen_board_region(screen_scale)
            if picked is None:
                print("Seleccion cancelada.")
                return 0
            region, logical_screen = picked
            print(f"Zona de tablero seleccionada: --screen-region {region[0]},{region[1]},{region[2]},{region[3]}")
        else:
            full_probe = _grab_screen_bgr(None)
            if full_probe is not None:
                logical_screen = (
                    max(640, round(full_probe.shape[1] / max(screen_scale, 1e-3))),
                    max(480, round(full_probe.shape[0] / max(screen_scale, 1e-3))),
                )

        args.full_frame_is_board = True
        source_feed: Any = LatestScreenFrame(region=region)
        if not source_feed.is_opened():
            print("No se pudo capturar la pantalla. Verifica permisos de Grabacion de Pantalla en macOS.", file=sys.stderr)
            return 1
        source_feed.start()
        inference = LiveInference(source_feed, pieces, board_segmenter, policies, args)
        inference.manual_locked = True
        inference.start()

        # Compact vertical layout by default so it fits beside Chess.com/Lichess without covering the board!
        compact_mode = True
        compact_size = (440, 820)
        wide_size = (960, 655)

        assistant_win = "Chess Assistant | 5 Modelos"
        cv2.namedWindow(assistant_win, cv2.WINDOW_NORMAL)
        win_w, win_h = compact_size
        wx, wy = compute_assistant_window_pos(region, logical_screen, (win_w, win_h))
        cv2.resizeWindow(assistant_win, win_w, win_h)
        cv2.moveWindow(assistant_win, wx, wy)
        print(
            f"Asistente de pantalla listo en ({wx}, {wy}) fuera de tu tablero.\n"
            "IMPORTANTE: NO pongas esta ventana encima del tablero de ajedrez.\n"
            "Teclas: [c] cambiar zona | [v] vista compacta/ancha | [t] turno | [f] girar (negras) | "
            "[k] enroque | [m] mascaras | [0/1-5] modelos | [s] guardar | [q/Esc] salir."
        )
        try:
            while True:
                snap = inference.state()
                board_vis = snap["board_image"]
                if board_vis is not None:
                    if compact_mode and board_vis.shape[1] == args.size + 370:
                        # Stack clear rectified board (480x480) above the telemetry sidebar + hotkey bar.
                        col_w = 480
                        board_part = cv2.resize(board_vis[:, :args.size], (col_w, col_w), interpolation=cv2.INTER_AREA)
                        panel_crop = board_vis[:500, args.size:]
                        panel_h = round(panel_crop.shape[0] * (col_w / panel_crop.shape[1]))
                        panel_part = cv2.resize(panel_crop, (col_w, panel_h), interpolation=cv2.INTER_AREA)
                        footer = np.full((46, col_w, 3), (22, 24, 28), dtype=np.uint8)
                        cv2.line(footer, (12, 2), (col_w - 12, 2), (70, 75, 85), 1)
                        cv2.putText(footer, "[c] Zona   [v] Vista   [t] Turno   [f] Girar", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 195, 210), 1, cv2.LINE_AA)
                        cv2.putText(footer, "[0/1-5] Modelos   [m] Mascaras   [q] Salir", (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 195, 210), 1, cv2.LINE_AA)
                        display_img = np.vstack([board_part, panel_part, footer])
                    else:
                        display_img = board_vis
                    cv2.imshow(assistant_win, display_img)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q")):
                    return 0
                if key == ord("v"):
                    compact_mode = not compact_mode
                    win_w, win_h = compact_size if compact_mode else wide_size
                    wx, wy = compute_assistant_window_pos(region, logical_screen, (win_w, win_h))
                    cv2.resizeWindow(assistant_win, win_w, win_h)
                    cv2.moveWindow(assistant_win, wx, wy)
                    continue
                if key == ord("c"):
                    _close_window(assistant_win)
                    picked = select_screen_board_region(screen_scale)
                    if picked is not None:
                        region, logical_screen = picked
                        source_feed.set_region(region)
                        inference.control(ord("r"))
                        inference.manual_locked = True
                        print(
                            f"Nueva zona seleccionada: --screen-region "
                            f"{region[0]},{region[1]},{region[2]},{region[3]}"
                        )
                    cv2.namedWindow(assistant_win, cv2.WINDOW_NORMAL)
                    win_w, win_h = compact_size if compact_mode else wide_size
                    wx, wy = compute_assistant_window_pos(region, logical_screen, (win_w, win_h))
                    cv2.resizeWindow(assistant_win, win_w, win_h)
                    cv2.moveWindow(assistant_win, wx, wy)
                    continue
                if key == ord("s") and board_vis is not None:
                    snap_dir = PROJECT_ROOT / "data" / "snapshots"
                    snap_dir.mkdir(parents=True, exist_ok=True)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = snap_dir / f"chess_{stamp}.png"
                    cv2.imwrite(str(out_path), board_vis)
                    print(f"Snapshot guardado: {out_path} | FEN: {snap['fen']}")
                    continue
                if key in (ord("t"), ord("f"), ord("k"), ord("m"), ord("0")) or (ord("1") <= key <= ord("9")):
                    inference.control(key)
        finally:
            inference.stop()
            source_feed.release()
            cv2.destroyAllWindows()

    camera_source = parse_camera_source(args.camera)
    source_feed = LatestFrameCamera(camera_source)
    if not source_feed.is_opened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    window_name = "Camera"

    source_feed.start()
    inference = LiveInference(source_feed, pieces, board_segmenter, policies, args)
    inference.start()

    calibrating = False
    clicked_points: list[tuple[int, int]] = []
    calibration_error = ""

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal calibrating, clicked_points, calibration_error
        if not calibrating or event != cv2.EVENT_LBUTTONDOWN:
            return
        clicked_points.append((x, y))
        if len(clicked_points) == 4:
            try:
                inference.set_manual_corners(np.array(clicked_points, dtype=np.float32))
                calibrating = False
                clicked_points = []
                calibration_error = ""
            except ValueError as err:
                calibration_error = f"Esquinas invalidas ({err}); intenta de nuevo"
                clicked_points = []

    board_win = "Board Analysis | YOLO-seg + 5 Move Models"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    board_win_ready = False
    print(
        "Live chess ready | [c] marcar 4 esquinas, [r] auto-tablero, [t] turno, [f] girar, "
        "[k] enroque, [m] mascaras, [0/1-5] modelos, [s] guardar, [q/Esc] salir."
    )

    try:
        while True:
            frame = source_feed.latest()
            if frame is None:
                cv2.waitKey(1)
                continue
            snap = inference.state()
            corners = snap["corners"]
            board_vis = snap["board_image"]
            visible_fen = snap["fen"]
            visible_move = snap["move"]
            inference_fps = snap["inference_fps"]
            latency_ms = snap["latency_ms"]
            all_results = snap["results"]
            visible_results = _filter_results(all_results, snap["active_model_idx"])

            preview = frame.copy()
            # Clean top status bar (never collides with draw_ar_overlay).
            _translucent_rect(preview, (0, 0), (preview.shape[1], 40), (20, 22, 26), 0.78)
            geometry = "manual" if snap["manual_locked"] else ("auto-locked" if corners is not None else "buscando")
            status_line = (
                f"{window_name.upper()} | YOLO {inference_fps:.1f} FPS | {latency_ms:.0f} ms | "
                f"board: {geometry} | [c] 4 esquinas [r] auto [t] turno [q] salir"
            )
            cv2.putText(preview, status_line, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (90, 255, 140), 1, cv2.LINE_AA)

            if calibrating:
                step_label = CALIBRATION_LABELS[min(len(clicked_points), 3)]
                _translucent_rect(preview, (12, 48), (520, 92), (20, 90, 220), 0.88)
                cv2.putText(
                    preview,
                    f"CALIBRAR: clic en {step_label}  (Esc cancela)",
                    (22, 76),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if calibration_error:
                    cv2.putText(preview, calibration_error, (22, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 100, 255), 2, cv2.LINE_AA)
                for idx, pt in enumerate(clicked_points):
                    cv2.circle(preview, pt, 7, (0, 255, 255), -1, cv2.LINE_AA)
                    cv2.putText(preview, str(idx + 1), (pt[0] + 10, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
                if len(clicked_points) >= 2:
                    cv2.polylines(preview, [np.array(clicked_points, np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
            elif corners is None:
                _translucent_rect(preview, (12, 48), (560, 86), (30, 30, 170), 0.80)
                cv2.putText(
                    preview,
                    "Buscando tablero 8x8... (o pulsa [c] para marcar 4 esquinas)",
                    (20, 73),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.polylines(preview, [corners.astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
                if visible_results:
                    draw_ar_overlay(
                        preview,
                        corners,
                        args.size,
                        visible_results,
                        snap["turn"],
                        snap["flip"],
                        visible_fen,
                    )

            # Bottom ticker on preview window
            _translucent_rect(preview, (0, preview.shape[0] - 32), (preview.shape[1], preview.shape[0]), (20, 22, 26), 0.78)
            bottom_txt = visible_move if visible_move else visible_fen
            cv2.putText(preview, bottom_txt[:115], (14, preview.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 240, 245), 1, cv2.LINE_AA)

            if board_vis is not None and not args.single_window:
                if not board_win_ready:
                    cv2.namedWindow(board_win, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(board_win, 936, 640)
                    board_win_ready = True
                cv2.imshow(board_win, board_vis)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                if calibrating:
                    calibrating = False
                    clicked_points = []
                    calibration_error = ""
                    continue
                return 0
            if key == ord("q"):
                return 0
            if key == ord("c"):
                calibrating = not calibrating
                clicked_points = []
                calibration_error = ""
                continue
            if key == ord("s") and board_vis is not None:
                snap_dir = PROJECT_ROOT / "data" / "snapshots"
                snap_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = snap_dir / f"chess_{stamp}.png"
                cv2.imwrite(str(out_path), board_vis)
                print(f"Snapshot guardado: {out_path} | FEN: {visible_fen}")
                continue
            if key in (ord("t"), ord("f"), ord("r"), ord("k"), ord("m"), ord("0")) or (ord("1") <= key <= ord("9")):
                inference.control(key)
    finally:
        inference.stop()
        source_feed.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
