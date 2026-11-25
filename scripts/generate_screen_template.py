"""Generate the 2D starting-position board template for screen YOLO-seg training.

Produces a borderless square RGB PNG (default 800x800) with Chess.com's default
green/cream square colours (``#ebecd0`` / ``#779556``) and the 32 starting pieces
rendered cleanly from ``chess.svg.PIECES`` without external SVG binaries.

Example:
    uv run python scripts/generate_screen_template.py
    uv run python scripts/generate_screen_template.py --output data/vision/screen-template.png --size 800
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import chess
import chess.svg
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "vision" / "screen-template.png"
DEFAULT_SIZE = 800
LIGHT_SQUARE = np.array((235, 236, 208), dtype=np.uint8)
DARK_SQUARE = np.array((119, 149, 86), dtype=np.uint8)
SVG_VIEWBOX = 45.0

_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_COMMAND_RE = re.compile(r"[MmLlHhVvCcSsAaZz]")
_MATRIX_RE = re.compile(r"matrix\(\s*([^)]+)\)")
_STYLE_KEYS = ("fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "fill-rule")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination PNG path")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Square board size in pixels (divisible by 8)")
    return parser.parse_args()


class _PathReader:
    """Sequential reader over an SVG ``d`` attribute supporting packed numbers and arc flags."""

    def __init__(self, d: str) -> None:
        self.d = d
        self.pos = 0
        self.length = len(d)

    def _skip_sep(self) -> None:
        while self.pos < self.length and self.d[self.pos] in " \t\r\n,":
            self.pos += 1

    def has_more_numbers(self) -> bool:
        self._skip_sep()
        if self.pos >= self.length:
            return False
        ch = self.d[self.pos]
        return ch.isdigit() or ch in "+-."

    def next_command(self) -> str | None:
        self._skip_sep()
        if self.pos >= self.length:
            return None
        match = _COMMAND_RE.match(self.d, self.pos)
        if match:
            self.pos = match.end()
            return match.group(0)
        return None

    def next_float(self) -> float:
        self._skip_sep()
        match = _NUMBER_RE.match(self.d, self.pos)
        if not match:
            raise ValueError(f"Expected number at index {self.pos} in SVG path: {self.d!r}")
        self.pos = match.end()
        return float(match.group(0))

    def next_pair(self) -> tuple[float, float]:
        return self.next_float(), self.next_float()

    def next_flag(self) -> int:
        self._skip_sep()
        if self.pos >= self.length or self.d[self.pos] not in "01":
            raise ValueError(f"Expected arc flag 0/1 at index {self.pos} in SVG path: {self.d!r}")
        flag = int(self.d[self.pos])
        self.pos += 1
        return flag


def _vector_angle(ux: float, uy: float, vx: float, vy: float) -> float:
    return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)


def _arc_points(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    x_rot_deg: float,
    large_arc: int,
    sweep: int,
    x2: float,
    y2: float,
) -> list[tuple[float, float]]:
    """Convert an SVG elliptical arc into polyline points via W3C endpoint-to-center math."""
    if x1 == x2 and y1 == y2:
        return []
    if rx == 0.0 or ry == 0.0:
        return [(x2, y2)]

    rx, ry = abs(rx), abs(ry)
    phi = math.radians(x_rot_deg % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2

    rx2, ry2 = rx * rx, ry * ry
    x1p2, y1p2 = x1p * x1p, y1p * y1p
    lam = x1p2 / rx2 + y1p2 / ry2
    if lam > 1.0:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale
        rx2, ry2 = rx * rx, ry * ry

    denom = rx2 * y1p2 + ry2 * x1p2
    num = max(0.0, rx2 * ry2 - denom)
    coef = math.sqrt(num / denom) if denom > 0.0 else 0.0
    if large_arc == sweep:
        coef = -coef

    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta1 = _vector_angle(1.0, 0.0, ux, uy)
    dtheta = _vector_angle(ux, uy, vx, vy)
    if not sweep and dtheta > 0.0:
        dtheta -= 2.0 * math.pi
    elif sweep and dtheta < 0.0:
        dtheta += 2.0 * math.pi

    steps = max(24, math.ceil(abs(dtheta) / (math.pi / 12.0)))
    points: list[tuple[float, float]] = []
    for step in range(1, steps + 1):
        if step == steps:
            points.append((x2, y2))
        else:
            theta = theta1 + dtheta * (step / steps)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            points.append((cx + cos_phi * rx * cos_t - sin_phi * ry * sin_t,
                           cy + sin_phi * rx * cos_t + cos_phi * ry * sin_t))
    return points


def _parse_svg_path(d: str) -> MplPath:
    """Parse an SVG path ``d`` string into a ``matplotlib.path.Path``."""
    reader = _PathReader(d)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    cur = np.array((0.0, 0.0))
    subpath_start = np.array((0.0, 0.0))
    prev_ctrl2: np.ndarray | None = None
    prev_cmd = ""

    while True:
        cmd = reader.next_command()
        if cmd is None:
            break
        if cmd in ("Z", "z"):
            vertices.append((float(subpath_start[0]), float(subpath_start[1])))
            codes.append(MplPath.CLOSEPOLY)
            cur = subpath_start.copy()
            prev_ctrl2 = None
            prev_cmd = cmd
            continue

        active_cmd = cmd
        while True:
            is_rel = active_cmd.islower()
            base = cur if is_rel else np.array((0.0, 0.0))
            upper = active_cmd.upper()

            if upper == "M":
                pt = base + np.array(reader.next_pair())
                vertices.append((float(pt[0]), float(pt[1])))
                codes.append(MplPath.MOVETO)
                cur = pt
                subpath_start = pt.copy()
                prev_ctrl2 = None
                prev_cmd = active_cmd
                # Per SVG spec, subsequent coordinate pairs after M/m are implicit L/l.
                active_cmd = "l" if is_rel else "L"
            elif upper == "L":
                pt = base + np.array(reader.next_pair())
                vertices.append((float(pt[0]), float(pt[1])))
                codes.append(MplPath.LINETO)
                cur = pt
                prev_ctrl2 = None
                prev_cmd = active_cmd
            elif upper == "H":
                x = (cur[0] if is_rel else 0.0) + reader.next_float()
                cur = np.array((x, cur[1]))
                vertices.append((float(cur[0]), float(cur[1])))
                codes.append(MplPath.LINETO)
                prev_ctrl2 = None
                prev_cmd = active_cmd
            elif upper == "V":
                y = (cur[1] if is_rel else 0.0) + reader.next_float()
                cur = np.array((cur[0], y))
                vertices.append((float(cur[0]), float(cur[1])))
                codes.append(MplPath.LINETO)
                prev_ctrl2 = None
                prev_cmd = active_cmd
            elif upper == "C":
                c1 = base + np.array(reader.next_pair())
                c2 = base + np.array(reader.next_pair())
                pt = base + np.array(reader.next_pair())
                vertices.extend([(float(c1[0]), float(c1[1])), (float(c2[0]), float(c2[1])), (float(pt[0]), float(pt[1]))])
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                cur = pt
                prev_ctrl2 = c2
                prev_cmd = active_cmd
            elif upper == "S":
                c1 = 2.0 * cur - prev_ctrl2 if prev_cmd.upper() in ("C", "S") and prev_ctrl2 is not None else cur.copy()
                c2 = base + np.array(reader.next_pair())
                pt = base + np.array(reader.next_pair())
                vertices.extend([(float(c1[0]), float(c1[1])), (float(c2[0]), float(c2[1])), (float(pt[0]), float(pt[1]))])
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
                cur = pt
                prev_ctrl2 = c2
                prev_cmd = active_cmd
            elif upper == "A":
                rx, ry = reader.next_pair()
                x_rot = reader.next_float()
                large_arc = reader.next_flag()
                sweep = reader.next_flag()
                pt = base + np.array(reader.next_pair())
                for px, py in _arc_points(float(cur[0]), float(cur[1]), rx, ry, x_rot, large_arc, sweep, float(pt[0]), float(pt[1])):
                    vertices.append((px, py))
                    codes.append(MplPath.LINETO)
                cur = pt
                prev_ctrl2 = None
                prev_cmd = active_cmd
            else:
                raise ValueError(f"Unsupported SVG path command {active_cmd!r}")

            if not reader.has_more_numbers():
                break

    return MplPath(vertices, codes)


def _merge_style(parent_style: dict[str, str], element: ET.Element) -> dict[str, str]:
    merged = dict(parent_style)
    for key in _STYLE_KEYS:
        if key in element.attrib:
            merged[key] = element.attrib[key].strip()
    inline = element.attrib.get("style", "")
    for item in inline.split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            key, value = key.strip(), value.strip()
            if key in _STYLE_KEYS:
                merged[key] = value
    return merged


def _parse_transform(element: ET.Element, parent_transform: Affine2D) -> Affine2D:
    spec = element.attrib.get("transform", "").strip()
    if not spec:
        return parent_transform
    match = _MATRIX_RE.fullmatch(spec)
    if not match:
        raise ValueError(f"Unsupported SVG transform: {spec!r}")
    a, b, c, d, e, f = (float(val) for val in re.split(r"[\s,]+", match.group(1).strip()))
    return Affine2D.from_values(a, b, c, d, e, f) + parent_transform


def _patch_kwargs(style: dict[str, str], stroke_scale: float) -> dict:
    fill = style.get("fill", "#000000")
    stroke = style.get("stroke", "none")
    stroke_width = float(style.get("stroke-width", "1"))
    capstyle = style.get("stroke-linecap", "butt")
    if capstyle == "square":
        capstyle = "projecting"
    joinstyle = style.get("stroke-linejoin", "miter")
    has_fill = fill.lower() != "none"
    has_stroke = stroke.lower() != "none" and stroke_width > 0.0
    return {
        "facecolor": fill if has_fill else "none",
        "edgecolor": stroke if has_stroke else "none",
        "linewidth": stroke_width * stroke_scale if has_stroke else 0.0,
        "capstyle": capstyle,
        "joinstyle": joinstyle,
        "antialiased": True,
    }


def _collect_patches(
    element: ET.Element,
    parent_style: dict[str, str],
    parent_transform: Affine2D,
    stroke_scale: float,
    patches: list[tuple[PathPatch | Circle, Affine2D]],
) -> None:
    style = _merge_style(parent_style, element)
    transform = _parse_transform(element, parent_transform)
    tag = element.tag.split("}")[-1]
    if tag == "path":
        d = element.attrib.get("d", "")
        if d:
            patches.append((PathPatch(_parse_svg_path(d), **_patch_kwargs(style, stroke_scale)), transform))
    elif tag == "circle":
        cx = float(element.attrib.get("cx", "0"))
        cy = float(element.attrib.get("cy", "0"))
        r = float(element.attrib.get("r", "0"))
        patches.append((Circle((cx, cy), r, **_patch_kwargs(style, stroke_scale)), transform))
    for child in element:
        _collect_patches(child, style, transform, stroke_scale, patches)


def render_piece_sprite(symbol: str, square_size: int) -> np.ndarray:
    """Rasterize one piece from ``chess.svg.PIECES`` into an RGBA ``square_size`` array."""
    dpi = 100.0
    fig = Figure(figsize=(square_size / dpi, square_size / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, SVG_VIEWBOX)
    ax.set_ylim(SVG_VIEWBOX, 0.0)
    ax.axis("off")
    ax.patch.set_alpha(0.0)

    stroke_scale = (square_size / dpi) * 72.0 / SVG_VIEWBOX
    root = ET.fromstring(chess.svg.PIECES[symbol])
    patches: list[tuple[PathPatch | Circle, Affine2D]] = []
    _collect_patches(root, {}, Affine2D(), stroke_scale, patches)
    for patch, transform in patches:
        patch.set_transform(transform + ax.transData)
        ax.add_patch(patch)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8).copy()


def empty_board(size: int) -> np.ndarray:
    """Create a borderless ``size x size`` checkerboard with exact flat square colours."""
    square_size = size // 8
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            colour = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            y1, x1 = row * square_size, col * square_size
            canvas[y1:y1 + square_size, x1:x1 + square_size] = colour
    return canvas


def generate_screen_template(output_path: Path = DEFAULT_OUTPUT, size: int = DEFAULT_SIZE) -> Path:
    """Generate the starting-position 2D screen template PNG and return ``output_path``."""
    if size < 8 or size % 8 != 0:
        raise ValueError("--size must be divisible by 8 and at least 8")

    square_size = size // 8
    canvas = empty_board(size)
    sprites = {symbol: render_piece_sprite(symbol, square_size) for symbol in chess.svg.PIECES}

    for square, piece in chess.Board().piece_map().items():
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        y1, x1 = row * square_size, col * square_size
        sprite = sprites[piece.symbol()]
        alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
        fg = sprite[:, :, :3].astype(np.float32)
        bg = canvas[y1:y1 + square_size, x1:x1 + square_size].astype(np.float32)
        blended = np.rint(fg * alpha + bg * (1.0 - alpha))
        canvas[y1:y1 + square_size, x1:x1 + square_size] = np.clip(blended, 0, 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(output_path)
    return output_path


def main() -> int:
    args = parse_args()
    if args.size < 8 or args.size % 8 != 0:
        print("--size must be divisible by 8 and at least 8", file=sys.stderr)
        return 2
    output = generate_screen_template(args.output, size=args.size)
    print(f"Screen template generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
