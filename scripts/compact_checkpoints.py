"""Rewrite saved checkpoints as compact half-precision files.

New checkpoints saved by the notebooks are already compact (half precision).
This converts older ones, for example ``data/models/backups/``, so Git LFS
stores the smallest version of every model. Ultralytics YOLO checkpoints are
not plain state dicts, so they are skipped untouched.

Example:
    uv run python scripts/compact_checkpoints.py --dry-run
    uv run python scripts/compact_checkpoints.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch

from lichess_pipeline.model.checkpoints import (
    CHECKPOINT_SUFFIXES,
    compact_state_dict,
    discover_policy_checkpoints,
    is_lfs_pointer,
)

EXCLUDED_DIRS = {"maia3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[PROJECT_ROOT / "data" / "models"],
        help="Files or directories to scan (default: data/models)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report the sizes")
    return parser.parse_args()


def iter_checkpoints(targets: list[Path]) -> Iterator[Path]:
    for target in targets:
        if target.is_dir():
            for label, path in discover_policy_checkpoints(target).items():
                # The Maia-3 folder is a Hugging Face cache: re-downloaded on
                # demand and left out of Git, so its files keep their dtype.
                if EXCLUDED_DIRS.intersection(label.split("/")):
                    continue
                yield path
        elif target.is_file() and target.suffix.lower() in CHECKPOINT_SUFFIXES:
            yield target


def tensor_state(path: Path) -> tuple[dict, str | None]:
    """Return the loaded payload and the key that holds the tensor state dict."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
        nested = payload["state"]
        if nested and all(isinstance(value, torch.Tensor) for value in nested.values()):
            return payload, "state"
    if (
        isinstance(payload, dict)
        and payload
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    ):
        return payload, None
    raise ValueError("not a tensor state dict")


def compact(path: Path, dry_run: bool) -> tuple[float, float] | None:
    """Rewrite ``path`` in half precision; return (before, after) megabytes."""
    if is_lfs_pointer(path):
        return None
    payload, key = tensor_state(path)
    state = payload if key is None else payload[key]
    compacted = compact_state_dict(state)
    if all(tensor.dtype == compacted[name].dtype for name, tensor in state.items()):
        return None
    before = path.stat().st_size / 1024**2
    after = sum(tensor.numel() * tensor.element_size() for tensor in compacted.values())
    after /= 1024**2
    if not dry_run:
        updated = compacted if key is None else dict(payload, **{key: compacted})
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(updated, temporary)
        temporary.replace(path)
    return before, after


def main() -> int:
    args = parse_args()
    total_before = total_after = 0.0
    converted = skipped = 0
    for path in iter_checkpoints(args.paths):
        try:
            sizes = compact(path, args.dry_run)
        except Exception as error:  # noqa: BLE001 - YOLO/other opaque files must be skipped
            skipped += 1
            print(f"skip  {path.name}: {str(error).splitlines()[0]}")
            continue
        if sizes is None:
            continue
        before, after = sizes
        total_before += before
        total_after += after
        converted += 1
        print(
            f"{(args.dry_run and 'would') or 'done '} {path.name}: {before:.1f} MB -> {after:.1f} MB"
        )
    print(
        f"{converted} checkpoints {'to convert' if args.dry_run else 'converted'}, "
        f"{skipped} skipped: {total_before:.1f} MB -> {total_after:.1f} MB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
