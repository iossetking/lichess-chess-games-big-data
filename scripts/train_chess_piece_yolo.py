"""Download the public chess-piece dataset and fine-tune a compact YOLO model.

The source is the CC-BY-4.0 ``dopaul/chess-pieces-merged`` mirror.  It merges
the public Roboflow chess-piece labels with a complementary dataset and exposes
train/valid/test splits with the 12 coloured piece classes.

Example (Apple Silicon):
    uv run python scripts/train_chess_piece_yolo.py --epochs 50 --device mps

The resulting checkpoint is copied to ``data/models/chess-pieces.pt`` and can
be consumed directly by ``scripts/live_chess_fen.py``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

DATASET_ID = "dopaul/chess-pieces-merged"
CLASS_NAMES = (
    "black-bishop",
    "black-king",
    "black-knight",
    "black-pawn",
    "black-queen",
    "black-rook",
    "white-bishop",
    "white-king",
    "white-knight",
    "white-pawn",
    "white-queen",
    "white-rook",
)
SPLITS = {"train": "train", "valid": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, help="Use an existing YOLO dataset, including mixed 2D/3D images")
    parser.add_argument("--shape-only", action="store_true", help="Export six piece types, merging white and black labels")
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset identifier")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "vision" / "chess-pieces-merged")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="Ultralytics device, e.g. mps, cpu, 0")
    parser.add_argument(
        "--base-model",
        default=str(PROJECT_ROOT / "data" / "models" / "yolo11n.pt"),
        help="Pretrained Ultralytics detection checkpoint",
    )
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "data" / "runs" / "vision")
    parser.add_argument("--name", default="chess-pieces-yolo11n")
    parser.add_argument("--download-only", action="store_true", help="Export the YOLO dataset but do not train")
    parser.add_argument("--force-export", action="store_true", help="Recreate dataset files even if data.yaml exists")
    parser.add_argument("--include-empty", action="store_true", help="Keep source rows with no piece annotations as background images")
    return parser.parse_args()


def _normalise(value: float, dimension: int) -> float:
    """Accept either YOLO-normalised or pixel annotations from a dataset mirror."""
    return value / dimension if value > 1 else value


def export_yolo_dataset(
    dataset_id: str, destination: Path, force: bool = False, include_empty: bool = False,
    shape_only: bool = False
) -> Path:
    """Download the public dataset and materialise its images/labels in YOLO layout."""
    if shape_only:
        destination = destination / "shape-only"
    data_yaml = destination / "data.yaml"
    if data_yaml.exists() and not force:
        return data_yaml

    from datasets import load_dataset

    dataset = load_dataset(dataset_id)
    for source_split, target_split in SPLITS.items():
        if source_split not in dataset:
            raise ValueError(f"Dataset {dataset_id!r} has no {source_split!r} split")
        split_dir = destination / target_split
        if force and split_dir.exists():
            shutil.rmtree(split_dir)
        image_dir = split_dir / "images"
        label_dir = split_dir / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for item in dataset[source_split]:
            image = item["image"].convert("RGB")
            annotations = item["annotations"]
            classes = annotations["class_id"]
            if not classes and not include_empty:
                continue
            x_centers = annotations["x_center"]
            y_centers = annotations["y_center"]
            widths = annotations["width"]
            heights = annotations["height"]
            if not (len(classes) == len(x_centers) == len(y_centers) == len(widths) == len(heights)):
                raise ValueError(f"Inconsistent annotation lengths in {source_split} item {item['image_id']}")
            image_path = image_dir / f"{written:06}.jpg"
            image.save(image_path, quality=95)
            labels = [
                f"{int(class_id) % 6 if shape_only else int(class_id)} {_normalise(float(x_center), image.width):.8f} "
                f"{_normalise(float(y_center), image.height):.8f} "
                f"{_normalise(float(width), image.width):.8f} "
                f"{_normalise(float(height), image.height):.8f}"
                for class_id, x_center, y_center, width, height in zip(
                    classes, x_centers, y_centers, widths, heights, strict=True
                )
            ]
            (label_dir / f"{written:06}.txt").write_text("\n".join(labels), encoding="utf-8")
            written += 1
        if written == 0:
            raise ValueError(f"No annotated images were exported for split {source_split!r}")

    class_names = tuple(name.removeprefix("black-") for name in CLASS_NAMES[:6]) if shape_only else CLASS_NAMES
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(
        f"path: {destination}\ntrain: train/images\nval: valid/images\ntest: test/images\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return data_yaml


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        print("--epochs, --imgsz and --batch must be positive", file=sys.stderr)
        return 2
    if args.data_yaml and args.shape_only:
        print("Use six-class labels in --data-yaml; --shape-only applies to exported datasets", file=sys.stderr)
        return 2
    data_yaml = args.data_yaml or export_yolo_dataset(
        args.dataset, args.data_dir, force=args.force_export, include_empty=args.include_empty,
        shape_only=args.shape_only,
    )
    if not data_yaml.is_file():
        print(f"Dataset configuration not found: {data_yaml}", file=sys.stderr)
        return 2
    print(f"YOLO dataset ready: {data_yaml}")
    if args.download_only:
        return 0

    from ultralytics import YOLO

    model = YOLO(args.base_model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=max(10, args.epochs // 5),
        seed=42,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"Ultralytics did not produce {best}")
    output = PROJECT_ROOT / "data" / "models" / ("chess-piece-shapes.pt" if args.shape_only else "chess-pieces.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    print(f"Trained weights copied to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
