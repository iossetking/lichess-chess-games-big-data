"""Train a per-piece crop classifier (second stage after the detector).

The detector finds *where* each piece is; this classifier looks at a single
square crop and decides *which* piece it is, which fixes shape confusions such
as knight/pawn or queen/king.

Training crops come from
* the ground-truth boxes of ``mixed-chess-shapes-aligned`` (real photos + 2D
  screen renders, six classes), several jittered crops per box so the
  classifier tolerates imperfect detector boxes, and
* the Kaggle "Chess Pieces Detection Images" folders (five classes, no king,
  single pieces) in ``data/vision/kaggle-chess-pieces/ChessPiece``.
Validation/test crops come only from real photos of the held-out splits.

Example:
    uv run python scripts/train_piece_classifier.py --epochs 25
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lichess_pipeline.chess_vision import PIECE_CLASSES, crop_piece  # noqa: E402

ALIGNED = ROOT / 'data/vision/mixed-chess-shapes-aligned'
KAGGLE = ROOT / 'data/vision/kaggle-chess-pieces/ChessPiece'
KAGGLE_FOLDERS = {'bishop': 'bishop_resized', 'knight': 'knight-resize', 'pawn': 'pawn_resized',
                  'queen': 'Queen-Resized', 'rook': 'Rook-resize'}
# Balance classes: Kaggle lacks kings and boards have 8x more pawns than kings/queens.
CLASS_COPY_MULTIPLIER = {
    'king': 3.0,
    'queen': 2.2,
    'bishop': 1.1,
    'knight': 1.1,
    'rook': 1.1,
    'pawn': 0.5,
}
DESTINATION = ROOT / 'data/vision/piece-crops'


def jitter(box, rng, strength):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    dx = rng.uniform(-strength, strength) * w
    # Keep top crown/cross inside the crop (crucial for king vs queen).
    dy = rng.uniform(-strength * 0.75, strength * 0.75) * h
    s = 1 + rng.uniform(-strength * 0.8, strength * 1.1, 2)
    cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
    return cx - w * s[0] / 2, cy - h * s[1] / 2, cx + w * s[0] / 2, cy + h * s[1] / 2


def build(size: int, copies: int, kaggle_copies: int, grayscale_prob: float = 0.2) -> dict:
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    rng = np.random.default_rng(42)
    counts: dict[str, dict[str, int]] = {}

    def save(split, name, image, stem):
        if split == 'train' and rng.random() < grayscale_prob:
            image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        folder = DESTINATION / split / name
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f'{stem}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        counts.setdefault(split, {}).setdefault(name, 0)
        counts[split][name] += 1

    for split in ('train', 'valid', 'test'):
        for domain in ('3d', '2d'):
            # Held-out splits: real photos only (3d); 2d is a single synthetic style.
            if split != 'train' and domain == '2d':
                continue
            for image_path in sorted((ALIGNED / split / domain / 'images').glob('*.jpg')):
                image = cv2.imread(str(image_path))
                height, width = image.shape[:2]
                for index, line in enumerate((ALIGNED / split / domain / 'labels' / f'{image_path.stem}.txt').read_text().splitlines()):
                    c, x, y, w, h = line.split()
                    box = ((float(x) - float(w) / 2) * width, (float(y) - float(h) / 2) * height,
                           (float(x) + float(w) / 2) * width, (float(y) + float(h) / 2) * height)
                    name = PIECE_CLASSES[int(c)]
                    stem = f'{domain}-{image_path.stem}-{index}'
                    if split == 'train':
                        mult = CLASS_COPY_MULTIPLIER[name]
                        base = copies if domain == '3d' else max(1, copies // 3)
                        n = max(1, round(base * mult))
                        for k in range(n):
                            b = box if k == 0 else jitter(box, rng, 0.10)
                            save(split, name, crop_piece(image, b, size), f'{stem}-{k}')
                    else:
                        save('val' if split == 'valid' else split, name, crop_piece(image, box, size), stem)
    for name, folder in KAGGLE_FOLDERS.items():
        k_copies = max(1, round(kaggle_copies * min(CLASS_COPY_MULTIPLIER[name], 1.5)))
        for path in sorted((KAGGLE / folder).glob('*.jpg')):
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            for k in range(k_copies):
                b = (0, 0, w, h) if k == 0 else jitter((0, 0, w, h), rng, 0.08)
                save('train', name, crop_piece(image, b, size, margin=1.1), f'kaggle-{path.stem}-{k}')
    (DESTINATION / 'counts.json').write_text(json.dumps(counts, indent=2))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=160)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--model', type=Path, default=ROOT / 'data/models/yolo11s-cls.pt')
    parser.add_argument('--copies', type=int, default=3, help='Base jittered crops per real 3D box')
    parser.add_argument('--kaggle-copies', type=int, default=3)
    parser.add_argument('--grayscale-prob', type=float, default=0.2, help='Probability of grayscale crop in train')
    parser.add_argument('--name', default='piece-classifier-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/models/chess-piece-classifier.pt')
    parser.add_argument('--skip-build', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    counts = json.loads((DESTINATION / 'counts.json').read_text()) if args.skip_build else build(
        args.imgsz, args.copies, args.kaggle_copies, args.grayscale_prob)
    print(json.dumps(counts, indent=2), flush=True)
    from ultralytics import YOLO

    if args.resume:
        result = YOLO(ROOT / 'data/runs/vision' / args.name / 'weights/last.pt').train(resume=True)
    else:
        result = YOLO(args.model).train(
            data=str(DESTINATION), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
            project=str(ROOT / 'data/runs/vision'), name=args.name, workers=0, patience=10, seed=42,
            hsv_h=.3, hsv_s=.6, hsv_v=.4, degrees=10, scale=.3, fliplr=.5, erasing=.3, cos_lr=True,
            label_smoothing=0.05, plots=True)
    shutil.copy2(Path(result.save_dir) / 'weights/best.pt', args.output)
    model = YOLO(args.output)
    report = {'counts': counts, 'weights': str(args.output.resolve().relative_to(ROOT))}
    for split in ('val', 'test'):
        metrics = model.val(data=str(DESTINATION), split=split, imgsz=args.imgsz, batch=args.batch,
                            device=args.device, workers=0, project=str(ROOT / 'data/runs/vision'),
                            name=f'{args.name}-{split}')
        report[split] = {'top1': float(metrics.top1), 'top5': float(metrics.top5)}
    args.output.with_name(args.output.stem + '-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
