"""Train the six-shape piece detector (v2): cleaned labels + larger model.

What changed compared with ``train_mixed_chess_shapes.py``:

* **Label/image orientation repair.**  Many 960x1280 and 4032x3024 photos in
  ``chess-pieces-merged`` have labels that were drawn on a 90-degree-rotated
  copy of the picture, so the boxes landed on empty board squares.  Training on
  them teaches the network wrong shapes and counts them as "missed" in
  evaluation.  For each such image the four rotations are tried and the one
  whose pieces agree with a reference detector is kept; images where no
  rotation agrees (the labels are unusable) are dropped.
* Large photos are saved with a 1280 px long side (smaller pieces stay visible,
  training is faster).
* Larger backbone (YOLO11s by default), 640 px, stronger classification loss
  and augmentation geared towards viewpoint changes.

Example:
    uv run python scripts/train_piece_shapes_v2.py --epochs 80
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('bishop', 'king', 'knight', 'pawn', 'queen', 'rook')
SOURCES = ('chess-pieces-merged', 'screen-chess-seg')
SAFE_SIZES = {(640, 640), (640, 480), (480, 640)}  # verified aligned by hand/automatically
ROTATIONS = {'0': None, 'cw': cv2.ROTATE_90_CLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE,
             '180': cv2.ROTATE_180}
MAX_SIDE = 1280
MIN_ALIGNMENT = 0.5


def _xyxy(box, width, height):
    x, y, w, h = box
    return np.array([(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height])


def alignment_score(truth: list[np.ndarray], predicted: np.ndarray) -> float:
    """Mean best IoU (class-agnostic) of each ground-truth box against detections."""
    if not len(predicted) or not truth:
        return 0.0
    total = 0.0
    for box in truth:
        x1 = np.maximum(box[0], predicted[:, 0]); y1 = np.maximum(box[1], predicted[:, 1])
        x2 = np.minimum(box[2], predicted[:, 2]); y2 = np.minimum(box[3], predicted[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = ((box[2] - box[0]) * (box[3] - box[1])
                 + (predicted[:, 2] - predicted[:, 0]) * (predicted[:, 3] - predicted[:, 1]) - inter)
        total += float((inter / np.maximum(union, 1e-9)).max())
    return total / len(truth)


def shrink(image: np.ndarray) -> np.ndarray:
    scale = MAX_SIDE / max(image.shape[:2])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image


def best_rotation(reference, image: np.ndarray, boxes: list[list[float]], device: str) -> tuple[str, float]:
    scores = {}
    for name, code in ROTATIONS.items():
        rotated = image if code is None else cv2.rotate(image, code)
        height, width = rotated.shape[:2]
        found = reference.predict(rotated, conf=.15, imgsz=1280, device=device, verbose=False)[0]
        predicted = found.boxes.xyxy.cpu().numpy()
        scores[name] = alignment_score([_xyxy(b, width, height) for b in boxes], predicted)
    name = max(scores, key=scores.get)
    return name, scores[name]


def prepare(destination: Path, reference_weights: Path, device: str) -> dict:
    from ultralytics import YOLO

    reference = YOLO(reference_weights)
    seen, counts = set(), {}
    duplicates = unlabelled = 0
    fixed: dict[str, int] = {}
    dropped: list[str] = []
    rng = np.random.default_rng(42)
    if destination.exists():
        shutil.rmtree(destination)
    for split in ('test', 'valid', 'train'):
        for source in SOURCES:
            domain = '2d' if source == 'screen-chess-seg' else '3d'
            key = f'{split}/{domain}'
            counts.setdefault(key, 0)
            folder = ROOT / 'data/vision' / source
            source_names = yaml.safe_load((folder / 'data.yaml').read_text())['names']
            for image_path in sorted((folder / split / 'images').glob('*')):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f'Unreadable image: {image_path}')
                digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                labels, raw_boxes = [], []
                label_path = folder / split / 'labels' / f'{image_path.stem}.txt'
                for line in label_path.read_text().splitlines():
                    values = line.split()
                    class_id = NAMES.index(source_names[int(values[0])].split('-')[-1])
                    coordinates = np.array(values[1:], dtype=float)
                    if len(coordinates) != 4:
                        polygon = coordinates.reshape(-1, 2)
                        low, high = polygon.min(axis=0), polygon.max(axis=0)
                        coordinates = np.concatenate(((low + high) / 2, high - low))
                    if not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
                        raise ValueError(f'Invalid box in {label_path}')
                    labels.append((class_id, coordinates))
                    raw_boxes.append(coordinates.tolist())
                if not labels:
                    unlabelled += 1
                    continue
                stem = f'{source}-{image_path.stem}'
                # Only the physical, large-format photos have shifted labels.
                if domain == '3d' and (image.shape[1], image.shape[0]) not in SAFE_SIZES:
                    image = shrink(image)
                    rotation, score = best_rotation(reference, image, raw_boxes, device)
                    if score < MIN_ALIGNMENT:
                        dropped.append(f'{split}/{stem} (best={rotation}, score={score:.2f})')
                        continue
                    if rotation != '0':
                        image = cv2.rotate(image, ROTATIONS[rotation])
                        fixed[rotation] = fixed.get(rotation, 0) + 1
                lines = [f'{c} ' + ' '.join(f'{x:.7f}' for x in co) for c, co in labels]
                target = destination / split / domain
                for kind in ('images', 'labels'):
                    (target / kind).mkdir(parents=True, exist_ok=True)
                # Training includes grayscale examples to reduce reliance on colour.
                if split == 'train' and rng.random() < .3:
                    image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                if not cv2.imwrite(str(target / 'images' / f'{stem}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f'Cannot write {stem}')
                (target / 'labels' / f'{stem}.txt').write_text('\n'.join(lines))
                counts[key] += 1
                print(f'  prepared {key}/{stem}', end='\r', flush=True)
    config = {'path': str(destination), 'names': dict(enumerate(NAMES))}
    for split, setting in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        config[setting] = [f'{split}/{domain}/images' for domain in ('2d', '3d')]
    (destination / 'data.yaml').write_text(yaml.safe_dump(config))
    for domain in ('2d', '3d'):
        domain_config = dict(config, val=f'valid/{domain}/images', test=f'test/{domain}/images')
        (destination / f'{domain}.yaml').write_text(yaml.safe_dump(domain_config))
    report = {'counts': counts, 'exact_duplicates_removed': duplicates, 'unlabelled_images_excluded': unlabelled,
              'sources': list(SOURCES), 'seed': 42, 'orientation_fixed': fixed,
              'dropped_unusable_labels': dropped, 'reference_weights': str(reference_weights.relative_to(ROOT)),
              'excluded_source': 'chess-pieces-roboflow: local labels are all empty despite visible pieces'}
    (destination / 'preparation.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--model', type=Path, default=ROOT / 'data/models/yolo11s.pt')
    parser.add_argument('--reference', type=Path, default=ROOT / 'data/models/chess-piece-shapes-v1.pt')
    parser.add_argument('--cls', type=float, default=1.0, help='Classification loss gain (default 0.5 upstream)')
    parser.add_argument('--name', default='shapes-v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/models/chess-piece-shapes-v2.pt')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-prepare', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted run from its last.pt')
    args = parser.parse_args()
    destination = ROOT / 'data/vision/mixed-chess-shapes-aligned'
    if args.skip_prepare:
        report = json.loads((destination / 'preparation.json').read_text())
    else:
        report = prepare(destination, args.reference, args.device)
    print(json.dumps(report, indent=2), flush=True)
    if args.prepare_only:
        return
    from ultralytics import YOLO

    started = perf_counter()
    if args.resume:
        last = ROOT / 'data/runs/vision' / args.name / 'weights/last.pt'
        result = YOLO(last).train(resume=True)
    else:
        result = YOLO(args.model).train(data=str(destination / 'data.yaml'), epochs=args.epochs, imgsz=args.imgsz,
                             batch=args.batch, device=args.device, project=str(ROOT / 'data/runs/vision'),
                             name=args.name, workers=0, patience=20, seed=42, pretrained=True, cls=args.cls,
                             hsv_h=.5, hsv_s=.7, hsv_v=.4, degrees=15, translate=.12, scale=.6, shear=2,
                             perspective=.0005, fliplr=.5, mosaic=1.0, mixup=.1, erasing=.3,
                             close_mosaic=10, cos_lr=True, plots=True)
    best = Path(result.save_dir) / 'weights/best.pt'
    shutil.copy2(best, args.output)
    model = YOLO(args.output)
    report['weights'] = str(args.output.relative_to(ROOT))
    report['base_model'] = str(args.model.relative_to(ROOT))
    report['training_args'] = {k: str(v) for k, v in vars(args).items()}
    report['training_minutes'] = (perf_counter() - started) / 60
    report['run_directory'] = str(Path(result.save_dir).relative_to(ROOT))
    report['evaluation'] = {}
    for domain in ('2d', '3d'):
        metrics = model.val(data=str(destination / f'{domain}.yaml'), split='test', imgsz=args.imgsz,
                            batch=args.batch, device=args.device, workers=0,
                            project=str(ROOT / 'data/runs/vision'), name=f'{args.name}-test-{domain}')
        report['evaluation'][domain] = {k: float(v) for k, v in metrics.results_dict.items()}
    sample = cv2.imread(str(next((destination / 'test/3d/images').glob('*.jpg'))))
    model.predict(sample, imgsz=args.imgsz, device=args.device, verbose=False)
    times = []
    for _ in range(30):
        t = perf_counter()
        model.predict(sample, imgsz=args.imgsz, device=args.device, verbose=False)
        times.append((perf_counter() - t) * 1000)
    report['single_image_latency_ms'] = {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)),
                                         'device': args.device, 'imgsz': args.imgsz}
    args.output.with_name(args.output.stem + '-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
