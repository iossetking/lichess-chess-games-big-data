"""Train six shape classes from local physical and synthetic-screen datasets."""
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
# The local Roboflow export has empty labels despite visible pieces.
SOURCES = ('chess-pieces-merged', 'screen-chess-seg')


def prepare(destination: Path) -> dict:
    """Preserve held-out splits; deduplicate decoded images before training."""
    seen = set()
    counts = {}
    duplicates = 0
    unlabelled = 0
    rng = np.random.default_rng(42)
    for split in ('test', 'valid', 'train'):
        for source in SOURCES:
            domain = '2d' if source == 'screen-chess-seg' else '3d'
            key = f'{split}/{domain}'
            counts.setdefault(key, 0)
            folder = ROOT / 'data/vision' / source
            config = yaml.safe_load((folder / 'data.yaml').read_text())
            source_names = config['names']
            for image_path in sorted((folder / split / 'images').glob('*')):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError(f'Unreadable image: {image_path}')
                digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                labels = []
                label_path = folder / split / 'labels' / f'{image_path.stem}.txt'
                for line in label_path.read_text().splitlines():
                    values = line.split()
                    name = source_names[int(values[0])].split('-')[-1]
                    class_id = NAMES.index(name)
                    coordinates = np.array(values[1:], dtype=float)
                    if len(coordinates) != 4:
                        polygon = coordinates.reshape(-1, 2)
                        low, high = polygon.min(axis=0), polygon.max(axis=0)
                        coordinates = np.concatenate(((low + high) / 2, high - low))
                    if not np.isfinite(coordinates).all() or (coordinates < 0).any() or (coordinates > 1).any():
                        raise ValueError(f'Invalid box in {label_path}')
                    labels.append(f'{class_id} ' + ' '.join(f'{x:.7f}' for x in coordinates))
                if not labels:
                    unlabelled += 1
                    continue
                target = destination / split / domain
                for kind in ('images', 'labels'):
                    (target / kind).mkdir(parents=True, exist_ok=True)
                stem = f'{source}-{image_path.stem}'
                # Training includes grayscale examples to reduce reliance on colour.
                if split == 'train' and rng.random() < .3:
                    image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                if not cv2.imwrite(str(target / 'images' / f'{stem}.jpg'), image):
                    raise RuntimeError(f'Cannot write {stem}')
                (target / 'labels' / f'{stem}.txt').write_text('\n'.join(labels))
                counts[key] += 1
    config = {'path': str(destination), 'names': dict(enumerate(NAMES))}
    for split, setting in (('train', 'train'), ('valid', 'val'), ('test', 'test')):
        config[setting] = [f'{split}/{domain}/images' for domain in ('2d', '3d')]
    (destination / 'data.yaml').write_text(yaml.safe_dump(config))
    for domain in ('2d', '3d'):
        domain_config = dict(config, val=f'valid/{domain}/images', test=f'test/{domain}/images')
        (destination / f'{domain}.yaml').write_text(yaml.safe_dump(domain_config))
    report = {'counts': counts, 'exact_duplicates_removed': duplicates, 'unlabelled_images_excluded': unlabelled,
              'sources': list(SOURCES), 'seed': 42,
              'excluded_source': 'chess-pieces-roboflow: local labels are all empty despite visible pieces',
              'limitations': '2D uses one sprite style; physical sources may contain near duplicates not caught by exact pixel hashing.'}
    (destination / 'preparation.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=512)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--name', default='mixed-chess-shapes-clean')
    args = parser.parse_args()
    destination = ROOT / 'data/vision/mixed-chess-shapes-clean'
    destination.mkdir(parents=True, exist_ok=True)
    report = prepare(destination)
    print(json.dumps(report, indent=2), flush=True)
    if args.prepare_only:
        return
    from ultralytics import YOLO

    initial = ROOT / 'data/models/chess-piece-shapes-initial.pt'
    if not initial.exists():
        previous = ROOT / 'data/models/chess-piece-shapes.pt'
        shutil.copy2(previous if previous.exists() else ROOT / 'data/models/chess-pieces.pt', initial)
    model = YOLO(initial)
    result = model.train(data=str(destination / 'data.yaml'), epochs=args.epochs,
                         imgsz=args.imgsz, batch=args.batch, device=args.device,
                         project=str(ROOT / 'data/runs/vision'), name=args.name,
                         workers=0, patience=8, seed=42, pretrained=True,
                         hsv_h=.5, hsv_s=.8, hsv_v=.4, degrees=12,
                         perspective=.0003, fliplr=.5, mosaic=.5, close_mosaic=5,
                         plots=True)
    best = Path(result.save_dir) / 'weights/best.pt'
    output = ROOT / 'data/models/chess-piece-shapes.pt'
    shutil.copy2(best, output)
    model = YOLO(output)
    report['weights'] = str(output.relative_to(ROOT))
    report['initial_weights'] = str(initial.relative_to(ROOT))
    report['initial_weights_sha256'] = hashlib.sha256(initial.read_bytes()).hexdigest()
    report['training_args'] = vars(args)
    report['run_directory'] = str(Path(result.save_dir).relative_to(ROOT))
    report['evaluation'] = {}
    for domain in ('2d', '3d'):
        metrics = model.val(data=str(destination / f'{domain}.yaml'), split='test',
                            imgsz=args.imgsz, batch=args.batch, device=args.device,
                            workers=0, project=str(ROOT / 'data/runs/vision'),
                            name=f'{args.name}-test-{domain}')
        report['evaluation'][domain] = {k: float(v) for k, v in metrics.results_dict.items()}
        report['evaluation'][domain]['speed_ms'] = metrics.speed
    sample = cv2.imread(str(next((destination / 'test/3d/images').glob('*.jpg'))))
    model.predict(sample, imgsz=args.imgsz, device=args.device, verbose=False)
    times = []
    for _ in range(20):
        started = perf_counter()
        model.predict(sample, imgsz=args.imgsz, device=args.device, verbose=False)
        times.append((perf_counter() - started) * 1000)
    report['single_image_latency_ms'] = {'median': float(np.median(times)), 'p95': float(np.percentile(times, 95)),
                                          'device': args.device, 'imgsz': args.imgsz,
                                          'note': 'Repeated still image; excludes camera and UI.'}
    (ROOT / 'data/models/chess-piece-shapes-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
