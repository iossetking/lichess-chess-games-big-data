"""Compare detector-only against detector + crop classifier fusion.

Each ground-truth piece is matched to the best-overlapping detection
(IoU >= 0.5, class-agnostic); the final label is then scored as
correct / confused / missed, for several classifier weights.

Example:
    uv run python scripts/evaluate_two_stage.py --split valid
    uv run python scripts/evaluate_two_stage.py --split test --weights-list 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lichess_pipeline.chess_vision import PIECE_CLASSES, crop_piece, fuse_piece_probabilities  # noqa: E402

DATASET = ROOT / 'data/vision/mixed-chess-shapes-aligned'


def iou(a, b):
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def classify(classifier, crops, device, imgsz, *, tta: bool = False):
    """Return an (n, 6) probability array ordered like PIECE_CLASSES."""
    if not crops:
        return np.zeros((0, len(PIECE_CLASSES)))

    def _run(batch):
        out = np.zeros((len(batch), len(PIECE_CLASSES)))
        for i, result in enumerate(classifier.predict(batch, imgsz=imgsz, device=device, verbose=False)):
            for class_id, name in result.names.items():
                if name in PIECE_CLASSES:
                    out[i, PIECE_CLASSES.index(name)] = float(result.probs.data[class_id])
        return out

    probs = _run(crops)
    if tta:
        flipped = [cv2.flip(crop, 1) for crop in crops]
        probs = 0.5 * (probs + _run(flipped))
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', type=Path, default=ROOT / 'data/models/chess-piece-shapes.pt')
    parser.add_argument('--classifier', type=Path, default=ROOT / 'data/models/chess-piece-classifier.pt')
    parser.add_argument('--split', nargs='+', default=['valid'])
    parser.add_argument('--conf', type=float, default=0.30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--crop-size', type=int, default=None,
                        help='Classifier crop size (auto-detected from checkpoint if omitted)')
    parser.add_argument('--tta', action='store_true', help='Average classifier predictions with horizontal flip')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--weights-list', type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 1000.0],
                        help='Classifier weights to try (0 = detector only, 1000 = classifier only)')
    args = parser.parse_args()
    from ultralytics import YOLO

    detector, classifier = YOLO(args.detector), YOLO(args.classifier)
    crop_size = args.crop_size or int(getattr(classifier, 'overrides', {}).get('imgsz', 128))
    tallies = {w: np.zeros((len(PIECE_CLASSES), len(PIECE_CLASSES) + 1), dtype=int) for w in args.weights_list}
    false_boxes = 0
    for split in args.split:
        for path in sorted((DATASET / split / '3d/images').glob('*.jpg')):
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            truth = []
            for line in (DATASET / split / '3d/labels' / f'{path.stem}.txt').read_text().splitlines():
                c, x, y, w, h = map(float, line.split())
                truth.append((int(c), np.array([(x - w / 2) * width, (y - h / 2) * height,
                                                (x + w / 2) * width, (y + h / 2) * height])))
            found = detector.predict(image, conf=args.conf, imgsz=args.imgsz, device=args.device,
                                     verbose=False, agnostic_nms=True, iou=0.5)[0]
            boxes = found.boxes.xyxy.cpu().numpy() if len(found.boxes) else np.zeros((0, 4))
            det_names = [found.names[int(c)] for c in found.boxes.cls.cpu().numpy()] if len(found.boxes) else []
            confs = found.boxes.conf.cpu().numpy() if len(found.boxes) else np.zeros(0)
            probs = classify(classifier, [crop_piece(image, b, crop_size) for b in boxes],
                             args.device, crop_size, tta=args.tta)
            for weight in args.weights_list:
                labels = [fuse_piece_probabilities(n, float(c), p, weight)[0] for n, c, p in zip(det_names, confs, probs)]
                used: set[int] = set()
                for gt_class, gt_box in truth:
                    if len(boxes) == 0:
                        tallies[weight][gt_class, -1] += 1
                        continue
                    overlaps = iou(gt_box, boxes)
                    for u in used:
                        overlaps[u] = 0
                    best = int(overlaps.argmax())
                    if overlaps[best] < 0.5:
                        tallies[weight][gt_class, -1] += 1
                        continue
                    used.add(best)
                    tallies[weight][gt_class, PIECE_CLASSES.index(labels[best])] += 1
                if weight == args.weights_list[0]:
                    false_boxes += len(boxes) - len(used)
    print(f'conf={args.conf:.2f} imgsz={args.imgsz} crop_size={crop_size} tta={args.tta} false_boxes={false_boxes}', flush=True)
    for weight, matrix in tallies.items():
        detected = matrix[:, :-1]
        correct, confused, missed, total = int(np.trace(detected)), int(detected.sum() - np.trace(detected)), int(matrix[:, -1].sum()), int(matrix.sum())
        print(f'classifier_weight={weight:>6}: pieces={total} correct={correct} confused={confused} missed={missed} '
              f'acc|det={correct / max(correct + confused, 1):.2%} recall={(correct + confused) / max(total, 1):.2%} '
              f'e2e={correct / total:.2%}', flush=True)
    last = tallies[args.weights_list[-1]]
    print('rows=truth; cols=' + ','.join(PIECE_CLASSES) + ',missed (last weight)')
    for i, name in enumerate(PIECE_CLASSES):
        print(f'{name:>7}', last[i].tolist())


if __name__ == '__main__':
    main()
