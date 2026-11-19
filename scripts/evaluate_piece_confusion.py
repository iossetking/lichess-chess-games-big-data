"""Measure piece *classification* errors at the live operating point.

For every ground-truth piece the best-overlapping prediction (IoU >= 0.5,
ignoring class) is found, so each piece is either ``correct``, ``confused``
(wrong type, e.g. knight -> pawn) or ``missed``.  This is the number that the
live camera view makes visible, unlike mAP.

Example:
    uv run python scripts/evaluate_piece_confusion.py \
      --weights data/models/chess-piece-shapes.pt --split valid test --domain 3d
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('bishop', 'king', 'knight', 'pawn', 'queen', 'rook')
DATASET = ROOT / 'data/vision/mixed-chess-shapes-clean'


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def evaluate(weights: Path, splits: list[str], domain: str, conf: float, imgsz: int,
             device: str, augment: bool, agnostic: bool, dataset: Path = DATASET) -> dict:
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    names = {int(k): v for k, v in model.names.items()}
    matrix = np.zeros((len(NAMES), len(NAMES) + 1), dtype=int)  # rows GT, last col = missed
    extra = 0
    for split in splits:
        for image_path in sorted((dataset / split / domain / 'images').glob('*.jpg')):
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            truth = []
            for line in (dataset / split / domain / 'labels' / f'{image_path.stem}.txt').read_text().splitlines():
                c, x, y, w, h = map(float, line.split())
                truth.append((int(c), [(x - w / 2) * width, (y - h / 2) * height,
                                       (x + w / 2) * width, (y + h / 2) * height]))
            result = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False,
                                   augment=augment, agnostic_nms=agnostic)[0]
            boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.zeros((0, 4))
            classes = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) else np.zeros(0, int)
            used = set()
            for gt_class, gt_box in truth:
                if len(boxes) == 0:
                    matrix[gt_class, -1] += 1
                    continue
                overlaps = iou(np.asarray(gt_box), boxes)
                for u in used:
                    overlaps[u] = 0
                best = int(overlaps.argmax())
                if overlaps[best] < 0.5:
                    matrix[gt_class, -1] += 1
                    continue
                used.add(best)
                matrix[gt_class, NAMES.index(names[classes[best]])] += 1
            extra += len(boxes) - len(used)
    detected = matrix[:, :-1]
    correct = int(np.trace(detected))
    confused = int(detected.sum() - correct)
    missed = int(matrix[:, -1].sum())
    total = int(matrix.sum())
    return {
        'weights': str(weights), 'conf': conf, 'imgsz': imgsz, 'augment': augment, 'agnostic_nms': agnostic,
        'pieces': total, 'correct': correct, 'confused': confused, 'missed': missed, 'false_boxes': int(extra),
        'accuracy_when_detected': correct / max(correct + confused, 1),
        'recall': (correct + confused) / max(total, 1),
        'end_to_end_correct': correct / max(total, 1),
        'matrix_rows_truth_cols_pred_then_missed': {NAMES[i]: matrix[i].tolist() for i in range(len(NAMES))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'data/models/chess-piece-shapes.pt')
    parser.add_argument('--split', nargs='+', default=['valid', 'test'])
    parser.add_argument('--domain', choices=('2d', '3d'), default='3d')
    parser.add_argument('--conf', type=float, nargs='+', default=[0.25, 0.4, 0.55])
    parser.add_argument('--imgsz', type=int, nargs='+', default=[512])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--augment', action='store_true', help='Ultralytics test-time augmentation')
    parser.add_argument('--agnostic', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    results = []
    for size in args.imgsz:
        for conf in args.conf:
            r = evaluate(args.weights, args.split, args.domain, conf, size, args.device, args.augment,
                         args.agnostic, args.dataset)
            results.append(r)
            print(f"imgsz={size} conf={conf:.2f} aug={args.augment} agn={args.agnostic} | pieces={r['pieces']} "
                  f"correct={r['correct']} confused={r['confused']} missed={r['missed']} false={r['false_boxes']} | "
                  f"acc|det={r['accuracy_when_detected']:.1%} recall={r['recall']:.1%} e2e={r['end_to_end_correct']:.1%}",
                  flush=True)
    last = results[-1]['matrix_rows_truth_cols_pred_then_missed']
    print('rows=truth; cols=' + ','.join(NAMES) + ',missed')
    for name, row in last.items():
        print(f'{name:>7}', row)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
