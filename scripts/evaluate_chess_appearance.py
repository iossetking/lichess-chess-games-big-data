"""Evaluate the shape detector on recoloured and grayscale held-out images."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--imgsz', type=int, default=512)
    args = parser.parse_args()
    source = ROOT / 'data/vision/mixed-chess-shapes-clean'
    target = ROOT / 'data/vision/chess-shapes-clean-appearance-test'
    config = yaml.safe_load((source / 'data.yaml').read_text())
    report = {}
    from ultralytics import YOLO

    model = YOLO(ROOT / 'data/models/chess-piece-shapes.pt')
    for appearance in ('grayscale', 'hue-shift'):
        for domain in ('2d', '3d'):
            destination = target / appearance / domain
            for kind in ('images', 'labels'):
                (destination / kind).mkdir(parents=True, exist_ok=True)
            for image_path in (source / 'test' / domain / 'images').glob('*.jpg'):
                image = cv2.imread(str(image_path))
                if appearance == 'grayscale':
                    image = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
                else:
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    hsv[:, :, 0] = (hsv[:, :, 0].astype('uint16') + 75) % 180
                    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(str(destination / 'images' / image_path.name), image)
                shutil.copy2(source / 'test' / domain / 'labels' / f'{image_path.stem}.txt', destination / 'labels')
            data = {'path': str(destination), 'names': config['names'], 'train': 'images', 'val': 'images'}
            data_path = destination / 'data.yaml'
            data_path.write_text(yaml.safe_dump(data))
            metrics = model.val(data=str(data_path), imgsz=args.imgsz, batch=8,
                                device=args.device, workers=0,
                                project=str(ROOT / 'data/runs/vision'), name=f'shapes-{appearance}-{domain}')
            report[f'{appearance}/{domain}'] = {k: float(v) for k, v in metrics.results_dict.items()}
    report['limitations'] = 'Same held-out images with colour transforms; not a test of unseen piece designs, lighting or materials.'
    output = ROOT / 'data/models/chess-piece-shapes-appearance-report.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
