"""Check label conversion and isolation of held-out training data."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

spec = importlib.util.spec_from_file_location(
    'train_mixed', Path(__file__).resolve().parents[1] / 'scripts/train_mixed_chess_shapes.py'
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def test_prepare_maps_labels_and_excludes_test_duplicates(tmp_path, monkeypatch):
    source = tmp_path / 'data/vision/screen-chess-seg'
    source.mkdir(parents=True)
    (source / 'data.yaml').write_text(yaml.safe_dump({'names': {0: 'white-pawn', 1: 'black-king'}}))
    for index, split in enumerate(('train', 'valid', 'test')):
        for kind in ('images', 'labels'):
            (source / split / kind).mkdir(parents=True)
        image = np.full((20, 20, 3), 30 + index * 50, dtype=np.uint8)
        cv2.imwrite(str(source / split / 'images/example.png'), image)
        # Segmentation polygon -> detection box.
        (source / split / 'labels/example.txt').write_text('0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8')
    duplicate = cv2.imread(str(source / 'test/images/example.png'))
    cv2.imwrite(str(source / 'train/images/duplicate.png'), duplicate)
    (source / 'train/labels/duplicate.txt').write_text('1 0.5 0.5 0.2 0.2')
    cv2.imwrite(str(source / 'train/images/unlabelled.png'), np.full((20, 20, 3), 200, dtype=np.uint8))
    (source / 'train/labels/unlabelled.txt').write_text('')
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'SOURCES', ('screen-chess-seg',))
    destination = tmp_path / 'mixed'
    destination.mkdir()
    report = training.prepare(destination)
    assert report['exact_duplicates_removed'] == 1
    assert report['unlabelled_images_excluded'] == 1
    assert report['counts']['train/2d'] == 1
    assert not (destination / 'train/2d/images/screen-chess-seg-duplicate.jpg').exists()
    values = (destination / 'train/2d/labels/screen-chess-seg-example.txt').read_text().split()
    assert values[0] == '3'  # pawn, independent of source class ordering or colour
    np.testing.assert_allclose([float(v) for v in values[1:]], [.3, .5, .4, .6])
    assert json.loads((destination / 'preparation.json').read_text()) == report
