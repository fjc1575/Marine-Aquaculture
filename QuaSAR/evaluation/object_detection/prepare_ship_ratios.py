"""Prepare ship quality-ratio datasets and COCO annotations.
"""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import random
import shutil

from PIL import Image


TIERS = ('high', 'medium', 'low')
PREFIXES = ('H_', 'M_', 'L_')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def generate_ratios():
    return [(h / 10, m / 10, (10 - h - m) / 10)
            for h in range(11) for m in range(11 - h)]


def allocate_counts(total, weights):
    """Use the source script's largest-remainder allocation, with H/M/L tie order."""
    exact = [total * weight / sum(weights) for weight in weights]
    counts = [int(value) for value in exact]
    fractions = sorted(range(3), key=lambda i: exact[i] - counts[i], reverse=True)
    for index in fractions[:total - sum(counts)]:
        counts[index] += 1
    return counts


def integer_weights(ratio):
    values = [Decimal(str(value)) for value in ratio]
    scale = 10 ** max(0, *(-value.as_tuple().exponent for value in values))
    return [int(value * scale) for value in values]


def find_label(image, label_dirs):
    stem = image.stem
    if stem.startswith(PREFIXES):
        stem = stem[2:]
    for directory in label_dirs:
        candidate = directory / (stem + '.txt')
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No YOLO label for {image}')


def read_boxes(label, width, height):
    boxes = []
    for line_number, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != '0':
            raise ValueError(f'{label}:{line_number}: expected class 0 and four YOLO coordinates')
        cx, cy, w, h = map(float, parts[1:])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f'{label}:{line_number}: invalid normalized coordinates')
        # Preserve the source conversion: clamp the upper-left corner to zero.
        boxes.append([max(0, (cx - w / 2) * width), max(0, (cy - h / 2) * height),
                      w * width, h * height])
    return boxes


def prepare_ratio(pools, label_dirs, output_root, total, ratio, rng, seed,
                  val_images=None, val_json=None):
    counts = allocate_counts(total, integer_weights(ratio))
    for tier, count in zip(TIERS, counts):
        if count > len(pools[tier]):
            raise ValueError(f'{tier}: requested {count}, available {len(pools[tier])}')
    destination = output_root / ('high_{}_medium_{}_low_{}'.format(*ratio))
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')

    selected = []
    for tier, prefix, count in zip(TIERS, PREFIXES, counts):
        for image in rng.sample(pools[tier], count):
            label = find_label(image, label_dirs)
            with Image.open(image) as opened:
                width, height = opened.size
            selected.append((tier, image, label, prefix + image.name, width, height,
                             read_boxes(label, width, height)))
    if len({row[3] for row in selected}) != total:
        raise ValueError('Selected output names are not unique')
    coco = {'info': {'description': 'MSAR ship quality-ratio training subset'},
            'licenses': [], 'images': [], 'annotations': [],
            'categories': [{'id': 1, 'name': 'ship'}]}
    (destination / 'train2017').mkdir(parents=True)
    (destination / 'annotations').mkdir()
    manifest = []
    for image_id, (tier, image, label, name, width, height, boxes) in enumerate(selected, 1):
        shutil.copy2(image, destination / 'train2017' / name)
        coco['images'].append(dict(id=image_id, file_name=name, width=width, height=height))
        for box in boxes:
            coco['annotations'].append(dict(id=len(coco['annotations']) + 1,
                image_id=image_id, category_id=1, bbox=box, area=box[2] * box[3], iscrowd=0))
        manifest.append(dict(quality=tier, source_image=str(image), source_label=str(label),
                             file_name=name, boxes=len(boxes)))
    (destination / 'annotations' / 'instances_train2017.json').write_text(
        json.dumps(coco, ensure_ascii=False, indent=2), encoding='utf-8')
    if val_images is not None:
        shutil.copytree(val_images, destination / 'val2017')
        shutil.copy2(val_json, destination / 'annotations' / 'instances_val2017.json')
    summary = dict(ratio=ratio, counts=dict(zip(TIERS, counts)), total_samples=total,
                   seed=seed, images=manifest, validation_included=val_images is not None)
    (destination / 'sampling.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(output=str(destination), counts=summary['counts'],
                images=len(coco['images']), annotations=len(coco['annotations']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True,
                        help='root containing high/train, medium/train and low/train')
    parser.add_argument('--label-dirs', type=Path, nargs='+', required=True,
                        help='YOLO txt directories, searched in the supplied order')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--total-samples', type=int, default=1293)
    parser.add_argument('--ratio', type=float, nargs=3, action='append',
                        metavar=('HIGH', 'MEDIUM', 'LOW'), help='repeatable; omit for all 66 ratios')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-images', type=Path)
    parser.add_argument('--val-json', type=Path)
    args = parser.parse_args()
    if args.total_samples <= 0:
        parser.error('--total-samples must be positive')
    ratios = [tuple(row) for row in args.ratio] if args.ratio else generate_ratios()
    if any(any(not math.isfinite(v) or v < 0 or v > 1 for v in row)
           or not math.isclose(sum(row), 1.0, abs_tol=1e-8) for row in ratios):
        parser.error('each ratio must contain three nonnegative values summing to 1')
    if (args.val_images is None) != (args.val_json is None):
        parser.error('--val-images and --val-json must be supplied together')
    pools = {}
    for tier in TIERS:
        directory = args.input_root / tier / 'train'
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        pools[tier] = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for directory in args.label_dirs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if args.val_images is not None:
        validation = json.loads(args.val_json.read_text(encoding='utf-8-sig'))
        for image in validation['images']:
            if not (args.val_images / image['file_name']).is_file():
                raise FileNotFoundError(args.val_images / image['file_name'])
    rng = random.Random(args.seed)
    for ratio in ratios:
        print(json.dumps(prepare_ratio(pools, args.label_dirs, args.output_root,
            args.total_samples, ratio, rng, args.seed, args.val_images, args.val_json),
            ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
