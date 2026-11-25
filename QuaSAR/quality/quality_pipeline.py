"""Gamma stretching and image-quality metrics in one CSV.
Metrics are measured on the saved JPEG pixels after Gamma stretching.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.stats as st
from joblib import Parallel, delayed
from osgeo import gdal
from tqdm import tqdm


METRIC_COLUMNS = ['Mean', 'MSE', 'STD', 'C', 'ENL', 'RL', 'GL', 'G', 'B', 'V', 'SNR', 'IE']
COLUMNS = ['File'] + METRIC_COLUMNS + ['Gamma wucha', 'optimal_scale', 'Status', 'Error']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def read_sar(path):
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise ValueError('Cannot read image: ' + str(path))
    width, height = dataset.RasterXSize, dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    dataset = None
    if data is None or data.size == 0:
        raise ValueError('Empty image: ' + str(path))
    return data[0:height, 0:width]


def stretch_bounds(img, scale, low_percentile=1, high_percentile=99.9):
    non_zero_pixels = img[img > 0]
    if non_zero_pixels.size == 0:
        non_zero_pixels = img.flatten()
    mean = non_zero_pixels.mean()
    std = non_zero_pixels.std()
    lower = np.percentile(non_zero_pixels, low_percentile)
    upper = np.percentile(non_zero_pixels, high_percentile)
    upper = min(upper, mean + scale * std)
    lower = max(lower, 0)
    if lower >= upper:
        upper = lower + 1e-5
    return lower, upper


def fit_error(scale, flat, x, reference_pdf):
    lower, upper = stretch_bounds(flat, scale)
    stretched = np.clip(flat, lower, upper)
    stretched = (stretched - lower) / (upper - lower) * 255
    valid = stretched[np.isfinite(stretched)]
    if valid.size == 0:
        return np.inf, scale
    mean, variance = np.mean(valid), np.var(valid)
    if mean <= 0 or variance <= 0 or not np.isfinite(mean + variance):
        return np.inf, scale
    shape = mean ** 2 / variance
    gamma_scale = variance / mean
    pdf = st.gamma.pdf(x, shape, scale=gamma_scale)
    region = (x > 0) & (x < 25)
    error = np.sum((reference_pdf[region] - pdf[region]) ** 2)
    return (error if np.isfinite(error) else np.inf), scale


def reference_distribution(path):
    flat = read_sar(path).astype(np.float32).flatten()
    if not np.isfinite(flat).all() or np.var(flat) <= 0 or np.mean(flat) <= 0:
        raise ValueError('Reference must have finite pixels, positive mean and nonzero variance')
    shape = np.mean(flat) ** 2 / np.var(flat)
    gamma_scale = np.var(flat) / np.mean(flat)
    x = np.linspace(0.001, 255, 2000)
    pdf = st.gamma.pdf(x, shape, scale=gamma_scale)
    if not np.isfinite(pdf).all():
        raise ValueError('Reference produced a nonfinite Gamma distribution')
    return x, pdf


def gamma_stretch(pixels, x, reference_pdf, jobs=1):
    original = pixels.astype(np.float32)
    flat = original.flatten()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0 or np.var(flat) <= 0:
        raise ValueError('Image has no varying finite pixels; Gamma fitting is undefined')
    coarse = Parallel(n_jobs=jobs)(
        delayed(fit_error)(i / 10, flat, x, reference_pdf) for i in range(0, 1000, 20))
    error, scale = min(coarse, key=lambda item: item[0])
    if not np.isfinite(error):
        raise ValueError('No finite Gamma fit in coarse search')
    fine_start = max(int(scale * 10) - 20, 0)
    fine_end = min(int(scale * 10) + 20, 1000)
    fine = Parallel(n_jobs=jobs)(
        delayed(fit_error)(i / 10, flat, x, reference_pdf)
        for i in range(fine_start, fine_end, 5))
    error, scale = min(fine, key=lambda item: item[0])
    if not np.isfinite(error):
        raise ValueError('No finite Gamma fit in fine search')
    lower, upper = stretch_bounds(original, scale)
    stretched = np.clip(original, lower, upper)
    stretched = (stretched - lower) / (upper - lower) * 255
    if stretched.ndim == 3:
        stretched = stretched[1, :, :]
    stretched = np.nan_to_num(stretched, nan=0.0, posinf=255, neginf=0.0)
    return np.clip(stretched, 0, 255).astype(np.uint8), error, scale


def image_metrics(gray):
    img = gray.astype(np.float32)
    mean, variance, std = np.mean(img), np.var(img), np.std(img)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = 10 * np.log10(np.mean(img ** 2) / variance)
        enl = mean ** 2 / variance
        resolution = 10 * np.log10(1 / (enl ** 0.5) + 1)
        low, high = np.percentile(img, [10, 90])
        gray_ratio = low / high
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.mean(np.sqrt(gx ** 2 + gy ** 2))
    bx = (img[:, 2:] - img[:, :-2]) ** 2
    by = (img[2:, :] - img[:-2, :]) ** 2
    brenner = bx.sum() + by.sum()
    vertical = (img[1:, :] - img[:-1, :]) ** 2
    horizontal = (img[:, 1:] - img[:, :-1]) ** 2
    count = vertical.size + horizontal.size
    contrast = (vertical.sum() + horizontal.sum()) / count if count else np.nan
    hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    vollath = np.sum(img[:-1] * img[1:]) - np.sum(img ** 2)
    return {'Mean': mean, 'MSE': variance, 'STD': std, 'C': contrast,
            'ENL': enl, 'RL': resolution, 'GL': gray_ratio, 'G': gradient,
            'B': brenner, 'V': vollath, 'SNR': snr, 'IE': entropy}


def process_image(source, target, x, reference_pdf, jobs=1):
    stretched, error, scale = gamma_stretch(read_sar(source), x, reference_pdf, jobs)
    ok, encoded = cv2.imencode('.jpg', stretched)
    if not ok:
        raise OSError('JPEG encoding failed: ' + str(source))
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise OSError('Encoded JPEG could not be decoded: ' + str(source))
    metrics = image_metrics(gray)
    target.write_bytes(encoded.tobytes())
    row = {'File': source.name, **metrics, 'Gamma wucha': error, 'optimal_scale': scale,
           'Status': 'ok', 'Error': ''}
    if not all(np.isfinite(value) for value in metrics.values()):
        row['Status'] = 'nonfinite_metrics'
        row['Error'] = 'Nonfinite metrics: ' + ', '.join(k for k, v in metrics.items() if not np.isfinite(v))
    return row


def parse_args():
    parser = argparse.ArgumentParser(description='Gamma stretch and extract image metrics into one CSV.')
    parser.add_argument('--input-dir', type=Path, required=True, help='Directory of original SAR images')
    parser.add_argument('--reference', type=Path, required=True, help='Reference SAR image')
    parser.add_argument('--output-dir', type=Path, required=True, help='Directory for stretched JPEG images')
    parser.add_argument('--output-csv', type=Path, required=True, help='Combined image metrics and Gamma errors')
    parser.add_argument('--files', nargs='+', help='Optional input filenames for a subset')
    parser.add_argument('--jobs', type=int, default=1, help='Parallel jobs within each Gamma search')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing existing output files')
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir, output_dir = args.input_dir.absolute(), args.output_dir.absolute()
    csv_path = args.output_csv.absolute()
    if not source_dir.is_dir():
        raise ValueError('Input directory does not exist: ' + str(source_dir))
    if source_dir == output_dir:
        raise ValueError('Output directory must differ from the original image directory')
    if args.jobs == 0:
        raise ValueError('--jobs must not be zero')
    if args.files:
        if any(Path(name).name != name for name in args.files):
            raise ValueError('--files accepts filenames only')
        files = [source_dir / name for name in args.files]
    else:
        files = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in EXTENSIONS and p.is_file())
    if not files:
        raise ValueError('No input images found')
    output_names = [p.stem + '.jpg' for p in files]
    if len({name.casefold() for name in output_names}) != len(output_names):
        raise ValueError('Duplicate output basenames; input images would overwrite each other')
    targets = [output_dir / name for name in output_names]
    originals = {p.absolute() for p in files} | {args.reference.absolute()}
    if csv_path in originals or any(target in originals for target in targets):
        raise ValueError('Output would overwrite an input or reference image')
    if not args.overwrite and (csv_path.exists() or any(p.exists() for p in targets)):
        raise FileExistsError('Output exists; choose a new location or use --overwrite')
    x, reference_pdf = reference_distribution(args.reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for source, target in tqdm(list(zip(files, targets)), desc='Gamma + metrics'):
        try:
            row = process_image(source, target, x, reference_pdf, args.jobs)
        except Exception as error:
            row = {'File': source.name, 'Status': 'error',
                   'Error': type(error).__name__ + ': ' + str(error)}
            print(row['File'] + ': ' + row['Error'], file=sys.stderr)
        rows.append(row)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False, encoding='utf-8-sig')
    errors = sum(row['Status'] == 'error' for row in rows)
    nonfinite = sum(row['Status'] == 'nonfinite_metrics' for row in rows)
    print(f'CSV: {csv_path}\nImages: {len(rows)}; failed: {errors}; nonfinite metrics: {nonfinite}')
    return 1 if errors else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileExistsError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
