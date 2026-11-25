import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


GAMMA_COLUMN = 'Gamma wucha'
RESULT_COLUMNS = ['CustomScore', 'kmeans_label', 'Quality grade', 'ScoreStatus', 'ScoreError']
TIERS = ['Low quality sample', 'Medium quality sample', 'High quality sample']


def feature_columns(requested):
    selected = []
    for name in requested:
        name = name.strip()
        if name and name != GAMMA_COLUMN and name not in selected:
            selected.append(name)
    if not selected:
        raise ValueError('Select at least one image metric; Gamma wucha is added automatically')
    return selected + [GAMMA_COLUMN]


def score_and_cluster(frame, requested, id_column='File'):
    columns = feature_columns(requested)
    required = [id_column] + columns
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    if id_column in columns or any(name in RESULT_COLUMNS for name in columns):
        raise ValueError('Select image metrics rather than identifiers or score output columns')
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError('Image identifiers must be present and unique')
    numeric = frame[columns].apply(pd.to_numeric, errors='coerce')
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    valid = finite.all(axis=1)
    if valid.sum() < 3:
        raise ValueError('At least three rows with finite selected metrics and Gamma wucha are required')
    x = numeric.loc[valid].to_numpy(dtype=float)
    if not np.isfinite(np.std(x, axis=0)).all():
        raise ValueError('Selected metrics overflow during standardization')
    constant = [columns[i] for i in np.flatnonzero(np.std(x, axis=0) == 0)]
    if constant:
        raise ValueError('Selected metrics must vary across valid rows: ' + ', '.join(constant))

    standardizer = StandardScaler()
    standardized = standardizer.fit_transform(x)
    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(standardized)
    loadings = pca.components_[0]
    if abs(loadings[-1]) <= 1e-12:
        raise ValueError('Gamma wucha has zero weight in PC1; revise the selected metric set')
    denominator = loadings.sum()
    if abs(denominator) <= 1e-12:
        raise ValueError('PC1 weights sum to zero; this metric set cannot use sum-normalized weights')
    weights = loadings / denominator
    weighted = standardized @ weights
    if not np.isfinite(weighted).all() or np.ptp(weighted) <= 0:
        raise ValueError('Selected metrics do not produce finite varying scores')
    normalizer = MinMaxScaler()
    normalized_score = normalizer.fit_transform(weighted.reshape(-1, 1)).ravel()
    # Ships: scores = 1 - normalized_score.
    # Floating rafts: change the next line to scores = normalized_score.
    scores = 1 - normalized_score
    scores = np.clip(scores, 0, 1)
    if np.unique(scores).size < 3:
        raise ValueError('At least three distinct scores are required for three quality tiers')
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scores.reshape(-1, 1))
    centers = kmeans.cluster_centers_.ravel()
    order = np.argsort(centers)
    if not (np.diff(centers[order]) > 0).all():
        raise ValueError('Clustering did not produce three distinct quality centers')
    label_to_tier = {int(label): tier for label, tier in zip(order, TIERS)}

    result = frame.drop(columns=RESULT_COLUMNS, errors='ignore').copy()
    result['CustomScore'] = np.nan
    result['kmeans_label'] = pd.array([pd.NA] * len(result), dtype='Int64')
    result['Quality grade'] = pd.Series(pd.NA, index=result.index, dtype='string')
    result['ScoreStatus'] = 'invalid_features'
    result['ScoreError'] = ''
    result.loc[valid, 'CustomScore'] = scores
    result.loc[valid, 'kmeans_label'] = labels
    result.loc[valid, 'Quality grade'] = [label_to_tier[int(label)] for label in labels]
    result.loc[valid, 'ScoreStatus'] = 'scored'
    for position in np.flatnonzero(~valid):
        bad = [name for name, ok in zip(columns, finite[position]) if not ok]
        result.loc[result.index[position], 'ScoreError'] = 'Non-numeric or nonfinite: ' + ', '.join(bad)

    ordered_centers = centers[order]
    parameters = {
        'selected_image_metrics': columns[:-1],
        'mandatory_metric': GAMMA_COLUMN,
        'feature_columns': columns,
        'input_rows': int(len(frame)),
        'fitted_rows': int(valid.sum()),
        'excluded_rows': int((~valid).sum()),
        # For floating rafts, also remove "1 -" from this saved formula.
        'score_formula': '1 - MinMax(StandardScaler(X) @ sum_normalized_PC1_weights)',
        'standardization': {
            name: {'mean': float(mean), 'scale': float(scale)}
            for name, mean, scale in zip(columns, standardizer.mean_, standardizer.scale_)
        },
        'pc1_loadings': {name: float(value) for name, value in zip(columns, loadings)},
        'normalized_weights': {name: float(value) for name, value in zip(columns, weights)},
        'pc1_explained_variance_ratio': float(pca.explained_variance_ratio_[0]),
        'weighted_score_min': float(normalizer.data_min_[0]),
        'weighted_score_max': float(normalizer.data_max_[0]),
        'clustering': {
            'n_clusters': 3, 'random_state': 42, 'n_init': 10,
            'clusters': [
                {'label': int(label), 'quality': label_to_tier[int(label)], 'center': float(centers[label]),
                 'count': int(np.sum(labels == label))} for label in order
            ],
            'thresholds': [float((ordered_centers[0] + ordered_centers[1]) / 2),
                           float((ordered_centers[1] + ordered_centers[2]) / 2)],
        },
    }
    return result, parameters


def parse_args():
    parser = argparse.ArgumentParser(description='PCA quality scoring and three-tier clustering.')
    parser.add_argument('--input-csv', type=Path, required=True, help='Image-metric table containing Gamma wucha')
    parser.add_argument('--output-csv', type=Path, required=True, help='Table with scores and quality tiers')
    parser.add_argument('--features', nargs='+', required=True, help='Image metric column names; Gamma wucha is always included')
    parser.add_argument('--id-column', default='File', help='Image identifier column')
    parser.add_argument('--overwrite', action='store_true', help='Allow replacing output files')
    return parser.parse_args()


def main():
    args = parse_args()
    source, output = args.input_csv.absolute(), args.output_csv.absolute()
    parameters_path = output.with_suffix('.parameters.json')
    if source in {output, parameters_path} or output == parameters_path:
        raise ValueError('Input CSV, output CSV and parameter JSON must be separate files')
    if not args.overwrite and (output.exists() or parameters_path.exists()):
        raise FileExistsError('Output exists; choose a new path or use --overwrite')
    frame = pd.read_csv(source, dtype=str)
    frame.columns = frame.columns.str.strip()
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names after removing surrounding spaces')
    result, parameters = score_and_cluster(frame, args.features, args.id_column)
    parameters['id_column'] = args.id_column
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, encoding='utf-8-sig')
    parameters_path.write_text(json.dumps(parameters, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print('Features: ' + ', '.join(parameters['feature_columns']))
    print(f"Scored: {parameters['fitted_rows']}; invalid rows retained: {parameters['excluded_rows']}")
    print('Scores: ' + str(output))
    print('Parameters: ' + str(parameters_path))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
