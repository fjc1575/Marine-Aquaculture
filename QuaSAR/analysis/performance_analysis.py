"""Fit the QuaSAR quadratic performance model and analyze quality proportions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PARAMETERS = ('alpha', 'beta', 'gamma', 'delta', 'eta', 'c')
FORMULA = 'alpha*Q_h + beta*Q_h**2 + gamma*Q_m + delta*Q_m**2 + eta*Q_h*Q_m + c'


def design_matrix(high, medium):
    high, medium = np.broadcast_arrays(np.asarray(high, float), np.asarray(medium, float))
    return np.stack((high, high**2, medium, medium**2, high*medium, np.ones_like(high)), axis=-1)


def predict(theta, high, medium):
    return design_matrix(high, medium) @ np.asarray(theta)


def marginal_effects(theta, high, medium):
    alpha, beta, gamma, delta, eta, _ = theta
    high, medium = np.broadcast_arrays(np.asarray(high, float), np.asarray(medium, float))
    me_high = alpha + 2*beta*high + eta*medium
    me_medium = gamma + 2*delta*medium + eta*high
    return me_high, me_medium, -(me_high + me_medium)


def fit_model(frame):
    x = design_matrix(frame.Q_h, frame.Q_m)
    y = frame.performance.to_numpy()
    theta, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    if rank != 6:
        raise ValueError(f'Quadratic model needs six independent design columns; rank={rank}')
    residual = y - x @ theta
    sse = float(residual @ residual)
    total_variance = float(np.sum((y - y.mean())**2))
    dof = len(y) - rank
    covariance = (sse/dof) * np.linalg.pinv(x.T @ x) if dof > 0 else None
    diagnostics = dict(n_observations=len(y), rank=int(rank), residual_dof=int(dof),
        r_squared=1 - sse/total_variance if np.ptp(y) > 0 and total_variance > 0 else None,
        rmse=float(np.sqrt(np.mean(residual**2))), mae=float(np.mean(np.abs(residual))),
        residual_sum_squares=sse, condition_number=float(singular_values[0]/singular_values[-1]),
        evaluation='in-sample fit diagnostics, not held-out performance')
    return theta, covariance, diagnostics


def maximize_on_simplex(theta):
    alpha, beta, gamma, delta, eta, _ = theta
    tolerance = 1e-12 * max(1., float(np.max(np.abs(theta[:5]))))
    candidates = []
    def add(high, medium, location):
        if not np.isfinite([high, medium]).all() or min(high, medium, 1-high-medium) < -1e-9:
            return
        high, medium = max(0., float(high)), max(0., float(medium))
        if high + medium > 1:
            high, medium = high/(high+medium), medium/(high+medium)
        if any(np.allclose([high, medium], [r['Q_h'], r['Q_m']], atol=1e-10, rtol=0) for r in candidates):
            return
        candidates.append(dict(Q_h=high, Q_m=medium, Q_l=max(0., 1-high-medium),
            predicted_performance=float(predict(theta, high, medium)), location=location))
    for high, medium in ((0., 0.), (1., 0.), (0., 1.)):
        add(high, medium, 'vertex')
    if abs(beta) > tolerance:
        add(-alpha/(2*beta), 0., 'Q_m=0 edge')
    if abs(delta) > tolerance:
        add(0., -gamma/(2*delta), 'Q_h=0 edge')
    edge_a, edge_b = beta + delta - eta, alpha - gamma - 2*delta + eta
    if abs(edge_a) > tolerance:
        high = -edge_b/(2*edge_a)
        add(high, 1-high, 'Q_l=0 edge')
    hessian = np.array([[2*beta, eta], [eta, 2*delta]])
    if np.linalg.matrix_rank(hessian, tol=tolerance) == 2:
        high, medium = np.linalg.solve(hessian, -np.array([alpha, gamma]))
        add(high, medium, 'interior stationary point')
    add(1/3, 1/3, 'simplex center')
    table = pd.DataFrame(candidates).sort_values('predicted_performance', ascending=False).reset_index(drop=True)
    best = table.iloc[0].to_dict()
    equal_best = np.isclose(table.predicted_performance, best['predicted_performance'], rtol=1e-10, atol=1e-12)
    best['equal_best_candidate_count'] = int(equal_best.sum())
    best['method'] = 'global quadratic candidate enumeration on the closed simplex'
    return best, table


def load_observations(paths, metric, high_column, medium_column, low_column,
                      filter_column=None, filter_value=None):
    frames = []
    for path in paths:
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip()
        if filter_column is not None:
            if filter_column not in raw:
                raise ValueError(f'{path}: missing filter column {filter_column}')
            raw = raw.loc[raw[filter_column].astype(str) == filter_value]
        required = [high_column, medium_column, metric]
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f'{path}: missing columns {missing}')
        frame = pd.DataFrame({
            'Q_h': pd.to_numeric(raw[high_column], errors='raise'),
            'Q_m': pd.to_numeric(raw[medium_column], errors='raise'),
            'performance': pd.to_numeric(raw[metric], errors='raise')})
        frame['Q_l'] = pd.to_numeric(raw[low_column], errors='raise') if low_column in raw else 1-frame.Q_h-frame.Q_m
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f'{path}: missing or non-finite ratios/performance values')
        ratios = frame[['Q_h', 'Q_m', 'Q_l']].to_numpy()
        if (ratios < -1e-9).any() or (ratios > 1+1e-9).any() or not np.allclose(ratios.sum(axis=1), 1, rtol=0, atol=1e-8):
            raise ValueError(f'{path}: ratios must be in [0,1] and sum to 1; use fractions, not percentages')
        frame['source_file'] = str(path)
        if 'Source' in raw:
            frame['source_group'] = raw['Source'].astype(str)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if len(result) < 6:
        raise ValueError('At least six independent proportion observations are required')
    return result


def surface_table(theta, resolution):
    points = [(h/resolution, m/resolution)
              for h in range(resolution+1) for m in range(resolution+1-h)]
    frame = pd.DataFrame(points, columns=['Q_h', 'Q_m'])
    frame['Q_l'] = np.maximum(0, 1-frame.Q_h-frame.Q_m)
    frame['predicted_performance'] = predict(theta, frame.Q_h, frame.Q_m)
    frame['ME_h'], frame['ME_m'], frame['ME_l'] = marginal_effects(theta, frame.Q_h, frame.Q_m)
    return frame


def marginal_curves(theta, reference, points):
    high, medium = reference
    weight_high = high/(high+medium) if high+medium > 1e-12 else 0.5
    weight_medium = 1-weight_high
    high_axis = np.linspace(0, max(0., 1-medium), points)
    medium_axis = np.linspace(0, max(0., 1-high), points)
    low_axis = np.linspace(0, 1, points)
    paths = [('high', high_axis, np.full(points, medium)),
             ('medium', np.full(points, high), medium_axis),
             ('low', (1-low_axis)*weight_high, (1-low_axis)*weight_medium)]
    tables = []
    for index, (tier, h, m) in enumerate(paths):
        me_high, me_medium, me_low = marginal_effects(theta, h, m)
        effect = (me_high, me_medium, me_low)[index]
        # A derivative along the fixed H:M path differs from the paper's ME_l.
        derivative = -weight_high*me_high-weight_medium*me_medium if tier == 'low' else effect
        tables.append(pd.DataFrame(dict(tier=tier, ratio=(h, m, low_axis)[index],
            Q_h=h, Q_m=m, Q_l=np.maximum(0., 1-h-m), marginal_effect=effect,
            path_derivative=derivative, predicted_performance=predict(theta, h, m))))
    return pd.concat(tables, ignore_index=True), dict(high_weight=weight_high, medium_weight=weight_medium,
        pure_low_reference_fallback=bool(high+medium <= 1e-12))


def save_plots(observations, surface, curves, optimum, label, output):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11, 'axes.labelweight': 'bold', 'axes.spines.top': False,
        'axes.spines.right': False, 'savefig.dpi': 300, 'pdf.fonttype': 42})
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(output / f'{name}.{suffix}', bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    plot = ax.tricontourf(surface.Q_h, surface.Q_m, surface.predicted_performance, levels=40, cmap='viridis')
    fig.colorbar(plot, ax=ax, label=f'Predicted {label}')
    ax.scatter(observations.Q_h, observations.Q_m, s=12, facecolors='none', edgecolors='white', linewidths=.5)
    ax.scatter([optimum['Q_h']], [optimum['Q_m']], s=100, color='#D55E00', marker='*',
               edgecolor='black', linewidth=.6, label='Model optimum', zorder=5)
    ax.set(xlabel='High-quality proportion', ylabel='Medium-quality proportion',
           xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    save(fig, 'performance_surface')
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for tier, color, name in [('high', '#0072B2', 'High'), ('medium', '#009E73', 'Medium'),
                             ('low', '#D55E00', 'Low (paper definition)')]:
        row = curves.loc[curves.tier == tier]
        ax.plot(row.ratio, row.marginal_effect, label=name, color=color, linewidth=2)
    ax.axhline(0, color='0.45', linewidth=.8, linestyle='--')
    ax.set(xlabel='Quality-tier proportion', ylabel=f'Marginal effect on {label}', xlim=(0, 1))
    ax.legend(fontsize=9)
    save(fig, 'marginal_effects')
    fig, ax = plt.subplots(figsize=(5, 4.5))
    if 'source_group' in observations and observations.source_group.nunique() > 1:
        for group, row in observations.groupby('source_group'):
            ax.scatter(row.performance, row.fitted_performance, s=22, alpha=.7, label=group)
        ax.legend(fontsize=9)
    else:
        ax.scatter(observations.performance, observations.fitted_performance, s=22, color='#0072B2', alpha=.7)
    lower = min(observations.performance.min(), observations.fitted_performance.min())
    upper = max(observations.performance.max(), observations.fitted_performance.max())
    ax.plot([lower, upper], [lower, upper], color='0.45', linestyle='--', linewidth=1)
    ax.set(xlabel=f'Observed {label}', ylabel=f'Fitted {label}')
    save(fig, 'observed_vs_fitted')


def analyze(args):
    observations = load_observations(args.inputs, args.metric, args.high_column,
        args.medium_column, args.low_column, args.filter_column, args.filter_value)
    theta, covariance, diagnostics = fit_model(observations)
    optimum, candidates = maximize_on_simplex(theta)
    surface = surface_table(theta, args.resolution)
    curves, path_weights = marginal_curves(theta, (optimum['Q_h'], optimum['Q_m']), args.curve_points)
    observations['fitted_performance'] = predict(theta, observations.Q_h, observations.Q_m)
    observations['residual'] = observations.performance-observations.fitted_performance
    observations['ME_h'], observations['ME_m'], observations['ME_l'] = marginal_effects(
        theta, observations.Q_h, observations.Q_m)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    standard_errors = np.sqrt(np.maximum(0., np.diag(covariance))) if covariance is not None else [None]*6
    pd.DataFrame(dict(parameter=PARAMETERS, estimate=theta, standard_error=standard_errors)).to_csv(
        output / 'coefficients.csv', index=False)
    for name, frame in [('observations', observations), ('surface', surface),
                        ('marginal_curves', curves), ('optimization_candidates', candidates)]:
        frame.to_csv(output / f'{name}.csv', index=False)
    pd.DataFrame([optimum]).to_csv(output / 'recommended_ratio.csv', index=False)
    summary = dict(formula=FORMULA, metric=args.metric, metric_label=args.metric_label or args.metric,
        coefficients=dict(zip(PARAMETERS, map(float, theta))), diagnostics=diagnostics,
        recommended_ratio=optimum, input_files=[str(path) for path in args.inputs],
        pooling='all retained input rows enter one unweighted least-squares fit',
        filter=dict(column=args.filter_column, value=args.filter_value),
        ME_h='alpha+2*beta*Q_h+eta*Q_m', ME_m='gamma+2*delta*Q_m+eta*Q_h',
        ME_l='-(ME_h+ME_m), as defined in the paper',
        low_path_derivative='-high_weight*ME_h-medium_weight*ME_m', low_path_weights=path_weights,
        performance_scale='unchanged from the input metric', figures_created=not args.no_plots)
    if 'source_group' in observations:
        summary['source_group_counts'] = observations.source_group.value_counts().to_dict()
    (output / 'model.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if not args.no_plots:
        save_plots(observations, surface, curves, optimum, args.metric_label or args.metric, output)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', nargs='+', type=Path, required=True,
                        help='CSV tables; multiple tables are explicitly pooled into one fit')
    parser.add_argument('--metric', required=True, help='performance column, e.g. AP50 or mIou')
    parser.add_argument('--metric-label', help='axis label; defaults to the selected column name')
    parser.add_argument('--high-column', default='Q_h')
    parser.add_argument('--medium-column', default='Q_m')
    parser.add_argument('--low-column', default='Q_l', help='derived from high/medium if absent')
    parser.add_argument('--filter-column')
    parser.add_argument('--filter-value')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=100, help='simplex grid subdivisions')
    parser.add_argument('--curve-points', type=int, default=201)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if (args.filter_column is None) != (args.filter_value is None):
        parser.error('--filter-column and --filter-value must be supplied together')
    if args.resolution < 2 or args.curve_points < 2:
        parser.error('resolution and curve-points must be at least 2')
    summary = analyze(args)
    print(json.dumps(dict(recommended_ratio=summary['recommended_ratio'],
        diagnostics=summary['diagnostics'], output=str(args.output_dir)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
