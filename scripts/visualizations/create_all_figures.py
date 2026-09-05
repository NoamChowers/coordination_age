#!/usr/bin/env python3
"""Regenerate the five final figures and diagnostic tables from saved results.

No model fitting or notebook execution is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.visualizations.create_model_comparison import create_model_comparison
from scripts.visualizations.create_dataset_split_overview import age_summary, make_figure, missingness_summary

KRR_NAME = "RBF Kernel Ridge"
KRR_COLOR, OTHER_COLOR, ACCENT_COLOR = "#0072B2", "#A7A9AC", "#D55E00"


def create_diagnostics(output_dir: Path) -> list[dict]:
    """Plot held-out intervals and full-cohort LOOCV diagnostics."""
    PROJECT_ROOT = ROOT
    TRAIN_OUTPUT_DIR = ROOT / "outputs/train115_jkplus"
    FULL_ARTIFACT_DIR = ROOT / "artifacts/jackknife_plus_krr"
    FIGURE_DIR = output_dir
    figure_records = []

    def save_png(fig, filename, description, sources):
        fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", dpi=200)
        figure_records.append(dict(filename=filename, description=description,
                                   sources="; ".join(sources)))

    train_predictions = pd.read_csv(ROOT / 'outputs/model_comparison/comparison_oof_predictions.csv')
    train_metrics = pd.read_csv(ROOT / 'outputs/model_comparison/comparison_metrics.csv')
    test_predictions = pd.read_csv(TRAIN_OUTPUT_DIR / 'train115_jkplus_test29_predictions.csv')
    full_residuals = pd.read_csv(FULL_ARTIFACT_DIR / 'coordination_age_krr_jackknife_plus_residuals.csv')
    metadata = json.loads((PROJECT_ROOT / 'Data' / 'modeling_input_metadata.json').read_text())
    subject_order = np.asarray(metadata['subject_order'], dtype=int)

    assert len(train_predictions) == 115 * 6
    assert train_predictions.groupby('procedure').size().eq(115).all()
    assert train_metrics['procedure'].is_unique and len(train_metrics) == 6
    assert len(test_predictions) == 29 and test_predictions['SubjectNumber'].is_unique
    assert len(full_residuals) == 144
    assert np.array_equal(full_residuals['loo_position'], np.arange(144))
    assert len(subject_order) == 144

    full_predictions = pd.DataFrame({
        'SubjectNumber': subject_order[full_residuals['loo_position'].to_numpy(int)],
        'chronological_age_years': full_residuals['observed_y'].to_numpy(float),
        'oof_prediction_years': full_residuals['loo_prediction'].to_numpy(float),
        'residual_years': full_residuals['signed_residual'].to_numpy(float),
        'absolute_error_years': full_residuals['absolute_residual'].to_numpy(float),
    })


    def point_metrics(observed, predicted):
        observed = np.asarray(observed, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        return {
            'n': len(observed),
            'rmse_years': float(np.sqrt(mean_squared_error(observed, predicted))),
            'mae_years': float(mean_absolute_error(observed, predicted)),
            'r_squared': float(r2_score(observed, predicted)),
        }

    train_krr = train_predictions.query('procedure == @KRR_NAME').copy()
    summary_metrics = pd.DataFrame([
        {'role': 'Train LOOCV (KRR)', **point_metrics(train_krr['chronological_age_years'], train_krr['oof_prediction_years'])},
        {'role': 'Held-out test (train-fitted KRR)', **point_metrics(test_predictions['chronological_age_years'], test_predictions['predicted_age_years'])},
        {'role': 'Full-data LOOCV (KRR)', **point_metrics(full_predictions['chronological_age_years'], full_predictions['oof_prediction_years'])},
    ])
    summary_metrics['empirical_interval_coverage'] = [np.nan, test_predictions['covered'].mean(), np.nan]
    summary_metrics['mean_interval_width_years'] = [np.nan, test_predictions['jkplus_width'].mean(), np.nan]
    summary_metrics.to_csv(FIGURE_DIR / 'visualization_summary_metrics.csv', index=False)


    test_plot = test_predictions.sort_values('chronological_age_years').copy()
    lower_error = test_plot['predicted_age_years'] - test_plot['jkplus_lower']
    upper_error = test_plot['jkplus_upper'] - test_plot['predicted_age_years']
    x_limits = [test_plot['chronological_age_years'].min() - 0.4, test_plot['chronological_age_years'].max() + 0.4]
    y_limits = [min(test_plot['jkplus_lower'].min() - 0.5, x_limits[0]), max(test_plot['jkplus_upper'].max() + 0.5, x_limits[1])]
    coverage = test_plot['covered'].mean()

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.errorbar(test_plot['chronological_age_years'], test_plot['predicted_age_years'],
                yerr=np.vstack([lower_error, upper_error]), fmt='o', ms=5, color=KRR_COLOR,
                ecolor='0.58', elinewidth=1.1, capsize=2, alpha=0.85, label='Prediction + 95% JK+ interval')
    ax.plot(x_limits, x_limits, '--', color='black', lw=1.1, label='Identity')
    ax.set(xlim=x_limits, ylim=y_limits, xlabel='Chronological age (years)', ylabel='Predicted age (years)')
    ax.legend(frameon=False)

    fig.tight_layout()
    save_png(fig, 'figure_3_test_jkplus_interval_diagnostics.png',
             'Held-out test age predictions and 95% jackknife+ prediction intervals for chronological age.',
             ['outputs/train115_jkplus/train115_jkplus_test29_predictions.csv'])
    plt.close(fig)

    observed = full_predictions['chronological_age_years'].to_numpy(float)
    predicted = full_predictions['oof_prediction_years'].to_numpy(float)
    slope, intercept = np.polyfit(predicted, observed, 1)
    limits = [min(observed.min(), predicted.min()) - 0.4, max(observed.max(), predicted.max()) + 0.4]
    grid = np.linspace(*limits, 100)
    assert observed.min() >= 4 and observed.max() < 19
    age_bin_frame = pd.DataFrame({
        'age_bin': pd.cut(observed, bins=[4, 8, 12, 16, 19],
                          labels=['[4, 8)', '[8, 12)', '[12, 16)', '[16, 19)'],
                          right=False, include_lowest=True),
        'squared_error': (observed - predicted) ** 2,
        'residual': observed - predicted,
    })
    age_bin_rmse = (age_bin_frame.groupby('age_bin', observed=True, sort=True)
                    .agg(n=('squared_error', 'size'),
                         rmse_years=('squared_error', lambda values: np.sqrt(values.mean())),
                         bias_years=('residual', 'mean'))
                    .reset_index())
    overall_rmse = np.sqrt(np.mean((observed - predicted) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    ax = axes[0]
    ax.scatter(predicted, observed, s=29, color=KRR_COLOR, alpha=0.7, edgecolor='white', linewidth=0.35)
    ax.plot(limits, limits, '--', color='black', lw=1.1, label='Identity')
    ax.plot(grid, intercept + slope * grid, color=ACCENT_COLOR, lw=1.6,
            label=f'Calibration: age = {intercept:.2f} + {slope:.2f} × prediction')
    ax.set(xlim=limits, ylim=limits, xlabel='OOF predicted age (years)', ylabel='Chronological age (years)')
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)

    ax = axes[1]
    positions = np.arange(len(age_bin_rmse))
    bar_width = 0.36
    rmse_bars = ax.bar(positions - bar_width / 2, age_bin_rmse['rmse_years'], width=bar_width,
                       color=KRR_COLOR, alpha=0.82,
                       label=f'RMSE (overall = {overall_rmse:.2f} years)')
    bias_bars = ax.bar(positions + bar_width / 2, age_bin_rmse['bias_years'], width=bar_width,
                       color=OTHER_COLOR, edgecolor='0.35', alpha=0.9,
                       label='Bias: mean(age − prediction)')
    ax.axhline(0, color='black', lw=0.9)
    ax.bar_label(rmse_bars, labels=[f'{rmse:.2f}' for rmse in age_bin_rmse['rmse_years']], padding=3)
    ax.bar_label(bias_bars, labels=[f'{bias:+.2f}' for bias in age_bin_rmse['bias_years']], padding=3)
    age_bin_tick_labels = [f'{age_bin}\n(n={n_subjects})'
                           for age_bin, n_subjects in zip(age_bin_rmse['age_bin'].astype(str), age_bin_rmse['n'])]
    ax.set_xticks(positions, age_bin_tick_labels)
    ax.set(xlabel='Chronological-age bin (years)', ylabel='LOOCV error (years)',
           ylim=(min(-0.8, age_bin_rmse['bias_years'].min() - 0.2),
                 age_bin_rmse['rmse_years'].max() + 0.5))
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.18)

    fig.tight_layout()
    save_png(fig, 'figure_4_full_data_loocv_krr_diagnostics.png',
             'Calibration plus chronological-age-bin RMSE and bias for full-data KRR OOF predictions.',
             ['artifacts/jackknife_plus_krr/coordination_age_krr_jackknife_plus_residuals.csv',
              'Data/modeling_input_metadata.json'])
    plt.close(fig)

    age_bin_rmse.to_csv(FIGURE_DIR / "full_data_age_bin_metrics.csv", index=False)
    return figure_records


def create_ablation(output_dir: Path) -> dict:
    """Plot saved single omissions and the cumulative training-selected path."""
    FIGURE_DIR = output_dir
    stage_a_report = pd.read_csv(ROOT / "outputs/task_ablation/leave_one_experiment_out_metrics.csv")
    path_report = pd.read_csv(ROOT / "outputs/task_ablation/backward_path_metrics.csv")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # Stage A: each bar compares one omitted experiment with the all-experiments baseline.
    leave_one_plot = stage_a_report.query(
        "omitted_experiment != 'None (all experiments)'"
    ).copy().sort_values('delta_train_oof_rmse')
    x = np.arange(len(leave_one_plot))
    width = 0.36
    axes[0].bar(
        x - width / 2, leave_one_plot['delta_train_oof_rmse'], width,
        label='Train OOF', color='#2878B5'
    )
    axes[0].bar(
        x + width / 2, leave_one_plot['delta_test_rmse'], width,
        label='Test', color='#F28E2B'
    )
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(leave_one_plot['omitted_experiment'], rotation=35, ha='right')
    axes[0].set_ylabel('Δ RMSE relative to all-experiments model (years)')
    axes[0].set_xlabel('Experiment omitted (one at a time)')
    axes[0].legend(title='Leave-one-experiment-out', frameon=False)

    # Stage B: step 0 is all experiments; each subsequent point adds one omission.
    path_plot = path_report.sort_values('step').copy()
    axes[1].plot(
        path_plot['step'], path_plot['train_oof_rmse'],
        marker='o', linewidth=2, label='Train OOF', color='#2878B5'
    )
    axes[1].plot(
        path_plot['step'], path_plot['test_rmse'],
        marker='s', linewidth=2, label='Test', color='#F28E2B'
    )
    step_labels = [
        '0\nRemoved: none' if row.step == 0
        else f'{int(row.step)}\nRemoved: {row.removed_at_step}'
        for row in path_plot.itertuples(index=False)
    ]
    axes[1].set_xticks(path_plot['step'])
    axes[1].set_xticklabels(step_labels, rotation=28, ha='right')
    axes[1].set_xlabel('Number of experiments dropped')
    axes[1].set_ylabel('RMSE (years)')
    axes[1].legend(frameon=False)

    fig.tight_layout()
    figure_path = FIGURE_DIR / 'task_ablation_rmse.png'
    fig.savefig(figure_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


    return dict(filename=figure_path.name, description="Task omission effects and cumulative backward elimination.", sources="outputs/task_ablation/leave_one_experiment_out_metrics.csv; outputs/task_ablation/backward_path_metrics.csv")


def create_all_figures(output_dir: str | Path = ROOT / "outputs/visualizations") -> Path:
    """Write all five figures to one directory and return their manifest path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"}):
        X = pd.read_csv(ROOT / "Data/X.csv")
        y = pd.read_csv(ROOT / "Data/y.csv")
        split = pd.read_csv(ROOT / "Data/train_test_split.csv")
        cohort, counts = age_summary(y, split)
        counts.to_csv(output_dir / "dataset_split_age_bin_counts.csv", index=False)
        missingness_summary(X).to_csv(output_dir / "dataset_missingness_by_group.csv", index=False)
        make_figure(cohort, counts, output_dir / "dataset_split_overview.png")
        create_model_comparison(output_dir=output_dir)
        records = pd.read_csv(output_dir / "figure_manifest.csv").query(
            "filename == 'figure_1_train_loocv_model_comparison.png'"
        ).to_dict("records")
        records.append(dict(filename="dataset_split_overview.png",
                            description="Chronological-age distributions in the saved training/test split.",
                            sources="Data/y.csv; Data/train_test_split.csv"))
        records.extend(create_diagnostics(output_dir))
        records.append(create_ablation(output_dir))
    manifest_path = output_dir / "figure_manifest.csv"
    pd.DataFrame(records).sort_values("filename").to_csv(manifest_path, index=False)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/visualizations")
    args = parser.parse_args()
    print(create_all_figures(args.output_dir))


if __name__ == "__main__":
    main()
