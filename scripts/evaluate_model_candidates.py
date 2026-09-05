#!/usr/bin/env python3
"""Rerun selected training-set candidates, resuming verified outer-fold results.

Comparison tables retain saved predictions for candidates not requested.
KRR production artifacts are fitted separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
import sklearn
from joblib import Memory, parallel_config
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, ParameterGrid
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.krr_pipeline import make_krr_spec

SEED, INNER_FOLDS = 42, 5
KRR_NAME = "RBF Kernel Ridge"
RF_GRID = {
    "model__max_depth": [4, 7, None],
    "model__min_samples_leaf": [1, 3, 5],
    "model__max_features": [0.5, 1.0],
}


def candidate_specs(features):
    """Final candidate pipelines and training-only hyperparameter grids."""
    krr = make_krr_spec(features, sex_column="Sex_0Female_1Male")
    estimators = {
        "ElasticNet": (ElasticNet(max_iter=100_000, random_state=SEED), {
            "model__alpha": np.logspace(-4, 1, 25),
            "model__l1_ratio": [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        }),
        "Lasso": (Lasso(max_iter=100_000, random_state=SEED), {
            "model__alpha": np.logspace(-4, 1, 25),
        }),
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 4, 29)}),
        "OLS": (LinearRegression(), None),
        "Random Forest": (RandomForestRegressor(
            n_estimators=200, max_depth=7, random_state=SEED,
            n_jobs=1,
        ), RF_GRID),
    }
    specs = {KRR_NAME: {"estimator": krr["estimator"], "grid": krr["param_grid"]}}
    for name, (model, grid) in estimators.items():
        steps = [(key, clone(step)) for key, step in krr["estimator"].steps[:-1]]
        specs[name] = {"estimator": Pipeline(steps + [("model", model)]), "grid": grid}
    return specs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_training_data():
    X = pd.read_csv(ROOT / "Data/X.csv")
    y = pd.read_csv(ROOT / "Data/y.csv")["AgeInYears"]
    ids = json.loads((ROOT / "Data/modeling_input_metadata.json").read_text())["subject_order"]
    split = pd.read_csv(ROOT / "Data/train_test_split.csv")
    if len(X) != len(ids) or len(y) != len(ids):
        raise ValueError("Input rows and participant metadata do not align.")
    if not np.array_equal(np.sort(split.production_row_position), np.arange(len(X))):
        raise ValueError("Split must cover every input row exactly once.")
    if not np.array_equal(np.asarray(ids)[split.production_row_position], split.SubjectNumber):
        raise ValueError("Split participant IDs disagree with input row metadata.")
    training = split.query("partition == 'train'").sort_values("partition_position")
    if len(training) != 115 or training.SubjectNumber.duplicated().any():
        raise ValueError("Expected the historical 115-participant training set.")
    if training.partition_position.tolist() != list(range(115)):
        raise ValueError("Training partition order is incomplete.")
    X = X.iloc[training.production_row_position].copy()
    y = y.iloc[training.production_row_position].copy()
    X.index = y.index = pd.Index(training.SubjectNumber, name="SubjectNumber")
    return X, y


def configuration(X, y, name, spec):
    grid = list(ParameterGrid(spec["grid"])) if spec["grid"] else [{}]
    grid = [{k: v.item() if isinstance(v, np.generic) else v for k, v in p.items()} for p in grid]
    return {
        "procedure": name, "seed": SEED, "inner_folds": INNER_FOLDS,
        "scoring": "neg_mean_squared_error", "outer_cv": "LeaveOneOut",
        "features": X.columns.tolist(), "participant_ids": X.index.tolist(),
        "training_data_sha256": hashlib.sha256(
            pd.concat([X, y.rename("AgeInYears")], axis=1).to_csv().encode()
        ).hexdigest(),
        "configurations": grid,
        "implementation_sha256": sha256(__file__),
        "preprocessing_sha256": sha256(ROOT / "src/krr_pipeline.py"),
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
    }


def evaluate_candidate(X, y, name, spec, output_dir, n_jobs):
    folder = output_dir / name.lower().replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path, folds_path = folder / "configuration.json", folder / "folds.json"
    expected = configuration(X, y, name, spec)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != expected:
            raise ValueError(f"Stale configuration in {folder}; use a new --output-dir.")
    elif folds_path.exists():
        raise ValueError(f"Checkpoint has no configuration: {folds_path}")
    else:
        write_json(manifest_path, expected)
    records = json.loads(folds_path.read_text()) if folds_path.exists() else []
    if len(records) > len(X):
        raise ValueError("Checkpoint contains more folds than training participants.")
    for i, record in enumerate(records):
        if (record["outer_fold"] != i + 1 or record["SubjectNumber"] != int(X.index[i])
                or record["chronological_age_years"] != float(y.iloc[i])):
            raise ValueError("Checkpoint folds do not match the training participants.")
    print(f"{name}: reusing {len(records)}/{len(X)} folds; grid size "
          f"{len(expected['configurations'])}", flush=True)
    start, initial = time.perf_counter(), len(records)
    for position, (fit_idx, held_idx) in enumerate(LeaveOneOut().split(X)):
        if position < initial:
            continue
        fold_start = time.perf_counter()
        with TemporaryDirectory(prefix="coordination_candidate_") as cache:
            estimator = clone(spec["estimator"]).set_params(memory=Memory(cache, verbose=0))
            if spec["grid"]:
                search = GridSearchCV(
                    estimator, spec["grid"], scoring="neg_mean_squared_error",
                    cv=KFold(INNER_FOLDS, shuffle=True, random_state=SEED),
                    refit=True, n_jobs=n_jobs, pre_dispatch=n_jobs, error_score="raise",
                )
                with parallel_config(backend="threading", n_jobs=n_jobs):
                    search.fit(X.iloc[fit_idx], y.iloc[fit_idx])
                fitted, parameters = search.best_estimator_, search.best_params_
                inner_rmse = float(np.sqrt(-search.best_score_))
            else:
                fitted = estimator.fit(X.iloc[fit_idx], y.iloc[fit_idx])
                parameters, inner_rmse = {}, None
            prediction = float(fitted.predict(X.iloc[held_idx])[0])
        parameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in parameters.items()}
        records.append({
            "outer_fold": position + 1, "SubjectNumber": int(X.index[position]),
            "procedure": name, "chronological_age_years": float(y.iloc[position]),
            "oof_prediction_years": prediction, "inner_cv_rmse_years": inner_rmse,
            "best_parameters": parameters, "fit_seconds": time.perf_counter() - fold_start,
        })
        write_json(folds_path, records)
        elapsed = time.perf_counter() - start
        eta = elapsed / (len(records) - initial) * (len(X) - len(records))
        print(f"{name} {len(records):3d}/{len(X)}; elapsed={elapsed/60:.1f} min; "
              f"ETA={eta/60:.1f} min", flush=True)
    return pd.DataFrame(records), manifest_path


def combine_results(baseline, updates, y):
    if baseline.duplicated(["procedure", "SubjectNumber"]).any():
        raise ValueError("Duplicate participant predictions in baseline.")
    for _, group in baseline.groupby("procedure"):
        ordered = group.set_index("SubjectNumber").reindex(y.index)
        if len(group) != len(y) or not np.allclose(ordered.chronological_age_years, y):
            raise ValueError("Baseline outcomes/participants differ from current training data.")
        if not np.isfinite(ordered.oof_prediction_years).all():
            raise ValueError("Baseline predictions are incomplete.")
    columns = ["SubjectNumber", "procedure", "chronological_age_years", "oof_prediction_years"]
    predictions = pd.concat([
        baseline.loc[~baseline.procedure.isin(updates), columns],
        *[frame[columns] for frame in updates.values()],
    ], ignore_index=True)
    metrics = []
    for name, group in predictions.groupby("procedure", sort=False):
        observed, predicted = group.chronological_age_years, group.oof_prediction_years
        metrics.append({"procedure": name, "n": len(group),
                        "rmse_years": float(np.sqrt(mean_squared_error(observed, predicted))),
                        "mae_years": float(mean_absolute_error(observed, predicted)),
                        "r_squared": float(r2_score(observed, predicted))})
    metrics = pd.DataFrame(metrics).sort_values(["rmse_years", "mae_years"]).reset_index(drop=True)
    metrics.insert(0, "rmse_rank", np.arange(1, len(metrics) + 1))
    return predictions, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["Random Forest"])
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/model_comparison")
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    X, y = load_training_data()
    specs = candidate_specs(X.columns)
    if len(set(args.models)) != len(args.models) or set(args.models) - specs.keys():
        parser.error(f"Choose distinct model names from {list(specs)}")
    baseline_path = args.output_dir / "comparison_oof_predictions.csv"
    if not baseline_path.exists():
        baseline_path = ROOT / "outputs/model_comparison/comparison_oof_predictions.csv"
    baseline = (pd.read_csv(baseline_path) if baseline_path.exists() else
                pd.DataFrame(columns=["SubjectNumber", "procedure",
                                      "chronological_age_years", "oof_prediction_years"]))
    if set(specs) - set(args.models) - set(baseline.procedure):
        parser.error("No saved predictions for all candidates; supply all six --models names.")
    if not baseline.empty:
        combine_results(baseline, {}, y)  # Validate before any expensive work.
    updates, manifests = {}, {}
    for name in args.models:
        frame, manifest = evaluate_candidate(X, y, name, specs[name], args.output_dir, args.n_jobs)
        updates[name] = frame
        manifests[name] = str(manifest.relative_to(args.output_dir))
    baseline_digest = sha256(baseline_path) if baseline_path.exists() else None
    predictions, metrics = combine_results(baseline, updates, y)
    predictions.to_csv(args.output_dir / "comparison_oof_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "comparison_metrics.csv", index=False)
    write_json(args.output_dir / "comparison_manifest.json", {
        "analysis": "Final training-set nested-LOOCV candidate comparison",
        "rerun_models": args.models, "model_configurations": manifests,
        "reused_models": sorted(set(baseline.procedure) - set(args.models)),
        "baseline_sha256_before_update": baseline_digest,
        "test_outcomes_used_for_tuning": False,
    })
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
