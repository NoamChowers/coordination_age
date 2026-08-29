#!/usr/bin/env python3
"""Fit and serialize the final KRR jackknife+ artifacts.

The caller supplies an already validated modeling matrix ``X`` and aligned
target ``y``. In particular, upstream domain rules such as the n>=5 trial
mask, invalid-timing cleanup, and approved-feature selection are deliberately
outside this script. Statistical preprocessing that must be learned without
leakage remains inside every fitted sklearn pipeline: median imputation,
winsorization, feature-wise Yeo-Johnson transformation, and standardization.

For ``n`` observations the script fits and saves exactly ``n + 1`` models:

* ``n`` leave-one-out KRR pipelines, each with inner-CV hyperparameter tuning;
* one production KRR pipeline tuned and refit on all observations.

The absolute LOO residuals are saved separately and aligned to the model list
by the zero-based ``loo_position`` column.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Mapping, Sequence

from joblib import Memory
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.krr_pipeline import make_krr_spec


ARTIFACT_VERSION = 1
DEFAULT_SEED = 42
DEFAULT_INNER_FOLDS = 5
DEFAULT_N_JOBS = 6
DEFAULT_SCORING = "neg_mean_squared_error"
DEFAULT_SEX_COLUMN = "Sex_0Female_1Male"

LOO_MODELS_FILENAME = "coordination_age_krr_jackknife_plus_loo_models.pkl"
PRODUCTION_MODEL_FILENAME = "coordination_age_krr_jackknife_plus_production_model.pkl"
RESIDUALS_FILENAME = "coordination_age_krr_jackknife_plus_residuals.csv"
SELECTIONS_FILENAME = "coordination_age_krr_jackknife_plus_selections.csv"
MANIFEST_FILENAME = "coordination_age_krr_jackknife_plus_manifest.json"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Convert numpy scalar hyperparameters into serializable values."""

    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in parameters.items()
    }


def validate_xy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sex_column: str,
    inner_folds: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the already domain-cleaned modeling inputs."""

    if inner_folds < 2:
        raise ValueError("inner_folds must be at least 2.")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named columns.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series aligned to X.index.")
    if len(X) != len(y):
        raise ValueError(f"X and y have different lengths: {len(X)} and {len(y)}.")
    if len(X) <= inner_folds:
        raise ValueError(
            "LOO training sets must contain at least inner_folds observations; "
            f"received n={len(X)}, inner_folds={inner_folds}."
        )
    if not X.index.is_unique or not y.index.is_unique:
        raise ValueError("X and y indices must be unique.")
    if not X.index.equals(y.index):
        raise ValueError("X and y indices must be identically ordered.")
    if not X.columns.is_unique:
        raise ValueError("X feature names must be unique.")
    if sex_column not in X.columns:
        raise ValueError(f"Missing required sex predictor: {sex_column}")
    if any(str(column).startswith("n_trials_") for column in X.columns):
        raise ValueError(
            "n_trials_* diagnostics must be removed before calling this script."
        )
    nonnumeric = X.select_dtypes(exclude="number").columns.tolist()
    if nonnumeric:
        raise ValueError(f"All modeling features must be numeric: {nonnumeric}")
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("X may contain NaN for pipeline imputation, but not infinity.")
    if X.isna().all(axis=0).any():
        empty = X.columns[X.isna().all(axis=0)].tolist()
        raise ValueError(f"Features cannot be entirely missing in the full data: {empty}")

    y_numeric = pd.to_numeric(y, errors="coerce").astype(float)
    if not np.isfinite(y_numeric.to_numpy()).all():
        raise ValueError("y must contain only finite numeric values.")
    return X.copy(), y_numeric.rename(y.name or "target")


def fit_tuned_krr(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    spec: Mapping[str, Any],
    seed: int,
    inner_folds: int,
    n_jobs: int,
    memory: Memory | None = None,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[Pipeline, dict[str, Any], float, pd.DataFrame]:
    """Tune KRR by inner CV and refit the complete pipeline on ``X, y``."""

    estimator = clone(spec["estimator"])
    if memory is not None:
        estimator.set_params(memory=memory)
    grid = spec["param_grid"] if param_grid is None else dict(param_grid)
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    tuner = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=DEFAULT_SCORING,
        cv=inner_cv,
        refit=True,
        n_jobs=n_jobs,
        pre_dispatch=n_jobs,
        error_score="raise",
        return_train_score=False,
    )
    tuner.fit(X, y)
    fitted = tuner.best_estimator_.set_params(memory=None)
    best_parameters = plain_parameters(tuner.best_params_)
    inner_rmse = float(np.sqrt(-tuner.best_score_))
    cv_results = pd.DataFrame(tuner.cv_results_)
    return fitted, best_parameters, inner_rmse, cv_results


def fit_jackknife_plus_krr(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sex_column: str = DEFAULT_SEX_COLUMN,
    seed: int = DEFAULT_SEED,
    inner_folds: int = DEFAULT_INNER_FOLDS,
    n_jobs: int = DEFAULT_N_JOBS,
    param_grid: Mapping[str, Sequence[Any]] | None = None,
    progress_every: int = 5,
) -> tuple[dict[str, Any], Pipeline, pd.DataFrame, pd.DataFrame]:
    """Fit ``n`` LOO pipelines and one full-data KRR production pipeline."""

    if n_jobs < 1:
        raise ValueError("n_jobs must be at least 1.")
    X, y = validate_xy(X, y, sex_column=sex_column, inner_folds=inner_folds)
    spec = make_krr_spec(X.columns, sex_column=sex_column)
    loo_models: list[Pipeline] = []
    residual_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for loo_position, (fit_indices, held_out_indices) in enumerate(
        LeaveOneOut().split(X)
    ):
        fold_started = time.perf_counter()
        held_out_position = int(held_out_indices[0])
        if held_out_position != loo_position:
            raise RuntimeError("Unexpected LeaveOneOut ordering.")
        X_train = X.iloc[fit_indices]
        y_train = y.iloc[fit_indices]
        X_held_out = X.iloc[held_out_indices]
        observed = float(y.iloc[held_out_position])

        with TemporaryDirectory(prefix="coordination_jkplus_loo_") as cache_dir:
            fitted, parameters, inner_rmse, _ = fit_tuned_krr(
                X_train,
                y_train,
                spec=spec,
                seed=seed,
                inner_folds=inner_folds,
                n_jobs=n_jobs,
                memory=Memory(cache_dir, verbose=0),
                param_grid=param_grid,
            )
        prediction = float(np.asarray(fitted.predict(X_held_out)).ravel()[0])
        signed_residual = observed - prediction
        loo_models.append(fitted)
        residual_records.append(
            {
                "loo_position": loo_position,
                "sample_id": X.index[held_out_position],
                "observed_y": observed,
                "loo_prediction": prediction,
                "signed_residual": signed_residual,
                "absolute_residual": abs(signed_residual),
            }
        )
        selection_records.append(
            {
                "fit_role": "leave_one_out",
                "loo_position": loo_position,
                "sample_id": X.index[held_out_position],
                "n_fit_observations": len(X_train),
                "inner_cv_rmse": inner_rmse,
                "fit_seconds": time.perf_counter() - fold_started,
                "best_parameters": repr(parameters),
                **parameters,
            }
        )

        completed = loo_position + 1
        if (
            progress_every > 0
            and (completed == 1 or completed % progress_every == 0 or completed == len(X))
        ):
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(X) - completed)
            print(
                f"LOO KRR {completed:3d}/{len(X)}; "
                f"elapsed={elapsed:.1f}s; ETA={eta:.1f}s",
                flush=True,
            )

    production_started = time.perf_counter()
    with TemporaryDirectory(prefix="coordination_jkplus_production_") as cache_dir:
        production_model, production_parameters, production_inner_rmse, _ = fit_tuned_krr(
            X,
            y,
            spec=spec,
            seed=seed,
            inner_folds=inner_folds,
            n_jobs=n_jobs,
            memory=Memory(cache_dir, verbose=0),
            param_grid=param_grid,
        )
    selection_records.append(
        {
            "fit_role": "production",
            "loo_position": np.nan,
            "sample_id": np.nan,
            "n_fit_observations": len(X),
            "inner_cv_rmse": production_inner_rmse,
            "fit_seconds": time.perf_counter() - production_started,
            "best_parameters": repr(production_parameters),
            **production_parameters,
        }
    )

    residuals = pd.DataFrame(residual_records)
    selections = pd.DataFrame(selection_records)
    if len(loo_models) != len(X) or len(residuals) != len(X):
        raise RuntimeError("JK+ fitting did not produce exactly one model and score per row.")
    if not np.array_equal(residuals["loo_position"], np.arange(len(X))):
        raise RuntimeError("LOO residual ordering is not contiguous.")

    loo_artifact = {
        "artifact_type": "coordination_age_krr_jackknife_plus_loo_models",
        "artifact_version": ARTIFACT_VERSION,
        "n_training_observations": len(X),
        "feature_columns": X.columns.tolist(),
        "sample_ids": X.index.tolist(),
        "sex_column": sex_column,
        "inner_folds": inner_folds,
        "cv_seed": seed,
        "cv_scoring": DEFAULT_SCORING,
        "loo_models": loo_models,
    }
    return loo_artifact, production_model, residuals, selections


def save_jackknife_plus_artifacts(
    *,
    loo_artifact: Mapping[str, Any],
    production_model: Pipeline,
    residuals: pd.DataFrame,
    selections: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Serialize the JK+ model collection, production model, and metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "loo_models": output_dir / LOO_MODELS_FILENAME,
        "production_model": output_dir / PRODUCTION_MODEL_FILENAME,
        "residuals": output_dir / RESIDUALS_FILENAME,
        "selections": output_dir / SELECTIONS_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }

    with paths["loo_models"].open("wb") as handle:
        pickle.dump(dict(loo_artifact), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with paths["production_model"].open("wb") as handle:
        pickle.dump(production_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    residuals.to_csv(paths["residuals"], index=False)
    selections.to_csv(paths["selections"], index=False)

    n_observations = int(loo_artifact["n_training_observations"])
    n_loo_models = len(loo_artifact["loo_models"])
    if n_loo_models != n_observations:
        raise ValueError("The LOO artifact does not contain exactly n models.")
    manifest = {
        "artifact_type": "coordination_age_krr_jackknife_plus",
        "artifact_version": ARTIFACT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_training_observations": n_observations,
        "n_loo_models": n_loo_models,
        "n_production_models": 1,
        "total_models": n_loo_models + 1,
        "feature_columns": list(loo_artifact["feature_columns"]),
        "input_contract": (
            "Upstream domain validation, n>=5 censoring, invalid-timing cleanup, "
            "and approved-feature selection already applied; statistical transforms "
            "remain embedded in each fitted pipeline."
        ),
        "embedded_pipeline_steps": [
            "median imputation",
            "1%/99% winsorization",
            "feature-wise Yeo-Johnson",
            "standardization",
            "target standardization",
            "RBF kernel ridge",
        ],
        "inner_folds": int(loo_artifact["inner_folds"]),
        "cv_seed": int(loo_artifact["cv_seed"]),
        "cv_scoring": loo_artifact["cv_scoring"],
        "residual_definition": "absolute_residual = abs(y_i - f_minus_i(X_i))",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with paths["loo_models"].open("rb") as handle:
        restored_loo = pickle.load(handle)
    with paths["production_model"].open("rb") as handle:
        restored_production = pickle.load(handle)
    if len(restored_loo["loo_models"]) != n_observations:
        raise RuntimeError("LOO-model serialization round trip failed.")
    if not hasattr(restored_production, "predict"):
        raise RuntimeError("Production-model serialization round trip failed.")
    return paths


def load_xy_csv(
    x_path: Path,
    y_path: Path,
    *,
    id_column: str | None,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a validated modeling matrix and aligned target from separate CSVs."""

    X = pd.read_csv(x_path)
    target_table = pd.read_csv(y_path)
    if target_column not in target_table.columns:
        raise ValueError(f"Target CSV is missing column: {target_column}")
    if id_column is not None:
        if id_column not in X.columns or id_column not in target_table.columns:
            raise ValueError(f"Both CSVs must contain ID column: {id_column}")
        X = X.set_index(id_column)
        target_table = target_table.set_index(id_column)
    y = target_table[target_column]
    return X, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-csv", type=Path, required=True)
    parser.add_argument("--y-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--sex-column", default=DEFAULT_SEX_COLUMN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    X, y = load_xy_csv(
        args.x_csv,
        args.y_csv,
        id_column=args.id_column,
        target_column=args.target_column,
    )
    fitted = fit_jackknife_plus_krr(
        X,
        y,
        sex_column=args.sex_column,
        seed=args.seed,
        inner_folds=args.inner_folds,
        n_jobs=args.n_jobs,
        progress_every=args.progress_every,
    )
    paths = save_jackknife_plus_artifacts(
        loo_artifact=fitted[0],
        production_model=fitted[1],
        residuals=fitted[2],
        selections=fitted[3],
        output_dir=args.output_dir,
    )
    print(f"Saved {len(fitted[0]['loo_models']) + 1} models:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
