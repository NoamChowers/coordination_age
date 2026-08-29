#!/usr/bin/env python3
"""Predict coordination age and a jackknife+ interval for new subjects.

The point prediction comes from the full-data production KRR pipeline. The
jackknife+ endpoints use all leave-one-out pipelines and their aligned absolute
OOF residuals. For requested total miscoverage ``alpha``, each tail uses
``alpha / 2`` and the finite-sample modified empirical order statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fit_jackknife_plus_krr import (
    LOO_MODELS_FILENAME,
    MANIFEST_FILENAME,
    PRODUCTION_MODEL_FILENAME,
    RESIDUALS_FILENAME,
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of ``path``."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jackknife_plus_artifacts(
    *,
    loo_models_path: Path,
    production_model_path: Path,
    residuals_path: Path,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    """Load and optionally checksum-validate serialized JK+ artifacts."""

    paths = {
        "loo_models": Path(loo_models_path),
        "production_model": Path(production_model_path),
        "residuals": Path(residuals_path),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    if manifest_path is not None:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        for name, path in paths.items():
            expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
            if expected is None:
                raise ValueError(f"Manifest has no checksum for {name}.")
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(
                    f"Checksum mismatch for {name}: expected {expected}, got {observed}."
                )

    with paths["loo_models"].open("rb") as handle:
        loo_artifact = pickle.load(handle)
    with paths["production_model"].open("rb") as handle:
        production_model = pickle.load(handle)
    residuals = pd.read_csv(paths["residuals"])
    return loo_artifact, production_model, residuals


def validate_inference_inputs(
    X: pd.DataFrame,
    loo_artifact: Mapping[str, Any],
    production_model: Any,
    residuals: pd.DataFrame,
) -> tuple[list[Any], np.ndarray]:
    """Validate model count, score alignment, and feature contract."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with named columns.")
    expected_columns = list(loo_artifact.get("feature_columns", []))
    if not expected_columns:
        raise ValueError("LOO artifact has no feature-column contract.")
    if X.columns.tolist() != expected_columns:
        missing = [column for column in expected_columns if column not in X.columns]
        extra = [column for column in X.columns if column not in expected_columns]
        raise ValueError(
            "Inference columns do not exactly match the fitted order. "
            f"Missing={missing}; extra={extra}."
        )
    if not X.index.is_unique:
        raise ValueError("Inference row identifiers must be unique.")

    loo_models = list(loo_artifact.get("loo_models", []))
    n_expected = int(loo_artifact.get("n_training_observations", -1))
    if len(loo_models) != n_expected:
        raise ValueError(
            f"Expected {n_expected} LOO models; artifact contains {len(loo_models)}."
        )
    required_residual_columns = {
        "loo_position",
        "sample_id",
        "absolute_residual",
    }
    missing_residual_columns = required_residual_columns - set(residuals.columns)
    if missing_residual_columns:
        raise ValueError(
            f"Residual table is missing columns: {sorted(missing_residual_columns)}"
        )
    ordered = residuals.sort_values("loo_position").reset_index(drop=True)
    positions = ordered["loo_position"].to_numpy()
    if not np.array_equal(positions, np.arange(n_expected)):
        raise ValueError("Residual loo_position values must be exactly 0,...,n-1.")
    sample_ids = [str(value) for value in loo_artifact.get("sample_ids", [])]
    residual_ids = ordered["sample_id"].astype(str).tolist()
    if len(sample_ids) != n_expected or sample_ids != residual_ids:
        raise ValueError("Residual rows are not aligned to the saved LOO model order.")
    absolute_residuals = ordered["absolute_residual"].to_numpy(dtype=float)
    if not np.isfinite(absolute_residuals).all() or (absolute_residuals < 0).any():
        raise ValueError("Absolute residuals must be finite and nonnegative.")
    if not hasattr(production_model, "predict"):
        raise TypeError("Production model does not provide predict().")
    if any(not hasattr(model, "predict") for model in loo_models):
        raise TypeError("Every LOO model must provide predict().")
    return loo_models, absolute_residuals


def modified_jackknife_plus_bounds(
    loo_predictions: np.ndarray,
    absolute_residuals: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Calculate two-sided JK+ bounds using finite-sample order statistics.

    ``loo_predictions`` has shape ``(n_training, n_new)``. The total requested
    miscoverage ``alpha`` is split equally across the two tails. If the modified
    rank lies beyond the available sample, the conformal convention returns an
    unbounded endpoint rather than silently truncating the rank.
    """

    predictions = np.asarray(loo_predictions, dtype=float)
    residuals = np.asarray(absolute_residuals, dtype=float)
    if predictions.ndim != 2:
        raise ValueError("loo_predictions must have shape (n_training, n_new).")
    if residuals.ndim != 1 or len(residuals) != predictions.shape[0]:
        raise ValueError("absolute_residuals must align to the prediction rows.")
    if not np.isfinite(predictions).all() or not np.isfinite(residuals).all():
        raise ValueError("Predictions and residuals must be finite.")
    if (residuals < 0).any():
        raise ValueError("Absolute residuals cannot be negative.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")

    n_training = predictions.shape[0]
    tail_alpha = alpha / 2.0
    upper_rank = int(np.ceil((1.0 - tail_alpha) * (n_training + 1)))
    lower_rank = n_training - upper_rank + 1
    lower_candidates = predictions - residuals[:, None]
    upper_candidates = predictions + residuals[:, None]

    if upper_rank > n_training:
        lower = np.full(predictions.shape[1], -np.inf)
        upper = np.full(predictions.shape[1], np.inf)
    else:
        lower = np.partition(lower_candidates, lower_rank - 1, axis=0)[
            lower_rank - 1
        ]
        upper = np.partition(upper_candidates, upper_rank - 1, axis=0)[
            upper_rank - 1
        ]
    rank_metadata: dict[str, int | float] = {
        "n_training": n_training,
        "alpha": alpha,
        "tail_alpha": tail_alpha,
        "lower_rank_one_based": lower_rank,
        "upper_rank_one_based": upper_rank,
    }
    return lower, upper, rank_metadata


def predict_with_jackknife_plus(
    X: pd.DataFrame,
    *,
    loo_artifact: Mapping[str, Any],
    production_model: Any,
    residuals: pd.DataFrame,
    alpha: float = 0.10,
    clip: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Return full-model predictions and JK+ intervals for ``X``."""

    loo_models, absolute_residuals = validate_inference_inputs(
        X, loo_artifact, production_model, residuals
    )
    point_predictions = np.asarray(production_model.predict(X), dtype=float).ravel()
    loo_predictions = np.vstack(
        [np.asarray(model.predict(X), dtype=float).ravel() for model in loo_models]
    )
    lower, upper, rank_metadata = modified_jackknife_plus_bounds(
        loo_predictions,
        absolute_residuals,
        alpha=alpha,
    )
    if len(point_predictions) != len(X) or loo_predictions.shape[1] != len(X):
        raise RuntimeError("A fitted model returned an unexpected prediction shape.")

    result = pd.DataFrame(
        {
            "prediction": point_predictions,
            "jkplus_lower": lower,
            "jkplus_upper": upper,
        },
        index=X.index,
    )
    if clip is not None:
        clip_lower, clip_upper = map(float, clip)
        if not clip_lower < clip_upper:
            raise ValueError("clip lower bound must be strictly below upper bound.")
        result.insert(0, "raw_prediction", result["prediction"])
        result["raw_jkplus_lower"] = result["jkplus_lower"]
        result["raw_jkplus_upper"] = result["jkplus_upper"]
        result["prediction"] = result["prediction"].clip(clip_lower, clip_upper)
        result["jkplus_lower"] = result["jkplus_lower"].clip(clip_lower, clip_upper)
        result["jkplus_upper"] = result["jkplus_upper"].clip(clip_lower, clip_upper)

    result["jkplus_width"] = result["jkplus_upper"] - result["jkplus_lower"]
    result["prediction_inside_interval"] = (
        (result["prediction"] >= result["jkplus_lower"])
        & (result["prediction"] <= result["jkplus_upper"])
    )
    result["requested_coverage"] = 1.0 - alpha
    result.attrs["jackknife_plus_rank"] = rank_metadata
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-csv", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--clip-lower", type=float, default=None)
    parser.add_argument("--clip-upper", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    X = pd.read_csv(args.x_csv)
    if args.id_column is not None:
        if args.id_column not in X.columns:
            raise ValueError(f"Inference CSV is missing ID column: {args.id_column}")
        X = X.set_index(args.id_column)

    clip = None
    if args.clip_lower is not None or args.clip_upper is not None:
        if args.clip_lower is None or args.clip_upper is None:
            raise ValueError("Both --clip-lower and --clip-upper must be supplied.")
        clip = (args.clip_lower, args.clip_upper)

    artifact_dir = args.artifact_dir
    loo_artifact, production_model, residuals = load_jackknife_plus_artifacts(
        loo_models_path=artifact_dir / LOO_MODELS_FILENAME,
        production_model_path=artifact_dir / PRODUCTION_MODEL_FILENAME,
        residuals_path=artifact_dir / RESIDUALS_FILENAME,
        manifest_path=artifact_dir / MANIFEST_FILENAME,
    )
    result = predict_with_jackknife_plus(
        X,
        loo_artifact=loo_artifact,
        production_model=production_model,
        residuals=residuals,
        alpha=args.alpha,
        clip=clip,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.output_csv,
        index=True,
        index_label=X.index.name or "row_id",
    )
    print(f"Saved {len(result)} predictions to {args.output_csv}")
    print("JK+ rank:", result.attrs["jackknife_plus_rank"])


if __name__ == "__main__":
    main()
