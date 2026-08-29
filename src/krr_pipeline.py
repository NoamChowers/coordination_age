"""KRR-only preprocessing pipeline and hyperparameter grid.

Domain-level cleaning belongs upstream. This module contains only statistical
transformations that must be learned from each training fold, followed by the
selected RBF kernel-ridge regressor.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.stats import yeojohnson
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FoldwiseWinsorizer(BaseEstimator, TransformerMixin):
    """Clip columns using quantiles learned only during ``fit``."""

    def __init__(
        self,
        feature_names: Sequence[str],
        exclude_features: Sequence[str] | None = None,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.feature_names = feature_names
        self.exclude_features = exclude_features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: Any, y: Any = None) -> FoldwiseWinsorizer:
        X_array = np.asarray(X, dtype=float)
        self.feature_names_ = list(self.feature_names)
        excluded = set(
            [] if self.exclude_features is None else self.exclude_features
        )
        if X_array.shape[1] != len(self.feature_names_):
            raise ValueError("X column count does not match feature_names.")
        if not 0 <= self.lower_quantile <= self.upper_quantile <= 1:
            raise ValueError(
                "Winsorization quantiles must satisfy 0 <= lower <= upper <= 1."
            )

        self.lower_bounds_ = np.empty(X_array.shape[1], dtype=float)
        self.upper_bounds_ = np.empty(X_array.shape[1], dtype=float)
        for index, feature in enumerate(self.feature_names_):
            if feature in excluded:
                self.lower_bounds_[index] = -np.inf
                self.upper_bounds_[index] = np.inf
            else:
                self.lower_bounds_[index] = np.quantile(
                    X_array[:, index], self.lower_quantile
                )
                self.upper_bounds_[index] = np.quantile(
                    X_array[:, index], self.upper_quantile
                )
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_array = np.asarray(X, dtype=float).copy()
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


class YeoJohnsonByFeature(BaseEstimator, TransformerMixin):
    """Fit a separate Yeo-Johnson transform to each non-excluded column."""

    def __init__(
        self,
        feature_names: Sequence[str],
        exclude_features: Sequence[str] | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.exclude_features = exclude_features

    def fit(self, X: Any, y: Any = None) -> YeoJohnsonByFeature:
        X_array = np.asarray(X, dtype=float)
        self.feature_names_ = list(self.feature_names)
        excluded = set(
            [] if self.exclude_features is None else self.exclude_features
        )
        if X_array.shape[1] != len(self.feature_names_):
            raise ValueError("X column count does not match feature_names.")

        self.lambdas_: dict[int, float] = {}
        for index, feature in enumerate(self.feature_names_):
            if feature in excluded or np.std(X_array[:, index]) < 1e-12:
                continue
            _, fitted_lambda = yeojohnson(X_array[:, index])
            self.lambdas_[index] = float(fitted_lambda)
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_array = np.asarray(X, dtype=float).copy()
        for index, fitted_lambda in self.lambdas_.items():
            X_array[:, index] = yeojohnson(
                X_array[:, index], lmbda=fitted_lambda
            )
        return X_array

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


def make_krr_spec(
    feature_names: Sequence[str],
    *,
    sex_column: str,
) -> dict[str, Any]:
    """Build the complete KRR pipeline and its prespecified tuning grid."""

    feature_names = list(feature_names)
    if sex_column not in feature_names:
        raise ValueError(f"Missing required sex predictor: {sex_column}")

    estimator = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            (
                "winsorizer",
                FoldwiseWinsorizer(
                    feature_names=feature_names,
                    exclude_features=[sex_column],
                    lower_quantile=0.01,
                    upper_quantile=0.99,
                ),
            ),
            (
                "yeo_johnson",
                YeoJohnsonByFeature(
                    feature_names=feature_names,
                    exclude_features=[sex_column],
                ),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=KernelRidge(kernel="rbf"),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )

    return {
        "estimator": estimator,
        "param_grid": {
            "model__regressor__alpha": np.logspace(-4, 3, 15),
            "model__regressor__gamma": [
                0.0003,
                0.001,
                0.003,
                0.01,
                0.03,
                0.1,
                0.3,
                1.0,
            ],
        },
        "architecture": "Standardized RBF kernel ridge with target scaling",
    }


__all__ = ["FoldwiseWinsorizer", "YeoJohnsonByFeature", "make_krr_spec"]
