"""Load the saved coordination-age model once and predict participant batches."""

from pathlib import Path

import pandas as pd

from scripts.fit_jackknife_plus_krr import (
    LOO_MODELS_FILENAME, MANIFEST_FILENAME, PRODUCTION_MODEL_FILENAME,
    RESIDUALS_FILENAME,
)
from scripts.infer_jackknife_plus_krr import (
    load_jackknife_plus_artifacts, predict_with_jackknife_plus,
)

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts/jackknife_plus_krr"


class CoordinationAgeModel:
    """Checksum-validated pipelines with a named-column input contract.

    Load only trusted pickle artifacts. The bundled files use the versions in
    requirements.txt. Predictions and interval bounds are in years, unclipped.
    """

    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR):
        directory = Path(artifact_dir)
        self._loo, self._point, self._residuals = load_jackknife_plus_artifacts(
            loo_models_path=directory / LOO_MODELS_FILENAME,
            production_model_path=directory / PRODUCTION_MODEL_FILENAME,
            residuals_path=directory / RESIDUALS_FILENAME,
            manifest_path=directory / MANIFEST_FILENAME,
        )

    @property
    def feature_columns(self) -> list[str]:
        return list(self._loo["feature_columns"])

    def predict(self, X: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
        """Predict one or more rows, retaining their input index."""
        return predict_with_jackknife_plus(
            X, loo_artifact=self._loo, production_model=self._point,
            residuals=self._residuals, alpha=alpha,
        )
