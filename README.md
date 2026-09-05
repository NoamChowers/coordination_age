# Coordination-age production model

This repository fits and serves the finalized RBF kernel-ridge coordination-age
model with jackknife+ prediction intervals.

Install the environment:

```bash
python -m pip install -r requirements.txt
```

## Included model data

- `Data/bothexperiments.csv`: finalized aggregate data for all 144 subjects.
- `Data/X.csv`: exactly the selected 32 predictors in production order.
- `Data/y.csv`: one row-aligned target column, `AgeInYears`.
- `Data/modeling_input_metadata.json`: subject ordering and construction audit.
- `Data/train_test_split.csv`: recovered historical 115/29 membership, production-row
  position, and original within-partition order.
- `train.csv`: training data for statistical inference

The FIT script consumes `X.csv` and `y.csv`; it does not redo laboratory data
processing or feature selection. Documentation for constructing these files is
kept separately in `scripts/data/README.md`.

## Required input-data assumptions

Both model fitting and inference assume that the input aggregates were
prepared under the following rules. The modeling scripts do not detect or
correct violations of these assumptions.

1. **The blacklist is authoritative.** Trial inclusion and exclusion must
   follow the laboratory blacklist as the source of truth. Blacklisted trials
   must not contribute to any aggregate.

2. **Copied trials are excluded.** Aggregates must not include repeated or
   copied trials. This includes the known block-copy scenario in which trials
   11–20 duplicate trials 1–10.

3. **At least five valid trials are required.** For repeated-trial
   measurements from the `pressing`, `reach-to-grasp`, and `lift-object`
   tasks, an aggregate must be set to `NaN` whenever fewer than five valid,
   non-blacklisted, non-copied trials contribute to it.

   For example, if a participant has only four valid `reach-to-grasp` trials,
   the associated `peakvel`, `movementtime`, `maxaperturePercent`, and
   `meandistance` aggregates must be `NaN`; they must not be calculated from
   those four trials.

New inference data must satisfy the same assumptions as the data used for
fitting.

## Fit the saved model objects

Run `notebooks/fit_and_infer_jackknife_plus_krr.ipynb` from top to bottom. The
notebook invokes:

```bash
python scripts/fit_jackknife_plus_krr.py \
  --x-csv Data/X.csv \
  --y-csv Data/y.csv \
  --target-column AgeInYears \
  --output-dir artifacts/jackknife_plus_krr \
  --inner-folds 5 \
  --n-jobs 6
```

For `N=144`, the fitting procedure tunes and fits 144 leave-one-out pipelines
and one full-data production pipeline. It writes five files:

```text
artifacts/jackknife_plus_krr/
├── coordination_age_krr_jackknife_plus_loo_models.pkl
├── coordination_age_krr_jackknife_plus_production_model.pkl
├── coordination_age_krr_jackknife_plus_residuals.csv
├── coordination_age_krr_jackknife_plus_selections.csv
└── coordination_age_krr_jackknife_plus_manifest.json
```

- `loo_models.pkl` is a dictionary containing the ordered list of 144 fitted
  leave-one-out sklearn pipelines and their feature/sample metadata.
- `production_model.pkl` is the fitted full-data sklearn pipeline used for the
  point prediction.
- `residuals.csv` contains each subject's observed age, LOO prediction, signed
  residual, and absolute residual. These residuals calibrate JK+ intervals.
- `selections.csv` records inner-CV error, selected KRR hyperparameters, and fit
  time for every LOO fit and the full-data fit.
- `manifest.json` records the schema, model counts, runtime versions, and
  SHA-256 digests used to verify artifact integrity.

At the Python-function level, `fit_jackknife_plus_krr(...)` returns a four-item
tuple:

```python
(loo_artifact, production_model, residuals_dataframe, selections_dataframe)
```

The command-line script serializes that tuple into the five files above.

## Train-only model selection and test-set interval audit

Run `notebooks/train_model_selection_and_jackknife_plus_test.ipynb` to:

- compare KRR, ElasticNet, Lasso, Ridge, OLS, and fixed Random Forest using
  train-only outer-LOOCV predictions;
- save a separate 115-subject KRR JK+ artifact set under
  `artifacts/train115_jackknife_plus_krr/`; and
- predict the 29 historical test subjects with 95% JK+ intervals, saving
  row-level results and diagnostics under `outputs/train115_jkplus/`.

The notebook is self-contained within this repository. Test outcomes enter
only after architecture selection, fitting, and interval construction. Because
they were inspected during earlier project work, the resulting test metrics
are a post-hoc audit rather than pristine external validation.

Run `notebooks/model_diagnostics_and_conformal_visualizations.ipynb` to create
the saved train-LOOCV comparison, held-out-test JK+ interval diagnostics, and
full-data LOOCV KRR diagnostics. This notebook performs no model fitting; it
reads the saved prediction tables and writes PNGs plus a figure manifest under
`outputs/visualizations/`.

## Infer on a new X

The new CSV must contain the same 32 columns, in the same order, as
`Data/X.csv`. Do not include identifiers or age. The values must already follow
the same laboratory processing, aggregation, validity, and feature-selection
rules as the training matrix. Missing predictor values may remain as `NaN`;
imputation is part of every saved pipeline.

For a 95% JK+ interval (`alpha = 0.05`):

```bash
python scripts/infer_jackknife_plus_krr.py \
  --x-csv path/to/new_X.csv \
  --artifact-dir artifacts/jackknife_plus_krr \
  --output-csv outputs/new_predictions.csv \
  --alpha 0.05
```

The CLI writes a CSV with one row per input subject. The Python function
`predict_with_jackknife_plus(...)` returns the same result as a pandas
`DataFrame`—not a single float or tuple. Without clipping, its columns are:

| column | meaning | type |
|---|---|---|
| `prediction` | Full-data production-model prediction, in years | float |
| `jkplus_lower` | Lower JK+ endpoint, in years | float |
| `jkplus_upper` | Upper JK+ endpoint, in years | float |
| `jkplus_width` | `jkplus_upper - jkplus_lower` | float |
| `prediction_inside_interval` | Whether the point prediction lies inside its interval | bool |
| `requested_coverage` | Requested marginal coverage, e.g. `0.95` | float |

Thus, a one-subject call has the shape:

```text
   prediction  jkplus_lower  jkplus_upper  jkplus_width  prediction_inside_interval  requested_coverage
0      <float>       <float>       <float>       <float>                        <bool>                0.95
```

To enforce the prespecified 4–18-year service range, add:

```text
--clip-lower 4 --clip-upper 18
```

When clipping is requested, the result additionally preserves
`raw_prediction`, `raw_jkplus_lower`, and `raw_jkplus_upper` before returning
the clipped `prediction` and interval columns.

The equivalent Python API is:

```python
from pathlib import Path
import pandas as pd

from scripts.fit_jackknife_plus_krr import (
    LOO_MODELS_FILENAME,
    MANIFEST_FILENAME,
    PRODUCTION_MODEL_FILENAME,
    RESIDUALS_FILENAME,
)
from scripts.infer_jackknife_plus_krr import (
    load_jackknife_plus_artifacts,
    predict_with_jackknife_plus,
)

artifact_dir = Path("artifacts/jackknife_plus_krr")
loo_artifact, production_model, residuals = load_jackknife_plus_artifacts(
    loo_models_path=artifact_dir / LOO_MODELS_FILENAME,
    production_model_path=artifact_dir / PRODUCTION_MODEL_FILENAME,
    residuals_path=artifact_dir / RESIDUALS_FILENAME,
    manifest_path=artifact_dir / MANIFEST_FILENAME,
)

new_X = pd.read_csv("path/to/new_X.csv")
result = predict_with_jackknife_plus(
    new_X,
    loo_artifact=loo_artifact,
    production_model=production_model,
    residuals=residuals,
    alpha=0.05,
)
```

The artifact directory must remain intact because the loader verifies file
digests and model/residual ordering. The coverage statement is finite-sample
marginal under exchangeability; it is not 95% conditional coverage for every
individual feature profile.

## Statistical Inference
To reproduce the statistical results in section 4 of the report, run the notebook `inference.ipynb`
