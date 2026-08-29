# Coordination-age production pipeline

This repository contains the final RBF kernel-ridge fitting/inference workflow
and the data-preparation steps required immediately before modeling.

Install the Python environment with:

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Data preparation is deliberately split into two stages.

### 1. Enforce blacklist truth and remove copied trials

```bash
python scripts/validate_all_trials.py \
  --input Data/bothexperiments_all_trials.csv \
  --blacklist Data/blacklist.csv \
  --output Data/bothexperiments_all_trials_valid.csv \
  --audit-output outputs/all_trials_corrections.csv
```

The normalized blacklist may be CSV or Parquet and must contain:

- `subject`, `task`, `condition`, and `trial` as the trial key;
- `blacklisted`, with the lab blacklist treated as authoritative.

The script sets every blacklisted trial-feature cell to missing. It detects
isolated copies through exact task-specific feature-vector matches. It infers
a copied second-half block only when at least three visible pairs support the
half-table offset and at least 80% of comparable pairs match. Every exclusion
is labeled as blacklist-, direct-match-, or block-inference-based in the audit.
Only embedded aggregates affected by corrections are recomputed.

### 2. Build aggregates, count valid trials, and apply n >= 5

```bash
python scripts/aggregate_all_trials.py \
  --input Data/bothexperiments_all_trials_valid.csv \
  --supplemental-aggregates Data/bothexperiments.csv \
  --output Data/bothexperiments_aggregate_n5.csv
```

The legacy all-trials table contains everything needed except `meandistance`.
That feature was computed inside the original lab pipeline and cannot be
reconstructed from its trial-feature columns. The supplemental aggregate file
is used only for `meandistance` and subject alignment. The light/heavy contrast
features are reconstructed as `Heavy - Light`.

The trial-level availability masks for `peakvel`, `movementtime`, and
`maxaperturePercent` are identical across every subject and trial position in
the reference data. Their shared reach-to-grasp count is therefore used as the
`n >= 5` validity proxy for `meandistance`. The script checks this equality and
fails rather than applying the proxy if the three observable counts disagree.

The output contains the original 45 aggregate columns plus 22 `n_trials_*`
diagnostics. Every count-backed aggregate is set to missing when its own valid
trial count is below five. Trial counts are diagnostics and must not be passed
to the model as predictors.

## Reproduction result

Using the original project inputs:

- The all-trials/blacklist heuristic detects three supported offset-10 blocks
  (subjects 141, 145, and 146), 18 directly visible vector matches, and the
  isolated subject-153 trial-20 copy. Their union is exactly the 31 documented
  copied trials.
- Stage 1 reproduces `bothexperiments_all_trials_v2.csv` with no differing
  cells.
- Stage 2 with `--minimum-valid-trials 1` reproduces the uncensored
  `bothexperiments_v2.csv` with no differing cells.
- Stage 2 with the default threshold of five reproduces
  `bothexperiments_v2.csv` after applying its family-specific `n_trials_* >= 5`
  masks and the shared reach-to-grasp count proxy to `meandistance`, with no
  differing cells.

## Model fitting and inference

`scripts/fit_jackknife_plus_krr.py` expects an already domain-cleaned numeric
`X` and aligned `y`. It fits and saves N leave-one-out KRR pipelines plus one
full-data production pipeline. `scripts/infer_jackknife_plus_krr.py` returns
the production prediction and the jackknife+ interval.

### Final checked-in modeling data

- `Data/bothexperiments.csv`: finalized 144-subject aggregate table, including
  22 trial-count diagnostics and all `n >= 5` masks (including the shared
  reach-to-grasp proxy for `meandistance`).
- `Data/X.csv`: exactly the frozen 32 predictors in production order.
- `Data/y.csv`: the single target column `AgeInYears`, row-aligned with `X.csv`.
- `Data/modeling_input_metadata.json`: subject order and construction audit.

The model inputs can be regenerated from the aggregate table with:

```bash
python scripts/create_modeling_inputs.py \
  --aggregate-csv Data/bothexperiments.csv \
  --x-output Data/X.csv \
  --y-output Data/y.csv \
  --metadata-output Data/modeling_input_metadata.json
```

### Fit all saved objects

Run `notebooks/fit_and_infer_jackknife_plus_krr.ipynb` from top to bottom. Its
fitting cell invokes the production script with five inner folds and writes:

```text
artifacts/jackknife_plus_krr/
├── coordination_age_krr_jackknife_plus_loo_models.pkl
├── coordination_age_krr_jackknife_plus_production_model.pkl
├── coordination_age_krr_jackknife_plus_residuals.csv
├── coordination_age_krr_jackknife_plus_selections.csv
└── coordination_age_krr_jackknife_plus_manifest.json
```

The files are one artifact set and must remain together. The inference loader
checks their SHA-256 digests and verifies model/residual ordering.

### Infer on a new X with saved objects

The new CSV must contain exactly the same 32 columns, in the same order, as
`Data/X.csv`. Do not include age, `SubjectNumber`, or `n_trials_*` columns.
Domain cleaning and aggregation must already have been performed; feature
missingness may remain as `NaN` because imputation is inside each saved model.

For a 95% JK+ interval (`alpha = 0.05`):

```bash
python scripts/infer_jackknife_plus_krr.py \
  --x-csv path/to/new_X.csv \
  --artifact-dir artifacts/jackknife_plus_krr \
  --output-csv outputs/new_predictions.csv \
  --alpha 0.05
```

To enforce the prespecified 4–18-year service range on the point prediction
and interval endpoints, additionally pass `--clip-lower 4 --clip-upper 18`.

The output includes:

- `prediction`: prediction from the full-data production model;
- `jkplus_lower`, `jkplus_upper`: jackknife+ interval endpoints;
- `jkplus_width`: interval width;
- `requested_coverage`: `1 - alpha`.

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

The coverage guarantee is finite-sample marginal under exchangeability. It is
not a guarantee of 95% conditional coverage for every individual feature
profile.
