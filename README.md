# Coordination age

Predict an age in years from a participant's motor-coordination measurements,
using RBF kernel ridge regression and jackknife+ prediction intervals. The saved
model is ready to use: **no training is needed for inference**.

Developed with Prof. Jason Friedman's laboratory at Tel Aviv University using
144 typically developing participants aged 4–19. The held-out test RMSE was
**1.40 years** (29 participants); the production model uses all 144 participants.

## Setup

Use **Python 3.13**, matching the saved artifacts. Run commands from this folder:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Predict a participant or a batch

```bash
python scripts/infer_jackknife_plus_krr.py \
  --x-csv examples/participant.csv \
  --output-csv outputs/inference/predictions.csv
```

Replace the example CSV with your participants. The example is an existing
reference row for demonstrating the input format, **not a new evaluation case**.

- Use the **32 columns in `Data/X.csv`, in that order**, one participant per row.
- Supply laboratory aggregates in the original units, before statistical
  transformation. Use `0` for female and `1` for male in `Sex_0Female_1Male`.
- Leave unavailable measurements empty. Apply the laboratory exclusions,
  copied-trial removal and minimum-five-valid-trial rule before inference;
  [data preparation](scripts/data/README.md) explains the feature-specific rules.
- Exclude age. An optional ID column can be retained using `--id-column SubjectNumber`.

The output contains `prediction`, `jkplus_lower`, `jkplus_upper`, and
`jkplus_width`, all in **years**, plus the row ID, requested coverage and an
indicator of whether the point prediction lies inside the interval. The default
is 95% requested coverage (`--alpha 0.05`), with no clipping. Missing-value
imputation and all fitted statistical transformations are included in the model.

For Python use, load once and predict any number of batches:

```python
import pandas as pd
from src.inference import CoordinationAgeModel

model = CoordinationAgeModel()
result = model.predict(pd.read_csv("examples/participant.csv"))
```

Interpret the prediction as a normative age comparison. Accuracy and interval
coverage in developmentally impaired populations have not been established;
the score is not a validated measure of years of developmental delay. Load only
trusted model files; the loader checks the bundled artifact hashes and sklearn version.

## Reproduce results

Regenerate **all five final figures** and diagnostic tables from saved results,
without training:

```bash
python scripts/visualizations/create_all_figures.py
```

Find the PNGs and their source manifest in [`outputs/visualizations/`](outputs/visualizations/).
The final model comparison includes the 18-configuration random-forest grid.
[Reproduction instructions](docs/reproducing.md) cover individual candidate
reruns, complete model refitting, held-out evaluation, task ablation and tests.

## Files

| Directory | Contents |
|---|---|
| `Data/` | Final aggregate data, 32-column inputs, ages and fixed 115/29 split |
| `artifacts/jackknife_plus_krr/` | Default production model, 144 LOO models and integrity manifest |
| `artifacts/train115_jackknife_plus_krr/` | Models used for the held-out evaluation |
| `artifacts/jackknife_plus_krr_without_box_and_blocks/` | Optional 31-predictor model; explicitly choose it with `--artifact-dir` |
| `outputs/` | Final comparison, held-out predictions, task ablation and figures |
| `scripts/`, `src/` | Runnable commands and shared modeling code |
| `notebooks/` | Fitting and analysis walkthroughs |

For the report's Section 4 statistical results, use the
[association-analysis notebook](notebooks/statistical_association_analysis.ipynb)
with `Data/association/train.csv`, `Data/association/variables.xlsx` and the
development dependencies. This analysis is independent of the saved prediction pipeline.
