# Reproducing the results

Run from the repository root in the environment described in the README.
Saved artifacts and prediction tables are included, so inspecting results,
predicting new participants and rebuilding figures do not require model fitting.

## Figures and recorded results (seconds)

```bash
python scripts/visualizations/create_all_figures.py
```

This writes the five final PNGs, age-bin error summaries and a source manifest to
`outputs/visualizations/`. Use `--output-dir path/to/figures` for a separate copy.
It reads the final candidate comparison, held-out predictions, full-cohort LOO
residuals and task-ablation tables. It never fits or selects a model.

The principal results are:

| Evaluation | RMSE (years) | Source |
|---|---:|---|
| Training outer LOOCV, RBF-KRR | 1.4215 | `outputs/model_comparison/comparison_metrics.csv` |
| Training outer LOOCV, Random Forest | 1.6843 | Same table |
| Held-out test, RBF-KRR | 1.4035 | `outputs/train115_jkplus/train115_jkplus_test29_predictions.csv` |
| Full-cohort LOOCV, RBF-KRR | 1.3862 | `artifacts/jackknife_plus_krr/*residuals.csv` |

## Rerun one candidate (resumable)

```bash
python scripts/evaluate_model_candidates.py --models "Random Forest" --n-jobs 6
python scripts/visualizations/create_all_figures.py
```

The RF grid has 200 trees, depth `[4, 7, None]`, minimum leaf size `[1, 3, 5]`,
and feature fraction `[0.5, 1.0]`: 18 configurations per inner CV. A fresh
115-fold RF run took approximately 17 minutes with six workers on the development
machine. A completed matching checkpoint is reused in seconds.

Available names: `"RBF Kernel Ridge"`, `"ElasticNet"`, `"Lasso"`, `"Ridge"`,
`"OLS"`, `"Random Forest"`. Pass multiple names together. Saved predictions for
unrequested candidates remain in the comparison. Each candidate validates its
training data, configuration, source and software versions before resuming.
Use a **new output directory** after changing a grid, implementation or dependency:

```bash
python scripts/evaluate_model_candidates.py --models "Random Forest" \
  --output-dir outputs/rebuilt_comparison --n-jobs 6
python scripts/visualizations/create_model_comparison.py \
  --comparison-dir outputs/rebuilt_comparison --output-dir outputs/rebuilt_figures
```

To rerun all six candidates, supply all six names. This regenerates comparison
predictions; it does not serialize production KRR models.

## Refit production artifacts

```bash
python scripts/fit_jackknife_plus_krr.py \
  --x-csv Data/X.csv --y-csv Data/y.csv --target-column AgeInYears \
  --output-dir artifacts/jackknife_plus_krr --inner-folds 5 --n-jobs 6
```

This tunes and fits 144 LOO pipelines and one full-data pipeline, saving models,
residuals, selected hyperparameters and a checksum manifest. It replaces that
artifact directory's files. Use a different `--output-dir` to preserve the
bundled version. Unlike candidate evaluation, this command has no fold resume;
allow substantial runtime. `--n-jobs 1` reduces CPU use.

For the optional model without Box and Blocks, add `--drop-column boxAndBlocks`
and use `--output-dir artifacts/jackknife_plus_krr_without_box_and_blocks`.
Inference then requires the corresponding 31-column input and `--artifact-dir`.

Model binaries can differ across platforms even with the same seed. Compare
predictions and selected parameters with numerical tolerance, not pickle hashes
between independently fitted runs. Manifest hashes check the integrity of one
saved artifact collection.

## Rebuild the held-out evaluation and task ablation

Install the notebook/test tools:

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name coordination-age
jupyter lab
```

Choose the `coordination-age` kernel and run the relevant notebook top to bottom:

1. `train_model_selection_and_jackknife_plus_test.ipynb` fits the six final
   candidates on the fixed 115 training participants, saves their comparison,
   fits training-only KRR artifacts and evaluates the 29 held-out participants.
2. `experiment_level_task_ablation.ipynb` retunes KRR for each task subset,
   selects the greedy elimination path on training predictions, then evaluates
   the frozen subsets on the test set and calculates the paired-error bootstrap.
   It reuses completed subset caches; a fresh run can take hours.
3. `fit_and_infer_jackknife_plus_krr.ipynb` demonstrates full-cohort fitting and
   inference. By default it loads the bundled model; set `REFIT=True` to refit.
4. `model_diagnostics_and_conformal_visualizations.ipynb` runs the same fast
   figure generator as the command-line entry point.

After any refitting, regenerate figures. The split file specifies the historical
membership and row order; its generation/stratification procedure is not available.
The displayed age bins are descriptive. The held-out evaluation is internal to
this cohort, not external clinical validation. The task-ablation bootstrap
resamples saved paired errors without refitting; it does not capture retraining
or task-selection uncertainty.

## Data and inference contract

The separate `statistical_association_analysis.ipynb` notebook uses
`Data/association/train.csv` (115 rows) and `variables.xlsx` for task grouping.
It fits randomized Lasso selective inference and does not produce the saved
coordination-age predictor or its intervals. Its supplied dataset and statistical
algorithm are preserved separately; the production audit checks its input
preparation but does not revalidate the selective-inference theory or rerun the
long Gibbs calculations. `requirements-dev.txt` includes its Excel reader.

`Data/bothexperiments.csv` is the finalized aggregate input. Its construction and
regeneration of `X.csv` / `y.csv` are documented in [data preparation](../scripts/data/README.md).
The raw trial delivery and laboratory blacklist are not bundled; they are needed
only to repeat upstream cleaning, not to reproduce the model from the included
aggregates. Do not estimate preprocessing anew on incoming participants.

Jackknife+ uses `tau = alpha / 2`; the default `alpha=0.05` uses `tau=0.025`.
Its general marginal lower bound is 95% under exchangeability and a symmetric
fitting procedure. The complete model-family selection and seeded tuning
workflow does not independently establish that theorem's assumptions, and
coverage is not conditional on each profile or guaranteed in clinical populations.
The 29 held-out intervals cover 29/29 ages with mean width 7.24 years.
The full-model point prediction need not be the interval midpoint.

Optional CLI clipping requires both `--clip-lower` and `--clip-upper` and retains
raw values in additional columns. Clipping is not used for the reported metrics
and does not establish coverage outside the chosen bounds.

## Tests

```bash
python -m pytest -q
```

Tests cover data preparation, artifact integrity, held-out interval reproduction,
input validation, fitting/serialization, candidate checkpoint reuse and figure
regeneration. They exercise small fits and saved models; the expensive full
model comparison and ablation are explicit reproduction steps above.
