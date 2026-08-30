# Data preparation scripts

This pipeline takes the laboratory's all-trials feature delivery and produces
an aggregate dataset that conforms to the model's required data-validity
rules: the blacklist is authoritative, copied trials do not contribute to
aggregates, and repeated-trial aggregates from `pressing`, `reach-to-grasp`,
and `lift-object` are set to `NaN` when fewer than five valid trials remain.

The resulting aggregate dataset is then reduced to the fixed 32-feature model
matrix and aligned age target. Data preparation remains separate from fitting
so the FIT and INFER scripts receive inputs that already satisfy these domain
rules.

## 1. Enforce the laboratory blacklist and remove copied trials

```bash
python scripts/data/validate_all_trials.py \
  --input Data/bothexperiments_all_trials.csv \
  --blacklist Data/blacklist.csv \
  --output Data/bothexperiments_all_trials_valid.csv \
  --audit-output outputs/all_trials_corrections.csv
```

The normalized blacklist may be CSV or Parquet and must contain `subject`,
`task`, `condition`, `trial`, and the binary `blacklisted` decision. The lab's
decision is authoritative.

Copied repetitions are detected only from the delivered all-trials features:

- isolated copies require an exact task-specific feature-vector match;
- a copied second-half block requires at least three visible matching pairs
  and at least 80% agreement among comparable pairs.

The audit distinguishes blacklist exclusions, direct matches, and inferred
blocks. Embedded aggregates are recomputed only when contributing values move.

## 2. Aggregate valid repetitions and require at least five

```bash
python scripts/data/aggregate_all_trials.py \
  --input Data/bothexperiments_all_trials_valid.csv \
  --supplemental-aggregates Data/bothexperiments_lab.csv \
  --output Data/bothexperiments.csv
```

Reach-to-grasp and lift-object features are aggregated by their established
medians. Pressing retains the synchronized laboratory aggregate because the
delivered repetition columns do not cover every trial used upstream.

`meandistance` has no repetition-level columns in the delivered all-trials
table, so its laboratory aggregate is supplied by
`--supplemental-aggregates`. The observable reach-to-grasp features have
identical trial availability for every subject and repetition; their shared
availability is used to apply the same minimum-five rule to `meandistance`.
The script aborts if those observable reach-to-grasp counts disagree.

## 3. Create the fixed modeling matrix and target

```bash
python scripts/data/create_modeling_inputs.py \
  --aggregate-csv Data/bothexperiments.csv \
  --x-output Data/X.csv \
  --y-output Data/y.csv \
  --metadata-output Data/modeling_input_metadata.json
```

This selects the frozen 32 predictors, applies the established invalid-timing
rule, and converts age in months to the one-column `AgeInYears` target. `X.csv`
and `y.csv` are row-aligned and are the direct inputs to the production FIT
script.

## Validation on the delivered dataset

- The correction heuristic identifies the three supported copied trial blocks
  and one isolated copied trial, totaling the 31 documented copied trials.
- The corrected all-trials table matches the finalized validation reference
  with zero differing cells.
- The aggregate output matches the finalized censored reference with zero
  differing cells.
