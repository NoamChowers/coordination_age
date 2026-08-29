#!/usr/bin/env python3
"""Create an aggregate dataset from a validated wide all-trials table.

The output contains aggregate features, ``n_trials_*`` diagnostics, and masks
each count-backed aggregate when fewer than the requested number of valid
trials remain. The legacy all-trials schema cannot represent ``meandistance``;
``--supplemental-aggregates`` supplies that one column and subject alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import (
    MINIMUM_VALID_TRIALS,
    build_aggregate_dataset,
    read_table,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Validated all-trials CSV.")
    parser.add_argument(
        "--supplemental-aggregates",
        type=Path,
        required=True,
        help="Aggregate CSV supplying the non-reconstructable meandistance column.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output aggregate CSV.")
    parser.add_argument(
        "--minimum-valid-trials",
        type=int,
        default=MINIMUM_VALID_TRIALS,
        help="Mask each count-backed aggregate below this count (default: 5).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    valid_all_trials = read_table(args.input)
    supplemental = read_table(args.supplemental_aggregates)
    aggregate, summary = build_aggregate_dataset(
        valid_all_trials,
        supplemental,
        minimum_valid_trials=args.minimum_valid_trials,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
