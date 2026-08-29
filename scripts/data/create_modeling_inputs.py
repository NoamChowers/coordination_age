#!/usr/bin/env python3
"""Create the frozen 32-feature X.csv and one-column AgeInYears y.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_inputs import build_model_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--x-output", type=Path, required=True)
    parser.add_argument("--y-output", type=Path, required=True)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional JSON containing subject order and preprocessing counts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate = pd.read_csv(args.aggregate_csv)
    X, y, metadata = build_model_inputs(aggregate)
    args.x_output.parent.mkdir(parents=True, exist_ok=True)
    args.y_output.parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.x_output, index=False)
    y.to_csv(args.y_output, index=False)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({key: value for key, value in metadata.items() if key != "subject_order"}, indent=2, sort_keys=True))
    print(f"Wrote {args.x_output}")
    print(f"Wrote {args.y_output}")


if __name__ == "__main__":
    main()
