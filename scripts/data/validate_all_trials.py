#!/usr/bin/env python3
"""Enforce blacklist decisions and heuristically remove copied trials.

The blacklist input must be a normalized CSV or Parquet table containing
``subject``, ``task``, ``condition``, ``trial``, and ``blacklisted``. Copied
trials are detected from exact task-specific feature-vector matches in the
all-trials table. A complete second-half block is inferred only when at least
three visible pairs support the half-table offset with at least 80% agreement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation import correct_all_trials, read_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Wide all-trials CSV.")
    parser.add_argument(
        "--blacklist",
        type=Path,
        required=True,
        help="Normalized long-form blacklist in CSV or Parquet format.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Corrected all-trials CSV.")
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Optional cell-level correction audit CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    all_trials = read_table(args.input)
    blacklist = read_table(args.blacklist)
    corrected, audit, summary = correct_all_trials(all_trials, blacklist)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(args.output, index=False)
    if args.audit_output is not None:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_output, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
