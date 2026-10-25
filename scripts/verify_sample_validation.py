#!/usr/bin/env python3
"""Compare a sample-validation run with the published expected predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = (
    "recurrence_probability",
    "complication_probability",
)
PREDICTION_COLUMNS = (
    "recurrence_prediction",
    "complication_prediction",
)
GATE_COLUMNS = tuple(
    f"{task}_gate_{modality}"
    for task in ("recurrence", "complication")
    for modality in ("ct", "tumor", "l3", "clinical")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", required=True)
    parser.add_argument(
        "--expected",
        default="examples/sample_validation_cohort/expected_predictions.csv",
    )
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    actual = pd.read_csv(Path(args.actual), dtype={"case_id": str})
    expected = pd.read_csv(Path(args.expected), dtype={"case_id": str})
    if actual["case_id"].tolist() != expected["case_id"].tolist():
        raise SystemExit("FAIL: case IDs or row order do not match")

    numeric_columns = [*PROBABILITY_COLUMNS, *GATE_COLUMNS]
    missing = [
        column
        for column in [*numeric_columns, *PREDICTION_COLUMNS]
        if column not in actual.columns or column not in expected.columns
    ]
    if missing:
        raise SystemExit("FAIL: missing comparison columns: " + ", ".join(missing))

    absolute_difference = np.abs(
        actual[numeric_columns].to_numpy(float)
        - expected[numeric_columns].to_numpy(float)
    )
    maximum_difference = float(absolute_difference.max(initial=0.0))
    labels_match = bool(
        np.array_equal(
            actual[list(PREDICTION_COLUMNS)].to_numpy(int),
            expected[list(PREDICTION_COLUMNS)].to_numpy(int),
        )
    )
    passed = labels_match and maximum_difference <= args.atol
    report = {
        "status": "PASS" if passed else "FAIL",
        "cases": int(len(actual)),
        "maximum_absolute_difference": maximum_difference,
        "absolute_tolerance": float(args.atol),
        "binary_predictions_match": labels_match,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
