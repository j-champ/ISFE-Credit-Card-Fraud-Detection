from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]

def validate(path: Path) -> dict:
    df = pd.read_csv(path)
    required = set(EXPECTED_COLUMNS)
    missing_columns = sorted(required - set(df.columns))
    extra_columns = sorted(set(df.columns) - required)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if extra_columns:
        print(f"WARNING: unexpected columns: {extra_columns}")

    if df["Class"].isna().any():
        raise ValueError("Target column Class contains missing values")
    labels = set(df["Class"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0 and 1; found {sorted(labels)}")

    counts = df["Class"].value_counts().sort_index()
    n = len(df)
    duplicates = int(df.duplicated().sum())
    duplicate_class_conflicts = 0
    if duplicates:
        # Count identical feature rows (excluding target) that have more than one label.
        duplicate_groups = df.groupby(df.columns.drop("Class").tolist(), dropna=False)["Class"].nunique()
        duplicate_class_conflicts = int((duplicate_groups > 1).sum())

    report = {
        "file": str(path.resolve()),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "rows": int(n),
        "columns": int(df.shape[1]),
        "feature_count": int(df.shape[1] - 1),
        "target": "Class",
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        "duplicate_rows": duplicates,
        "duplicate_feature_groups_with_conflicting_labels": duplicate_class_conflicts,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_percentages": {str(k): round(float(v / n * 100), 6) for k, v in counts.items()},
        "imbalance_ratio_majority_to_minority": round(float(counts.max() / counts.min()), 3),
        "amount_summary": {k: float(v) for k, v in df["Amount"].describe().items()},
        "time_summary": {k: float(v) for k, v in df["Time"].describe().items()},
    }
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the ISFE credit-card dataset")
    parser.add_argument("--data", default="data/raw/creditcard.csv", help="Path to CSV")
    args = parser.parse_args()
    result = validate(Path(args.data))
    print(json.dumps(result, indent=2))
