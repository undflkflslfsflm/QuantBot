from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_dates(
    input_path: Path,
    output_path: Path,
    input_format: str,
    date_column: str,
) -> None:
    df = pd.read_csv(input_path)
    if date_column not in df.columns:
        raise ValueError(f"{input_path} must have a {date_column!r} column")

    df[date_column] = pd.to_datetime(df[date_column], format=input_format).dt.strftime("%Y-%m-%d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, lineterminator="\n")
    print(f"Wrote: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a CSV date column to YYYY-MM-DD.")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--input-format", default="%m/%d/%Y")
    parser.add_argument("--date-column", default="date")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    normalize_dates(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        input_format=args.input_format,
        date_column=args.date_column,
    )


if __name__ == "__main__":
    main()
