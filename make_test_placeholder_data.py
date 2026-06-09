from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).parent / "data" / "test_fake"
TICKERS = ["VTI", "VXUS", "TLT", "IAU", "PDBC", "SGOV"]


def main() -> None:
    np.random.seed(42)
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start="2014-01-01", end="2025-12-31", freq="B")
    downloaded_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "source": "make_test_placeholder_data.py",
        "downloaded_at_utc": downloaded_at,
        "adjusted": False,
        "price_column": "adj_close",
        "warning": "Synthetic placeholder data for tests only. Do not use for strategy evaluation.",
        "tickers": {},
    }

    for ticker in TICKERS:
        price = 100.0
        prices = []

        for _ in range(len(dates)):
            price *= 1 + np.random.normal(0.0003, 0.01)
            prices.append(price)

        df = pd.DataFrame({"date": dates, "adj_close": prices})
        df.to_csv(BASE_DIR / f"{ticker}.csv", index=False, lineterminator="\n")
        metadata["tickers"][ticker] = {
            "ticker": ticker,
            "source": "make_test_placeholder_data.py",
            "downloaded_at_utc": downloaded_at,
            "adjusted": False,
            "price_column": "adj_close",
            "rows": int(len(df)),
            "first_date": str(df["date"].iloc[0].date()),
            "last_date": str(df["date"].iloc[-1].date()),
            "file": f"{ticker}.csv",
        }

    (BASE_DIR / "_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Placeholder CSV files created in {BASE_DIR}")


if __name__ == "__main__":
    main()
