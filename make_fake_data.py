import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

BASE_DIR = Path(__file__).parent / "data"
BASE_DIR.mkdir(exist_ok=True)

tickers = ["VTI", "VXUS", "TLT", "IAU", "PDBC", "SGOV"]

dates = pd.date_range(start="2014-01-01", end="2025-12-31", freq="B")

for ticker in tickers:
    price = 100.0
    prices = []

    for _ in range(len(dates)):
        # small daily random walk
        price *= (1 + np.random.normal(0.0003, 0.01))
        prices.append(price)

    df = pd.DataFrame({
        "date": dates,
        "close": prices
    })

    df.to_csv(BASE_DIR / f"{ticker}.csv", index=False)

print("Fake CSV files created in ./data/")
