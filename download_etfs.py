import os

# Point these to the PEM you created/exported (corporate root, or a merged bundle)
CORP_CA_PEM = r"C:\certs\corp-root.pem"

os.environ["SSL_CERT_FILE"] = CORP_CA_PEM
os.environ["REQUESTS_CA_BUNDLE"] = CORP_CA_PEM
os.environ["CURL_CA_BUNDLE"] = CORP_CA_PEM

import pandas as pd
import yfinance as yf
from pathlib import Path

TICKERS = ["VTI", "VXUS", "TLT", "IAU", "PDBC", "SGOV"]
START_DATE = "2005-01-01"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker} ...")

    df = yf.download(
        ticker,
        start=START_DATE,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"  ⚠ no data returned for {ticker}")
        continue

    out = df[["Adj Close"]].copy()
    out.index = pd.to_datetime(out.index).strftime("%Y-%m-%d")
    out.reset_index(inplace=True)
    out.columns = ["date", "close"]

    out.to_csv(DATA_DIR / f"{ticker}.csv", index=False, lineterminator="\n")
    print(f"  ✓ wrote data/{ticker}.csv ({len(out)} rows)")

print("Done.")

