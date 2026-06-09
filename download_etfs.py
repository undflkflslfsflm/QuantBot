from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["VTI", "VXUS", "TLT", "IAU", "PDBC", "SGOV"]
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_DATA_DIR = Path("data/live")
CA_BUNDLE_ENV = "QUANTBOT_CA_BUNDLE"
SOURCE_CHOICES = {"auto", "yfinance", "yahoo-chart"}


def _configure_ca_bundle(path: Optional[str]) -> Optional[str]:
    ca_bundle = path or os.environ.get(CA_BUNDLE_ENV)
    if not ca_bundle:
        return None

    ca_path = Path(ca_bundle).expanduser().resolve()
    if not ca_path.exists():
        raise FileNotFoundError(f"CA bundle does not exist: {ca_path}")

    ca_value = str(ca_path)
    os.environ["SSL_CERT_FILE"] = ca_value
    os.environ["REQUESTS_CA_BUNDLE"] = ca_value
    os.environ["CURL_CA_BUNDLE"] = ca_value
    return ca_value


def _parse_tickers(raw: str) -> List[str]:
    tickers = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not tickers:
        raise ValueError("At least one ticker is required")
    return list(dict.fromkeys(tickers))


def _download_yfinance(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: no data returned")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", ticker) in df.columns:
            adj = df[("Adj Close", ticker)]
        elif "Adj Close" in df.columns.get_level_values(0):
            adj = df["Adj Close"].iloc[:, 0]
        else:
            raise RuntimeError(f"{ticker}: yfinance response missing Adj Close")
    else:
        if "Adj Close" not in df.columns:
            raise RuntimeError(f"{ticker}: yfinance response missing Adj Close")
        adj = df["Adj Close"]

    out = adj.to_frame(name="adj_close")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index().dropna()
    out = out[out["adj_close"] > 0]
    if out.empty:
        raise RuntimeError(f"{ticker}: no positive adjusted-close values")

    out.index = out.index.strftime("%Y-%m-%d")
    out.index.name = "date"
    return out.reset_index()


def _download_yahoo_chart(ticker: str, start: str, end: Optional[str]) -> pd.DataFrame:
    start_dt = pd.Timestamp(start).to_pydatetime().replace(tzinfo=timezone.utc)
    if end:
        end_dt = pd.Timestamp(end).to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        end_dt = datetime.now(timezone.utc) + timedelta(days=1)

    params = urllib.parse.urlencode(
        {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    errors = chart.get("error")
    if errors:
        raise RuntimeError(f"{ticker}: Yahoo chart error: {errors}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"{ticker}: Yahoo chart returned no result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    adjclose_blocks = result.get("indicators", {}).get("adjclose") or []
    if not timestamps or not adjclose_blocks:
        raise RuntimeError(f"{ticker}: Yahoo chart response missing adjusted close")

    adj_values = adjclose_blocks[0].get("adjclose") or []
    if len(timestamps) != len(adj_values):
        raise RuntimeError(f"{ticker}: Yahoo chart timestamp/price length mismatch")

    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).strftime("%Y-%m-%d")
    out = pd.DataFrame({"date": dates, "adj_close": adj_values})
    out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    out = out.dropna()
    out = out[out["adj_close"] > 0]
    if out.empty:
        raise RuntimeError(f"{ticker}: no positive adjusted-close values")

    return out


def _download_adjusted_close(
    ticker: str,
    start: str,
    end: Optional[str],
    source: str,
) -> Tuple[pd.DataFrame, str]:
    source = source.strip().lower()
    if source not in SOURCE_CHOICES:
        raise ValueError(f"Unknown source: {source}")

    if source in {"auto", "yahoo-chart"}:
        try:
            return _download_yahoo_chart(ticker=ticker, start=start, end=end), "yahoo-chart"
        except Exception:
            if source == "yahoo-chart":
                raise

    if source == "yfinance" or source == "auto":
        try:
            return _download_yfinance(ticker=ticker, start=start, end=end), "yfinance"
        except Exception:
            if source == "yfinance":
                raise
            raise


def download_all(
    tickers: List[str],
    start: str,
    end: Optional[str],
    data_dir: Path,
    ca_bundle: Optional[str],
    source: str,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata: Dict[str, object] = {
        "source": source,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "adjusted": True,
        "price_column": "adj_close",
        "start": start,
        "end": end,
        "ca_bundle": ca_bundle,
        "tickers": {},
    }

    failures: List[str] = []
    for ticker in tickers:
        print(f"Downloading {ticker} ...")
        try:
            df, actual_source = _download_adjusted_close(
                ticker=ticker,
                start=start,
                end=end,
                source=source,
            )
            out_path = data_dir / f"{ticker}.csv"
            df.to_csv(out_path, index=False, lineterminator="\n")

            ticker_meta = {
                "ticker": ticker,
                "source": actual_source,
                "downloaded_at_utc": metadata["downloaded_at_utc"],
                "adjusted": True,
                "price_column": "adj_close",
                "rows": int(len(df)),
                "first_date": str(df["date"].iloc[0]),
                "last_date": str(df["date"].iloc[-1]),
                "file": out_path.name,
            }
            metadata["tickers"][ticker] = ticker_meta  # type: ignore[index]
            print(f"  wrote {out_path} ({len(df)} rows, source={actual_source})")
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            print(f"  failed: {exc}")

    metadata_path = data_dir / "_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    if failures:
        raise SystemExit("Download failed:\n" + "\n".join(failures))

    print(f"Metadata: {metadata_path}")
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download adjusted ETF closes for QuantBot.")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker list",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_CHOICES),
        default="auto",
        help="Data source. auto tries Yahoo chart API, then yfinance.",
    )
    parser.add_argument(
        "--ca-bundle",
        default=None,
        help=f"Optional PEM bundle path. Also supported via {CA_BUNDLE_ENV}.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ca_bundle = _configure_ca_bundle(args.ca_bundle)
    tickers = _parse_tickers(args.tickers)
    download_all(
        tickers=tickers,
        start=args.start,
        end=args.end,
        data_dir=Path(args.data_dir),
        ca_bundle=ca_bundle,
        source=args.source,
    )


if __name__ == "__main__":
    main()
