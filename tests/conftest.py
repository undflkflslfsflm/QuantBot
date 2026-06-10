from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matvm_ibkr_bot import MatvmConfig


RISK_TICKERS = ["VTI", "VXUS", "TLT", "IAU", "PDBC"]
CASH_TICKER = "SGOV"


def _business_index(periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=periods)


def _price_frame(kind: str, periods: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _business_index(periods)
    x = np.arange(periods, dtype=float)

    if kind == "trending_up":
        base = 100.0 * np.exp(0.0015 * x)
        data = {
            "VTI": base,
            "VXUS": 95.0 * np.exp(0.0012 * x),
            "TLT": 90.0 * np.exp(0.0008 * x),
            "IAU": 80.0 * np.exp(0.0010 * x),
            "PDBC": 70.0 * np.exp(0.0006 * x),
        }
    elif kind == "trending_down":
        base = 100.0 * np.exp(-0.0015 * x)
        data = {
            "VTI": base,
            "VXUS": 98.0 * np.exp(-0.0012 * x),
            "TLT": 96.0 * np.exp(-0.0008 * x),
            "IAU": 94.0 * np.exp(-0.0010 * x),
            "PDBC": 92.0 * np.exp(-0.0006 * x),
        }
    elif kind == "flat_choppy":
        data = {}
        for i, ticker in enumerate(RISK_TICKERS):
            noise = rng.normal(0.0, 0.002 + i * 0.0002, size=periods)
            data[ticker] = 100.0 * np.exp(np.cumsum(noise))
    elif kind == "sharp_drawdown":
        base = 100.0 * np.exp(0.0010 * x)
        shock = np.ones(periods)
        shock[periods // 2 :] *= 0.80
        data = {
            "VTI": base * shock,
            "VXUS": 97.0 * np.exp(0.0008 * x) * shock,
            "TLT": 94.0 * np.exp(0.0005 * x),
            "IAU": 91.0 * np.exp(0.0007 * x),
            "PDBC": 88.0 * np.exp(0.0004 * x) * shock,
        }
    else:
        raise ValueError(f"unknown synthetic price kind: {kind}")

    data[CASH_TICKER] = 100.0 * np.exp(0.00005 * x)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def trending_up_prices() -> pd.DataFrame:
    return _price_frame("trending_up")


@pytest.fixture
def trending_down_prices() -> pd.DataFrame:
    return _price_frame("trending_down")


@pytest.fixture
def flat_choppy_prices() -> pd.DataFrame:
    return _price_frame("flat_choppy")


@pytest.fixture
def sharp_drawdown_prices() -> pd.DataFrame:
    return _price_frame("sharp_drawdown")


@pytest.fixture
def small_config() -> MatvmConfig:
    return MatvmConfig(
        risk_tickers=list(RISK_TICKERS),
        cash_ticker=CASH_TICKER,
        cash_return_mode="zero",
        cash_return_ticker=None,
        risk_free_mode="zero",
        risk_free_ticker=None,
        vol_window=10,
        sma_window=20,
        momentum_windows=(5, 10, 20),
        min_history_days=25,
        confirm_signals=1,
        breadth_min=1,
        weight_cap=0.40,
        vol_target=1.00,
        rebalance_freq="W-FRI",
        rebalance_abs_threshold=0.0,
        rebalance_rel_threshold=0.0,
        tcost_bps=0.0,
        slippage_bps=0.0,
    )

