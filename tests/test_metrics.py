from __future__ import annotations

import math

import numpy as np
import pandas as pd

from matvm_ibkr_bot import (
    TRADING_DAYS_PER_YEAR,
    cagr,
    log_returns,
    max_drawdown,
    rolling_realized_vol,
    sharpe_ratio,
    simple_returns,
    sma,
    sortino_ratio,
)


def test_simple_returns_tiny_hand_computed_values() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})

    got = simple_returns(prices)

    np.testing.assert_allclose(got["A"].to_numpy(), [0.0, 0.10, -0.10])


def test_log_returns_tiny_hand_computed_values() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})

    got = log_returns(prices)

    expected = [0.0, math.log(1.10), math.log(0.90)]
    np.testing.assert_allclose(got["A"].to_numpy(), expected)


def test_rolling_realized_vol_uses_population_std_and_annualizes() -> None:
    log_rets = pd.DataFrame({"A": [0.01, 0.03, -0.01]})

    got = rolling_realized_vol(log_rets, window=2)

    expected_last = np.std([0.03, -0.01], ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert math.isnan(got["A"].iloc[0])
    assert got["A"].iloc[-1] == expected_last


def test_sma_tiny_hand_computed_values() -> None:
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]})

    got = sma(prices, window=2)

    assert math.isnan(got["A"].iloc[0])
    np.testing.assert_allclose(got["A"].iloc[1:].to_numpy(), [1.5, 2.5])


def test_max_drawdown_tiny_hand_computed_values() -> None:
    equity = pd.Series([100.0, 120.0, 90.0, 150.0])

    assert max_drawdown(equity) == 0.25


def test_sharpe_ratio_matches_formula() -> None:
    returns = pd.Series([0.01, 0.02, -0.01, 0.00])
    rf = pd.Series([0.001, 0.001, 0.001, 0.001])
    excess = returns - rf

    got = sharpe_ratio(returns, daily_rf=rf)

    expected = excess.mean() / excess.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert got == expected


def test_sortino_ratio_matches_formula() -> None:
    returns = pd.Series([0.03, -0.01, 0.02, -0.02])

    got = sortino_ratio(returns)

    downside = returns[returns < 0.0]
    expected = returns.mean() / downside.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert got == expected


def test_cagr_matches_calendar_year_formula() -> None:
    equity = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2020-01-01", "2021-01-01"]),
    )
    years = 366.0 / 365.25

    assert cagr(equity) == (1.10 ** (1.0 / years) - 1.0)


def test_metric_edge_cases_empty_single_zero_and_nan() -> None:
    empty = pd.Series(dtype=float)
    single = pd.Series([100.0], index=pd.to_datetime(["2020-01-01"]))
    zeros = pd.Series([0.0, 0.0, 0.0])
    prices_with_nan = pd.DataFrame({"A": [100.0, np.nan, 110.0]})

    assert cagr(empty) == 0.0
    assert cagr(single) == 0.0
    assert math.isnan(max_drawdown(empty))
    assert math.isnan(sharpe_ratio(zeros))
    assert math.isnan(sortino_ratio(zeros))
    assert not simple_returns(prices_with_nan).isna().any().any()
    assert not log_returns(prices_with_nan).isna().any().any()

