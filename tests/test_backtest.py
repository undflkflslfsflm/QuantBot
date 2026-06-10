from __future__ import annotations

import numpy as np
import pandas as pd

from matvm_ibkr_bot import MatvmConfig, audit_price_data, backtest


def test_backtest_is_deterministic(
    trending_up_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    first = backtest(trending_up_prices, small_config)
    second = backtest(trending_up_prices, small_config)

    pd.testing.assert_series_equal(first.equity_curve, second.equity_curve, check_exact=True)
    pd.testing.assert_frame_equal(first.weights, second.weights, check_exact=True)


def test_transaction_costs_reduce_final_equity_on_uptrend(
    trending_up_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    no_cost = small_config
    with_cost = MatvmConfig(**{**no_cost.__dict__, "tcost_bps": 25.0, "slippage_bps": 25.0})

    no_cost_result = backtest(trending_up_prices, no_cost)
    with_cost_result = backtest(trending_up_prices, with_cost)

    assert no_cost_result.equity_curve.iloc[-1] >= with_cost_result.equity_curve.iloc[-1]


def test_backtest_no_lookahead_weights_match_truncated_runs(
    sharp_drawdown_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    full = backtest(sharp_drawdown_prices, small_config)
    check_dates = [
        sharp_drawdown_prices.index[90],
        sharp_drawdown_prices.index[140],
        sharp_drawdown_prices.index[210],
        sharp_drawdown_prices.index[280],
    ]

    for dt in check_dates:
        truncated = backtest(sharp_drawdown_prices.loc[:dt], small_config)
        pd.testing.assert_series_equal(
            full.weights.loc[dt],
            truncated.weights.loc[dt],
            check_exact=False,
            atol=1e-12,
            rtol=1e-12,
            check_names=False,
        )


def test_audit_price_data_flags_missing_nan_and_stale_prices(
    trending_up_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    missing = trending_up_prices.drop(columns=[small_config.risk_tickers[0]])
    missing_audit = audit_price_data(missing, small_config)
    assert any("missing price column" in msg for msg in missing_audit.errors)

    with_nan = trending_up_prices.copy()
    with_nan.loc[with_nan.index[30:35], small_config.risk_tickers[1]] = np.nan
    nan_audit = audit_price_data(with_nan, small_config)
    assert any("missing price values" in msg for msg in nan_audit.warnings)

    stale = trending_up_prices.copy()
    stale[small_config.risk_tickers[2]] = 100.0
    stale_audit = audit_price_data(stale, small_config)
    assert any("suspiciously flat price history" in msg for msg in stale_audit.warnings)

