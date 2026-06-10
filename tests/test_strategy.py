from __future__ import annotations

import json

import numpy as np
import pandas as pd

from matvm_ibkr_bot import MatvmConfig, MatvmStrategy, StrategyState, load_state, save_state


def test_target_weights_are_nonnegative_sum_to_one_and_respect_cap(
    trending_up_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    cfg = small_config
    strategy = MatvmStrategy(cfg)
    asof = trending_up_prices.index[80]

    weights = strategy.generate_target_weights(
        trending_up_prices,
        asof=asof,
        equity=100_000.0,
        peak_equity=100_000.0,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= -1e-12).all()
    assert (weights[cfg.risk_tickers] <= cfg.weight_cap + 1e-12).all()


def test_hysteresis_requires_persistent_signal_before_allocation(
    trending_up_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    cfg = small_config
    cfg.confirm_signals = 2
    cfg.weight_cap = 1.0
    strategy = MatvmStrategy(cfg)

    first = strategy.generate_target_weights(
        trending_up_prices,
        asof=trending_up_prices.index[80],
        equity=100_000.0,
        peak_equity=100_000.0,
    )
    second = strategy.generate_target_weights(
        trending_up_prices,
        asof=trending_up_prices.index[81],
        equity=100_000.0,
        peak_equity=100_000.0,
    )

    assert first[cfg.risk_tickers].sum() == 0.0
    assert second[cfg.risk_tickers].sum() > 0.0


def test_drawdown_overlay_halves_then_exits_and_ramps_back(
    trending_up_prices: pd.DataFrame,
    small_config: MatvmConfig,
) -> None:
    cfg = small_config
    cfg.weight_cap = 1.0
    cfg.dd_half = 0.05
    cfg.dd_safe = 0.10
    cfg.dd_exit = 0.02
    cfg.safe_mode_ramp_weeks = 1
    strategy = MatvmStrategy(cfg)
    asofs = list(trending_up_prices.index[80:84])

    half = strategy.generate_target_weights(
        trending_up_prices,
        asof=asofs[0],
        equity=94_000.0,
        peak_equity=100_000.0,
    )
    safe = strategy.generate_target_weights(
        trending_up_prices,
        asof=asofs[1],
        equity=88_000.0,
        peak_equity=100_000.0,
    )
    ramp = strategy.generate_target_weights(
        trending_up_prices,
        asof=asofs[2],
        equity=100_000.0,
        peak_equity=100_000.0,
    )
    recovered = strategy.generate_target_weights(
        trending_up_prices,
        asof=asofs[3],
        equity=100_000.0,
        peak_equity=100_000.0,
    )

    assert half[cfg.cash_ticker] >= 0.50
    assert safe[cfg.cash_ticker] == 1.0
    assert 0.49 <= ramp[cfg.risk_tickers].sum() <= 0.51
    assert recovered[cfg.risk_tickers].sum() > ramp[cfg.risk_tickers].sum()


def test_state_round_trip_preserves_all_fields(tmp_path) -> None:
    path = tmp_path / "state.json"
    state = StrategyState(
        held={"VTI": 1, "VXUS": 0},
        enter_count={"VTI": 2, "VXUS": 1},
        exit_count={"VTI": 0, "VXUS": 3},
        safe_mode=True,
        ramp_weeks_remaining=2,
        breadth_good_count=4,
        last_signal_date="2026-06-10",
        last_rebalance_period="2026-06-06/2026-06-12",
        expected_positions={"VTI": 12.0, "SGOV": 3.0},
        positions_baseline_accepted=True,
    )

    save_state(path, state)
    loaded = load_state(path)

    assert loaded == state
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "saved_at_utc" in data
