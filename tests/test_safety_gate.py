from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from matvm_ibkr_bot import (
    IbkrConfig,
    SafetyGate,
    SafetyGateConfig,
    SafetyGateFailure,
)


def _fresh_prices(config, today: str = "2026-06-10") -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(today), periods=config.min_history_days + 5)
    x = np.arange(len(idx), dtype=float)
    data = {}
    for i, ticker in enumerate(config.risk_tickers):
        data[ticker] = (100.0 + i) * np.exp((0.0005 + i * 0.00005) * x)
    if config.cash_ticker:
        data[config.cash_ticker] = 100.0 * np.exp(0.00005 * x)
    return pd.DataFrame(data, index=idx)


def _assert_gate_fails(capsys, callback, expected: str) -> None:
    with pytest.raises(SafetyGateFailure):
        callback()
    err = capsys.readouterr().err
    assert "SAFETY CHECK FAILED" in err
    assert expected in err


def test_halt_file_aborts(tmp_path, capsys) -> None:
    (tmp_path / "HALT").write_text("", encoding="utf-8")
    gate = SafetyGate(SafetyGateConfig(), tmp_path)

    _assert_gate_fails(capsys, gate.check_halt_file, "HALT")


def test_stale_data_aborts(tmp_path, capsys, small_config) -> None:
    gate = SafetyGate(SafetyGateConfig(max_stale_business_days=3), tmp_path)
    prices = _fresh_prices(small_config, today="2026-05-29")

    _assert_gate_fails(
        capsys,
        lambda: gate.check_data_freshness(prices, today=pd.Timestamp("2026-06-10")),
        "last price bar is stale",
    )


def test_data_audit_errors_abort(tmp_path, capsys, small_config) -> None:
    gate = SafetyGate(SafetyGateConfig(), tmp_path)
    prices = _fresh_prices(small_config).drop(columns=[small_config.risk_tickers[0]])

    _assert_gate_fails(
        capsys,
        lambda: gate.check_data_audit(prices, small_config),
        "data audit errors",
    )


def test_data_audit_warnings_can_abort_when_configured(tmp_path, capsys, small_config) -> None:
    gate = SafetyGate(SafetyGateConfig(allow_data_warnings=False), tmp_path)
    prices = _fresh_prices(small_config)
    prices.loc[prices.index[-1], small_config.risk_tickers[0]] *= 1.50

    _assert_gate_fails(
        capsys,
        lambda: gate.check_data_audit(prices, small_config),
        "data audit warnings are not allowed",
    )


def test_first_run_without_accepted_positions_aborts(tmp_path, capsys) -> None:
    gate = SafetyGate(SafetyGateConfig(accept_positions=False), tmp_path)

    _assert_gate_fails(
        capsys,
        lambda: gate.check_position_reconciliation(
            actual_positions={"VTI": 10.0},
            expected_positions={},
            baseline_accepted=False,
        ),
        "no persisted broker-position baseline",
    )


def test_first_run_with_accept_positions_snapshots_baseline(tmp_path, capsys) -> None:
    gate = SafetyGate(SafetyGateConfig(accept_positions=True), tmp_path)

    snapshot = gate.check_position_reconciliation(
        actual_positions={"VTI": 10.0, "SGOV": 2.0},
        expected_positions={},
        baseline_accepted=False,
    )

    assert snapshot == {"SGOV": 2.0, "VTI": 10.0}
    assert "accepting current broker positions" in capsys.readouterr().err


def test_position_mismatch_aborts_with_diff(tmp_path, capsys) -> None:
    gate = SafetyGate(SafetyGateConfig(position_tolerance_shares=0.5), tmp_path)

    _assert_gate_fails(
        capsys,
        lambda: gate.check_position_reconciliation(
            actual_positions={"VTI": 11.0},
            expected_positions={"VTI": 10.0},
            baseline_accepted=True,
        ),
        "VTI: actual=11, expected=10, diff=1",
    )


def test_rebalance_lock_aborts_without_force(tmp_path, capsys) -> None:
    gate = SafetyGate(SafetyGateConfig(force_rebalance=False), tmp_path)

    _assert_gate_fails(
        capsys,
        lambda: gate.check_rebalance_lock(
            current_period="2026-06-06/2026-06-12",
            last_rebalance_period="2026-06-06/2026-06-12",
        ),
        "already traded",
    )


def test_single_order_notional_cap_aborts(tmp_path, capsys) -> None:
    gate = SafetyGate(SafetyGateConfig(max_order_notional_frac=0.25), tmp_path)

    _assert_gate_fails(
        capsys,
        lambda: gate.check_order_caps_and_limit_prices(
            current_positions={},
            target_shares={"VTI": 3},
            last_prices={"VTI": 100.0},
            equity=1_000.0,
            ibkr_cfg=IbkrConfig(),
        ),
        "order notional",
    )


def test_total_turnover_cap_aborts(tmp_path, capsys) -> None:
    gate = SafetyGate(
        SafetyGateConfig(max_order_notional_frac=1.0, max_turnover_frac=0.30),
        tmp_path,
    )

    _assert_gate_fails(
        capsys,
        lambda: gate.check_order_caps_and_limit_prices(
            current_positions={},
            target_shares={"VTI": 2, "TLT": 2},
            last_prices={"VTI": 100.0, "TLT": 100.0},
            equity=1_000.0,
            ibkr_cfg=IbkrConfig(),
        ),
        "total turnover",
    )


def test_limit_price_sanity_aborts(tmp_path, capsys) -> None:
    gate = SafetyGate(SafetyGateConfig(limit_price_max_deviation=0.05), tmp_path)

    _assert_gate_fails(
        capsys,
        lambda: gate.check_order_caps_and_limit_prices(
            current_positions={},
            target_shares={"VTI": 1},
            last_prices={"VTI": 100.0},
            equity=10_000.0,
            ibkr_cfg=IbkrConfig(limit_buffer_bps=600.0),
        ),
        "computed limit price",
    )


def test_fully_clean_gate_passes(tmp_path, small_config) -> None:
    gate = SafetyGate(SafetyGateConfig(), tmp_path)
    prices = _fresh_prices(small_config)

    gate.check_halt_file()
    gate.check_data_freshness(prices, today=pd.Timestamp("2026-06-10"))
    audit = gate.check_data_audit(prices, small_config)
    snapshot = gate.check_position_reconciliation(
        actual_positions={"VTI": 10.0, "SGOV": 2.0},
        expected_positions={"VTI": 10.0, "SGOV": 2.0},
        baseline_accepted=True,
    )
    gate.check_rebalance_lock(
        current_period="2026-06-06/2026-06-12",
        last_rebalance_period="2026-05-30/2026-06-05",
    )
    gate.check_order_caps_and_limit_prices(
        current_positions={"VTI": 10.0},
        target_shares={"VTI": 11, "TLT": 1},
        last_prices={"VTI": 100.0, "TLT": 90.0},
        equity=10_000.0,
        ibkr_cfg=IbkrConfig(limit_buffer_bps=10.0),
    )

    assert not audit.has_issues()
    assert snapshot is None

