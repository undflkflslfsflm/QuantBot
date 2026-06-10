from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from matvm_ibkr_bot import IbkrClient, IbkrConfig, compute_target_shares_from_weights


def test_compute_target_shares_from_weights_rounds_down_and_skips_bad_prices() -> None:
    weights = pd.Series({"VTI": 0.50, "TLT": 0.25, "SGOV": 0.25, "BAD": 0.10})
    prices = {"VTI": 33.0, "TLT": 20.0, "SGOV": 100.0, "BAD": 0.0}

    got = compute_target_shares_from_weights(weights, equity=1_000.0, last_prices=prices)

    assert got == {"VTI": 15, "TLT": 12, "SGOV": 2}


def test_place_rebalance_orders_computes_sell_then_buy_deltas(monkeypatch) -> None:
    client = IbkrClient(IbkrConfig(use_limit_orders=True, limit_buffer_bps=100.0))
    client.ib = SimpleNamespace()

    monkeypatch.setattr(client, "current_positions", lambda account: {"VTI": 10.0, "TLT": 0.0})
    monkeypatch.setattr(client, "last_prices", lambda symbols: {"VTI": 100.0, "TLT": 50.0})
    monkeypatch.setattr(
        client,
        "qualify_contracts",
        lambda symbols: [SimpleNamespace(symbol=symbol) for symbol in symbols],
    )

    orders = client.place_rebalance_orders(
        account="DU123",
        target_shares={"VTI": 5, "TLT": 3},
        dry_run=True,
    )

    assert [order["symbol"] for order in orders] == ["VTI", "TLT"]
    assert [order["delta"] for order in orders] == [-5, 3]
    assert orders[0]["orderType"] == "LMT"
    assert orders[0]["totalQuantity"] == 5
    assert orders[0]["lmtPrice"] == 99.0
    assert orders[1]["orderType"] == "LMT"
    assert orders[1]["totalQuantity"] == 3
    assert orders[1]["lmtPrice"] == 50.5
    assert all(order["dry_run"] for order in orders)


def test_place_rebalance_orders_respects_max_order_count(monkeypatch) -> None:
    client = IbkrClient(IbkrConfig(max_orders=1))
    client.ib = SimpleNamespace()

    monkeypatch.setattr(client, "current_positions", lambda account: {})
    monkeypatch.setattr(client, "last_prices", lambda symbols: {"VTI": 100.0, "TLT": 50.0})
    monkeypatch.setattr(
        client,
        "qualify_contracts",
        lambda symbols: [SimpleNamespace(symbol=symbol) for symbol in symbols],
    )

    try:
        client.place_rebalance_orders(
            account="DU123",
            target_shares={"VTI": 1, "TLT": 1},
            dry_run=True,
        )
    except RuntimeError as exc:
        assert "max_orders=1" in str(exc)
    else:
        raise AssertionError("Expected max order guard to raise RuntimeError")

