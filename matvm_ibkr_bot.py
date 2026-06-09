from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is only used for plots. The bot runs fine without it.
try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# -----------------------------
# Utilities
# -----------------------------

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_MODES = {"zero", "constant", "ticker"}


def _dt_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_timestamp(x: str) -> pd.Timestamp:
    """Parse YYYY-MM-DD into pandas Timestamp."""
    return pd.Timestamp(x).tz_localize(None)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("prices DataFrame index must be a pandas DatetimeIndex")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    return df


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rolling_realized_vol(log_rets: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized realized vol from daily log returns."""
    return log_rets.rolling(window=window, min_periods=window).std(ddof=0) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )


def sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    return prices.rolling(window=window, min_periods=window).mean()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = 1.0 - equity / peak
    return float(dd.max())


def sharpe_ratio(daily_returns: pd.Series, daily_rf: Optional[pd.Series] = None) -> float:
    """Annualized Sharpe using daily returns."""
    r = daily_returns.copy()
    if daily_rf is not None:
        daily_rf = daily_rf.reindex_like(r).fillna(0.0)
        r = r - daily_rf
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((mu / sd) * math.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_returns: pd.Series, daily_rf: Optional[pd.Series] = None) -> float:
    r = daily_returns.copy()
    if daily_rf is not None:
        daily_rf = daily_rf.reindex_like(r).fillna(0.0)
        r = r - daily_rf
    mu = r.mean()
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float((mu / dd) * math.sqrt(TRADING_DAYS_PER_YEAR))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start = equity.index[0]
    end = equity.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _safe_float(x: float, default: float = 0.0) -> float:
    try:
        if x is None or np.isnan(x) or np.isinf(x):
            return default
        return float(x)
    except Exception:
        return default


def _price_cagr(prices: pd.Series) -> float:
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0
    start = prices.index[0]
    end = prices.index[-1]
    if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
        years = (end - start).days / 365.25
        if years > 0:
            return float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1)
    periods = max(len(prices) - 1, 1)
    return float((prices.iloc[-1] / prices.iloc[0]) ** (TRADING_DAYS_PER_YEAR / periods) - 1)


# -----------------------------
# Config and State
# -----------------------------


@dataclass
class MatvmConfig:
    # Universe
    risk_tickers: List[str] = field(
        default_factory=lambda: ["VTI", "VXUS", "TLT", "IAU", "PDBC"]
    )
    cash_ticker: str = "SGOV"  # fallback: "BIL" for longer history

    # Performance benchmark. This is intentionally separate from cash_ticker:
    # cash_ticker is investable, risk_free_* controls excess-return metrics.
    risk_free_mode: str = "zero"  # "zero", "constant", or "ticker"
    risk_free_ticker: Optional[str] = None
    annual_risk_free_rate: float = 0.0

    # Signal windows (trading days)
    vol_window: int = 60
    sma_window: int = 200
    momentum_windows: Tuple[int, int, int] = (63, 126, 252)

    # Hysteresis
    confirm_signals: int = 2

    # Portfolio construction
    weight_cap: float = 0.35
    cov_shrink_lambda: float = 0.50
    vol_target: float = 0.08  # annualized

    # Capital preservation overlay (drawdown thresholds)
    dd_half: float = 0.08
    dd_safe: float = 0.12
    dd_exit: float = 0.06
    breadth_min: int = 3
    safe_mode_ramp_weeks: int = 4

    # Rebalancing / costs
    rebalance_abs_threshold: float = 0.02
    rebalance_rel_threshold: float = 0.15
    tcost_bps: float = 1.0
    slippage_bps: float = 2.0

    # Data warmup
    min_history_days: int = 260  # minimum bars before strategy acts

    def all_tickers(self) -> List[str]:
        return list(dict.fromkeys(self.risk_tickers + [self.cash_ticker]))

    def required_price_tickers(self) -> List[str]:
        tickers = self.all_tickers()
        if str(self.risk_free_mode).strip().lower() == "ticker" and self.risk_free_ticker:
            tickers.append(self.risk_free_ticker)
        return list(dict.fromkeys(tickers))


@dataclass
class StrategyState:
    held: Dict[str, int] = field(default_factory=dict)  # 1 if "eligible/held" via hysteresis
    enter_count: Dict[str, int] = field(default_factory=dict)
    exit_count: Dict[str, int] = field(default_factory=dict)

    # Safe mode latch
    safe_mode: bool = False
    ramp_weeks_remaining: int = 0

    # Breadth confirmation counter (raw eligibility)
    breadth_good_count: int = 0

    # Last processed signal date
    last_signal_date: Optional[str] = None

    def ensure_assets(self, assets: Iterable[str]) -> None:
        for a in assets:
            self.held.setdefault(a, 0)
            self.enter_count.setdefault(a, 0)
            self.exit_count.setdefault(a, 0)


# -----------------------------
# Strategy
# -----------------------------


class MatvmStrategy:
    def __init__(self, config: MatvmConfig, state: Optional[StrategyState] = None):
        self.cfg = config
        self.state = state or StrategyState()
        self.state.ensure_assets(self.cfg.risk_tickers)

    def _compute_q_scores(
        self, prices: pd.DataFrame, asof: pd.Timestamp
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Return (q_score, sigma, sma200) for risk tickers at asof."""
        risk = prices[self.cfg.risk_tickers]
        risk = risk.loc[:asof]
        if len(risk) < max(self.cfg.momentum_windows + (self.cfg.sma_window, self.cfg.vol_window + 1)):
            raise ValueError(
                f"Not enough history up to {asof.date()} for signals. "
                f"Have {len(risk)} days."
            )

        lr = log_returns(risk)
        sigma = rolling_realized_vol(lr, window=self.cfg.vol_window).iloc[-1]
        sigma = sigma.replace(0.0, np.nan)

        sma200 = sma(risk, window=self.cfg.sma_window).iloc[-1]
        p0 = risk.iloc[-1]

        q_parts = []
        for L in self.cfg.momentum_windows:
            pL = risk.iloc[-L]
            m = np.log(p0 / pL)
            denom = sigma * math.sqrt(L / TRADING_DAYS_PER_YEAR)
            qL = m / denom
            q_parts.append(qL)
        q = pd.concat(q_parts, axis=1).mean(axis=1)

        # clean
        q = q.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sigma = sigma.fillna(sigma.median()).fillna(0.0)
        # pandas deprecates fillna(method=...), so use ffill()
        sma200 = sma200.ffill()

        return q, sigma, sma200

    def _raw_eligibility(
        self, prices: pd.DataFrame, asof: pd.Timestamp
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Compute raw eligibility E_raw (before hysteresis)."""
        q, sigma, sma200 = self._compute_q_scores(prices, asof)
        p0 = prices.loc[asof, self.cfg.risk_tickers]
        trend = p0 > sma200
        e_raw = (q > 0.0) & trend
        e_raw = e_raw.astype(int)
        return e_raw, sigma, sma200

    def _update_hysteresis(self, e_raw: pd.Series) -> pd.Series:
        """Update held status using confirm_signals hysteresis and return E_hyst."""
        confirm = self.cfg.confirm_signals
        e_hyst = {}
        for a in self.cfg.risk_tickers:
            raw = int(e_raw.get(a, 0))
            held = int(self.state.held.get(a, 0))

            if held == 1:
                if raw == 0:
                    self.state.exit_count[a] = self.state.exit_count.get(a, 0) + 1
                else:
                    self.state.exit_count[a] = 0

                if self.state.exit_count[a] >= confirm:
                    held = 0
                    self.state.exit_count[a] = 0
                    self.state.enter_count[a] = 0

            else:
                if raw == 1:
                    self.state.enter_count[a] = self.state.enter_count.get(a, 0) + 1
                else:
                    self.state.enter_count[a] = 0

                if self.state.enter_count[a] >= confirm:
                    held = 1
                    self.state.enter_count[a] = 0
                    self.state.exit_count[a] = 0

            self.state.held[a] = held
            e_hyst[a] = held

        return pd.Series(e_hyst, dtype=float)

    @staticmethod
    def _cap_weights(orig: pd.Series, cap: float) -> pd.Series:
        """Apply an upper cap per asset. Leftover remains unallocated (cash)."""
        w = orig.copy()
        w = w[w > 0]
        if w.empty:
            return orig * 0.0
        w = w / w.sum()

        fixed = pd.Series(0.0, index=w.index)
        free = w.index.tolist()
        budget = 1.0

        while True:
            if not free:
                break
            # allocate remaining budget to free assets proportional to original weights
            free_w = w.loc[free]
            if free_w.sum() <= 0:
                alloc = pd.Series(budget / len(free), index=free)
            else:
                alloc = free_w / free_w.sum() * budget

            over = alloc[alloc > cap + 1e-12]
            if over.empty:
                fixed.loc[free] = alloc
                budget = 1.0 - fixed.sum()
                break

            # cap the overweight ones
            for a in over.index:
                fixed[a] = cap
                if a in free:
                    free.remove(a)

            budget = 1.0 - fixed.sum()
            if budget <= 1e-12:
                budget = 0.0
                break

        # fixed may sum < 1 if not enough free capacity -> leftover becomes cash
        out = pd.Series(0.0, index=orig.index, dtype=float)
        out.loc[fixed.index] = fixed
        return out

    def _portfolio_vol(
        self, prices: pd.DataFrame, asof: pd.Timestamp, weights: pd.Series
    ) -> float:
        """Annualized portfolio vol estimate using covariance shrinkage."""
        risk = prices[self.cfg.risk_tickers].loc[:asof]
        lr = log_returns(risk)
        lr_win = lr.iloc[-self.cfg.vol_window :]

        active = weights[weights > 0].index.tolist()
        if not active:
            return 0.0

        w = weights.loc[active].values.reshape(-1, 1)

        if len(active) == 1:
            sigma = rolling_realized_vol(lr[[active[0]]], window=self.cfg.vol_window).iloc[-1, 0]
            return _safe_float(sigma, 0.0)

        X = lr_win[active]
        S = np.cov(X.values, rowvar=False, ddof=0)  # daily covariance

        # shrink toward diagonal
        lam = float(self.cfg.cov_shrink_lambda)
        diag = np.diag(np.diag(S))
        Sigma = (1.0 - lam) * S + lam * diag

        var = float((w.T @ Sigma @ w).squeeze())
        var = max(var, 0.0)
        return float(math.sqrt(TRADING_DAYS_PER_YEAR * var))

    def _risk_multiplier(self, dd: float, breadth_good: bool) -> float:
        """Drawdown-based k with safe-mode latch + ramp-out."""
        dd = float(dd)

        # Base drawdown rule
        if dd >= self.cfg.dd_safe:
            dd_k = 0.0
        elif dd >= self.cfg.dd_half:
            dd_k = 0.5
        else:
            dd_k = 1.0

        # Enter safe mode if dd is severe
        if dd_k == 0.0:
            self.state.safe_mode = True
            self.state.ramp_weeks_remaining = 0
            return 0.0

        # While in safe mode, only exit when recovery conditions met
        if self.state.safe_mode:
            if (dd <= self.cfg.dd_exit) and breadth_good and (self.state.breadth_good_count >= self.cfg.confirm_signals):
                self.state.safe_mode = False
                self.state.ramp_weeks_remaining = int(self.cfg.safe_mode_ramp_weeks)
            else:
                return 0.0

        # Ramp out of safe mode
        if self.state.ramp_weeks_remaining > 0:
            self.state.ramp_weeks_remaining -= 1
            return min(dd_k, 0.5)

        return dd_k

    def generate_target_weights(
        self,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        equity: float,
        peak_equity: float,
    ) -> pd.Series:
        """Compute target weights for all tickers (risk + cash) as of `asof`.

        Returns a Series indexed by all tickers that sums to 1.
        """
        prices = _ensure_datetime_index(prices)
        prices = prices.sort_index()

        # Warmup guard
        if len(prices.loc[:asof]) < self.cfg.min_history_days:
            w = pd.Series(0.0, index=self.cfg.all_tickers())
            w[self.cfg.cash_ticker] = 1.0
            return w

        # Raw eligibility and vol
        e_raw, sigma, _sma200 = self._raw_eligibility(prices, asof)

        breadth_good = int(e_raw.sum()) >= int(self.cfg.breadth_min)
        if breadth_good:
            self.state.breadth_good_count += 1
        else:
            self.state.breadth_good_count = 0

        # Apply hysteresis to get E_hyst
        e_hyst = self._update_hysteresis(e_raw)

        # Base inverse-vol weights
        inv = pd.Series(0.0, index=self.cfg.risk_tickers, dtype=float)
        for a in self.cfg.risk_tickers:
            if e_hyst.get(a, 0.0) > 0.0:
                s = float(sigma.get(a, 0.0))
                inv[a] = 0.0 if s <= 0 else 1.0 / s

        if inv.sum() <= 0:
            base = pd.Series(0.0, index=self.cfg.risk_tickers, dtype=float)
        else:
            base = inv / inv.sum()

        # Cap concentration; leftover becomes cash
        capped = self._cap_weights(base, cap=float(self.cfg.weight_cap))

        # Vol targeting (no leverage: scale down only)
        port_vol = self._portfolio_vol(prices, asof, capped)
        if port_vol <= 0:
            a = 1.0
        else:
            a = min(1.0, float(self.cfg.vol_target) / port_vol)

        w_risk = capped * a

        # Drawdown circuit breaker
        peak_equity = max(float(peak_equity), 1e-12)
        dd = 1.0 - float(equity) / peak_equity
        k = self._risk_multiplier(dd=dd, breadth_good=breadth_good)
        w_risk *= k

        # Assemble full weights
        w = pd.Series(0.0, index=self.cfg.all_tickers(), dtype=float)
        for a in self.cfg.risk_tickers:
            w[a] = float(w_risk.get(a, 0.0))

        w_cash = 1.0 - float(w[self.cfg.risk_tickers].sum())
        w[self.cfg.cash_ticker] = max(0.0, w_cash)

        # Normalize tiny rounding error
        s = float(w.sum())
        if s <= 0:
            w[:] = 0.0
            w[self.cfg.cash_ticker] = 1.0
        else:
            w = w / s

        self.state.last_signal_date = str(asof.date())
        return w


# -----------------------------
# Backtesting
# -----------------------------


@dataclass
class DataAuditResult:
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def has_issues(self) -> bool:
        return bool(self.warnings or self.errors)


def get_daily_rf(rets: pd.DataFrame, config: MatvmConfig) -> pd.Series:
    """Build the benchmark return series for excess-return metrics."""
    mode = str(config.risk_free_mode).strip().lower()
    if mode not in RISK_FREE_MODES:
        raise ValueError(f"Unknown risk_free_mode: {config.risk_free_mode}")

    if mode == "zero":
        return pd.Series(0.0, index=rets.index, name="RF_ZERO")

    if mode == "constant":
        annual = float(config.annual_risk_free_rate)
        if annual <= -1.0:
            raise ValueError("annual_risk_free_rate must be greater than -1.0")
        daily = (1.0 + annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        return pd.Series(daily, index=rets.index, name="RF_CONSTANT")

    if not config.risk_free_ticker:
        raise ValueError("risk_free_ticker must be set when risk_free_mode='ticker'")
    if config.risk_free_ticker not in rets.columns:
        raise ValueError(f"Missing risk-free ticker: {config.risk_free_ticker}")
    return rets[config.risk_free_ticker].rename(f"RF_{config.risk_free_ticker}")


def _looks_like_make_fake_data(prices: pd.Series) -> bool:
    clean = prices.dropna()
    if len(clean) < 3000:
        return False
    idx = pd.DatetimeIndex(clean.index)
    if idx[0] != pd.Timestamp("2014-01-01"):
        return False
    if pd.Timestamp("2025-12-31") not in idx:
        return False
    if not 95.0 <= float(clean.iloc[0]) <= 105.0:
        return False

    synthetic_calendar = pd.date_range("2014-01-01", "2025-12-31", freq="B")
    return synthetic_calendar.isin(idx).all()


def audit_cash_proxy(prices: pd.Series, ticker: str) -> List[str]:
    warnings: List[str] = []
    clean = prices.dropna()
    if len(clean) < 2:
        return [f"{ticker}: no return data"]

    rets = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return [f"{ticker}: no return data"]

    ann_vol = float(rets.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
    cagr_val = _price_cagr(clean)
    max_daily_abs = float(rets.abs().max())

    if ann_vol > 0.03:
        warnings.append(f"{ticker}: suspicious cash-proxy volatility: {ann_vol:.2%}")
    if abs(cagr_val) > 0.08:
        warnings.append(f"{ticker}: suspicious cash-proxy CAGR: {cagr_val:.2%}")
    if cagr_val < 0:
        warnings.append(f"{ticker}: negative long-term cash-proxy CAGR: {cagr_val:.2%}")
    if max_daily_abs > 0.01:
        warnings.append(f"{ticker}: suspicious cash-proxy daily move: {max_daily_abs:.2%}")

    return warnings


def audit_price_data(prices: pd.DataFrame, config: MatvmConfig) -> DataAuditResult:
    """Validate market data before using it for backtest interpretation."""
    result = DataAuditResult()
    prices = _ensure_datetime_index(prices)

    if prices.index.has_duplicates:
        result.errors.append("price data: duplicate dates found")
    if not prices.index.is_monotonic_increasing:
        result.errors.append("price data: dates are not monotonic increasing")

    for ticker in config.required_price_tickers():
        if ticker not in prices.columns:
            result.errors.append(f"{ticker}: missing price column")
            continue

        s = prices[ticker]
        clean = s.dropna()
        if clean.empty:
            result.errors.append(f"{ticker}: no price data")
            continue
        if s.isna().any():
            result.warnings.append(f"{ticker}: missing price values")
        if (clean <= 0).any():
            result.errors.append(f"{ticker}: zero or negative prices found")
            continue
        if len(clean) < config.min_history_days:
            result.warnings.append(
                f"{ticker}: short history ({len(clean)} rows, need at least {config.min_history_days})"
            )

        rets = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if not rets.empty and float(rets.abs().max()) > 0.25:
            result.warnings.append(f"{ticker}: extreme daily return: {rets.abs().max():.2%}")
        if clean.nunique() <= max(2, int(len(clean) * 0.01)):
            result.warnings.append(f"{ticker}: suspiciously flat price history")
        if _looks_like_make_fake_data(clean):
            result.warnings.append(
                f"{ticker}: looks like placeholder data generated by make_test_placeholder_data.py"
            )

    if config.cash_ticker in prices.columns:
        result.warnings.extend(audit_cash_proxy(prices[config.cash_ticker], config.cash_ticker))

    if (
        str(config.risk_free_mode).strip().lower() == "ticker"
        and config.risk_free_ticker
        and config.risk_free_ticker in prices.columns
        and config.risk_free_ticker != config.cash_ticker
    ):
        result.warnings.extend(
            audit_cash_proxy(prices[config.risk_free_ticker], config.risk_free_ticker)
        )

    return result


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    stats: Dict[str, float]


def backtest(
    prices: pd.DataFrame,
    config: MatvmConfig,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Sequential weekly backtest.

    - Signals evaluated on last trading day of each week (W-FRI resample).
    - Trades executed on next available trading day close.
    - Portfolio held with constant weights between rebalances.
    - Costs applied on rebalance days based on turnover.
    """
    prices = _ensure_datetime_index(prices).sort_index()

    tickers = config.all_tickers()
    required_tickers = config.required_price_tickers()
    missing = [t for t in required_tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Prices missing columns: {missing}")

    prices = prices[required_tickers].copy()
    prices = prices.ffill().dropna()

    rets = simple_returns(prices)
    portfolio_rets = rets[tickers]

    # Determine signal dates: actual last trading day of each week
    last_trading_each_week = prices.index.to_series().resample("W-FRI").last().dropna()
    signal_dates = pd.DatetimeIndex(last_trading_each_week.values)
    signal_set = set(signal_dates)

    strategy = MatvmStrategy(config=config)

    w_current = pd.Series(0.0, index=tickers, dtype=float)
    w_current[config.cash_ticker] = 1.0

    V = float(initial_capital)
    peak = V

    equity = []
    weights_daily = []
    trade_rows = []

    pending_target: Optional[pd.Series] = None
    pending_trade_date: Optional[pd.Timestamp] = None

    cost_rate = (config.tcost_bps + config.slippage_bps) / 10_000.0

    idx = prices.index

    for i in range(len(idx)):
        dt = idx[i]

        # Apply daily return using weights held from previous close to this close
        if i > 0:
            r = float((w_current * portfolio_rets.loc[dt]).sum())
            V *= (1.0 + r)

        # Track peak/drawdown
        peak = max(peak, V)

        # Execute pending rebalance at close of trade date
        if pending_trade_date is not None and dt == pending_trade_date and pending_target is not None:
            # Decide whether to rebalance (simple: full rebalance only if meaningful drift)
            diff = (pending_target - w_current).abs()
            rel = diff / (w_current.abs() + 1e-12)

            trigger = (diff > config.rebalance_abs_threshold) | (rel > config.rebalance_rel_threshold)
            do_rebalance = bool(trigger.any())

            if do_rebalance:
                turnover = 0.5 * float((pending_target - w_current).abs().sum())
                cost = turnover * cost_rate
                V *= max(0.0, 1.0 - cost)

                trade_rows.append(
                    {
                        "date": dt,
                        "turnover": turnover,
                        "cost_rate": cost_rate,
                        "cost_frac": cost,
                        "equity_after": V,
                    }
                )
                w_current = pending_target.copy()
            else:
                trade_rows.append(
                    {
                        "date": dt,
                        "turnover": 0.0,
                        "cost_rate": cost_rate,
                        "cost_frac": 0.0,
                        "equity_after": V,
                    }
                )

            pending_target = None
            pending_trade_date = None

        # On signal date, compute weights to execute next trading day
        if dt in signal_set:
            w_target = strategy.generate_target_weights(
                prices=prices,
                asof=dt,
                equity=V,
                peak_equity=peak,
            )
            # schedule trade for next trading day
            if i + 1 < len(idx):
                pending_target = w_target
                pending_trade_date = idx[i + 1]

        equity.append(V)
        weights_daily.append(w_current.copy())

    equity_curve = pd.Series(equity, index=idx, name="equity")
    weights_df = pd.DataFrame(weights_daily, index=idx)
    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty:
        trades_df = trades_df.set_index("date")

    daily_port_ret = equity_curve.pct_change().fillna(0.0)

    daily_rf_zero = pd.Series(0.0, index=daily_port_ret.index)
    daily_rf_config = get_daily_rf(rets, config).reindex_like(daily_port_ret).fillna(0.0)

    stats = {
        "CAGR": cagr(equity_curve),
        "MaxDrawdown": max_drawdown(equity_curve),
        "Sharpe_RF0": sharpe_ratio(daily_port_ret, daily_rf=daily_rf_zero),
        "Sortino_RF0": sortino_ratio(daily_port_ret, daily_rf=daily_rf_zero),
        "Sharpe_Excess": sharpe_ratio(daily_port_ret, daily_rf=daily_rf_config),
        "Sortino_Excess": sortino_ratio(daily_port_ret, daily_rf=daily_rf_config),
    }
    stats["Calmar"] = (
        stats["CAGR"] / stats["MaxDrawdown"] if stats["MaxDrawdown"] > 0 else 0.0
    )

    # Turnover summary
    if not trades_df.empty:
        stats["AvgWeeklyTurnover"] = float(trades_df["turnover"].mean())
        stats["TotalCostFrac"] = float(trades_df["cost_frac"].sum())
    else:
        stats["AvgWeeklyTurnover"] = 0.0
        stats["TotalCostFrac"] = 0.0

    return BacktestResult(
        equity_curve=equity_curve,
        weights=weights_df,
        trades=trades_df,
        stats=stats,
    )


def plot_equity_and_drawdown(equity: pd.Series, outpath: Optional[Path] = None) -> None:
    if not _HAVE_MPL:
        print("matplotlib not available; skipping plot")
        return
    peak = equity.cummax()
    dd = 1.0 - equity / peak

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(equity.index, equity.values)
    ax1.set_title("Equity Curve")

    ax2.plot(dd.index, dd.values)
    ax2.set_title("Drawdown")

    fig.tight_layout()
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=150)
    else:
        plt.show()


# -----------------------------
# Persistence helpers (for live)
# -----------------------------


def load_state(path: Path) -> StrategyState:
    if not path.exists():
        return StrategyState()
    data = json.loads(path.read_text())
    st = StrategyState()
    for k in ["held", "enter_count", "exit_count"]:
        if k in data and isinstance(data[k], dict):
            setattr(st, k, {str(kk): int(vv) for kk, vv in data[k].items()})
    st.safe_mode = bool(data.get("safe_mode", False))
    st.ramp_weeks_remaining = int(data.get("ramp_weeks_remaining", 0))
    st.breadth_good_count = int(data.get("breadth_good_count", 0))
    st.last_signal_date = data.get("last_signal_date")
    return st


def save_state(path: Path, state: StrategyState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "held": state.held,
        "enter_count": state.enter_count,
        "exit_count": state.exit_count,
        "safe_mode": state.safe_mode,
        "ramp_weeks_remaining": state.ramp_weeks_remaining,
        "breadth_good_count": state.breadth_good_count,
        "last_signal_date": state.last_signal_date,
        "saved_at_utc": _dt_utc_now().isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def update_equity_history(path: Path, date: pd.Timestamp, equity: float) -> pd.Series:
    """Append (date,equity) if new and return full history as a Series."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
    else:
        df = pd.DataFrame(columns=["date", "equity"])

    date = pd.Timestamp(date).normalize()

    if df.empty or pd.Timestamp(df["date"].iloc[-1]).normalize() != date:
        df = pd.concat(
            [df, pd.DataFrame({"date": [date], "equity": [float(equity)]})],
            ignore_index=True,
        )

    df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    df.to_csv(path, index=False)

    s = pd.Series(df["equity"].values, index=pd.DatetimeIndex(df["date"]), name="equity")
    return s


# -----------------------------
# Data sources
# -----------------------------


def load_prices_from_csv(folder: Path, tickers: Sequence[str]) -> pd.DataFrame:
    """Load per-ticker CSV files with columns: date, adj_close (or close).

    File naming: <TICKER>.csv
    """
    live_folder = folder / "live"
    if live_folder.exists() and all((live_folder / f"{t}.csv").exists() for t in tickers):
        folder = live_folder

    frames = []
    for t in tickers:
        f = folder / f"{t}.csv"
        if not f.exists():
            raise FileNotFoundError(f"Missing {f}. Expected one CSV per ticker.")
        df = pd.read_csv(f)
        if "date" not in df.columns:
            raise ValueError(f"{f} must have a 'date' column")
        df["date"] = pd.to_datetime(df["date"])
        price_col = None
        for c in ["adj_close", "Adj Close", "adjusted_close", "close", "Close"]:
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            raise ValueError(f"{f} must have one of: adj_close/Adj Close/close/Close")
        s = df.set_index("date")[price_col].rename(t)
        frames.append(s)

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.ffill().dropna()
    return prices


def load_prices_from_yfinance(
    tickers: Sequence[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Optional convenience source for backtests (NOT used for live execution).

    Requires: pip install yfinance
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance") from e

    data = yf.download(list(tickers), start=start, end=end, progress=False, auto_adjust=False)
    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned no data")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            adj = data["Adj Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(1):
            adj = data.xs("Adj Close", level=1, axis=1).copy()
        else:
            raise RuntimeError("Could not find 'Adj Close' in yfinance data")
    else:
        # single ticker
        if "Adj Close" not in data.columns:
            raise RuntimeError("Could not find 'Adj Close' in yfinance data")
        t = tickers[0]
        adj = data["Adj Close"].to_frame(name=t)

    adj.index = pd.to_datetime(adj.index)
    adj = adj.sort_index().ffill().dropna()

    # Ensure we have exactly requested columns
    adj = adj[[t for t in tickers if t in adj.columns]]
    if adj.shape[1] != len(tickers):
        missing = [t for t in tickers if t not in adj.columns]
        raise RuntimeError(f"Missing tickers from yfinance result: {missing}")

    return adj


# -----------------------------
# IBKR integration (ib_insync)
# -----------------------------


@dataclass
class IbkrConfig:
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper: 7497, TWS live: 7496 (common defaults)
    client_id: int = 42
    account: Optional[str] = None  # if None, first managed account

    currency: str = "USD"
    exchange: str = "SMART"

    # Execution
    use_limit_orders: bool = True
    limit_buffer_bps: float = 10.0  # 10 bps price cushion

    # Safety
    max_orders: int = 50


class IbkrClient:
    """Thin wrapper around ib_insync that keeps imports optional."""

    def __init__(self, cfg: IbkrConfig):
        self.cfg = cfg
        self.ib = None

    def connect(self):
        try:
            from ib_insync import IB  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ib_insync not installed. Run: pip install ib_insync"
            ) from e

        self.ib = IB()
        ok = self.ib.connect(self.cfg.host, int(self.cfg.port), clientId=int(self.cfg.client_id))
        if not ok:
            raise RuntimeError("Failed to connect to IBKR. Is TWS / IB Gateway running?")
        return self

    def disconnect(self):
        if self.ib is not None:
            try:
                self.ib.disconnect()
            except Exception:
                pass

    def managed_account(self) -> str:
        assert self.ib is not None
        accounts = list(self.ib.managedAccounts())
        if not accounts:
            raise RuntimeError("No managed accounts found")
        if self.cfg.account:
            if self.cfg.account not in accounts:
                raise RuntimeError(f"Account {self.cfg.account} not in managed accounts {accounts}")
            return self.cfg.account
        return accounts[0]

    def net_liquidation(self, account: str) -> float:
        """Net liquidation (equity) in base currency."""
        assert self.ib is not None
        vals = self.ib.accountValues(account=account)
        for v in vals:
            if v.tag == "NetLiquidation" and (v.currency == self.cfg.currency or v.currency == ""):
                return float(v.value)
        # Fallback: try without currency match
        for v in vals:
            if v.tag == "NetLiquidation":
                return float(v.value)
        raise RuntimeError("Could not read NetLiquidation from account values")

    def _contract_for_symbol(self, symbol: str):
        from ib_insync import Stock  # type: ignore

        return Stock(symbol, self.cfg.exchange, self.cfg.currency)

    def qualify_contracts(self, symbols: Sequence[str]):
        assert self.ib is not None
        contracts = [self._contract_for_symbol(s) for s in symbols]
        self.ib.qualifyContracts(*contracts)
        return contracts

    def historical_daily_prices(
        self,
        symbols: Sequence[str],
        duration: str = "3 Y",
        what_to_show: str = "ADJUSTED_LAST",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Fetch daily adjusted (if available) closes from IBKR."""
        assert self.ib is not None

        from ib_insync import util  # type: ignore

        contracts = self.qualify_contracts(symbols)
        series = []

        for sym, c in zip(symbols, contracts):
            bars = self.ib.reqHistoricalData(
                c,
                endDateTime="",
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
                keepUpToDate=False,
            )

            if not bars:
                # fallback to TRADES
                bars = self.ib.reqHistoricalData(
                    c,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=use_rth,
                    formatDate=1,
                    keepUpToDate=False,
                )

            df = util.df(bars)
            if df.empty:
                raise RuntimeError(f"No historical data for {sym}")

            # IB bars have 'date' and 'close'
            df["date"] = pd.to_datetime(df["date"])
            s = df.set_index("date")["close"].rename(sym)
            series.append(s)

        prices = pd.concat(series, axis=1).sort_index()
        prices = prices.ffill().dropna()
        return prices

    def last_prices(self, symbols: Sequence[str]) -> Dict[str, float]:
        """Try to get last prices. Falls back to latest historical close if live quotes unavailable."""
        assert self.ib is not None

        contracts = self.qualify_contracts(symbols)
        out: Dict[str, float] = {}

        # Request snapshot market data (requires subscriptions). If not available, values may be NaN.
        try:
            for sym, c in zip(symbols, contracts):
                ticker = self.ib.reqMktData(c, "", snapshot=True, regulatorySnapshot=False)
                self.ib.sleep(0.5)
                px = ticker.marketPrice()
                if px is None or np.isnan(px) or px <= 0:
                    # Try last / close fields
                    px = ticker.last if ticker.last else ticker.close
                if px is None or np.isnan(px) or px <= 0:
                    continue
                out[sym] = float(px)
        except Exception:
            # ignore; will fill from historical
            pass

        # Fill missing using latest historical close
        missing = [s for s in symbols if s not in out]
        if missing:
            hist = self.historical_daily_prices(missing, duration="10 D")
            for sym in missing:
                out[sym] = float(hist[sym].iloc[-1])

        return out

    def current_positions(self, account: str) -> Dict[str, float]:
        """Return current position size (shares) by symbol."""
        assert self.ib is not None
        pos = self.ib.positions(account=account)
        out: Dict[str, float] = {}
        for p in pos:
            sym = getattr(p.contract, "symbol", None)
            if sym:
                out[str(sym)] = float(p.position)
        return out

    def place_rebalance_orders(
        self,
        account: str,
        target_shares: Dict[str, int],
        dry_run: bool = True,
    ) -> List[Dict[str, object]]:
        """Place orders to move current positions to target_shares.

        - Sells first, then buys.
        - Uses limit orders with a configurable buffer, or market orders.
        """
        assert self.ib is not None

        try:
            from ib_insync import LimitOrder, MarketOrder  # type: ignore
        except Exception as e:
            raise RuntimeError("ib_insync not installed") from e

        current = self.current_positions(account=account)
        symbols = list(target_shares.keys())
        prices = self.last_prices(symbols)
        contracts = {c.symbol: c for c in self.qualify_contracts(symbols)}

        deltas: List[Tuple[str, int]] = []
        for sym in symbols:
            cur = int(round(current.get(sym, 0.0)))
            tgt = int(target_shares[sym])
            delta = tgt - cur
            if delta != 0:
                deltas.append((sym, delta))

        if len(deltas) > self.cfg.max_orders:
            raise RuntimeError(f"Refusing to place {len(deltas)} orders (max_orders={self.cfg.max_orders})")

        # sells first
        sells = [(s, d) for s, d in deltas if d < 0]
        buys = [(s, d) for s, d in deltas if d > 0]

        placed: List[Dict[str, object]] = []

        def _mk_order(sym: str, delta: int):
            side = "BUY" if delta > 0 else "SELL"
            qty = abs(int(delta))
            px = float(prices[sym])
            if self.cfg.use_limit_orders:
                buf = float(self.cfg.limit_buffer_bps) / 10_000.0
                limit_px = px * (1 + buf) if side == "BUY" else px * (1 - buf)
                return LimitOrder(side, qty, round(limit_px, 2))
            return MarketOrder(side, qty)

        for sym, delta in sells + buys:
            order = _mk_order(sym, delta)
            contract = contracts.get(sym)
            if contract is None:
                raise RuntimeError(f"Missing contract for {sym}")

            placed.append(
                {
                    "symbol": sym,
                    "delta": int(delta),
                    "orderType": order.orderType,
                    "totalQuantity": order.totalQuantity,
                    "lmtPrice": getattr(order, "lmtPrice", None),
                    "dry_run": dry_run,
                }
            )

            if not dry_run:
                trade = self.ib.placeOrder(contract, order)
                self.ib.sleep(0.2)
                # best-effort status capture
                try:
                    status = trade.orderStatus.status
                except Exception:
                    status = "UNKNOWN"
                placed[-1]["status"] = status

        return placed


# -----------------------------
# Live runner (IBKR)
# -----------------------------


def compute_target_shares_from_weights(
    weights: pd.Series,
    equity: float,
    last_prices: Dict[str, float],
) -> Dict[str, int]:
    tgt: Dict[str, int] = {}
    for sym, w in weights.items():
        px = float(last_prices.get(sym, np.nan))
        if np.isnan(px) or px <= 0:
            continue
        dollars = float(equity) * float(w)
        shares = int(math.floor(dollars / px))
        tgt[str(sym)] = max(0, shares)
    return tgt


def run_live_ibkr(
    strat_cfg: MatvmConfig,
    ibkr_cfg: IbkrConfig,
    state_path: Path,
    equity_history_path: Path,
    dry_run: bool = True,
    duration: str = "3 Y",
    log_path: Optional[Path] = None,
) -> None:
    """Run the strategy once against an IBKR account.

    Typical usage:
    - Run once per week (e.g., via cron) or run daily in dry-run mode.
    - In dry-run, you inspect the proposed orders.
    - When you're satisfied, run with --place-orders.

    State is persisted to disk so hysteresis + safe-mode work across runs.
    """

    state = load_state(state_path)
    strategy = MatvmStrategy(config=strat_cfg, state=state)

    client = IbkrClient(cfg=ibkr_cfg).connect()
    try:
        account = client.managed_account()
        equity = client.net_liquidation(account=account)

        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        equity_hist = update_equity_history(equity_history_path, today, equity)
        peak = float(equity_hist.cummax().iloc[-1])

        symbols = strat_cfg.all_tickers()
        prices = client.historical_daily_prices(symbols, duration=duration)
        asof = prices.index[-1]

        weights = strategy.generate_target_weights(
            prices=prices,
            asof=asof,
            equity=equity,
            peak_equity=peak,
        )

        # Rebalance threshold check (weight drift) uses current positions -> approximate current weights
        last_px = client.last_prices(symbols)

        # Compute target shares (whole shares)
        target_shares = compute_target_shares_from_weights(weights, equity=equity, last_prices=last_px)

        # Place orders to reach target
        orders = client.place_rebalance_orders(account=account, target_shares=target_shares, dry_run=dry_run)

        # Persist state
        save_state(state_path, strategy.state)

        # Log
        payload = {
            "run_at_utc": _dt_utc_now().isoformat(),
            "account": account,
            "net_liquidation": float(equity),
            "asof_price_date": str(asof.date()),
            "weights": {k: float(v) for k, v in weights.items()},
            "target_shares": target_shares,
            "orders": orders,
            "dry_run": dry_run,
            "state": dataclasses.asdict(strategy.state),
        }

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

        # Human-readable output
        print("\n=== MA-TVM / IBKR run ===")
        print(f"Account:           {account}")
        print(f"Net liquidation:   {equity:,.2f} {ibkr_cfg.currency}")
        print(f"Signal as-of date: {asof.date()}")
        print(f"Mode:             {'DRY RUN' if dry_run else 'PLACE ORDERS'}")

        print("\nTarget weights:")
        for sym in strat_cfg.risk_tickers + [strat_cfg.cash_ticker]:
            print(f"  {sym:>6}  {weights.get(sym, 0.0):7.2%}")

        print("\nOrders:")
        if not orders:
            print("  (no changes)")
        else:
            for o in orders:
                sym = o["symbol"]
                delta = int(o["delta"])
                side = "BUY" if delta > 0 else "SELL"
                qty = abs(delta)
                ot = o.get("orderType")
                lp = o.get("lmtPrice")
                extra = f" @ {lp}" if lp is not None else ""
                print(f"  {side:>4} {qty:>6} {sym}  ({ot}){extra}")

    finally:
        client.disconnect()


# -----------------------------
# CLI
# -----------------------------


def _make_default_config() -> MatvmConfig:
    return MatvmConfig()


def cli_backtest(args: argparse.Namespace) -> None:
    cfg = _make_default_config()
    if args.risk:
        cfg.risk_tickers = [s.strip().upper() for s in args.risk.split(",") if s.strip()]
    if args.cash:
        cfg.cash_ticker = args.cash.strip().upper()
    cfg.risk_free_mode = args.risk_free_mode.strip().lower()
    cfg.risk_free_ticker = (
        args.risk_free_ticker.strip().upper() if args.risk_free_ticker else None
    )
    cfg.annual_risk_free_rate = float(args.annual_risk_free_rate)

    if cfg.risk_free_mode == "ticker" and not cfg.risk_free_ticker:
        raise SystemExit("--risk-free-ticker is required when --risk-free-mode ticker")

    tickers = cfg.required_price_tickers()

    if args.csv_folder:
        prices = load_prices_from_csv(Path(args.csv_folder), tickers)
    else:
        prices = load_prices_from_yfinance(tickers, start=args.start, end=args.end)

    audit = audit_price_data(prices, cfg)
    if audit.has_issues():
        print("\n=== DATA AUDIT ===")
        for err in audit.errors:
            print(f"ERROR:   {err}")
        for warning in audit.warnings:
            print(f"WARNING: {warning}")
        if audit.errors:
            raise SystemExit("Data audit failed with errors.")
        if args.strict_data:
            raise SystemExit("Strict data audit failed because warnings were found.")

    res = backtest(prices=prices, config=cfg, initial_capital=float(args.initial))

    print("\n=== Backtest config ===")
    if audit.has_issues():
        print(f"{'Backtest status':>22}: DIAGNOSTIC ONLY")
        print(f"{'Reason':>22}: data audit warnings detected")
    else:
        print(f"{'Backtest status':>22}: CLEAN DATA CHECK PASSED")
    print(f"{'Cash asset':>22}: {cfg.cash_ticker}")
    print(f"{'Risk-free mode':>22}: {cfg.risk_free_mode}")
    if cfg.risk_free_mode == "ticker":
        print(f"{'Risk-free ticker':>22}: {cfg.risk_free_ticker}")
    elif cfg.risk_free_mode == "constant":
        print(f"{'Annual risk-free rate':>22}: {cfg.annual_risk_free_rate:.2%}")

    print("\n=== Backtest stats ===")
    for k, v in res.stats.items():
        if k in {"CAGR", "MaxDrawdown"}:
            print(f"{k:>22}: {v:8.2%}")
        else:
            print(f"{k:>22}: {v:8.3f}")

    outdir = Path(args.outdir) if args.outdir else Path("./matvm_out")
    outdir.mkdir(parents=True, exist_ok=True)

    res.equity_curve.to_csv(outdir / "equity.csv")
    res.weights.to_csv(outdir / "weights_daily.csv")
    if not res.trades.empty:
        res.trades.to_csv(outdir / "trades.csv")

    if args.plot:
        plot_equity_and_drawdown(res.equity_curve, outpath=outdir / "equity_drawdown.png")

    print(f"\nWrote results to: {outdir.resolve()}")


def cli_live_ibkr(args: argparse.Namespace) -> None:
    strat_cfg = _make_default_config()

    if args.risk:
        strat_cfg.risk_tickers = [s.strip().upper() for s in args.risk.split(",") if s.strip()]
    if args.cash:
        strat_cfg.cash_ticker = args.cash.strip().upper()

    ibcfg = IbkrConfig(
        host=args.host,
        port=int(args.port),
        client_id=int(args.client_id),
        account=args.account,
        use_limit_orders=not args.market,
        limit_buffer_bps=float(args.limit_bps),
    )

    state_path = Path(args.state_file)
    eq_path = Path(args.equity_file)
    log_path = Path(args.log_file) if args.log_file else None

    run_live_ibkr(
        strat_cfg=strat_cfg,
        ibkr_cfg=ibcfg,
        state_path=state_path,
        equity_history_path=eq_path,
        dry_run=not args.place_orders,
        duration=args.duration,
        log_path=log_path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MA-TVM quantitative trading bot with IBKR execution (single-file)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Backtest
    b = sub.add_parser("backtest", help="Run an offline backtest")
    b.add_argument("--start", default="2014-01-01")
    b.add_argument("--end", default="2025-12-31")
    b.add_argument("--initial", default="100000")
    b.add_argument("--risk", default=None, help="Comma-separated risk tickers")
    b.add_argument("--cash", default=None, help="Cash ticker")
    b.add_argument(
        "--risk-free-mode",
        choices=sorted(RISK_FREE_MODES),
        default="zero",
        help="Benchmark for excess-return Sharpe/Sortino",
    )
    b.add_argument(
        "--risk-free-ticker",
        default=None,
        help="Ticker to use when --risk-free-mode ticker",
    )
    b.add_argument(
        "--annual-risk-free-rate",
        type=float,
        default=0.0,
        help="Decimal annual rate to use when --risk-free-mode constant, e.g. 0.05",
    )
    b.add_argument(
        "--csv-folder",
        default=None,
        help="Folder with <TICKER>.csv files. If it contains data/live, that folder is preferred.",
    )
    b.add_argument("--outdir", default="./matvm_out")
    b.add_argument("--plot", action="store_true")
    b.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail the backtest when data audit warnings are found",
    )
    b.set_defaults(func=cli_backtest)

    # Live IBKR
    l = sub.add_parser("live-ibkr", help="Run one live rebalance against IBKR")
    l.add_argument("--host", default="127.0.0.1")
    l.add_argument("--port", default="7497")
    l.add_argument("--client-id", default="42")
    l.add_argument("--account", default=None)

    l.add_argument("--risk", default=None, help="Comma-separated risk tickers")
    l.add_argument("--cash", default=None, help="Cash ticker")

    l.add_argument("--duration", default="3 Y", help="Historical duration for signals")

    l.add_argument(
        "--place-orders",
        action="store_true",
        help="Actually submit orders. Default is DRY RUN.",
    )
    l.add_argument(
        "--market",
        action="store_true",
        help="Use market orders (default: limit orders with buffer)",
    )
    l.add_argument(
        "--limit-bps",
        default="10",
        help="Limit price cushion in bps (default 10). Buy: +bps, Sell: -bps.",
    )

    l.add_argument(
        "--state-file",
        default="./matvm_state/state.json",
        help="Persistent hysteresis/safe-mode state",
    )
    l.add_argument(
        "--equity-file",
        default="./matvm_state/equity_history.csv",
        help="Local equity history used for drawdown calc",
    )
    l.add_argument(
        "--log-file",
        default="./matvm_state/runs.jsonl",
        help="Append-only JSONL log of each run",
    )

    l.set_defaults(func=cli_live_ibkr)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
