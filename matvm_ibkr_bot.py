from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import subprocess
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
CASH_RETURN_MODES = {"zero", "constant", "ticker"}
CASH_POLICY_MODES = {"dynamic", "fixed"}
ASSET_SELECTION_MODES = {
    "base",
    "equal_weight_selected",
    "top_momentum_only",
    "top_momentum_no_vol_filter",
    "min_vol_only",
    "risk_parity_selected",
    "random_selected",
}
MIN_DAILY_RETURN_STD = 1e-5
DECISION_MIN_EXCESS_SHARPE = 0.50
DECISION_MATERIAL_DD_REDUCTION = 0.08
VOL_TARGET_BENCHMARKS = (0.08, 0.10, 0.12)
VOL_TARGET_LOOKBACK_DAYS = 63
EXPOSURE_MATCHED_PREFIXES = ("StaticMatched_", "SameCashSchedule_")
DEFAULT_RANDOM_NULL_SEEDS = 100
WALK_FORWARD_CANDIDATE_VARIANTS = (
    "baseline",
    "CashTiming_ThresholdSweep_Loose",
    "NoCashTiming_StaticAverageCash",
    "NoCashTiming_FixedCash_30",
    "NoCashTiming_FixedCash_40",
    "AssetSelection_EqualWeightSelected",
    "AssetSelection_TopMomentumOnly",
    "AssetSelection_MinVolOnly",
    "AssetSelection_RiskParitySelected",
)
DIAGNOSTIC_OUTPUT_TABLES = {
    "allocation_history",
    "asset_weight_summary",
    "cash_exposure_summary",
    "return_contribution_by_asset",
    "best_allocation_decisions",
    "worst_allocation_decisions",
}


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
    if sd < MIN_DAILY_RETURN_STD or np.isnan(sd):
        return float("nan")
    return float((mu / sd) * math.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(daily_returns: pd.Series, daily_rf: Optional[pd.Series] = None) -> float:
    r = daily_returns.copy()
    if daily_rf is not None:
        daily_rf = daily_rf.reindex_like(r).fillna(0.0)
        r = r - daily_rf
    mu = r.mean()
    downside = r[r < 0]
    dd = downside.std(ddof=0)
    if dd < MIN_DAILY_RETURN_STD or np.isnan(dd):
        return float("nan")
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

    # Backtest cash return model. This is intentionally separate from cash_ticker:
    # cash_ticker is the live execution instrument, cash_return_* controls what
    # return the cash sleeve earns in historical simulations.
    cash_return_mode: str = "ticker"  # "zero", "constant", or "ticker"
    cash_return_ticker: Optional[str] = "SGOV"
    annual_cash_return_rate: float = 0.0

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
    cash_policy: str = "dynamic"  # "dynamic" or "fixed"; research variants only
    fixed_cash_weight: Optional[float] = None
    asset_selection_mode: str = "base"
    asset_selection_seed: Optional[int] = None

    # Capital preservation overlay (drawdown thresholds)
    dd_half: float = 0.08
    dd_safe: float = 0.12
    dd_exit: float = 0.06
    breadth_min: int = 3
    safe_mode_ramp_weeks: int = 4

    # Rebalancing / costs
    rebalance_freq: str = "W-FRI"
    rebalance_abs_threshold: float = 0.02
    rebalance_rel_threshold: float = 0.15
    tcost_bps: float = 1.0
    slippage_bps: float = 2.0

    # Data warmup
    min_history_days: int = 260  # minimum bars before strategy acts

    def all_tickers(self) -> List[str]:
        return list(dict.fromkeys(self.risk_tickers + [self.cash_ticker]))

    def required_price_tickers(self) -> List[str]:
        tickers = list(self.risk_tickers)
        if str(self.cash_return_mode).strip().lower() == "ticker" and self.cash_return_ticker:
            tickers.append(self.cash_return_ticker)
        if str(self.risk_free_mode).strip().lower() == "ticker" and self.risk_free_ticker:
            tickers.append(self.risk_free_ticker)
        return list(dict.fromkeys(tickers))

    def required_history_days(self) -> int:
        signal_need = max(self.momentum_windows + (self.sma_window, self.vol_window + 1))
        return max(int(self.min_history_days), int(signal_need))


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

    def _raw_momentum_scores(self, prices: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
        """Momentum scores without volatility normalization."""
        risk = prices[self.cfg.risk_tickers].loc[:asof]
        if len(risk) < max(self.cfg.momentum_windows):
            return pd.Series(0.0, index=self.cfg.risk_tickers, dtype=float)

        p0 = risk.iloc[-1]
        parts = []
        for lookback in self.cfg.momentum_windows:
            pL = risk.iloc[-lookback]
            parts.append(np.log(p0 / pL))
        raw = pd.concat(parts, axis=1).mean(axis=1)
        return raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

    def _asset_selection_weights(
        self,
        prices: pd.DataFrame,
        asof: pd.Timestamp,
        q: pd.Series,
        sigma: pd.Series,
        trend: pd.Series,
        e_hyst: pd.Series,
    ) -> pd.Series:
        mode = str(self.cfg.asset_selection_mode).strip().lower()
        if mode not in ASSET_SELECTION_MODES:
            raise ValueError(f"Unknown asset_selection_mode: {self.cfg.asset_selection_mode}")

        weights = pd.Series(0.0, index=self.cfg.risk_tickers, dtype=float)

        if mode == "top_momentum_no_vol_filter":
            raw_momentum = self._raw_momentum_scores(prices, asof)
            active = [
                ticker
                for ticker in self.cfg.risk_tickers
                if bool(trend.get(ticker, False)) and float(raw_momentum.get(ticker, 0.0)) > 0.0
            ]
            if not active:
                return weights
            chosen = max(active, key=lambda ticker: float(raw_momentum.get(ticker, -np.inf)))
            weights[chosen] = 1.0
            return weights

        active = [
            ticker
            for ticker in self.cfg.risk_tickers
            if float(e_hyst.get(ticker, 0.0)) > 0.0
        ]
        if not active:
            return weights

        if mode == "equal_weight_selected":
            for ticker in active:
                weights[ticker] = 1.0 / len(active)
            return weights

        if mode == "top_momentum_only":
            chosen = max(active, key=lambda ticker: float(q.get(ticker, -np.inf)))
            weights[chosen] = 1.0
            return weights

        if mode == "min_vol_only":
            chosen = min(active, key=lambda ticker: float(sigma.get(ticker, np.inf)))
            weights[chosen] = 1.0
            return weights

        if mode == "random_selected":
            seed = 0 if self.cfg.asset_selection_seed is None else int(self.cfg.asset_selection_seed)
            day_key = int(pd.Timestamp(asof).strftime("%Y%m%d"))
            rng = np.random.default_rng(seed * 1_000_003 + day_key)
            chosen = str(rng.choice(active))
            weights[chosen] = 1.0
            return weights

        inv = pd.Series(0.0, index=self.cfg.risk_tickers, dtype=float)
        for ticker in active:
            vol = float(sigma.get(ticker, 0.0))
            inv[ticker] = 0.0 if vol <= 0 else 1.0 / vol

        if inv.sum() <= 0:
            return weights
        return inv / inv.sum()

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
        if len(prices.loc[:asof]) < self.cfg.required_history_days():
            w = pd.Series(0.0, index=self.cfg.all_tickers())
            w[self.cfg.cash_ticker] = 1.0
            return w

        # Raw eligibility and vol
        q, sigma, sma200 = self._compute_q_scores(prices, asof)
        p0 = prices.loc[asof, self.cfg.risk_tickers]
        trend = p0 > sma200
        e_raw = ((q > 0.0) & trend).astype(int)

        breadth_good = int(e_raw.sum()) >= int(self.cfg.breadth_min)
        if breadth_good:
            self.state.breadth_good_count += 1
        else:
            self.state.breadth_good_count = 0

        # Apply hysteresis to get E_hyst
        e_hyst = self._update_hysteresis(e_raw)

        base = self._asset_selection_weights(
            prices=prices,
            asof=asof,
            q=q,
            sigma=sigma,
            trend=trend,
            e_hyst=e_hyst,
        )

        # Cap concentration; leftover becomes cash
        capped = self._cap_weights(base, cap=float(self.cfg.weight_cap))

        cash_policy = str(self.cfg.cash_policy).strip().lower()
        if cash_policy not in CASH_POLICY_MODES:
            raise ValueError(f"Unknown cash_policy: {self.cfg.cash_policy}")

        if cash_policy == "fixed":
            fixed_cash = 0.0 if self.cfg.fixed_cash_weight is None else float(self.cfg.fixed_cash_weight)
            fixed_cash = min(max(fixed_cash, 0.0), 1.0)
            if capped.sum() > 0:
                w_risk = capped / capped.sum() * (1.0 - fixed_cash)
            else:
                w_risk = capped.copy()
        else:
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


def get_daily_cash_returns(rets: pd.DataFrame, config: MatvmConfig) -> pd.Series:
    """Build the return series earned by the backtested cash sleeve."""
    mode = str(config.cash_return_mode).strip().lower()
    if mode not in CASH_RETURN_MODES:
        raise ValueError(f"Unknown cash_return_mode: {config.cash_return_mode}")

    if mode == "zero":
        return pd.Series(0.0, index=rets.index, name=config.cash_ticker)

    if mode == "constant":
        annual = float(config.annual_cash_return_rate)
        if annual <= -1.0:
            raise ValueError("annual_cash_return_rate must be greater than -1.0")
        daily = (1.0 + annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
        return pd.Series(daily, index=rets.index, name=config.cash_ticker)

    if not config.cash_return_ticker:
        raise ValueError("cash_return_ticker must be set when cash_return_mode='ticker'")
    if config.cash_return_ticker not in rets.columns:
        raise ValueError(f"Missing cash return ticker: {config.cash_return_ticker}")
    return rets[config.cash_return_ticker].rename(config.cash_ticker)


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

    if (
        str(config.cash_return_mode).strip().lower() == "ticker"
        and config.cash_return_ticker
        and config.cash_return_ticker in prices.columns
    ):
        result.warnings.extend(
            audit_cash_proxy(prices[config.cash_return_ticker], config.cash_return_ticker)
        )

    if (
        str(config.risk_free_mode).strip().lower() == "ticker"
        and config.risk_free_ticker
        and config.risk_free_ticker in prices.columns
        and config.risk_free_ticker != config.cash_return_ticker
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
    portfolio_rets = pd.DataFrame(index=prices.index)
    for ticker in config.risk_tickers:
        portfolio_rets[ticker] = rets[ticker]
    portfolio_rets[config.cash_ticker] = get_daily_cash_returns(rets, config)
    portfolio_rets = portfolio_rets[tickers]

    # Determine signal dates: actual last trading day in each rebalance period.
    last_trading_each_period = prices.index.to_series().resample(config.rebalance_freq).last().dropna()
    signal_dates = pd.DatetimeIndex(last_trading_each_period.values)
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
# Robustness analysis
# -----------------------------


ROBUSTNESS_START_DATES: Tuple[str, ...] = (
    "2018-01-01",
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
    "2022-01-01",
    "2023-01-01",
)

REGIME_PERIODS: Tuple[Tuple[str, str, str], ...] = (
    ("2020 crash", "2020-02-19", "2020-04-30"),
    ("2022 inflation/rate shock", "2022-01-01", "2022-12-31"),
    ("2023-2024 rebound", "2023-01-01", "2024-12-31"),
)


def _clone_config(config: MatvmConfig, **updates: object) -> MatvmConfig:
    cfg = dataclasses.replace(config)
    cfg.risk_tickers = list(config.risk_tickers)
    cfg.momentum_windows = tuple(config.momentum_windows)
    for key, value in updates.items():
        setattr(cfg, key, value)
    return cfg


def _parse_date_list(raw: str) -> List[str]:
    dates = [part.strip() for part in raw.split(",") if part.strip()]
    if not dates:
        raise ValueError("At least one date is required")
    for value in dates:
        pd.Timestamp(value)
    return dates


def _slice_prices(
    prices: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    out = prices
    if start:
        out = out.loc[pd.Timestamp(start) :]
    if end:
        out = out.loc[: pd.Timestamp(end)]
    return out.copy()


def _stats_from_equity(equity: pd.Series, daily_rf: pd.Series) -> Dict[str, float]:
    daily_returns = equity.pct_change().fillna(0.0)
    daily_rf = daily_rf.reindex_like(daily_returns).fillna(0.0)
    stats = {
        "CAGR": cagr(equity),
        "MaxDrawdown": max_drawdown(equity),
        "Sharpe_RF0": sharpe_ratio(daily_returns),
        "Sortino_RF0": sortino_ratio(daily_returns),
        "Sharpe_Excess": sharpe_ratio(daily_returns, daily_rf=daily_rf),
        "Sortino_Excess": sortino_ratio(daily_returns, daily_rf=daily_rf),
    }
    stats["Calmar"] = (
        stats["CAGR"] / stats["MaxDrawdown"] if stats["MaxDrawdown"] > 0 else 0.0
    )
    return stats


def _period_stats_row(
    label: str,
    requested_start: Optional[str],
    requested_end: Optional[str],
    equity: pd.Series,
    daily_rf: pd.Series,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "Label": label,
        "RequestedStart": requested_start,
        "RequestedEnd": requested_end,
        "Status": "NO_DATA",
        "ActualStart": None,
        "ActualEnd": None,
        "Days": 0,
    }
    if extra:
        row.update(extra)

    equity = equity.dropna()
    if len(equity) < 2:
        return row

    row.update(
        {
            "Status": "OK",
            "ActualStart": str(equity.index[0].date()),
            "ActualEnd": str(equity.index[-1].date()),
            "Days": int(len(equity)),
        }
    )
    row.update(_stats_from_equity(equity, daily_rf=daily_rf))
    return row


def _daily_rf_for_prices(prices: pd.DataFrame, config: MatvmConfig) -> pd.Series:
    rets = simple_returns(prices)
    return get_daily_rf(rets, config)


def _git_commit() -> Optional[str]:
    try:
        repo = Path(__file__).resolve().parent
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        commit = res.stdout.strip()
        return commit or None
    except Exception:
        return None


def _run_metadata_columns(
    config: MatvmConfig,
    backtest_status: str,
    data_audit_status: str,
    git_commit: Optional[str],
) -> Dict[str, object]:
    return {
        "CashExecution": config.cash_ticker,
        "CashReturnMode": config.cash_return_mode,
        "CashReturnTicker": config.cash_return_ticker,
        "AnnualCashReturnRate": float(config.annual_cash_return_rate),
        "RiskFreeMode": config.risk_free_mode,
        "RiskFreeTicker": config.risk_free_ticker,
        "AnnualRiskFreeRate": float(config.annual_risk_free_rate),
        "BacktestStatus": backtest_status,
        "DataAuditStatus": data_audit_status,
        "GitCommit": git_commit,
    }


def _add_run_metadata(df: pd.DataFrame, metadata: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    for key, value in reversed(list(metadata.items())):
        out.insert(0, key, value)
    return out


def _json_value(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if np.isnan(value) or np.isinf(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    return value


def _metric_value(value: object) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if np.isnan(out) or np.isinf(out):
        return None
    return out


def _run_mode(config: MatvmConfig) -> str:
    if config.cash_return_mode == "constant":
        cash = f"cash=constant:{config.annual_cash_return_rate:.2%}"
    elif config.cash_return_mode == "ticker":
        cash = f"cash=ticker:{config.cash_return_ticker}"
    else:
        cash = "cash=zero"

    if config.risk_free_mode == "constant":
        rf = f"rf=constant:{config.annual_risk_free_rate:.2%}"
    elif config.risk_free_mode == "ticker":
        rf = f"rf=ticker:{config.risk_free_ticker}"
    else:
        rf = "rf=zero"

    return f"{cash}; {rf}"


def _first_ok_row(df: pd.DataFrame, key: str, value: str) -> Optional[pd.Series]:
    if df.empty or key not in df.columns:
        return None
    rows = df[(df[key] == value) & (df.get("Status", "") == "OK")]
    if rows.empty:
        return None
    return rows.iloc[0]


def _best_benchmark(df: pd.DataFrame, metric: str, mode: str) -> Optional[str]:
    if df.empty or metric not in df.columns:
        return None

    candidates = df[df.get("Status", "") == "OK"].copy()
    if "BenchmarkNote" in candidates.columns:
        candidates = candidates[candidates["BenchmarkNote"] != "DIAGNOSTIC_ONLY"]
    candidates[metric] = pd.to_numeric(candidates[metric], errors="coerce")
    candidates = candidates.dropna(subset=[metric])
    if candidates.empty:
        return None

    if mode == "min":
        row = candidates.loc[candidates[metric].idxmin()]
    else:
        row = candidates.loc[candidates[metric].idxmax()]
    return str(row.get("Label") or row.get("Benchmark"))


def _active_rows_with_prefixes(df: pd.DataFrame, prefixes: Sequence[str]) -> pd.DataFrame:
    if df.empty or "Benchmark" not in df.columns:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    names = df["Benchmark"].astype(str)
    for prefix in prefixes:
        mask = mask | names.str.startswith(prefix)
    rows = df[mask & (df.get("Status", "") == "OK")].copy()
    if rows.empty:
        return rows
    rows["Benchmark_Sharpe_Excess"] = pd.to_numeric(
        rows["Benchmark_Sharpe_Excess"], errors="coerce"
    )
    return rows.dropna(subset=["Benchmark_Sharpe_Excess"])


def _best_active_row(df: pd.DataFrame, prefixes: Sequence[str]) -> Optional[pd.Series]:
    rows = _active_rows_with_prefixes(df, prefixes)
    if rows.empty:
        return None
    return rows.loc[rows["Benchmark_Sharpe_Excess"].idxmax()]


def _classify_strategy(summary: Dict[str, object]) -> Tuple[str, List[str]]:
    reasons: List[str] = []

    if summary.get("DataAuditStatus") != "PASS":
        return "DIAGNOSTIC_ONLY", ["Data audit did not pass"]

    worst_wf = _metric_value(summary.get("WalkForwardWorstSharpe"))
    negative_folds = int(summary.get("WalkForwardNegativeFoldCount") or 0)
    candidate_negative_folds = int(summary.get("WalkForwardCandidateNegativeFoldCount") or 0)
    strategy_sharpe = _metric_value(summary.get("Strategy_Sharpe_Excess"))
    ew_sharpe_delta = _metric_value(summary.get("EqualWeightRisk_Sharpe_Delta"))
    ew_dd_reduction = _metric_value(summary.get("EqualWeightRisk_Drawdown_Reduction"))
    vol_target_sharpe_delta = _metric_value(summary.get("BestVolTarget_Sharpe_Delta"))
    exposure_sharpe_delta = _metric_value(summary.get("ExposureMatched_Sharpe_Excess_Delta"))
    same_cash_sharpe_delta = _metric_value(summary.get("SameCashSchedule_Sharpe_Excess_Delta"))
    cash_timing = _metric_value(summary.get("CashTimingContribution"))
    asset_selection = _metric_value(summary.get("AssetSelectionContribution"))
    dynamic_cash_policy_delta = _metric_value(
        summary.get("DynamicCashTiming_Sharpe_Delta_vs_BestCashPolicy")
    )
    base_vs_eq_selected = _metric_value(
        summary.get("BaseVsEqualWeightSelected_Sharpe_Excess_Delta")
    )
    base_vs_top_momentum = _metric_value(summary.get("BaseVsTopMomentum_Sharpe_Excess_Delta"))
    base_vs_min_vol = _metric_value(summary.get("BaseVsMinVol_Sharpe_Excess_Delta"))
    base_vs_random_median = _metric_value(
        summary.get("BaseVsRandomMedian_Sharpe_Excess_Delta")
    )
    base_random_percentile_sharpe = _metric_value(
        summary.get("BaseRandomPercentile_Sharpe_Excess")
    )
    selected_random_percentile_sharpe = _metric_value(
        summary.get("SelectedCandidateRandomPercentile_Sharpe_Excess")
    )
    random_beat_base_rate_sharpe = _metric_value(
        summary.get("RandomBeatBaseRate_Sharpe_Excess")
    )
    selected_static_sharpe_delta = _metric_value(
        summary.get("SelectedCandidate_Sharpe_Delta_vs_StaticMatched_EqualWeightRisk_Cash")
    )
    selected_same_cash_sharpe_delta = _metric_value(
        summary.get("SelectedCandidate_Sharpe_Delta_vs_SameCashSchedule_EqualWeightRisk")
    )
    selected_dd_reduction = _metric_value(
        summary.get("SelectedCandidate_Drawdown_Reduction_vs_EqualWeightRisk")
    )
    candidate_mean_active_sharpe = _metric_value(
        summary.get("WalkForwardCandidateMeanActiveSharpe")
    )
    best_cash_policy = summary.get("BestCashPolicyVariant")

    if negative_folds > 0:
        reasons.append("At least one walk-forward fold has negative excess Sharpe")
    if candidate_negative_folds > 0:
        reasons.append("At least one walk-forward candidate-selection fold has negative excess Sharpe")
    if worst_wf is not None:
        reasons.append(f"Worst walk-forward excess Sharpe is {worst_wf:.3f}")
    if strategy_sharpe is None or strategy_sharpe < DECISION_MIN_EXCESS_SHARPE:
        reasons.append(
            f"Strategy excess Sharpe is below {DECISION_MIN_EXCESS_SHARPE:.2f}"
        )
    if ew_sharpe_delta is not None and ew_sharpe_delta < 0:
        reasons.append("Strategy underperforms EqualWeightRisk on excess Sharpe")
    if vol_target_sharpe_delta is not None and vol_target_sharpe_delta < 0:
        reasons.append("Strategy underperforms best VolTarget VTI/cash benchmark on excess Sharpe")
    if exposure_sharpe_delta is not None:
        if exposure_sharpe_delta >= 0:
            reasons.append("Strategy beats best exposure-matched benchmark on excess Sharpe")
        else:
            reasons.append("Strategy loses to best exposure-matched benchmark on excess Sharpe")
    if same_cash_sharpe_delta is not None:
        if same_cash_sharpe_delta >= 0:
            reasons.append("Strategy beats best same-cash-schedule benchmark on excess Sharpe")
        else:
            reasons.append("Strategy loses to best same-cash-schedule benchmark on excess Sharpe")
    if ew_dd_reduction is not None and ew_dd_reduction >= DECISION_MATERIAL_DD_REDUCTION:
        reasons.append("Strategy materially reduces drawdown versus EqualWeightRisk")
    if cash_timing is not None:
        if cash_timing > 0:
            if asset_selection is not None and cash_timing > max(asset_selection, 0.0):
                reasons.append("Strategy return is mostly explained by cash timing versus static exposure")
            else:
                reasons.append("Cash timing contribution is positive versus static exposure")
        elif cash_timing < 0:
            reasons.append("Cash timing contribution is negative versus static exposure")
    if asset_selection is not None:
        if asset_selection > 0:
            reasons.append("Asset selection contribution is positive versus same cash schedule")
        elif asset_selection < 0:
            reasons.append("Asset selection contribution is negative versus same cash schedule")
    if dynamic_cash_policy_delta is not None:
        if dynamic_cash_policy_delta > 1e-6:
            reasons.append("Dynamic cash timing adds value versus tested cash-policy variants")
        elif dynamic_cash_policy_delta < -1e-6:
            reasons.append(
                f"Dynamic cash timing destroys value versus best cash policy: {best_cash_policy}"
            )
        else:
            reasons.append(f"Dynamic cash timing is tied with best cash policy: {best_cash_policy}")
    if base_vs_eq_selected is not None:
        if base_vs_eq_selected >= 0:
            reasons.append("MATVM base selection beats equal-weight selected assets on excess Sharpe")
        else:
            reasons.append("MATVM base selection loses to equal-weight selected assets on excess Sharpe")
    if base_vs_top_momentum is not None:
        if base_vs_top_momentum >= 0:
            reasons.append("MATVM base selection beats top-momentum-only selection on excess Sharpe")
        else:
            reasons.append("MATVM base selection loses to top-momentum-only selection on excess Sharpe")
    if base_vs_min_vol is not None:
        if base_vs_min_vol >= 0:
            reasons.append("MATVM base selection beats min-vol-only selection on excess Sharpe")
        else:
            reasons.append("MATVM base selection loses to min-vol-only selection on excess Sharpe")
    if base_vs_random_median is not None:
        if base_vs_random_median >= 0:
            reasons.append("MATVM base selection beats median random same-cash selection on excess Sharpe")
        else:
            reasons.append("MATVM base selection loses to median random same-cash selection on excess Sharpe")
    reasons.append("Random variants are diagnostic-only and excluded from candidate selection")
    if random_beat_base_rate_sharpe is not None:
        reasons.append(
            "Random null model beats base on Sharpe_Excess in "
            f"{random_beat_base_rate_sharpe:.1%} of runs"
        )
    if base_random_percentile_sharpe is not None:
        reasons.append(
            "Base Sharpe_Excess percentile versus random null is "
            f"{base_random_percentile_sharpe:.1%}"
        )
        if base_random_percentile_sharpe < 0.50:
            reasons.append("Asset-selection edge is not proven: base fails the random-null test")
        elif base_random_percentile_sharpe < 0.75:
            reasons.append("Asset-selection edge is not proven: base random-null percentile is below 75%")
    if selected_static_sharpe_delta is not None:
        if selected_static_sharpe_delta >= 0:
            reasons.append("Selected non-diagnostic candidate beats StaticMatched EqualWeightRisk/Cash on excess Sharpe")
        else:
            reasons.append("Selected non-diagnostic candidate loses to StaticMatched EqualWeightRisk/Cash on excess Sharpe")
    if selected_same_cash_sharpe_delta is not None:
        if selected_same_cash_sharpe_delta >= 0:
            reasons.append("Selected non-diagnostic candidate beats SameCashSchedule EqualWeightRisk on excess Sharpe")
        else:
            reasons.append("Selected non-diagnostic candidate loses to SameCashSchedule EqualWeightRisk on excess Sharpe")
    if candidate_mean_active_sharpe is not None:
        if candidate_mean_active_sharpe > 1e-6:
            reasons.append("Walk-forward candidate selection improves base on average test excess Sharpe")
        elif candidate_mean_active_sharpe < -1e-6:
            reasons.append("Walk-forward candidate selection worsens base on average test excess Sharpe")
        else:
            reasons.append("Walk-forward candidate selection is tied with base on average test excess Sharpe")

    if (
        strategy_sharpe is not None
        and strategy_sharpe >= DECISION_MIN_EXCESS_SHARPE
        and selected_dd_reduction is not None
        and selected_dd_reduction >= DECISION_MATERIAL_DD_REDUCTION
        and selected_static_sharpe_delta is not None
        and selected_static_sharpe_delta >= 0
        and selected_same_cash_sharpe_delta is not None
        and selected_same_cash_sharpe_delta >= 0
        and negative_folds == 0
        and candidate_negative_folds == 0
        and random_beat_base_rate_sharpe is not None
        and random_beat_base_rate_sharpe <= 0.25
        and max(
            base_random_percentile_sharpe
            if base_random_percentile_sharpe is not None
            else -1.0,
            selected_random_percentile_sharpe
            if selected_random_percentile_sharpe is not None
            else -1.0,
        )
        >= 0.75
    ):
        return "PASS_DEFENSIVE_CANDIDATE", reasons

    if ew_dd_reduction is not None and ew_dd_reduction >= DECISION_MATERIAL_DD_REDUCTION:
        return "MIXED_DEFENSIVE_CANDIDATE", reasons

    return "FAIL_OR_WEAK", reasons


def _fmt_metric(value: object, pct: bool = False) -> str:
    number = _metric_value(value)
    if number is None:
        return "n/a"
    return f"{number:.2%}" if pct else f"{number:.3f}"


def _robustness_decision_summary(
    config: MatvmConfig,
    baseline: BacktestResult,
    tables: Dict[str, pd.DataFrame],
    metadata: Dict[str, object],
    effective_start: str,
    effective_end: str,
) -> Dict[str, object]:
    active = tables["active_benchmarks"]
    benchmarks = tables["benchmarks"]
    walk_forward = tables["walk_forward"]
    walkforward_candidate = tables.get("walkforward_candidate_selection", pd.DataFrame())
    parameter_sweep = tables["parameter_sweep"]
    random_null = tables.get("random_null_model", pd.DataFrame())

    ew = _first_ok_row(active, "Benchmark", "EqualWeightRisk")
    static_ew = _first_ok_row(active, "Benchmark", "StaticMatched_EqualWeightRisk_Cash")
    same_ew = _first_ok_row(active, "Benchmark", "SameCashSchedule_EqualWeightRisk")
    wf_sharpe = (
        pd.to_numeric(walk_forward["Sharpe_Excess"], errors="coerce").dropna()
        if "Sharpe_Excess" in walk_forward.columns
        else pd.Series(dtype=float)
    )
    wf_candidate_sharpe = (
        pd.to_numeric(walkforward_candidate["Test_Sharpe_Excess"], errors="coerce").dropna()
        if "Test_Sharpe_Excess" in walkforward_candidate.columns
        else pd.Series(dtype=float)
    )
    wf_candidate_active_sharpe = (
        pd.to_numeric(walkforward_candidate["Test_Active_Sharpe_Excess"], errors="coerce").dropna()
        if "Test_Active_Sharpe_Excess" in walkforward_candidate.columns
        else pd.Series(dtype=float)
    )
    wf_candidate_active_cagr = (
        pd.to_numeric(walkforward_candidate["Test_Active_CAGR"], errors="coerce").dropna()
        if "Test_Active_CAGR" in walkforward_candidate.columns
        else pd.Series(dtype=float)
    )
    wf_candidate_counts = (
        walkforward_candidate["SelectedVariant"].astype(str).value_counts().to_dict()
        if "SelectedVariant" in walkforward_candidate.columns
        else {}
    )

    summary: Dict[str, object] = {
        "RunMode": _run_mode(config),
        "GitCommit": metadata.get("GitCommit"),
        "DataAuditStatus": metadata.get("DataAuditStatus"),
        "BacktestStatus": metadata.get("BacktestStatus"),
        "CashExecution": config.cash_ticker,
        "CashReturnMode": config.cash_return_mode,
        "CashReturnTicker": config.cash_return_ticker,
        "AnnualCashReturnRate": float(config.annual_cash_return_rate),
        "RiskFreeMode": config.risk_free_mode,
        "RiskFreeTicker": config.risk_free_ticker,
        "AnnualRiskFreeRate": float(config.annual_risk_free_rate),
        "ActualStart": effective_start,
        "ActualEnd": effective_end,
        "Strategy_CAGR": baseline.stats.get("CAGR"),
        "Strategy_MaxDD": baseline.stats.get("MaxDrawdown"),
        "Strategy_Sharpe_Excess": baseline.stats.get("Sharpe_Excess"),
        "Strategy_Calmar": baseline.stats.get("Calmar"),
        "BestBenchmarkBySharpe": _best_benchmark(benchmarks, "Sharpe_Excess", "max"),
        "BestBenchmarkByCAGR": _best_benchmark(benchmarks, "CAGR", "max"),
        "BestBenchmarkByDrawdown": _best_benchmark(benchmarks, "MaxDrawdown", "min"),
        "WalkForwardWorstSharpe": float(wf_sharpe.min()) if not wf_sharpe.empty else None,
        "WalkForwardNegativeFoldCount": int((wf_sharpe < 0).sum()) if not wf_sharpe.empty else 0,
        "WalkForwardSelectedVariantCounts": wf_candidate_counts,
        "WalkForwardCandidateWorstSharpe": (
            float(wf_candidate_sharpe.min()) if not wf_candidate_sharpe.empty else None
        ),
        "WalkForwardCandidateNegativeFoldCount": (
            int((wf_candidate_sharpe < 0).sum()) if not wf_candidate_sharpe.empty else 0
        ),
        "WalkForwardCandidateMedianSharpe": (
            float(wf_candidate_sharpe.median()) if not wf_candidate_sharpe.empty else None
        ),
        "WalkForwardCandidateMeanSharpe": (
            float(wf_candidate_sharpe.mean()) if not wf_candidate_sharpe.empty else None
        ),
        "WalkForwardCandidateMedianActiveSharpe": (
            float(wf_candidate_active_sharpe.median())
            if not wf_candidate_active_sharpe.empty
            else None
        ),
        "WalkForwardCandidateMeanActiveSharpe": (
            float(wf_candidate_active_sharpe.mean())
            if not wf_candidate_active_sharpe.empty
            else None
        ),
        "WalkForwardCandidateMeanActiveCAGR": (
            float(wf_candidate_active_cagr.mean()) if not wf_candidate_active_cagr.empty else None
        ),
    }

    if ew is not None:
        summary.update(
            {
                "EqualWeightRisk_CAGR_Delta": ew.get("Active_CAGR"),
                "EqualWeightRisk_Sharpe_Delta": ew.get("Active_Sharpe_Excess"),
                "EqualWeightRisk_Drawdown_Reduction": ew.get("Drawdown_Reduction"),
            }
        )
    else:
        summary.update(
            {
                "EqualWeightRisk_CAGR_Delta": None,
                "EqualWeightRisk_Sharpe_Delta": None,
                "EqualWeightRisk_Drawdown_Reduction": None,
            }
        )

    best_vol = _best_active_row(active, ("VolTarget_VTI_Cash_",))
    if best_vol is not None:
        summary.update(
            {
                "BestVolTargetBenchmark": best_vol.get("Benchmark"),
                "BestVolTarget_CAGR_Delta": best_vol.get("Active_CAGR"),
                "BestVolTarget_Sharpe_Delta": best_vol.get("Active_Sharpe_Excess"),
                "BestVolTarget_Drawdown_Reduction": best_vol.get("Drawdown_Reduction"),
            }
        )
    else:
        summary.update(
            {
                "BestVolTargetBenchmark": None,
                "BestVolTarget_CAGR_Delta": None,
                "BestVolTarget_Sharpe_Delta": None,
                "BestVolTarget_Drawdown_Reduction": None,
            }
        )

    best_static = _best_active_row(active, ("StaticMatched_",))
    best_same = _best_active_row(active, ("SameCashSchedule_",))
    best_exposure = _best_active_row(active, EXPOSURE_MATCHED_PREFIXES)

    if best_exposure is not None:
        summary.update(
            {
                "BestExposureMatchedBenchmark": best_exposure.get("Benchmark"),
                "ExposureMatched_CAGR_Delta": best_exposure.get("Active_CAGR"),
                "ExposureMatched_Sharpe_Excess_Delta": best_exposure.get("Active_Sharpe_Excess"),
                "ExposureMatched_MaxDD_Reduction": best_exposure.get("Drawdown_Reduction"),
            }
        )
    else:
        summary.update(
            {
                "BestExposureMatchedBenchmark": None,
                "ExposureMatched_CAGR_Delta": None,
                "ExposureMatched_Sharpe_Excess_Delta": None,
                "ExposureMatched_MaxDD_Reduction": None,
            }
        )

    if best_static is not None:
        summary.update(
            {
                "BestStaticMatchedBenchmark": best_static.get("Benchmark"),
                "StaticMatched_CAGR_Delta": best_static.get("Active_CAGR"),
                "StaticMatched_Sharpe_Excess_Delta": best_static.get("Active_Sharpe_Excess"),
                "StaticMatched_MaxDD_Reduction": best_static.get("Drawdown_Reduction"),
            }
        )
    else:
        summary.update(
            {
                "BestStaticMatchedBenchmark": None,
                "StaticMatched_CAGR_Delta": None,
                "StaticMatched_Sharpe_Excess_Delta": None,
                "StaticMatched_MaxDD_Reduction": None,
            }
        )

    if best_same is not None:
        summary.update(
            {
                "BestSameCashScheduleBenchmark": best_same.get("Benchmark"),
                "SameCashSchedule_CAGR_Delta": best_same.get("Active_CAGR"),
                "SameCashSchedule_Sharpe_Excess_Delta": best_same.get("Active_Sharpe_Excess"),
                "SameCashSchedule_MaxDD_Reduction": best_same.get("Drawdown_Reduction"),
            }
        )
    else:
        summary.update(
            {
                "BestSameCashScheduleBenchmark": None,
                "SameCashSchedule_CAGR_Delta": None,
                "SameCashSchedule_Sharpe_Excess_Delta": None,
                "SameCashSchedule_MaxDD_Reduction": None,
            }
        )

    def _row_metric(row: Optional[pd.Series], key: str) -> Optional[float]:
        if row is None:
            return None
        return _metric_value(row.get(key))

    static_ew_cagr = _row_metric(static_ew, "Benchmark_CAGR")
    same_ew_cagr = _row_metric(same_ew, "Benchmark_CAGR")
    ew_cagr = _row_metric(ew, "Benchmark_CAGR")
    strategy_cagr = _metric_value(summary.get("Strategy_CAGR"))

    static_ew_sharpe = _row_metric(static_ew, "Benchmark_Sharpe_Excess")
    same_ew_sharpe = _row_metric(same_ew, "Benchmark_Sharpe_Excess")
    ew_sharpe = _row_metric(ew, "Benchmark_Sharpe_Excess")
    strategy_sharpe = _metric_value(summary.get("Strategy_Sharpe_Excess"))

    summary.update(
        {
            "StaticMatched_EqualWeightRisk_CAGR": static_ew_cagr,
            "SameCashSchedule_EqualWeightRisk_CAGR": same_ew_cagr,
            "EqualWeightRisk_CAGR": ew_cagr,
            "AssetSelectionContribution": (
                strategy_cagr - same_ew_cagr
                if strategy_cagr is not None and same_ew_cagr is not None
                else None
            ),
            "CashTimingContribution": (
                same_ew_cagr - static_ew_cagr
                if same_ew_cagr is not None and static_ew_cagr is not None
                else None
            ),
            "ExposureEffect": (
                static_ew_cagr - ew_cagr
                if static_ew_cagr is not None and ew_cagr is not None
                else None
            ),
            "StaticMatched_EqualWeightRisk_Sharpe_Excess": static_ew_sharpe,
            "SameCashSchedule_EqualWeightRisk_Sharpe_Excess": same_ew_sharpe,
            "EqualWeightRisk_Sharpe_Excess": ew_sharpe,
            "AssetSelectionContribution_Sharpe_Excess": (
                strategy_sharpe - same_ew_sharpe
                if strategy_sharpe is not None and same_ew_sharpe is not None
                else None
            ),
            "CashTimingContribution_Sharpe_Excess": (
                same_ew_sharpe - static_ew_sharpe
                if same_ew_sharpe is not None and static_ew_sharpe is not None
                else None
            ),
            "ExposureEffect_Sharpe_Excess": (
                static_ew_sharpe - ew_sharpe
                if static_ew_sharpe is not None and ew_sharpe is not None
                else None
            ),
        }
    )

    cash_policy_rows = parameter_sweep[
        parameter_sweep.get("VariantGroup", pd.Series(dtype=str)).isin(
            ["DynamicCashTiming", "NoCashTiming", "CashTiming_ThresholdSweep"]
        )
        & (parameter_sweep.get("Status", "") == "OK")
    ].copy()
    if not cash_policy_rows.empty:
        cash_policy_rows["Sharpe_Excess"] = pd.to_numeric(
            cash_policy_rows["Sharpe_Excess"], errors="coerce"
        )
        cash_policy_rows = cash_policy_rows.dropna(subset=["Sharpe_Excess"])

    baseline_row = _first_ok_row(parameter_sweep, "Label", "baseline")

    def _best_variant_row(group: str) -> Optional[pd.Series]:
        rows = parameter_sweep[
            (parameter_sweep.get("VariantGroup", "") == group)
            & (parameter_sweep.get("Status", "") == "OK")
        ].copy()
        if rows.empty:
            return None
        rows["Sharpe_Excess"] = pd.to_numeric(rows["Sharpe_Excess"], errors="coerce")
        rows = rows.dropna(subset=["Sharpe_Excess"])
        if rows.empty:
            return None
        return rows.loc[rows["Sharpe_Excess"].idxmax()]

    best_cash_policy = (
        cash_policy_rows.loc[cash_policy_rows["Sharpe_Excess"].idxmax()]
        if not cash_policy_rows.empty
        else None
    )
    best_no_cash = _best_variant_row("NoCashTiming")
    best_threshold = _best_variant_row("CashTiming_ThresholdSweep")

    def _variant_value(row: Optional[pd.Series], key: str) -> Optional[float]:
        if row is None:
            return None
        return _metric_value(row.get(key))

    dynamic_sharpe = _variant_value(baseline_row, "Sharpe_Excess")
    dynamic_cagr = _variant_value(baseline_row, "CAGR")
    dynamic_maxdd = _variant_value(baseline_row, "MaxDrawdown")
    best_cash_sharpe = _variant_value(best_cash_policy, "Sharpe_Excess")
    best_cash_cagr = _variant_value(best_cash_policy, "CAGR")
    best_cash_maxdd = _variant_value(best_cash_policy, "MaxDrawdown")

    summary.update(
        {
            "BestCashPolicyVariant": best_cash_policy.get("Label") if best_cash_policy is not None else None,
            "BestCashPolicyGroup": best_cash_policy.get("VariantGroup") if best_cash_policy is not None else None,
            "BestCashPolicy_Sharpe_Excess": best_cash_sharpe,
            "BestCashPolicy_CAGR": best_cash_cagr,
            "BestCashPolicy_MaxDrawdown": best_cash_maxdd,
            "DynamicCashTiming_Sharpe_Excess": dynamic_sharpe,
            "DynamicCashTiming_CAGR": dynamic_cagr,
            "DynamicCashTiming_MaxDrawdown": dynamic_maxdd,
            "DynamicCashTiming_Sharpe_Delta_vs_BestCashPolicy": (
                dynamic_sharpe - best_cash_sharpe
                if dynamic_sharpe is not None and best_cash_sharpe is not None
                else None
            ),
            "DynamicCashTiming_CAGR_Delta_vs_BestCashPolicy": (
                dynamic_cagr - best_cash_cagr
                if dynamic_cagr is not None and best_cash_cagr is not None
                else None
            ),
            "DynamicCashTiming_MaxDD_Reduction_vs_BestCashPolicy": (
                best_cash_maxdd - dynamic_maxdd
                if dynamic_maxdd is not None and best_cash_maxdd is not None
                else None
            ),
            "BestNoCashTimingVariant": best_no_cash.get("Label") if best_no_cash is not None else None,
            "BestNoCashTiming_Sharpe_Excess": _variant_value(best_no_cash, "Sharpe_Excess"),
            "BestNoCashTiming_CAGR": _variant_value(best_no_cash, "CAGR"),
            "BestNoCashTiming_MaxDrawdown": _variant_value(best_no_cash, "MaxDrawdown"),
            "BestThresholdSweepVariant": best_threshold.get("Label") if best_threshold is not None else None,
            "BestThresholdSweep_Sharpe_Excess": _variant_value(best_threshold, "Sharpe_Excess"),
            "BestThresholdSweep_CAGR": _variant_value(best_threshold, "CAGR"),
            "BestThresholdSweep_MaxDrawdown": _variant_value(best_threshold, "MaxDrawdown"),
        }
    )

    def _variant_delta(
        left: Optional[pd.Series], right: Optional[pd.Series], key: str
    ) -> Optional[float]:
        left_value = _variant_value(left, key)
        right_value = _variant_value(right, key)
        if left_value is None or right_value is None:
            return None
        return left_value - right_value

    asset_rows = parameter_sweep[
        (parameter_sweep.get("VariantGroup", "") == "AssetSelection")
        & (parameter_sweep.get("Status", "") == "OK")
    ].copy()
    candidate_rows = parameter_sweep[
        (parameter_sweep.get("VariantRole", "") == "CANDIDATE")
        & (parameter_sweep.get("Status", "") == "OK")
    ].copy()
    for rows_df in (asset_rows, candidate_rows):
        if not rows_df.empty:
            rows_df["Sharpe_Excess"] = pd.to_numeric(
                rows_df["Sharpe_Excess"], errors="coerce"
            )
            rows_df["CAGR"] = pd.to_numeric(rows_df["CAGR"], errors="coerce")
            rows_df["MaxDrawdown"] = pd.to_numeric(
                rows_df["MaxDrawdown"], errors="coerce"
            )
            rows_df["Calmar"] = pd.to_numeric(rows_df["Calmar"], errors="coerce")

    asset_candidate_rows = asset_rows[asset_rows.get("VariantRole", "") == "CANDIDATE"].copy()
    asset_candidate_rows = asset_candidate_rows.dropna(subset=["Sharpe_Excess"])
    candidate_rows = candidate_rows.dropna(subset=["Sharpe_Excess"])

    asset_base = _first_ok_row(parameter_sweep, "Label", "AssetSelection_Base")
    asset_eq = _first_ok_row(parameter_sweep, "Label", "AssetSelection_EqualWeightSelected")
    asset_top = _first_ok_row(parameter_sweep, "Label", "AssetSelection_TopMomentumOnly")
    asset_min_vol = _first_ok_row(parameter_sweep, "Label", "AssetSelection_MinVolOnly")

    best_asset = (
        asset_candidate_rows.loc[asset_candidate_rows["Sharpe_Excess"].idxmax()]
        if not asset_candidate_rows.empty
        else None
    )
    selected_candidate = (
        candidate_rows.loc[candidate_rows["Sharpe_Excess"].idxmax()]
        if not candidate_rows.empty
        else None
    )

    base_cagr = _variant_value(asset_base, "CAGR")
    base_sharpe = _variant_value(asset_base, "Sharpe_Excess")
    base_calmar = _variant_value(asset_base, "Calmar")
    selected_sharpe = _variant_value(selected_candidate, "Sharpe_Excess")
    selected_maxdd = _variant_value(selected_candidate, "MaxDrawdown")

    asset_comparison_deltas = [
        _variant_delta(asset_base, asset_eq, "Sharpe_Excess"),
        _variant_delta(asset_base, asset_top, "Sharpe_Excess"),
        _variant_delta(asset_base, asset_min_vol, "Sharpe_Excess"),
    ]
    base_beats_simple = all(
        delta is not None and delta >= 0 for delta in asset_comparison_deltas
    )

    random_stats = random_null.copy()
    if not random_stats.empty:
        for col in ["CAGR", "Sharpe_Excess", "Calmar", "MaxDrawdown"]:
            random_stats[col] = pd.to_numeric(random_stats[col], errors="coerce")
    random_count = int(len(random_stats))
    random_median_cagr = (
        float(random_stats["CAGR"].median()) if random_count and "CAGR" in random_stats else None
    )
    random_median_sharpe = (
        float(random_stats["Sharpe_Excess"].median())
        if random_count and "Sharpe_Excess" in random_stats
        else None
    )
    random_median_calmar = (
        float(random_stats["Calmar"].median())
        if random_count and "Calmar" in random_stats
        else None
    )
    random_best_cagr = (
        float(random_stats["CAGR"].max()) if random_count and "CAGR" in random_stats else None
    )
    random_best_sharpe = (
        float(random_stats["Sharpe_Excess"].max())
        if random_count and "Sharpe_Excess" in random_stats
        else None
    )
    random_best_calmar = (
        float(random_stats["Calmar"].max()) if random_count and "Calmar" in random_stats else None
    )
    random_worst_maxdd = (
        float(random_stats["MaxDrawdown"].max())
        if random_count and "MaxDrawdown" in random_stats
        else None
    )

    def _random_percentile(metric: str, value: Optional[float]) -> Optional[float]:
        if value is None or random_stats.empty or metric not in random_stats.columns:
            return None
        series = pd.to_numeric(random_stats[metric], errors="coerce").dropna()
        if series.empty:
            return None
        return float((series <= value).mean())

    def _random_beat_rate(metric: str, value: Optional[float]) -> Optional[float]:
        if value is None or random_stats.empty or metric not in random_stats.columns:
            return None
        series = pd.to_numeric(random_stats[metric], errors="coerce").dropna()
        if series.empty:
            return None
        return float((series > value).mean())

    base_random_percentile_cagr = _random_percentile("CAGR", base_cagr)
    base_random_percentile_sharpe = _random_percentile("Sharpe_Excess", base_sharpe)
    base_random_percentile_calmar = _random_percentile("Calmar", base_calmar)
    selected_random_percentile_sharpe = _random_percentile(
        "Sharpe_Excess", selected_sharpe
    )
    random_beat_base_rate_cagr = _random_beat_rate("CAGR", base_cagr)
    random_beat_base_rate_sharpe = _random_beat_rate("Sharpe_Excess", base_sharpe)
    random_beat_base_rate_calmar = _random_beat_rate("Calmar", base_calmar)

    base_vs_random_median_sharpe = (
        base_sharpe - random_median_sharpe
        if base_sharpe is not None and random_median_sharpe is not None
        else None
    )
    base_vs_random_median_calmar = (
        base_calmar - random_median_calmar
        if base_calmar is not None and random_median_calmar is not None
        else None
    )
    if base_random_percentile_sharpe is None:
        asset_selection_decision = "INSUFFICIENT_DATA"
    elif base_random_percentile_sharpe < 0.50:
        asset_selection_decision = "BASE_FAILS_RANDOM_NULL"
    elif base_random_percentile_sharpe >= 0.75 and base_beats_simple:
        asset_selection_decision = "BASE_BEATS_SIMPLE_AND_RANDOM_NULL"
    elif base_random_percentile_sharpe >= 0.75:
        asset_selection_decision = "BASE_BEATS_RANDOM_NULL_BUT_SIMPLE_VARIANT_BEATS_BASE"
    else:
        asset_selection_decision = "INSUFFICIENT_DATA"

    selected_candidate_static_delta = (
        selected_sharpe - static_ew_sharpe
        if selected_sharpe is not None and static_ew_sharpe is not None
        else None
    )
    selected_candidate_same_cash_delta = (
        selected_sharpe - same_ew_sharpe
        if selected_sharpe is not None and same_ew_sharpe is not None
        else None
    )
    selected_candidate_dd_reduction = (
        _row_metric(ew, "Benchmark_MaxDD") - selected_maxdd
        if selected_maxdd is not None and _row_metric(ew, "Benchmark_MaxDD") is not None
        else None
    )

    summary.update(
        {
            "BestAssetSelectionVariant": best_asset.get("Label") if best_asset is not None else None,
            "BestAssetSelectionVariant_Sharpe_Excess": _variant_value(best_asset, "Sharpe_Excess"),
            "BestAssetSelectionVariant_CAGR": _variant_value(best_asset, "CAGR"),
            "SelectedCandidateVariant": (
                selected_candidate.get("Label") if selected_candidate is not None else None
            ),
            "SelectedCandidateVariantGroup": (
                selected_candidate.get("VariantGroup") if selected_candidate is not None else None
            ),
            "SelectedCandidate_Sharpe_Excess": selected_sharpe,
            "SelectedCandidate_CAGR": _variant_value(selected_candidate, "CAGR"),
            "SelectedCandidate_MaxDrawdown": selected_maxdd,
            "SelectedCandidate_Sharpe_Delta_vs_StaticMatched_EqualWeightRisk_Cash": (
                selected_candidate_static_delta
            ),
            "SelectedCandidate_Sharpe_Delta_vs_SameCashSchedule_EqualWeightRisk": (
                selected_candidate_same_cash_delta
            ),
            "SelectedCandidate_Drawdown_Reduction_vs_EqualWeightRisk": (
                selected_candidate_dd_reduction
            ),
            "SelectedCandidateRandomPercentile_Sharpe_Excess": (
                selected_random_percentile_sharpe
            ),
            "BaseVsEqualWeightSelected_CAGR_Delta": _variant_delta(asset_base, asset_eq, "CAGR"),
            "BaseVsEqualWeightSelected_Sharpe_Excess_Delta": _variant_delta(
                asset_base, asset_eq, "Sharpe_Excess"
            ),
            "BaseVsTopMomentum_CAGR_Delta": _variant_delta(asset_base, asset_top, "CAGR"),
            "BaseVsTopMomentum_Sharpe_Excess_Delta": _variant_delta(
                asset_base, asset_top, "Sharpe_Excess"
            ),
            "BaseVsMinVol_CAGR_Delta": _variant_delta(asset_base, asset_min_vol, "CAGR"),
            "BaseVsMinVol_Sharpe_Excess_Delta": _variant_delta(
                asset_base, asset_min_vol, "Sharpe_Excess"
            ),
            "BaseVsRandomMedian_CAGR_Delta": (
                base_cagr - random_median_cagr
                if base_cagr is not None and random_median_cagr is not None
                else None
            ),
            "BaseVsRandomMedian_Sharpe_Excess_Delta": base_vs_random_median_sharpe,
            "BaseVsRandomMedian_Calmar_Delta": base_vs_random_median_calmar,
            "RandomNullSeedCount": random_count,
            "RandomNullMedian_CAGR": random_median_cagr,
            "RandomNullMedian_Sharpe_Excess": random_median_sharpe,
            "RandomNullMedian_Calmar": random_median_calmar,
            "RandomNullBest_CAGR": random_best_cagr,
            "RandomNullBest_Sharpe_Excess": random_best_sharpe,
            "RandomNullBest_Calmar": random_best_calmar,
            "RandomNullWorst_MaxDD": random_worst_maxdd,
            "BaseRandomPercentile_CAGR": base_random_percentile_cagr,
            "BaseRandomPercentile_Sharpe_Excess": base_random_percentile_sharpe,
            "BaseRandomPercentile_Calmar": base_random_percentile_calmar,
            "RandomBeatBaseRate_CAGR": random_beat_base_rate_cagr,
            "RandomBeatBaseRate_Sharpe_Excess": random_beat_base_rate_sharpe,
            "RandomBeatBaseRate_Calmar": random_beat_base_rate_calmar,
            "AssetSelectionDecision": asset_selection_decision,
        }
    )

    decision, reasons = _classify_strategy(summary)
    summary["Decision"] = decision
    summary["Reason"] = reasons
    return {str(k): _json_value(v) for k, v in summary.items()}


def _write_summary_markdown(outdir: Path, summary: Dict[str, object]) -> None:
    reasons = summary.get("Reason") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    lines = [
        "# Robustness Decision Summary",
        "",
        f"Decision: {summary.get('Decision')}",
        f"Run mode: {summary.get('RunMode')}",
        f"Git commit: {summary.get('GitCommit')}",
        f"Data audit status: {summary.get('DataAuditStatus')}",
        f"Backtest status: {summary.get('BacktestStatus')}",
        "",
        "## Key Metrics",
        "",
        f"- Actual period: {summary.get('ActualStart')} to {summary.get('ActualEnd')}",
        f"- Strategy CAGR: {_fmt_metric(summary.get('Strategy_CAGR'), pct=True)}",
        f"- Strategy max drawdown: {_fmt_metric(summary.get('Strategy_MaxDD'), pct=True)}",
        f"- Strategy excess Sharpe: {_fmt_metric(summary.get('Strategy_Sharpe_Excess'))}",
        f"- Strategy Calmar: {_fmt_metric(summary.get('Strategy_Calmar'))}",
        "",
        "## EqualWeightRisk Comparison",
        "",
        f"- Active CAGR: {_fmt_metric(summary.get('EqualWeightRisk_CAGR_Delta'), pct=True)}",
        f"- Active excess Sharpe: {_fmt_metric(summary.get('EqualWeightRisk_Sharpe_Delta'))}",
        f"- Drawdown reduction: {_fmt_metric(summary.get('EqualWeightRisk_Drawdown_Reduction'), pct=True)}",
        "",
        "## VolTarget VTI/Cash Comparison",
        "",
        f"- Best VolTarget benchmark: {summary.get('BestVolTargetBenchmark')}",
        f"- Active CAGR: {_fmt_metric(summary.get('BestVolTarget_CAGR_Delta'), pct=True)}",
        f"- Active excess Sharpe: {_fmt_metric(summary.get('BestVolTarget_Sharpe_Delta'))}",
        f"- Drawdown reduction: {_fmt_metric(summary.get('BestVolTarget_Drawdown_Reduction'), pct=True)}",
        "",
        "## Exposure-Matched Comparison",
        "",
        f"- Best exposure-matched benchmark: {summary.get('BestExposureMatchedBenchmark')}",
        f"- Active CAGR: {_fmt_metric(summary.get('ExposureMatched_CAGR_Delta'), pct=True)}",
        f"- Active excess Sharpe: {_fmt_metric(summary.get('ExposureMatched_Sharpe_Excess_Delta'))}",
        f"- Drawdown reduction: {_fmt_metric(summary.get('ExposureMatched_MaxDD_Reduction'), pct=True)}",
        f"- Best static matched benchmark: {summary.get('BestStaticMatchedBenchmark')}",
        f"- Static matched active excess Sharpe: {_fmt_metric(summary.get('StaticMatched_Sharpe_Excess_Delta'))}",
        f"- Best same-cash-schedule benchmark: {summary.get('BestSameCashScheduleBenchmark')}",
        f"- Same-cash active excess Sharpe: {_fmt_metric(summary.get('SameCashSchedule_Sharpe_Excess_Delta'))}",
        "",
        "## Attribution",
        "",
        f"- Asset selection contribution: {_fmt_metric(summary.get('AssetSelectionContribution'), pct=True)}",
        f"- Cash timing contribution: {_fmt_metric(summary.get('CashTimingContribution'), pct=True)}",
        f"- Exposure effect: {_fmt_metric(summary.get('ExposureEffect'), pct=True)}",
        f"- Asset selection Sharpe contribution: {_fmt_metric(summary.get('AssetSelectionContribution_Sharpe_Excess'))}",
        f"- Cash timing Sharpe contribution: {_fmt_metric(summary.get('CashTimingContribution_Sharpe_Excess'))}",
        f"- Exposure effect Sharpe contribution: {_fmt_metric(summary.get('ExposureEffect_Sharpe_Excess'))}",
        "",
        "## Cash Policy Ablation",
        "",
        f"- Best cash policy variant: {summary.get('BestCashPolicyVariant')}",
        f"- Best cash policy group: {summary.get('BestCashPolicyGroup')}",
        f"- Best cash policy excess Sharpe: {_fmt_metric(summary.get('BestCashPolicy_Sharpe_Excess'))}",
        f"- Best cash policy CAGR: {_fmt_metric(summary.get('BestCashPolicy_CAGR'), pct=True)}",
        f"- Dynamic cash Sharpe delta: {_fmt_metric(summary.get('DynamicCashTiming_Sharpe_Delta_vs_BestCashPolicy'))}",
        f"- Dynamic cash CAGR delta: {_fmt_metric(summary.get('DynamicCashTiming_CAGR_Delta_vs_BestCashPolicy'), pct=True)}",
        f"- Dynamic cash max-DD reduction: {_fmt_metric(summary.get('DynamicCashTiming_MaxDD_Reduction_vs_BestCashPolicy'), pct=True)}",
        f"- Best no-cash-timing variant: {summary.get('BestNoCashTimingVariant')}",
        f"- Best threshold-sweep variant: {summary.get('BestThresholdSweepVariant')}",
        "",
        "## Asset Selection Ablation",
        "",
        f"- Best asset-selection variant: {summary.get('BestAssetSelectionVariant')}",
        f"- Asset-selection decision: {summary.get('AssetSelectionDecision')}",
        f"- Base vs equal-weight selected Sharpe delta: {_fmt_metric(summary.get('BaseVsEqualWeightSelected_Sharpe_Excess_Delta'))}",
        f"- Base vs top momentum Sharpe delta: {_fmt_metric(summary.get('BaseVsTopMomentum_Sharpe_Excess_Delta'))}",
        f"- Base vs min-vol Sharpe delta: {_fmt_metric(summary.get('BaseVsMinVol_Sharpe_Excess_Delta'))}",
        f"- Base vs median random Sharpe delta: {_fmt_metric(summary.get('BaseVsRandomMedian_Sharpe_Excess_Delta'))}",
        f"- Selected non-diagnostic candidate: {summary.get('SelectedCandidateVariant')}",
        f"- Selected candidate static-matched Sharpe delta: {_fmt_metric(summary.get('SelectedCandidate_Sharpe_Delta_vs_StaticMatched_EqualWeightRisk_Cash'))}",
        f"- Selected candidate same-cash Sharpe delta: {_fmt_metric(summary.get('SelectedCandidate_Sharpe_Delta_vs_SameCashSchedule_EqualWeightRisk'))}",
        "",
        "## Random Null Model",
        "",
        f"- Random null seed count: {summary.get('RandomNullSeedCount')}",
        f"- Random median excess Sharpe: {_fmt_metric(summary.get('RandomNullMedian_Sharpe_Excess'))}",
        f"- Random best excess Sharpe: {_fmt_metric(summary.get('RandomNullBest_Sharpe_Excess'))}",
        f"- Base excess-Sharpe percentile: {_fmt_metric(summary.get('BaseRandomPercentile_Sharpe_Excess'), pct=True)}",
        f"- Selected candidate excess-Sharpe percentile: {_fmt_metric(summary.get('SelectedCandidateRandomPercentile_Sharpe_Excess'), pct=True)}",
        f"- Random beat-base rate, excess Sharpe: {_fmt_metric(summary.get('RandomBeatBaseRate_Sharpe_Excess'), pct=True)}",
        f"- Base vs random median Calmar delta: {_fmt_metric(summary.get('BaseVsRandomMedian_Calmar_Delta'))}",
        "",
        "## Benchmarks",
        "",
        f"- Best by excess Sharpe: {summary.get('BestBenchmarkBySharpe')}",
        f"- Best by CAGR: {summary.get('BestBenchmarkByCAGR')}",
        f"- Best by lowest drawdown: {summary.get('BestBenchmarkByDrawdown')}",
        "",
        "## Walk Forward",
        "",
        f"- Worst excess Sharpe: {_fmt_metric(summary.get('WalkForwardWorstSharpe'))}",
        f"- Negative fold count: {summary.get('WalkForwardNegativeFoldCount')}",
        "",
        "## Walk-Forward Candidate Selection",
        "",
        f"- Selected variant counts: {summary.get('WalkForwardSelectedVariantCounts')}",
        f"- Worst test excess Sharpe: {_fmt_metric(summary.get('WalkForwardCandidateWorstSharpe'))}",
        f"- Negative fold count: {summary.get('WalkForwardCandidateNegativeFoldCount')}",
        f"- Median test excess Sharpe: {_fmt_metric(summary.get('WalkForwardCandidateMedianSharpe'))}",
        f"- Mean test excess Sharpe: {_fmt_metric(summary.get('WalkForwardCandidateMeanSharpe'))}",
        f"- Mean active test excess Sharpe: {_fmt_metric(summary.get('WalkForwardCandidateMeanActiveSharpe'))}",
        "",
        "## Reasons",
        "",
    ]
    if reasons:
        lines.extend([f"- {reason}" for reason in reasons])
    else:
        lines.append("- No gate warnings.")

    (outdir / "robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _robustness_variants(
    base_config: MatvmConfig,
    baseline_avg_cash_weight: Optional[float] = None,
) -> List[Tuple[str, MatvmConfig]]:
    avg_cash = 0.0 if baseline_avg_cash_weight is None else float(baseline_avg_cash_weight)
    avg_cash = min(max(avg_cash, 0.0), 1.0)
    definitions: List[Tuple[str, Dict[str, object]]] = [
        ("baseline", {}),
        (
            "NoCashTiming_StaticAverageCash",
            {"cash_policy": "fixed", "fixed_cash_weight": avg_cash},
        ),
        ("NoCashTiming_FixedCash_20", {"cash_policy": "fixed", "fixed_cash_weight": 0.20}),
        ("NoCashTiming_FixedCash_30", {"cash_policy": "fixed", "fixed_cash_weight": 0.30}),
        ("NoCashTiming_FixedCash_40", {"cash_policy": "fixed", "fixed_cash_weight": 0.40}),
        ("NoCashTiming_FixedCash_50", {"cash_policy": "fixed", "fixed_cash_weight": 0.50}),
        (
            "CashTiming_ThresholdSweep_Tight",
            {"dd_half": 0.04, "dd_safe": 0.08, "dd_exit": 0.03},
        ),
        (
            "CashTiming_ThresholdSweep_Medium",
            {"dd_half": 0.06, "dd_safe": 0.10, "dd_exit": 0.04},
        ),
        (
            "CashTiming_ThresholdSweep_Loose",
            {"dd_half": 0.10, "dd_safe": 0.16, "dd_exit": 0.08},
        ),
        (
            "CashTiming_ThresholdSweep_VeryLoose",
            {"dd_half": 0.12, "dd_safe": 0.20, "dd_exit": 0.10},
        ),
        (
            "AssetSelection_Base",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "base",
            },
        ),
        (
            "AssetSelection_EqualWeightSelected",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "equal_weight_selected",
            },
        ),
        (
            "AssetSelection_TopMomentumOnly",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "top_momentum_only",
            },
        ),
        (
            "AssetSelection_TopMomentumNoVolFilter",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "top_momentum_no_vol_filter",
            },
        ),
        (
            "AssetSelection_MinVolOnly",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "min_vol_only",
            },
        ),
        (
            "AssetSelection_RiskParitySelected",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "risk_parity_selected",
            },
        ),
        (
            "AssetSelection_RandomSameCashSeed_1",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "random_selected",
                "asset_selection_seed": 1,
            },
        ),
        (
            "AssetSelection_RandomSameCashSeed_2",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "random_selected",
                "asset_selection_seed": 2,
            },
        ),
        (
            "AssetSelection_RandomSameCashSeed_3",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "random_selected",
                "asset_selection_seed": 3,
            },
        ),
        (
            "AssetSelection_RandomSameCashSeed_4",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "random_selected",
                "asset_selection_seed": 4,
            },
        ),
        (
            "AssetSelection_RandomSameCashSeed_5",
            {
                "cash_policy": "fixed",
                "fixed_cash_weight": avg_cash,
                "asset_selection_mode": "random_selected",
                "asset_selection_seed": 5,
            },
        ),
        ("vol_window_40", {"vol_window": 40}),
        ("vol_window_90", {"vol_window": 90}),
        ("sma_window_150", {"sma_window": 150}),
        ("sma_window_250", {"sma_window": 250}),
        ("momentum_short_21_63_126", {"momentum_windows": (21, 63, 126)}),
        ("momentum_long_126_252_504", {"momentum_windows": (126, 252, 504)}),
        ("vol_target_6pct", {"vol_target": 0.06}),
        ("vol_target_10pct", {"vol_target": 0.10}),
        ("weight_cap_25pct", {"weight_cap": 0.25}),
        ("weight_cap_50pct", {"weight_cap": 0.50}),
        ("confirm_1", {"confirm_signals": 1}),
        ("confirm_3", {"confirm_signals": 3}),
        ("breadth_min_2", {"breadth_min": 2}),
        ("breadth_min_4", {"breadth_min": 4}),
        ("rebalance_2w", {"rebalance_freq": "2W-FRI"}),
        ("rebalance_month_end", {"rebalance_freq": "ME"}),
    ]
    return [(name, _clone_config(base_config, **updates)) for name, updates in definitions]


def _start_date_sensitivity(
    prices: pd.DataFrame,
    config: MatvmConfig,
    start_dates: Sequence[str],
    initial_capital: float,
) -> pd.DataFrame:
    rows = []
    for start in start_dates:
        sliced = _slice_prices(prices, start=start)
        try:
            res = backtest(prices=sliced, config=config, initial_capital=initial_capital)
            daily_rf = _daily_rf_for_prices(sliced, config).reindex(res.equity_curve.index)
            row = _period_stats_row(
                label=start,
                requested_start=start,
                requested_end=None,
                equity=res.equity_curve,
                daily_rf=daily_rf,
                extra={"Trades": int(len(res.trades))},
            )
        except Exception as exc:
            row = {
                "Label": start,
                "RequestedStart": start,
                "RequestedEnd": None,
                "Status": f"ERROR: {exc}",
                "ActualStart": None,
                "ActualEnd": None,
                "Days": 0,
                "Trades": 0,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _precompute_variant_results(
    prices: pd.DataFrame,
    variants: Sequence[Tuple[str, MatvmConfig]],
    initial_capital: float,
) -> Dict[str, BacktestResult]:
    out: Dict[str, BacktestResult] = {}
    for name, cfg in variants:
        out[name] = backtest(prices=prices, config=cfg, initial_capital=initial_capital)
    return out


def _variant_group(name: str) -> str:
    if name == "baseline":
        return "DynamicCashTiming"
    if name.startswith("NoCashTiming_"):
        return "NoCashTiming"
    if name.startswith("CashTiming_ThresholdSweep"):
        return "CashTiming_ThresholdSweep"
    if name.startswith("AssetSelection_"):
        return "AssetSelection"
    return "ParameterSweep"


def _variant_role(name: str) -> str:
    if name.startswith("AssetSelection_RandomSameCashSeed_"):
        return "DIAGNOSTIC_ONLY"
    return "CANDIDATE"


def _variant_avg_cash_weight(res: BacktestResult, config: MatvmConfig) -> float:
    if config.cash_ticker not in res.weights.columns or res.weights.empty:
        return float("nan")
    return float(res.weights[config.cash_ticker].mean())


def _parameter_sweep_table(
    results: Dict[str, BacktestResult],
    daily_rf: pd.Series,
    variant_configs: Dict[str, MatvmConfig],
) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        cfg = variant_configs.get(name)
        extra: Dict[str, object] = {
            "Trades": int(len(res.trades)),
            "VariantGroup": _variant_group(name),
            "VariantRole": _variant_role(name),
        }
        if cfg is not None:
            avg_cash = _variant_avg_cash_weight(res, cfg)
            extra.update(
                {
                    "CashPolicy": cfg.cash_policy,
                    "FixedCashWeight": cfg.fixed_cash_weight,
                    "AvgCashWeight": avg_cash,
                    "AvgRiskWeight": 1.0 - avg_cash if not np.isnan(avg_cash) else np.nan,
                    "AssetSelectionMode": cfg.asset_selection_mode,
                    "AssetSelectionSeed": cfg.asset_selection_seed,
                    "DDHalf": cfg.dd_half,
                    "DDSafe": cfg.dd_safe,
                    "DDExit": cfg.dd_exit,
                }
            )
        row = _period_stats_row(
            label=name,
            requested_start=None,
            requested_end=None,
            equity=res.equity_curve,
            daily_rf=daily_rf,
            extra=extra,
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Sharpe_Excess", ascending=False, na_position="last")


def _regime_table(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    daily_rf: pd.Series,
) -> pd.DataFrame:
    periods = list(REGIME_PERIODS)
    recent_start = (prices.index[-1] - pd.DateOffset(years=1)).date().isoformat()
    recent_end = prices.index[-1].date().isoformat()
    periods.append(("recent 1y", recent_start, recent_end))

    rows = []
    for label, start, end in periods:
        equity = baseline.equity_curve.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        rows.append(
            _period_stats_row(
                label=label,
                requested_start=start,
                requested_end=end,
                equity=equity,
                daily_rf=daily_rf,
            )
        )
    return pd.DataFrame(rows)


def _benchmark_return_frame(prices: pd.DataFrame, config: MatvmConfig) -> pd.DataFrame:
    rets = simple_returns(prices)
    out = pd.DataFrame(index=prices.index)
    for ticker in config.risk_tickers:
        if ticker in rets.columns:
            out[ticker] = rets[ticker]
    out[config.cash_ticker] = get_daily_cash_returns(rets, config)
    return out


def _benchmark_daily_returns(returns: pd.DataFrame, weights: Dict[str, float]) -> Optional[pd.Series]:
    active = {t: w for t, w in weights.items() if t in returns.columns}
    if not active:
        return None

    w = pd.Series(active, dtype=float)
    w = w / w.sum()
    return (returns[w.index] * w).sum(axis=1)


def _benchmark_equity(returns: pd.DataFrame, weights: Dict[str, float]) -> Optional[pd.Series]:
    bench_daily = _benchmark_daily_returns(returns, weights)
    if bench_daily is None:
        return None
    return (1.0 + bench_daily).cumprod()


def _risk_sleeve_specs(config: MatvmConfig) -> List[Tuple[str, Dict[str, float]]]:
    return [
        ("VTI", {"VTI": 1.0}),
        ("EqualWeightRisk", {t: 1.0 / len(config.risk_tickers) for t in config.risk_tickers}),
        (
            "EqualWeightFull",
            {t: 1.0 / len(config.risk_tickers) for t in config.risk_tickers},
        ),
        ("60_40", {"VTI": 0.60, "TLT": 0.40}),
    ]


def _equity_from_daily_returns(daily_returns: pd.Series, name: str) -> pd.Series:
    return (1.0 + daily_returns.fillna(0.0)).cumprod().rename(name)


def _baseline_weights_for_returns(
    baseline: BacktestResult,
    returns: pd.DataFrame,
    config: MatvmConfig,
) -> pd.DataFrame:
    weights = baseline.weights.reindex(returns.index).ffill().fillna(0.0)
    for ticker in config.all_tickers():
        if ticker not in weights.columns:
            weights[ticker] = 0.0
    return weights[config.all_tickers()]


def _static_matched_benchmark_results(
    returns: pd.DataFrame,
    config: MatvmConfig,
    baseline: BacktestResult,
) -> List[Dict[str, object]]:
    if config.cash_ticker not in returns.columns:
        return []

    weights = _baseline_weights_for_returns(
        baseline=baseline,
        returns=returns,
        config=config,
    )
    avg_cash_weight = float(weights[config.cash_ticker].mean())
    avg_cash_weight = min(max(avg_cash_weight, 0.0), 1.0)
    risk_weight = 1.0 - avg_cash_weight

    results: List[Dict[str, object]] = []
    for sleeve_name, sleeve_weights in _risk_sleeve_specs(config):
        risk_daily = _benchmark_daily_returns(returns, sleeve_weights)
        if risk_daily is None:
            results.append(
                {
                    "label": f"StaticMatched_{sleeve_name}_Cash",
                    "note": "",
                    "equity": None,
                    "avg_turnover": 0.0,
                    "matched_cash_weight": avg_cash_weight,
                }
            )
            continue
        daily = risk_weight * risk_daily + avg_cash_weight * returns[config.cash_ticker]
        results.append(
            {
                "label": f"StaticMatched_{sleeve_name}_Cash",
                "note": "",
                "equity": _equity_from_daily_returns(daily, f"StaticMatched_{sleeve_name}_Cash"),
                "avg_turnover": 0.0,
                "matched_cash_weight": avg_cash_weight,
            }
        )

    return results


def _same_cash_schedule_benchmark_results(
    returns: pd.DataFrame,
    config: MatvmConfig,
    baseline: BacktestResult,
) -> List[Dict[str, object]]:
    if config.cash_ticker not in returns.columns:
        return []

    weights = _baseline_weights_for_returns(
        baseline=baseline,
        returns=returns,
        config=config,
    )
    weights_for_return = weights.shift(1).fillna(0.0)
    cash_weight = weights_for_return[config.cash_ticker].reindex(returns.index).fillna(1.0)
    cash_weight = cash_weight.clip(lower=0.0, upper=1.0)
    risk_weight = 1.0 - cash_weight

    results: List[Dict[str, object]] = []
    for sleeve_name, sleeve_weights in _risk_sleeve_specs(config):
        risk_daily = _benchmark_daily_returns(returns, sleeve_weights)
        if risk_daily is None:
            results.append(
                {
                    "label": f"SameCashSchedule_{sleeve_name}",
                    "note": "",
                    "equity": None,
                    "avg_turnover": 0.0,
                    "matched_cash_weight": float(cash_weight.mean()),
                }
            )
            continue
        daily = risk_weight * risk_daily + cash_weight * returns[config.cash_ticker]
        results.append(
            {
                "label": f"SameCashSchedule_{sleeve_name}",
                "note": "",
                "equity": _equity_from_daily_returns(daily, f"SameCashSchedule_{sleeve_name}"),
                "avg_turnover": 0.0,
                "matched_cash_weight": float(cash_weight.mean()),
            }
        )

    return results


def _vol_target_vti_cash_benchmark(
    returns: pd.DataFrame,
    config: MatvmConfig,
    target_vol: float,
    lookback_days: int = VOL_TARGET_LOOKBACK_DAYS,
) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame], float]:
    if "VTI" not in returns.columns or config.cash_ticker not in returns.columns:
        return None, None, 0.0

    idx = returns.index
    if len(idx) < 2:
        return None, None, 0.0

    signal_dates = pd.DatetimeIndex(
        idx.to_series().resample(config.rebalance_freq).last().dropna().values
    )
    signal_set = set(signal_dates)

    w_current = pd.Series({"VTI": 0.0, config.cash_ticker: 1.0}, dtype=float)
    pending_target: Optional[pd.Series] = None
    pending_trade_date: Optional[pd.Timestamp] = None

    equity = []
    weights = []
    trade_turnover: List[float] = []
    value = 1.0
    vti_rets = returns["VTI"].fillna(0.0)
    cash_rets = returns[config.cash_ticker].fillna(0.0)

    for i, dt in enumerate(idx):
        if i > 0:
            daily_return = (
                float(w_current.get("VTI", 0.0)) * float(vti_rets.loc[dt])
                + float(w_current.get(config.cash_ticker, 0.0)) * float(cash_rets.loc[dt])
            )
            value *= 1.0 + daily_return

        if pending_trade_date is not None and dt == pending_trade_date and pending_target is not None:
            turnover = 0.5 * float((pending_target - w_current).abs().sum())
            trade_turnover.append(turnover)
            w_current = pending_target.copy()
            pending_target = None
            pending_trade_date = None

        if dt in signal_set:
            trailing = vti_rets.loc[:dt].tail(lookback_days)
            if len(trailing) >= lookback_days:
                realized_vol = float(trailing.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
                if realized_vol > 0.0 and not np.isnan(realized_vol):
                    vti_weight = min(max(target_vol / realized_vol, 0.0), 1.0)
                else:
                    vti_weight = 0.0
            else:
                vti_weight = 0.0

            pending_target = pd.Series(
                {"VTI": float(vti_weight), config.cash_ticker: 1.0 - float(vti_weight)},
                dtype=float,
            )
            if i + 1 < len(idx):
                pending_trade_date = idx[i + 1]

        equity.append(value)
        weights.append(w_current.copy())

    equity_series = pd.Series(equity, index=idx, name=f"VolTarget_VTI_Cash_{int(target_vol * 100)}pct")
    weights_df = pd.DataFrame(weights, index=idx).fillna(0.0)
    avg_turnover = float(np.mean(trade_turnover)) if trade_turnover else 0.0
    return equity_series, weights_df, avg_turnover


def _benchmark_specs(
    prices: pd.DataFrame,
    config: MatvmConfig,
    daily_rf: pd.Series,
) -> List[Tuple[str, Dict[str, float], str]]:
    returns = _benchmark_return_frame(prices, config)
    specs: List[Tuple[str, Dict[str, float], str]] = [
        ("VTI", {"VTI": 1.0}, ""),
        ("EqualWeightRisk", {t: 1.0 / len(config.risk_tickers) for t in config.risk_tickers}, ""),
        (
            "EqualWeightFull",
            {t: 1.0 / (len(config.risk_tickers) + 1) for t in config.risk_tickers + [config.cash_ticker]},
            "",
        ),
        ("60_40_VTI_TLT", {"VTI": 0.60, "TLT": 0.40}, ""),
        ("Cash", {config.cash_ticker: 1.0}, ""),
    ]

    single_asset_scores = []
    for ticker in returns.columns:
        equity = _benchmark_equity(returns, {ticker: 1.0})
        if equity is None:
            continue
        stats = _stats_from_equity(equity, daily_rf=daily_rf)
        score = _safe_float(float(stats.get("Sharpe_Excess", np.nan)), default=-1e9)
        single_asset_scores.append((score, ticker))

    if single_asset_scores:
        _score, ticker = max(single_asset_scores, key=lambda x: x[0])
        specs.append((f"BestSingleAssetHindsight_{ticker}", {ticker: 1.0}, "DIAGNOSTIC_ONLY"))

    return specs


def _benchmark_results(
    prices: pd.DataFrame,
    config: MatvmConfig,
    daily_rf: pd.Series,
    baseline: Optional[BacktestResult] = None,
) -> List[Dict[str, object]]:
    returns = _benchmark_return_frame(prices, config)
    results: List[Dict[str, object]] = []

    for label, weights, note in _benchmark_specs(prices, config, daily_rf):
        results.append(
            {
                "label": label,
                "note": note,
                "equity": _benchmark_equity(returns, weights),
                "avg_turnover": 0.0,
            }
        )

    if baseline is not None:
        results.extend(
            _static_matched_benchmark_results(
                returns=returns,
                config=config,
                baseline=baseline,
            )
        )
        results.extend(
            _same_cash_schedule_benchmark_results(
                returns=returns,
                config=config,
                baseline=baseline,
            )
        )

    for target_vol in VOL_TARGET_BENCHMARKS:
        equity, weights_df, avg_turnover = _vol_target_vti_cash_benchmark(
            returns=returns,
            config=config,
            target_vol=target_vol,
        )
        results.append(
            {
                "label": f"VolTarget_VTI_Cash_{int(target_vol * 100)}pct",
                "note": "",
                "equity": equity,
                "weights": weights_df,
                "avg_turnover": avg_turnover,
            }
        )

    return results


def _benchmark_table(
    prices: pd.DataFrame,
    config: MatvmConfig,
    daily_rf: pd.Series,
    baseline: Optional[BacktestResult] = None,
) -> pd.DataFrame:
    rows = []
    for result in _benchmark_results(prices, config, daily_rf, baseline=baseline):
        label = str(result["label"])
        note = str(result.get("note") or "")
        equity = result.get("equity")
        if equity is None:
            rows.append(
                {
                    "Label": label,
                    "BenchmarkNote": note,
                    "Status": "NO_DATA",
                    "ActualStart": None,
                    "ActualEnd": None,
                    "Days": 0,
                }
            )
            continue

        rows.append(
            _period_stats_row(
                label=label,
                requested_start=None,
                requested_end=None,
                equity=equity,
                daily_rf=daily_rf,
                extra={
                    "BenchmarkNote": note,
                    "AvgWeeklyTurnover": float(result.get("avg_turnover") or 0.0),
                    "MatchedCashWeight": result.get("matched_cash_weight"),
                },
            )
        )
    return pd.DataFrame(rows).sort_values("Sharpe_Excess", ascending=False, na_position="last")


def _active_benchmark_table(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
    daily_rf: pd.Series,
) -> pd.DataFrame:
    rows = []
    strategy_turnover = float(baseline.stats.get("AvgWeeklyTurnover", 0.0))

    for result in _benchmark_results(prices, config, daily_rf, baseline=baseline):
        label = str(result["label"])
        note = str(result.get("note") or "")
        benchmark_equity = result.get("equity")
        if benchmark_equity is None:
            rows.append({"Benchmark": label, "BenchmarkNote": note, "Status": "NO_DATA"})
            continue

        idx = baseline.equity_curve.index.intersection(benchmark_equity.index)
        if len(idx) < 2:
            rows.append(
                {
                    "Benchmark": label,
                    "BenchmarkNote": note,
                    "Status": "NO_DATA",
                    "ActualStart": None,
                    "ActualEnd": None,
                    "Days": int(len(idx)),
                }
            )
            continue

        strategy_equity = baseline.equity_curve.loc[idx]
        benchmark_equity = benchmark_equity.loc[idx]
        rf = daily_rf.reindex(idx).fillna(0.0)

        strategy_stats = _stats_from_equity(strategy_equity, daily_rf=rf)
        benchmark_stats = _stats_from_equity(benchmark_equity, daily_rf=rf)

        benchmark_turnover = float(result.get("avg_turnover") or 0.0)
        row = {
            "Benchmark": label,
            "BenchmarkNote": note,
            "Status": "OK",
            "ActualStart": str(idx[0].date()),
            "ActualEnd": str(idx[-1].date()),
            "Days": int(len(idx)),
            "Strategy_CAGR": strategy_stats.get("CAGR"),
            "Benchmark_CAGR": benchmark_stats.get("CAGR"),
            "Active_CAGR": strategy_stats.get("CAGR") - benchmark_stats.get("CAGR"),
            "Strategy_MaxDD": strategy_stats.get("MaxDrawdown"),
            "Benchmark_MaxDD": benchmark_stats.get("MaxDrawdown"),
            "Drawdown_Reduction": benchmark_stats.get("MaxDrawdown")
            - strategy_stats.get("MaxDrawdown"),
            "Strategy_Sharpe_Excess": strategy_stats.get("Sharpe_Excess"),
            "Benchmark_Sharpe_Excess": benchmark_stats.get("Sharpe_Excess"),
            "Active_Sharpe_Excess": strategy_stats.get("Sharpe_Excess")
            - benchmark_stats.get("Sharpe_Excess"),
            "Strategy_Calmar": strategy_stats.get("Calmar"),
            "Benchmark_Calmar": benchmark_stats.get("Calmar"),
            "Active_Calmar": strategy_stats.get("Calmar") - benchmark_stats.get("Calmar"),
            "Strategy_AvgWeeklyTurnover": strategy_turnover,
            "Benchmark_AvgWeeklyTurnover": benchmark_turnover,
            "Turnover_Delta": strategy_turnover - benchmark_turnover,
            "MatchedCashWeight": result.get("matched_cash_weight"),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Active_Sharpe_Excess", ascending=False, na_position="last")


def _strategy_return_inputs(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    returns = _benchmark_return_frame(prices, config).reindex(baseline.weights.index).fillna(0.0)
    tickers = config.all_tickers()
    for ticker in tickers:
        if ticker not in returns.columns:
            returns[ticker] = 0.0

    weights = baseline.weights.reindex(returns.index).ffill().fillna(0.0)
    for ticker in tickers:
        if ticker not in weights.columns:
            weights[ticker] = 0.0

    returns = returns[tickers]
    weights = weights[tickers]
    weights_for_return = weights.shift(1).fillna(0.0)
    portfolio_returns = baseline.equity_curve.pct_change().reindex(returns.index).fillna(0.0)
    return returns, weights, weights_for_return, portfolio_returns


def _allocation_history_table(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> pd.DataFrame:
    returns, weights, weights_for_return, portfolio_returns = _strategy_return_inputs(
        baseline=baseline,
        prices=prices,
        config=config,
    )
    rows = []
    for ticker in config.all_tickers():
        contribution = weights_for_return[ticker] * returns[ticker]
        frame = pd.DataFrame(
            {
                "Date": returns.index,
                "Ticker": ticker,
                "Weight": weights_for_return[ticker].values,
                "EndOfDayWeight": weights[ticker].values,
                "IsCash": ticker == config.cash_ticker,
                "PortfolioReturn": portfolio_returns.values,
                "AssetReturn": returns[ticker].values,
                "Contribution": contribution.values,
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _asset_weight_summary_table(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> pd.DataFrame:
    returns, weights, weights_for_return, _portfolio_returns = _strategy_return_inputs(
        baseline=baseline,
        prices=prices,
        config=config,
    )

    rows = []
    for ticker in config.all_tickers():
        held = weights_for_return[ticker] > 1e-6
        rows.append(
            {
                "Ticker": ticker,
                "IsCash": ticker == config.cash_ticker,
                "AverageWeight": float(weights[ticker].mean()),
                "MedianWeight": float(weights[ticker].median()),
                "MaxWeight": float(weights[ticker].max()),
                "PercentDaysHeld": float((weights[ticker] > 1e-6).mean()),
                "AverageAppliedWeight": float(weights_for_return[ticker].mean()),
                "AvgReturnWhenHeld": float(returns.loc[held, ticker].mean()) if held.any() else np.nan,
                "HitRateWhenHeld": float((returns.loc[held, ticker] > 0.0).mean()) if held.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _cash_exposure_summary_table(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> pd.DataFrame:
    _returns, weights, _weights_for_return, _portfolio_returns = _strategy_return_inputs(
        baseline=baseline,
        prices=prices,
        config=config,
    )

    cash_weight = weights[config.cash_ticker]
    risk_weight = weights[config.risk_tickers].sum(axis=1)
    changes = weights.diff().abs().fillna(0.0)
    cash_turnover = 0.5 * float(changes[config.cash_ticker].sum())
    risk_turnover = 0.5 * float(changes[config.risk_tickers].sum(axis=1).sum())

    row = {
        "ActualStart": str(weights.index[0].date()) if len(weights.index) else None,
        "ActualEnd": str(weights.index[-1].date()) if len(weights.index) else None,
        "CashTicker": config.cash_ticker,
        "Percent_Time_In_Cash": float((cash_weight > 1e-6).mean()),
        "Percent_Majority_Cash": float((cash_weight >= 0.5).mean()),
        "Average_Cash_Weight": float(cash_weight.mean()),
        "Median_Cash_Weight": float(cash_weight.median()),
        "Max_Cash_Weight": float(cash_weight.max()),
        "Average_Risk_Asset_Weight": float(risk_weight.mean()),
        "Median_Risk_Asset_Weight": float(risk_weight.median()),
        "Turnover_From_Cash": cash_turnover,
        "Turnover_From_Risk_Assets": risk_turnover,
    }
    return pd.DataFrame([row])


def _return_contribution_by_asset_table(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> pd.DataFrame:
    returns, weights, weights_for_return, _portfolio_returns = _strategy_return_inputs(
        baseline=baseline,
        prices=prices,
        config=config,
    )
    contributions = weights_for_return * returns
    total_contribution = float(contributions.sum().sum())

    rows = []
    for ticker in config.all_tickers():
        held = weights_for_return[ticker] > 1e-6
        contribution = contributions[ticker]
        rows.append(
            {
                "Ticker": ticker,
                "IsCash": ticker == config.cash_ticker,
                "AverageWeight": float(weights[ticker].mean()),
                "ReturnContribution": float(contribution.sum()),
                "ContributionShare": (
                    float(contribution.sum() / total_contribution)
                    if abs(total_contribution) > 1e-12
                    else np.nan
                ),
                "AvgReturnWhenHeld": float(returns.loc[held, ticker].mean()) if held.any() else np.nan,
                "HitRateWhenHeld": float((returns.loc[held, ticker] > 0.0).mean()) if held.any() else np.nan,
                "WorstContribution": float(contribution.min()),
                "BestContribution": float(contribution.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("ReturnContribution", ascending=False)


def _allocation_decision_reason(
    benchmark: str,
    selected_assets: str,
    cash_weight: float,
    portfolio_return: float,
    benchmark_return: float,
    active_return: float,
) -> str:
    if cash_weight >= 0.5 and benchmark_return > portfolio_return:
        return f"Cash drag versus {benchmark}"
    if cash_weight >= 0.5 and benchmark_return < 0.0 and active_return > 0.0:
        return f"Defensive cash allocation helped versus {benchmark}"
    if not selected_assets:
        return "Cash-only allocation"
    if active_return > 0.0:
        return f"Selected risk assets beat {benchmark}"
    return f"Selected risk assets lagged {benchmark}"


def _allocation_decision_tables(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    returns, weights, _weights_for_return, portfolio_returns = _strategy_return_inputs(
        baseline=baseline,
        prices=prices,
        config=config,
    )
    benchmark_specs = [
        ("EqualWeightRisk", {t: 1.0 / len(config.risk_tickers) for t in config.risk_tickers}),
        ("VTI", {"VTI": 1.0}),
    ]

    rows = []
    strategy_weekly = (1.0 + portfolio_returns).resample("W-FRI").prod() - 1.0
    weekly_cash_weight = weights[config.cash_ticker].resample("W-FRI").mean()
    weekly_last_weights = weights.resample("W-FRI").last()

    for benchmark, benchmark_weights in benchmark_specs:
        benchmark_daily = _benchmark_daily_returns(returns, benchmark_weights)
        if benchmark_daily is None:
            continue
        benchmark_weekly = (1.0 + benchmark_daily).resample("W-FRI").prod() - 1.0
        idx = strategy_weekly.index.intersection(benchmark_weekly.index).intersection(weekly_last_weights.index)
        for dt in idx:
            row_weights = weekly_last_weights.loc[dt]
            selected_assets = [
                ticker for ticker in config.risk_tickers if float(row_weights.get(ticker, 0.0)) > 1e-4
            ]
            selected = ",".join(selected_assets)
            portfolio_return = float(strategy_weekly.loc[dt])
            benchmark_return = float(benchmark_weekly.loc[dt])
            active_return = portfolio_return - benchmark_return
            cash_weight = float(weekly_cash_weight.loc[dt])
            rows.append(
                {
                    "Date": dt,
                    "Benchmark": benchmark,
                    "SelectedAssets": selected,
                    "CashWeight": cash_weight,
                    "PortfolioReturn": portfolio_return,
                    "BenchmarkReturn": benchmark_return,
                    "ActiveReturn": active_return,
                    "Reason": _allocation_decision_reason(
                        benchmark=benchmark,
                        selected_assets=selected,
                        cash_weight=cash_weight,
                        portfolio_return=portfolio_return,
                        benchmark_return=benchmark_return,
                        active_return=active_return,
                    ),
                }
            )

    columns = [
        "Date",
        "Benchmark",
        "SelectedAssets",
        "CashWeight",
        "PortfolioReturn",
        "BenchmarkReturn",
        "ActiveReturn",
        "Reason",
    ]
    decisions = pd.DataFrame(rows, columns=columns)
    if decisions.empty:
        return decisions.copy(), decisions.copy()

    worst = decisions.sort_values("ActiveReturn", ascending=True).head(top_n).reset_index(drop=True)
    best = decisions.sort_values("ActiveReturn", ascending=False).head(top_n).reset_index(drop=True)
    return best, worst


def _allocation_diagnostic_tables(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> Dict[str, pd.DataFrame]:
    best, worst = _allocation_decision_tables(
        baseline=baseline,
        prices=prices,
        config=config,
    )
    return {
        "allocation_history": _allocation_history_table(
            baseline=baseline,
            prices=prices,
            config=config,
        ),
        "asset_weight_summary": _asset_weight_summary_table(
            baseline=baseline,
            prices=prices,
            config=config,
        ),
        "cash_exposure_summary": _cash_exposure_summary_table(
            baseline=baseline,
            prices=prices,
            config=config,
        ),
        "return_contribution_by_asset": _return_contribution_by_asset_table(
            baseline=baseline,
            prices=prices,
            config=config,
        ),
        "best_allocation_decisions": best,
        "worst_allocation_decisions": worst,
    }


def _robustness_table_filename(name: str) -> str:
    if name == "parameter_sweep":
        return "robustness_parameter_variants.csv"
    if name in DIAGNOSTIC_OUTPUT_TABLES:
        return f"{name}.csv"
    return f"robustness_{name}.csv"


def _plot_defensive_benchmark_charts(
    baseline: BacktestResult,
    prices: pd.DataFrame,
    config: MatvmConfig,
    daily_rf: pd.Series,
    outdir: Path,
) -> List[str]:
    if not _HAVE_MPL:
        return []

    benchmark_results = _benchmark_results(prices, config, daily_rf)
    wanted = {
        "VTI",
        "EqualWeightRisk",
        "VolTarget_VTI_Cash_8pct",
        "VolTarget_VTI_Cash_10pct",
        "VolTarget_VTI_Cash_12pct",
    }

    series: Dict[str, pd.Series] = {
        "MATVM": baseline.equity_curve / baseline.equity_curve.iloc[0],
    }
    for result in benchmark_results:
        label = str(result["label"])
        equity = result.get("equity")
        if label in wanted and isinstance(equity, pd.Series) and len(equity) > 1:
            series[label] = equity / equity.iloc[0]

    if len(series) < 2:
        return []

    common_index = series["MATVM"].index
    for values in series.values():
        common_index = common_index.intersection(values.index)
    if len(common_index) < 2:
        return []

    paths: List[str] = []

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    for label, values in series.items():
        values.loc[common_index].plot(ax=ax, label=label)
    ax.set_title("Equity Curve vs Benchmarks")
    ax.set_xlabel("")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    equity_path = outdir / "equity_curve_vs_benchmarks.png"
    fig.savefig(equity_path, dpi=150)
    plt.close(fig)
    paths.append(equity_path.name)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    for label, values in series.items():
        aligned = values.loc[common_index]
        drawdown = 1.0 - aligned / aligned.cummax()
        drawdown.plot(ax=ax, label=label)
    ax.set_title("Drawdown vs Benchmarks")
    ax.set_xlabel("")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    drawdown_path = outdir / "drawdown_vs_benchmarks.png"
    fig.savefig(drawdown_path, dpi=150)
    plt.close(fig)
    paths.append(drawdown_path.name)

    return paths


def _random_null_signal_schedule(
    prices: pd.DataFrame,
    config: MatvmConfig,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, List[str]]]:
    signal_dates = pd.DatetimeIndex(
        prices.index.to_series().resample(config.rebalance_freq).last().dropna().values
    )
    signal_set = set(signal_dates)
    strategy = MatvmStrategy(config=config)
    schedule: List[Tuple[pd.Timestamp, pd.Timestamp, List[str]]] = []
    idx = prices.index

    for i, dt in enumerate(idx):
        if dt not in signal_set:
            continue
        if i + 1 >= len(idx):
            continue
        if len(prices.loc[:dt]) < config.required_history_days():
            schedule.append((idx[i + 1], dt, []))
            continue

        q, _sigma, sma200 = strategy._compute_q_scores(prices, dt)
        p0 = prices.loc[dt, config.risk_tickers]
        trend = p0 > sma200
        e_raw = ((q > 0.0) & trend).astype(int)
        e_hyst = strategy._update_hysteresis(e_raw)
        active = [
            ticker
            for ticker in config.risk_tickers
            if float(e_hyst.get(ticker, 0.0)) > 0.0
        ]
        schedule.append((idx[i + 1], dt, active))

    return schedule


def _random_null_target_weights(
    config: MatvmConfig,
    seed: int,
    signal_date: pd.Timestamp,
    active: Sequence[str],
) -> pd.Series:
    target = pd.Series(0.0, index=config.all_tickers(), dtype=float)
    fixed_cash = 0.0 if config.fixed_cash_weight is None else float(config.fixed_cash_weight)
    fixed_cash = min(max(fixed_cash, 0.0), 1.0)

    if not active:
        target[config.cash_ticker] = 1.0
        return target

    day_key = int(pd.Timestamp(signal_date).strftime("%Y%m%d"))
    rng = np.random.default_rng(int(seed) * 1_000_003 + day_key)
    chosen = str(rng.choice(list(active)))
    target[chosen] = 1.0 - fixed_cash
    target[config.cash_ticker] = fixed_cash
    return target / target.sum()


def _simulate_random_null_seed(
    returns: pd.DataFrame,
    config: MatvmConfig,
    schedule: Sequence[Tuple[pd.Timestamp, pd.Timestamp, List[str]]],
    seed: int,
) -> Tuple[pd.Series, pd.Series]:
    tickers = config.all_tickers()
    targets = {
        trade_date: _random_null_target_weights(
            config=config,
            seed=seed,
            signal_date=signal_date,
            active=active,
        )
        for trade_date, signal_date, active in schedule
    }

    w_current = pd.Series(0.0, index=tickers, dtype=float)
    w_current[config.cash_ticker] = 1.0
    value = 1.0
    cost_rate = (config.tcost_bps + config.slippage_bps) / 10_000.0

    equity: List[float] = []
    cash_weights: List[float] = []
    idx = returns.index

    for i, dt in enumerate(idx):
        if i > 0:
            daily_return = float((w_current * returns.loc[dt, tickers]).sum())
            value *= 1.0 + daily_return

        target = targets.get(dt)
        if target is not None:
            diff = (target - w_current).abs()
            rel = diff / (w_current.abs() + 1e-12)
            trigger = (diff > config.rebalance_abs_threshold) | (
                rel > config.rebalance_rel_threshold
            )
            if bool(trigger.any()):
                turnover = 0.5 * float(diff.sum())
                value *= max(0.0, 1.0 - turnover * cost_rate)
                w_current = target.copy()

        equity.append(value)
        cash_weights.append(float(w_current.get(config.cash_ticker, 0.0)))

    return (
        pd.Series(equity, index=idx, name=f"random_null_{seed}"),
        pd.Series(cash_weights, index=idx, name=f"cash_weight_{seed}"),
    )


def _random_null_model_table(
    prices: pd.DataFrame,
    config: MatvmConfig,
    daily_rf: pd.Series,
    base_result: BacktestResult,
    baseline_avg_cash_weight: float,
    seed_count: int,
) -> pd.DataFrame:
    seed_count = max(int(seed_count), 0)
    columns = [
        "Seed",
        "CAGR",
        "MaxDrawdown",
        "Sharpe_RF0",
        "Sharpe_Excess",
        "Sortino_RF0",
        "Sortino_Excess",
        "Calmar",
        "AvgCashWeight",
        "PercentTimeInCash",
        "BeatsBase_CAGR",
        "BeatsBase_Sharpe_Excess",
        "BeatsBase_Calmar",
        "BeatsBase_MaxDD",
    ]
    if seed_count <= 0:
        return pd.DataFrame(columns=columns)

    null_config = _clone_config(
        config,
        cash_policy="fixed",
        fixed_cash_weight=float(min(max(baseline_avg_cash_weight, 0.0), 1.0)),
        asset_selection_mode="random_selected",
    )
    returns = _benchmark_return_frame(prices, null_config).reindex(prices.index).fillna(0.0)
    for ticker in null_config.all_tickers():
        if ticker not in returns.columns:
            returns[ticker] = 0.0
    returns = returns[null_config.all_tickers()]

    schedule = _random_null_signal_schedule(prices=prices, config=null_config)
    base_stats = _stats_from_equity(base_result.equity_curve, daily_rf=daily_rf)

    rows = []
    for seed in range(1, seed_count + 1):
        equity, cash_weights = _simulate_random_null_seed(
            returns=returns,
            config=null_config,
            schedule=schedule,
            seed=seed,
        )
        stats = _stats_from_equity(equity, daily_rf=daily_rf)
        rows.append(
            {
                "Seed": seed,
                "CAGR": stats.get("CAGR"),
                "MaxDrawdown": stats.get("MaxDrawdown"),
                "Sharpe_RF0": stats.get("Sharpe_RF0"),
                "Sharpe_Excess": stats.get("Sharpe_Excess"),
                "Sortino_RF0": stats.get("Sortino_RF0"),
                "Sortino_Excess": stats.get("Sortino_Excess"),
                "Calmar": stats.get("Calmar"),
                "AvgCashWeight": float(cash_weights.mean()),
                "PercentTimeInCash": float((cash_weights > 1e-6).mean()),
                "BeatsBase_CAGR": stats.get("CAGR") > base_stats.get("CAGR"),
                "BeatsBase_Sharpe_Excess": stats.get("Sharpe_Excess")
                > base_stats.get("Sharpe_Excess"),
                "BeatsBase_Calmar": stats.get("Calmar") > base_stats.get("Calmar"),
                "BeatsBase_MaxDD": stats.get("MaxDrawdown")
                < base_stats.get("MaxDrawdown"),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _walk_forward_table(
    results: Dict[str, BacktestResult],
    prices: pd.DataFrame,
    daily_rf: pd.Series,
    score_metric: str,
    min_test_days: int = 126,
) -> pd.DataFrame:
    rows = []
    data_start = prices.index[0]
    data_end = prices.index[-1]
    fold_train_start = data_start
    fold_num = 1

    while True:
        train_end = fold_train_start + pd.DateOffset(years=2) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = min(test_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), data_end)
        if test_start >= data_end or train_end >= data_end:
            break
        if len(prices.loc[test_start:test_end]) < min_test_days:
            break

        scored: List[Tuple[float, str, Dict[str, float]]] = []
        for name, res in results.items():
            if _variant_role(name) == "DIAGNOSTIC_ONLY":
                continue
            train_equity = res.equity_curve.loc[fold_train_start:train_end]
            if len(train_equity) < 2:
                continue
            train_stats = _stats_from_equity(train_equity, daily_rf=daily_rf)
            score = _safe_float(float(train_stats.get(score_metric, np.nan)), default=-1e9)
            scored.append((score, name, train_stats))

        if not scored:
            break

        scored.sort(reverse=True, key=lambda x: x[0])
        train_score, selected_name, selected_train_stats = scored[0]
        selected_equity = results[selected_name].equity_curve.loc[test_start:test_end]
        row = _period_stats_row(
            label=f"fold_{fold_num}",
            requested_start=str(test_start.date()),
            requested_end=str(test_end.date()),
            equity=selected_equity,
            daily_rf=daily_rf,
            extra={
                "TrainStart": str(fold_train_start.date()),
                "TrainEnd": str(train_end.date()),
                "SelectedVariant": selected_name,
                "TrainScoreMetric": score_metric,
                "TrainScore": train_score,
                "TrainSharpe_Excess": selected_train_stats["Sharpe_Excess"],
                "TrainCalmar": selected_train_stats["Calmar"],
            },
        )
        rows.append(row)

        fold_num += 1
        fold_train_start = fold_train_start + pd.DateOffset(years=1)

    return pd.DataFrame(rows)


def _candidate_variant_label(name: str) -> str:
    return "Base" if name == "baseline" else name


def _walk_forward_candidate_selection_table(
    results: Dict[str, BacktestResult],
    prices: pd.DataFrame,
    daily_rf: pd.Series,
    score_metric: str,
    min_test_days: int = 126,
) -> pd.DataFrame:
    rows = []
    data_start = prices.index[0]
    data_end = prices.index[-1]
    fold_train_start = data_start
    fold_num = 1
    candidate_names = [
        name
        for name in WALK_FORWARD_CANDIDATE_VARIANTS
        if name in results and _variant_role(name) != "DIAGNOSTIC_ONLY"
    ]
    if "baseline" not in results or not candidate_names:
        return pd.DataFrame(rows)

    while True:
        train_end = fold_train_start + pd.DateOffset(years=2) - pd.DateOffset(days=1)
        test_start = train_end + pd.DateOffset(days=1)
        test_end = min(test_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), data_end)
        if test_start >= data_end or train_end >= data_end:
            break
        if len(prices.loc[test_start:test_end]) < min_test_days:
            break

        scored: List[Tuple[float, str, Dict[str, float]]] = []
        for name in candidate_names:
            train_equity = results[name].equity_curve.loc[fold_train_start:train_end]
            if len(train_equity) < 2:
                continue
            train_stats = _stats_from_equity(train_equity, daily_rf=daily_rf)
            score_value = _metric_value(train_stats.get(score_metric))
            score = -1e9 if score_value is None else score_value
            scored.append((score, name, train_stats))

        if not scored:
            break

        scored.sort(reverse=True, key=lambda x: x[0])
        _train_score, selected_name, selected_train_stats = scored[0]
        selected_equity = results[selected_name].equity_curve.loc[test_start:test_end]
        baseline_equity = results["baseline"].equity_curve.loc[test_start:test_end]
        idx = selected_equity.index.intersection(baseline_equity.index)
        if len(idx) < 2:
            break

        test_equity = selected_equity.loc[idx]
        benchmark_equity = baseline_equity.loc[idx]
        rf = daily_rf.reindex(idx).fillna(0.0)
        test_stats = _stats_from_equity(test_equity, daily_rf=rf)
        benchmark_stats = _stats_from_equity(benchmark_equity, daily_rf=rf)

        rows.append(
            {
                "Fold": f"fold_{fold_num}",
                "TrainStart": str(fold_train_start.date()),
                "TrainEnd": str(train_end.date()),
                "TestStart": str(idx[0].date()),
                "TestEnd": str(idx[-1].date()),
                "SelectedVariant": _candidate_variant_label(selected_name),
                "Train_CAGR": selected_train_stats.get("CAGR"),
                "Train_MaxDD": selected_train_stats.get("MaxDrawdown"),
                "Train_Sharpe_Excess": selected_train_stats.get("Sharpe_Excess"),
                "Train_Calmar": selected_train_stats.get("Calmar"),
                "Test_CAGR": test_stats.get("CAGR"),
                "Test_MaxDD": test_stats.get("MaxDrawdown"),
                "Test_Sharpe_Excess": test_stats.get("Sharpe_Excess"),
                "Test_Calmar": test_stats.get("Calmar"),
                "Test_Benchmark": "Base",
                "Test_Active_CAGR": test_stats.get("CAGR") - benchmark_stats.get("CAGR"),
                "Test_Active_Sharpe_Excess": test_stats.get("Sharpe_Excess")
                - benchmark_stats.get("Sharpe_Excess"),
                "Test_Drawdown_Reduction": benchmark_stats.get("MaxDrawdown")
                - test_stats.get("MaxDrawdown"),
            }
        )

        fold_num += 1
        fold_train_start = fold_train_start + pd.DateOffset(years=1)

    return pd.DataFrame(rows)


def run_robustness_analysis(
    prices: pd.DataFrame,
    config: MatvmConfig,
    initial_capital: float,
    start_dates: Sequence[str],
    outdir: Path,
    score_metric: str = "Sharpe_Excess",
    backtest_status: str = "UNKNOWN",
    data_audit_status: str = "UNKNOWN",
    random_null_seeds: int = DEFAULT_RANDOM_NULL_SEEDS,
) -> Dict[str, pd.DataFrame]:
    prices = _ensure_datetime_index(prices).sort_index()
    required = config.required_price_tickers()
    prices = prices[required].ffill().dropna()
    daily_rf = _daily_rf_for_prices(prices, config)

    baseline = backtest(prices=prices, config=config, initial_capital=initial_capital)
    baseline_avg_cash = _variant_avg_cash_weight(baseline, config)
    variants = _robustness_variants(config, baseline_avg_cash_weight=baseline_avg_cash)
    variant_configs = {name: cfg for name, cfg in variants}
    variant_results = _precompute_variant_results(
        prices=prices,
        variants=variants,
        initial_capital=initial_capital,
    )

    tables = {
        "start_dates": _start_date_sensitivity(
            prices=prices,
            config=config,
            start_dates=start_dates,
            initial_capital=initial_capital,
        ),
        "parameter_sweep": _parameter_sweep_table(
            results=variant_results,
            daily_rf=daily_rf,
            variant_configs=variant_configs,
        ),
        "random_null_model": _random_null_model_table(
            prices=prices,
            config=config,
            daily_rf=daily_rf,
            base_result=variant_results["AssetSelection_Base"],
            baseline_avg_cash_weight=baseline_avg_cash,
            seed_count=random_null_seeds,
        ),
        "regimes": _regime_table(
            baseline=baseline,
            prices=prices,
            daily_rf=daily_rf,
        ),
        "benchmarks": _benchmark_table(
            prices=prices,
            config=config,
            daily_rf=daily_rf,
            baseline=baseline,
        ),
        "active_benchmarks": _active_benchmark_table(
            baseline=baseline,
            prices=prices,
            config=config,
            daily_rf=daily_rf,
        ),
        "walk_forward": _walk_forward_table(
            results=variant_results,
            prices=prices,
            daily_rf=daily_rf,
            score_metric=score_metric,
        ),
        "walkforward_candidate_selection": _walk_forward_candidate_selection_table(
            results=variant_results,
            prices=prices,
            daily_rf=daily_rf,
            score_metric=score_metric,
        ),
    }
    tables.update(
        _allocation_diagnostic_tables(
            baseline=baseline,
            prices=prices,
            config=config,
        )
    )

    git_commit = _git_commit()
    metadata = _run_metadata_columns(
        config=config,
        backtest_status=backtest_status,
        data_audit_status=data_audit_status,
        git_commit=git_commit,
    )
    decision_summary = _robustness_decision_summary(
        config=config,
        baseline=baseline,
        tables=tables,
        metadata=metadata,
        effective_start=str(prices.index[0].date()),
        effective_end=str(prices.index[-1].date()),
    )
    tables = {name: _add_run_metadata(df, metadata) for name, df in tables.items()}

    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(outdir / _robustness_table_filename(name), index=False, lineterminator="\n")
    plot_files = _plot_defensive_benchmark_charts(
        baseline=baseline,
        prices=prices,
        config=config,
        daily_rf=daily_rf,
        outdir=outdir,
    )

    summary = {
        "run_at_utc": _dt_utc_now().isoformat(),
        "score_metric": score_metric,
        "tables": {name: _robustness_table_filename(name) for name in tables},
        "plots": plot_files,
    }
    summary.update(decision_summary)
    (outdir / "robustness_summary.json").write_text(
        json.dumps(_json_value(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_summary_markdown(outdir, decision_summary)
    return tables


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


def _apply_common_config_args(cfg: MatvmConfig, args: argparse.Namespace) -> MatvmConfig:
    if args.risk:
        cfg.risk_tickers = [s.strip().upper() for s in args.risk.split(",") if s.strip()]
    if args.cash:
        cfg.cash_ticker = args.cash.strip().upper()
    cfg.cash_return_mode = args.cash_return_mode.strip().lower()
    cfg.cash_return_ticker = (
        args.cash_return_ticker.strip().upper() if args.cash_return_ticker else cfg.cash_ticker
    )
    cfg.annual_cash_return_rate = float(args.annual_cash_return_rate)
    cfg.risk_free_mode = args.risk_free_mode.strip().lower()
    cfg.risk_free_ticker = (
        args.risk_free_ticker.strip().upper() if args.risk_free_ticker else None
    )
    cfg.annual_risk_free_rate = float(args.annual_risk_free_rate)

    if cfg.risk_free_mode == "ticker" and not cfg.risk_free_ticker:
        raise SystemExit("--risk-free-ticker is required when --risk-free-mode ticker")
    if cfg.cash_return_mode == "ticker" and not cfg.cash_return_ticker:
        raise SystemExit("--cash-return-ticker is required when --cash-return-mode ticker")

    return cfg


def _print_data_audit(audit: DataAuditResult) -> None:
    if not audit.has_issues():
        return
    print("\n=== DATA AUDIT ===")
    for err in audit.errors:
        print(f"ERROR:   {err}")
    for warning in audit.warnings:
        print(f"WARNING: {warning}")


def cli_backtest(args: argparse.Namespace) -> None:
    cfg = _apply_common_config_args(_make_default_config(), args)

    tickers = cfg.required_price_tickers()

    if args.csv_folder:
        prices = load_prices_from_csv(Path(args.csv_folder), tickers)
    else:
        prices = load_prices_from_yfinance(tickers, start=args.start, end=args.end)

    audit = audit_price_data(prices, cfg)
    if audit.has_issues():
        _print_data_audit(audit)
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
    print(f"{'Cash execution':>22}: {cfg.cash_ticker}")
    print(f"{'Cash return mode':>22}: {cfg.cash_return_mode}")
    if cfg.cash_return_mode == "ticker":
        print(f"{'Cash return ticker':>22}: {cfg.cash_return_ticker}")
    elif cfg.cash_return_mode == "constant":
        print(f"{'Annual cash return':>22}: {cfg.annual_cash_return_rate:.2%}")
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


def cli_robustness(args: argparse.Namespace) -> None:
    cfg = _apply_common_config_args(_make_default_config(), args)
    tickers = cfg.required_price_tickers()

    if args.csv_folder:
        prices = load_prices_from_csv(Path(args.csv_folder), tickers)
    else:
        prices = load_prices_from_yfinance(tickers, start=args.start, end=args.end)

    audit = audit_price_data(prices, cfg)
    if audit.has_issues():
        _print_data_audit(audit)
        if audit.errors:
            raise SystemExit("Data audit failed with errors.")
        if not args.allow_data_warnings:
            raise SystemExit("Robustness analysis requires clean data. Use --allow-data-warnings to override.")

    start_dates = _parse_date_list(args.start_dates)
    outdir = Path(args.outdir)
    backtest_status = "DIAGNOSTIC ONLY" if audit.has_issues() else "CLEAN DATA CHECK PASSED"
    data_audit_status = "WARNINGS" if audit.warnings else "PASS"
    tables = run_robustness_analysis(
        prices=prices,
        config=cfg,
        initial_capital=float(args.initial),
        start_dates=start_dates,
        outdir=outdir,
        score_metric=args.score_metric,
        backtest_status=backtest_status,
        data_audit_status=data_audit_status,
        random_null_seeds=int(args.random_null_seeds),
    )

    print("\n=== Robustness status ===")
    if audit.has_issues():
        print(f"{'Backtest status':>22}: DIAGNOSTIC ONLY")
        print(f"{'Reason':>22}: data audit warnings detected")
    else:
        print(f"{'Backtest status':>22}: CLEAN DATA CHECK PASSED")
    print(f"{'Cash execution':>22}: {cfg.cash_ticker}")
    print(f"{'Cash return mode':>22}: {cfg.cash_return_mode}")
    if cfg.cash_return_mode == "ticker":
        print(f"{'Cash return ticker':>22}: {cfg.cash_return_ticker}")
    elif cfg.cash_return_mode == "constant":
        print(f"{'Annual cash return':>22}: {cfg.annual_cash_return_rate:.2%}")
    print(f"{'Risk-free mode':>22}: {cfg.risk_free_mode}")
    if cfg.risk_free_mode == "ticker":
        print(f"{'Risk-free ticker':>22}: {cfg.risk_free_ticker}")
    print(f"{'Output folder':>22}: {outdir.resolve()}")

    summary_path = outdir / "robustness_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print("\n=== Decision summary ===")
    print(f"{'Decision':>22}: {summary.get('Decision')}")
    print(f"{'Strategy excess Sharpe':>22}: {_fmt_metric(summary.get('Strategy_Sharpe_Excess'))}")
    print(f"{'EW Risk Sharpe delta':>22}: {_fmt_metric(summary.get('EqualWeightRisk_Sharpe_Delta'))}")
    print(f"{'EW Risk DD reduction':>22}: {_fmt_metric(summary.get('EqualWeightRisk_Drawdown_Reduction'), pct=True)}")
    print(f"{'Best VolTarget':>22}: {summary.get('BestVolTargetBenchmark')}")
    print(f"{'VolTarget Sharpe delta':>22}: {_fmt_metric(summary.get('BestVolTarget_Sharpe_Delta'))}")
    print(f"{'VolTarget DD reduction':>22}: {_fmt_metric(summary.get('BestVolTarget_Drawdown_Reduction'), pct=True)}")
    print(f"{'Best exposure match':>22}: {summary.get('BestExposureMatchedBenchmark')}")
    print(f"{'Exposure Sharpe delta':>22}: {_fmt_metric(summary.get('ExposureMatched_Sharpe_Excess_Delta'))}")
    print(f"{'Exposure DD reduction':>22}: {_fmt_metric(summary.get('ExposureMatched_MaxDD_Reduction'), pct=True)}")
    print(f"{'Best same-cash':>22}: {summary.get('BestSameCashScheduleBenchmark')}")
    print(f"{'Same-cash Sharpe delta':>22}: {_fmt_metric(summary.get('SameCashSchedule_Sharpe_Excess_Delta'))}")
    print(f"{'Same-cash DD reduction':>22}: {_fmt_metric(summary.get('SameCashSchedule_MaxDD_Reduction'), pct=True)}")
    print(f"{'Cash timing contrib':>22}: {_fmt_metric(summary.get('CashTimingContribution'), pct=True)}")
    print(f"{'Asset select contrib':>22}: {_fmt_metric(summary.get('AssetSelectionContribution'), pct=True)}")
    print(f"{'Best cash policy':>22}: {summary.get('BestCashPolicyVariant')}")
    print(f"{'Cash policy group':>22}: {summary.get('BestCashPolicyGroup')}")
    print(f"{'Dyn cash Sharpe delta':>22}: {_fmt_metric(summary.get('DynamicCashTiming_Sharpe_Delta_vs_BestCashPolicy'))}")
    print(f"{'Dyn cash CAGR delta':>22}: {_fmt_metric(summary.get('DynamicCashTiming_CAGR_Delta_vs_BestCashPolicy'), pct=True)}")
    print(f"{'Best asset select':>22}: {summary.get('BestAssetSelectionVariant')}")
    print(f"{'Asset select decision':>22}: {summary.get('AssetSelectionDecision')}")
    print(f"{'Base vs EqSel Sharpe':>22}: {_fmt_metric(summary.get('BaseVsEqualWeightSelected_Sharpe_Excess_Delta'))}")
    print(f"{'Base vs TopMom Sharpe':>22}: {_fmt_metric(summary.get('BaseVsTopMomentum_Sharpe_Excess_Delta'))}")
    print(f"{'Base vs MinVol Sharpe':>22}: {_fmt_metric(summary.get('BaseVsMinVol_Sharpe_Excess_Delta'))}")
    print(f"{'Base vs RandMed':>22}: {_fmt_metric(summary.get('BaseVsRandomMedian_Sharpe_Excess_Delta'))}")
    print(f"{'Selected candidate':>22}: {summary.get('SelectedCandidateVariant')}")
    print(f"{'Sel vs static Sharpe':>22}: {_fmt_metric(summary.get('SelectedCandidate_Sharpe_Delta_vs_StaticMatched_EqualWeightRisk_Cash'))}")
    print(f"{'Sel vs same Sharpe':>22}: {_fmt_metric(summary.get('SelectedCandidate_Sharpe_Delta_vs_SameCashSchedule_EqualWeightRisk'))}")
    print(f"{'Random null seeds':>22}: {summary.get('RandomNullSeedCount')}")
    print(f"{'Base random pctile':>22}: {_fmt_metric(summary.get('BaseRandomPercentile_Sharpe_Excess'), pct=True)}")
    print(f"{'Random beat base':>22}: {_fmt_metric(summary.get('RandomBeatBaseRate_Sharpe_Excess'), pct=True)}")
    print(f"{'WF negative folds':>22}: {summary.get('WalkForwardNegativeFoldCount')}")
    print(f"{'WF cand neg folds':>22}: {summary.get('WalkForwardCandidateNegativeFoldCount')}")
    print(f"{'WF cand mean Sharpe':>22}: {_fmt_metric(summary.get('WalkForwardCandidateMeanSharpe'))}")
    reasons = summary.get("Reason") or []
    if reasons:
        print(f"{'Primary reason':>22}: {reasons[0]}")

    exposure_cols = [
        "Percent_Time_In_Cash",
        "Average_Cash_Weight",
        "Median_Cash_Weight",
        "Max_Cash_Weight",
        "Average_Risk_Asset_Weight",
        "Turnover_From_Cash",
        "Turnover_From_Risk_Assets",
    ]
    print("\n=== Allocation exposure summary ===")
    exposure = tables["cash_exposure_summary"]
    print(exposure[[c for c in exposure_cols if c in exposure.columns]].to_string(index=False))

    contribution_cols = [
        "Ticker",
        "AverageWeight",
        "ReturnContribution",
        "ContributionShare",
        "AvgReturnWhenHeld",
        "HitRateWhenHeld",
        "WorstContribution",
        "BestContribution",
    ]
    print("\n=== Return contribution by asset ===")
    contrib = tables["return_contribution_by_asset"]
    print(contrib[[c for c in contribution_cols if c in contrib.columns]].to_string(index=False))

    display_cols = [
        "Label",
        "VariantGroup",
        "VariantRole",
        "CashPolicy",
        "FixedCashWeight",
        "AvgCashWeight",
        "AssetSelectionMode",
        "AssetSelectionSeed",
        "Status",
        "ActualStart",
        "ActualEnd",
        "CAGR",
        "MaxDrawdown",
        "Sharpe_RF0",
        "Sharpe_Excess",
        "Calmar",
    ]
    print("\n=== Start-date sensitivity ===")
    print(tables["start_dates"][[c for c in display_cols if c in tables["start_dates"].columns]].to_string(index=False))

    print("\n=== Top parameter variants by Sharpe_Excess ===")
    sweep = tables["parameter_sweep"].head(8)
    print(sweep[[c for c in display_cols if c in sweep.columns]].to_string(index=False))

    print("\n=== Regime breakdown ===")
    print(tables["regimes"][[c for c in display_cols if c in tables["regimes"].columns]].to_string(index=False))

    print("\n=== Benchmarks ===")
    print(tables["benchmarks"][[c for c in display_cols if c in tables["benchmarks"].columns]].to_string(index=False))

    active_cols = [
        "Benchmark",
        "BenchmarkNote",
        "Status",
        "MatchedCashWeight",
        "Strategy_CAGR",
        "Benchmark_CAGR",
        "Active_CAGR",
        "Strategy_MaxDD",
        "Benchmark_MaxDD",
        "Drawdown_Reduction",
        "Strategy_Sharpe_Excess",
        "Benchmark_Sharpe_Excess",
        "Active_Sharpe_Excess",
        "Strategy_Calmar",
        "Benchmark_Calmar",
        "Turnover_Delta",
    ]
    print("\n=== Active benchmark comparison ===")
    print(tables["active_benchmarks"][[c for c in active_cols if c in tables["active_benchmarks"].columns]].to_string(index=False))

    if not tables["walk_forward"].empty:
        wf_cols = [
            "Label",
            "Status",
            "TrainStart",
            "TrainEnd",
            "RequestedStart",
            "RequestedEnd",
            "SelectedVariant",
            "TrainScore",
            "Sharpe_Excess",
            "Calmar",
        ]
        print("\n=== Walk-forward selected variants ===")
        print(tables["walk_forward"][[c for c in wf_cols if c in tables["walk_forward"].columns]].to_string(index=False))

    if not tables["walkforward_candidate_selection"].empty:
        candidate_cols = [
            "Fold",
            "TrainStart",
            "TrainEnd",
            "TestStart",
            "TestEnd",
            "SelectedVariant",
            "Train_Sharpe_Excess",
            "Test_Sharpe_Excess",
            "Test_Active_Sharpe_Excess",
            "Test_Drawdown_Reduction",
        ]
        print("\n=== Walk-forward candidate selection ===")
        print(
            tables["walkforward_candidate_selection"][
                [c for c in candidate_cols if c in tables["walkforward_candidate_selection"].columns]
            ].to_string(index=False)
        )


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
        "--cash-return-mode",
        choices=sorted(CASH_RETURN_MODES),
        default="ticker",
        help="Historical return model for the backtested cash sleeve",
    )
    b.add_argument(
        "--cash-return-ticker",
        default=None,
        help="Ticker to use when --cash-return-mode ticker. Defaults to --cash.",
    )
    b.add_argument(
        "--annual-cash-return-rate",
        type=float,
        default=0.0,
        help="Decimal annual cash return when --cash-return-mode constant, e.g. 0.02",
    )
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

    # Robustness
    r = sub.add_parser("robustness", help="Run robustness analysis on clean backtest data")
    r.add_argument("--start", default="2014-01-01")
    r.add_argument("--end", default="2025-12-31")
    r.add_argument("--initial", default="100000")
    r.add_argument("--risk", default=None, help="Comma-separated risk tickers")
    r.add_argument("--cash", default=None, help="Cash ticker")
    r.add_argument(
        "--cash-return-mode",
        choices=sorted(CASH_RETURN_MODES),
        default="ticker",
        help="Historical return model for the backtested cash sleeve",
    )
    r.add_argument(
        "--cash-return-ticker",
        default=None,
        help="Ticker to use when --cash-return-mode ticker. Defaults to --cash.",
    )
    r.add_argument(
        "--annual-cash-return-rate",
        type=float,
        default=0.0,
        help="Decimal annual cash return when --cash-return-mode constant, e.g. 0.02",
    )
    r.add_argument(
        "--risk-free-mode",
        choices=sorted(RISK_FREE_MODES),
        default="zero",
        help="Benchmark for excess-return Sharpe/Sortino",
    )
    r.add_argument(
        "--risk-free-ticker",
        default=None,
        help="Ticker to use when --risk-free-mode ticker",
    )
    r.add_argument(
        "--annual-risk-free-rate",
        type=float,
        default=0.0,
        help="Decimal annual rate to use when --risk-free-mode constant, e.g. 0.05",
    )
    r.add_argument(
        "--csv-folder",
        default="./data",
        help="Folder with <TICKER>.csv files. If it contains data/live, that folder is preferred.",
    )
    r.add_argument(
        "--start-dates",
        default=",".join(ROBUSTNESS_START_DATES),
        help="Comma-separated start dates for sensitivity analysis",
    )
    r.add_argument(
        "--score-metric",
        choices=["Sharpe_Excess", "Sharpe_RF0", "Calmar", "CAGR"],
        default="Sharpe_Excess",
        help="Metric used to select variants in walk-forward analysis",
    )
    r.add_argument(
        "--random-null-seeds",
        type=int,
        default=DEFAULT_RANDOM_NULL_SEEDS,
        help="Number of random same-cash null-model seeds to evaluate",
    )
    r.add_argument("--outdir", default="./matvm_out/robustness")
    r.add_argument(
        "--allow-data-warnings",
        action="store_true",
        help="Run even when the data audit reports warnings",
    )
    r.set_defaults(func=cli_robustness)

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
