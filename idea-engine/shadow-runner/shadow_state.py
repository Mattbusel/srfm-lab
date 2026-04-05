"""
Per-shadow state management
============================
Each shadow strategy carries its own physics, statistical models, and
account state.  Everything is serialisable to JSON so state can be
checkpointed and restored from idea_engine.db.

Classes
-------
ShadowState    — top-level container; positions, equity, trade log, genome
ShadowPhysics  — EMA-based BH (Black Hole) momentum physics
ShadowGARCH    — Rolling GARCH(1,1)-like volatility tracker
ShadowOU       — Ornstein-Uhlenbeck mean-reversion detector
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class VirtualTrade:
    """A single paper trade executed by a shadow strategy."""

    trade_id: str
    shadow_id: str
    symbol: str
    side: str            # 'buy' | 'sell'
    qty: float
    price: float
    ts: float            # Unix timestamp
    signal_source: str   # 'bh' | 'garch' | 'ou' | 'combined'
    pnl: float = 0.0     # Realised P&L (filled on close)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "shadow_id": self.shadow_id,
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "price": self.price,
            "ts": self.ts,
            "signal_source": self.signal_source,
            "pnl": self.pnl,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VirtualTrade":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# BH physics
# ---------------------------------------------------------------------------

class ShadowPhysics:
    """
    BH (Black Hole) EMA momentum physics for a single genome.

    The BH model tracks a "gravity well" EMA and its collapse threshold.
    When the fast-EMA crosses beyond the collapse threshold the model
    emits a directional signal.

    Parameters
    ----------
    bh_form : float
        EMA shape exponent (controls how sharply the well focuses).
    bh_decay : float
        EMA decay factor (≈ 2/(N+1) for an N-bar EMA).
    bh_collapse : float
        Fraction of the EMA range that triggers a collapse signal.
    bh_ctl_min : int
        Minimum bars required in a trend before a signal fires.
    """

    def __init__(
        self,
        bh_form: float = 1.84,
        bh_decay: float = 0.955,
        bh_collapse: float = 0.75,
        bh_ctl_min: int = 3,
    ) -> None:
        self.bh_form = bh_form
        self.bh_decay = bh_decay
        self.bh_collapse = bh_collapse
        self.bh_ctl_min = int(bh_ctl_min)

        # State
        self._ema: float | None = None
        self._ema_slow: float | None = None
        self._trend_bars: int = 0
        self._prev_price: float | None = None
        self._gravity_well: float = 0.0
        self._range_tracker: float = 0.0    # Running range estimate

    def update(self, price: float) -> float:
        """
        Feed a new price and return the BH signal in [-1, +1].

        +1 = strong upward collapse, -1 = strong downward collapse.
        Values between indicate the degree of gravitational pull.
        """
        if self._ema is None:
            self._ema = price
            self._ema_slow = price
            self._prev_price = price
            return 0.0

        # Fast EMA (short period derived from bh_decay)
        alpha_fast = 1.0 - self.bh_decay
        alpha_slow = alpha_fast * 0.5

        self._ema = self._ema + alpha_fast * (price - self._ema)
        self._ema_slow = self._ema_slow + alpha_slow * (price - self._ema_slow)

        # Range tracker (exponentially weighted absolute change)
        abs_move = abs(price - self._prev_price)
        self._range_tracker = 0.9 * self._range_tracker + 0.1 * abs_move

        # Gravity well: how far is fast EMA from slow EMA relative to range
        diff = self._ema - self._ema_slow
        rng = max(self._range_tracker, 1e-8)

        # Apply bh_form power law — sharpens the well
        raw_signal = math.copysign(
            min(1.0, (abs(diff) / rng) ** self.bh_form),
            diff,
        )

        # Collapse threshold gate
        collapse_threshold = self.bh_collapse
        if abs(raw_signal) >= collapse_threshold:
            if math.copysign(1, raw_signal) == math.copysign(1, self._prev_signal if hasattr(self, "_prev_signal") else raw_signal):
                self._trend_bars += 1
            else:
                self._trend_bars = 1
        else:
            self._trend_bars = max(0, self._trend_bars - 1)

        self._prev_price = price
        self._prev_signal = raw_signal  # type: ignore[attr-defined]

        # Emit signal only after ctl_min bars of sustained trend
        if self._trend_bars >= self.bh_ctl_min:
            return float(raw_signal)
        return 0.0

    @property
    def ema(self) -> float | None:
        return self._ema

    def reset(self) -> None:
        self._ema = None
        self._ema_slow = None
        self._trend_bars = 0
        self._prev_price = None
        self._gravity_well = 0.0
        self._range_tracker = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bh_form": self.bh_form,
            "bh_decay": self.bh_decay,
            "bh_collapse": self.bh_collapse,
            "bh_ctl_min": self.bh_ctl_min,
            "_ema": self._ema,
            "_ema_slow": self._ema_slow,
            "_trend_bars": self._trend_bars,
            "_prev_price": self._prev_price,
            "_gravity_well": self._gravity_well,
            "_range_tracker": self._range_tracker,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShadowPhysics":
        obj = cls(
            bh_form=d["bh_form"],
            bh_decay=d["bh_decay"],
            bh_collapse=d["bh_collapse"],
            bh_ctl_min=d["bh_ctl_min"],
        )
        obj._ema = d.get("_ema")
        obj._ema_slow = d.get("_ema_slow")
        obj._trend_bars = d.get("_trend_bars", 0)
        obj._prev_price = d.get("_prev_price")
        obj._gravity_well = d.get("_gravity_well", 0.0)
        obj._range_tracker = d.get("_range_tracker", 0.0)
        return obj


# ---------------------------------------------------------------------------
# GARCH volatility tracker
# ---------------------------------------------------------------------------

class ShadowGARCH:
    """
    Lightweight GARCH(1,1)-like rolling volatility tracker.

    Tracks conditional variance σ² using:
        σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}

    where ω, α, β are derived from ``garch_target_vol`` so the unconditional
    variance equals (garch_target_vol / √252)².

    The signal returned is ``current_vol / target_vol`` — values > 1 indicate
    elevated volatility (risk-off), < 1 indicate calm (risk-on).

    Parameters
    ----------
    garch_target_vol : float
        Annualised target volatility (e.g. 1.0 = 100 %).
    """

    # Standard GARCH(1,1) persistence params
    _ALPHA = 0.10
    _BETA  = 0.85

    def __init__(self, garch_target_vol: float = 1.0) -> None:
        self.garch_target_vol = garch_target_vol
        # Daily target vol
        self._daily_target = garch_target_vol / math.sqrt(252)
        self._target_var = self._daily_target ** 2
        # ω chosen so unconditional variance = target_var
        self._omega = self._target_var * (1.0 - self._ALPHA - self._BETA)
        # State
        self._cond_var: float = self._target_var
        self._prev_return: float = 0.0
        self._prev_price: float | None = None
        self._bar_count: int = 0

    def update(self, price: float) -> float:
        """
        Feed a new price and return the volatility ratio (current / target).

        Returns 1.0 until at least 2 bars have been observed.
        """
        if self._prev_price is None:
            self._prev_price = price
            return 1.0

        r = math.log(price / max(self._prev_price, 1e-12))
        self._cond_var = (
            self._omega
            + self._ALPHA * self._prev_return ** 2
            + self._BETA * self._cond_var
        )
        self._cond_var = max(self._cond_var, 1e-10)
        self._prev_return = r
        self._prev_price = price
        self._bar_count += 1

        current_vol = math.sqrt(self._cond_var)
        return current_vol / max(self._daily_target, 1e-10)

    @property
    def current_vol(self) -> float:
        """Current daily conditional volatility."""
        return math.sqrt(self._cond_var)

    @property
    def annualised_vol(self) -> float:
        """Annualised conditional volatility."""
        return math.sqrt(self._cond_var * 252)

    def to_dict(self) -> dict[str, Any]:
        return {
            "garch_target_vol": self.garch_target_vol,
            "_cond_var": self._cond_var,
            "_prev_return": self._prev_return,
            "_prev_price": self._prev_price,
            "_bar_count": self._bar_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShadowGARCH":
        obj = cls(garch_target_vol=d["garch_target_vol"])
        obj._cond_var = d.get("_cond_var", obj._target_var)
        obj._prev_return = d.get("_prev_return", 0.0)
        obj._prev_price = d.get("_prev_price")
        obj._bar_count = d.get("_bar_count", 0)
        return obj


# ---------------------------------------------------------------------------
# OU mean-reversion detector
# ---------------------------------------------------------------------------

class ShadowOU:
    """
    Ornstein-Uhlenbeck mean-reversion detector.

    Estimates the OU spread:
        Z_t = (price_t − μ) / σ_ou

    where μ and σ_ou are estimated from a rolling window.
    The ``ou_frac`` parameter scales the reversion signal strength.

    Parameters
    ----------
    ou_frac : float
        Signal scale factor (default 0.10).
    window : int
        Rolling window length for μ / σ estimation (default 20 bars).
    """

    def __init__(self, ou_frac: float = 0.10, window: int = 20) -> None:
        self.ou_frac = ou_frac
        self.window = window
        self._prices: list[float] = []

    def update(self, price: float) -> float:
        """
        Feed a new price and return the scaled OU z-score in [-1, +1].

        Negative values → price above mean (revert short opportunity).
        Positive values → price below mean (revert long opportunity).
        (Sign convention: positive signal = go long expecting mean reversion.)
        """
        self._prices.append(price)
        if len(self._prices) > self.window:
            self._prices.pop(0)

        if len(self._prices) < 4:
            return 0.0

        arr = self._prices
        mu = sum(arr) / len(arr)
        variance = sum((x - mu) ** 2 for x in arr) / max(len(arr) - 1, 1)
        sigma = math.sqrt(max(variance, 1e-12))

        z = (price - mu) / sigma
        # Invert: if price is above mean we expect reversion downward = negative signal
        signal = -math.tanh(self.ou_frac * z)
        return float(signal)

    def half_life(self) -> float:
        """
        Estimate the OU half-life from the autocorrelation of the price series.

        Returns ``inf`` if the series appears non-stationary.
        """
        if len(self._prices) < 6:
            return float("inf")
        arr = self._prices
        n = len(arr)
        mu = sum(arr) / n
        c0 = sum((x - mu) ** 2 for x in arr) / n
        c1 = sum((arr[i] - mu) * (arr[i - 1] - mu) for i in range(1, n)) / n
        if c0 < 1e-12 or c1 / c0 >= 1.0:
            return float("inf")
        # OU half-life: -ln(2) / ln(rho) where rho = c1/c0
        rho = c1 / c0
        if rho <= 0:
            return float("inf")
        return -math.log(2) / math.log(rho)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ou_frac": self.ou_frac,
            "window": self.window,
            "_prices": list(self._prices),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShadowOU":
        obj = cls(ou_frac=d["ou_frac"], window=d["window"])
        obj._prices = list(d.get("_prices", []))
        return obj


# ---------------------------------------------------------------------------
# ShadowState — top-level per-shadow container
# ---------------------------------------------------------------------------

@dataclass
class ShadowState:
    """
    Complete per-shadow state.

    Fields
    ------
    shadow_id    : unique identifier for this shadow instance
    genome_id    : FK to hall_of_fame.id
    genome       : dict of genome params
    equity       : current virtual equity ($)
    positions    : {symbol -> qty} (positive = long, negative = short)
    trades       : list of VirtualTrade
    physics      : ShadowPhysics instance
    garch        : ShadowGARCH instance
    ou           : ShadowOU instance
    bar_count    : number of bars processed
    last_ts      : Unix timestamp of last bar
    """

    shadow_id: str
    genome_id: int
    genome: dict[str, Any]
    equity: float = 100_000.0
    positions: dict[str, float] = field(default_factory=dict)
    trades: list[VirtualTrade] = field(default_factory=list)
    physics: ShadowPhysics = field(default_factory=ShadowPhysics)
    garch: ShadowGARCH = field(default_factory=ShadowGARCH)
    ou: ShadowOU = field(default_factory=ShadowOU)
    bar_count: int = 0
    last_ts: float = 0.0
    initial_equity: float = 100_000.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_genome(cls, genome_id: int, genome: dict[str, Any]) -> "ShadowState":
        """Construct a fresh ShadowState from a genome dict."""
        shadow_id = f"shadow-{genome_id}-{uuid.uuid4().hex[:8]}"
        physics = ShadowPhysics(
            bh_form=genome.get("bh_form", 1.84),
            bh_decay=genome.get("bh_decay", 0.955),
            bh_collapse=genome.get("bh_collapse", 0.75),
            bh_ctl_min=int(genome.get("bh_ctl_min", 3)),
        )
        garch = ShadowGARCH(
            garch_target_vol=genome.get("garch_target_vol", 1.0),
        )
        ou = ShadowOU(
            ou_frac=genome.get("ou_frac", 0.10),
        )
        return cls(
            shadow_id=shadow_id,
            genome_id=genome_id,
            genome=genome,
            physics=physics,
            garch=garch,
            ou=ou,
        )

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def compute_signal(self, bar: dict[str, Any]) -> dict[str, float]:
        """
        Feed a bar and return a dict of signal components.

        Parameters
        ----------
        bar : dict with keys: open, high, low, close, volume, ts, symbol

        Returns
        -------
        dict:
            bh_signal    — BH physics signal [-1, +1]
            garch_ratio  — vol ratio (>1 = high vol)
            ou_signal    — OU reversion signal [-1, +1]
            combined     — blended signal [-1, +1]
        """
        close = float(bar.get("close", bar.get("c", 0.0)))
        bh = self.physics.update(close)
        garch_r = self.garch.update(close)
        ou = self.ou.update(close)
        self.bar_count += 1
        self.last_ts = float(bar.get("ts", time.time()))

        # Position size scaling from genome
        cf_scale = 1.0
        # Apply regime-dependent CF scale (simple heuristic: compare short vs long ema)
        if self.physics.ema is not None:
            if bh > 0.3:
                cf_scale = float(self.genome.get("cf_scale_bull", 1.0))
            elif bh < -0.3:
                cf_scale = float(self.genome.get("cf_scale_bear", 1.0))
            else:
                cf_scale = float(self.genome.get("cf_scale_neutral", 1.0))

        # Vol dampener: if vol > 1.5× target, scale down signal
        vol_damp = min(1.0, 1.0 / max(garch_r, 0.1))

        # Combine: BH drives direction, OU adds/subtracts mean-rev edge
        bh_weight = 0.70
        ou_weight = 0.30
        combined = float((bh_weight * bh + ou_weight * ou) * vol_damp * cf_scale)
        combined = max(-1.0, min(1.0, combined))

        return {
            "bh_signal": bh,
            "garch_ratio": garch_r,
            "ou_signal": ou,
            "combined": combined,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise entire shadow state to JSON string."""
        return json.dumps({
            "shadow_id": self.shadow_id,
            "genome_id": self.genome_id,
            "genome": self.genome,
            "equity": self.equity,
            "initial_equity": self.initial_equity,
            "positions": self.positions,
            "trades": [t.to_dict() for t in self.trades],
            "physics": self.physics.to_dict(),
            "garch": self.garch.to_dict(),
            "ou": self.ou.to_dict(),
            "bar_count": self.bar_count,
            "last_ts": self.last_ts,
        })

    @classmethod
    def from_json(cls, data: str | dict) -> "ShadowState":
        """Deserialise from JSON string or dict."""
        d = json.loads(data) if isinstance(data, str) else data
        trades = [VirtualTrade.from_dict(t) for t in d.get("trades", [])]
        physics = ShadowPhysics.from_dict(d["physics"])
        garch = ShadowGARCH.from_dict(d["garch"])
        ou = ShadowOU.from_dict(d["ou"])
        obj = cls(
            shadow_id=d["shadow_id"],
            genome_id=d["genome_id"],
            genome=d["genome"],
            equity=d.get("equity", 100_000.0),
            initial_equity=d.get("initial_equity", 100_000.0),
            positions=d.get("positions", {}),
            trades=trades,
            physics=physics,
            garch=garch,
            ou=ou,
            bar_count=d.get("bar_count", 0),
            last_ts=d.get("last_ts", 0.0),
        )
        return obj

    # ------------------------------------------------------------------
    # Performance helpers
    # ------------------------------------------------------------------

    def total_return(self) -> float:
        """Return (equity − initial_equity) / initial_equity."""
        return (self.equity - self.initial_equity) / max(self.initial_equity, 1e-8)

    def realised_pnl(self) -> float:
        """Sum of all realised trade P&Ls."""
        return sum(t.pnl for t in self.trades)

    def win_rate(self) -> float:
        """Fraction of profitable closed trades."""
        closed = [t for t in self.trades if t.pnl != 0.0]
        if not closed:
            return 0.0
        return sum(1 for t in closed if t.pnl > 0) / len(closed)

    def equity_curve(self) -> list[float]:
        """
        Reconstruct a simple equity curve from trade P&Ls.

        Returns a list of cumulative equity values starting from initial.
        """
        curve = [self.initial_equity]
        running = self.initial_equity
        for t in self.trades:
            running += t.pnl
            curve.append(running)
        return curve
