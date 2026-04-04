"""
Thin Python wrapper around the larsa_core Rust extension.
Falls back to pure-Python implementation if Rust is not compiled.
"""

import math
import random

# ── Attempt to import the compiled Rust extension ────────────────────────────
try:
    import larsa_core as _lc
    _RUST = True
except ImportError:
    _lc = None
    _RUST = False

# ── Regime constants (mirrored from Rust) ────────────────────────────────────
REGIME_SIDEWAYS = 0
REGIME_BULL = 1
REGIME_BEAR = 2
REGIME_HIGH_VOL = 3

# ── Default physics parameters ───────────────────────────────────────────────
_DEF_CF = 0.02
_DEF_BH_FORM = 1.5
_DEF_BH_DECAY = 0.95
_DEF_BH_COLLAPSE = 1.0
_DEF_CTL_REQ = 5


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema_series(closes, period=20):
    alpha = 2.0 / (period + 1)
    ema = closes[0]
    out = [closes[0]]
    for p in closes[1:]:
        ema = ema * (1.0 - alpha) + p * alpha
        out.append(ema)
    return out


def _run_bh_physics_py(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req):
    """Returns (masses, active, ctls, betas) — all aligned to closes."""
    n = len(closes)
    masses = [0.0] * n
    active = [False] * n
    ctls = [0] * n
    betas = [0.0] * n

    mass = 0.0
    ctl = 0
    bh_is_active = False

    for i in range(1, n):
        beta = abs(closes[i] - closes[i - 1]) / (closes[i - 1] * cf + 1e-12)
        betas[i] = beta

        if beta < 1.0:
            mass = mass * 0.97 + 0.03
            ctl += 1
        else:
            mass *= bh_decay
            ctl = 0

        if not bh_is_active and mass >= bh_form and ctl >= ctl_req:
            bh_is_active = True
        if bh_is_active and mass < bh_collapse:
            bh_is_active = False

        masses[i] = mass
        active[i] = bh_is_active
        ctls[i] = ctl

    return masses, active, ctls, betas


def _max_dd_py(equity):
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_approx_py(equity):
    if len(equity) < 2:
        return 0.0
    rets = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / n
    std = math.sqrt(var)
    if std < 1e-12:
        return 0.0
    return mean / std * math.sqrt(252)


# ─────────────────────────────────────────────────────────────────────────────
# 1. bh_mass_series
# ─────────────────────────────────────────────────────────────────────────────

def _bh_mass_series_py(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req):
    masses, active, ctls, _ = _run_bh_physics_py(
        closes, cf, bh_form, bh_decay, bh_collapse, ctl_req
    )
    return masses, active, ctls


def bh_mass_series(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req):
    """
    Compute BH mass series and active flags.
    Returns (masses: list[float], active: list[bool], ctl: list[int])
    """
    if _RUST:
        return _lc.bh_mass_series(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req)
    return _bh_mass_series_py(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req)


# ─────────────────────────────────────────────────────────────────────────────
# 2. full_backtest
# ─────────────────────────────────────────────────────────────────────────────

def _full_backtest_py(closes, highs, lows, cf, bh_form, bh_decay, bh_collapse,
                      ctl_req, long_only=False):
    n = len(closes)
    if n < 10:
        raise ValueError("Need at least 10 bars")

    masses, bh_active_vec, ctl_series, betas = _run_bh_physics_py(
        closes, cf, bh_form, bh_decay, bh_collapse, ctl_req
    )

    ema20_series = _ema_series(closes, 20)

    # Regime
    regime = [REGIME_SIDEWAYS] * n
    for i in range(1, n):
        price = closes[i]
        ema = ema20_series[i]
        act = bh_active_vec[i]
        beta = betas[i]
        if act and price > ema:
            regime[i] = REGIME_BULL
        elif act and price < ema:
            regime[i] = REGIME_BEAR
        elif not act and beta > 2.0:
            regime[i] = REGIME_HIGH_VOL
        else:
            regime[i] = REGIME_SIDEWAYS

    max_lev = 1.0
    equity = [1.0] * n
    positions = [0.0] * n
    last_pos = 0.0

    # Trade state
    in_trade = False
    trade_entry_bar = 0
    trade_entry_price = 0.0
    trade_pos = 0.0
    trade_best_price = 0.0
    trade_worst_price = 0.0
    trade_regime_at_entry = REGIME_SIDEWAYS
    trade_mass_at_entry = 0.0
    trades = []

    for i in range(1, n):
        ema = ema20_series[i]
        direction = 1.0 if closes[i] > ema else -1.0
        if long_only and direction < 0.0:
            direction = 0.0

        if bh_active_vec[i]:
            target = max_lev * direction
        elif ctl_series[i] >= 3:
            target = max_lev * 0.5 * direction
        else:
            target = 0.0

        pos_i = target if i >= 50 else 0.0
        positions[i] = pos_i

        pos_changed = abs(pos_i - last_pos) > 0.01

        if pos_changed:
            # Close existing trade
            if in_trade:
                exit_price = closes[i]
                ep = trade_entry_price
                if trade_pos > 0.0:
                    mfe_frac = (trade_best_price - ep) / ep
                    mae_frac = (ep - trade_worst_price) / ep
                    pnl_frac = (exit_price - ep) / ep
                elif trade_pos < 0.0:
                    mfe_frac = (ep - trade_worst_price) / ep
                    mae_frac = (trade_best_price - ep) / ep
                    pnl_frac = (ep - exit_price) / ep
                else:
                    mfe_frac = mae_frac = pnl_frac = 0.0
                trades.append({
                    "entry_bar": trade_entry_bar,
                    "exit_bar": i,
                    "entry_price": ep,
                    "exit_price": exit_price,
                    "pnl_frac": pnl_frac,
                    "hold_bars": i - trade_entry_bar,
                    "mfe_frac": mfe_frac,
                    "mae_frac": mae_frac,
                    "regime_at_entry": trade_regime_at_entry,
                    "bh_mass_at_entry": trade_mass_at_entry,
                    "tf_score": 1,
                })
                in_trade = False

            # Open new trade
            if abs(pos_i) > 0.01:
                in_trade = True
                trade_entry_bar = i
                trade_entry_price = closes[i]
                trade_pos = pos_i
                trade_best_price = closes[i]
                trade_worst_price = closes[i]
                trade_regime_at_entry = regime[i]
                trade_mass_at_entry = masses[i]

            last_pos = pos_i

        # Update MFE/MAE
        if in_trade:
            if highs[i] > trade_best_price:
                trade_best_price = highs[i]
            if lows[i] < trade_worst_price:
                trade_worst_price = lows[i]

        # Equity update
        ret = closes[i] / closes[i - 1] - 1.0
        equity[i] = equity[i - 1] * (1.0 + positions[i - 1] * ret)

    # Close open trade at end
    if in_trade:
        exit_price = closes[-1]
        ep = trade_entry_price
        if trade_pos > 0.0:
            mfe_frac = (trade_best_price - ep) / ep
            mae_frac = (ep - trade_worst_price) / ep
            pnl_frac = (exit_price - ep) / ep
        elif trade_pos < 0.0:
            mfe_frac = (ep - trade_worst_price) / ep
            mae_frac = (trade_best_price - ep) / ep
            pnl_frac = (ep - exit_price) / ep
        else:
            mfe_frac = mae_frac = pnl_frac = 0.0
        trades.append({
            "entry_bar": trade_entry_bar,
            "exit_bar": n - 1,
            "entry_price": ep,
            "exit_price": exit_price,
            "pnl_frac": pnl_frac,
            "hold_bars": (n - 1) - trade_entry_bar,
            "mfe_frac": mfe_frac,
            "mae_frac": mae_frac,
            "regime_at_entry": trade_regime_at_entry,
            "bh_mass_at_entry": trade_mass_at_entry,
            "tf_score": 1,
        })

    return {
        "equity_curve": equity,
        "positions": positions,
        "bh_masses": masses,
        "bh_active": bh_active_vec,
        "ctl_series": ctl_series,
        "regime": regime,
        "trades": trades,
    }


def full_backtest(closes, highs, lows, cf, bh_form, bh_decay, bh_collapse,
                  ctl_req, long_only=False):
    """
    Full SRFM backtest with MFE/MAE tracking and regime classification.

    Returns a dict with:
      equity_curve, positions, bh_masses, bh_active, ctl_series, regime, trades
    """
    if _RUST:
        return _lc.full_backtest(
            closes, highs, lows, cf, bh_form, bh_decay, bh_collapse,
            ctl_req, long_only
        )
    return _full_backtest_py(
        closes, highs, lows, cf, bh_form, bh_decay, bh_collapse,
        ctl_req, long_only
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. mc_simulation
# ─────────────────────────────────────────────────────────────────────────────

def _mc_simulation_py(returns_vec, n_sims, n_trades_per_sim, position_frac, serial_corr):
    if not returns_vec:
        raise ValueError("returns_vec must not be empty")

    neg_returns = [r for r in returns_vec if r < 0.0]
    pos_returns = [r for r in returns_vec if r >= 0.0]

    final_equities = []
    max_drawdowns = []
    blowup_count = 0

    for _ in range(max(1, n_sims)):
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        last_was_neg = False
        blowup = False

        for _ in range(max(1, n_trades_per_sim)):
            use_serial = (
                serial_corr > 1e-9
                and last_was_neg
                and neg_returns
                and pos_returns
            )
            if use_serial:
                p = random.random()
                threshold = min(0.5 + serial_corr, 1.0)
                if p < threshold:
                    drawn = random.choice(neg_returns)
                else:
                    drawn = random.choice(pos_returns)
            else:
                drawn = random.choice(returns_vec)

            last_was_neg = drawn < 0.0
            equity *= 1.0 + position_frac * drawn

            if equity <= 0.0:
                equity = 0.0
                blowup = True
                break

            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        if blowup:
            blowup_count += 1

        final_equities.append(equity)
        max_drawdowns.append(max_dd)

    return final_equities, max_drawdowns, blowup_count


def mc_simulation(returns_vec, n_sims, n_trades_per_sim, position_frac, serial_corr):
    """
    Regime-naive Monte Carlo simulation over historical trade returns.

    Parameters
    ----------
    returns_vec : list[float]
        Historical trade returns (pnl / position_dollar).
    n_sims : int
        Number of simulation paths.
    n_trades_per_sim : int
        Trades per path.
    position_frac : float
        Fraction of equity committed per trade.
    serial_corr : float
        AR(1) coefficient for loss clustering (0 = none, 0.3 = moderate).

    Returns
    -------
    (final_equities, max_drawdowns, blowup_count)
    """
    if _RUST:
        return _lc.mc_simulation(
            returns_vec, n_sims, n_trades_per_sim, position_frac, serial_corr
        )
    return _mc_simulation_py(
        returns_vec, n_sims, n_trades_per_sim, position_frac, serial_corr
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. sensitivity_sweep
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity_sweep_py(closes, highs, lows, param_name, base_value, perturbations):
    results = []
    for mult in perturbations:
        perturbed = base_value * mult

        cf = perturbed if param_name == "cf" else _DEF_CF
        bh_form = perturbed if param_name == "bh_form" else _DEF_BH_FORM
        bh_decay = perturbed if param_name == "bh_decay" else _DEF_BH_DECAY
        bh_collapse = perturbed if param_name == "bh_collapse" else _DEF_BH_COLLAPSE
        ctl_req = (
            int(round(perturbed)) if param_name == "ctl_req" else _DEF_CTL_REQ
        )

        bt = _full_backtest_py(
            closes, highs, lows,
            cf, bh_form, bh_decay, bh_collapse, ctl_req,
            long_only=False,
        )
        eq = bt["equity_curve"]
        final_eq = eq[-1] if eq else 1.0
        sh = _sharpe_approx_py(eq)
        dd = _max_dd_py(eq)
        results.append((mult, final_eq, sh, dd))
    return results


def sensitivity_sweep(closes, highs, lows, param_name, base_value, perturbations):
    """
    Sweep a single SRFM parameter across multipliers.

    Parameters
    ----------
    closes, highs, lows : list[float]
    param_name : str
        One of "cf", "bh_form", "bh_decay", "bh_collapse", "ctl_req".
    base_value : float
        Nominal value of the parameter.
    perturbations : list[float]
        Multipliers, e.g. [0.5, 0.75, 1.0, 1.25, 1.5].

    Returns
    -------
    list of (perturbation_mult, final_equity, sharpe_approx, max_drawdown)
    """
    if _RUST:
        return _lc.sensitivity_sweep(
            closes, highs, lows, param_name, base_value, perturbations
        )
    return _sensitivity_sweep_py(
        closes, highs, lows, param_name, base_value, perturbations
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. bh_correlation
# ─────────────────────────────────────────────────────────────────────────────

def _bh_correlation_py(closes_a, closes_b, cf_a, cf_b, bh_form, bh_decay, bh_collapse):
    ctl_req = _DEF_CTL_REQ
    _, active_a, _, _ = _run_bh_physics_py(
        closes_a, cf_a, bh_form, bh_decay, bh_collapse, ctl_req
    )
    _, active_b, _, _ = _run_bh_physics_py(
        closes_b, cf_b, bh_form, bh_decay, bh_collapse, ctl_req
    )

    length = min(len(active_a), len(active_b))
    if length == 0:
        return 0.0, 0.0

    a = [1.0 if x else 0.0 for x in active_a[:length]]
    b = [1.0 if x else 0.0 for x in active_b[:length]]

    both_active = sum(ai * bi for ai, bi in zip(a, b))
    either_active = sum(1.0 for ai, bi in zip(a, b) if ai > 0 or bi > 0)
    jaccard = both_active / either_active if either_active >= 1 else 0.0

    n = length
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    cov = sum((ai - mean_a) * (bi - mean_b) for ai, bi in zip(a, b)) / n
    std_a = math.sqrt(sum((ai - mean_a) ** 2 for ai in a) / n)
    std_b = math.sqrt(sum((bi - mean_b) ** 2 for bi in b) / n)
    pearson = cov / (std_a * std_b) if std_a > 1e-12 and std_b > 1e-12 else 0.0

    return jaccard, pearson


def bh_correlation(closes_a, closes_b, cf_a, cf_b, bh_form, bh_decay, bh_collapse):
    """
    Compute Jaccard and Pearson correlation between BH active series.

    Returns
    -------
    (jaccard: float, pearson: float)
    """
    if _RUST:
        return _lc.bh_correlation(
            closes_a, closes_b, cf_a, cf_b, bh_form, bh_decay, bh_collapse
        )
    return _bh_correlation_py(
        closes_a, closes_b, cf_a, cf_b, bh_form, bh_decay, bh_collapse
    )


# ─────────────────────────────────────────────────────────────────────────────
# Module info
# ─────────────────────────────────────────────────────────────────────────────

def backend():
    """Return which backend is active: 'rust' or 'python'."""
    return "rust" if _RUST else "python"


__all__ = [
    "bh_mass_series",
    "full_backtest",
    "mc_simulation",
    "sensitivity_sweep",
    "bh_correlation",
    "backend",
    "REGIME_SIDEWAYS",
    "REGIME_BULL",
    "REGIME_BEAR",
    "REGIME_HIGH_VOL",
]
