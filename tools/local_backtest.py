"""
Local BH Backtester
Uses SPY/QQQ/DIA as ES/NQ/YM proxies via yfinance.
- Daily data: 2010-present (full history)
- Hourly data: last ~3 years
- Proper mark-to-market equity at every bar
- No contract discretization (pure fractional P&L)
"""
import yfinance as yf
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INSTRUMENTS = {
    "ES": {"ticker": "SPY", "cf_1h": 0.001,  "cf_1d": 0.005},
    "NQ": {"ticker": "QQQ", "cf_1h": 0.0012, "cf_1d": 0.006},
    "YM": {"ticker": "DIA", "cf_1h": 0.0008, "cf_1d": 0.004},
}

START_DATE           = "2010-01-01"
STARTING_EQUITY      = 1_000_000.0
TAIL_FIXED_CAPITAL   = 3_000_000.0
PORTFOLIO_DAILY_RISK = 0.01
N_INST               = 3
CORR                 = 0.90
CORR_FACTOR          = math.sqrt(N_INST + N_INST * (N_INST - 1) * CORR)
PER_INST_RISK        = PORTFOLIO_DAILY_RISK / CORR_FACTOR   # ~0.00345

TF_CAP      = {6: 0.55, 4: 0.35, 2: 0.25, 0: 0.0}
MIN_HOLD    = 4
BH_DECAY    = 0.95
BH_FORM     = 1.5
BH_COLLAPSE = 1.0

OUT_DIR = Path(__file__).parent / "backtest_output"
OUT_DIR.mkdir(exist_ok=True)


# ── BH State ──────────────────────────────────────────────────────────────────
class BHState:
    def __init__(self, cf):
        self.cf       = cf
        self.cf_scale = 1.0
        self.mass     = 0.0
        self.active   = False
        self.bh_dir   = 0
        self.ctl      = 0
        self.prices   = []

    def update(self, price):
        self.prices.append(float(price))
        if len(self.prices) > 25:
            self.prices.pop(0)
        if len(self.prices) < 2:
            return
        prev = self.prices[-2]
        beta = abs(self.prices[-1] - prev) / (prev + 1e-9) / (self.cf * self.cf_scale + 1e-9)
        was  = self.active
        if beta < 1.0:
            self.ctl += 1
            sb = min(2.0, 1.0 + self.ctl * 0.1)
            self.mass = self.mass * 0.97 + 0.03 * sb
        else:
            self.ctl = 0
            self.mass *= BH_DECAY
        self.active = (self.mass > BH_FORM and self.ctl >= 3) if not was else \
                      (self.mass > BH_COLLAPSE and self.ctl >= 3)
        if not was and self.active:
            lb = min(20, len(self.prices) - 1)
            self.bh_dir = 1 if self.prices[-1] > self.prices[-1 - lb] else -1
        elif was and not self.active:
            self.bh_dir = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
def ema_s(s, n):  return s.ewm(span=n, adjust=False).mean()

def atr_s(hi, lo, cl, n=14):
    tr = pd.concat([(hi - lo), (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def bull_scale(close):
    e12  = ema_s(close, 12);  e26 = ema_s(close, 26)
    e50  = ema_s(close, 50);  e200 = ema_s(close, 200)
    return ((close > e200) & (e12 > e26) & (e26 > e50)).map({True: 3.0, False: 1.0})


# ── Download ──────────────────────────────────────────────────────────────────
def download_data():
    """
    Hourly: prefer Twelve Data cache (tools/data_cache/*_1h.csv) for full history.
    Falls back to yfinance last-730-days if cache not found.
    Daily: always from yfinance (full history, no API limit).
    """
    try:
        from download_twelvedata import load_cached, CACHE_DIR
        use_cache = True
    except ImportError:
        use_cache = False

    daily, hourly = {}, {}
    for sym, cfg in INSTRUMENTS.items():
        tk = cfg["ticker"]
        print(f"  {tk} daily...", end=" ", flush=True)
        d = yf.download(tk, start=START_DATE, interval="1d", progress=False, auto_adjust=True)
        d.columns = d.columns.get_level_values(0)
        daily[sym] = d
        print(f"{len(d)} bars")

        if use_cache:
            h = load_cached(sym)
            if not h.empty:
                print(f"  {tk} hourly (cache)... {len(h)} bars "
                      f"[{h.index.min().date()} to {h.index.max().date()}]")
                hourly[sym] = h
                continue

        # Fallback: yfinance last 730 days
        print(f"  {tk} hourly (yfinance fallback)...", end=" ", flush=True)
        h = yf.download(tk, period="730d", interval="1h", progress=False, auto_adjust=True)
        h.columns = h.columns.get_level_values(0)
        h.index = h.index.tz_convert("America/New_York")
        h = h.between_time("09:30", "16:00")
        if h.index.tz is not None:
            h.index = h.index.tz_localize(None)
        hourly[sym] = h
        print(f"{len(h)} bars")

    return daily, hourly


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(daily, hourly):
    syms = list(INSTRUMENTS.keys())

    # Pre-compute indicators
    d_atr   = {s: atr_s(daily[s]["High"], daily[s]["Low"], daily[s]["Close"]) for s in syms}
    d_scale = {s: bull_scale(daily[s]["Close"]) for s in syms}
    h_atr   = {}
    for s in syms:
        h = hourly[s]
        if not h.empty:
            h_atr[s] = atr_s(h["High"], h["Low"], h["Close"])

    d_bh = {s: BHState(INSTRUMENTS[s]["cf_1d"]) for s in syms}
    h_bh = {s: BHState(INSTRUMENTS[s]["cf_1h"]) for s in syms}

    all_days     = daily["ES"].index[daily["ES"].index >= pd.Timestamp(START_DATE)]
    hourly_dates = {s: set(hourly[s].index.date) for s in syms if not hourly[s].empty}

    # Position state
    # dollar_pos: actual dollar value of position (fixed at entry, changes on resize)
    dollar_pos   = {s: 0.0 for s in syms}   # dollars in position
    entry_price  = {s: None for s in syms}   # price when position was last set
    last_frac    = {s: 0.0 for s in syms}    # fraction target
    bars_held    = {s: 0 for s in syms}
    pos_floor    = {s: 0.0 for s in syms}

    equity       = STARTING_EQUITY
    peak         = STARTING_EQUITY
    equity_curve = []   # (date, equity)
    trades       = []   # completed round trips

    WARMUP = 200

    for day_idx, day in enumerate(all_days):
        day_date = day.date()

        # Update daily BH
        for s in syms:
            if day not in daily[s].index: continue
            d_bh[s].cf_scale = float(d_scale[s].get(day, 1.0))
            d_bh[s].update(daily[s].loc[day, "Close"])

        if day_idx < WARMUP:
            equity_curve.append((day_date, equity))
            continue

        use_hourly = (not hourly["ES"].empty and day_date in hourly_dates.get("ES", set()))

        if use_hourly:
            h_bars = {s: hourly[s][hourly[s].index.date == day_date] for s in syms}
            bar_times = h_bars["ES"].index
        else:
            bar_times = [day]

        for bar_time in bar_times:
            # Update hourly BH
            if use_hourly:
                for s in syms:
                    b = h_bars[s]
                    if bar_time not in b.index: continue
                    h_bh[s].cf_scale = float(d_scale[s].get(day, 1.0))
                    h_bh[s].update(b.loc[bar_time, "Close"])

            # Current prices
            curr_price = {}
            for s in syms:
                if use_hourly and bar_time in h_bars[s].index:
                    curr_price[s] = float(h_bars[s].loc[bar_time, "Close"])
                elif day in daily[s].index:
                    curr_price[s] = float(daily[s].loc[day, "Close"])
                else:
                    curr_price[s] = entry_price[s] if entry_price[s] else 1.0

            # ── Mark-to-market: update equity from open positions ─────────────
            mtm_pnl = 0.0
            for s in syms:
                if dollar_pos[s] != 0.0 and entry_price[s] is not None and entry_price[s] > 0:
                    ret = (curr_price[s] - entry_price[s]) / entry_price[s]
                    mtm_pnl += dollar_pos[s] * ret

            equity_live = equity + mtm_pnl
            if equity_live > peak:
                peak = equity_live

            # ── Compute targets ───────────────────────────────────────────────
            tail_frac = min(TAIL_FIXED_CAPITAL, equity_live) / equity_live
            raw_targets = {}

            for s in syms:
                tf = (4 if d_bh[s].active else 0) + (2 if (use_hourly and h_bh[s].active) else 0)
                ceiling = TF_CAP.get(tf, 0.0)

                if tf == 2 and abs(last_frac[s]) < 0.02:
                    ceiling = 0.0

                if ceiling == 0.0:
                    raw_targets[s] = 0.0
                    continue

                direction = 0
                if d_bh[s].active and d_bh[s].bh_dir != 0:
                    direction = d_bh[s].bh_dir
                elif use_hourly and h_bh[s].active and h_bh[s].bh_dir != 0:
                    direction = h_bh[s].bh_dir
                elif d_bh[s].active and len(d_bh[s].prices) >= 5:
                    direction = 1 if d_bh[s].prices[-1] > d_bh[s].prices[-5] else -1

                if direction == 0:
                    raw_targets[s] = 0.0
                    continue

                # Vol sizing
                if use_hourly and s in h_atr and bar_time in h_atr[s].index:
                    vol_pct = float(h_atr[s].loc[bar_time]) / (curr_price[s] + 1e-9) * math.sqrt(6.5)
                elif day in d_atr[s].index:
                    vol_pct = float(d_atr[s].loc[day]) / (curr_price[s] + 1e-9)
                else:
                    vol_pct = 0.01

                cap = min(PER_INST_RISK / (vol_pct + 1e-9), ceiling)
                raw_targets[s] = cap * direction

            # pos_floor
            for s in syms:
                tgt = raw_targets.get(s, 0.0)
                tf  = (4 if d_bh[s].active else 0) + (2 if (use_hourly and h_bh[s].active) else 0)
                if (tf >= 6 and abs(tgt) > 0.15 and not np.isclose(tgt, 0.0) and h_bh[s].ctl >= 5):
                    pos_floor[s] = max(pos_floor[s], 0.70 * abs(tgt))
                if pos_floor[s] > 0.0 and tf >= 4 and not np.isclose(last_frac[s], 0.0):
                    raw_targets[s] = float(np.sign(last_frac[s]) * max(abs(tgt), pos_floor[s]))
                    pos_floor[s] *= 0.95
                if tf < 4 or np.isclose(tgt, 0.0): pos_floor[s] = 0.0
                if not d_bh[s].active and not (use_hourly and h_bh[s].active): pos_floor[s] = 0.0

            # Normalize + tail_frac
            total_exp = sum(abs(v) for v in raw_targets.values())
            scale     = 1.0 / total_exp if total_exp > 1.0 else 1.0

            for s in syms:
                tgt   = raw_targets.get(s, 0.0)
                final = tgt * scale * tail_frac

                # Min hold reversal gate
                if (not np.isclose(last_frac[s], 0.0) and not np.isclose(final, 0.0) and
                        np.sign(final) != np.sign(last_frac[s]) and bars_held[s] < MIN_HOLD):
                    final = last_frac[s]

                if abs(final - last_frac[s]) > 0.02:
                    # Realize current position P&L
                    if dollar_pos[s] != 0.0 and entry_price[s] is not None:
                        ret  = (curr_price[s] - entry_price[s]) / entry_price[s]
                        pnl  = dollar_pos[s] * ret
                        equity += pnl
                        trades.append({
                            "exit_time":    bar_time if use_hourly else day_date,
                            "sym":          s,
                            "direction":    "Long" if dollar_pos[s] > 0 else "Short",
                            "entry_price":  entry_price[s],
                            "exit_price":   curr_price[s],
                            "dollar_pos":   dollar_pos[s],
                            "pnl":          pnl,
                        })

                    # Open new position
                    if np.isclose(final, 0.0):
                        dollar_pos[s]  = 0.0
                        entry_price[s] = None
                        bars_held[s]   = 0
                    else:
                        dollar_pos[s]  = final * equity  # fixed dollar amount at entry
                        entry_price[s] = curr_price[s]
                        if np.sign(final) != np.sign(last_frac[s]):
                            bars_held[s] = 0

                    last_frac[s] = final

                if abs(last_frac[s]) > 0.02:
                    bars_held[s] += 1

        # End-of-day equity snapshot (mark to market)
        eod_pnl = sum(
            dollar_pos[s] * (curr_price[s] - entry_price[s]) / entry_price[s]
            for s in syms
            if dollar_pos[s] != 0.0 and entry_price[s] and entry_price[s] > 0
        )
        equity_curve.append((day_date, max(0.0, equity + eod_pnl)))

    return equity_curve, trades, peak


# ── Stats ──────────────────────────────────────────────────────────────────────
def print_stats(equity_curve, trades, peak):
    dates  = [e[0] for e in equity_curve]
    values = np.array([e[1] for e in equity_curve])

    final      = values[-1]
    total_ret  = (final - STARTING_EQUITY) / STARTING_EQUITY
    years      = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    cagr       = (final / STARTING_EQUITY) ** (1 / years) - 1 if years > 0 and final > 0 else 0
    running_pk = np.maximum.accumulate(values)
    dd_series  = (values - running_pk) / running_pk
    max_dd     = dd_series.min()
    rets       = pd.Series(values).pct_change().dropna()
    sharpe     = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(252)

    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / (abs(sum(losses)) + 1e-9) if losses else float("inf")

    print("\n" + "=" * 60)
    print("  LOCAL BH BACKTEST  (Geeky params, SPY/QQQ/DIA proxies)")
    print("=" * 60)
    print(f"  Period:        {dates[0]}  to  {dates[-1]}")
    print(f"  Start equity:  ${STARTING_EQUITY:>13,.0f}")
    print(f"  Peak equity:   ${peak:>13,.0f}")
    print(f"  Final equity:  ${final:>13,.0f}")
    print(f"  Total return:  {total_ret:>13.1%}")
    print(f"  CAGR:          {cagr:>13.1%}")
    print(f"  Max drawdown:  {max_dd:>13.1%}")
    print(f"  Sharpe:        {sharpe:>13.2f}")
    print(f"  Trades:        {len(trades):>13,}")
    print(f"  Win rate:      {len(wins)/(len(trades)+1e-9):>13.1%}")
    print(f"  Profit factor: {pf:>13.2f}")
    if wins:   print(f"  Avg win:       ${np.mean(wins):>12,.0f}")
    if losses: print(f"  Avg loss:      ${np.mean(losses):>12,.0f}")
    print("=" * 60)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        for spine in ax.spines.values(): spine.set_edgecolor("#333")

    fig.suptitle(
        f"BH Engine Local Backtest   CAGR {cagr:.1%}   MaxDD {max_dd:.1%}   Sharpe {sharpe:.2f}",
        color="white", fontsize=12
    )
    ax1.plot(dates, values, color="#00d4aa", linewidth=1.0)
    ax1.fill_between(dates, values, alpha=0.12, color="#00d4aa")
    ax1.axhline(STARTING_EQUITY, color="#555", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Equity ($)", color="white")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax1.set_yscale("log")
    ax1.grid(alpha=0.15, color="white")

    ax2.fill_between(dates, dd_series * 100, color="#ff4444", alpha=0.7)
    ax2.set_ylabel("Drawdown %", color="white")
    ax2.set_xlabel("Date", color="white")
    ax2.grid(alpha=0.15, color="white")

    plt.tight_layout()
    png = OUT_DIR / "equity_curve.png"
    plt.savefig(png, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Chart: {png}")

    # ── Trade CSV ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(trades)
    csv_path = OUT_DIR / "trades.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Trades: {csv_path}")

    # Top trades by P&L
    df_s = df.sort_values("pnl", ascending=False)
    print(f"\n  TOP 15 TRADES:")
    print(f"  {'Date':<12}  {'Sym':3}  {'Dir':5}  {'P&L':>12}  {'Entry':>8}  {'Exit':>8}")
    print("  " + "-" * 58)
    for _, r in df_s.head(15).iterrows():
        dt = str(r["exit_time"])[:10]
        print(f"  {dt:<12}  {r['sym']:3}  {r['direction']:5}  "
              f"${r['pnl']:>11,.0f}  {r['entry_price']:>8.2f}  {r['exit_price']:>8.2f}")

    # Worst trades
    print(f"\n  WORST 10 TRADES:")
    print(f"  {'Date':<12}  {'Sym':3}  {'Dir':5}  {'P&L':>12}")
    print("  " + "-" * 40)
    for _, r in df_s.tail(10).iterrows():
        dt = str(r["exit_time"])[:10]
        print(f"  {dt:<12}  {r['sym']:3}  {r['direction']:5}  ${r['pnl']:>11,.0f}")

    # Annual P&L
    df["year"] = pd.to_datetime(df["exit_time"].astype(str).str[:10]).dt.year
    annual = df.groupby("year")["pnl"].sum()
    print(f"\n  ANNUAL P&L:")
    for yr, pnl in annual.items():
        bar = "#" * int(abs(pnl) / 20000)
        sign = "+" if pnl >= 0 else ""
        print(f"  {yr}:  {sign}${pnl:>10,.0f}  {bar}")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Downloading data...")
    daily, hourly = download_data()
    print("\nRunning backtest (2010-present, Geeky params)...")
    eq, tr, pk = run_backtest(daily, hourly)
    print_stats(eq, tr, pk)
