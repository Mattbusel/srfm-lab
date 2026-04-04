"""
research/regime_lab/stress.py
==============================
Historical stress-scenario library and stress-testing framework.

Classes
-------
StressScenario   — dataclass describing one historical stress event
StressResult     — outcome of applying a scenario to a trade set
WorstCaseResult  — aggregated worst-case analysis
StressTester     — orchestrates scenario execution

Functions
---------
plot_stress_results(results, save_path)
stress_report(results) -> pd.DataFrame
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime constants
# ---------------------------------------------------------------------------
BULL     = "BULL"
BEAR     = "BEAR"
SIDEWAYS = "SIDEWAYS"
HIGH_VOL = "HIGH_VOL"


# ===========================================================================
# 1. Data structures
# ===========================================================================

@dataclass
class StressScenario:
    """Description of a historical or synthetic stress event."""
    name:               str
    description:        str
    start_date:         str                       # ISO-8601 e.g. "2020-02-20"
    end_date:           str
    peak_drawdown:      float                     # max observed drawdown (negative, e.g. -0.35)
    duration_days:      int                       # calendar days
    asset_class:        str                       # "equity", "crypto", "rates", "mixed"
    regime:             str = HIGH_VOL
    return_multipliers: Dict[str, float] = field(default_factory=dict)
    # return_multipliers: regime → scaling factor for strategy returns during stress
    # e.g. {"BULL": 1.5, "BEAR": 0.5} means BULL-regime trades earn 1.5× normal
    vol_multiplier:     float = 2.0               # vol expansion factor
    correlation_shift:  float = 0.3              # increase in cross-asset correlation
    notes:              str = ""

    def daily_drawdown_rate(self) -> float:
        """Average daily log-return implied by the peak drawdown."""
        if self.duration_days <= 0:
            return 0.0
        return math.log(1 + self.peak_drawdown) / self.duration_days

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name":           self.name,
            "description":    self.description,
            "start_date":     self.start_date,
            "end_date":       self.end_date,
            "peak_drawdown":  self.peak_drawdown,
            "duration_days":  self.duration_days,
            "asset_class":    self.asset_class,
            "regime":         self.regime,
            "vol_multiplier": self.vol_multiplier,
        }


@dataclass
class StressResult:
    scenario:           StressScenario
    gross_pnl_impact:   float      # estimated total P&L impact ($)
    pct_equity_impact:  float      # as fraction of starting equity
    regime_breakdown:   Dict[str, float] = field(default_factory=dict)
    # regime → estimated P&L impact for trades in that regime
    trade_count:        int   = 0
    worst_trade_pnl:    float = 0.0
    best_trade_pnl:     float = 0.0
    blowup:             bool  = False   # equity < 10% of start
    notes:              str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario":          self.scenario.name,
            "gross_pnl_impact":  round(self.gross_pnl_impact, 2),
            "pct_equity_impact": round(self.pct_equity_impact * 100, 2),
            "trade_count":       self.trade_count,
            "worst_trade_pnl":   round(self.worst_trade_pnl, 2),
            "best_trade_pnl":    round(self.best_trade_pnl, 2),
            "blowup":            self.blowup,
        }


@dataclass
class WorstCaseResult:
    percentile:        float
    worst_pnl:         float
    worst_scenario:    Optional[str]
    var_estimate:      float    # Value-at-Risk
    cvar_estimate:     float    # Conditional VaR
    blowup_scenarios:  List[str] = field(default_factory=list)
    summary_df:        Optional[pd.DataFrame] = None


# ===========================================================================
# 2. Historical scenario catalogue
# ===========================================================================

def _build_scenario_catalogue() -> List[StressScenario]:
    """
    Returns the 20 canonical historical stress scenarios.
    """
    return [
        # ------------------------------------------------------------------ #
        # 1. COVID Crash
        # ------------------------------------------------------------------ #
        StressScenario(
            name="COVID_CRASH_2020",
            description="COVID-19 pandemic shock — fastest -34% S&P 500 decline on record",
            start_date="2020-02-20",
            end_date="2020-03-23",
            peak_drawdown=-0.340,
            duration_days=32,
            asset_class="equity",
            regime=HIGH_VOL,
            return_multipliers={BULL: -3.0, BEAR: 2.0, SIDEWAYS: -1.5, HIGH_VOL: 1.5},
            vol_multiplier=4.5,
            correlation_shift=0.5,
            notes="VIX peaked at 82.69. Fastest bear market in history.",
        ),
        # ------------------------------------------------------------------ #
        # 2. 2022 Crypto Winter
        # ------------------------------------------------------------------ #
        StressScenario(
            name="CRYPTO_WINTER_2022",
            description="2022 crypto bear market: BTC -75% from ATH over ~12 months",
            start_date="2021-11-10",
            end_date="2022-11-21",
            peak_drawdown=-0.770,
            duration_days=376,
            asset_class="crypto",
            regime=BEAR,
            return_multipliers={BULL: -4.0, BEAR: 3.0, SIDEWAYS: -2.0, HIGH_VOL: 2.0},
            vol_multiplier=2.5,
            correlation_shift=0.4,
            notes="BTC from $69K ATH to $15.5K. Multiple contagion events.",
        ),
        # ------------------------------------------------------------------ #
        # 3. 2022 Fed Tightening / QQQ Bear
        # ------------------------------------------------------------------ #
        StressScenario(
            name="FED_TIGHTENING_2022",
            description="Fed rate hikes trigger tech selloff: QQQ -33%, NDX -35%",
            start_date="2022-01-03",
            end_date="2022-10-14",
            peak_drawdown=-0.330,
            duration_days=284,
            asset_class="equity",
            regime=BEAR,
            return_multipliers={BULL: -2.5, BEAR: 2.5, SIDEWAYS: -1.0, HIGH_VOL: 1.5},
            vol_multiplier=2.0,
            correlation_shift=0.3,
            notes="Fastest rate-hiking cycle in 40 years. 10yr from 1.5% to 4.2%.",
        ),
        # ------------------------------------------------------------------ #
        # 4. FTX Collapse
        # ------------------------------------------------------------------ #
        StressScenario(
            name="FTX_COLLAPSE_NOV2022",
            description="FTX exchange collapse: BTC/ETH -30% in <1 week, contagion across crypto",
            start_date="2022-11-07",
            end_date="2022-11-14",
            peak_drawdown=-0.320,
            duration_days=7,
            asset_class="crypto",
            regime=HIGH_VOL,
            return_multipliers={BULL: -5.0, BEAR: 3.5, SIDEWAYS: -3.0, HIGH_VOL: 2.5},
            vol_multiplier=6.0,
            correlation_shift=0.6,
            notes="FTX/Alameda insolvency. Second-largest crypto exchange failure.",
        ),
        # ------------------------------------------------------------------ #
        # 5. SVB Collapse / Bank Stress
        # ------------------------------------------------------------------ #
        StressScenario(
            name="SVB_BANK_STRESS_2023",
            description="Silicon Valley Bank failure triggers regional bank contagion",
            start_date="2023-03-08",
            end_date="2023-03-17",
            peak_drawdown=-0.200,
            duration_days=9,
            asset_class="equity",
            regime=HIGH_VOL,
            return_multipliers={BULL: -2.0, BEAR: 2.0, SIDEWAYS: -1.5, HIGH_VOL: 1.5},
            vol_multiplier=3.5,
            correlation_shift=0.4,
            notes="KRE (regional bank ETF) -28%. FDIC seized SVB, Signature Bank.",
        ),
        # ------------------------------------------------------------------ #
        # 6. May 2021 Crypto Crash
        # ------------------------------------------------------------------ #
        StressScenario(
            name="CRYPTO_CRASH_MAY2021",
            description="BTC -53% in 3 weeks: Musk Tesla tweet + China mining ban fears",
            start_date="2021-05-12",
            end_date="2021-05-19",
            peak_drawdown=-0.530,
            duration_days=21,
            asset_class="crypto",
            regime=HIGH_VOL,
            return_multipliers={BULL: -4.0, BEAR: 3.0, SIDEWAYS: -2.5, HIGH_VOL: 2.0},
            vol_multiplier=5.0,
            correlation_shift=0.5,
            notes="BTC $63K→$30K. Leveraged long liquidations cascade.",
        ),
        # ------------------------------------------------------------------ #
        # 7. China Mining Ban 2021
        # ------------------------------------------------------------------ #
        StressScenario(
            name="CHINA_MINING_BAN_2021",
            description="China bans crypto mining/transactions: BTC -25% in 2 days",
            start_date="2021-06-18",
            end_date="2021-06-22",
            peak_drawdown=-0.255,
            duration_days=4,
            asset_class="crypto",
            regime=HIGH_VOL,
            return_multipliers={BULL: -3.5, BEAR: 2.5, SIDEWAYS: -2.0, HIGH_VOL: 2.0},
            vol_multiplier=5.5,
            correlation_shift=0.45,
            notes="BTC hashrate drops 50%. Forced miner selling.",
        ),
        # ------------------------------------------------------------------ #
        # 8. Flash Crash May 2010
        # ------------------------------------------------------------------ #
        StressScenario(
            name="FLASH_CRASH_MAY2010",
            description="May 6 2010 Flash Crash: DJIA -9% intraday, recovered same day",
            start_date="2010-05-06",
            end_date="2010-05-06",
            peak_drawdown=-0.092,
            duration_days=1,
            asset_class="equity",
            regime=HIGH_VOL,
            return_multipliers={BULL: -2.0, BEAR: 2.0, SIDEWAYS: -1.0, HIGH_VOL: 2.5},
            vol_multiplier=8.0,
            correlation_shift=0.7,
            notes="Algorithmic trading cascade. 1 trillion $ vanished briefly.",
        ),
        # ------------------------------------------------------------------ #
        # 9. European Sovereign Debt Crisis 2011-12
        # ------------------------------------------------------------------ #
        StressScenario(
            name="EUR_DEBT_CRISIS_2011",
            description="Eurozone sovereign debt crisis: periphery yields spike, equities -25%",
            start_date="2011-07-22",
            end_date="2012-07-26",
            peak_drawdown=-0.250,
            duration_days=369,
            asset_class="mixed",
            regime=HIGH_VOL,
            return_multipliers={BULL: -2.0, BEAR: 2.5, SIDEWAYS: -1.0, HIGH_VOL: 1.5},
            vol_multiplier=2.5,
            correlation_shift=0.35,
            notes="PIIGS crisis. ECB 'whatever it takes' speech July 26 2012 ended it.",
        ),
        # ------------------------------------------------------------------ #
        # 10. Global Financial Crisis 2008-09
        # ------------------------------------------------------------------ #
        StressScenario(
            name="GFC_2008_2009",
            description="Global Financial Crisis: S&P -57% peak to trough",
            start_date="2007-10-09",
            end_date="2009-03-09",
            peak_drawdown=-0.569,
            duration_days=517,
            asset_class="equity",
            regime=BEAR,
            return_multipliers={BULL: -5.0, BEAR: 4.0, SIDEWAYS: -2.0, HIGH_VOL: 3.0},
            vol_multiplier=4.0,
            correlation_shift=0.6,
            notes="Lehman failure Sep 2008. VIX hit 89.53.",
        ),
        # ------------------------------------------------------------------ #
        # 11. Dot-com Bust 2000-02
        # ------------------------------------------------------------------ #
        StressScenario(
            name="DOTCOM_BUST_2000_2002",
            description="Dot-com crash: NDX -83%, S&P -49% over ~2.5 years",
            start_date="2000-03-10",
            end_date="2002-10-09",
            peak_drawdown=-0.830,
            duration_days=943,
            asset_class="equity",
            regime=BEAR,
            return_multipliers={BULL: -6.0, BEAR: 5.0, SIDEWAYS: -2.5, HIGH_VOL: 2.5},
            vol_multiplier=2.5,
            correlation_shift=0.4,
            notes="Internet bubble deflation. Many tech stocks lost 90-100%.",
        ),
        # ------------------------------------------------------------------ #
        # 12. 9/11 Market Aftermath
        # ------------------------------------------------------------------ #
        StressScenario(
            name="NINE_ELEVEN_2001",
            description="9/11 attacks: markets closed 4 days, S&P -12% on reopening week",
            start_date="2001-09-17",
            end_date="2001-09-21",
            peak_drawdown=-0.118,
            duration_days=4,
            asset_class="equity",
            regime=HIGH_VOL,
            return_multipliers={BULL: -3.0, BEAR: 2.5, SIDEWAYS: -1.5, HIGH_VOL: 2.0},
            vol_multiplier=4.5,
            correlation_shift=0.55,
            notes="Markets closed Sep 11-14. Largest one-week drop since 1933.",
        ),
        # ------------------------------------------------------------------ #
        # 13. Asian Financial Crisis 1997
        # ------------------------------------------------------------------ #
        StressScenario(
            name="ASIAN_CRISIS_1997",
            description="Asian financial crisis: EM currencies collapse, Hang Seng -63%",
            start_date="1997-07-02",
            end_date="1998-01-12",
            peak_drawdown=-0.630,
            duration_days=194,
            asset_class="equity",
            regime=BEAR,
            return_multipliers={BULL: -3.0, BEAR: 3.0, SIDEWAYS: -1.5, HIGH_VOL: 2.0},
            vol_multiplier=3.0,
            correlation_shift=0.45,
            notes="Thai baht devaluation sparked contagion across SE Asia.",
        ),
        # ------------------------------------------------------------------ #
        # 14. Black Monday 1987
        # ------------------------------------------------------------------ #
        StressScenario(
            name="BLACK_MONDAY_1987",
            description="Black Monday Oct 19 1987: DJIA -22.6% in single session",
            start_date="1987-10-19",
            end_date="1987-10-19",
            peak_drawdown=-0.226,
            duration_days=1,
            asset_class="equity",
            regime=HIGH_VOL,
            return_multipliers={BULL: -5.0, BEAR: 4.0, SIDEWAYS: -2.5, HIGH_VOL: 3.0},
            vol_multiplier=10.0,
            correlation_shift=0.8,
            notes="Portfolio insurance / program trading cascade. Largest single-day % drop.",
        ),
        # ------------------------------------------------------------------ #
        # 15. LTCM Collapse 1998
        # ------------------------------------------------------------------ #
        StressScenario(
            name="LTCM_COLLAPSE_1998",
            description="LTCM near-failure after Russian default: spreads blow out, -21% S&P",
            start_date="1998-08-17",
            end_date="1998-10-08",
            peak_drawdown=-0.210,
            duration_days=51,
            asset_class="mixed",
            regime=HIGH_VOL,
            return_multipliers={BULL: -2.5, BEAR: 3.0, SIDEWAYS: -1.5, HIGH_VOL: 2.0},
            vol_multiplier=3.5,
            correlation_shift=0.5,
            notes="Russia default Aug 17. Fed coordinated LTCM bailout Sep 23.",
        ),
        # ------------------------------------------------------------------ #
        # 16. 1994 Bond Massacre
        # ------------------------------------------------------------------ #
        StressScenario(
            name="BOND_MASSACRE_1994",
            description="Fed surprise rate hikes cause global bond market selloff, -3.5% SPX",
            start_date="1994-01-31",
            end_date="1994-11-15",
            peak_drawdown=-0.086,
            duration_days=288,
            asset_class="rates",
            regime=BEAR,
            return_multipliers={BULL: -1.5, BEAR: 2.0, SIDEWAYS: -1.0, HIGH_VOL: 1.5},
            vol_multiplier=2.0,
            correlation_shift=0.3,
            notes="10yr yield +250bps. Orange County bankruptcy. Mexican peso crisis.",
        ),
        # ------------------------------------------------------------------ #
        # 17. Bitcoin Crash 2013 (Mt. Gox era)
        # ------------------------------------------------------------------ #
        StressScenario(
            name="BTC_CRASH_2013",
            description="Bitcoin -80% from $1,163 to $152 after Mt. Gox problems emerge",
            start_date="2013-12-04",
            end_date="2015-01-14",
            peak_drawdown=-0.869,
            duration_days=406,
            asset_class="crypto",
            regime=BEAR,
            return_multipliers={BULL: -6.0, BEAR: 5.0, SIDEWAYS: -3.0, HIGH_VOL: 2.5},
            vol_multiplier=3.0,
            correlation_shift=0.3,
            notes="Mt. Gox halted withdrawals Jan 2014, declared bankruptcy Feb 2014.",
        ),
        # ------------------------------------------------------------------ #
        # 18. Crypto Winter 2018
        # ------------------------------------------------------------------ #
        StressScenario(
            name="CRYPTO_WINTER_2018",
            description="2018 crypto crash: BTC -84%, altcoins -95%+",
            start_date="2017-12-17",
            end_date="2018-12-15",
            peak_drawdown=-0.840,
            duration_days=363,
            asset_class="crypto",
            regime=BEAR,
            return_multipliers={BULL: -5.0, BEAR: 4.0, SIDEWAYS: -2.5, HIGH_VOL: 2.0},
            vol_multiplier=2.5,
            correlation_shift=0.4,
            notes="ICO bubble burst. Bitcoin from $19,783 to $3,122.",
        ),
        # ------------------------------------------------------------------ #
        # 19. Terra/Luna Collapse May 2022
        # ------------------------------------------------------------------ #
        StressScenario(
            name="TERRA_LUNA_MAY2022",
            description="Terra/Luna de-peg: LUNA -99.9%, UST lost peg, BTC -30% in 1 week",
            start_date="2022-05-07",
            end_date="2022-05-13",
            peak_drawdown=-0.650,
            duration_days=6,
            asset_class="crypto",
            regime=HIGH_VOL,
            return_multipliers={BULL: -5.0, BEAR: 4.0, SIDEWAYS: -3.0, HIGH_VOL: 2.5},
            vol_multiplier=7.0,
            correlation_shift=0.65,
            notes="$40B market cap evaporated. Stablecoin de-peg contagion to BTC/ETH.",
        ),
        # ------------------------------------------------------------------ #
        # 20. Interest Rate Shock 2023
        # ------------------------------------------------------------------ #
        StressScenario(
            name="RATE_SHOCK_2023",
            description="Higher-for-longer rates: 10yr hits 5%, equities -10%, bonds -15%",
            start_date="2023-07-19",
            end_date="2023-10-23",
            peak_drawdown=-0.108,
            duration_days=96,
            asset_class="mixed",
            regime=HIGH_VOL,
            return_multipliers={BULL: -2.0, BEAR: 2.5, SIDEWAYS: -1.0, HIGH_VOL: 1.8},
            vol_multiplier=2.5,
            correlation_shift=0.35,
            notes="10yr Treasury yield reached 5.02% (highest since 2007).",
        ),
    ]


# Keep module-level catalogue accessible
SCENARIO_CATALOGUE: List[StressScenario] = _build_scenario_catalogue()
SCENARIO_BY_NAME:   Dict[str, StressScenario] = {s.name: s for s in SCENARIO_CATALOGUE}


# ===========================================================================
# 3. Trade helpers
# ===========================================================================

def _extract_trades(trades: Any) -> List[Dict[str, Any]]:
    """
    Normalise trades to a list of dicts.
    Accepts: list[dict], pd.DataFrame, or anything with .to_dict(orient='records').
    """
    if isinstance(trades, pd.DataFrame):
        return trades.to_dict(orient="records")
    if isinstance(trades, list):
        return [dict(t) if not isinstance(t, dict) else t for t in trades]
    if hasattr(trades, "to_dict"):
        return trades.to_dict(orient="records")
    return []


def _trade_regime(trade: Dict[str, Any]) -> str:
    """Extract regime from a trade record."""
    for key in ("regime", "entry_regime", "market_regime"):
        if key in trade:
            r = str(trade[key]).upper()
            if r in (BULL, BEAR, SIDEWAYS, HIGH_VOL):
                return r
    return SIDEWAYS


def _trade_pnl(trade: Dict[str, Any]) -> float:
    """Extract P&L from a trade record."""
    for key in ("pnl", "profit_loss", "realized_pnl", "net_pnl"):
        if key in trade:
            try:
                return float(trade[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _trade_entry_value(trade: Dict[str, Any]) -> float:
    """Estimate trade entry value for scaling."""
    for key in ("dollar_pos", "entry_value", "notional", "position_value"):
        if key in trade and trade[key]:
            try:
                return float(trade[key])
            except (TypeError, ValueError):
                pass
    entry_price = float(trade.get("entry_price", 0.0) or 0.0)
    qty         = float(trade.get("quantity", 1.0) or 1.0)
    if entry_price > 0:
        return entry_price * qty
    return 1_000.0   # default fallback


# ===========================================================================
# 4. StressTester
# ===========================================================================

class StressTester:
    """
    Apply historical stress scenarios to a set of trades.

    Parameters
    ----------
    starting_equity   : float — account equity to normalise P&L impacts
    custom_scenarios  : optional list of additional StressScenario objects
    blowup_threshold  : fraction of starting_equity below which blowup = True
    """

    def __init__(self,
                 starting_equity: float = 1_000_000.0,
                 custom_scenarios: Optional[List[StressScenario]] = None,
                 blowup_threshold: float = 0.10):
        self.starting_equity  = starting_equity
        self.blowup_threshold = blowup_threshold

        self._scenarios: List[StressScenario] = SCENARIO_CATALOGUE.copy()
        if custom_scenarios:
            self._scenarios.extend(custom_scenarios)

    # ------------------------------------------------------------------ #
    # Core scenario runner
    # ------------------------------------------------------------------ #

    def scenario_pnl_impact(self, trades: Any,
                             scenario: StressScenario) -> float:
        """
        Estimate the aggregate P&L impact of a stress scenario on a set of trades.

        Method
        ------
        For each trade in *trades*:
          1. Determine its regime at entry.
          2. Look up scenario.return_multipliers[regime] as the P&L scaling factor.
             A multiplier of -3.0 means the trade earns 3× its absolute P&L in the
             *opposite* direction (i.e. a losing amplification).
             A positive multiplier > 1 means the trade benefits more than expected.
          3. Additionally apply a drawdown scaling proportional to abs(peak_drawdown).

        Parameters
        ----------
        trades   : trade records
        scenario : StressScenario

        Returns
        -------
        float — estimated change in P&L (can be negative = loss)
        """
        trade_list = _extract_trades(trades)
        if not trade_list:
            return 0.0

        total_impact = 0.0
        dd_scale     = abs(scenario.peak_drawdown)  # e.g. 0.34

        for trade in trade_list:
            regime      = _trade_regime(trade)
            base_pnl    = _trade_pnl(trade)
            entry_val   = _trade_entry_value(trade)

            mult = scenario.return_multipliers.get(regime, -dd_scale)

            # Stress impact = (multiplier - 1) × base_pnl
            # + direct drawdown hit on position value
            impact = (mult - 1.0) * base_pnl - dd_scale * entry_val * 0.01
            total_impact += impact

        return total_impact

    def run_scenario(self, trades: Any, scenario: StressScenario) -> StressResult:
        """
        Apply a stress scenario and produce a StressResult.

        Parameters
        ----------
        trades   : trade records (dict list, DataFrame, or compatible)
        scenario : StressScenario

        Returns
        -------
        StressResult
        """
        trade_list = _extract_trades(trades)
        gross_pnl  = self.scenario_pnl_impact(trade_list, scenario)

        pct_impact = gross_pnl / self.starting_equity if self.starting_equity else 0.0
        blowup     = (self.starting_equity + gross_pnl) < self.starting_equity * self.blowup_threshold

        # Per-regime breakdown
        regime_breakdown: Dict[str, float] = {r: 0.0 for r in (BULL, BEAR, SIDEWAYS, HIGH_VOL)}
        trade_pnls: List[float] = []

        dd_scale = abs(scenario.peak_drawdown)
        for trade in trade_list:
            regime    = _trade_regime(trade)
            base_pnl  = _trade_pnl(trade)
            entry_val = _trade_entry_value(trade)
            mult      = scenario.return_multipliers.get(regime, -dd_scale)
            impact    = (mult - 1.0) * base_pnl - dd_scale * entry_val * 0.01
            regime_breakdown[regime] += impact
            trade_pnls.append(impact)

        worst_trade = min(trade_pnls) if trade_pnls else 0.0
        best_trade  = max(trade_pnls) if trade_pnls else 0.0

        return StressResult(
            scenario=scenario,
            gross_pnl_impact=gross_pnl,
            pct_equity_impact=pct_impact,
            regime_breakdown=regime_breakdown,
            trade_count=len(trade_list),
            worst_trade_pnl=worst_trade,
            best_trade_pnl=best_trade,
            blowup=blowup,
        )

    def run_all_scenarios(self, trades: Any,
                          scenario_names: Optional[List[str]] = None
                          ) -> List[StressResult]:
        """
        Run all scenarios (or a named subset) against the trade set.

        Parameters
        ----------
        trades          : trade records
        scenario_names  : if provided, only run these scenarios by name

        Returns
        -------
        List[StressResult] — one per scenario, sorted by pct_equity_impact ascending
        """
        scenarios = self._scenarios
        if scenario_names is not None:
            name_set  = set(scenario_names)
            scenarios = [s for s in scenarios if s.name in name_set]

        results = [self.run_scenario(trades, sc) for sc in scenarios]
        results.sort(key=lambda r: r.pct_equity_impact)
        return results

    # ------------------------------------------------------------------ #
    # Worst-case analysis
    # ------------------------------------------------------------------ #

    def worst_case_analysis(self, trades: Any,
                             percentile: float = 5.0) -> WorstCaseResult:
        """
        Identify the worst-case stress outcomes.

        Parameters
        ----------
        trades     : trade records
        percentile : bottom X% of outcomes considered "worst case"

        Returns
        -------
        WorstCaseResult
        """
        results = self.run_all_scenarios(trades)
        if not results:
            return WorstCaseResult(percentile=percentile, worst_pnl=0.0,
                                   worst_scenario=None, var_estimate=0.0,
                                   cvar_estimate=0.0)

        pnls       = np.array([r.gross_pnl_impact for r in results])
        var_idx    = int(np.percentile(np.arange(len(pnls)),
                                        percentile / 100 * len(pnls)))
        sorted_pnl = np.sort(pnls)
        var_est    = float(-np.percentile(pnls, percentile))
        # CVaR: mean of bottom-percentile outcomes
        n_tail     = max(1, int(len(pnls) * percentile / 100))
        cvar_est   = float(-np.mean(sorted_pnl[:n_tail]))

        worst_pnl  = float(sorted_pnl[0])
        worst_sc   = results[int(np.argmin(pnls))].scenario.name

        blowup_scs = [r.scenario.name for r in results if r.blowup]

        df = stress_report(results)

        return WorstCaseResult(
            percentile=percentile,
            worst_pnl=worst_pnl,
            worst_scenario=worst_sc,
            var_estimate=var_est,
            cvar_estimate=cvar_est,
            blowup_scenarios=blowup_scs,
            summary_df=df,
        )

    # ------------------------------------------------------------------ #
    # Custom scenario builder
    # ------------------------------------------------------------------ #

    def add_custom_scenario(self, scenario: StressScenario) -> None:
        """Register an additional stress scenario."""
        self._scenarios.append(scenario)
        logger.debug("Added custom scenario: %s", scenario.name)

    def build_parametric_scenario(self,
                                   name: str,
                                   drawdown: float,
                                   duration_days: int = 30,
                                   vol_multiplier: float = 3.0,
                                   asset_class: str = "equity") -> StressScenario:
        """
        Build a synthetic parametric stress scenario from a drawdown magnitude.

        Parameters
        ----------
        name          : scenario identifier
        drawdown      : peak drawdown (negative, e.g. -0.20)
        duration_days : length of stress period
        vol_multiplier: volatility expansion factor
        asset_class   : 'equity', 'crypto', 'rates', or 'mixed'

        Returns
        -------
        StressScenario (also registered in this StressTester)
        """
        dd = float(drawdown)
        if dd > 0:
            dd = -dd  # ensure negative

        # Scale multipliers proportional to drawdown severity
        severity = min(abs(dd) / 0.30, 3.0)  # cap at 3×

        sc = StressScenario(
            name=name,
            description=f"Parametric scenario: {dd*100:.1f}% drawdown over {duration_days} days",
            start_date="2000-01-01",
            end_date="2000-01-01",
            peak_drawdown=dd,
            duration_days=duration_days,
            asset_class=asset_class,
            regime=HIGH_VOL if abs(dd) > 0.15 else BEAR,
            return_multipliers={
                BULL:     -severity * 1.5,
                BEAR:      severity * 1.0,
                SIDEWAYS: -severity * 0.8,
                HIGH_VOL:  severity * 0.5,
            },
            vol_multiplier=vol_multiplier,
        )
        self.add_custom_scenario(sc)
        return sc


# ===========================================================================
# 5. Reporting utilities
# ===========================================================================

def stress_report(results: List[StressResult]) -> pd.DataFrame:
    """
    Build a summary DataFrame from a list of StressResults.

    Columns: scenario, asset_class, duration_days, peak_drawdown,
             gross_pnl_impact, pct_equity_impact, trade_count, blowup

    Parameters
    ----------
    results : list of StressResult objects

    Returns
    -------
    pd.DataFrame sorted by pct_equity_impact ascending (worst first)
    """
    rows = []
    for r in results:
        sc = r.scenario
        rows.append({
            "scenario":          sc.name,
            "asset_class":       sc.asset_class,
            "peak_drawdown_pct": round(sc.peak_drawdown * 100, 1),
            "duration_days":     sc.duration_days,
            "vol_multiplier":    sc.vol_multiplier,
            "gross_pnl_impact":  round(r.gross_pnl_impact, 2),
            "pct_equity_impact": round(r.pct_equity_impact * 100, 2),
            "trade_count":       r.trade_count,
            "worst_trade_pnl":   round(r.worst_trade_pnl, 2),
            "best_trade_pnl":    round(r.best_trade_pnl, 2),
            "blowup":            r.blowup,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("pct_equity_impact", ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def plot_stress_results(results: List[StressResult],
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 8)) -> Any:
    """
    Horizontal bar chart of P&L impact per scenario.

    Parameters
    ----------
    results   : list of StressResult
    save_path : file path to save image (PNG). If None, shows interactively.
    figsize   : matplotlib figure size

    Returns
    -------
    matplotlib Figure object (or None if matplotlib not available)
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as mpatches  # type: ignore
    except ImportError:
        logger.warning("matplotlib not installed; cannot plot stress results.")
        return None

    df = stress_report(results)
    if df.empty:
        logger.warning("No stress results to plot.")
        return None

    # Sort worst→best (ascending pct_equity_impact)
    df = df.sort_values("pct_equity_impact", ascending=True).head(20)

    fig, ax = plt.subplots(figsize=figsize)
    colors  = ["#d32f2f" if v < 0 else "#388e3c" for v in df["pct_equity_impact"]]
    bars    = ax.barh(df["scenario"], df["pct_equity_impact"], color=colors, edgecolor="black",
                      linewidth=0.5)

    ax.axvline(x=0, color="black", linewidth=1.2)
    ax.set_xlabel("P&L Impact (% of Equity)", fontsize=12)
    ax.set_title("Stress Test Results — P&L Impact by Scenario", fontsize=14, fontweight="bold")
    ax.set_xlim(df["pct_equity_impact"].min() * 1.15, max(df["pct_equity_impact"].max() * 1.1, 5))
    ax.grid(axis="x", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, df["pct_equity_impact"]):
        x_pos = bar.get_width() + (0.3 if val >= 0 else -0.3)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%", va="center", ha="left" if val >= 0 else "right",
                fontsize=8)

    # Blowup markers
    blowup_names = {r.scenario.name for r in results if r.blowup}
    for i, (_, row) in enumerate(df.iterrows()):
        if row["scenario"] in blowup_names:
            ax.get_yticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_fontweight("bold")

    red_patch   = mpatches.Patch(color="#d32f2f", label="Loss")
    green_patch = mpatches.Patch(color="#388e3c", label="Gain")
    ax.legend(handles=[red_patch, green_patch], loc="lower right")

    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Stress plot saved to %s", save_path)

    return fig


def scenario_correlation_matrix(results: List[StressResult]) -> pd.DataFrame:
    """
    Build a (scenarios × scenarios) correlation matrix based on
    the per-regime P&L breakdown vectors.

    Useful for identifying which scenarios tend to co-occur.

    Returns
    -------
    pd.DataFrame — correlation matrix
    """
    regime_keys = [BULL, BEAR, SIDEWAYS, HIGH_VOL]
    names   = [r.scenario.name for r in results]
    vectors = np.array([[r.regime_breakdown.get(rk, 0.0) for rk in regime_keys]
                         for r in results])

    if vectors.shape[0] < 2:
        return pd.DataFrame()

    corr = np.corrcoef(vectors)
    return pd.DataFrame(corr, index=names, columns=names)


def find_worst_n_scenarios(results: List[StressResult], n: int = 5) -> List[StressResult]:
    """Return the n worst scenarios by P&L impact."""
    return sorted(results, key=lambda r: r.pct_equity_impact)[:n]


def scenario_by_asset_class(scenarios: Optional[List[StressScenario]] = None,
                              asset_class: str = "crypto") -> List[StressScenario]:
    """Filter the catalogue by asset class."""
    catalogue = scenarios or SCENARIO_CATALOGUE
    return [s for s in catalogue if s.asset_class == asset_class]
