"""
Macro scenario generation engine — models broad economic/market scenarios for
portfolio stress testing and Monte Carlo simulation.

Scenarios modelled:
  - Rate shock (up / down)
  - Credit crisis
  - Liquidity squeeze
  - Stagflation
  - Deflation scare
  - Geopolitical escalation
  - Pandemic wave
  - Commodity super-cycle
  - Tech bubble pop
  - Currency crisis

Capabilities:
  - Monte Carlo across scenario space
  - Portfolio stress test (weights × scenario impacts)
  - Conditional scenario probability
  - Reverse stress test (find scenario producing target loss)
  - Probability calibration from VIX / credit-spread levels
  - Historical analog matching
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDefinition:
    """Full description of a macro scenario."""
    name: str
    description: str
    probability: float                     # base probability [0,1]
    asset_impacts: Dict[str, float]        # asset_class -> % impact
    duration_days: int = 60
    trigger_conditions: List[str] = field(default_factory=list)
    equity_impact: float = 0.0             # %
    bond_impact: float = 0.0
    commodity_impact: float = 0.0
    vol_impact: float = 0.0
    credit_impact: float = 0.0            # spread widening in bps
    severity: float = 1.0                 # 0-1 scale, 1 = extreme


@dataclass
class PortfolioSpec:
    """Simple portfolio definition for stress testing."""
    weights: Dict[str, float]             # asset_class -> weight
    total_value: float = 1_000_000.0


@dataclass
class StressResult:
    """Result of applying one scenario to a portfolio."""
    scenario_name: str
    pnl: float
    pnl_pct: float
    worst_asset: str
    worst_asset_pnl: float
    duration_days: int


@dataclass
class ReverseStressResult:
    """Result of reverse stress test."""
    target_loss_pct: float
    scenario_name: str
    required_severity: float
    scenario_pnl_pct: float


@dataclass
class HistoricalAnalog:
    """Past period that matches a scenario."""
    start_date: str
    end_date: str
    similarity_score: float
    realized_equity: float
    realized_vol: float
    description: str


# ---------------------------------------------------------------------------
# Pre-built scenario library
# ---------------------------------------------------------------------------

def _build_scenario_library() -> Dict[str, ScenarioDefinition]:
    """Return the full library of pre-built macro scenarios."""
    lib: Dict[str, ScenarioDefinition] = {}

    lib["rate_shock_up"] = ScenarioDefinition(
        name="rate_shock_up",
        description="Central bank hikes 100bp+ in a single meeting or inter-meeting emergency",
        probability=0.05,
        asset_impacts={"equity": -0.12, "bond": -0.08, "commodity": -0.05,
                        "crypto": -0.18, "real_estate": -0.10},
        duration_days=30,
        trigger_conditions=["CPI > 6%", "unemployment < 4%", "wage_growth > 5%"],
        equity_impact=-12.0, bond_impact=-8.0, commodity_impact=-5.0,
        vol_impact=60.0, credit_impact=80.0, severity=0.7,
    )

    lib["rate_shock_down"] = ScenarioDefinition(
        name="rate_shock_down",
        description="Emergency rate cut of 75bp+ signalling recession fear",
        probability=0.04,
        asset_impacts={"equity": 0.05, "bond": 0.06, "commodity": 0.02,
                        "crypto": 0.10, "real_estate": 0.04},
        duration_days=20,
        trigger_conditions=["payrolls < -200K", "ISM < 45", "credit_spreads > 500bp"],
        equity_impact=5.0, bond_impact=6.0, commodity_impact=2.0,
        vol_impact=-15.0, credit_impact=-40.0, severity=0.5,
    )

    lib["credit_crisis"] = ScenarioDefinition(
        name="credit_crisis",
        description="Major credit event — HY spreads blow out, defaults spike",
        probability=0.03,
        asset_impacts={"equity": -0.25, "bond": 0.04, "commodity": -0.10,
                        "crypto": -0.35, "real_estate": -0.15},
        duration_days=90,
        trigger_conditions=["HY_spread > 800bp", "IG_spread > 300bp",
                            "default_rate > 5%"],
        equity_impact=-25.0, bond_impact=4.0, commodity_impact=-10.0,
        vol_impact=120.0, credit_impact=400.0, severity=0.9,
    )

    lib["liquidity_squeeze"] = ScenarioDefinition(
        name="liquidity_squeeze",
        description="Sudden evaporation of market depth across asset classes",
        probability=0.06,
        asset_impacts={"equity": -0.08, "bond": -0.03, "commodity": -0.06,
                        "crypto": -0.25, "real_estate": -0.02},
        duration_days=14,
        trigger_conditions=["bid_ask_widening > 3x", "volume_drop > 50%",
                            "repo_rate_spike"],
        equity_impact=-8.0, bond_impact=-3.0, commodity_impact=-6.0,
        vol_impact=80.0, credit_impact=60.0, severity=0.6,
    )

    lib["stagflation"] = ScenarioDefinition(
        name="stagflation",
        description="High inflation + stagnant growth — worst of both worlds",
        probability=0.07,
        asset_impacts={"equity": -0.15, "bond": -0.10, "commodity": 0.15,
                        "crypto": -0.10, "real_estate": -0.05},
        duration_days=180,
        trigger_conditions=["CPI > 5%", "GDP < 1%", "unemployment > 5%"],
        equity_impact=-15.0, bond_impact=-10.0, commodity_impact=15.0,
        vol_impact=40.0, credit_impact=120.0, severity=0.75,
    )

    lib["deflation_scare"] = ScenarioDefinition(
        name="deflation_scare",
        description="CPI turns negative, demand collapses, debt deflation risk",
        probability=0.03,
        asset_impacts={"equity": -0.20, "bond": 0.12, "commodity": -0.20,
                        "crypto": -0.30, "real_estate": -0.12},
        duration_days=120,
        trigger_conditions=["CPI < 0%", "PPI < -3%", "velocity_of_money_falling"],
        equity_impact=-20.0, bond_impact=12.0, commodity_impact=-20.0,
        vol_impact=50.0, credit_impact=150.0, severity=0.8,
    )

    lib["geopolitical_escalation"] = ScenarioDefinition(
        name="geopolitical_escalation",
        description="Major geopolitical conflict — trade disruption, sanctions, supply shock",
        probability=0.08,
        asset_impacts={"equity": -0.10, "bond": 0.03, "commodity": 0.20,
                        "crypto": -0.12, "real_estate": -0.03},
        duration_days=60,
        trigger_conditions=["conflict_breakout", "sanctions_imposed",
                            "shipping_disruption"],
        equity_impact=-10.0, bond_impact=3.0, commodity_impact=20.0,
        vol_impact=45.0, credit_impact=70.0, severity=0.65,
    )

    lib["pandemic_wave"] = ScenarioDefinition(
        name="pandemic_wave",
        description="New pandemic wave with lockdowns and demand destruction",
        probability=0.04,
        asset_impacts={"equity": -0.18, "bond": 0.05, "commodity": -0.15,
                        "crypto": -0.20, "real_estate": -0.08},
        duration_days=90,
        trigger_conditions=["WHO_alert", "hospitalization_surge",
                            "lockdown_announced"],
        equity_impact=-18.0, bond_impact=5.0, commodity_impact=-15.0,
        vol_impact=90.0, credit_impact=100.0, severity=0.8,
    )

    lib["commodity_supercycle"] = ScenarioDefinition(
        name="commodity_supercycle",
        description="Broad commodity rally — energy, metals, agriculture all surge",
        probability=0.06,
        asset_impacts={"equity": 0.05, "bond": -0.06, "commodity": 0.30,
                        "crypto": 0.08, "real_estate": 0.04},
        duration_days=360,
        trigger_conditions=["DXY < 90", "global_PMI > 55",
                            "capex_underinvestment"],
        equity_impact=5.0, bond_impact=-6.0, commodity_impact=30.0,
        vol_impact=15.0, credit_impact=-20.0, severity=0.4,
    )

    lib["tech_bubble_pop"] = ScenarioDefinition(
        name="tech_bubble_pop",
        description="Tech/growth valuations collapse as discount rates normalize",
        probability=0.05,
        asset_impacts={"equity": -0.30, "bond": 0.02, "commodity": -0.03,
                        "crypto": -0.40, "real_estate": -0.05},
        duration_days=180,
        trigger_conditions=["NASDAQ_PE > 35", "rate_hike_cycle",
                            "earnings_miss_wave"],
        equity_impact=-30.0, bond_impact=2.0, commodity_impact=-3.0,
        vol_impact=70.0, credit_impact=60.0, severity=0.85,
    )

    lib["currency_crisis"] = ScenarioDefinition(
        name="currency_crisis",
        description="Major EM or DM currency collapse — capital flight, reserves drain",
        probability=0.04,
        asset_impacts={"equity": -0.14, "bond": -0.05, "commodity": 0.08,
                        "crypto": 0.15, "real_estate": -0.08},
        duration_days=60,
        trigger_conditions=["FX_vol_spike", "reserves_drawdown",
                            "current_account_deficit > 6%"],
        equity_impact=-14.0, bond_impact=-5.0, commodity_impact=8.0,
        vol_impact=55.0, credit_impact=90.0, severity=0.7,
    )

    return lib


SCENARIO_LIBRARY = _build_scenario_library()


# ---------------------------------------------------------------------------
# Scenario engine — Monte Carlo & stress testing
# ---------------------------------------------------------------------------

class ScenarioEngine:
    """Run Monte Carlo across scenarios and compute expected portfolio P&L."""

    def __init__(self, scenarios: Optional[Dict[str, ScenarioDefinition]] = None,
                 seed: int = 42):
        self.scenarios = scenarios or dict(SCENARIO_LIBRARY)
        self.rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------------
    # Monte Carlo
    # -----------------------------------------------------------------------

    def monte_carlo(self, portfolio: PortfolioSpec, n_sims: int = 10_000,
                    horizon_days: int = 252) -> Dict[str, object]:
        """
        Run Monte Carlo across the scenario space.

        Each simulation:
          1. Sample which scenarios occur (Bernoulli on probability).
          2. For active scenarios, apply asset impacts (with noise).
          3. Compute portfolio P&L.

        Returns distribution statistics.
        """
        pnl_array = np.zeros(n_sims)
        scenario_counts: Dict[str, int] = {s: 0 for s in self.scenarios}

        for i in range(n_sims):
            total_impact: Dict[str, float] = {}
            for sname, sdef in self.scenarios.items():
                # Scale probability to horizon
                prob_adj = 1.0 - (1.0 - sdef.probability) ** (horizon_days / 252.0)
                if self.rng.random() < prob_adj:
                    scenario_counts[sname] += 1
                    noise = self.rng.normal(1.0, 0.15)
                    for asset, impact in sdef.asset_impacts.items():
                        prev = total_impact.get(asset, 0.0)
                        total_impact[asset] = prev + impact * noise

            sim_pnl = 0.0
            for asset, weight in portfolio.weights.items():
                asset_ret = total_impact.get(asset, self.rng.normal(0.06 / 252 * horizon_days, 0.16 / math.sqrt(252) * math.sqrt(horizon_days)))
                sim_pnl += weight * asset_ret

            pnl_array[i] = sim_pnl

        pnl_dollar = pnl_array * portfolio.total_value
        sorted_pnl = np.sort(pnl_array)

        return {
            "mean_pnl_pct": float(np.mean(pnl_array) * 100),
            "std_pnl_pct": float(np.std(pnl_array) * 100),
            "var_95_pct": float(sorted_pnl[int(0.05 * n_sims)] * 100),
            "var_99_pct": float(sorted_pnl[int(0.01 * n_sims)] * 100),
            "cvar_95_pct": float(np.mean(sorted_pnl[:int(0.05 * n_sims)]) * 100),
            "max_loss_pct": float(sorted_pnl[0] * 100),
            "max_gain_pct": float(sorted_pnl[-1] * 100),
            "mean_pnl_dollar": float(np.mean(pnl_dollar)),
            "var_95_dollar": float(np.sort(pnl_dollar)[int(0.05 * n_sims)]),
            "scenario_frequencies": {k: v / n_sims for k, v in scenario_counts.items()},
            "pnl_distribution": pnl_array,
        }

    # -----------------------------------------------------------------------
    # Stress test — deterministic application of each scenario
    # -----------------------------------------------------------------------

    def stress_test(self, portfolio: PortfolioSpec,
                    severity_override: Optional[float] = None) -> List[StressResult]:
        """Apply each scenario to the portfolio and return ranked results."""
        results: List[StressResult] = []
        for sname, sdef in self.scenarios.items():
            sev = severity_override if severity_override is not None else sdef.severity
            pnl = 0.0
            worst_asset = ""
            worst_pnl = 0.0
            for asset, weight in portfolio.weights.items():
                impact = sdef.asset_impacts.get(asset, 0.0) * sev
                asset_pnl = weight * impact * portfolio.total_value
                pnl += asset_pnl
                if asset_pnl < worst_pnl:
                    worst_pnl = asset_pnl
                    worst_asset = asset
            pnl_pct = pnl / portfolio.total_value * 100 if portfolio.total_value else 0.0
            results.append(StressResult(
                scenario_name=sname, pnl=pnl, pnl_pct=pnl_pct,
                worst_asset=worst_asset, worst_asset_pnl=worst_pnl,
                duration_days=sdef.duration_days,
            ))
        results.sort(key=lambda r: r.pnl)
        return results

    def stress_test_single(self, portfolio: PortfolioSpec,
                           scenario_name: str,
                           severity: float = 1.0) -> StressResult:
        """Stress test a single named scenario with given severity."""
        sdef = self.scenarios[scenario_name]
        pnl = 0.0
        worst_asset = ""
        worst_pnl = 0.0
        for asset, weight in portfolio.weights.items():
            impact = sdef.asset_impacts.get(asset, 0.0) * severity
            asset_pnl = weight * impact * portfolio.total_value
            pnl += asset_pnl
            if asset_pnl < worst_pnl:
                worst_pnl = asset_pnl
                worst_asset = asset
        pnl_pct = pnl / portfolio.total_value * 100 if portfolio.total_value else 0.0
        return StressResult(
            scenario_name=scenario_name, pnl=pnl, pnl_pct=pnl_pct,
            worst_asset=worst_asset, worst_asset_pnl=worst_pnl,
            duration_days=sdef.duration_days,
        )

    # -----------------------------------------------------------------------
    # Conditional scenario probability
    # -----------------------------------------------------------------------

    def conditional_probability(self, scenario_b: str, given_a: str) -> float:
        """
        Estimate P(scenario_B | scenario_A already happening).

        Uses a simple copula-like model: scenarios that share trigger
        conditions or have similar asset impacts are correlated.
        """
        sdef_a = self.scenarios[given_a]
        sdef_b = self.scenarios[scenario_b]

        # Trigger overlap score
        triggers_a = set(sdef_a.trigger_conditions)
        triggers_b = set(sdef_b.trigger_conditions)
        if triggers_a and triggers_b:
            overlap = len(triggers_a & triggers_b) / max(len(triggers_a | triggers_b), 1)
        else:
            overlap = 0.0

        # Impact correlation score
        common_assets = set(sdef_a.asset_impacts) & set(sdef_b.asset_impacts)
        if common_assets:
            impacts_a = np.array([sdef_a.asset_impacts[a] for a in common_assets])
            impacts_b = np.array([sdef_b.asset_impacts[a] for a in common_assets])
            if np.std(impacts_a) > 0 and np.std(impacts_b) > 0:
                corr = float(np.corrcoef(impacts_a, impacts_b)[0, 1])
            else:
                corr = 0.0
        else:
            corr = 0.0

        # Combine: higher overlap / correlation → higher conditional prob
        base_prob = sdef_b.probability
        boost = 0.5 * overlap + 0.5 * max(corr, 0.0)
        cond_prob = min(base_prob + boost * (1.0 - base_prob), 0.95)
        return cond_prob

    def conditional_scenario_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build full conditional probability matrix P(B | A)."""
        names = list(self.scenarios.keys())
        matrix: Dict[str, Dict[str, float]] = {}
        for a in names:
            matrix[a] = {}
            for b in names:
                if a == b:
                    matrix[a][b] = 1.0
                else:
                    matrix[a][b] = self.conditional_probability(b, a)
        return matrix

    # -----------------------------------------------------------------------
    # Reverse stress test
    # -----------------------------------------------------------------------

    def reverse_stress_test(self, portfolio: PortfolioSpec,
                            target_loss_pct: float) -> List[ReverseStressResult]:
        """
        Find scenarios (and required severity) that produce the target loss.

        target_loss_pct: negative number, e.g. -20 for a 20% portfolio loss.
        """
        results: List[ReverseStressResult] = []
        for sname, sdef in self.scenarios.items():
            # Compute P&L at severity=1
            base_pnl = 0.0
            for asset, weight in portfolio.weights.items():
                impact = sdef.asset_impacts.get(asset, 0.0)
                base_pnl += weight * impact
            base_pnl_pct = base_pnl * 100

            if abs(base_pnl_pct) < 1e-9:
                continue

            required_severity = target_loss_pct / base_pnl_pct
            if required_severity < 0:
                continue  # scenario moves portfolio in opposite direction

            actual_pnl_pct = base_pnl_pct * required_severity
            results.append(ReverseStressResult(
                target_loss_pct=target_loss_pct,
                scenario_name=sname,
                required_severity=required_severity,
                scenario_pnl_pct=actual_pnl_pct,
            ))
        results.sort(key=lambda r: r.required_severity)
        return results

    # -----------------------------------------------------------------------
    # Probability calibration from market data
    # -----------------------------------------------------------------------

    def calibrate_probabilities(self, vix: float, hy_spread_bps: float,
                                ig_spread_bps: float) -> Dict[str, float]:
        """
        Adjust scenario probabilities based on current market stress levels.

        Higher VIX and wider spreads → higher probability of stress scenarios.
        """
        # Stress index: 0 = calm, 1 = extreme stress
        vix_stress = min(max((vix - 12) / 40, 0.0), 1.0)
        hy_stress = min(max((hy_spread_bps - 300) / 700, 0.0), 1.0)
        ig_stress = min(max((ig_spread_bps - 80) / 300, 0.0), 1.0)
        stress_index = 0.4 * vix_stress + 0.35 * hy_stress + 0.25 * ig_stress

        calibrated: Dict[str, float] = {}
        for sname, sdef in self.scenarios.items():
            base = sdef.probability
            if sdef.severity > 0.5:
                # Stress scenarios get probability boost when market is stressed
                adjusted = base * (1.0 + 2.0 * stress_index)
            else:
                # Benign scenarios get reduced probability
                adjusted = base * (1.0 - 0.5 * stress_index)
            calibrated[sname] = min(max(adjusted, 0.001), 0.50)
        return calibrated

    def apply_calibrated_probabilities(self, vix: float, hy_spread_bps: float,
                                        ig_spread_bps: float) -> None:
        """Mutate scenario probabilities in-place based on market data."""
        calibrated = self.calibrate_probabilities(vix, hy_spread_bps, ig_spread_bps)
        for sname, new_prob in calibrated.items():
            self.scenarios[sname].probability = new_prob

    # -----------------------------------------------------------------------
    # Historical analog matching
    # -----------------------------------------------------------------------

    def find_historical_analog(self, scenario_name: str,
                               historical_periods: Optional[List[Dict]] = None
                               ) -> List[HistoricalAnalog]:
        """
        Find historical periods most similar to a given scenario.

        If no historical_periods provided, uses a built-in library.
        """
        if historical_periods is None:
            historical_periods = _default_historical_periods()

        sdef = self.scenarios[scenario_name]
        target_vec = np.array([
            sdef.equity_impact, sdef.bond_impact, sdef.commodity_impact,
            sdef.vol_impact / 100.0, sdef.credit_impact / 100.0,
        ])

        analogs: List[HistoricalAnalog] = []
        for period in historical_periods:
            period_vec = np.array([
                period["equity_return"], period["bond_return"],
                period["commodity_return"], period["vol_change"],
                period["credit_change"],
            ])
            # Cosine similarity
            dot = float(np.dot(target_vec, period_vec))
            norm_t = float(np.linalg.norm(target_vec))
            norm_p = float(np.linalg.norm(period_vec))
            if norm_t > 0 and norm_p > 0:
                similarity = dot / (norm_t * norm_p)
            else:
                similarity = 0.0

            analogs.append(HistoricalAnalog(
                start_date=period["start"],
                end_date=period["end"],
                similarity_score=similarity,
                realized_equity=period["equity_return"],
                realized_vol=period.get("realized_vol", 0.0),
                description=period["description"],
            ))

        analogs.sort(key=lambda a: a.similarity_score, reverse=True)
        return analogs

    def best_analog(self, scenario_name: str) -> Optional[HistoricalAnalog]:
        """Return the single best historical analog for a scenario."""
        analogs = self.find_historical_analog(scenario_name)
        return analogs[0] if analogs else None


# ---------------------------------------------------------------------------
# Built-in historical periods for analog matching
# ---------------------------------------------------------------------------

def _default_historical_periods() -> List[Dict]:
    """Library of notable historical market episodes."""
    return [
        {
            "start": "2008-09-15", "end": "2009-03-09",
            "description": "Global Financial Crisis — Lehman collapse",
            "equity_return": -0.45, "bond_return": 0.08,
            "commodity_return": -0.35, "vol_change": 0.80,
            "credit_change": 0.60, "realized_vol": 0.65,
        },
        {
            "start": "2020-02-19", "end": "2020-03-23",
            "description": "COVID crash — fastest bear market in history",
            "equity_return": -0.34, "bond_return": 0.05,
            "commodity_return": -0.25, "vol_change": 0.70,
            "credit_change": 0.35, "realized_vol": 0.80,
        },
        {
            "start": "2022-01-03", "end": "2022-10-12",
            "description": "2022 rate shock — aggressive Fed tightening",
            "equity_return": -0.25, "bond_return": -0.15,
            "commodity_return": 0.10, "vol_change": 0.25,
            "credit_change": 0.15, "realized_vol": 0.28,
        },
        {
            "start": "2000-03-10", "end": "2002-10-09",
            "description": "Dot-com bust — tech bubble collapse",
            "equity_return": -0.49, "bond_return": 0.12,
            "commodity_return": -0.10, "vol_change": 0.35,
            "credit_change": 0.25, "realized_vol": 0.30,
        },
        {
            "start": "2011-07-22", "end": "2011-10-03",
            "description": "Euro debt crisis + US downgrade",
            "equity_return": -0.19, "bond_return": 0.06,
            "commodity_return": -0.12, "vol_change": 0.40,
            "credit_change": 0.20, "realized_vol": 0.35,
        },
        {
            "start": "2018-10-03", "end": "2018-12-24",
            "description": "Q4 2018 selloff — Fed over-tightening fears",
            "equity_return": -0.20, "bond_return": 0.02,
            "commodity_return": -0.15, "vol_change": 0.30,
            "credit_change": 0.12, "realized_vol": 0.25,
        },
        {
            "start": "2015-08-18", "end": "2015-08-25",
            "description": "China devaluation shock — flash crash",
            "equity_return": -0.11, "bond_return": 0.01,
            "commodity_return": -0.08, "vol_change": 0.55,
            "credit_change": 0.08, "realized_vol": 0.45,
        },
        {
            "start": "1973-01-01", "end": "1974-12-31",
            "description": "1973-74 stagflation — oil embargo + recession",
            "equity_return": -0.42, "bond_return": -0.08,
            "commodity_return": 0.40, "vol_change": 0.30,
            "credit_change": 0.20, "realized_vol": 0.25,
        },
        {
            "start": "1997-07-02", "end": "1998-10-08",
            "description": "Asian / Russian crisis + LTCM collapse",
            "equity_return": -0.22, "bond_return": 0.10,
            "commodity_return": -0.15, "vol_change": 0.50,
            "credit_change": 0.40, "realized_vol": 0.30,
        },
        {
            "start": "2022-05-01", "end": "2022-06-18",
            "description": "Terra/Luna + 3AC collapse — crypto credit crisis",
            "equity_return": -0.08, "bond_return": -0.04,
            "commodity_return": -0.05, "vol_change": 0.20,
            "credit_change": 0.10, "realized_vol": 0.22,
        },
    ]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def combined_scenario(scenarios: List[ScenarioDefinition],
                      correlation: float = 0.3) -> ScenarioDefinition:
    """
    Combine multiple scenarios into one joint scenario.

    Impacts are summed with a diversification factor based on correlation.
    """
    if not scenarios:
        raise ValueError("Must supply at least one scenario")

    n = len(scenarios)
    combined_impacts: Dict[str, float] = {}
    for s in scenarios:
        for asset, impact in s.asset_impacts.items():
            combined_impacts[asset] = combined_impacts.get(asset, 0.0) + impact

    # Diversification: sqrt of sum-of-squares when correlation < 1
    if n > 1:
        div_factor = math.sqrt(n + n * (n - 1) * correlation) / n
        for asset in combined_impacts:
            combined_impacts[asset] *= div_factor

    # Joint probability: assume some dependence
    joint_prob = 1.0
    for s in scenarios:
        joint_prob *= s.probability
    joint_prob = joint_prob ** (1.0 / max(n * (1.0 - correlation * 0.5), 1.0))

    names = [s.name for s in scenarios]
    return ScenarioDefinition(
        name="combined_" + "+".join(names),
        description=f"Combined scenario: {', '.join(names)}",
        probability=joint_prob,
        asset_impacts=combined_impacts,
        duration_days=max(s.duration_days for s in scenarios),
        trigger_conditions=list({t for s in scenarios for t in s.trigger_conditions}),
        equity_impact=sum(s.equity_impact for s in scenarios) * (div_factor if n > 1 else 1.0),
        bond_impact=sum(s.bond_impact for s in scenarios) * (div_factor if n > 1 else 1.0),
        commodity_impact=sum(s.commodity_impact for s in scenarios) * (div_factor if n > 1 else 1.0),
        vol_impact=max(s.vol_impact for s in scenarios),
        credit_impact=max(s.credit_impact for s in scenarios),
        severity=max(s.severity for s in scenarios),
    )


def scenario_summary_table(results: List[StressResult]) -> List[Dict[str, object]]:
    """Format stress results into a summary table."""
    rows: List[Dict[str, object]] = []
    for r in results:
        rows.append({
            "scenario": r.scenario_name,
            "pnl": round(r.pnl, 2),
            "pnl_pct": round(r.pnl_pct, 2),
            "worst_asset": r.worst_asset,
            "worst_asset_pnl": round(r.worst_asset_pnl, 2),
            "duration": r.duration_days,
        })
    return rows


def rank_scenarios_by_expected_loss(engine: ScenarioEngine,
                                    portfolio: PortfolioSpec) -> List[Tuple[str, float]]:
    """Rank scenarios by probability-weighted expected loss."""
    ranked: List[Tuple[str, float]] = []
    for sname, sdef in engine.scenarios.items():
        pnl = 0.0
        for asset, weight in portfolio.weights.items():
            impact = sdef.asset_impacts.get(asset, 0.0)
            pnl += weight * impact
        expected_loss = pnl * sdef.probability * portfolio.total_value
        ranked.append((sname, expected_loss))
    ranked.sort(key=lambda x: x[1])
    return ranked


def tail_scenario_blend(engine: ScenarioEngine, portfolio: PortfolioSpec,
                        n_worst: int = 3) -> ScenarioDefinition:
    """Create a blended scenario from the N worst scenarios for this portfolio."""
    results = engine.stress_test(portfolio)
    worst_names = [r.scenario_name for r in results[:n_worst]]
    worst_defs = [engine.scenarios[n] for n in worst_names]
    return combined_scenario(worst_defs, correlation=0.5)


def portfolio_scenario_heatmap(engine: ScenarioEngine,
                               portfolio: PortfolioSpec) -> Dict[str, Dict[str, float]]:
    """
    Build a heatmap: scenario × asset → P&L contribution.

    Useful for identifying which asset is most exposed to which scenario.
    """
    heatmap: Dict[str, Dict[str, float]] = {}
    for sname, sdef in engine.scenarios.items():
        heatmap[sname] = {}
        for asset, weight in portfolio.weights.items():
            impact = sdef.asset_impacts.get(asset, 0.0) * sdef.severity
            heatmap[sname][asset] = round(weight * impact * portfolio.total_value, 2)
    return heatmap
