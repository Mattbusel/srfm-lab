"""
Market analogy database — cross-domain analogies for creative idea generation.

Maps market phenomena to analogies from:
  - Biology: predator-prey, epidemics, evolution, ecology
  - Physics: phase transitions, resonance, entropy, gravity
  - Game theory: auctions, Nash equilibria, arms races
  - Engineering: control systems, feedback loops, signal processing
  - Psychology: herding, anchoring, loss aversion, overconfidence
  - Military strategy: flanking, attrition, logistics, deception

Each analogy suggests a hypothetical trading strategy or risk scenario.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketAnalogy:
    """A cross-domain analogy mapped to a market phenomenon."""
    source_domain: str          # physics / biology / game_theory / etc.
    source_concept: str         # e.g., "predator-prey dynamics"
    market_phenomenon: str      # e.g., "momentum and mean reversion cycles"
    mechanism: str              # how the analogy works mechanically
    trading_hypothesis: str     # the actionable trading hypothesis
    key_variables: list[str]    # variables to monitor
    regime_applicability: list[str]  # which regimes this applies to
    confidence: float           # 0-1 how well the analogy holds
    historical_examples: list[str] = field(default_factory=list)


# ── Analogy Library ───────────────────────────────────────────────────────────

ANALOGY_LIBRARY: list[MarketAnalogy] = [
    MarketAnalogy(
        source_domain="biology",
        source_concept="predator-prey (Lotka-Volterra)",
        market_phenomenon="momentum-mean reversion cycle",
        mechanism=(
            "Momentum traders (predators) feed on trending price moves (prey). "
            "As trend exhausts, mean reversion traders (new predators) emerge. "
            "Population cycles: trend dominates → overshoots → reverses → mean reverts."
        ),
        trading_hypothesis=(
            "After strong momentum signal (>2 std), fade the trend for mean reversion. "
            "Entry: when trend-following AUM peaks (COT data), initiate counter-trend."
        ),
        key_variables=["trend_strength", "cot_positioning", "momentum_crowding", "vol_of_vol"],
        regime_applicability=["trending_bull", "trending_bear", "mean_reverting"],
        confidence=0.72,
        historical_examples=["2020 tech momentum unwind", "2021-2022 growth → value rotation"],
    ),
    MarketAnalogy(
        source_domain="physics",
        source_concept="phase transition (critical point)",
        market_phenomenon="market crash / regime change",
        mechanism=(
            "Near a critical point, fluctuations diverge (like density near gas/liquid transition). "
            "In markets: correlation spikes, vol-of-vol rises, autocorrelation changes sign. "
            "The system is metastable — small trigger causes cascade."
        ),
        trading_hypothesis=(
            "Buy tail protection (puts, variance swaps) when: "
            "(1) realized corr crosses 0.7, (2) VIX/VVIX ratio < 5, "
            "(3) LPPL log-periodic oscillation detected in 252-day price series."
        ),
        key_variables=["realized_correlation", "vix_vvix_ratio", "lppl_fit_quality", "bid_ask_spread_index"],
        regime_applicability=["high_volatility", "crisis"],
        confidence=0.65,
        historical_examples=["2008 GFC", "2020 COVID crash", "2022 rate shock"],
    ),
    MarketAnalogy(
        source_domain="biology",
        source_concept="epidemic spread (SIR model)",
        market_phenomenon="information cascade / contagion",
        mechanism=(
            "Ideas/news spread through market participants like a virus. "
            "Susceptible (S) = uninformed participants. "
            "Infected (I) = active traders. Recovered (R) = already positioned. "
            "When I peaks, most momentum is captured; reversal likely."
        ),
        trading_hypothesis=(
            "Measure information spread rate via social media volume and options activity. "
            "When 'infection rate' peaks (dI/dt = 0), expect momentum exhaustion within 3-5 days."
        ),
        key_variables=["social_media_volume", "options_volume_vs_avg", "news_velocity", "retail_flow"],
        regime_applicability=["trending_bull", "trending_bear"],
        confidence=0.58,
        historical_examples=["GME short squeeze 2021", "NVDA AI narrative 2023"],
    ),
    MarketAnalogy(
        source_domain="physics",
        source_concept="resonance and natural frequency",
        market_phenomenon="technical level confluence / support-resistance",
        mechanism=(
            "Oscillating systems respond most strongly at their natural frequency. "
            "Price has 'natural frequencies' from options gamma/charm at key strikes, "
            "fund rebalancing cycles, and index reconstitution dates. "
            "When multiple frequencies align, expect amplified moves."
        ),
        trading_hypothesis=(
            "Identify price levels where: gamma exposure changes sign, "
            "52-week and 200-day MAs converge, and monthly open/close aligns. "
            "Position for breakout OR strong bounce at such levels."
        ),
        key_variables=["gamma_exposure_by_strike", "ma_confluence", "volume_at_price", "options_oi_concentration"],
        regime_applicability=["trending_bull", "trending_bear", "mean_reverting"],
        confidence=0.70,
        historical_examples=["S&P 4200 level Q3 2022", "BTC 30k support 2022"],
    ),
    MarketAnalogy(
        source_domain="game_theory",
        source_concept="Colonel Blotto / resource allocation game",
        market_phenomenon="institutional portfolio rebalancing",
        mechanism=(
            "Large asset managers allocate capital like Blotto strategies — "
            "limited 'troops' distributed to maximize return across 'battlefields'. "
            "Month-end rebalancing creates predictable flows: "
            "equities vs bonds vs alternatives all compete for AUM."
        ),
        trading_hypothesis=(
            "Month-end rebalancing: equities outperform vs bonds in final 3 days of month "
            "when equity/bond ratio has drifted >5%. Front-run the rebalance."
        ),
        key_variables=["equity_bond_ratio_drift", "month_end_proximity", "pension_fund_rebalancing_signal"],
        regime_applicability=["trending_bull", "unknown"],
        confidence=0.63,
        historical_examples=["Documented month-end equity momentum effect"],
    ),
    MarketAnalogy(
        source_domain="engineering",
        source_concept="PID control loop (proportional-integral-derivative)",
        market_phenomenon="central bank policy and market feedback",
        mechanism=(
            "Central banks act as PID controllers: "
            "P (proportional) = respond to current inflation gap, "
            "I (integral) = respond to cumulative inflation history, "
            "D (derivative) = respond to rate of change of inflation. "
            "Market anticipates future policy based on PID parameters."
        ),
        trading_hypothesis=(
            "When realized inflation deviates from Fed target by >150bps for >3 months (I term large), "
            "expect aggressive policy overshoot. Short duration bonds, long volatility."
        ),
        key_variables=["inflation_gap", "cumulative_inflation_gap", "policy_rate_gap", "breakeven_10y"],
        regime_applicability=["macro_tightening", "macro_easing", "crisis"],
        confidence=0.68,
        historical_examples=["2022 Fed hiking cycle", "1979-1981 Volcker shock"],
    ),
    MarketAnalogy(
        source_domain="psychology",
        source_concept="anchoring bias with updating",
        market_phenomenon="earnings estimate revision momentum",
        mechanism=(
            "Analysts anchor to prior estimates and update insufficiently to new info. "
            "First revision tends to understate true change; subsequent revisions continue. "
            "Creates predictable estimate revision momentum."
        ),
        trading_hypothesis=(
            "Go long stocks with positive EPS revision breadth (>60% of analysts raising). "
            "Hold for 1-3 months. Fade stocks with 3+ consecutive downward revisions."
        ),
        key_variables=["eps_revision_breadth", "estimate_revision_magnitude", "analyst_count", "time_since_last_revision"],
        regime_applicability=["trending_bull", "unknown"],
        confidence=0.74,
        historical_examples=["Well-documented in academic literature (Chan, Jegadeesh, Lakonishok 1996)"],
    ),
    MarketAnalogy(
        source_domain="military",
        source_concept="logistics and supply chain constraints",
        market_phenomenon="commodity supply chain disruption trading",
        mechanism=(
            "Military logistics: the army that runs out of supplies loses. "
            "Commodity traders: downstream demand is limited by upstream supply. "
            "Supply chain bottlenecks create price spikes at constrained nodes."
        ),
        trading_hypothesis=(
            "Map commodity supply chains. When upstream constraint detected "
            "(via freight rates, inventory data, producer surveys), "
            "go long downstream commodities that cannot substitute quickly."
        ),
        key_variables=["shipping_rates", "producer_inventory", "lead_times", "import_export_data"],
        regime_applicability=["macro_tightening", "crisis", "unknown"],
        confidence=0.61,
        historical_examples=["2021-2022 semiconductor shortage", "2022 natural gas crisis"],
    ),
    MarketAnalogy(
        source_domain="ecology",
        source_concept="niche competition and carrying capacity",
        market_phenomenon="strategy crowding and capacity constraints",
        mechanism=(
            "In ecology, too many organisms competing for same niche depletes resources. "
            "In markets, too many strategies with same factor exposure = diminishing returns. "
            "Carrying capacity = total alpha available. Crowded strategies fight over shrinking pie."
        ),
        trading_hypothesis=(
            "Measure strategy crowding via: (1) factor exposure similarity in 13F filings, "
            "(2) correlation of hedge fund returns to factor. "
            "Underweight crowded factors; allocate to uncrowded factors with similar characteristics."
        ),
        key_variables=["factor_crowding_score", "13f_concentration", "hedge_fund_return_correlation", "factor_capacity"],
        regime_applicability=["trending_bull", "high_volatility"],
        confidence=0.70,
    ),
    MarketAnalogy(
        source_domain="physics",
        source_concept="entropy and information theory",
        market_phenomenon="market efficiency and alpha decay",
        mechanism=(
            "Entropy increases as information diffuses into prices (EMH). "
            "Low entropy = concentrated order flow = predictable. "
            "High entropy = random walk = no alpha. "
            "Alpha exists in low-entropy pockets (illiquidity, complexity, behavioral)."
        ),
        trading_hypothesis=(
            "Measure market entropy via permutation entropy of price returns. "
            "When entropy < 0.5 (ordered), trend-following works. "
            "When entropy > 0.85 (chaotic), switch to mean reversion or exit."
        ),
        key_variables=["permutation_entropy", "approx_entropy", "hurst_exponent", "autocorrelation_lag1"],
        regime_applicability=["trending_bull", "mean_reverting", "high_volatility"],
        confidence=0.67,
    ),
    MarketAnalogy(
        source_domain="biology",
        source_concept="immune system and memory cells",
        market_phenomenon="institutional memory and historical price levels",
        mechanism=(
            "Immune memory: body remembers past pathogens and responds faster. "
            "Market memory: participants remember past price levels and react at them. "
            "'Immune memory' = volume at price / historical support/resistance. "
            "Strong institutional memory at round numbers and prior highs/lows."
        ),
        trading_hypothesis=(
            "Prior all-time highs / year lows with high volume-at-price = strong S/R. "
            "After break of key level: retest probability >60%. "
            "Use volume profile to identify highest-memory price levels."
        ),
        key_variables=["volume_at_price", "time_at_price", "historical_turn_count", "options_open_interest"],
        regime_applicability=["trending_bull", "trending_bear", "mean_reverting"],
        confidence=0.73,
    ),
    MarketAnalogy(
        source_domain="game_theory",
        source_concept="repeated prisoner's dilemma and cooperation",
        market_phenomenon="market maker coordination and bid-ask spread stability",
        mechanism=(
            "Multiple market makers face prisoner's dilemma: "
            "narrow spread (cooperate) = low profit but stable market, "
            "wide spread (defect) = high profit but invites competition. "
            "In illiquid/volatile conditions, cooperation breaks down → spreads widen."
        ),
        trading_hypothesis=(
            "Bid-ask spread widening preceded by: options vol spike, VIX > 25, "
            "or major news event. Front-run spread widening by reducing limit orders, "
            "switch to market orders for fills."
        ),
        key_variables=["bid_ask_spread_history", "vix_level", "market_maker_inventory", "quote_stuffing_count"],
        regime_applicability=["high_volatility", "crisis"],
        confidence=0.65,
    ),
]


# ── Analogy Engine ────────────────────────────────────────────────────────────

class MarketAnalogyEngine:
    """
    Search and apply cross-domain analogies to current market conditions.
    """

    def __init__(self):
        self.library = ANALOGY_LIBRARY
        self._usage_counts: dict[str, int] = {}

    def search_by_domain(self, domain: str) -> list[MarketAnalogy]:
        return [a for a in self.library if a.source_domain == domain]

    def search_by_regime(self, regime: str) -> list[MarketAnalogy]:
        return [a for a in self.library
                if regime in a.regime_applicability or "unknown" in a.regime_applicability]

    def search_by_variable(self, variable_keyword: str) -> list[MarketAnalogy]:
        kw = variable_keyword.lower()
        return [a for a in self.library
                if any(kw in v.lower() for v in a.key_variables)]

    def rank_analogies(
        self,
        available_signals: list[str],
        current_regime: str,
        min_confidence: float = 0.5,
    ) -> list[dict]:
        """
        Rank analogies by relevance to current market conditions.
        Score = confidence * regime_match * signal_coverage.
        """
        results = []
        for analogy in self.library:
            if analogy.confidence < min_confidence:
                continue

            # Regime match
            regime_match = 1.0 if (current_regime in analogy.regime_applicability
                                    or "unknown" in analogy.regime_applicability) else 0.3

            # Signal coverage: how many key variables we have data for
            available_lower = [s.lower() for s in available_signals]
            covered = sum(1 for v in analogy.key_variables
                          if any(word in available_lower for word in v.lower().split("_")))
            signal_cov = float(covered) / max(len(analogy.key_variables), 1)

            score = float(analogy.confidence * regime_match * (0.5 + 0.5 * signal_cov))

            results.append({
                "analogy": analogy,
                "score": score,
                "regime_match": regime_match,
                "signal_coverage": signal_cov,
                "source": f"{analogy.source_domain}:{analogy.source_concept}",
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def generate_hypothesis_from_analogy(
        self,
        analogy: MarketAnalogy,
        current_values: dict[str, float],
    ) -> dict:
        """
        Generate a concrete trading hypothesis from an analogy given current signal values.
        """
        # Check if key variables suggest the analogy is active
        relevant_vars = {}
        for v in analogy.key_variables:
            if v in current_values:
                relevant_vars[v] = current_values[v]

        # Simple heuristic: if >50% of key variables available and non-zero
        data_quality = len(relevant_vars) / max(len(analogy.key_variables), 1)
        is_active = data_quality > 0.4

        return {
            "analogy_name": f"{analogy.source_domain}:{analogy.source_concept}",
            "market_phenomenon": analogy.market_phenomenon,
            "trading_hypothesis": analogy.trading_hypothesis,
            "is_active": is_active,
            "data_quality": data_quality,
            "available_variables": relevant_vars,
            "confidence": float(analogy.confidence * data_quality),
            "regime_applicability": analogy.regime_applicability,
        }

    def add_analogy(self, analogy: MarketAnalogy) -> None:
        """Add a new analogy to the library."""
        self.library.append(analogy)

    def domain_summary(self) -> dict:
        domains = {}
        for a in self.library:
            domains[a.source_domain] = domains.get(a.source_domain, 0) + 1
        avg_conf = float(np.mean([a.confidence for a in self.library]))
        return {
            "total_analogies": len(self.library),
            "by_domain": domains,
            "avg_confidence": avg_conf,
            "domains": list(domains.keys()),
        }


# ── Lotka-Volterra Market Model ────────────────────────────────────────────────

def lotka_volterra_market(
    x0: float,             # initial trend-follower population
    y0: float,             # initial mean-reversion trader population
    alpha: float = 0.3,    # trend-follower birth rate
    beta: float = 0.1,     # predation rate
    gamma: float = 0.2,    # mean-rev death rate
    delta: float = 0.05,   # mean-rev growth from trend exhaustion
    n_periods: int = 252,
    dt: float = 1.0,
) -> dict:
    """
    Lotka-Volterra model of momentum vs mean-reversion trader dynamics.
    dx/dt = alpha*x - beta*x*y  (trend-followers grow, die when crowded)
    dy/dt = -gamma*y + delta*x*y (mean-rev die naturally, grow when trend exhausts)
    """
    x = np.zeros(n_periods)
    y = np.zeros(n_periods)
    x[0] = x0
    y[0] = y0

    for t in range(1, n_periods):
        dx = (alpha * x[t-1] - beta * x[t-1] * y[t-1]) * dt
        dy = (-gamma * y[t-1] + delta * x[t-1] * y[t-1]) * dt
        x[t] = max(x[t-1] + dx, 0.01)
        y[t] = max(y[t-1] + dy, 0.01)

    # Implied price impact: trend-followers push prices away from mean
    # net_pressure > 0 = trending market
    net_pressure = x - y
    trading_signal = np.tanh(net_pressure / max(abs(net_pressure).max(), 1e-10))

    return {
        "trend_follower_pop": x.tolist(),
        "mean_rev_pop": y.tolist(),
        "net_pressure": net_pressure.tolist(),
        "trading_signal": trading_signal.tolist(),
        "current_regime": "trending" if net_pressure[-1] > 0.5 else
                           "mean_reverting" if net_pressure[-1] < -0.5 else "transitioning",
    }


# ── SIR Information Cascade ───────────────────────────────────────────────────

def sir_information_cascade(
    n_participants: int = 1000,
    initial_informed_frac: float = 0.01,
    beta_spread: float = 0.3,     # infection rate
    gamma_recover: float = 0.05,  # recovery rate (become "positioned" / exhausted)
    n_periods: int = 30,
) -> dict:
    """
    SIR model for information cascade through market participants.
    S: uninformed, I: actively trading on news, R: already positioned/exhausted.
    """
    S = np.zeros(n_periods)
    I = np.zeros(n_periods)
    R = np.zeros(n_periods)

    S[0] = n_participants * (1 - initial_informed_frac)
    I[0] = n_participants * initial_informed_frac
    R[0] = 0.0
    N = float(n_participants)

    for t in range(1, n_periods):
        new_infected = beta_spread * I[t-1] * S[t-1] / N
        new_recovered = gamma_recover * I[t-1]
        S[t] = max(S[t-1] - new_infected, 0)
        I[t] = max(I[t-1] + new_infected - new_recovered, 0)
        R[t] = R[t-1] + new_recovered

    # Peak active trading
    peak_t = int(np.argmax(I))
    peak_I_frac = float(I.max() / N)

    # Market pressure: I / N as buying pressure proxy
    pressure = I / N

    return {
        "susceptible": S.tolist(),
        "infected_trading": I.tolist(),
        "recovered_positioned": R.tolist(),
        "pressure": pressure.tolist(),
        "peak_period": peak_t,
        "peak_active_fraction": peak_I_frac,
        "cascade_complete": bool(I[-1] < N * 0.01),
        "momentum_exhaustion_signal": bool(peak_t > 0 and np.argmax(I) < n_periods - 3),
    }
