"""
Mean Field Games (MFG) for financial markets.

MFG models rational agents interacting through aggregate (mean field) quantities.
Applications:
  - Optimal execution with many players
  - Systemic risk: contagion via mean field coupling
  - Herding equilibrium: agents couple through price impact
  - Nash equilibrium execution schedules
  - Price formation in competitive trading
  - Mean field optimal stopping (American options with competition)
  - Flocking model for correlated trading
  - MFG for market making

References: Carmona-Delarue (2018), Lasry-Lions (2007), Cardaliaguet (2010).
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


# ── MFG State Distribution ────────────────────────────────────────────────────

@dataclass
class MeanFieldState:
    """State of the mean field (distribution of agents)."""
    t: float
    mean_inventory: float    # average inventory across agents
    std_inventory: float     # dispersion of inventory
    mean_price: float        # current price level
    aggregate_flow: float    # total market order flow (sum of all agent actions)
    n_agents: int            # number of agents (for normalization)


# ── MFG Optimal Execution ─────────────────────────────────────────────────────

@dataclass
class MFGExecutionParams:
    """Parameters for MFG optimal execution."""
    X0: float                # initial inventory (same for all agents)
    T: float                 # horizon (days)
    sigma: float             # price volatility
    phi: float               # inventory penalty (running)
    A: float                 # terminal inventory penalty
    kappa: float             # temporary impact coefficient
    lambda_mf: float = 0.5   # mean field coupling (impact from others' trades)
    n_agents: int = 100      # number of competing agents


def mfg_optimal_execution_nash(
    params: MFGExecutionParams,
    n_steps: int = 50,
) -> dict:
    """
    Nash equilibrium execution in MFG with N competing agents.
    Each agent solves: min E[int (phi*X^2 + kappa*v^2) dt + A*X_T^2]
    subject to: dX = -v dt, dS = kappa*v dt + lambda_mf * V_bar dt + sigma*dW
    where V_bar = mean field (average trading rate).

    In symmetric Nash: all agents use same strategy.
    Solve via: Riccati ODE for value function.
    """
    p = params
    dt = p.T / n_steps
    t_grid = np.linspace(0, p.T, n_steps + 1)

    # Riccati ODE for MFG:
    # dP/dt = 2*phi - P^2/(kappa + lambda_mf*P_bar)  (approximately)
    # In symmetric case: P_bar = P

    # Initialize from terminal condition P(T) = 2*A
    P = np.zeros(n_steps + 1)
    P[-1] = 2 * p.A

    # Backward integrate
    for i in range(n_steps - 1, -1, -1):
        effective_kappa = max(p.kappa + p.lambda_mf * P[i+1] / p.n_agents, 1e-6)
        dP = 2 * p.phi - P[i+1]**2 / effective_kappa
        P[i] = P[i+1] - dP * dt  # backward Euler
        P[i] = max(P[i], 0)

    # Optimal trading rate: v*(t) = -P(t) * X(t) / (2*kappa)
    # Simulate inventory trajectory
    X = np.zeros(n_steps + 1)
    X[0] = p.X0
    V = np.zeros(n_steps)  # trading rates

    for i in range(n_steps):
        effective_kappa = max(p.kappa + p.lambda_mf * P[i] / p.n_agents, 1e-6)
        v = P[i] * X[i] / (2 * effective_kappa)  # optimal rate
        V[i] = v
        X[i+1] = max(X[i] - v * dt, 0)

    # Expected cost
    inventory_cost = float(p.phi * np.sum(X[:-1]**2) * dt)
    impact_cost = float(p.kappa * np.sum(V**2) * dt)
    terminal_cost = float(p.A * X[-1]**2)
    total_cost = inventory_cost + impact_cost + terminal_cost

    # MFG correction: cost increases with number of competing agents
    mfg_cost_premium = float(p.lambda_mf * np.sum(V**2) * dt / max(p.n_agents, 1))

    return {
        "inventory_schedule": X.tolist(),
        "trading_rates": V.tolist(),
        "t_grid": t_grid.tolist(),
        "total_expected_cost": float(total_cost),
        "inventory_cost": float(inventory_cost),
        "impact_cost": float(impact_cost),
        "terminal_cost": float(terminal_cost),
        "mfg_cost_premium": float(mfg_cost_premium),
        "riccati_solution": P.tolist(),
        "competitive_disadvantage_pct": float(mfg_cost_premium / max(total_cost, 1e-10) * 100),
    }


def mfg_price_impact_amplification(
    n_agents: int,
    individual_order: float,
    daily_volume: float,
    lambda_coupling: float = 0.5,
) -> dict:
    """
    MFG amplification factor: how much does competition amplify price impact?
    When N agents all try to execute same direction, impact is N * individual impact
    minus netting from competition.
    """
    total_demand = n_agents * individual_order
    participation = total_demand / max(daily_volume, 1e-10)

    # Without competition: each agent alone
    individual_impact_bps = float(math.sqrt(individual_order / max(daily_volume, 1e-10)) * 10000 * 0.1)

    # With competition (MFG Nash): amplification
    amplification = float(1 + lambda_coupling * math.log(max(n_agents, 1)))
    competitive_impact_bps = individual_impact_bps * amplification

    # Optimal delay: wait for others to reduce their impact
    optimal_wait_periods = int(math.ceil(math.log(n_agents) / 2)) if n_agents > 1 else 1

    return {
        "individual_impact_bps": float(individual_impact_bps),
        "competitive_impact_bps": float(competitive_impact_bps),
        "amplification_factor": float(amplification),
        "total_market_demand": float(total_demand),
        "participation_rate": float(participation),
        "optimal_wait_periods": optimal_wait_periods,
        "first_mover_advantage_bps": float(competitive_impact_bps - individual_impact_bps),
    }


# ── Herding Equilibrium ───────────────────────────────────────────────────────

def herding_equilibrium(
    n_agents: int = 100,
    sigma_private: float = 0.1,    # private signal noise
    price_sensitivity: float = 0.5,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    n_rounds: int = 20,
) -> dict:
    """
    Herding model: agents observe private signal + public price.
    Nash equilibrium: agents weight private vs public info.
    Banerjee/Bikhchandani herding cascade.
    """
    rng = np.random.default_rng(42)

    # True value
    true_value = rng.normal(prior_mean, prior_std)

    # Each agent gets noisy signal
    signals = rng.normal(true_value, sigma_private, n_agents)

    actions = []
    price_history = [prior_mean]
    current_public_belief = prior_mean
    public_precision = 1 / prior_std**2

    for t in range(n_rounds):
        round_actions = []
        for i in range(min(n_agents, 10)):  # process 10 agents per round
            agent_idx = (t * 10 + i) % n_agents
            private_signal = signals[agent_idx]
            private_precision = 1 / sigma_private**2

            # Optimal Bayesian weight: precision-weighted average
            posterior_mean = (private_signal * private_precision + current_public_belief * public_precision) / \
                             (private_precision + public_precision)

            # Action: buy if posterior > current price, else sell
            action = 1.0 if posterior_mean > price_history[-1] else -1.0
            round_actions.append(float(action))

        # Price update: respond to net order flow
        net_flow = float(np.mean(round_actions))
        new_price = price_history[-1] + price_sensitivity * net_flow
        price_history.append(new_price)
        actions.extend(round_actions)

        # Update public belief from price signal
        current_public_belief = new_price
        public_precision = min(public_precision + 0.1, 10.0)  # precision grows with price history

    # Herding measure: fraction of agents taking same action as majority
    if actions:
        action_arr = np.array(actions)
        majority = float(np.sign(action_arr.mean()))
        herding_frac = float(np.mean(action_arr == majority))
    else:
        herding_frac = 0.5

    return {
        "true_value": float(true_value),
        "final_price": float(price_history[-1]),
        "price_error": float(abs(price_history[-1] - true_value)),
        "herding_fraction": float(herding_frac),
        "price_history": price_history,
        "cascade_detected": bool(herding_frac > 0.8),
        "informationally_efficient": bool(abs(price_history[-1] - true_value) < sigma_private),
    }


# ── MFG Market Making ─────────────────────────────────────────────────────────

def mfg_market_maker_equilibrium(
    n_makers: int = 5,
    sigma: float = 0.02,       # underlying vol
    mu_order_arrival: float = 10.0,  # orders/period
    inventory_aversion: float = 0.01,
    adverse_selection: float = 0.3,
    n_periods: int = 100,
) -> dict:
    """
    MFG equilibrium for market making with N competing market makers.
    Avellaneda-Stoikov extended to N players.
    Equilibrium: each maker widens spread as N decreases (monopoly) or narrows as N -> inf.
    """
    # Single maker optimal spread (Avellaneda-Stoikov)
    T = n_periods / 252  # convert to years
    # bid_ask = gamma * sigma^2 * T + 2/gamma * log(1 + gamma/kappa)
    # Simplified: use Avellaneda formula
    gamma = inventory_aversion
    kappa_order = mu_order_arrival  # order arrival "intensity"

    single_maker_spread = float(
        gamma * sigma**2 * T +
        (2 / gamma) * math.log(1 + gamma / max(kappa_order, 1e-6))
    )

    # With N makers: competition narrows spread
    # Nash equilibrium spread ≈ single_spread / sqrt(n_makers)  (Cournot-like)
    equilibrium_spread = float(single_maker_spread / math.sqrt(n_makers))

    # Adverse selection component doesn't change with N makers
    adverse_component = float(2 * adverse_selection * sigma * math.sqrt(T))
    equilibrium_spread = max(equilibrium_spread, adverse_component)

    # Simulate inventory paths for representative maker
    inventories = np.zeros(n_periods)
    pnl = np.zeros(n_periods)
    rng = np.random.default_rng(42)

    for t in range(1, n_periods):
        # Order arrivals (Poisson)
        buy_orders = rng.poisson(mu_order_arrival / 2 / n_makers)
        sell_orders = rng.poisson(mu_order_arrival / 2 / n_makers)

        net_flow = float(buy_orders - sell_orders)
        inventories[t] = inventories[t-1] + net_flow

        # P&L: spread capture - inventory risk
        spread_pnl = (buy_orders + sell_orders) * equilibrium_spread / 2
        inventory_risk = gamma * inventories[t]**2 * sigma**2
        pnl[t] = pnl[t-1] + spread_pnl - inventory_risk

    return {
        "single_maker_spread": float(single_maker_spread),
        "equilibrium_spread": float(equilibrium_spread),
        "adverse_selection_component": float(adverse_component),
        "n_makers": n_makers,
        "competition_discount": float(1 - equilibrium_spread / max(single_maker_spread, 1e-10)),
        "avg_inventory": float(abs(inventories).mean()),
        "pnl_series": pnl.tolist(),
        "total_pnl": float(pnl[-1]),
        "sharpe": float(pnl.mean() / max(pnl.std(), 1e-10) * math.sqrt(252)),
    }


# ── Systemic Risk MFG ──────────────────────────────────────────────────────────

def systemic_risk_mean_field(
    n_banks: int = 50,
    initial_capital_mean: float = 1.0,
    initial_capital_std: float = 0.2,
    interbank_exposure: float = 0.1,    # fraction of capital in interbank
    recovery_rate: float = 0.4,
    fire_sale_elasticity: float = 2.0,
    n_steps: int = 50,
) -> dict:
    """
    MFG model of systemic risk: bank capital dynamics with mean-field coupling.
    When average capital falls (mean field), each bank faces correlated losses.
    dX_i = mu_i * dt - lambda * (X_bar - X_i) * dt + sigma * dW_i
    Default: X_i < 0
    """
    rng = np.random.default_rng(42)
    dt = 1.0 / n_steps

    # Initialize capital
    capital = rng.normal(initial_capital_mean, initial_capital_std, n_banks)
    capital = np.maximum(capital, 0.1)

    # Track defaults
    n_defaults = np.zeros(n_steps)
    mean_capital = np.zeros(n_steps)
    capital_history = [capital.copy()]

    for t in range(n_steps):
        alive = capital > 0
        x_bar = float(capital[alive].mean()) if alive.any() else 0.0
        mean_capital[t] = x_bar

        # Shocks
        idio_shock = rng.normal(0, 0.05, n_banks) * math.sqrt(dt)
        systemic_shock = rng.normal(0, 0.03) * math.sqrt(dt)  # common shock

        # Mean field dynamics: pull toward mean
        mf_pull = 0.5 * (x_bar - capital) * dt

        # Fire sale: when mean capital drops, assets devalue
        if x_bar < initial_capital_mean * 0.8:
            fire_sale_loss = fire_sale_elasticity * (initial_capital_mean - x_bar) * 0.01
        else:
            fire_sale_loss = 0.0

        # Interbank contagion: defaults cause losses to connected banks
        default_frac = float((~alive).sum() / max(n_banks, 1))
        contagion_loss = interbank_exposure * default_frac * (1 - recovery_rate)

        capital = capital + mf_pull + idio_shock + systemic_shock - fire_sale_loss - contagion_loss
        capital = np.where(alive, capital, 0.0)  # once dead, stays dead

        n_defaults[t] = float((capital <= 0).sum())
        capital_history.append(capital.copy())

    final_alive = (capital > 0).sum()
    total_defaults = n_banks - final_alive

    return {
        "n_defaults_final": int(total_defaults),
        "default_rate": float(total_defaults / n_banks),
        "mean_capital_path": mean_capital.tolist(),
        "cumulative_defaults": n_defaults.tolist(),
        "systemic_event": bool(total_defaults > n_banks * 0.1),
        "capital_distribution_final": {
            "mean": float(capital[capital > 0].mean()) if (capital > 0).any() else 0.0,
            "std": float(capital[capital > 0].std()) if (capital > 0).any() else 0.0,
            "min": float(capital[capital > 0].min()) if (capital > 0).any() else 0.0,
        },
        "cascade_wave": int(np.argmax(np.diff(np.concatenate([[0], n_defaults]))) + 1),
    }


# ── Flocking / Correlated Trading ─────────────────────────────────────────────

def vicsek_trading_model(
    n_traders: int = 50,
    speed: float = 1.0,         # position change per step
    coupling_radius: float = 5.0,
    noise: float = 0.3,
    n_steps: int = 100,
) -> dict:
    """
    Vicsek flocking model applied to trader behavior.
    Traders align their 'trading direction' with nearby peers.
    High alignment = herding = correlated positions.
    """
    rng = np.random.default_rng(42)

    # Initialize: random positions (portfolio composition) and angles (trading direction)
    positions = rng.uniform(0, 10, (n_traders, 2))
    angles = rng.uniform(-math.pi, math.pi, n_traders)

    order_params = np.zeros(n_steps)  # Vicsek order parameter (herding measure)
    position_history = []

    for t in range(n_steps):
        # Compute pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))

        # New angles: average of neighbors within radius + noise
        new_angles = np.zeros(n_traders)
        for i in range(n_traders):
            neighbors = np.where(dist[i] < coupling_radius)[0]
            if len(neighbors) > 0:
                avg_sin = float(np.sin(angles[neighbors]).mean())
                avg_cos = float(np.cos(angles[neighbors]).mean())
                avg_angle = math.atan2(avg_sin, avg_cos)
            else:
                avg_angle = angles[i]
            new_angles[i] = avg_angle + rng.uniform(-noise, noise)

        angles = new_angles

        # Update positions
        positions[:, 0] += speed * np.cos(angles)
        positions[:, 1] += speed * np.sin(angles)

        # Order parameter: |<e^{i*theta}>| = alignment measure
        op = float(abs(np.exp(1j * angles).mean()))
        order_params[t] = op

        if t % 10 == 0:
            position_history.append(positions.copy())

    current_alignment = float(order_params[-1])
    return {
        "order_parameter": order_params.tolist(),
        "current_alignment": current_alignment,
        "herding_regime": current_alignment > 0.7,
        "dispersed_regime": current_alignment < 0.3,
        "avg_alignment": float(order_params.mean()),
        "n_traders": n_traders,
        "coupling_radius": coupling_radius,
    }


# ── MFG Signal Generator ──────────────────────────────────────────────────────

class MeanFieldSignal:
    """
    Derives trading signals from mean field game theory.
    Monitors when individual agent is misaligned with equilibrium.
    """

    def __init__(self, n_agents_est: int = 100, lambda_coupling: float = 0.5):
        self.n_agents = n_agents_est
        self.lambda_coupling = lambda_coupling
        self._flow_history: list[float] = []

    def update_flow(self, aggregate_flow: float) -> None:
        self._flow_history.append(aggregate_flow)

    def equilibrium_deviation(self, current_inventory: float) -> dict:
        """
        How far is current inventory from MFG equilibrium?
        Equilibrium: hold average inventory = 0 (for liquidity provision).
        """
        if not self._flow_history:
            return {"deviation": 0.0, "signal": 0.0}

        avg_flow = float(np.mean(self._flow_history[-20:]))
        expected_inventory_share = avg_flow / max(self.n_agents, 1)
        deviation = current_inventory - expected_inventory_share

        # Signal: if deviation > threshold, adjust position toward equilibrium
        signal = float(-np.tanh(deviation * 2))  # mean-revert to equilibrium

        return {
            "deviation": float(deviation),
            "signal": float(signal),
            "expected_inventory": float(expected_inventory_share),
            "equilibrium_distance": float(abs(deviation)),
            "n_agents_est": self.n_agents,
        }

    def competitive_execution_cost(
        self,
        order_size: float,
        daily_volume: float,
    ) -> dict:
        """Estimate execution cost accounting for other agents."""
        return mfg_price_impact_amplification(
            self.n_agents,
            order_size,
            daily_volume,
            self.lambda_coupling,
        )
