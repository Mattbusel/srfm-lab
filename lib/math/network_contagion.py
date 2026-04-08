"""
network_contagion.py — Financial contagion modelling through network structure.

Covers: dynamic correlation networks, threshold graphs, spectral gap / Fiedler value,
DebtRank, contagion probability, fire-sale dynamics, interbank solvency cascade,
Systemic Risk Index, network entropy, and robustness score.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg
from scipy.sparse import csgraph
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CorrelationNetwork:
    """Snapshot of a dynamic correlation network."""
    assets: List[str]
    corr_matrix: np.ndarray        # (n, n) correlation matrix
    adj_matrix: np.ndarray         # (n, n) binary adjacency (thresholded)
    threshold: float
    n_edges: int
    density: float                 # edges / possible edges


@dataclass
class SpectralRiskResult:
    fiedler_value: float           # algebraic connectivity (λ₂ of Laplacian)
    spectral_gap: float            # λ₂ − λ₁ (λ₁ = 0 for connected graph)
    largest_eigenvalue: float      # spectral radius of adjacency
    fiedler_vector: np.ndarray     # corresponding eigenvector
    systemic_risk_score: float     # normalised [0, 1]; lower λ₂ → higher risk


@dataclass
class DebtRankResult:
    impact_vector: np.ndarray      # relative economic value lost at each node
    total_impact: float            # sum of impact_vector (DebtRank score)
    propagation_rounds: int
    shocked_node: int
    impact_by_round: List[np.ndarray]


@dataclass
class ContagionProbResult:
    prob_beyond_k: Dict[int, float]  # {k: P(shock spreads beyond k hops)}
    expected_reach: float
    shock_node: int
    transmission_prob: float


@dataclass
class FireSaleResult:
    price_path: np.ndarray         # (n_steps, n_assets) price evolution
    holding_path: np.ndarray       # (n_steps, n_assets) holdings evolution
    leverage_path: np.ndarray      # (n_steps, n_assets)
    liquidation_amounts: np.ndarray  # total sold per asset
    spiral_index: float            # price drop amplification factor
    converged: bool
    n_steps: int


@dataclass
class SolvencyCascadeResult:
    initial_capital: np.ndarray
    final_capital: np.ndarray
    defaulted_nodes: List[int]
    cascade_rounds: int
    loss_given_default: np.ndarray  # per-node fraction recovered by creditors
    total_system_loss: float


@dataclass
class NetworkRiskMetrics:
    systemic_risk_index: np.ndarray   # per-node SRI
    degree_centrality: np.ndarray
    eigenvector_centrality: np.ndarray
    betweenness_approx: np.ndarray    # approximate betweenness
    network_entropy: float
    robustness_score: float
    clustering_coefficient: np.ndarray


# ---------------------------------------------------------------------------
# 1. Dynamic Correlation Network
# ---------------------------------------------------------------------------

def rolling_correlation_network(returns: np.ndarray,
                                 assets: Optional[List[str]] = None,
                                 window: int = 60,
                                 threshold: float = 0.5) -> List[CorrelationNetwork]:
    """
    Build a sequence of correlation networks using a rolling window over returns.

    Parameters
    ----------
    returns   : (T, n_assets) return matrix
    assets    : asset names
    window    : rolling window length
    threshold : |corr| threshold for edge inclusion
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape
    if assets is None:
        assets = [f"asset_{i}" for i in range(n)]
    networks: List[CorrelationNetwork] = []
    for t in range(window, T + 1):
        chunk = returns[t - window:t]
        corr = np.corrcoef(chunk.T)
        corr = np.nan_to_num(corr, nan=0.0)
        adj = (np.abs(corr) > threshold).astype(float)
        np.fill_diagonal(adj, 0)
        n_edges = int(adj.sum()) // 2
        possible = n * (n - 1) // 2
        density = n_edges / possible if possible > 0 else 0.0
        networks.append(CorrelationNetwork(
            assets=list(assets),
            corr_matrix=corr.copy(),
            adj_matrix=adj.copy(),
            threshold=threshold,
            n_edges=n_edges,
            density=density))
    return networks


def threshold_graph(corr_matrix: np.ndarray, threshold: float = 0.5,
                    absolute: bool = True) -> np.ndarray:
    """
    Construct binary adjacency matrix from correlation matrix.

    Parameters
    ----------
    absolute : if True, edge when |corr| > threshold; else corr > threshold
    """
    corr = np.asarray(corr_matrix, dtype=float)
    if absolute:
        adj = (np.abs(corr) > threshold).astype(float)
    else:
        adj = (corr > threshold).astype(float)
    np.fill_diagonal(adj, 0)
    return adj


# ---------------------------------------------------------------------------
# 2. Spectral Gap / Fiedler Value
# ---------------------------------------------------------------------------

def spectral_risk(adj_matrix: np.ndarray) -> SpectralRiskResult:
    """
    Compute Laplacian spectral properties of the network.

    The Fiedler value (λ₂ of Laplacian) measures algebraic connectivity.
    Lower Fiedler value → graph is closer to disconnected → higher fragility.
    """
    A = np.asarray(adj_matrix, dtype=float)
    n = A.shape[0]
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A  # unnormalised Laplacian
    # Eigenvalues of L (all real, ≥ 0 for undirected)
    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L)))
    fiedler_value = float(eigenvalues[1]) if n > 1 else 0.0
    spectral_gap = float(eigenvalues[1] - eigenvalues[0]) if n > 1 else 0.0
    # Fiedler vector
    eigvals, eigvecs = np.linalg.eigh(L)
    idx_sorted = np.argsort(eigvals)
    fiedler_vector = eigvecs[:, idx_sorted[1]] if n > 1 else eigvecs[:, 0]
    # Adjacency spectral radius
    adj_eigvals = np.real(np.linalg.eigvals(A))
    largest_ev = float(np.max(np.abs(adj_eigvals)))
    # Normalise systemic risk: invert Fiedler, scale to [0, 1]
    max_possible_fiedler = n  # rough upper bound
    risk_score = 1.0 - min(fiedler_value / max_possible_fiedler, 1.0)
    return SpectralRiskResult(fiedler_value=fiedler_value,
                              spectral_gap=spectral_gap,
                              largest_eigenvalue=largest_ev,
                              fiedler_vector=fiedler_vector,
                              systemic_risk_score=risk_score)


# ---------------------------------------------------------------------------
# 3. DebtRank Algorithm
# ---------------------------------------------------------------------------

def debt_rank(exposure_matrix: np.ndarray,
              capital: np.ndarray,
              shocked_node: int,
              shock_fraction: float = 1.0,
              max_rounds: int = 100,
              tol: float = 1e-6) -> DebtRankResult:
    """
    Compute DebtRank: recursive measure of systemic impact when a node fails.

    Based on Battiston et al. (2012).

    Parameters
    ----------
    exposure_matrix : (n, n) matrix where E[i,j] = exposure of j to i
                      (fraction of i's liabilities owed to j)
    capital         : (n,) economic value (equity capital) at each node
    shocked_node    : index of initially distressed node
    shock_fraction  : initial distress fraction applied to shocked_node (0–1)
    """
    n = len(capital)
    E = np.asarray(exposure_matrix, dtype=float)
    V = np.asarray(capital, dtype=float)
    # Normalise: W[j,i] = fraction of j's capital lost if i defaults
    # W[j,i] = E[i,j] × V[i] / V[j]
    W = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if V[j] > 0:
                W[j, i] = E[i, j] * V[i] / V[j]
    # h[i]: distress level of node i ∈ [0, 1]
    h = np.zeros(n)
    h[shocked_node] = shock_fraction
    # status: 0 = undistressed, 1 = distressed, 2 = inactive (already propagated)
    status = np.zeros(n, dtype=int)
    status[shocked_node] = 1
    impact_by_round: List[np.ndarray] = [h.copy()]
    prev_h = h.copy()
    for rnd in range(max_rounds):
        h_new = h.copy()
        for i in range(n):
            if status[i] == 1:
                for j in range(n):
                    if status[j] == 0:
                        delta = W[j, i] * h[i]
                        h_new[j] = min(1.0, h_new[j] + delta)
                        if h_new[j] > 0:
                            status[j] = 1
                status[i] = 2  # mark as propagated
        delta_h = np.max(np.abs(h_new - prev_h))
        h = h_new
        impact_by_round.append(h.copy())
        prev_h = h.copy()
        if delta_h < tol:
            break
    # Impact vector: h[i] weighted by economic value
    impact_vector = h * V / (V.sum() + 1e-15)
    total_impact = float(impact_vector.sum())
    return DebtRankResult(impact_vector=impact_vector, total_impact=total_impact,
                          propagation_rounds=rnd + 1,
                          shocked_node=shocked_node,
                          impact_by_round=impact_by_round)


# ---------------------------------------------------------------------------
# 4. Contagion Probability
# ---------------------------------------------------------------------------

def contagion_probability(adj_matrix: np.ndarray,
                           shock_node: int,
                           transmission_prob: float = 0.3,
                           max_hops: int = 5,
                           n_simulations: int = 5000) -> ContagionProbResult:
    """
    Monte Carlo estimate of the probability that a shock at shock_node
    propagates beyond k hops via independent cascade model.
    """
    A = np.asarray(adj_matrix, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(42)
    reach_counts: Dict[int, int] = {k: 0 for k in range(1, max_hops + 1)}
    total_reaches: List[int] = []

    for _ in range(n_simulations):
        infected = {shock_node}
        frontier = {shock_node}
        hop = 0
        while frontier and hop < max_hops:
            new_frontier: set = set()
            for node in frontier:
                neighbors = np.where(A[node] > 0)[0]
                for nb in neighbors:
                    if nb not in infected:
                        if rng.random() < transmission_prob:
                            infected.add(nb)
                            new_frontier.add(nb)
            frontier = new_frontier
            hop += 1
        spread = len(infected) - 1  # exclude shock node
        total_reaches.append(spread)
        for k in range(1, max_hops + 1):
            if spread >= k:
                reach_counts[k] += 1

    prob_beyond_k = {k: reach_counts[k] / n_simulations for k in reach_counts}
    expected_reach = float(np.mean(total_reaches))
    return ContagionProbResult(prob_beyond_k=prob_beyond_k,
                               expected_reach=expected_reach,
                               shock_node=shock_node,
                               transmission_prob=transmission_prob)


# ---------------------------------------------------------------------------
# 5. Fire-Sale Dynamics
# ---------------------------------------------------------------------------

def fire_sale_dynamics(prices: np.ndarray,
                       holdings: np.ndarray,
                       leverage: np.ndarray,
                       leverage_target: float,
                       price_impact: float = 0.01,
                       max_steps: int = 200,
                       tol: float = 1e-6) -> FireSaleResult:
    """
    Simulate deleveraging fire-sale spiral.

    When leverage exceeds target, agents sell assets proportionally.
    Selling depresses prices, which increases leverage further → spiral.

    Parameters
    ----------
    prices        : (n_assets,) initial prices
    holdings      : (n_agents, n_assets) initial portfolio holdings
    leverage      : (n_agents,) initial leverage ratios
    leverage_target : target leverage ratio (agents deleverage towards this)
    price_impact  : fraction of price drop per unit of normalised selling pressure
    """
    prices = np.asarray(prices, dtype=float).copy()
    holdings = np.asarray(holdings, dtype=float).copy()
    leverage = np.asarray(leverage, dtype=float).copy()
    n_agents, n_assets = holdings.shape
    price_history = [prices.copy()]
    holding_history = [holdings.copy()]
    leverage_history = [leverage.copy()]
    liquidation_amounts = np.zeros(n_assets)
    initial_prices = prices.copy()
    converged = False

    for step in range(max_steps):
        excess = np.maximum(leverage - leverage_target, 0.0)
        if np.max(excess) < tol:
            converged = True
            break
        # Each agent sells proportionally to portfolio value per asset
        portfolio_values = holdings * prices[np.newaxis, :]  # (n_agents, n_assets)
        total_portfolio = portfolio_values.sum(axis=1, keepdims=True) + 1e-15
        sell_fractions = excess[:, np.newaxis] * portfolio_values / total_portfolio
        sold_qty = sell_fractions / (prices[np.newaxis, :] + 1e-15)
        # Aggregate selling pressure per asset
        total_sold = sold_qty.sum(axis=0)
        liquidation_amounts += total_sold
        # Price impact: price drops proportionally to selling volume / market cap
        market_cap = (holdings.sum(axis=0) * prices) + 1e-15
        price_drop = price_impact * total_sold * prices / market_cap
        prices = prices * (1.0 - price_drop)
        prices = np.maximum(prices, 1e-6)
        # Update holdings
        holdings = holdings - sold_qty
        holdings = np.maximum(holdings, 0.0)
        # Recalculate leverage: assets / equity
        asset_values = (holdings * prices[np.newaxis, :]).sum(axis=1)
        equity = asset_values / (leverage + 1e-15)  # approximate equity
        leverage = asset_values / (equity + 1e-15)
        price_history.append(prices.copy())
        holding_history.append(holdings.copy())
        leverage_history.append(leverage.copy())

    price_arr = np.array(price_history)
    holding_arr = np.array(holding_history)
    leverage_arr = np.array(leverage_history)
    price_drop_total = (initial_prices - prices) / (initial_prices + 1e-15)
    direct_price_drop = price_impact * liquidation_amounts / (initial_prices + 1e-15)
    spiral_index = float(np.mean(price_drop_total / (direct_price_drop + 1e-15)))
    return FireSaleResult(price_path=price_arr, holding_path=holding_arr,
                          leverage_path=leverage_arr,
                          liquidation_amounts=liquidation_amounts,
                          spiral_index=spiral_index, converged=converged,
                          n_steps=len(price_history) - 1)


# ---------------------------------------------------------------------------
# 6. Interbank Exposure Matrix & Solvency Cascade
# ---------------------------------------------------------------------------

def solvency_cascade(interbank_matrix: np.ndarray,
                     capital: np.ndarray,
                     external_assets: np.ndarray,
                     recovery_rate: float = 0.4,
                     initial_shock: Optional[np.ndarray] = None,
                     max_rounds: int = 50) -> SolvencyCascadeResult:
    """
    Simulate Eisenberg-Noe solvency cascade on interbank network.

    Parameters
    ----------
    interbank_matrix : (n, n) bilateral exposures: row i owes column j the amount
    capital          : (n,) equity capital of each bank
    external_assets  : (n,) external (non-interbank) assets
    recovery_rate    : fraction of assets recovered on default
    initial_shock    : (n,) initial haircut on external assets (fraction lost)
    """
    n = len(capital)
    liabilities = np.asarray(interbank_matrix, dtype=float)  # L[i,j]: i owes j
    cap = np.asarray(capital, dtype=float).copy()
    ext = np.asarray(external_assets, dtype=float).copy()
    if initial_shock is not None:
        shock = np.asarray(initial_shock, dtype=float)
        ext = ext * (1.0 - shock)
    # Total liabilities per bank
    total_liab = liabilities.sum(axis=1)
    # Claims: what each bank is owed by others
    claims = liabilities.sum(axis=0)
    defaulted: List[int] = []
    lgd = np.zeros(n)
    rounds = 0
    prev_defaulted = set()

    for rnd in range(max_rounds):
        newly_defaulted = []
        for i in range(n):
            if i in prev_defaulted:
                continue
            # Total assets = external + interbank claims (at recovery if creditor defaulted)
            interbank_receivables = 0.0
            for j in range(n):
                if j in prev_defaulted:
                    interbank_receivables += liabilities[j, i] * recovery_rate
                else:
                    interbank_receivables += liabilities[j, i]
            total_assets = ext[i] + interbank_receivables
            net_worth = total_assets - total_liab[i]
            if net_worth < 0:
                newly_defaulted.append(i)
                lgd[i] = max(0.0, -net_worth / (total_liab[i] + 1e-15))
                cap[i] = net_worth
        if not newly_defaulted:
            break
        for d in newly_defaulted:
            prev_defaulted.add(d)
            defaulted.append(d)
        rounds += 1

    initial_cap = np.asarray(capital, dtype=float)
    final_cap = np.maximum(cap, -initial_cap)  # bounded loss
    total_loss = float(np.sum(np.maximum(initial_cap - final_cap, 0.0)))
    return SolvencyCascadeResult(initial_capital=initial_cap,
                                 final_capital=cap,
                                 defaulted_nodes=defaulted,
                                 cascade_rounds=rounds,
                                 loss_given_default=lgd,
                                 total_system_loss=total_loss)


# ---------------------------------------------------------------------------
# 7. Systemic Risk Index (SRI)
# ---------------------------------------------------------------------------

def systemic_risk_index(adj_matrix: np.ndarray,
                         exposures: np.ndarray) -> np.ndarray:
    """
    SRI_i = eigenvector_centrality_i × total_exposure_i (normalised).

    Captures both network position and bilateral exposure size.

    Parameters
    ----------
    adj_matrix : (n, n) adjacency matrix
    exposures  : (n,) total interbank exposure per node
    """
    A = np.asarray(adj_matrix, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    n = A.shape[0]
    # Power iteration for leading eigenvector
    v = np.ones(n)
    for _ in range(200):
        v_new = A @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            break
        v_new /= norm
        if np.max(np.abs(v_new - v)) < 1e-10:
            v = v_new
            break
        v = v_new
    eig_centrality = np.abs(v)
    eig_centrality /= eig_centrality.sum() + 1e-15
    sri = eig_centrality * exposures
    sri /= sri.sum() + 1e-15
    return sri


# ---------------------------------------------------------------------------
# 8. Network Entropy
# ---------------------------------------------------------------------------

def network_entropy(adj_matrix: np.ndarray) -> float:
    """
    Compute network entropy based on degree distribution.

    H = -Σ p(k) log p(k) where p(k) = fraction of nodes with degree k.
    """
    A = np.asarray(adj_matrix, dtype=float)
    degrees = A.sum(axis=1).astype(int)
    max_deg = int(degrees.max()) if len(degrees) > 0 else 0
    if max_deg == 0:
        return 0.0
    counts = np.bincount(degrees, minlength=max_deg + 1)
    pk = counts / counts.sum()
    pk = pk[pk > 0]
    return float(-np.sum(pk * np.log(pk)))


# ---------------------------------------------------------------------------
# 9. Robustness Score
# ---------------------------------------------------------------------------

def robustness_score(adj_matrix: np.ndarray,
                     strategy: str = 'high_degree',
                     n_trials: int = 20) -> float:
    """
    Estimate robustness as the fraction of nodes that must be removed
    to disconnect the largest connected component to < 50% of original size.

    Parameters
    ----------
    strategy : 'high_degree' (targeted attack) or 'random'
    """
    A = np.asarray(adj_matrix, dtype=float).copy()
    n = A.shape[0]
    if n < 3:
        return 1.0

    def _lcc_fraction(mat: np.ndarray) -> float:
        # Largest connected component fraction using BFS
        visited = np.zeros(n, dtype=bool)
        max_size = 0
        for start in range(n):
            if not visited[start] and mat[start].sum() > 0:
                queue = [start]
                size = 0
                while queue:
                    node = queue.pop()
                    if visited[node]:
                        continue
                    visited[node] = True
                    size += 1
                    neighbors = np.where(mat[node] > 0)[0]
                    queue.extend(neighbors[~visited[neighbors]].tolist())
                max_size = max(max_size, size)
        return max_size / n

    if strategy == 'high_degree':
        mat = A.copy()
        for removed in range(1, n):
            degrees = mat.sum(axis=1)
            target = int(np.argmax(degrees))
            mat[target, :] = 0
            mat[:, target] = 0
            if _lcc_fraction(mat) < 0.5:
                return removed / n
        return 1.0
    else:  # random
        rng = np.random.default_rng(0)
        threshold_fracs = []
        for _ in range(n_trials):
            mat = A.copy()
            order = rng.permutation(n)
            for k, target in enumerate(order):
                mat[target, :] = 0
                mat[:, target] = 0
                if _lcc_fraction(mat) < 0.5:
                    threshold_fracs.append((k + 1) / n)
                    break
            else:
                threshold_fracs.append(1.0)
        return float(np.mean(threshold_fracs))


# ---------------------------------------------------------------------------
# 10. Full Network Risk Metrics
# ---------------------------------------------------------------------------

def compute_network_risk_metrics(adj_matrix: np.ndarray,
                                  exposures: Optional[np.ndarray] = None) -> NetworkRiskMetrics:
    """
    Compute the full suite of per-node and global network risk metrics.
    """
    A = np.asarray(adj_matrix, dtype=float)
    n = A.shape[0]
    if exposures is None:
        exposures = np.ones(n)
    exposures = np.asarray(exposures, dtype=float)

    # Degree centrality
    degree = A.sum(axis=1)
    degree_centrality = degree / (n - 1) if n > 1 else degree

    # Eigenvector centrality (power iteration)
    v = np.ones(n) / n
    for _ in range(300):
        v_new = A @ v
        nrm = np.linalg.norm(v_new)
        if nrm < 1e-12:
            break
        v_new /= nrm
        if np.max(np.abs(v_new - v)) < 1e-10:
            v = v_new
            break
        v = v_new
    eig_centrality = np.abs(v)
    eig_centrality /= eig_centrality.sum() + 1e-15

    # Approximate betweenness via random paths (cheap proxy)
    betweenness = np.zeros(n)
    rng = np.random.default_rng(1)
    sample_size = min(100, n * (n - 1))
    pairs = [(rng.integers(n), rng.integers(n)) for _ in range(sample_size)]
    for (s, t) in pairs:
        if s == t:
            continue
        # BFS to find shortest path from s to t
        parent = -np.ones(n, dtype=int)
        visited = np.zeros(n, dtype=bool)
        queue = [s]
        visited[s] = True
        found = False
        while queue and not found:
            node = queue.pop(0)
            for nb in np.where(A[node] > 0)[0]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = node
                    queue.append(nb)
                    if nb == t:
                        found = True
                        break
        if found:
            # Trace path and increment betweenness of intermediates
            node = t
            while parent[node] != s and parent[node] != -1:
                node = parent[node]
                betweenness[node] += 1.0

    betweenness /= sample_size + 1e-15

    # Clustering coefficient per node
    clustering = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
            continue
        # Count triangles through i
        triangles = 0
        for j in neighbors:
            for l in neighbors:
                if j != l and A[j, l] > 0:
                    triangles += 1
        clustering[i] = triangles / (k * (k - 1))

    sri = systemic_risk_index(A, exposures)
    ent = network_entropy(A)
    rob = robustness_score(A)

    return NetworkRiskMetrics(systemic_risk_index=sri,
                               degree_centrality=degree_centrality,
                               eigenvector_centrality=eig_centrality,
                               betweenness_approx=betweenness,
                               network_entropy=ent,
                               robustness_score=rob,
                               clustering_coefficient=clustering)
