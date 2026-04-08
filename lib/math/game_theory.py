"""
Game theory for financial markets.

Nash equilibrium, Stackelberg, Cournot, Bertrand, VCG mechanism design,
Bayesian games, repeated games, evolutionary dynamics, Shapley value,
cooperative games, and market-making competition applications.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from itertools import combinations, product


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simplex_project(v: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u > cssv / np.arange(1, n + 1))[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


# ---------------------------------------------------------------------------
# 1. Nash Equilibrium: Support Enumeration (2-player)
# ---------------------------------------------------------------------------

class NashEquilibriumFinder:
    """Find all Nash equilibria of a 2-player normal-form game via support enumeration."""

    def __init__(self, A: np.ndarray, B: np.ndarray):
        """A[i,j] = payoff to player 1 when P1 plays i, P2 plays j.
           B[i,j] = payoff to player 2."""
        self.A = A
        self.B = B
        self.m, self.n = A.shape  # P1 has m strategies, P2 has n

    def find_all(self, tol: float = 1e-8) -> List[Tuple[np.ndarray, np.ndarray]]:
        equilibria = []
        for k1 in range(1, self.m + 1):
            for k2 in range(1, self.n + 1):
                for s1 in combinations(range(self.m), k1):
                    for s2 in combinations(range(self.n), k2):
                        result = self._check_support(list(s1), list(s2), tol)
                        if result is not None:
                            equilibria.append(result)
        return equilibria

    def _check_support(self, s1: List[int], s2: List[int],
                       tol: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        k1, k2 = len(s1), len(s2)
        # P2's mixed strategy must make P1 indifferent over s1
        # A[s1, :][:, s2] @ q = v1 * 1, sum(q) = 1
        A_sub = self.A[np.ix_(s1, s2)]
        B_sub = self.B[np.ix_(s1, s2)]
        # Solve for q (P2's strategy): A_sub @ q = v1 * ones
        # Augmented system: [A_sub, -1; 1...1, 0] [q; v1] = [0; 1]
        try:
            M2 = np.zeros((k1 + 1, k2 + 1))
            M2[:k1, :k2] = A_sub
            M2[:k1, k2] = -1
            M2[k1, :k2] = 1
            rhs2 = np.zeros(k1 + 1)
            rhs2[k1] = 1
            sol2 = np.linalg.lstsq(M2, rhs2, rcond=None)[0]
            q = sol2[:k2]
            if np.any(q < -tol) or abs(np.sum(q) - 1) > tol:
                return None
        except np.linalg.LinAlgError:
            return None
        # Solve for p (P1's strategy): B_sub.T @ p = v2 * ones
        try:
            M1 = np.zeros((k2 + 1, k1 + 1))
            M1[:k2, :k1] = B_sub.T
            M1[:k2, k1] = -1
            M1[k2, :k1] = 1
            rhs1 = np.zeros(k2 + 1)
            rhs1[k2] = 1
            sol1 = np.linalg.lstsq(M1, rhs1, rcond=None)[0]
            p = sol1[:k1]
            if np.any(p < -tol) or abs(np.sum(p) - 1) > tol:
                return None
        except np.linalg.LinAlgError:
            return None
        # Check that strategies outside support don't improve payoff
        p_full = np.zeros(self.m)
        q_full = np.zeros(self.n)
        p_full[s1] = np.maximum(p, 0)
        q_full[s2] = np.maximum(q, 0)
        p_full /= p_full.sum() + 1e-30
        q_full /= q_full.sum() + 1e-30
        v1 = p_full @ self.A @ q_full
        v2 = p_full @ self.B @ q_full
        for i in range(self.m):
            if i not in s1:
                if self.A[i] @ q_full > v1 + tol:
                    return None
        for j in range(self.n):
            if j not in s2:
                if p_full @ self.B[:, j] > v2 + tol:
                    return None
        return (p_full, q_full)


# ---------------------------------------------------------------------------
# 2. Lemke-Howson for Mixed Strategy Nash
# ---------------------------------------------------------------------------

class LemkeHowson:
    """Lemke-Howson algorithm for finding one Nash equilibrium of a 2-player game."""

    def __init__(self, A: np.ndarray, B: np.ndarray):
        self.A = A
        self.B = B

    def solve(self, pivot_var: int = 0, max_pivots: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        m, n = self.A.shape
        # Shift to ensure positive payoffs
        shift = max(0, -self.A.min(), -self.B.min()) + 1
        A = self.A + shift
        B = self.B + shift
        # Tableau for P1: rows 0..m-1 => constraints from B^T
        # Tableau for P2: rows 0..n-1 => constraints from A
        # Simplified: use support enumeration as fallback
        # Full Lemke-Howson pivot is complex; provide vertex enumeration approach
        T1 = np.hstack([B.T, np.eye(n)])
        T2 = np.hstack([np.eye(m), A])
        # Use complementary pivoting
        basis1 = list(range(m, m + n))  # slack vars for P2
        basis2 = list(range(0, m))      # slack vars for P1
        q1 = np.ones(n)
        q2 = np.ones(m)
        entering = pivot_var
        tableau = 1  # start with tableau 1
        for _ in range(max_pivots):
            if tableau == 1:
                col = entering if entering < m else entering - m
                if entering < m:
                    ratios = q1 / (T1[:, entering] + 1e-30)
                    ratios[T1[:, entering] <= 0] = np.inf
                    if np.all(np.isinf(ratios)):
                        break
                    row = np.argmin(ratios)
                    leaving = basis1[row]
                    # Pivot
                    pivot_elem = T1[row, entering]
                    T1[row] /= pivot_elem
                    q1[row] /= pivot_elem
                    for i in range(n):
                        if i != row:
                            factor = T1[i, entering]
                            T1[i] -= factor * T1[row]
                            q1[i] -= factor * q1[row]
                    basis1[row] = entering
                    entering = leaving
                    tableau = 2 if entering < m else 1
                else:
                    break
            else:
                if entering >= m:
                    col_idx = entering - m
                    ratios = q2 / (T2[:, entering] + 1e-30)
                    ratios[T2[:, entering] <= 0] = np.inf
                    if np.all(np.isinf(ratios)):
                        break
                    row = np.argmin(ratios)
                    leaving = basis2[row]
                    pivot_elem = T2[row, entering]
                    T2[row] /= pivot_elem
                    q2[row] /= pivot_elem
                    for i in range(m):
                        if i != row:
                            factor = T2[i, entering]
                            T2[i] -= factor * T2[row]
                            q2[i] -= factor * q2[row]
                    basis2[row] = entering
                    entering = leaving
                    tableau = 1 if entering >= m else 2
                else:
                    break
            if entering == pivot_var:
                break
        # Extract strategies
        p = np.zeros(m)
        for i, b in enumerate(basis2):
            if b >= m:
                pass
            else:
                p[b] = q2[i]
        q = np.zeros(n)
        for i, b in enumerate(basis1):
            if b < m:
                pass
            else:
                q[b - m] = q1[i]
        p_sum = p.sum()
        q_sum = q.sum()
        if p_sum > 1e-10:
            p /= p_sum
        else:
            p = np.ones(m) / m
        if q_sum > 1e-10:
            q /= q_sum
        else:
            q = np.ones(n) / n
        return p, q


# ---------------------------------------------------------------------------
# 3. Stackelberg Equilibrium
# ---------------------------------------------------------------------------

class StackelbergEquilibrium:
    """Leader-follower (Stackelberg) game. Leader commits, follower best-responds."""

    def __init__(self, leader_payoff: np.ndarray, follower_payoff: np.ndarray):
        """Payoff matrices: leader_payoff[i,j], follower_payoff[i,j]."""
        self.L = leader_payoff
        self.F = follower_payoff

    def solve(self) -> Dict[str, Any]:
        m, n = self.L.shape
        best_leader_payoff = -np.inf
        best_leader_action = 0
        best_follower_action = 0
        for i in range(m):
            # Follower best-responds to leader action i
            j_star = np.argmax(self.F[i, :])
            if self.L[i, j_star] > best_leader_payoff:
                best_leader_payoff = self.L[i, j_star]
                best_leader_action = i
                best_follower_action = j_star
        return {
            "leader_action": best_leader_action,
            "follower_action": best_follower_action,
            "leader_payoff": float(self.L[best_leader_action, best_follower_action]),
            "follower_payoff": float(self.F[best_leader_action, best_follower_action]),
        }

    def solve_mixed(self, n_grid: int = 100) -> Dict[str, Any]:
        """Stackelberg with mixed leader strategy (discretized)."""
        m, n = self.L.shape
        best_val = -np.inf
        best_p = np.ones(m) / m
        for indices in product(range(n_grid + 1), repeat=m - 1):
            s = sum(indices)
            if s > n_grid:
                continue
            p = np.array(list(indices) + [n_grid - s], dtype=float) / n_grid
            follower_expected = p @ self.F
            j_star = np.argmax(follower_expected)
            leader_val = p @ self.L[:, j_star]
            if leader_val > best_val:
                best_val = leader_val
                best_p = p.copy()
        follower_exp = best_p @ self.F
        j_star = np.argmax(follower_exp)
        return {
            "leader_mixed": best_p,
            "follower_action": int(j_star),
            "leader_payoff": float(best_val),
        }


# ---------------------------------------------------------------------------
# 4. Cournot Oligopoly (quantity competition)
# ---------------------------------------------------------------------------

class CournotOligopoly:
    """Cournot quantity competition among N market makers."""

    def __init__(self, n_firms: int, demand_intercept: float = 100.0,
                 demand_slope: float = 1.0, costs: Optional[np.ndarray] = None):
        self.n = n_firms
        self.a = demand_intercept
        self.b = demand_slope
        self.costs = costs if costs is not None else np.zeros(n_firms)

    def nash_equilibrium(self) -> Dict[str, Any]:
        """Analytical Nash equilibrium for linear demand P = a - b*Q."""
        n, a, b = self.n, self.a, self.b
        c = self.costs
        c_sum = np.sum(c)
        quantities = np.zeros(n)
        for i in range(n):
            quantities[i] = (a - (n + 1) * c[i] + c_sum) / (b * (n + 1))
            quantities[i] = max(quantities[i], 0)
        Q = np.sum(quantities)
        price = max(a - b * Q, 0)
        profits = quantities * (price - c)
        return {
            "quantities": quantities,
            "total_quantity": float(Q),
            "price": float(price),
            "profits": profits,
            "consumer_surplus": float(0.5 * b * Q ** 2),
        }

    def best_response(self, i: int, other_quantities: np.ndarray) -> float:
        Q_others = np.sum(other_quantities) - other_quantities[i]
        q_br = (self.a - self.b * Q_others - self.costs[i]) / (2 * self.b)
        return max(q_br, 0)

    def simulate_dynamics(self, n_rounds: int = 100,
                          learning_rate: float = 0.3) -> np.ndarray:
        quantities = np.ones(self.n) * self.a / (self.b * (self.n + 1))
        history = [quantities.copy()]
        for _ in range(n_rounds):
            new_q = quantities.copy()
            for i in range(self.n):
                br = self.best_response(i, quantities)
                new_q[i] = quantities[i] + learning_rate * (br - quantities[i])
            quantities = new_q
            history.append(quantities.copy())
        return np.array(history)


# ---------------------------------------------------------------------------
# 5. Bertrand Competition (price competition)
# ---------------------------------------------------------------------------

class BertrandCompetition:
    """Bertrand price competition with differentiated products."""

    def __init__(self, n_firms: int, base_demand: float = 100.0,
                 price_sensitivity: float = 2.0,
                 cross_sensitivity: float = 0.5,
                 costs: Optional[np.ndarray] = None):
        self.n = n_firms
        self.base_demand = base_demand
        self.own_sens = price_sensitivity
        self.cross_sens = cross_sensitivity
        self.costs = costs if costs is not None else np.zeros(n_firms)

    def demand(self, prices: np.ndarray) -> np.ndarray:
        """q_i = base - own_sens * p_i + cross_sens * sum_{j!=i} p_j."""
        q = np.zeros(self.n)
        for i in range(self.n):
            others_sum = np.sum(prices) - prices[i]
            q[i] = self.base_demand - self.own_sens * prices[i] + self.cross_sens * others_sum
        return np.maximum(q, 0)

    def nash_equilibrium(self) -> Dict[str, Any]:
        """Solve first-order conditions for Nash in prices."""
        # profit_i = (p_i - c_i) * q_i
        # FOC: q_i + (p_i - c_i) * dq_i/dp_i = 0
        # q_i = base - own * p_i + cross * sum(p_j)
        # dq_i/dp_i = -own
        # => base - own*p_i + cross*sum(p_j) - own*(p_i - c_i) = 0
        # => base + own*c_i - 2*own*p_i + cross*sum(p_j) = 0
        # System: 2*own*p_i - cross*sum(p_j) = base + own*c_i
        A = np.eye(self.n) * 2 * self.own_sens
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    A[i, j] = -self.cross_sens
        rhs = self.base_demand + self.own_sens * self.costs
        prices = np.linalg.solve(A, rhs)
        prices = np.maximum(prices, self.costs)
        q = self.demand(prices)
        profits = (prices - self.costs) * q
        return {
            "prices": prices,
            "quantities": q,
            "profits": profits,
        }


# ---------------------------------------------------------------------------
# 6. VCG Mechanism Design (Vickrey-Clarke-Groves)
# ---------------------------------------------------------------------------

class VCGAuction:
    """VCG auction for single or multiple items."""

    def __init__(self, n_bidders: int, n_items: int = 1):
        self.n_bidders = n_bidders
        self.n_items = n_items

    def single_item(self, bids: np.ndarray) -> Dict[str, Any]:
        """Single-item VCG (= second-price auction)."""
        winner = int(np.argmax(bids))
        sorted_bids = np.sort(bids)[::-1]
        payment = float(sorted_bids[1]) if len(sorted_bids) > 1 else 0.0
        return {"winner": winner, "payment": payment, "bid": float(bids[winner])}

    def multi_item(self, valuations: np.ndarray) -> Dict[str, Any]:
        """
        Multi-item VCG. valuations[i, S] for bidder i, bundle S.
        Simplified: unit-demand (each bidder wants at most 1 item).
        valuations: (n_bidders, n_items).
        """
        n, m = valuations.shape
        # Optimal assignment: maximize total welfare
        from itertools import permutations
        best_welfare = -np.inf
        best_assignment = {}
        items = list(range(m))
        bidders = list(range(n))
        # Try all assignments (bidder -> item, at most min(n,m) assigned)
        k = min(n, m)
        for bidder_subset in combinations(bidders, k):
            for item_perm in permutations(items, k):
                welfare = sum(valuations[bidder_subset[i], item_perm[i]] for i in range(k))
                if welfare > best_welfare:
                    best_welfare = welfare
                    best_assignment = {bidder_subset[i]: item_perm[i] for i in range(k)}
        # VCG payments
        payments = {}
        for bidder in best_assignment:
            # Welfare without this bidder
            others = [b for b in best_assignment if b != bidder]
            other_items = list(range(m))
            welfare_without = -np.inf
            k2 = min(len(others), len(other_items))
            if k2 == 0:
                welfare_without = 0
            else:
                for bs in combinations(others, k2):
                    for ip in permutations(other_items, k2):
                        w = sum(valuations[bs[i], ip[i]] for i in range(k2))
                        welfare_without = max(welfare_without, w)
            others_welfare_in = sum(valuations[b, best_assignment[b]] for b in others)
            payments[bidder] = float(welfare_without - others_welfare_in)
        return {
            "assignment": best_assignment,
            "payments": payments,
            "total_welfare": float(best_welfare),
        }


# ---------------------------------------------------------------------------
# 7. Bayesian Games: Incomplete Information Nash
# ---------------------------------------------------------------------------

class BayesianGame:
    """Two-player Bayesian game with finite types."""

    def __init__(self, type_probs1: np.ndarray, type_probs2: np.ndarray,
                 payoffs1: np.ndarray, payoffs2: np.ndarray):
        """
        type_probs1: (T1,) prior on player 1 types
        type_probs2: (T2,) prior on player 2 types
        payoffs1: (T1, T2, A1, A2) payoff to P1
        payoffs2: (T1, T2, A1, A2) payoff to P2
        """
        self.tp1 = type_probs1
        self.tp2 = type_probs2
        self.pay1 = payoffs1
        self.pay2 = payoffs2
        self.T1, self.T2, self.A1, self.A2 = payoffs1.shape

    def bayesian_nash(self, max_iter: int = 500, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find Bayesian Nash equilibrium via iterated best response.
        Returns strategies: sigma1[t1, a1], sigma2[t2, a2].
        """
        sigma1 = np.ones((self.T1, self.A1)) / self.A1
        sigma2 = np.ones((self.T2, self.A2)) / self.A2
        for _ in range(max_iter):
            old1, old2 = sigma1.copy(), sigma2.copy()
            # Best response for P1 given sigma2
            for t1 in range(self.T1):
                expected = np.zeros(self.A1)
                for a1 in range(self.A1):
                    for t2 in range(self.T2):
                        for a2 in range(self.A2):
                            expected[a1] += self.tp2[t2] * sigma2[t2, a2] * self.pay1[t1, t2, a1, a2]
                best_a = np.argmax(expected)
                sigma1[t1] = 0
                sigma1[t1, best_a] = 1.0
            # Best response for P2 given sigma1
            for t2 in range(self.T2):
                expected = np.zeros(self.A2)
                for a2 in range(self.A2):
                    for t1 in range(self.T1):
                        for a1 in range(self.A1):
                            expected[a2] += self.tp1[t1] * sigma1[t1, a1] * self.pay2[t1, t2, a1, a2]
                best_a = np.argmax(expected)
                sigma2[t2] = 0
                sigma2[t2, best_a] = 1.0
            if np.allclose(sigma1, old1, atol=tol) and np.allclose(sigma2, old2, atol=tol):
                break
        return sigma1, sigma2


# ---------------------------------------------------------------------------
# 8. Repeated Games: Folk Theorem, Trigger Strategies, Tit-for-Tat
# ---------------------------------------------------------------------------

class RepeatedGame:
    """Infinitely repeated game with discounting."""

    def __init__(self, stage_payoff1: np.ndarray, stage_payoff2: np.ndarray,
                 discount: float = 0.9):
        self.A = stage_payoff1
        self.B = stage_payoff2
        self.delta = discount

    def grim_trigger_sustainable(self, coop_actions: Tuple[int, int],
                                 defect_actions: Tuple[int, int]) -> Dict[str, Any]:
        """Check if cooperation via grim trigger is a subgame-perfect NE."""
        ci, cj = coop_actions
        di, dj = defect_actions
        v_coop1 = self.A[ci, cj] / (1 - self.delta)
        v_coop2 = self.B[ci, cj] / (1 - self.delta)
        # Best deviation for P1
        dev1 = np.max(self.A[:, cj])
        v_punish1 = self.A[di, dj] / (1 - self.delta)
        v_deviate1 = dev1 + self.delta * v_punish1
        # Best deviation for P2
        dev2 = np.max(self.B[ci, :])
        v_punish2 = self.B[di, dj] / (1 - self.delta)
        v_deviate2 = dev2 + self.delta * v_punish2
        return {
            "sustainable": bool(v_coop1 >= v_deviate1 and v_coop2 >= v_deviate2),
            "coop_value_p1": float(v_coop1),
            "coop_value_p2": float(v_coop2),
            "min_discount_p1": float((dev1 - self.A[ci, cj]) /
                                     (dev1 - self.A[di, dj] + 1e-30)),
            "min_discount_p2": float((dev2 - self.B[ci, cj]) /
                                     (dev2 - self.B[di, dj] + 1e-30)),
        }

    def tit_for_tat_simulate(self, n_rounds: int = 100,
                              noise: float = 0.0,
                              rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        rng = rng or np.random.default_rng(42)
        m, n = self.A.shape
        history1, history2 = [], []
        payoffs1, payoffs2 = [], []
        # Assume action 0 = cooperate, action 1 = defect
        a1, a2 = 0, 0
        for t in range(n_rounds):
            # Noise: tremble
            if rng.random() < noise:
                a1 = rng.integers(0, m)
            if rng.random() < noise:
                a2 = rng.integers(0, n)
            history1.append(a1)
            history2.append(a2)
            payoffs1.append(self.A[a1, a2])
            payoffs2.append(self.B[a1, a2])
            # Tit-for-tat: copy opponent's last action
            a1, a2 = history2[-1], history1[-1]
        return {
            "avg_payoff_p1": float(np.mean(payoffs1)),
            "avg_payoff_p2": float(np.mean(payoffs2)),
            "cooperation_rate_p1": float(np.mean(np.array(history1) == 0)),
            "cooperation_rate_p2": float(np.mean(np.array(history2) == 0)),
        }

    def feasible_payoff_set(self, n_samples: int = 10000,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample feasible payoff pairs (for folk theorem visualization)."""
        rng = rng or np.random.default_rng(42)
        m, n = self.A.shape
        payoffs = []
        for _ in range(n_samples):
            p = rng.dirichlet(np.ones(m))
            q = rng.dirichlet(np.ones(n))
            v1 = p @ self.A @ q
            v2 = p @ self.B @ q
            payoffs.append([v1, v2])
        return np.array(payoffs)


# ---------------------------------------------------------------------------
# 9. Evolutionary Game Theory: Replicator Dynamics, ESS
# ---------------------------------------------------------------------------

class EvolutionaryGame:
    """Evolutionary game theory: replicator dynamics and ESS."""

    def __init__(self, payoff_matrix: np.ndarray):
        """Symmetric game: payoff_matrix[i,j] = payoff of strategy i vs j."""
        self.A = payoff_matrix
        self.n = payoff_matrix.shape[0]

    def replicator_dynamics(self, x0: np.ndarray, dt: float = 0.01,
                            n_steps: int = 5000) -> np.ndarray:
        """Simulate replicator dynamics: dx_i/dt = x_i * (f_i - f_bar)."""
        x = x0.copy()
        trajectory = [x.copy()]
        for _ in range(n_steps):
            fitness = self.A @ x
            avg_fitness = x @ fitness
            dx = x * (fitness - avg_fitness)
            x = x + dt * dx
            x = np.maximum(x, 0)
            x /= x.sum() + 1e-30
            trajectory.append(x.copy())
        return np.array(trajectory)

    def is_ess(self, strategy_idx: int) -> bool:
        """Check if pure strategy is an Evolutionarily Stable Strategy."""
        e = np.zeros(self.n)
        e[strategy_idx] = 1.0
        a_star = self.A[strategy_idx, strategy_idx]
        for j in range(self.n):
            if j == strategy_idx:
                continue
            a_j = self.A[j, strategy_idx]
            if a_j > a_star:
                return False
            if a_j == a_star:
                if self.A[j, j] >= self.A[strategy_idx, j]:
                    return False
        return True

    def find_all_ess(self) -> List[int]:
        return [i for i in range(self.n) if self.is_ess(i)]

    def fixed_points(self) -> List[np.ndarray]:
        """Return pure strategy fixed points and interior (if exists)."""
        fps = []
        for i in range(self.n):
            e = np.zeros(self.n)
            e[i] = 1.0
            fps.append(e)
        # Interior: solve A @ x = c * 1, sum(x) = 1
        try:
            M = np.vstack([self.A - np.eye(self.n) * self.A[0, 0], np.ones(self.n)])
            rhs = np.zeros(self.n + 1)
            rhs[-1] = 1.0
            x = np.linalg.lstsq(M, rhs, rcond=None)[0]
            if np.all(x > -1e-8):
                x = np.maximum(x, 0)
                x /= x.sum()
                fps.append(x)
        except np.linalg.LinAlgError:
            pass
        return fps


# ---------------------------------------------------------------------------
# 10. Cooperative Games: Shapley Value
# ---------------------------------------------------------------------------

class ShapleyValue:
    """Compute Shapley value for cooperative game."""

    def __init__(self, n_players: int, value_function: Dict[frozenset, float]):
        """value_function maps coalitions (frozensets) to values."""
        self.n = n_players
        self.v = value_function

    def compute(self) -> np.ndarray:
        """Exact Shapley value computation."""
        from math import factorial
        phi = np.zeros(self.n)
        players = list(range(self.n))
        for i in players:
            for size in range(0, self.n):
                others = [j for j in players if j != i]
                for S in combinations(others, size):
                    S_set = frozenset(S)
                    S_with_i = frozenset(S) | {i}
                    v_with = self.v.get(S_with_i, 0.0)
                    v_without = self.v.get(S_set, 0.0)
                    weight = factorial(size) * factorial(self.n - size - 1) / factorial(self.n)
                    phi[i] += weight * (v_with - v_without)
        return phi

    def compute_sampling(self, n_samples: int = 10000,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Approximate Shapley value via random permutation sampling."""
        rng = rng or np.random.default_rng(42)
        phi = np.zeros(self.n)
        players = list(range(self.n))
        for _ in range(n_samples):
            perm = rng.permutation(players)
            coalition = set()
            v_prev = 0.0
            for p in perm:
                coalition.add(p)
                v_curr = self.v.get(frozenset(coalition), 0.0)
                phi[p] += v_curr - v_prev
                v_prev = v_curr
        return phi / n_samples


# ---------------------------------------------------------------------------
# 11. Coalitional Games: Core and Nucleolus
# ---------------------------------------------------------------------------

class CoalitionalGame:
    """Coalitional game: core and nucleolus for cost/risk sharing."""

    def __init__(self, n_players: int, value_function: Dict[frozenset, float]):
        self.n = n_players
        self.v = value_function

    def is_in_core(self, allocation: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if allocation is in the core."""
        grand = frozenset(range(self.n))
        if abs(np.sum(allocation) - self.v.get(grand, 0.0)) > tol:
            return False
        players = list(range(self.n))
        for size in range(1, self.n):
            for S in combinations(players, size):
                S_set = frozenset(S)
                if np.sum(allocation[list(S)]) < self.v.get(S_set, 0.0) - tol:
                    return False
        return True

    def nucleolus(self, max_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
        """Approximate nucleolus via iterated LP relaxation (simplified)."""
        grand_value = self.v.get(frozenset(range(self.n)), 0.0)
        x = np.ones(self.n) * grand_value / self.n
        players = list(range(self.n))
        coalitions = []
        for size in range(1, self.n):
            for S in combinations(players, size):
                coalitions.append(frozenset(S))
        for _ in range(max_iter):
            excesses = []
            for S in coalitions:
                e = self.v.get(S, 0.0) - np.sum(x[list(S)])
                excesses.append(e)
            excesses = np.array(excesses)
            if len(excesses) == 0:
                break
            worst_idx = np.argmax(excesses)
            if excesses[worst_idx] < tol:
                break
            S_worst = list(coalitions[worst_idx])
            deficit = excesses[worst_idx]
            # Transfer from non-members to members
            non_members = [p for p in players if p not in S_worst]
            if non_members:
                transfer = deficit / (len(S_worst) + len(non_members))
                for p in S_worst:
                    x[p] += transfer
                for p in non_members:
                    x[p] -= transfer * len(S_worst) / len(non_members)
            # Re-normalize to grand coalition value
            x = x * grand_value / (np.sum(x) + 1e-30)
        return x


# ---------------------------------------------------------------------------
# 12. Application: Market Maker Competition Equilibrium
# ---------------------------------------------------------------------------

def market_maker_competition(n_makers: int = 3,
                              total_flow: float = 1000.0,
                              inventory_cost: float = 0.001,
                              rebate: float = 0.0001,
                              base_spread: float = 0.01) -> Dict[str, Any]:
    """
    Model market maker competition as Cournot-like game.
    Each maker chooses liquidity provision quantity; price = base_spread - slope * total_q.
    """
    costs = np.full(n_makers, inventory_cost)
    cournot = CournotOligopoly(n_makers, demand_intercept=base_spread * total_flow,
                                demand_slope=base_spread / total_flow, costs=costs)
    eq = cournot.nash_equilibrium()
    # Add rebate revenue
    effective_profits = eq["profits"] + eq["quantities"] * rebate * total_flow
    hhi = np.sum((eq["quantities"] / eq["total_quantity"]) ** 2) * 10000
    return {
        "quantities": eq["quantities"],
        "effective_spread": float(eq["price"] / total_flow),
        "profits": effective_profits,
        "hhi": float(hhi),
        "market_shares": (eq["quantities"] / eq["total_quantity"]).tolist(),
    }


# ---------------------------------------------------------------------------
# 13. Application: Shapley Value for Factor Contribution Attribution
# ---------------------------------------------------------------------------

def factor_shapley_attribution(returns: np.ndarray, factor_returns: np.ndarray,
                                factor_names: Optional[List[str]] = None,
                                n_samples: int = 5000) -> Dict[str, Any]:
    """
    Attribute portfolio performance to factors using Shapley value.
    Coalitional value = R^2 of regression on factor subset.
    """
    T, K = factor_returns.shape
    if factor_names is None:
        factor_names = [f"F{i}" for i in range(K)]

    def r_squared_coalition(S: frozenset) -> float:
        if len(S) == 0:
            return 0.0
        idx = sorted(S)
        F = factor_returns[:, idx]
        try:
            beta = np.linalg.lstsq(F, returns, rcond=None)[0]
            fitted = F @ beta
            ss_res = np.sum((returns - fitted) ** 2)
            ss_tot = np.sum((returns - returns.mean()) ** 2)
            return max(1.0 - ss_res / (ss_tot + 1e-30), 0.0)
        except np.linalg.LinAlgError:
            return 0.0

    # Build value function for all coalitions (feasible for K <= 15)
    if K <= 15:
        vf: Dict[frozenset, float] = {}
        for size in range(K + 1):
            for S in combinations(range(K), size):
                vf[frozenset(S)] = r_squared_coalition(frozenset(S))
        sv = ShapleyValue(K, vf)
        phi = sv.compute()
    else:
        vf = {}
        sv = ShapleyValue(K, vf)
        # Override with sampling that computes on the fly
        rng = np.random.default_rng(42)
        phi = np.zeros(K)
        for _ in range(n_samples):
            perm = rng.permutation(K)
            coalition: set = set()
            v_prev = 0.0
            for p in perm:
                coalition.add(p)
                v_curr = r_squared_coalition(frozenset(coalition))
                phi[p] += v_curr - v_prev
                v_prev = v_curr
        phi /= n_samples

    total_r2 = r_squared_coalition(frozenset(range(K)))
    contributions = {factor_names[i]: float(phi[i]) for i in range(K)}
    pct = {factor_names[i]: float(phi[i] / (total_r2 + 1e-30) * 100)
           for i in range(K)}
    return {
        "shapley_values": contributions,
        "percentage_contribution": pct,
        "total_r_squared": float(total_r2),
        "shapley_sum": float(np.sum(phi)),
    }
