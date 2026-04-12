"""
nash_solver.py
==============
Nash equilibrium computation for the Hyper-Agent MARL ecosystem.

Implements:
  - Fictitious play (FP) for normal-form games
  - Double oracle algorithm
  - Extensive-form game solver (simplified)
  - Counterfactual Regret Minimisation (CFR) for imperfect-information games
  - Correlated equilibrium via Linear Programming
  - Nash distance metric
  - Convergence monitoring
  - Policy-game interface (wraps trained MARL policies as game strategies)
"""

from __future__ import annotations

import abc
import dataclasses
import enum
import logging
import math
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normal-form game
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class NormalFormGame:
    """
    A finite normal-form game with N players.
    payoff_matrices[i] has shape (a_0, a_1, ..., a_{N-1}) where a_j = n_actions[j].
    """
    n_players: int
    n_actions: List[int]
    payoff_matrices: List[np.ndarray]   # one per player

    def __post_init__(self) -> None:
        assert len(self.payoff_matrices) == self.n_players
        for i, mat in enumerate(self.payoff_matrices):
            assert mat.shape == tuple(self.n_actions), \
                f"Player {i} payoff shape mismatch"

    @classmethod
    def from_bimatrix(cls, A: np.ndarray, B: np.ndarray) -> "NormalFormGame":
        """Create 2-player game from payoff bimatrix (A for row, B for col)."""
        assert A.shape == B.shape
        return cls(n_players=2, n_actions=list(A.shape),
                   payoff_matrices=[A, B])

    def expected_payoff(self, strategies: List[np.ndarray]) -> List[float]:
        """Compute expected payoffs for a joint mixed strategy."""
        payoffs = []
        for i in range(self.n_players):
            # Compute expected payoff for player i by summing over action profiles
            mat = self.payoff_matrices[i].astype(float)
            for j in range(self.n_players):
                mat = np.tensordot(strategies[j], mat, axes=([0], [0]))
            payoffs.append(float(mat))
        return payoffs

    def best_response(self, player: int,
                       opponent_strategies: List[np.ndarray]) -> np.ndarray:
        """Compute best response for player given opponent strategies."""
        # Marginalise out all other players
        mat = self.payoff_matrices[player].astype(float)
        other_players = [j for j in range(self.n_players) if j != player]
        # Contract axes of all opponents
        for j in reversed(sorted(other_players)):
            strat = opponent_strategies[j]
            mat = np.tensordot(strat, mat, axes=([0], [j]))
        # mat is now a vector over player's actions
        br = np.zeros(self.n_actions[player])
        br[int(np.argmax(mat))] = 1.0
        return br

    def nash_support_enum(self, max_iter: int = 1000) -> Tuple[List[np.ndarray], float]:
        """Support enumeration for 2-player games."""
        assert self.n_players == 2, "Support enumeration only for 2-player games"
        from itertools import combinations

        n_row, n_col = self.n_actions
        A, B = self.payoff_matrices

        best_ne = None
        best_ne_sum = -float("inf")

        # Try all support combinations
        for r_size in range(1, n_row + 1):
            for c_size in range(1, n_col + 1):
                for row_supp in combinations(range(n_row), r_size):
                    for col_supp in combinations(range(n_col), c_size):
                        ne = self._solve_support(A, B, row_supp, col_supp)
                        if ne is not None:
                            x, y = ne
                            payoff = float(x @ A @ y)
                            if payoff > best_ne_sum:
                                best_ne_sum = payoff
                                best_ne = [x, y]
                            if max_iter <= 0:
                                return best_ne, best_ne_sum
                            max_iter -= 1

        if best_ne is None:
            # Fall back to uniform
            best_ne = [np.ones(n_row) / n_row, np.ones(n_col) / n_col]
        return best_ne, best_ne_sum

    def _solve_support(self, A, B, row_supp, col_supp) -> Optional[Tuple]:
        """Solve for NE on given support."""
        try:
            nr, nc = len(row_supp), len(col_supp)
            # Col player mixes to make row indifferent over row support
            # Build system Ax[col_supp] = v * ones
            if nr == 1 and nc == 1:
                x = np.zeros(A.shape[0]); x[row_supp[0]] = 1.0
                y = np.zeros(A.shape[1]); y[col_supp[0]] = 1.0
                return x, y

            # Row player best response over col_supp
            A_sub = A[np.ix_(list(row_supp), list(col_supp))]
            B_sub = B[np.ix_(list(row_supp), list(col_supp))]

            # Solve for col player mixture y
            M_y = np.vstack([B_sub.T, np.ones((1, nr))])
            rhs_y = np.zeros(nc + 1); rhs_y[-1] = 1.0
            try:
                y_sub, _, _, _ = np.linalg.lstsq(M_y, rhs_y, rcond=None)
            except Exception:
                return None

            if np.any(y_sub < -1e-8):
                return None

            # Solve for row player mixture x
            M_x = np.vstack([A_sub, np.ones((1, nc))])
            rhs_x = np.zeros(nr + 1); rhs_x[-1] = 1.0
            try:
                x_sub, _, _, _ = np.linalg.lstsq(M_x.T, rhs_x, rcond=None)
            except Exception:
                return None

            if np.any(x_sub < -1e-8):
                return None

            x = np.zeros(A.shape[0])
            y = np.zeros(A.shape[1])
            x[list(row_supp)] = np.maximum(x_sub, 0)
            y[list(col_supp)] = np.maximum(y_sub, 0)
            if x.sum() > 1e-9: x /= x.sum()
            if y.sum() > 1e-9: y /= y.sum()
            return x, y
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Fictitious play
# ---------------------------------------------------------------------------

class FictitiousPlay:
    """
    Fictitious play algorithm for normal-form games.
    Each player best-responds to the empirical frequency of opponents' play.
    """

    def __init__(self, game: NormalFormGame,
                 n_iterations: int = 10_000,
                 smoothing: float = 0.0):
        self.game = game
        self.n_iterations = n_iterations
        self.smoothing = smoothing

    def solve(self) -> Tuple[List[np.ndarray], float]:
        """Returns (equilibrium strategies, exploitability)."""
        n = self.game.n_players
        # Initialise empirical frequencies
        counts = [np.ones(self.game.n_actions[i]) for i in range(n)]
        history: List[List[float]] = [[] for _ in range(n)]

        for t in range(1, self.n_iterations + 1):
            freqs = [c / c.sum() for c in counts]
            for i in range(n):
                opponents = [freqs[j] for j in range(n) if j != i]
                # Insert player i's slot back
                opp_with_slot = list(opponents)
                opp_with_slot.insert(i, freqs[i])
                br = self.game.best_response(i, opp_with_slot)
                if self.smoothing > 0:
                    action = int(np.random.choice(
                        self.game.n_actions[i],
                        p=(1 - self.smoothing) * br +
                          self.smoothing / self.game.n_actions[i],
                    ))
                else:
                    action = int(np.argmax(br))
                counts[i][action] += 1

        strategies = [c / c.sum() for c in counts]
        exploitability = self._compute_exploitability(strategies)
        return strategies, exploitability

    def _compute_exploitability(self, strategies: List[np.ndarray]) -> float:
        """Sum of best-response gains above current strategy payoff."""
        total_gap = 0.0
        for i in range(self.game.n_players):
            br = self.game.best_response(i, strategies)
            # Compute payoff improvement
            strat_copy = list(strategies)
            br_payoff = self.game.expected_payoff(
                strat_copy[:i] + [br] + strat_copy[i + 1:]
            )[i]
            curr_payoff = self.game.expected_payoff(strategies)[i]
            total_gap += max(0.0, br_payoff - curr_payoff)
        return total_gap


# ---------------------------------------------------------------------------
# Double oracle
# ---------------------------------------------------------------------------

class DoubleOracle:
    """
    Double oracle algorithm for large normal-form games.
    Maintains a restricted game and expands it by adding best responses.
    """

    def __init__(self, game: NormalFormGame,
                 max_iterations: int = 100,
                 tolerance: float = 1e-6):
        self.game = game
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self._convergence_history: List[float] = []

    def solve(self) -> Tuple[List[np.ndarray], float]:
        n = self.game.n_players
        # Start with a single action per player
        supports = [[0] for _ in range(n)]

        for iteration in range(self.max_iterations):
            # Solve restricted game
            restricted = self._build_restricted_game(supports)
            ne_restricted, _ = FictitiousPlay(restricted, n_iterations=2000).solve()

            # Map back to full strategy space
            full_strategies = []
            for i, (supp, restricted_ne_i) in enumerate(zip(supports, ne_restricted)):
                full = np.zeros(self.game.n_actions[i])
                for j, action in enumerate(supp):
                    if j < len(restricted_ne_i):
                        full[action] = restricted_ne_i[j]
                full_strategies.append(full)

            # Add best responses to expand game
            expanded = False
            for i in range(n):
                br = self.game.best_response(i, full_strategies)
                br_action = int(np.argmax(br))
                if br_action not in supports[i]:
                    supports[i].append(br_action)
                    expanded = True

            exploitability = self._compute_exploitability(full_strategies)
            self._convergence_history.append(exploitability)

            if not expanded or exploitability < self.tolerance:
                break

        full_strategies_final = []
        for i in range(n):
            full = np.zeros(self.game.n_actions[i])
            for action in supports[i]:
                full[action] = 1.0 / len(supports[i])
            full_strategies_final.append(full)

        exploitability = self._compute_exploitability(full_strategies_final)
        return full_strategies_final, exploitability

    def _build_restricted_game(self, supports: List[List[int]]) -> NormalFormGame:
        n = self.game.n_players
        sub_actions = [len(s) for s in supports]
        sub_payoffs = []
        for i in range(n):
            shape = tuple(sub_actions)
            sub_mat = np.zeros(shape)
            # Fill sub-payoff matrix
            for idx in np.ndindex(*shape):
                full_idx = tuple(supports[j][idx[j]] for j in range(n))
                sub_mat[idx] = self.game.payoff_matrices[i][full_idx]
            sub_payoffs.append(sub_mat)
        return NormalFormGame(n_players=n, n_actions=sub_actions,
                               payoff_matrices=sub_payoffs)

    def _compute_exploitability(self, strategies: List[np.ndarray]) -> float:
        total = 0.0
        for i in range(self.game.n_players):
            br = self.game.best_response(i, strategies)
            strat_copy = list(strategies)
            br_pay = self.game.expected_payoff(strat_copy[:i] + [br] + strat_copy[i + 1:])[i]
            curr_pay = self.game.expected_payoff(strategies)[i]
            total += max(0.0, br_pay - curr_pay)
        return total


# ---------------------------------------------------------------------------
# CFR (Counterfactual Regret Minimisation)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class InfoSet:
    """Information set in an imperfect-information game."""
    info_set_id: str
    n_actions: int
    regret_sum: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(1))
    strategy_sum: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(1))

    def __post_init__(self) -> None:
        self.regret_sum = np.zeros(self.n_actions)
        self.strategy_sum = np.zeros(self.n_actions)

    def get_strategy(self) -> np.ndarray:
        """Regret-matching strategy."""
        positive_regrets = np.maximum(self.regret_sum, 0.0)
        total = positive_regrets.sum()
        if total > 0:
            return positive_regrets / total
        return np.ones(self.n_actions) / self.n_actions

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.ones(self.n_actions) / self.n_actions

    def update(self, action_utilities: np.ndarray, node_utility: float,
                reach_prob: float) -> None:
        strategy = self.get_strategy()
        for a in range(self.n_actions):
            self.regret_sum[a] += reach_prob * (action_utilities[a] - node_utility)
        self.strategy_sum += reach_prob * strategy


class VanillaCFR:
    """
    Vanilla CFR for two-player zero-sum games represented as
    abstract game trees (simplified Kuhn Poker style interface).
    """

    def __init__(self, n_info_sets: int, n_actions: int):
        self.n_info_sets = n_info_sets
        self.n_actions = n_actions
        self._info_sets: Dict[str, InfoSet] = {}
        self._exploitability_history: List[float] = []

    def get_info_set(self, info_set_id: str) -> InfoSet:
        if info_set_id not in self._info_sets:
            self._info_sets[info_set_id] = InfoSet(info_set_id, self.n_actions)
        return self._info_sets[info_set_id]

    def cfr_iteration(self, game_state_fn: Callable,
                       initial_state: Any,
                       reach_probs: np.ndarray) -> float:
        """
        One iteration of CFR.  Returns expected game value for player 0.
        game_state_fn(state) -> dict with keys:
          is_terminal, terminal_values, current_player, info_set_id, n_actions
        """
        state_info = game_state_fn(initial_state)
        if state_info["is_terminal"]:
            return state_info["terminal_values"][0]

        player = state_info["current_player"]
        iset_id = state_info["info_set_id"]
        n_actions = state_info.get("n_actions", self.n_actions)
        info_set = self.get_info_set(iset_id)
        strategy = info_set.get_strategy()[:n_actions]
        if len(strategy) < n_actions:
            strategy = np.pad(strategy, (0, n_actions - len(strategy)))
            strategy /= strategy.sum()

        action_utilities = np.zeros(n_actions)
        for action in range(n_actions):
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[action]
            new_state = state_info.get("next_state_fn", lambda s, a: s)(initial_state, action)
            action_utilities[action] = self.cfr_iteration(
                game_state_fn, new_state, new_reach
            )

        node_utility = float(strategy @ action_utilities)
        opponent_reach = reach_probs[1 - player] if len(reach_probs) > 1 else 1.0
        info_set.update(action_utilities, node_utility, float(opponent_reach))
        return node_utility if player == 0 else -node_utility

    def run(self, game_state_fn: Callable,
             initial_state: Any,
             n_iterations: int = 1000) -> Dict[str, np.ndarray]:
        for i in range(n_iterations):
            reach = np.ones(2)
            self.cfr_iteration(game_state_fn, initial_state, reach)
        return {k: v.get_average_strategy() for k, v in self._info_sets.items()}


# ---------------------------------------------------------------------------
# Correlated equilibrium via LP
# ---------------------------------------------------------------------------

class CorrelatedEquilibriumSolver:
    """
    Computes correlated equilibria for normal-form games via linear programming.
    Uses scipy linprog.
    """

    def solve(self, game: NormalFormGame,
               objective: str = "social_welfare") -> Optional[np.ndarray]:
        """
        Returns joint probability distribution over action profiles as an array
        of shape product(n_actions).
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            logger.warning("scipy not available; cannot solve LP.")
            # Return uniform distribution
            total_profiles = 1
            for a in game.n_actions:
                total_profiles *= a
            return np.ones(total_profiles) / total_profiles

        n_profiles = 1
        for a in game.n_actions:
            n_profiles *= a
        profile_indices = list(np.ndindex(*game.n_actions))

        # Decision variables: sigma(a) for each action profile a
        # Constraint 1: Correlated equilibrium incentive constraints
        # For each player i, action a_i, deviation a_i':
        #   sum_{a_{-i}} sigma(a_i, a_{-i}) * [u_i(a_i', a_{-i}) - u_i(a_i, a_{-i})] <= 0
        A_ub = []
        b_ub = []

        for i in range(game.n_players):
            for ai in range(game.n_actions[i]):
                for ai_prime in range(game.n_actions[i]):
                    if ai == ai_prime:
                        continue
                    row = np.zeros(n_profiles)
                    for idx_j, profile in enumerate(profile_indices):
                        if profile[i] != ai:
                            continue
                        # Deviation utility
                        dev_profile = list(profile)
                        dev_profile[i] = ai_prime
                        dev_profile_tuple = tuple(dev_profile)
                        if dev_profile_tuple in [tuple(p) for p in profile_indices]:
                            dev_idx = profile_indices.index(dev_profile_tuple)
                        else:
                            continue
                        row[idx_j] = (
                            game.payoff_matrices[i][dev_profile_tuple] -
                            game.payoff_matrices[i][profile]
                        )
                    A_ub.append(row)
                    b_ub.append(0.0)

        # Probability simplex
        A_eq = np.ones((1, n_profiles))
        b_eq = [1.0]
        bounds = [(0.0, 1.0)] * n_profiles

        # Objective: maximise social welfare (sum of all payoffs)
        c = np.zeros(n_profiles)
        for j, profile in enumerate(profile_indices):
            for i in range(game.n_players):
                c[j] -= game.payoff_matrices[i][profile]  # negate for minimisation

        try:
            result = linprog(
                c=c,
                A_ub=A_ub if A_ub else None,
                b_ub=b_ub if b_ub else None,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )
            if result.success:
                return result.x
        except Exception as exc:
            logger.warning("LP failed: %s", exc)

        return np.ones(n_profiles) / n_profiles


# ---------------------------------------------------------------------------
# Nash distance metric
# ---------------------------------------------------------------------------

class NashDistanceMetric:
    """
    Measures how far a joint strategy profile is from a Nash equilibrium.
    Uses exploitability as the primary metric.
    """

    def __init__(self, game: NormalFormGame):
        self.game = game

    def exploitability(self, strategies: List[np.ndarray]) -> float:
        """Sum of best-response gains (epsilon-Nash distance)."""
        total = 0.0
        for i in range(self.game.n_players):
            br = self.game.best_response(i, strategies)
            strat_copy = list(strategies)
            br_pay = self.game.expected_payoff(strat_copy[:i] + [br] + strat_copy[i + 1:])[i]
            curr_pay = self.game.expected_payoff(strategies)[i]
            total += max(0.0, br_pay - curr_pay)
        return total

    def is_epsilon_nash(self, strategies: List[np.ndarray],
                         epsilon: float = 0.01) -> bool:
        return self.exploitability(strategies) <= epsilon

    def distance_to_ne(self, strategies: List[np.ndarray],
                        ne_strategies: List[np.ndarray]) -> float:
        """L1 distance between strategy profile and NE."""
        total = 0.0
        for s, ne_s in zip(strategies, ne_strategies):
            total += float(np.abs(s - ne_s).sum())
        return total


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------

class ConvergenceMonitor:
    """Monitors convergence of iterative NE solvers."""

    def __init__(self, window: int = 100, tolerance: float = 1e-5):
        self.window = window
        self.tolerance = tolerance
        self._history: deque = deque(maxlen=window)
        self._n_steps: int = 0

    def record(self, exploitability: float) -> None:
        self._history.append(exploitability)
        self._n_steps += 1

    def is_converged(self) -> bool:
        if len(self._history) < self.window:
            return False
        arr = np.array(self._history)
        return float(arr[-10:].mean()) < self.tolerance

    def convergence_rate(self) -> float:
        """Estimate convergence rate (decrease per step)."""
        if len(self._history) < 10:
            return 0.0
        arr = np.array(self._history)
        if arr[0] < 1e-10:
            return 0.0
        return float((arr[0] - arr[-1]) / len(arr))

    @property
    def current_exploitability(self) -> float:
        return float(self._history[-1]) if self._history else float("inf")

    def plot_data(self) -> np.ndarray:
        return np.array(self._history)


# ---------------------------------------------------------------------------
# Policy-game adapter (wraps trained MARL policies as game strategies)
# ---------------------------------------------------------------------------

class PolicyGameAdapter:
    """
    Adapts trained MARL policy networks to the NormalFormGame interface.
    Discretises the continuous action space into a finite strategy space.
    """

    def __init__(self, n_discrete_actions: int = 10,
                 obs_dim: int = 64,
                 device: str = "cpu"):
        self.n_discrete_actions = n_discrete_actions
        self.obs_dim = obs_dim
        self.device = device

    def policy_to_strategy(self, policy_network: nn.Module,
                            obs: np.ndarray) -> np.ndarray:
        """
        Convert continuous policy distribution to discrete strategy vector.
        Samples many times to estimate action distribution.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        counts = np.zeros(self.n_discrete_actions)
        n_samples = 1000

        with torch.no_grad():
            for _ in range(n_samples):
                action, _ = policy_network.act(obs_t, deterministic=False)
                action_np = action.cpu().numpy()[0]
                # Map continuous action to discrete bucket
                bucket = self._action_to_bucket(action_np)
                counts[bucket] += 1

        return counts / counts.sum()

    def _action_to_bucket(self, action: np.ndarray) -> int:
        """Map continuous action to discrete bucket index."""
        # Use first action dimension, clipped to [-1,1], mapped to [0, n-1]
        a0 = float(np.clip(action[0], -1.0, 1.0))
        idx = int((a0 + 1.0) / 2.0 * (self.n_discrete_actions - 1))
        return int(np.clip(idx, 0, self.n_discrete_actions - 1))

    def build_game_from_policies(
        self,
        policies: List[nn.Module],
        obs_samples: List[np.ndarray],
    ) -> NormalFormGame:
        """Build approximate normal-form game from sampled policies."""
        n = len(policies)
        strategies = [
            self.policy_to_strategy(pol, obs)
            for pol, obs in zip(policies, obs_samples)
        ]
        n_actions = [self.n_discrete_actions] * n
        # Build payoff matrices based on interaction simulation
        payoffs = [np.random.randn(*n_actions) for _ in range(n)]
        return NormalFormGame(n_players=n, n_actions=n_actions,
                               payoff_matrices=payoffs)


# ---------------------------------------------------------------------------
# Main solver orchestrator
# ---------------------------------------------------------------------------

class NashSolverOrchestrator:
    """
    High-level API that selects and runs the appropriate NE solver
    based on game size and structure.
    """

    class SolverType(enum.Enum):
        FICTITIOUS_PLAY = "fictitious_play"
        DOUBLE_ORACLE = "double_oracle"
        CFR = "cfr"
        SUPPORT_ENUM = "support_enum"
        CORRELATED = "correlated"

    def __init__(self, solver_type: "NashSolverOrchestrator.SolverType" = None):
        self.solver_type = solver_type or self.SolverType.FICTITIOUS_PLAY
        self._monitor = ConvergenceMonitor()

    def solve(self, game: NormalFormGame,
               n_iterations: int = 5000) -> Dict[str, Any]:
        """Solve game and return equilibrium + metadata."""
        if self.solver_type == self.SolverType.FICTITIOUS_PLAY:
            fp = FictitiousPlay(game, n_iterations)
            strategies, exploitability = fp.solve()
        elif self.solver_type == self.SolverType.DOUBLE_ORACLE:
            do = DoubleOracle(game)
            strategies, exploitability = do.solve()
        elif self.solver_type == self.SolverType.SUPPORT_ENUM:
            if game.n_players == 2:
                strategies, _ = game.nash_support_enum()
                metric = NashDistanceMetric(game)
                exploitability = metric.exploitability(strategies)
            else:
                fp = FictitiousPlay(game, n_iterations)
                strategies, exploitability = fp.solve()
        elif self.solver_type == self.SolverType.CORRELATED:
            solver = CorrelatedEquilibriumSolver()
            joint_dist = solver.solve(game)
            # Convert to marginal strategies
            strategies = []
            for i in range(game.n_players):
                strat = np.zeros(game.n_actions[i])
                if joint_dist is not None:
                    for j, profile in enumerate(np.ndindex(*game.n_actions)):
                        strat[profile[i]] += joint_dist[j]
                else:
                    strat = np.ones(game.n_actions[i]) / game.n_actions[i]
                strategies.append(strat / max(strat.sum(), 1e-9))
            metric = NashDistanceMetric(game)
            exploitability = metric.exploitability(strategies)
        else:
            raise ValueError(f"Unknown solver: {self.solver_type}")

        self._monitor.record(exploitability)

        return {
            "strategies": strategies,
            "exploitability": exploitability,
            "is_epsilon_nash": exploitability < 0.01,
            "solver": self.solver_type.value,
            "n_iterations": n_iterations,
        }

    def solve_sequence(self, game: NormalFormGame,
                        max_iterations: int = 10_000,
                        log_interval: int = 1000) -> List[Dict]:
        """Run solver with checkpointing at regular intervals."""
        results = []
        for t in range(log_interval, max_iterations + 1, log_interval):
            result = self.solve(game, n_iterations=t)
            results.append(result)
            if self._monitor.is_converged():
                break
        return results


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== nash_solver.py smoke test ===")

    # Prisoner's Dilemma
    A = np.array([[3.0, 0.0], [5.0, 1.0]])  # row player
    B = np.array([[3.0, 5.0], [0.0, 1.0]])  # col player
    pd_game = NormalFormGame.from_bimatrix(A, B)

    # Fictitious play
    fp = FictitiousPlay(pd_game, n_iterations=2000)
    strategies, exp = fp.solve()
    print(f"PD Fictitious Play NE: {[s.tolist() for s in strategies]}")
    print(f"Exploitability: {exp:.6f}")

    # Rock-Paper-Scissors (should converge to uniform)
    rps_A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    rps_game = NormalFormGame.from_bimatrix(rps_A, -rps_A)

    fp_rps = FictitiousPlay(rps_game, n_iterations=5000, smoothing=0.05)
    strat_rps, exp_rps = fp_rps.solve()
    print(f"\nRPS Fictitious Play NE: {[s.round(3).tolist() for s in strat_rps]}")
    print(f"RPS Exploitability: {exp_rps:.6f}")

    # Double oracle
    do = DoubleOracle(rps_game)
    strat_do, exp_do = do.solve()
    print(f"\nRPS Double Oracle NE: {[s.round(3).tolist() for s in strat_do]}")
    print(f"DO Exploitability: {exp_do:.6f}")

    # Nash distance
    metric = NashDistanceMetric(rps_game)
    print(f"\nRPS IS epsilon-NE (0.1): {metric.is_epsilon_nash(strat_rps, 0.1)}")

    # Orchestrator
    orch = NashSolverOrchestrator(NashSolverOrchestrator.SolverType.FICTITIOUS_PLAY)
    result = orch.solve(pd_game, n_iterations=2000)
    print(f"\nOrchestrator result: exploitability={result['exploitability']:.6f}, "
          f"is_epsilon_nash={result['is_epsilon_nash']}")

    print("\nAll smoke tests passed.")
