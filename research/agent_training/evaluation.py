"""
research/agent_training/evaluation.py

Comprehensive agent evaluation for SRFM-Lab.

Covers:
- Per-regime performance breakdown
- Robustness testing via observation perturbation
- Multi-agent comparison
- Action distribution analysis vs BH signals
- Behavioral cloning loss vs expert trades
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from research.agent_training.environment import (
    TradingEnvironment,
    EnvironmentConfig,
    episode_stats,
    EpisodeStats,
    REGIME_MAP,
)


# ---------------------------------------------------------------------------
# EvalResult (standalone for this module)
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Per-episode evaluation result."""

    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    calmar_ratio: float
    regime: str = "ALL"


@dataclass
class RegimeEvalResult:
    """Aggregated results broken down by market regime."""

    regime: str
    mean_return: float
    std_return: float
    mean_sharpe: float
    mean_max_drawdown: float
    mean_n_trades: float
    n_episodes: int
    per_episode: list[EvalResult]


@dataclass
class RobustnessResult:
    """Result of perturbation-based robustness evaluation."""

    base_return: float
    base_sharpe: float
    mean_perturbed_return: float
    std_perturbed_return: float
    mean_perturbed_sharpe: float
    std_perturbed_sharpe: float
    robustness_score: float  # 1 - std_return / (|base_return| + 1e-8)
    n_perturbations: int


@dataclass
class AgentComparison:
    """Comparison of multiple agents over the same episode set."""

    agent_names: list[str]
    mean_returns: dict[str, float]
    std_returns: dict[str, float]
    mean_sharpes: dict[str, float]
    mean_drawdowns: dict[str, float]
    mean_n_trades: dict[str, float]
    comparison_df: pd.DataFrame


@dataclass
class ActionAnalysis:
    """Analysis of an agent's action distribution and BH signal correlation."""

    action_mean: float
    action_std: float
    action_skew: float
    action_histogram: np.ndarray   # (20,) bins
    action_bins: np.ndarray        # (20,) bin edges
    corr_with_tf_score: float
    corr_with_mass: float
    corr_with_ensemble_signal: float
    corr_with_momentum5: float
    long_fraction: float
    short_fraction: float
    flat_fraction: float
    action_series: np.ndarray


# ---------------------------------------------------------------------------
# AgentEvaluator
# ---------------------------------------------------------------------------


class AgentEvaluator:
    """
    Comprehensive agent evaluation suite.

    Args:
        n_eval_episodes : Default number of evaluation episodes.
        seed            : RNG seed for reproducibility.
        verbose         : Print progress.
    """

    def __init__(
        self,
        n_eval_episodes: int = 50,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.n_eval_episodes = n_eval_episodes
        self._rng = np.random.default_rng(seed)
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Core: run agent for N episodes and collect stats
    # ------------------------------------------------------------------

    def _run_episodes(
        self,
        agent: Any,
        env: TradingEnvironment,
        n_episodes: int,
        obs_noise_scale: float = 0.0,
    ) -> list[EpisodeStats]:
        """
        Run agent for n_episodes and return list of EpisodeStats.

        Args:
            obs_noise_scale : Std of Gaussian noise added to observations (robustness testing).
        """
        stats_list = []
        for ep_idx in range(n_episodes):
            obs = env.reset()
            transitions = []
            done = False

            while not done:
                if obs_noise_scale > 0.0:
                    noisy_obs = obs + self._rng.normal(0, obs_noise_scale, obs.shape)
                else:
                    noisy_obs = obs

                action = self._get_action(agent, noisy_obs)
                next_obs, reward, done, info = env.step(action)
                transitions.append({
                    "obs": obs.copy(),
                    "action": action,
                    "reward": reward,
                    "next_obs": next_obs.copy(),
                    "equity": info.get("equity", 0.0),
                    "info": info,
                })
                obs = next_obs

            if transitions:
                stats_list.append(episode_stats(transitions))

        return stats_list

    @staticmethod
    def _get_action(agent: Any, obs: np.ndarray) -> float:
        """Dispatch to the appropriate greedy action method."""
        if hasattr(agent, "act_greedy"):
            return float(agent.act_greedy(obs))
        elif hasattr(agent, "act"):
            result = agent.act(obs)
            if isinstance(result, tuple):
                return float(result[0])
            return float(result)
        return 0.0

    # ------------------------------------------------------------------
    # Regime evaluation
    # ------------------------------------------------------------------

    def evaluate_on_regimes(
        self,
        agent: Any,
        price_data: pd.DataFrame,
        features: np.ndarray,
        regimes: Optional[list[str]] = None,
        n_episodes_per_regime: int = 20,
        config: Optional[EnvironmentConfig] = None,
    ) -> dict[str, RegimeEvalResult]:
        """
        Evaluate agent separately for each market regime.

        The regime is determined by the 'regime' column in price_data.
        For each regime, episodes are started at time steps that belong to
        that regime, so the agent sees predominantly that market condition.

        Args:
            agent                  : Trained agent.
            price_data             : DataFrame with 'close' and 'regime' columns.
            features               : Feature matrix aligned with price_data.
            regimes                : List of regime names to evaluate.
                                     Defaults to all unique regimes in data.
            n_episodes_per_regime  : Episodes per regime.
            config                 : Optional EnvironmentConfig override.

        Returns:
            Dict mapping regime name to RegimeEvalResult.
        """
        if "regime" not in price_data.columns:
            raise ValueError("price_data must have a 'regime' column.")

        all_regimes = (
            regimes if regimes is not None
            else list(price_data["regime"].dropna().unique())
        )

        results: dict[str, RegimeEvalResult] = {}

        for regime in all_regimes:
            # Find start indices that are in this regime (with enough lookahead)
            regime_mask = (
                price_data["regime"].str.upper() == regime.upper()
            ).values
            valid_indices = np.where(regime_mask)[0]
            # Filter: must have at least 200 steps remaining
            valid_indices = valid_indices[valid_indices < len(price_data) - 250]
            valid_indices = valid_indices[valid_indices > 30]

            if len(valid_indices) < 5:
                if self.verbose:
                    print(f"Regime {regime}: not enough data, skipping.")
                continue

            env_cfg = config or EnvironmentConfig()
            env = TradingEnvironment(price_data, features, config=env_cfg)

            per_ep_results = []
            for ep_idx in range(n_episodes_per_regime):
                start_idx = int(
                    valid_indices[self._rng.integers(0, len(valid_indices))]
                )
                obs = env.reset(start_idx=start_idx)
                transitions = []
                done = False

                while not done:
                    action = self._get_action(agent, obs)
                    next_obs, reward, done, info = env.step(action)
                    transitions.append({
                        "obs": obs.copy(),
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs.copy(),
                        "equity": info.get("equity", 0.0),
                        "info": info,
                    })
                    obs = next_obs

                if transitions:
                    st = episode_stats(transitions)
                    per_ep_results.append(EvalResult(
                        total_return=st.total_return,
                        sharpe_ratio=st.sharpe_ratio,
                        sortino_ratio=st.sortino_ratio,
                        max_drawdown=st.max_drawdown,
                        n_trades=st.n_trades,
                        win_rate=st.win_rate,
                        calmar_ratio=st.calmar_ratio,
                        regime=regime,
                    ))

            if per_ep_results:
                rets = [r.total_return for r in per_ep_results]
                sharpes = [r.sharpe_ratio for r in per_ep_results]
                dds = [r.max_drawdown for r in per_ep_results]
                trades = [r.n_trades for r in per_ep_results]

                results[regime] = RegimeEvalResult(
                    regime=regime,
                    mean_return=float(np.mean(rets)),
                    std_return=float(np.std(rets)),
                    mean_sharpe=float(np.mean(sharpes)),
                    mean_max_drawdown=float(np.mean(dds)),
                    mean_n_trades=float(np.mean(trades)),
                    n_episodes=len(per_ep_results),
                    per_episode=per_ep_results,
                )

                if self.verbose:
                    print(
                        f"Regime {regime:20s} | "
                        f"ret={results[regime].mean_return:.4f} | "
                        f"sharpe={results[regime].mean_sharpe:.3f} | "
                        f"dd={results[regime].mean_max_drawdown:.3f}"
                    )

        return results

    # ------------------------------------------------------------------
    # Robustness evaluation
    # ------------------------------------------------------------------

    def evaluate_robustness(
        self,
        agent: Any,
        env: TradingEnvironment,
        n_perturb: int = 50,
        noise_scale: float = 0.01,
        n_base_episodes: int = 20,
    ) -> RobustnessResult:
        """
        Evaluate agent robustness to small perturbations in observations.

        For each perturbation run, Gaussian noise (std=noise_scale) is
        added to all observations before passing to the agent. Robustness
        is measured as 1 - coefficient_of_variation of episode returns.

        Args:
            agent           : Trained agent.
            env             : TradingEnvironment.
            n_perturb       : Number of noisy evaluation runs.
            noise_scale     : Std of observation noise.
            n_base_episodes : Episodes for unperturbed baseline.

        Returns:
            RobustnessResult.
        """
        # Baseline
        base_stats = self._run_episodes(agent, env, n_base_episodes, obs_noise_scale=0.0)
        base_rets = np.array([s.total_return for s in base_stats])
        base_sharpes = np.array([s.sharpe_ratio for s in base_stats])
        base_return = float(np.mean(base_rets))
        base_sharpe = float(np.mean(base_sharpes))

        # Perturbed runs
        perturbed_rets = []
        perturbed_sharpes = []
        n_eps_per_run = max(1, n_base_episodes // 5)

        for _ in range(n_perturb):
            perturb_stats = self._run_episodes(agent, env, n_eps_per_run, obs_noise_scale=noise_scale)
            if perturb_stats:
                perturbed_rets.append(float(np.mean([s.total_return for s in perturb_stats])))
                perturbed_sharpes.append(float(np.mean([s.sharpe_ratio for s in perturb_stats])))

        mean_pert_ret = float(np.mean(perturbed_rets)) if perturbed_rets else 0.0
        std_pert_ret = float(np.std(perturbed_rets)) if perturbed_rets else 0.0
        mean_pert_sharpe = float(np.mean(perturbed_sharpes)) if perturbed_sharpes else 0.0
        std_pert_sharpe = float(np.std(perturbed_sharpes)) if perturbed_sharpes else 0.0

        # Robustness: 1 minus relative degradation
        cv = std_pert_ret / (abs(base_return) + 1e-8)
        robustness = float(max(0.0, 1.0 - cv))

        return RobustnessResult(
            base_return=base_return,
            base_sharpe=base_sharpe,
            mean_perturbed_return=mean_pert_ret,
            std_perturbed_return=std_pert_ret,
            mean_perturbed_sharpe=mean_pert_sharpe,
            std_perturbed_sharpe=std_pert_sharpe,
            robustness_score=robustness,
            n_perturbations=n_perturb,
        )

    # ------------------------------------------------------------------
    # Multi-agent comparison
    # ------------------------------------------------------------------

    def compare_agents(
        self,
        agents: dict[str, Any],
        env: TradingEnvironment,
        n_episodes: int = 100,
    ) -> AgentComparison:
        """
        Compare multiple agents over the same set of episodes.

        All agents are evaluated with the same random start seeds so
        comparisons are fair.

        Args:
            agents     : Dict mapping agent name to agent object.
            env        : TradingEnvironment.
            n_episodes : Number of evaluation episodes per agent.

        Returns:
            AgentComparison with per-agent metrics and a comparison DataFrame.
        """
        # Generate fixed episode seeds for fairness
        seeds = [int(self._rng.integers(0, 100_000)) for _ in range(n_episodes)]

        results: dict[str, list[EpisodeStats]] = {}

        for name, agent in agents.items():
            ep_stats = []
            for seed in seeds:
                env.seed(seed)
                obs = env.reset()
                transitions = []
                done = False

                while not done:
                    action = self._get_action(agent, obs)
                    next_obs, reward, done, info = env.step(action)
                    transitions.append({
                        "obs": obs.copy(),
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs.copy(),
                        "equity": info.get("equity", 0.0),
                        "info": info,
                    })
                    obs = next_obs

                if transitions:
                    ep_stats.append(episode_stats(transitions))

            results[name] = ep_stats

        # Aggregate
        mean_returns: dict[str, float] = {}
        std_returns: dict[str, float] = {}
        mean_sharpes: dict[str, float] = {}
        mean_drawdowns: dict[str, float] = {}
        mean_trades: dict[str, float] = {}

        for name, stats in results.items():
            rets = np.array([s.total_return for s in stats])
            sharpes = np.array([s.sharpe_ratio for s in stats])
            dds = np.array([s.max_drawdown for s in stats])
            trades = np.array([s.n_trades for s in stats])

            mean_returns[name] = float(np.mean(rets))
            std_returns[name] = float(np.std(rets))
            mean_sharpes[name] = float(np.mean(sharpes))
            mean_drawdowns[name] = float(np.mean(dds))
            mean_trades[name] = float(np.mean(trades))

        # DataFrame
        rows = []
        for name in agents:
            rows.append({
                "agent": name,
                "mean_return": mean_returns[name],
                "std_return": std_returns[name],
                "mean_sharpe": mean_sharpes[name],
                "mean_max_drawdown": mean_drawdowns[name],
                "mean_n_trades": mean_trades[name],
                "sharpe_per_dd": mean_sharpes[name] / (mean_drawdowns[name] + 1e-8),
            })
        df = pd.DataFrame(rows).set_index("agent").sort_values("mean_return", ascending=False)

        return AgentComparison(
            agent_names=list(agents.keys()),
            mean_returns=mean_returns,
            std_returns=std_returns,
            mean_sharpes=mean_sharpes,
            mean_drawdowns=mean_drawdowns,
            mean_n_trades=mean_trades,
            comparison_df=df,
        )

    # ------------------------------------------------------------------
    # Action analysis
    # ------------------------------------------------------------------

    def action_analysis(
        self,
        agent: Any,
        env: TradingEnvironment,
        n_episodes: int = 20,
    ) -> ActionAnalysis:
        """
        Analyse the agent's action distribution and correlation with BH signals.

        Collects actions and observations across multiple episodes, then
        computes summary statistics and correlations with key BH features.

        BH signal indices in observation vector:
            [0]  price_pct_change
            [1]  tf_score
            [2]  mass
            [3]  atr_norm
            [4]  regime_encoded
            [5]  equity_ratio
            [6]  position_ratio
            [7]  unrealized_pnl_pct
            [8]  rolling_vol_5
            [9]  rolling_vol_20
            [10] momentum_5
            [11] momentum_20
            [12] ensemble_signal
            [13] bh_active

        Args:
            agent      : Trained agent.
            env        : TradingEnvironment.
            n_episodes : Number of episodes for data collection.

        Returns:
            ActionAnalysis with statistics and correlations.
        """
        all_actions: list[float] = []
        all_obs: list[np.ndarray] = []

        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self._get_action(agent, obs)
                all_actions.append(float(action))
                all_obs.append(obs.copy())
                next_obs, _, done, _ = env.step(action)
                obs = next_obs

        actions_arr = np.array(all_actions, dtype=np.float64)
        obs_arr = np.array(all_obs, dtype=np.float64)  # (N, 14)

        # Histogram
        counts, bin_edges = np.histogram(actions_arr, bins=20, range=(-1.0, 1.0))
        hist = counts.astype(np.float64) / (len(actions_arr) + 1e-12)

        def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
            if x.std() < 1e-10 or y.std() < 1e-10:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        # Skewness
        mu = actions_arr.mean()
        sigma = actions_arr.std() + 1e-12
        skew = float(np.mean(((actions_arr - mu) / sigma) ** 3))

        return ActionAnalysis(
            action_mean=float(actions_arr.mean()),
            action_std=float(actions_arr.std()),
            action_skew=skew,
            action_histogram=hist,
            action_bins=bin_edges,
            corr_with_tf_score=safe_corr(actions_arr, obs_arr[:, 1]),
            corr_with_mass=safe_corr(actions_arr, obs_arr[:, 2]),
            corr_with_ensemble_signal=safe_corr(actions_arr, obs_arr[:, 12]),
            corr_with_momentum5=safe_corr(actions_arr, obs_arr[:, 10]),
            long_fraction=float(np.mean(actions_arr > 0.05)),
            short_fraction=float(np.mean(actions_arr < -0.05)),
            flat_fraction=float(np.mean(np.abs(actions_arr) <= 0.05)),
            action_series=actions_arr,
        )

    # ------------------------------------------------------------------
    # Behavioral cloning loss
    # ------------------------------------------------------------------

    def behavioral_cloning_loss(
        self,
        agent: Any,
        expert_trades: pd.DataFrame,
        obs_columns: Optional[list[str]] = None,
        action_column: str = "action",
    ) -> dict[str, float]:
        """
        Measure how similar the agent's policy is to a set of expert trades.

        The expert trades DataFrame must contain observation columns and
        an action column.

        Args:
            agent          : Trained agent.
            expert_trades  : DataFrame with observations and expert actions.
            obs_columns    : Column names for observations. If None, uses
                             first 14 numeric columns.
            action_column  : Column name for expert actions.

        Returns:
            Dict with 'mse', 'mae', 'direction_accuracy', 'correlation'.
        """
        if obs_columns is None:
            numeric_cols = expert_trades.select_dtypes(include=[np.number]).columns.tolist()
            obs_columns = [c for c in numeric_cols if c != action_column][:14]

        if len(obs_columns) == 0:
            raise ValueError("No observation columns found in expert_trades.")

        obs_matrix = expert_trades[obs_columns].values.astype(np.float64)
        expert_actions = expert_trades[action_column].values.astype(np.float64)

        # Pad or trim to obs_dim
        obs_dim = obs_matrix.shape[1]
        expected_dim = 14
        if obs_dim < expected_dim:
            pad = np.zeros((len(obs_matrix), expected_dim - obs_dim))
            obs_matrix = np.concatenate([obs_matrix, pad], axis=1)
        elif obs_dim > expected_dim:
            obs_matrix = obs_matrix[:, :expected_dim]

        # Get agent actions
        agent_actions = np.array(
            [self._get_action(agent, obs_matrix[i]) for i in range(len(obs_matrix))],
            dtype=np.float64,
        )

        mse = float(np.mean((agent_actions - expert_actions) ** 2))
        mae = float(np.mean(np.abs(agent_actions - expert_actions)))

        # Direction accuracy: same sign of action
        dir_match = (np.sign(agent_actions) == np.sign(expert_actions)).mean()
        direction_accuracy = float(dir_match)

        # Correlation
        if agent_actions.std() > 1e-10 and expert_actions.std() > 1e-10:
            correlation = float(np.corrcoef(agent_actions, expert_actions)[0, 1])
        else:
            correlation = 0.0

        return {
            "mse": mse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "correlation": correlation,
            "n_samples": len(expert_actions),
        }

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def full_evaluation_report(
        self,
        agent: Any,
        env: TradingEnvironment,
        agent_name: str = "agent",
        n_episodes: int = 50,
    ) -> dict[str, Any]:
        """
        Run all evaluations and return a summary dict.

        Args:
            agent       : Trained agent.
            env         : TradingEnvironment.
            agent_name  : Name for report.
            n_episodes  : Episodes for each evaluation.

        Returns:
            Dict with keys: 'name', 'eval_stats', 'robustness', 'action_analysis'.
        """
        # Basic eval
        stats = self._run_episodes(agent, env, n_episodes)
        rets = np.array([s.total_return for s in stats])
        sharpes = np.array([s.sharpe_ratio for s in stats])
        dds = np.array([s.max_drawdown for s in stats])

        eval_summary = {
            "mean_return": float(np.mean(rets)),
            "std_return": float(np.std(rets)),
            "mean_sharpe": float(np.mean(sharpes)),
            "mean_max_drawdown": float(np.mean(dds)),
            "n_episodes": n_episodes,
        }

        # Robustness
        robustness = self.evaluate_robustness(agent, env, n_perturb=20, noise_scale=0.01)

        # Action analysis
        action_an = self.action_analysis(agent, env, n_episodes=min(10, n_episodes))

        return {
            "name": agent_name,
            "eval_stats": eval_summary,
            "robustness": robustness,
            "action_analysis": action_an,
        }

    def print_report(self, report: dict[str, Any]) -> None:
        """Pretty-print a full evaluation report."""
        print(f"\n{'='*60}")
        print(f"Agent Evaluation Report: {report['name']}")
        print(f"{'='*60}")

        es = report["eval_stats"]
        print(f"\nOverall Performance ({es['n_episodes']} episodes):")
        print(f"  Mean Return:     {es['mean_return']:.4f} ± {es['std_return']:.4f}")
        print(f"  Mean Sharpe:     {es['mean_sharpe']:.3f}")
        print(f"  Mean MaxDrawdown:{es['mean_max_drawdown']:.3f}")

        rb = report["robustness"]
        print(f"\nRobustness (noise_scale=0.01, {rb.n_perturbations} perturbations):")
        print(f"  Base Return:           {rb.base_return:.4f}")
        print(f"  Perturbed Return:      {rb.mean_perturbed_return:.4f} ± {rb.std_perturbed_return:.4f}")
        print(f"  Robustness Score:      {rb.robustness_score:.3f}")

        aa = report["action_analysis"]
        print(f"\nAction Distribution:")
        print(f"  Mean: {aa.action_mean:.4f}  Std: {aa.action_std:.4f}  Skew: {aa.action_skew:.3f}")
        print(f"  Long: {aa.long_fraction:.1%}  Short: {aa.short_fraction:.1%}  Flat: {aa.flat_fraction:.1%}")
        print(f"\nBH Signal Correlations:")
        print(f"  tf_score:        {aa.corr_with_tf_score:.3f}")
        print(f"  mass:            {aa.corr_with_mass:.3f}")
        print(f"  ensemble_signal: {aa.corr_with_ensemble_signal:.3f}")
        print(f"  momentum_5:      {aa.corr_with_momentum5:.3f}")
        print(f"{'='*60}\n")
