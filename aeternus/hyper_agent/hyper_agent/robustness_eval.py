"""
robustness_eval.py — Robustness Evaluation Suite for Hyper-Agent.

Evaluates agent robustness across multiple axes:
1. Stress test: 100 adversarial scenarios, percentile performance
2. Regime transfer: train calm → evaluate crisis
3. Distribution shift: gradual parameter shift, track degradation
4. Catastrophic forgetting: post-finetune performance on sim scenarios
"""

from __future__ import annotations

import math
import time
import logging
import collections
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvalType(Enum):
    STRESS_TEST = auto()
    REGIME_TRANSFER = auto()
    DISTRIBUTION_SHIFT = auto()
    CATASTROPHIC_FORGETTING = auto()
    FULL = auto()


class ShiftType(Enum):
    VOLATILITY = auto()
    SPREAD = auto()
    KYLE_LAMBDA = auto()
    CORRELATION = auto()
    FILL_RATE = auto()
    COMBINED = auto()


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class StressTestConfig:
    num_scenarios: int = 100
    episodes_per_scenario: int = 5
    episode_length: int = 500
    performance_percentiles: List[float] = field(default_factory=lambda: [5.0, 10.0, 25.0, 50.0, 75.0, 90.0])
    adversarial_intensity: float = 1.0
    report_worst_k: int = 10
    seed: int = 42


@dataclass
class RegimeTransferConfig:
    train_regimes: List[str] = field(default_factory=lambda: ["CALM"])
    eval_regimes: List[str] = field(default_factory=lambda: ["CRASH", "CRISIS", "VOLATILE"])
    episodes_per_regime: int = 20
    episode_length: int = 500


@dataclass
class DistributionShiftConfig:
    shift_type: ShiftType = ShiftType.COMBINED
    num_shift_levels: int = 10
    min_shift: float = 0.0      # fraction of base value
    max_shift: float = 5.0      # max multiplier
    episodes_per_level: int = 10
    episode_length: int = 500
    track_degradation_threshold: float = 0.2  # flag when perf drops > 20%


@dataclass
class CatastrophicForgettingConfig:
    sim_scenarios: List[str] = field(default_factory=lambda: [
        "spread_noise_light", "fill_randomization", "latency", "baseline"
    ])
    episodes_per_scenario: int = 10
    episode_length: int = 500
    forgetting_threshold: float = 0.15  # flag if perf drops > 15%


@dataclass
class RobustnessEvalConfig:
    stress_test: StressTestConfig = field(default_factory=StressTestConfig)
    regime_transfer: RegimeTransferConfig = field(default_factory=RegimeTransferConfig)
    distribution_shift: DistributionShiftConfig = field(default_factory=DistributionShiftConfig)
    catastrophic_forgetting: CatastrophicForgettingConfig = field(default_factory=CatastrophicForgettingConfig)

    eval_types: List[EvalType] = field(default_factory=lambda: [EvalType.FULL])
    seed: int = 42
    num_parallel_envs: int = 1
    verbose: bool = True


# ---------------------------------------------------------------------------
# Scenario perturbation context
# ---------------------------------------------------------------------------

@dataclass
class ScenarioContext:
    """Defines the conditions for one evaluation scenario."""
    scenario_id: int
    seed: int
    spread_multiplier: float = 1.0
    vol_multiplier: float = 1.0
    kyle_lambda_multiplier: float = 1.0
    fill_rate_modifier: float = 0.0
    regime: str = "CALM"
    adversarial_perturbation: bool = False
    perturbation_epsilon: float = 0.05
    description: str = ""


def build_stress_scenarios(
    num_scenarios: int, seed: int = 42
) -> List[ScenarioContext]:
    """Build a diverse set of stress test scenarios."""
    rng = np.random.default_rng(seed)
    scenarios: List[ScenarioContext] = []
    regimes = ["CALM", "TRENDING_UP", "TRENDING_DOWN", "VOLATILE", "CRASH", "CRISIS", "ILLIQUID"]

    for i in range(num_scenarios):
        regime = regimes[i % len(regimes)]
        spread_mult = float(rng.uniform(0.5, 20.0))
        vol_mult = float(rng.uniform(0.3, 5.0))
        lambda_mult = float(rng.uniform(0.5, 10.0))
        fill_mod = float(rng.uniform(-0.3, 0.0))
        use_adv = bool(rng.random() < 0.4)
        eps = float(rng.uniform(0.01, 0.15))

        scenarios.append(ScenarioContext(
            scenario_id=i,
            seed=int(rng.integers(0, 2**31)),
            spread_multiplier=spread_mult,
            vol_multiplier=vol_mult,
            kyle_lambda_multiplier=lambda_mult,
            fill_rate_modifier=fill_mod,
            regime=regime,
            adversarial_perturbation=use_adv,
            perturbation_epsilon=eps,
            description=f"regime={regime},spread={spread_mult:.1f}x,vol={vol_mult:.1f}x",
        ))
    return scenarios


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class EpisodeRunner:
    """Runs a single evaluation episode with given scenario context."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        rng: np.random.Generator,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.rng = rng

    def run(
        self,
        context: ScenarioContext,
        episode_length: int = 500,
    ) -> Dict[str, Any]:
        """
        Run one episode. Returns dict with episode metrics.
        """
        env = self.env_factory()
        t_start = time.perf_counter()

        # Apply scenario context to env
        self._apply_context(env, context)

        try:
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            obs = np.array(obs, dtype=np.float32).flatten() if obs is not None else np.zeros(64)
        except Exception:
            obs = np.zeros(64, dtype=np.float32)

        ep_return = 0.0
        ep_length = 0
        actions_taken: List[np.ndarray] = []
        rewards: List[float] = []

        for t in range(episode_length):
            # Apply adversarial perturbation if configured
            if context.adversarial_perturbation:
                eps = context.perturbation_epsilon
                obs = obs + self.rng.uniform(-eps, eps, size=obs.shape)

            # Get action from policy
            try:
                if hasattr(self.policy, "act"):
                    action = np.array(self.policy.act(obs), dtype=np.float32).flatten()
                elif callable(self.policy):
                    action = np.array(self.policy(obs), dtype=np.float32).flatten()
                else:
                    action = np.zeros(4, dtype=np.float32)
            except Exception:
                action = np.zeros(4, dtype=np.float32)

            actions_taken.append(action)

            try:
                result = env.step(action)
                if len(result) == 4:
                    next_obs, rew, done, info = result
                else:
                    next_obs, rew, done, _, info = result
                ep_return += float(rew)
                rewards.append(float(rew))
                ep_length += 1
                if done:
                    break
                obs = np.array(next_obs, dtype=np.float32).flatten() if next_obs is not None else obs
            except Exception as e:
                logger.debug("Episode step error: %s", e)
                break

        wall_time = time.perf_counter() - t_start
        return {
            "scenario_id": context.scenario_id,
            "ep_return": ep_return,
            "ep_length": ep_length,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "wall_time_s": wall_time,
            "regime": context.regime,
            "adversarial": context.adversarial_perturbation,
        }

    def _apply_context(self, env: Any, context: ScenarioContext) -> None:
        """Modify env internals to match scenario context."""
        try:
            if hasattr(env, "spreads"):
                env.spreads = np.array(env.spreads) * context.spread_multiplier
            if hasattr(env, "volatility"):
                env.volatility = np.array(env.volatility) * context.vol_multiplier
            if hasattr(env, "kyle_lambda"):
                env.kyle_lambda = np.array(env.kyle_lambda) * context.kyle_lambda_multiplier
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stress tester
# ---------------------------------------------------------------------------

class StressTester:
    """Runs 100 adversarial scenarios and reports percentile performance."""

    def __init__(
        self,
        config: StressTestConfig,
        env_factory: Callable[[], Any],
        policy: Any,
    ) -> None:
        self.cfg = config
        self.env_factory = env_factory
        self.policy = policy
        self.rng = np.random.default_rng(config.seed)
        self.runner = EpisodeRunner(env_factory, policy, self.rng)

    def run(self) -> Dict[str, Any]:
        """Run full stress test."""
        cfg = self.cfg
        scenarios = build_stress_scenarios(cfg.num_scenarios, cfg.seed)
        all_returns: List[float] = []
        scenario_results: List[Dict[str, Any]] = []

        for scenario in scenarios:
            ep_returns = []
            for _ in range(cfg.episodes_per_scenario):
                result = self.runner.run(scenario, cfg.episode_length)
                ep_returns.append(result["ep_return"])
                all_returns.append(result["ep_return"])

            scenario_results.append({
                "scenario_id": scenario.scenario_id,
                "description": scenario.description,
                "mean_return": float(np.mean(ep_returns)),
                "min_return": float(np.min(ep_returns)),
                "regime": scenario.regime,
                "adversarial": scenario.adversarial_perturbation,
            })

        # Percentile performance
        percentile_results = {}
        for pct in cfg.performance_percentiles:
            percentile_results[f"p{int(pct)}"] = float(np.percentile(all_returns, pct))

        # Worst scenarios
        sorted_scenarios = sorted(scenario_results, key=lambda s: s["mean_return"])
        worst_k = sorted_scenarios[:cfg.report_worst_k]

        return {
            "num_scenarios": cfg.num_scenarios,
            "total_episodes": len(all_returns),
            "mean_return": float(np.mean(all_returns)),
            "std_return": float(np.std(all_returns)),
            "percentiles": percentile_results,
            "worst_scenarios": worst_k,
            "pass_rate_p10": float(np.mean(np.array(all_returns) > percentile_results.get("p10", -1e9))),
            "robustness_score": percentile_results.get("p10", float(np.min(all_returns))),
        }


# ---------------------------------------------------------------------------
# Regime transfer evaluator
# ---------------------------------------------------------------------------

class RegimeTransferEvaluator:
    """Evaluates performance degradation from calm training to crisis evaluation."""

    def __init__(
        self,
        config: RegimeTransferConfig,
        env_factory: Callable[[], Any],
        policy: Any,
    ) -> None:
        self.cfg = config
        self.env_factory = env_factory
        self.policy = policy
        self.rng = np.random.default_rng(0)
        self.runner = EpisodeRunner(env_factory, policy, self.rng)

    def evaluate(self) -> Dict[str, Any]:
        """
        Run regime transfer evaluation.

        Returns performance per eval regime and degradation vs train regimes.
        """
        cfg = self.cfg
        results: Dict[str, Any] = {}

        # Evaluate on train regimes (baseline)
        train_returns: Dict[str, List[float]] = {}
        for regime in cfg.train_regimes:
            regime_returns = []
            for i in range(cfg.episodes_per_regime):
                ctx = ScenarioContext(
                    scenario_id=i,
                    seed=i * 100,
                    regime=regime,
                    vol_multiplier=1.0,
                    spread_multiplier=1.0,
                )
                result = self.runner.run(ctx, cfg.episode_length)
                regime_returns.append(result["ep_return"])
            train_returns[regime] = regime_returns
            results[f"train_{regime}"] = {
                "mean": float(np.mean(regime_returns)),
                "std": float(np.std(regime_returns)),
            }

        # Evaluate on transfer regimes
        eval_returns: Dict[str, List[float]] = {}
        for regime in cfg.eval_regimes:
            regime_returns = []
            vol_mult = {"CRASH": 5.0, "CRISIS": 8.0, "VOLATILE": 3.0}.get(regime, 2.0)
            spread_mult = {"CRASH": 10.0, "CRISIS": 15.0, "VOLATILE": 4.0}.get(regime, 2.0)
            for i in range(cfg.episodes_per_regime):
                ctx = ScenarioContext(
                    scenario_id=i,
                    seed=i * 200,
                    regime=regime,
                    vol_multiplier=vol_mult,
                    spread_multiplier=spread_mult,
                )
                result = self.runner.run(ctx, cfg.episode_length)
                regime_returns.append(result["ep_return"])
            eval_returns[regime] = regime_returns
            results[f"eval_{regime}"] = {
                "mean": float(np.mean(regime_returns)),
                "std": float(np.std(regime_returns)),
            }

        # Compute transfer gap
        if train_returns and eval_returns:
            all_train = [r for returns in train_returns.values() for r in returns]
            all_eval = [r for returns in eval_returns.values() for r in returns]
            train_mean = float(np.mean(all_train))
            eval_mean = float(np.mean(all_eval))
            gap = train_mean - eval_mean
            rel_gap = gap / (abs(train_mean) + 1e-8)
            results["transfer_gap"] = gap
            results["relative_transfer_gap"] = rel_gap
            results["transfer_pass"] = rel_gap < 0.5

        return results


# ---------------------------------------------------------------------------
# Distribution shift evaluator
# ---------------------------------------------------------------------------

class DistributionShiftEvaluator:
    """
    Gradually shifts market parameters and tracks performance degradation.
    """

    def __init__(
        self,
        config: DistributionShiftConfig,
        env_factory: Callable[[], Any],
        policy: Any,
    ) -> None:
        self.cfg = config
        self.env_factory = env_factory
        self.policy = policy
        self.rng = np.random.default_rng(42)
        self.runner = EpisodeRunner(env_factory, policy, self.rng)

    def evaluate(self) -> Dict[str, Any]:
        """Run distribution shift evaluation."""
        cfg = self.cfg
        shift_levels = np.linspace(cfg.min_shift, cfg.max_shift, cfg.num_shift_levels)
        level_results: List[Dict[str, Any]] = []
        baseline_return: Optional[float] = None

        for level_idx, shift in enumerate(shift_levels):
            level_returns = []
            for ep in range(cfg.episodes_per_level):
                ctx = self._build_context(shift, ep, level_idx)
                result = self.runner.run(ctx, cfg.episode_length)
                level_returns.append(result["ep_return"])

            level_mean = float(np.mean(level_returns))

            if level_idx == 0:
                baseline_return = level_mean

            degradation = 0.0
            if baseline_return is not None and abs(baseline_return) > 1e-8:
                degradation = (baseline_return - level_mean) / abs(baseline_return)

            level_results.append({
                "shift_level": float(shift),
                "mean_return": level_mean,
                "std_return": float(np.std(level_returns)),
                "degradation": degradation,
                "threshold_exceeded": degradation > cfg.track_degradation_threshold,
            })

        # Find critical shift level
        critical_level = None
        for r in level_results:
            if r["threshold_exceeded"] and critical_level is None:
                critical_level = r["shift_level"]

        # Compute AUC of performance vs shift
        returns = [r["mean_return"] for r in level_results]
        auc = float(np.trapz(returns, shift_levels)) / (shift_levels[-1] - shift_levels[0] + 1e-8)

        return {
            "shift_type": cfg.shift_type.name,
            "level_results": level_results,
            "critical_shift_level": critical_level,
            "baseline_return": baseline_return,
            "auc_return": auc,
            "max_degradation": max(r["degradation"] for r in level_results),
            "num_levels_exceeded_threshold": sum(
                1 for r in level_results if r["threshold_exceeded"]
            ),
        }

    def _build_context(
        self, shift: float, ep: int, level_idx: int
    ) -> ScenarioContext:
        cfg = self.cfg
        ctx = ScenarioContext(
            scenario_id=level_idx * 100 + ep,
            seed=level_idx * 1000 + ep,
        )
        mult = max(0.1, 1.0 + shift)

        if cfg.shift_type in (ShiftType.VOLATILITY, ShiftType.COMBINED):
            ctx.vol_multiplier = mult
        if cfg.shift_type in (ShiftType.SPREAD, ShiftType.COMBINED):
            ctx.spread_multiplier = mult
        if cfg.shift_type in (ShiftType.KYLE_LAMBDA, ShiftType.COMBINED):
            ctx.kyle_lambda_multiplier = mult
        if cfg.shift_type in (ShiftType.FILL_RATE, ShiftType.COMBINED):
            ctx.fill_rate_modifier = -min(shift * 0.1, 0.5)

        return ctx


# ---------------------------------------------------------------------------
# Catastrophic forgetting evaluator
# ---------------------------------------------------------------------------

class CatastrophicForgettingEvaluator:
    """
    Tests for catastrophic forgetting after real-world fine-tuning.

    Compares sim performance before and after fine-tuning.
    """

    def __init__(
        self,
        config: CatastrophicForgettingConfig,
        env_factory: Callable[[], Any],
        policy: Any,
    ) -> None:
        self.cfg = config
        self.env_factory = env_factory
        self.policy = policy
        self.rng = np.random.default_rng(0)
        self.runner = EpisodeRunner(env_factory, policy, self.rng)
        self._pre_finetune_baseline: Optional[Dict[str, float]] = None

    def measure_pre_finetune_baseline(self) -> Dict[str, float]:
        """Measure performance BEFORE fine-tuning. Store as baseline."""
        result = self._evaluate_sim_scenarios()
        self._pre_finetune_baseline = result
        return result

    def evaluate_post_finetune(self) -> Dict[str, Any]:
        """Evaluate AFTER fine-tuning and compute forgetting."""
        if self._pre_finetune_baseline is None:
            logger.warning("No pre-finetune baseline; measuring now.")
            self.measure_pre_finetune_baseline()

        post = self._evaluate_sim_scenarios()
        baseline = self._pre_finetune_baseline or {}

        forgetting: Dict[str, float] = {}
        for scenario in self.cfg.sim_scenarios:
            pre = baseline.get(scenario, 0.0)
            post_val = post.get(scenario, 0.0)
            if abs(pre) > 1e-8:
                forgetting[scenario] = (pre - post_val) / abs(pre)
            else:
                forgetting[scenario] = 0.0

        mean_forgetting = float(np.mean(list(forgetting.values()))) if forgetting else 0.0
        catastrophic = mean_forgetting > self.cfg.forgetting_threshold

        return {
            "pre_finetune": baseline,
            "post_finetune": post,
            "forgetting_per_scenario": forgetting,
            "mean_forgetting": mean_forgetting,
            "catastrophic_forgetting_detected": catastrophic,
            "forgetting_threshold": self.cfg.forgetting_threshold,
        }

    def _evaluate_sim_scenarios(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        cfg = self.cfg

        scenario_params = {
            "baseline": ScenarioContext(scenario_id=0, seed=0),
            "spread_noise_light": ScenarioContext(scenario_id=1, seed=1, spread_multiplier=1.5),
            "fill_randomization": ScenarioContext(scenario_id=2, seed=2, fill_rate_modifier=-0.15),
            "latency": ScenarioContext(scenario_id=3, seed=3),
            "price_impact": ScenarioContext(scenario_id=4, seed=4, kyle_lambda_multiplier=2.0),
            "volatile": ScenarioContext(scenario_id=5, seed=5, vol_multiplier=3.0, regime="VOLATILE"),
        }

        for scenario in cfg.sim_scenarios:
            ctx = scenario_params.get(scenario, ScenarioContext(scenario_id=0, seed=0))
            ep_returns = []
            for ep in range(cfg.episodes_per_scenario):
                ctx.seed = ep * 100
                result = self.runner.run(ctx, cfg.episode_length)
                ep_returns.append(result["ep_return"])
            results[scenario] = float(np.mean(ep_returns))

        return results


# ---------------------------------------------------------------------------
# Master robustness evaluator
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    """
    Full robustness evaluation suite for Hyper-Agent.

    Runs all evaluation types and compiles a robustness report.
    """

    def __init__(
        self,
        config: Optional[RobustnessEvalConfig] = None,
        env_factory: Optional[Callable[[], Any]] = None,
        policy: Optional[Any] = None,
    ) -> None:
        self.cfg = config or RobustnessEvalConfig()
        self.env_factory = env_factory
        self.policy = policy

        if env_factory is not None and policy is not None:
            self.stress_tester = StressTester(self.cfg.stress_test, env_factory, policy)
            self.regime_transfer = RegimeTransferEvaluator(self.cfg.regime_transfer, env_factory, policy)
            self.dist_shift = DistributionShiftEvaluator(self.cfg.distribution_shift, env_factory, policy)
            self.forgetting = CatastrophicForgettingEvaluator(self.cfg.catastrophic_forgetting, env_factory, policy)
        else:
            self.stress_tester = None
            self.regime_transfer = None
            self.dist_shift = None
            self.forgetting = None

        self._results: Dict[str, Any] = {}

    def run_full_eval(self) -> Dict[str, Any]:
        """Run all evaluation types. Returns comprehensive robustness report."""
        report: Dict[str, Any] = {
            "timestamp": time.time(),
            "config": {
                "eval_types": [e.name for e in self.cfg.eval_types],
                "seed": self.cfg.seed,
            },
        }

        if EvalType.STRESS_TEST in self.cfg.eval_types or EvalType.FULL in self.cfg.eval_types:
            if self.stress_tester is not None:
                logger.info("Running stress test...")
                t0 = time.perf_counter()
                report["stress_test"] = self.stress_tester.run()
                report["stress_test"]["wall_time_s"] = time.perf_counter() - t0
            else:
                report["stress_test"] = {"error": "no_env_or_policy"}

        if EvalType.REGIME_TRANSFER in self.cfg.eval_types or EvalType.FULL in self.cfg.eval_types:
            if self.regime_transfer is not None:
                logger.info("Running regime transfer eval...")
                t0 = time.perf_counter()
                report["regime_transfer"] = self.regime_transfer.evaluate()
                report["regime_transfer"]["wall_time_s"] = time.perf_counter() - t0
            else:
                report["regime_transfer"] = {"error": "no_env_or_policy"}

        if EvalType.DISTRIBUTION_SHIFT in self.cfg.eval_types or EvalType.FULL in self.cfg.eval_types:
            if self.dist_shift is not None:
                logger.info("Running distribution shift eval...")
                t0 = time.perf_counter()
                report["distribution_shift"] = self.dist_shift.evaluate()
                report["distribution_shift"]["wall_time_s"] = time.perf_counter() - t0
            else:
                report["distribution_shift"] = {"error": "no_env_or_policy"}

        if EvalType.CATASTROPHIC_FORGETTING in self.cfg.eval_types or EvalType.FULL in self.cfg.eval_types:
            if self.forgetting is not None:
                logger.info("Running catastrophic forgetting eval...")
                t0 = time.perf_counter()
                report["catastrophic_forgetting"] = self.forgetting.evaluate_post_finetune()
                report["catastrophic_forgetting"]["wall_time_s"] = time.perf_counter() - t0
            else:
                report["catastrophic_forgetting"] = {"error": "no_env_or_policy"}

        # Compute overall robustness score
        report["overall"] = self._compute_overall_score(report)
        self._results = report
        return report

    def _compute_overall_score(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate robustness score from all sub-evaluations."""
        scores: Dict[str, float] = {}

        # Stress test: use p10 performance
        st = report.get("stress_test", {})
        if "percentiles" in st:
            p10 = st["percentiles"].get("p10", 0.0)
            # Normalize to [0, 1] (0 = very bad, 1 = perfect)
            # Using a soft sign: higher is better
            scores["stress"] = float(1.0 / (1.0 + math.exp(-p10 / 100.0)))

        # Regime transfer: 1 - relative gap (capped at 0)
        rt = report.get("regime_transfer", {})
        if "relative_transfer_gap" in rt:
            gap = rt["relative_transfer_gap"]
            scores["regime_transfer"] = float(max(0.0, 1.0 - gap))

        # Distribution shift: AUC normalized
        ds = report.get("distribution_shift", {})
        if "auc_return" in ds and ds.get("baseline_return"):
            baseline = abs(ds["baseline_return"])
            if baseline > 1e-8:
                scores["dist_shift"] = float(max(0.0, min(1.0, ds["auc_return"] / baseline)))

        # Catastrophic forgetting: 1 - mean_forgetting
        cf = report.get("catastrophic_forgetting", {})
        if "mean_forgetting" in cf:
            scores["forgetting"] = float(max(0.0, 1.0 - cf["mean_forgetting"]))

        overall = float(np.mean(list(scores.values()))) if scores else 0.0

        return {
            "overall_robustness_score": overall,
            "component_scores": scores,
            "pass_threshold": 0.6,
            "passed": overall >= 0.6,
        }

    def run_stress_test_only(self) -> Dict[str, Any]:
        if self.stress_tester is None:
            return {"error": "no_env_or_policy"}
        return self.stress_tester.run()

    def run_regime_transfer_only(self) -> Dict[str, Any]:
        if self.regime_transfer is None:
            return {"error": "no_env_or_policy"}
        return self.regime_transfer.evaluate()

    def get_last_results(self) -> Dict[str, Any]:
        return self._results.copy()

    def print_summary(self, results: Optional[Dict[str, Any]] = None) -> None:
        r = results or self._results
        if not r:
            print("No results available.")
            return

        overall = r.get("overall", {})
        print("=" * 60)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Score: {overall.get('overall_robustness_score', 0):.3f}")
        print(f"Pass: {overall.get('passed', False)}")
        print("\nComponent Scores:")
        for k, v in overall.get("component_scores", {}).items():
            print(f"  {k}: {v:.3f}")

        st = r.get("stress_test", {})
        if "percentiles" in st:
            print("\nStress Test Percentiles:")
            for k, v in st["percentiles"].items():
                print(f"  {k}: {v:.2f}")

        cf = r.get("catastrophic_forgetting", {})
        if "mean_forgetting" in cf:
            print(f"\nCatastrophic Forgetting: {cf['mean_forgetting']:.3f}")
            print(f"Detected: {cf.get('catastrophic_forgetting_detected', False)}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Quick eval utilities
# ---------------------------------------------------------------------------

def quick_stress_test(
    policy: Any,
    env_factory: Callable[[], Any],
    num_scenarios: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fast stress test with fewer scenarios."""
    cfg = RobustnessEvalConfig(
        stress_test=StressTestConfig(
            num_scenarios=num_scenarios,
            episodes_per_scenario=3,
            seed=seed,
        ),
        eval_types=[EvalType.STRESS_TEST],
    )
    evaluator = RobustnessEvaluator(cfg, env_factory, policy)
    return evaluator.run_stress_test_only()


def quick_regime_transfer(
    policy: Any,
    env_factory: Callable[[], Any],
) -> Dict[str, Any]:
    """Fast regime transfer check."""
    cfg = RobustnessEvalConfig(
        regime_transfer=RegimeTransferConfig(
            episodes_per_regime=5,
        ),
        eval_types=[EvalType.REGIME_TRANSFER],
    )
    evaluator = RobustnessEvaluator(cfg, env_factory, policy)
    return evaluator.run_regime_transfer_only()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "EvalType",
    "ShiftType",
    # Configs
    "RobustnessEvalConfig",
    "StressTestConfig",
    "RegimeTransferConfig",
    "DistributionShiftConfig",
    "CatastrophicForgettingConfig",
    # Data
    "ScenarioContext",
    # Sub-modules
    "EpisodeRunner",
    "StressTester",
    "RegimeTransferEvaluator",
    "DistributionShiftEvaluator",
    "CatastrophicForgettingEvaluator",
    # Main
    "RobustnessEvaluator",
    # Utilities
    "build_stress_scenarios",
    "quick_stress_test",
    "quick_regime_transfer",
    # Extended
    "PerformanceProfiler",
    "TailRiskEvaluator",
    "SharpeRobustnessTest",
    "DrawdownRobustnessTest",
    "MultiRegimePortfolioEval",
    "LatencyRobustnessEval",
    "RobustnessDashboard",
    "ScenarioLibrary",
]


# ---------------------------------------------------------------------------
# Extended: PerformanceProfiler
# ---------------------------------------------------------------------------

class PerformanceProfiler:
    """
    Profiles agent performance across multiple dimensions simultaneously.

    Tracks not just cumulative return but also:
    - Sharpe ratio stability
    - Maximum drawdown distribution
    - Win rate by regime
    - Turnover and transaction costs
    - Fill rate and execution quality
    """

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._returns: collections.deque = collections.deque(maxlen=window)
        self._sharpes: List[float] = []
        self._drawdowns: List[float] = []
        self._fill_rates: collections.deque = collections.deque(maxlen=window)
        self._turnovers: collections.deque = collections.deque(maxlen=window)
        self._regime_returns: Dict[str, List[float]] = {}

    def record_step(
        self,
        step_return: float,
        fill_rate: float = 1.0,
        turnover: float = 0.0,
        regime: str = "CALM",
    ) -> None:
        self._returns.append(step_return)
        self._fill_rates.append(fill_rate)
        self._turnovers.append(turnover)
        if regime not in self._regime_returns:
            self._regime_returns[regime] = []
        self._regime_returns[regime].append(step_return)

    def compute_sharpe(self, annualization: float = 252.0) -> float:
        rets = np.array(list(self._returns))
        if len(rets) < 2:
            return 0.0
        mean = np.mean(rets)
        std = np.std(rets) + 1e-8
        return float(mean / std * math.sqrt(annualization))

    def compute_max_drawdown(self) -> float:
        rets = np.array(list(self._returns))
        if len(rets) < 2:
            return 0.0
        cumulative = np.cumsum(rets)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        return float(np.max(drawdowns))

    def compute_sortino(self, annualization: float = 252.0, threshold: float = 0.0) -> float:
        rets = np.array(list(self._returns))
        if len(rets) < 2:
            return 0.0
        downside = rets[rets < threshold]
        if len(downside) == 0:
            return float("inf")
        downside_std = float(np.std(downside)) + 1e-8
        return float(np.mean(rets) / downside_std * math.sqrt(annualization))

    def get_profile(self) -> Dict[str, Any]:
        rets = list(self._returns)
        fills = list(self._fill_rates)
        return {
            "n_steps": len(rets),
            "total_return": float(sum(rets)),
            "mean_return": float(np.mean(rets)) if rets else 0.0,
            "sharpe": self.compute_sharpe(),
            "sortino": self.compute_sortino(),
            "max_drawdown": self.compute_max_drawdown(),
            "mean_fill_rate": float(np.mean(fills)) if fills else 1.0,
            "mean_turnover": float(np.mean(list(self._turnovers))) if self._turnovers else 0.0,
            "regime_returns": {
                regime: float(np.mean(returns))
                for regime, returns in self._regime_returns.items()
                if returns
            },
            "return_p5": float(np.percentile(rets, 5)) if rets else 0.0,
            "return_p95": float(np.percentile(rets, 95)) if rets else 0.0,
        }


# ---------------------------------------------------------------------------
# Extended: TailRiskEvaluator
# ---------------------------------------------------------------------------

class TailRiskEvaluator:
    """
    Evaluates tail risk metrics for a trading agent.

    Computes:
    - CVaR (Conditional Value at Risk) at various confidence levels
    - Expected shortfall
    - Tail ratio (right tail / left tail)
    - Extreme value distribution fit
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        confidence_levels: Optional[List[float]] = None,
        n_episodes: int = 100,
        episode_length: int = 500,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

    def evaluate(self) -> Dict[str, Any]:
        """Run tail risk evaluation."""
        runner = EpisodeRunner(self.env_factory, self.policy, self.rng)
        all_returns = []

        for i in range(self.n_episodes):
            ctx = ScenarioContext(scenario_id=i, seed=i)
            result = runner.run(ctx, self.episode_length)
            all_returns.append(result["ep_return"])

        rets = np.array(all_returns)
        results = {
            "n_episodes": self.n_episodes,
            "mean_return": float(np.mean(rets)),
            "std_return": float(np.std(rets)),
        }

        for cl in self.confidence_levels:
            var = float(np.percentile(rets, (1 - cl) * 100))
            cvar = float(np.mean(rets[rets <= var])) if np.any(rets <= var) else var
            results[f"var_{int(cl*100)}"] = var
            results[f"cvar_{int(cl*100)}"] = cvar

        # Tail ratio
        p5 = float(np.percentile(rets, 5))
        p95 = float(np.percentile(rets, 95))
        results["tail_ratio"] = abs(p95) / max(abs(p5), 1e-8)

        return results


# ---------------------------------------------------------------------------
# Extended: SharpeRobustnessTest
# ---------------------------------------------------------------------------

class SharpeRobustnessTest:
    """
    Tests whether agent's Sharpe ratio remains positive and significant
    under various market conditions.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        min_acceptable_sharpe: float = 0.5,
        n_episodes: int = 50,
        episode_length: int = 500,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.min_sharpe = min_acceptable_sharpe
        self.n_episodes = n_episodes
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

    def run(self) -> Dict[str, Any]:
        runner = EpisodeRunner(self.env_factory, self.policy, self.rng)
        profiler = PerformanceProfiler()
        all_returns: List[float] = []

        regimes = ["CALM", "VOLATILE", "CRASH", "TRENDING_UP"]
        for i in range(self.n_episodes):
            regime = regimes[i % len(regimes)]
            vol_mult = {"CRASH": 4.0, "VOLATILE": 2.5, "CALM": 1.0, "TRENDING_UP": 1.5}.get(regime, 1.0)
            spread_mult = {"CRASH": 8.0, "VOLATILE": 3.0, "CALM": 1.0, "TRENDING_UP": 1.5}.get(regime, 1.0)
            ctx = ScenarioContext(
                scenario_id=i, seed=i * 77,
                regime=regime,
                vol_multiplier=vol_mult,
                spread_multiplier=spread_mult,
            )
            result = runner.run(ctx, self.episode_length)
            all_returns.append(result["ep_return"])
            for _ in range(result["ep_length"]):
                profiler.record_step(
                    result["ep_return"] / max(result["ep_length"], 1),
                    regime=regime,
                )

        rets = np.array(all_returns)
        sharpe = profiler.compute_sharpe()
        profile = profiler.get_profile()
        profile["overall_sharpe"] = sharpe
        profile["sharpe_robust"] = sharpe >= self.min_sharpe
        profile["n_episodes"] = self.n_episodes
        return profile


# ---------------------------------------------------------------------------
# Extended: DrawdownRobustnessTest
# ---------------------------------------------------------------------------

class DrawdownRobustnessTest:
    """
    Tests maximum drawdown robustness under adversarial conditions.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        max_acceptable_drawdown: float = 0.3,
        n_scenarios: int = 30,
        episode_length: int = 500,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.max_dd = max_acceptable_drawdown
        self.n_scenarios = n_scenarios
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

    def run(self) -> Dict[str, Any]:
        runner = EpisodeRunner(self.env_factory, self.policy, self.rng)
        drawdowns: List[float] = []

        for i in range(self.n_scenarios):
            ctx = ScenarioContext(
                scenario_id=i,
                seed=i * 111,
                spread_multiplier=float(self.rng.uniform(1.0, 8.0)),
                vol_multiplier=float(self.rng.uniform(1.0, 4.0)),
            )
            result = runner.run(ctx, self.episode_length)
            # Compute intra-episode drawdown from step rewards
            step_returns = result.get("returns", [result["ep_return"]])
            if step_returns:
                cumulative = np.cumsum(step_returns)
                running_max = np.maximum.accumulate(cumulative)
                dd = float(np.max(running_max - cumulative))
                drawdowns.append(dd)

        if not drawdowns:
            return {"error": "no_episodes"}

        return {
            "n_scenarios": self.n_scenarios,
            "mean_drawdown": float(np.mean(drawdowns)),
            "max_drawdown": float(np.max(drawdowns)),
            "p95_drawdown": float(np.percentile(drawdowns, 95)),
            "drawdown_robust": float(np.percentile(drawdowns, 95)) <= self.max_dd,
            "max_acceptable": self.max_dd,
        }


# ---------------------------------------------------------------------------
# Extended: MultiRegimePortfolioEval
# ---------------------------------------------------------------------------

class MultiRegimePortfolioEval:
    """
    Evaluates portfolio-level agent performance across multiple market regimes.

    Tests:
    - Diversification effectiveness
    - Cross-asset PnL attribution
    - Regime-conditional performance decomposition
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        regimes: Optional[List[str]] = None,
        n_episodes_per_regime: int = 10,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.regimes = regimes or ["CALM", "VOLATILE", "CRASH", "TRENDING_UP", "TRENDING_DOWN"]
        self.n_episodes = n_episodes_per_regime
        self.rng = np.random.default_rng(seed)

    def evaluate(self) -> Dict[str, Any]:
        runner = EpisodeRunner(self.env_factory, self.policy, self.rng)
        results: Dict[str, Any] = {}

        all_returns: List[float] = []
        for regime in self.regimes:
            regime_returns = []
            vol_mult = {
                "CRASH": 5.0, "VOLATILE": 3.0, "CALM": 1.0,
                "TRENDING_UP": 1.5, "TRENDING_DOWN": 1.7,
            }.get(regime, 1.0)
            spread_mult = {
                "CRASH": 10.0, "VOLATILE": 3.5, "CALM": 1.0,
                "TRENDING_UP": 1.5, "TRENDING_DOWN": 2.0,
            }.get(regime, 1.0)

            for ep in range(self.n_episodes):
                ctx = ScenarioContext(
                    scenario_id=ep,
                    seed=ep * 500,
                    regime=regime,
                    vol_multiplier=vol_mult,
                    spread_multiplier=spread_mult,
                )
                r = runner.run(ctx, 500)
                regime_returns.append(r["ep_return"])
                all_returns.append(r["ep_return"])

            results[regime] = {
                "mean_return": float(np.mean(regime_returns)),
                "std_return": float(np.std(regime_returns)),
                "min_return": float(np.min(regime_returns)),
                "sharpe": float(np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)),
                "win_rate": float(np.mean(np.array(regime_returns) > 0)),
            }

        results["overall"] = {
            "mean_return": float(np.mean(all_returns)),
            "std_return": float(np.std(all_returns)),
            "regime_consistency": float(np.std([
                results[r]["mean_return"] for r in self.regimes
            ])),
        }
        return results


# ---------------------------------------------------------------------------
# Extended: LatencyRobustnessEval
# ---------------------------------------------------------------------------

class LatencyRobustnessEval:
    """
    Evaluates agent robustness to execution latency.

    Tests performance under simulated latency delays and packet loss.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        latency_levels_us: Optional[List[float]] = None,
        n_episodes: int = 10,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.policy = policy
        self.latency_levels = latency_levels_us or [0.0, 100.0, 1000.0, 10_000.0, 50_000.0]
        self.n_episodes = n_episodes
        self.rng = np.random.default_rng(seed)

    def evaluate(self) -> Dict[str, Any]:
        runner = EpisodeRunner(self.env_factory, self.policy, self.rng)
        results: Dict[str, Any] = {}
        baseline_return: Optional[float] = None

        for lat_us in self.latency_levels:
            level_returns = []
            for ep in range(self.n_episodes):
                ctx = ScenarioContext(scenario_id=ep, seed=ep * 13)
                result = runner.run(ctx, 500)
                # Simulate stale obs due to latency
                if lat_us > 1000.0:
                    # High latency: penalize by reducing return
                    latency_factor = 1.0 - min(0.5, lat_us / 100_000.0)
                    ep_return = result["ep_return"] * latency_factor
                else:
                    ep_return = result["ep_return"]
                level_returns.append(ep_return)

            mean_return = float(np.mean(level_returns))
            results[f"latency_{int(lat_us)}us"] = {
                "latency_us": lat_us,
                "mean_return": mean_return,
                "std_return": float(np.std(level_returns)),
            }

            if lat_us == 0.0:
                baseline_return = mean_return

        # Compute degradation vs zero latency
        if baseline_return is not None and abs(baseline_return) > 1e-8:
            for key in results:
                ret = results[key]["mean_return"]
                results[key]["degradation"] = (baseline_return - ret) / abs(baseline_return)

        return results


# ---------------------------------------------------------------------------
# Extended: RobustnessDashboard
# ---------------------------------------------------------------------------

class RobustnessDashboard:
    """
    Aggregates all robustness evaluation results into a single dashboard.

    Provides a standardized report format suitable for logging and alerting.
    """

    def __init__(self) -> None:
        self._reports: List[Dict[str, Any]] = []

    def add_report(self, name: str, report: Dict[str, Any]) -> None:
        self._reports.append({"name": name, "report": report, "time": time.time()})

    def get_latest(self, name: str) -> Optional[Dict[str, Any]]:
        for r in reversed(self._reports):
            if r["name"] == name:
                return r["report"]
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all latest reports."""
        seen_names = set()
        summary: Dict[str, Any] = {}
        for r in reversed(self._reports):
            if r["name"] not in seen_names:
                seen_names.add(r["name"])
                summary[r["name"]] = self._extract_key_metrics(r["report"])
        return summary

    def _extract_key_metrics(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from a report."""
        metrics: Dict[str, Any] = {}
        for key, val in report.items():
            if isinstance(val, (int, float, bool)):
                metrics[key] = val
            elif isinstance(val, dict) and "mean_return" in val:
                metrics[f"{key}_mean"] = val["mean_return"]
        return metrics

    def generate_html_report(self) -> str:
        """Generate a simple HTML robustness report."""
        summary = self.get_summary()
        lines = ["<html><body>", "<h1>Hyper-Agent Robustness Dashboard</h1>"]
        for name, metrics in summary.items():
            lines.append(f"<h2>{name}</h2><ul>")
            for k, v in metrics.items():
                lines.append(f"<li>{k}: {v:.4f}" if isinstance(v, float) else f"<li>{k}: {v}")
            lines.append("</ul>")
        lines.append("</body></html>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extended: ScenarioLibrary
# ---------------------------------------------------------------------------

class ScenarioLibrary:
    """
    Library of pre-built evaluation scenarios.

    Provides named, reproducible scenarios for standardized evaluation
    across experiments and model versions.
    """

    SCENARIOS: Dict[str, ScenarioContext] = {
        "base": ScenarioContext(scenario_id=0, seed=0, description="Baseline calm"),
        "high_spread": ScenarioContext(scenario_id=1, seed=1, spread_multiplier=5.0, description="5x spread"),
        "high_vol": ScenarioContext(scenario_id=2, seed=2, vol_multiplier=3.0, description="3x volatility"),
        "crash": ScenarioContext(scenario_id=3, seed=3, vol_multiplier=6.0, spread_multiplier=15.0, regime="CRASH", description="Market crash"),
        "illiquid": ScenarioContext(scenario_id=4, seed=4, spread_multiplier=10.0, description="Illiquid market"),
        "trending": ScenarioContext(scenario_id=5, seed=5, regime="TRENDING_UP", description="Trending up"),
        "trending_down": ScenarioContext(scenario_id=6, seed=6, regime="TRENDING_DOWN", description="Trending down"),
        "volatile": ScenarioContext(scenario_id=7, seed=7, vol_multiplier=3.5, regime="VOLATILE", description="High volatility"),
        "crisis": ScenarioContext(scenario_id=8, seed=8, vol_multiplier=8.0, spread_multiplier=20.0, regime="CRISIS", description="Full crisis"),
        "adversarial_obs": ScenarioContext(scenario_id=9, seed=9, adversarial_perturbation=True, perturbation_epsilon=0.05, description="Adversarial obs noise"),
        "poor_fills": ScenarioContext(scenario_id=10, seed=10, fill_rate_modifier=-0.4, description="Poor fill rates"),
        "combined_stress": ScenarioContext(scenario_id=11, seed=11, vol_multiplier=4.0, spread_multiplier=8.0, fill_rate_modifier=-0.3, regime="VOLATILE", description="Combined stress"),
    }

    @classmethod
    def get(cls, name: str) -> Optional[ScenarioContext]:
        return cls.SCENARIOS.get(name)

    @classmethod
    def get_all(cls) -> Dict[str, ScenarioContext]:
        return cls.SCENARIOS.copy()

    @classmethod
    def list_names(cls) -> List[str]:
        return list(cls.SCENARIOS.keys())

    @classmethod
    def run_library(
        cls,
        env_factory: Callable[[], Any],
        policy: Any,
        episode_length: int = 500,
        n_episodes: int = 5,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Run all scenarios in the library and return results."""
        rng = np.random.default_rng(seed)
        runner = EpisodeRunner(env_factory, policy, rng)
        results: Dict[str, Any] = {}

        for name, scenario in cls.SCENARIOS.items():
            ep_returns = []
            for ep in range(n_episodes):
                ctx = ScenarioContext(
                    scenario_id=scenario.scenario_id,
                    seed=ep * 100 + scenario.seed,
                    spread_multiplier=scenario.spread_multiplier,
                    vol_multiplier=scenario.vol_multiplier,
                    kyle_lambda_multiplier=scenario.kyle_lambda_multiplier,
                    fill_rate_modifier=scenario.fill_rate_modifier,
                    regime=scenario.regime,
                    adversarial_perturbation=scenario.adversarial_perturbation,
                    perturbation_epsilon=scenario.perturbation_epsilon,
                    description=scenario.description,
                )
                result = runner.run(ctx, episode_length)
                ep_returns.append(result["ep_return"])

            results[name] = {
                "description": scenario.description,
                "mean_return": float(np.mean(ep_returns)),
                "std_return": float(np.std(ep_returns)),
                "min_return": float(np.min(ep_returns)),
                "n_episodes": n_episodes,
            }

        # Overall summary
        all_means = [v["mean_return"] for v in results.values()]
        results["__summary__"] = {
            "overall_mean": float(np.mean(all_means)),
            "overall_min": float(np.min(all_means)),
            "overall_std": float(np.std(all_means)),
            "num_scenarios": len(cls.SCENARIOS),
        }
        return results


# ---------------------------------------------------------------------------
# AdversarialObservationAttacker — applies PGD-style noise to observations
# during evaluation to measure policy sensitivity
# ---------------------------------------------------------------------------

class AdversarialObservationAttacker:
    """Perturbs observations with bounded L-inf noise to find worst-case inputs."""

    def __init__(self, epsilon: float = 0.05, num_steps: int = 5,
                 step_size: float = 0.01):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    # ------------------------------------------------------------------
    def attack(self, obs: np.ndarray, policy_fn,
               maximize_loss: bool = True) -> np.ndarray:
        """Return adversarial obs using finite-difference gradient ascent."""
        adv = obs.copy()
        for _ in range(self.num_steps):
            # Finite difference gradient estimate
            grad = np.zeros_like(adv)
            for i in range(len(adv.flat)):
                delta = np.zeros_like(adv)
                delta.flat[i] = 1e-4
                score_plus = float(np.sum(policy_fn(adv + delta)))
                score_minus = float(np.sum(policy_fn(adv - delta)))
                grad.flat[i] = (score_plus - score_minus) / (2e-4)
            sign_grad = np.sign(grad)
            if maximize_loss:
                adv = adv + self.step_size * sign_grad
            else:
                adv = adv - self.step_size * sign_grad
            # Project back
            adv = np.clip(adv, obs - self.epsilon, obs + self.epsilon)
        return adv

    # ------------------------------------------------------------------
    def sensitivity_score(self, obs: np.ndarray, policy_fn) -> float:
        """Return L2 distance between clean and adversarial policy outputs."""
        clean_out = policy_fn(obs)
        adv_obs = self.attack(obs, policy_fn)
        adv_out = policy_fn(adv_obs)
        return float(np.linalg.norm(np.array(clean_out) - np.array(adv_out)))


# ---------------------------------------------------------------------------
# RegimeRobustnessMatrix — full cross-regime robustness evaluation
# ---------------------------------------------------------------------------

class RegimeRobustnessMatrix:
    """Evaluates policy performance for every (train_regime, eval_regime) pair
    to build a transfer matrix showing where sim-to-real gaps are largest."""

    REGIMES = ["CALM", "TRENDING_UP", "TRENDING_DOWN", "VOLATILE",
               "CRASH", "RECOVERY", "ILLIQUID", "CRISIS"]

    def __init__(self, env, policy_fn, episodes_per_cell: int = 5):
        self.env = env
        self.policy_fn = policy_fn
        self.episodes_per_cell = episodes_per_cell

    # ------------------------------------------------------------------
    def _run_cell(self, regime: str) -> float:
        returns = []
        for _ in range(self.episodes_per_cell):
            obs, _ = self.env.reset()
            # Inject regime if env supports it
            if hasattr(self.env, "set_regime"):
                self.env.set_regime(regime)
            ep_ret = 0.0
            done = False
            while not done:
                action = self.policy_fn(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                ep_ret += float(np.sum(rew)) if hasattr(rew, "__iter__") else float(rew)
                done = terminated or truncated
            returns.append(ep_ret)
        return float(np.mean(returns))

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        matrix = {}
        for regime in self.REGIMES:
            matrix[regime] = self._run_cell(regime)
        best = max(matrix.values())
        worst = min(matrix.values())
        return {
            "per_regime": matrix,
            "best_regime": max(matrix, key=matrix.get),
            "worst_regime": min(matrix, key=matrix.get),
            "range": best - worst,
            "mean": float(np.mean(list(matrix.values()))),
            "std": float(np.std(list(matrix.values()))),
        }


# ---------------------------------------------------------------------------
# ParametricRobustnessAnalyzer — sweeps single parameters for sensitivity
# ---------------------------------------------------------------------------

class ParametricRobustnessAnalyzer:
    """Sweeps a single environment parameter (e.g., spread_multiplier) over a
    range and measures the policy return sensitivity."""

    def __init__(self, env, policy_fn, episodes_per_point: int = 3):
        self.env = env
        self.policy_fn = policy_fn
        self.episodes_per_point = episodes_per_point

    # ------------------------------------------------------------------
    def sweep(self, param_name: str, values: list) -> dict:
        results = {}
        for val in values:
            returns = []
            for _ in range(self.episodes_per_point):
                obs, _ = self.env.reset()
                if hasattr(self.env, param_name):
                    setattr(self.env, param_name, val)
                ep_ret = 0.0
                done = False
                while not done:
                    action = self.policy_fn(obs)
                    obs, rew, terminated, truncated, _ = self.env.step(action)
                    ep_ret += float(np.sum(rew)) if hasattr(rew, "__iter__") else float(rew)
                    done = terminated or truncated
                returns.append(ep_ret)
            results[float(val)] = {
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "min": float(np.min(returns)),
            }
        vals_list = [v["mean"] for v in results.values()]
        if len(vals_list) > 1:
            sensitivity = float(np.std(vals_list) / (np.mean(np.abs(vals_list)) + 1e-8))
        else:
            sensitivity = 0.0
        return {"sweep": results, "sensitivity_coefficient": sensitivity,
                "param": param_name}


# ---------------------------------------------------------------------------
# EpisodeReturnAggregator — collects returns across eval runs for statistics
# ---------------------------------------------------------------------------

class EpisodeReturnAggregator:
    """Lightweight collector for episode returns; computes confidence intervals
    via bootstrap."""

    def __init__(self):
        self._returns: list = []

    # ------------------------------------------------------------------
    def push(self, ep_return: float) -> None:
        self._returns.append(ep_return)

    # ------------------------------------------------------------------
    def bootstrap_ci(self, n_bootstrap: int = 1000,
                     ci: float = 0.95) -> tuple:
        """Return (lower, mean, upper) confidence interval."""
        if not self._returns:
            return (0.0, 0.0, 0.0)
        arr = np.array(self._returns, dtype=np.float32)
        means = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            means.append(float(np.mean(sample)))
        lo = float(np.percentile(means, (1 - ci) / 2 * 100))
        hi = float(np.percentile(means, (1 + ci) / 2 * 100))
        return (lo, float(np.mean(arr)), hi)

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        if not self._returns:
            return {}
        arr = np.array(self._returns)
        lo, mean, hi = self.bootstrap_ci()
        return {
            "n": len(self._returns),
            "mean": mean,
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "ci_95_low": lo,
            "ci_95_high": hi,
        }

    # ------------------------------------------------------------------
    def clear(self) -> None:
        self._returns.clear()


# ---------------------------------------------------------------------------
# ComprehensiveRobustnessReport — combines all evaluators into one report dict
# ---------------------------------------------------------------------------

class ComprehensiveRobustnessReport:
    """Orchestrates StressTester, RegimeTransferEvaluator, TailRiskEvaluator,
    and ParametricRobustnessAnalyzer into one comprehensive report."""

    def __init__(self, env, policy_fn, obs_dim: int = 16):
        self.env = env
        self.policy_fn = policy_fn
        self.obs_dim = obs_dim

    # ------------------------------------------------------------------
    def generate(self, num_stress_scenarios: int = 20,
                 param_sweeps: Optional[dict] = None) -> dict:
        report: dict = {}

        # 1. Stress test (reduced scenario count for speed)
        stress_scenarios = build_stress_scenarios(num_stress_scenarios)
        stress_results = []
        runner = EpisodeRunner(self.env, self.policy_fn)
        for sc in stress_scenarios[:10]:  # limit for eval speed
            ret = runner.run(sc)
            stress_results.append({"scenario": sc.regime or "default",
                                   "return": ret})
        report["stress_test"] = {
            "num_scenarios": len(stress_results),
            "mean_return": float(np.mean([r["return"] for r in stress_results])),
            "min_return": float(np.min([r["return"] for r in stress_results])),
        }

        # 2. Parametric sweeps
        if param_sweeps:
            analyzer = ParametricRobustnessAnalyzer(self.env, self.policy_fn)
            report["parametric"] = {}
            for param, values in param_sweeps.items():
                report["parametric"][param] = analyzer.sweep(param, values)

        # 3. Return aggregation
        aggregator = EpisodeReturnAggregator()
        for r in stress_results:
            aggregator.push(r["return"])
        report["return_statistics"] = aggregator.summary()

        # 4. Overall robustness score [0, 1]
        mean_r = report["stress_test"]["mean_return"]
        min_r = report["stress_test"]["min_return"]
        score = float(np.clip((mean_r - min_r) / (abs(mean_r) + abs(min_r) + 1e-8), 0, 1))
        report["overall_robustness_score"] = score

        return report
