"""
research/agent_training/cli.py

Click CLI for agent training operations.

Commands:
    train run      — Train an agent on historical data
    train eval     — Evaluate a saved agent
    train compare  — Compare multiple agents
    train hyperopt — Run hyperparameter search

Usage examples:
    python -m research.agent_training.cli run --instrument BTC/USD --agent d3qn --episodes 1000
    python -m research.agent_training.cli eval --weights checkpoints/best_agent --agent d3qn
    python -m research.agent_training.cli compare --weights-dir checkpoints/ --agents d3qn,ddqn,td3
    python -m research.agent_training.cli hyperopt --agent d3qn --trials 50
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import click
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_price_data(data_path: str) -> pd.DataFrame:
    """Load price data from CSV or Parquet."""
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path, parse_dates=True)


def _make_synthetic_features(price_df: pd.DataFrame) -> np.ndarray:
    """
    Generate a minimal synthetic feature matrix from a price DataFrame.

    Used when no pre-computed feature file is provided.
    Returns an array of shape (T, 23) compatible with lib/features.py layout.
    """
    n = len(price_df)
    closes = price_df["close"].values.astype(np.float64)
    features = np.zeros((n, 23), dtype=np.float64)

    # RSI proxy (index 0)
    delta = np.diff(closes, prepend=closes[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.convolve(gain, np.ones(14) / 14, mode="same")
    avg_loss = np.convolve(loss, np.ones(14) / 14, mode="same") + 1e-12
    rs = avg_gain / avg_loss
    features[:, 0] = 1.0 - 1.0 / (1.0 + rs)

    # Log return (index 21)
    features[1:, 21] = np.log(closes[1:] / (closes[:-1] + 1e-12))

    # ATR% proxy (index 6)
    if "high" in price_df.columns and "low" in price_df.columns:
        atr = (price_df["high"].values - price_df["low"].values).astype(np.float64)
        features[:, 6] = atr / (closes + 1e-12)
    else:
        roll_std = pd.Series(closes).rolling(14).std().fillna(0).values
        features[:, 6] = roll_std / (closes + 1e-12)

    return features


def _build_agent(agent_type: str, obs_dim: int, **kwargs):
    """Factory for agent construction by name."""
    from research.agent_training.agents import (
        DQNAgent, DDQNAgent, D3QNAgent, TD3Agent, PPOAgent, EnsembleAgent
    )
    mapping = {
        "dqn": DQNAgent,
        "ddqn": DDQNAgent,
        "d3qn": D3QNAgent,
        "td3": TD3Agent,
        "ppo": PPOAgent,
        "ensemble": EnsembleAgent,
    }
    cls = mapping.get(agent_type.lower())
    if cls is None:
        raise click.BadParameter(
            f"Unknown agent '{agent_type}'. Choose from: {list(mapping.keys())}"
        )
    return cls(obs_dim=obs_dim, **kwargs)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """SRFM-Lab Agent Training CLI."""
    pass


# ---------------------------------------------------------------------------
# train run
# ---------------------------------------------------------------------------


@cli.command("run")
@click.option("--data", default=None, help="Path to price data CSV/Parquet.")
@click.option("--features", default=None, help="Path to features .npy file.")
@click.option("--instrument", default="BTC/USD", show_default=True, help="Instrument label (informational).")
@click.option("--agent", default="d3qn", show_default=True, help="Agent type: dqn | ddqn | d3qn | td3 | ppo | ensemble.")
@click.option("--episodes", default=500, show_default=True, type=int, help="Training episodes.")
@click.option("--batch-size", default=64, show_default=True, type=int, help="Replay buffer batch size.")
@click.option("--lr", default=1e-3, show_default=True, type=float, help="Learning rate.")
@click.option("--gamma", default=0.99, show_default=True, type=float, help="Discount factor.")
@click.option("--eval-every", default=50, show_default=True, type=int, help="Evaluate every N episodes.")
@click.option("--checkpoint-dir", default="checkpoints", show_default=True, help="Checkpoint directory.")
@click.option("--reward", default="sharpe", show_default=True, help="Reward shaping: sharpe | sortino | log_return | calmar.")
@click.option("--equity", default=1_000_000, show_default=True, type=float, help="Starting equity.")
@click.option("--per/--no-per", default=False, show_default=True, help="Use Prioritized Experience Replay.")
@click.option("--curriculum/--no-curriculum", default=True, show_default=True, help="Use curriculum learning.")
@click.option("--plot", is_flag=True, default=False, help="Plot training curves after training.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--verbose/--quiet", default=True, show_default=True)
def run_cmd(
    data, features, instrument, agent, episodes, batch_size, lr, gamma,
    eval_every, checkpoint_dir, reward, equity, per, curriculum, plot, seed, verbose
):
    """Train an RL agent on historical price data."""
    from research.agent_training.environment import TradingEnvironment, EnvironmentConfig
    from research.agent_training.trainer import AgentTrainer, TrainingConfig

    click.echo(f"Training {agent.upper()} agent on {instrument} for {episodes} episodes.")

    # Load data
    if data is None:
        click.echo("No --data provided. Generating synthetic price data for demonstration.")
        rng = np.random.default_rng(seed or 42)
        n = 5000
        prices = np.cumprod(1.0 + rng.normal(0.0001, 0.015, n)) * 50_000
        price_df = pd.DataFrame({"close": prices})
        feat_matrix = _make_synthetic_features(price_df)
    else:
        price_df = _load_price_data(data)
        if features is not None:
            feat_matrix = np.load(features)
        else:
            click.echo("No --features provided. Generating synthetic features.")
            feat_matrix = _make_synthetic_features(price_df)

    env_cfg = EnvironmentConfig(
        starting_equity=equity,
        reward_shaping=reward,
        seed=seed,
    )
    env = TradingEnvironment(price_df, feat_matrix, config=env_cfg)
    obs_dim = env.obs_dim

    try:
        rl_agent = _build_agent(agent, obs_dim, lr=lr, gamma=gamma)
    except Exception as e:
        raise click.ClickException(str(e))

    cfg = TrainingConfig(
        n_episodes=episodes,
        eval_every=eval_every,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        checkpoint_dir=checkpoint_dir,
        use_per=per,
        use_curriculum=curriculum,
        verbose=verbose,
        seed=seed,
    )

    trainer = AgentTrainer(rl_agent, env, cfg)
    result = trainer.train()

    click.echo(f"\nTraining complete.")
    click.echo(f"  Best eval return:  {result.best_eval_return:.4f}")
    click.echo(f"  Best episode:      {result.best_episode}")
    click.echo(f"  Total steps:       {result.total_steps}")
    click.echo(f"  Training time:     {result.training_time_s:.1f}s")
    click.echo(f"  Early stopped:     {result.early_stopped}")
    click.echo(f"  Best weights:      {result.best_weights_path}")

    if plot:
        trainer.plot_training_curves(result, save_path=os.path.join(checkpoint_dir, "training_curves.png"))


# ---------------------------------------------------------------------------
# train eval
# ---------------------------------------------------------------------------


@cli.command("eval")
@click.option("--data", default=None, help="Path to price data CSV/Parquet.")
@click.option("--features", default=None, help="Path to features .npy file.")
@click.option("--weights", required=False, default=None, help="Path to saved agent weights (.npz).")
@click.option("--agent", default="d3qn", show_default=True)
@click.option("--episodes", default=50, show_default=True, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--regime-breakdown/--no-regime-breakdown", default=False)
@click.option("--robustness/--no-robustness", default=False)
def eval_cmd(data, features, weights, agent, episodes, seed, regime_breakdown, robustness):
    """Evaluate a saved agent and print performance metrics."""
    from research.agent_training.environment import TradingEnvironment, EnvironmentConfig
    from research.agent_training.evaluation import AgentEvaluator

    click.echo(f"Evaluating {agent.upper()} agent...")

    if data is None:
        rng = np.random.default_rng(seed or 42)
        n = 3000
        prices = np.cumprod(1.0 + rng.normal(0.0001, 0.015, n)) * 50_000
        price_df = pd.DataFrame({"close": prices})
        feat_matrix = _make_synthetic_features(price_df)
    else:
        price_df = _load_price_data(data)
        feat_matrix = np.load(features) if features else _make_synthetic_features(price_df)

    env_cfg = EnvironmentConfig(seed=seed)
    env = TradingEnvironment(price_df, feat_matrix, config=env_cfg)

    rl_agent = _build_agent(agent, env.obs_dim)
    if weights is not None:
        if hasattr(rl_agent, "load"):
            rl_agent.load(weights)
            click.echo(f"Loaded weights from {weights}")

    evaluator = AgentEvaluator(n_eval_episodes=episodes, seed=seed, verbose=True)
    report = evaluator.full_evaluation_report(rl_agent, env, agent_name=agent, n_episodes=episodes)
    evaluator.print_report(report)

    if regime_breakdown and "regime" in price_df.columns:
        click.echo("\nRegime breakdown:")
        regime_results = evaluator.evaluate_on_regimes(rl_agent, price_df, feat_matrix)
        for regime, res in regime_results.items():
            click.echo(
                f"  {regime:20s}: return={res.mean_return:.4f}, "
                f"sharpe={res.mean_sharpe:.3f}, dd={res.mean_max_drawdown:.3f}"
            )

    if robustness:
        click.echo("\nRobustness evaluation...")
        rob = evaluator.evaluate_robustness(rl_agent, env, n_perturb=30, noise_scale=0.02)
        click.echo(f"  Robustness score: {rob.robustness_score:.3f}")
        click.echo(f"  Base return:      {rob.base_return:.4f}")
        click.echo(f"  Perturbed return: {rob.mean_perturbed_return:.4f} ± {rob.std_perturbed_return:.4f}")


# ---------------------------------------------------------------------------
# train compare
# ---------------------------------------------------------------------------


@cli.command("compare")
@click.option("--data", default=None, help="Path to price data CSV/Parquet.")
@click.option("--features", default=None, help="Path to features .npy file.")
@click.option("--agents", default="d3qn,ddqn,td3", show_default=True, help="Comma-separated agent types.")
@click.option("--episodes", default=50, show_default=True, type=int)
@click.option("--train-episodes", default=200, show_default=True, type=int, help="Episodes to train each agent first.")
@click.option("--seed", default=None, type=int)
@click.option("--output", default=None, help="Output CSV path for comparison table.")
def compare_cmd(data, features, agents, episodes, train_episodes, seed, output):
    """Train and compare multiple agent types on the same data."""
    from research.agent_training.environment import TradingEnvironment, EnvironmentConfig
    from research.agent_training.trainer import AgentTrainer, TrainingConfig
    from research.agent_training.evaluation import AgentEvaluator

    agent_types = [a.strip().lower() for a in agents.split(",")]
    click.echo(f"Comparing agents: {agent_types}")

    if data is None:
        rng = np.random.default_rng(seed or 42)
        n = 4000
        prices = np.cumprod(1.0 + rng.normal(0.0001, 0.015, n)) * 50_000
        price_df = pd.DataFrame({"close": prices})
        feat_matrix = _make_synthetic_features(price_df)
    else:
        price_df = _load_price_data(data)
        feat_matrix = np.load(features) if features else _make_synthetic_features(price_df)

    env_cfg = EnvironmentConfig(seed=seed)

    trained_agents = {}
    for at in agent_types:
        click.echo(f"Training {at.upper()} for {train_episodes} episodes...")
        env = TradingEnvironment(price_df, feat_matrix, config=env_cfg)
        try:
            agent_obj = _build_agent(at, env.obs_dim)
        except click.BadParameter as e:
            click.echo(f"  Skipping {at}: {e}")
            continue

        cfg = TrainingConfig(
            n_episodes=train_episodes,
            eval_every=train_episodes + 1,
            verbose=False,
            seed=seed,
        )
        trainer = AgentTrainer(agent_obj, env, cfg)
        trainer.train()
        trained_agents[at] = agent_obj
        click.echo(f"  Done.")

    click.echo(f"\nComparing {len(trained_agents)} agents over {episodes} eval episodes...")
    eval_env = TradingEnvironment(price_df, feat_matrix, config=env_cfg)
    evaluator = AgentEvaluator(seed=seed)
    comparison = evaluator.compare_agents(trained_agents, eval_env, n_episodes=episodes)

    click.echo("\nComparison Results:")
    click.echo(comparison.comparison_df.to_string())

    if output is not None:
        comparison.comparison_df.to_csv(output)
        click.echo(f"\nSaved to {output}")


# ---------------------------------------------------------------------------
# train hyperopt
# ---------------------------------------------------------------------------


@cli.command("hyperopt")
@click.option("--data", default=None, help="Path to price data CSV/Parquet.")
@click.option("--features", default=None, help="Path to features .npy file.")
@click.option("--agent", default="d3qn", show_default=True)
@click.option("--trials", default=30, show_default=True, type=int, help="Number of random trials.")
@click.option("--train-episodes", default=100, show_default=True, type=int)
@click.option("--cv-folds", default=3, show_default=True, type=int, help="Cross-validation folds for best params.")
@click.option("--seed", default=None, type=int)
@click.option("--output", default="hyperopt_result.json", show_default=True, help="Output JSON path.")
def hyperopt_cmd(data, features, agent, trials, train_episodes, cv_folds, seed, output):
    """Run hyperparameter optimisation for an agent."""
    import json
    from research.agent_training.environment import TradingEnvironment, EnvironmentConfig
    from research.agent_training.hyperopt import AgentHyperSearch
    from research.agent_training.agents import (
        DQNAgent, DDQNAgent, D3QNAgent, TD3Agent, PPOAgent, EnsembleAgent
    )

    agent_map = {
        "dqn": DQNAgent, "ddqn": DDQNAgent, "d3qn": D3QNAgent,
        "td3": TD3Agent, "ppo": PPOAgent, "ensemble": EnsembleAgent,
    }
    agent_cls = agent_map.get(agent.lower())
    if agent_cls is None:
        raise click.ClickException(f"Unknown agent: {agent}")

    if data is None:
        rng = np.random.default_rng(seed or 42)
        n = 4000
        prices = np.cumprod(1.0 + rng.normal(0.0001, 0.015, n)) * 50_000
        price_df = pd.DataFrame({"close": prices})
        feat_matrix = _make_synthetic_features(price_df)
    else:
        price_df = _load_price_data(data)
        feat_matrix = np.load(features) if features else _make_synthetic_features(price_df)

    def env_fn():
        return TradingEnvironment(price_df, feat_matrix, config=EnvironmentConfig(seed=seed))

    searcher = AgentHyperSearch(verbose=True, seed=seed)

    click.echo(f"Running {trials} random trials for {agent.upper()}...")
    result = searcher.random_search(
        env_fn=env_fn,
        agent_class=agent_cls,
        n_trials=trials,
        train_episodes=train_episodes,
    )

    click.echo(f"\nBest score: {result.best_score:.4f}")
    click.echo(f"Best params: {result.best_params}")

    # Cross-validate best params
    if cv_folds > 0:
        click.echo(f"\nCross-validating best params ({cv_folds} folds)...")
        cv_result = searcher.cross_validate_agent(
            agent_class=agent_cls,
            params=result.best_params,
            env_fn=env_fn,
            n_folds=cv_folds,
            train_episodes=train_episodes,
        )
        click.echo(f"CV Score: {cv_result.mean_score:.4f} ± {cv_result.std_score:.4f}")

    # Save result
    out_data = {
        "agent": agent,
        "n_trials": trials,
        "best_score": float(result.best_score),
        "best_params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                        for k, v in result.best_params.items()},
        "top5": [
            {"trial_id": t.trial_id, "score": float(t.score), "params": {
                k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                for k, v in t.params.items()
            }}
            for t in result.top_k_trials(5)
        ],
    }
    with open(output, "w") as f:
        import json
        json.dump(out_data, f, indent=2)
    click.echo(f"\nResults saved to {output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
