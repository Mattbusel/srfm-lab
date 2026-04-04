"""
research/portfolio_lab/cli.py

Click CLI for portfolio lab operations.

Commands:
    portfolio construct  — Build a portfolio from returns data
    portfolio risk       — Run risk analytics on a portfolio
    portfolio rebalance  — Compare rebalancing strategies

Usage examples:
    python -m research.portfolio_lab.cli construct --returns returns.csv --method hrp --output weights.csv
    python -m research.portfolio_lab.cli risk --returns returns.csv --weights weights.csv
    python -m research.portfolio_lab.cli rebalance --returns returns.csv --weights weights.csv
"""

from __future__ import annotations

import json
import sys
from typing import Optional

import click
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_returns(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_weights(path: str) -> dict[str, float]:
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    df = pd.read_csv(path, index_col=0)
    if "weight" in df.columns:
        return df["weight"].to_dict()
    return df.iloc[:, 0].to_dict()


def _save_weights(weights: dict[str, float], path: str) -> None:
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(weights, f, indent=2)
    else:
        pd.DataFrame.from_dict(
            {"asset": list(weights.keys()), "weight": list(weights.values())}
        ).to_csv(path, index=False)


def _make_synthetic_returns(n_assets: int = 5, n_periods: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic correlated returns for testing."""
    rng = np.random.default_rng(seed)
    # Cholesky-correlated returns
    corr = np.full((n_assets, n_assets), 0.4)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_periods, n_assets))
    R = z @ L.T * 0.015 + 0.0003  # ~15% ann vol, ~7% ann ret
    cols = [f"Asset_{chr(65+i)}" for i in range(n_assets)]
    return pd.DataFrame(R, columns=cols)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """SRFM-Lab Portfolio Construction CLI."""
    pass


# ---------------------------------------------------------------------------
# portfolio construct
# ---------------------------------------------------------------------------


@cli.command("construct")
@click.option("--returns", default=None, help="Path to returns CSV (rows=dates, cols=assets).")
@click.option("--method", default="hrp", show_default=True,
              help="Method: equal | inv_vol | min_var | max_sharpe | hrp | black_litterman | risk_parity | kelly.")
@click.option("--output", default="weights.csv", show_default=True, help="Output weights file (.csv or .json).")
@click.option("--cov-method", default="ledoit_wolf", show_default=True,
              help="Covariance estimation: sample | ledoit_wolf | shrinkage | oas.")
@click.option("--risk-free", default=0.04, show_default=True, type=float, help="Annual risk-free rate (for MaxSharpe).")
@click.option("--long-only/--allow-short", default=True, show_default=True)
@click.option("--max-weight", default=1.0, show_default=True, type=float, help="Max weight per asset.")
@click.option("--kelly-fraction", default=0.5, show_default=True, type=float, help="Kelly fraction (for Kelly method).")
@click.option("--frontier", is_flag=True, default=False, help="Plot efficient frontier (MaxSharpe only).")
@click.option("--verbose/--quiet", default=True)
def construct_cmd(returns, method, output, cov_method, risk_free, long_only, max_weight,
                  kelly_fraction, frontier, verbose):
    """Construct a portfolio using the specified method."""
    from research.portfolio_lab.construction import make_portfolio, MaxSharpePortfolio

    if returns is None:
        click.echo("No --returns provided. Using synthetic data (5 assets, 500 periods).")
        returns_df = _make_synthetic_returns()
    else:
        returns_df = _load_returns(returns)

    click.echo(f"Building {method.upper()} portfolio on {returns_df.shape[1]} assets, {len(returns_df)} periods.")

    try:
        kwargs: dict = {"cov_method": cov_method}
        if method in ("max_sharpe", "tangency"):
            kwargs["risk_free_rate"] = risk_free
            kwargs["long_only"] = long_only
            kwargs["max_weight"] = max_weight
        elif method in ("min_var", "min_variance", "gmv"):
            kwargs["long_only"] = long_only
            kwargs["max_weight"] = max_weight
        elif method in ("kelly",):
            kwargs["fraction"] = kelly_fraction
            kwargs["long_only"] = long_only
            kwargs["max_weight"] = max_weight

        portfolio = make_portfolio(method, **kwargs)
        weights = portfolio.fit(returns_df)
    except Exception as e:
        raise click.ClickException(f"Portfolio construction failed: {e}")

    if verbose:
        click.echo("\nPortfolio weights:")
        for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
            click.echo(f"  {asset:30s}: {w:.4f} ({w*100:.1f}%)")

    _save_weights(weights, output)
    click.echo(f"\nWeights saved to {output}")

    if frontier and method in ("max_sharpe", "tangency"):
        assert isinstance(portfolio, MaxSharpePortfolio)
        eff = portfolio.efficient_frontier(returns_df)
        click.echo(f"\nEfficient frontier computed ({len(eff)} points).")
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(eff["volatility"], eff["return"], c=eff["sharpe"], cmap="viridis", s=20)
            ax.set_xlabel("Volatility")
            ax.set_ylabel("Return")
            ax.set_title("Efficient Frontier")
            plt.colorbar(ax.collections[0], ax=ax, label="Sharpe")
            plt.tight_layout()
            plt.savefig("efficient_frontier.png", dpi=150)
            click.echo("Efficient frontier saved to efficient_frontier.png")
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# portfolio risk
# ---------------------------------------------------------------------------


@cli.command("risk")
@click.option("--returns", default=None, help="Path to returns CSV.")
@click.option("--weights", default=None, help="Path to weights file (.csv or .json).")
@click.option("--benchmark", default=None, help="Path to benchmark returns CSV (single column).")
@click.option("--confidence", default=0.99, show_default=True, type=float, help="VaR/CVaR confidence level.")
@click.option("--var-method", default="historical", show_default=True,
              help="VaR method: historical | parametric | cornish_fisher.")
@click.option("--cov-method", default="ledoit_wolf", show_default=True)
@click.option("--plot/--no-plot", default=False, show_default=True, help="Plot risk contributions.")
@click.option("--output", default=None, help="Output JSON report path.")
def risk_cmd(returns, weights, benchmark, confidence, var_method, cov_method, plot, output):
    """Run comprehensive portfolio risk analytics."""
    from research.portfolio_lab.risk import PortfolioRiskAnalyzer

    if returns is None:
        click.echo("No --returns provided. Using synthetic data.")
        returns_df = _make_synthetic_returns()
    else:
        returns_df = _load_returns(returns)

    if weights is None:
        click.echo("No --weights provided. Using equal weights.")
        n = returns_df.shape[1]
        w_dict = {col: 1.0 / n for col in returns_df.columns}
    else:
        w_dict = _load_weights(weights)

    bench_series = None
    if benchmark is not None:
        bench_df = _load_returns(benchmark)
        bench_series = bench_df.iloc[:, 0]

    analyzer = PortfolioRiskAnalyzer(cov_method=cov_method)
    report = analyzer.full_risk_report(w_dict, returns_df, benchmark_returns=bench_series)

    click.echo("\nPortfolio Risk Report:")
    click.echo(f"  Annualised Volatility:  {report['volatility']*100:.2f}%")
    click.echo(f"  Annualised Return:      {report['annualised_return']*100:.2f}%")
    click.echo(f"  Sharpe Ratio:           {report['sharpe']:.3f}")
    click.echo(f"  Sortino Ratio:          {report['sortino']:.3f}")
    click.echo(f"  Calmar Ratio:           {report['calmar']:.3f}")
    click.echo(f"  Max Drawdown:           {report['max_drawdown']*100:.2f}%")
    click.echo(f"  VaR {confidence*100:.0f}%:              {report['var_99']*100:.2f}%")
    click.echo(f"  CVaR {confidence*100:.0f}%:             {report['cvar_99']*100:.2f}%")

    if "beta" in report:
        click.echo(f"  Beta:                   {report['beta']:.3f}")
        click.echo(f"  Tracking Error:         {report['tracking_error']*100:.2f}%")
        click.echo(f"  Information Ratio:      {report['information_ratio']:.3f}")

    click.echo("\nComponent Risk Contributions:")
    for asset, crc in report["component_risk"].items():
        click.echo(f"  {asset:30s}: {crc*100:.1f}%")

    if output is not None:
        serialisable = {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in report.items()
                        if k != "component_risk"}
        serialisable["component_risk"] = {k: float(v) for k, v in report["component_risk"].items()}
        with open(output, "w") as f:
            json.dump(serialisable, f, indent=2)
        click.echo(f"\nReport saved to {output}")

    if plot:
        cov = analyzer.covariance_estimation(returns_df)
        w_arr = np.array([w_dict.get(a, 0.0) for a in returns_df.columns])
        analyzer.plot_risk_contributions(
            w_arr, cov * 252,
            asset_names=list(returns_df.columns),
            save_path="risk_contributions.png",
        )
        click.echo("Risk contributions chart saved to risk_contributions.png")


# ---------------------------------------------------------------------------
# portfolio rebalance
# ---------------------------------------------------------------------------


@cli.command("rebalance")
@click.option("--returns", default=None, help="Path to returns CSV.")
@click.option("--weights", default=None, help="Path to target weights file.")
@click.option("--strategy", default="all", show_default=True,
              help="Strategy: calendar | threshold | all. For calendar, use --freq.")
@click.option("--freq", default="monthly", show_default=True,
              help="Calendar frequency: daily | weekly | monthly | quarterly | annually.")
@click.option("--threshold", default=0.05, show_default=True, type=float,
              help="Drift threshold for threshold rebalancing.")
@click.option("--cost", default=0.0002, show_default=True, type=float,
              help="Transaction cost per unit of turnover.")
@click.option("--plot/--no-plot", default=False, show_default=True)
@click.option("--output", default=None, help="Output CSV for comparison table.")
def rebalance_cmd(returns, weights, strategy, freq, threshold, cost, plot, output):
    """Analyse rebalancing strategies and their cost/performance tradeoffs."""
    from research.portfolio_lab.rebalancing import RebalancingAnalyzer

    if returns is None:
        click.echo("No --returns provided. Using synthetic data.")
        returns_df = _make_synthetic_returns()
    else:
        returns_df = _load_returns(returns)

    if weights is None:
        n = returns_df.shape[1]
        w_dict = {col: 1.0 / n for col in returns_df.columns}
    else:
        w_dict = _load_weights(weights)

    analyzer = RebalancingAnalyzer(verbose=True)

    if strategy == "all":
        click.echo("\nComparing all rebalancing strategies...")
        df = analyzer.compare_strategies(w_dict, returns_df, transaction_cost=cost)
        click.echo("\nStrategy Comparison:")
        click.echo(df.to_string(index=False))
        if output:
            df.to_csv(output, index=False)
            click.echo(f"\nSaved to {output}")
    elif strategy == "calendar":
        result = analyzer.calendar_rebalance(w_dict, returns_df, freq=freq, transaction_cost=cost)
        _print_rebal_result(result)
    elif strategy == "threshold":
        result = analyzer.threshold_rebalance(w_dict, returns_df, threshold=threshold, transaction_cost=cost)
        _print_rebal_result(result)
    else:
        raise click.BadParameter(f"Unknown strategy: {strategy}")

    if plot and strategy != "all":
        _plot_equity_curve(result)


def _print_rebal_result(result) -> None:
    click.echo(f"\nRebalancing Result — {result.strategy}:")
    click.echo(f"  Total Return:     {result.total_return*100:.2f}%")
    click.echo(f"  Sharpe Ratio:     {result.sharpe_ratio:.3f}")
    click.echo(f"  Max Drawdown:     {result.max_drawdown*100:.2f}%")
    click.echo(f"  N Rebalances:     {result.n_rebalances}")
    click.echo(f"  Total Cost:       {result.total_cost:.4f}")
    click.echo(f"  Tracking Error:   {result.tracking_error*100:.2f}%")
    click.echo(f"  Avg Turnover:     {result.turnover:.4f}")


def _plot_equity_curve(result) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.equity_curve, color="steelblue", linewidth=1.5)
    for rd in result.rebalance_dates:
        ax.axvline(rd, color="orange", alpha=0.3, linewidth=0.8)
    ax.set_title(f"Equity Curve — {result.strategy}")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=150)
    click.echo("Equity curve saved to equity_curve.png")


# ---------------------------------------------------------------------------
# portfolio correlate
# ---------------------------------------------------------------------------


@cli.command("correlate")
@click.option("--returns", default=None, help="Path to returns CSV.")
@click.option("--window", default=60, show_default=True, type=int, help="Rolling correlation window.")
@click.option("--method", default="sample", show_default=True, help="Correlation type: sample | dcc.")
@click.option("--pair", default=None, help="Asset pair for rolling plot (e.g. 'BTC,ETH').")
@click.option("--plot/--no-plot", default=True, show_default=True)
@click.option("--output", default=None, help="Save heatmap to file.")
def correlate_cmd(returns, window, method, pair, plot, output):
    """Compute and visualise portfolio correlations."""
    from research.portfolio_lab.correlation import (
        rolling_correlation_matrix,
        dynamic_conditional_correlation,
        plot_correlation_heatmap,
        plot_rolling_correlation,
        diversification_ratio,
        effective_n_bets,
    )
    from research.portfolio_lab.construction import _estimate_covariance

    if returns is None:
        click.echo("No --returns provided. Using synthetic data.")
        returns_df = _make_synthetic_returns()
    else:
        returns_df = _load_returns(returns)

    if method == "dcc":
        click.echo("Running DCC-GARCH model...")
        dcc = dynamic_conditional_correlation(returns_df)
        last_corr = dcc["corr_matrices"][-1]
        click.echo(f"Final DCC correlation matrix:\n{np.round(last_corr, 3)}")
    else:
        last_corr = _sample_corr(returns_df.values.astype(np.float64))

    cov = _estimate_covariance(returns_df, "ledoit_wolf")
    n = returns_df.shape[1]
    eq_weights = np.ones(n) / n

    dr = diversification_ratio(eq_weights, cov)
    enb = effective_n_bets(eq_weights, cov)
    click.echo(f"\nDiversification Ratio (equal weight): {dr:.3f}")
    click.echo(f"Effective N Bets (equal weight):       {enb}")

    if plot:
        save = output or ("correlation.png" if output else None)
        plot_correlation_heatmap(
            last_corr,
            asset_names=list(returns_df.columns),
            title="Correlation Matrix",
            save_path=save,
        )
        if save:
            click.echo(f"Heatmap saved to {save}")

    if pair is not None:
        parts = [p.strip() for p in pair.split(",")]
        if len(parts) == 2:
            cols = list(returns_df.columns)
            if parts[0] in cols and parts[1] in cols:
                p1_idx = cols.index(parts[0])
                p2_idx = cols.index(parts[1])
                rolling = rolling_correlation_matrix(returns_df, window=window)
                plot_rolling_correlation(
                    rolling, p1_idx, p2_idx,
                    dates=returns_df.index,
                    asset_names=cols,
                    save_path=f"rolling_corr_{parts[0]}_{parts[1]}.png",
                )
                click.echo(f"Rolling correlation saved.")
            else:
                click.echo(f"Could not find assets {parts} in columns {cols}")


def _sample_corr(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    X_c = X - X.mean(axis=0)
    std = X_c.std(axis=0) + 1e-12
    X_norm = X_c / std
    corr = (X_norm.T @ X_norm) / max(1, n - 1)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    cli()
