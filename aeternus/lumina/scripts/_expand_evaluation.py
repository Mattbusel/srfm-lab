"""Expand evaluation.py with comprehensive backtesting and metrics."""

PATH = r"C:\Users\Matthew\srfm-lab\aeternus\lumina\lumina\evaluation.py"

CONTENT = r'''

# =============================================================================
# SECTION: Advanced Portfolio Analytics
# =============================================================================

class AttributionAnalyzer:
    """Brinson-Hood-Beebower (BHB) performance attribution.

    Decomposes portfolio excess return into:
    - Allocation effect: over/underweighting sector vs benchmark
    - Selection effect: stock selection within sectors
    - Interaction effect: joint allocation and selection

    Reference: Brinson & Beebower, "Determinants of Portfolio Performance"
    Financial Analysts Journal 1986.

    Args:
        frequency: Data frequency ('daily', 'weekly', 'monthly')
    """

    def __init__(self, frequency: str = "daily") -> None:
        self.frequency = frequency
        self._annualization = {"daily": 252, "weekly": 52, "monthly": 12}.get(frequency, 252)

    def compute_bhb(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        asset_returns: np.ndarray,
        sector_map: Dict[int, str],
    ) -> Dict[str, float]:
        """Compute BHB attribution decomposition.

        Args:
            portfolio_weights: (N,) portfolio asset weights
            benchmark_weights: (N,) benchmark asset weights
            asset_returns: (N,) realized asset returns this period
            sector_map: Dict mapping asset index to sector name
        Returns:
            Dict with allocation, selection, interaction, total effects
        """
        sectors = list(set(sector_map.values()))
        alloc, select, interact = 0.0, 0.0, 0.0

        for sector in sectors:
            mask = np.array([sector_map.get(i, "") == sector for i in range(len(asset_returns))])
            if mask.sum() == 0:
                continue

            wp = portfolio_weights[mask].sum()  # Portfolio sector weight
            wb = benchmark_weights[mask].sum()   # Benchmark sector weight

            # Sector returns
            rp = (portfolio_weights[mask] * asset_returns[mask]).sum() / (wp + 1e-10)
            rb = (benchmark_weights[mask] * asset_returns[mask]).sum() / (wb + 1e-10)
            rb_total = (benchmark_weights * asset_returns).sum()  # Total benchmark return

            # BHB effects
            alloc += (wp - wb) * (rb - rb_total)
            select += wb * (rp - rb)
            interact += (wp - wb) * (rp - rb)

        total = alloc + select + interact
        return {
            "allocation_effect": alloc,
            "selection_effect": select,
            "interaction_effect": interact,
            "total_active_return": total,
        }

    def rolling_attribution(
        self,
        portfolio_weights: np.ndarray,
        benchmark_weights: np.ndarray,
        asset_returns: np.ndarray,
        sector_map: Dict[int, str],
        window: int = 21,
    ) -> Dict[str, np.ndarray]:
        """Rolling BHB attribution over time.

        Args:
            portfolio_weights: (T, N)
            benchmark_weights: (T, N)
            asset_returns: (T, N)
            sector_map: Dict
            window: Rolling window
        Returns:
            Dict of (T,) attribution time series
        """
        T = len(asset_returns)
        allocs = np.zeros(T)
        selects = np.zeros(T)
        interacts = np.zeros(T)

        for t in range(window, T):
            w_start = max(0, t - window)
            pw_avg = portfolio_weights[w_start:t].mean(axis=0)
            bw_avg = benchmark_weights[w_start:t].mean(axis=0)
            r_avg = asset_returns[w_start:t].mean(axis=0)
            result = self.compute_bhb(pw_avg, bw_avg, r_avg, sector_map)
            allocs[t] = result["allocation_effect"]
            selects[t] = result["selection_effect"]
            interacts[t] = result["interaction_effect"]

        return {
            "allocation": allocs,
            "selection": selects,
            "interaction": interacts,
            "total": allocs + selects + interacts,
        }


class FactorExposureAnalyzer:
    """Analyze factor exposures and alpha decomposition.

    Regresses portfolio returns on systematic risk factors
    to identify alpha (unexplained return) and beta exposures.

    Supported factor models:
    - CAPM: single market factor
    - Fama-French 3-factor
    - Carhart 4-factor (FF3 + momentum)
    - Custom factor models

    Args:
        factors: (T, F) factor return matrix
        factor_names: List of factor names
    """

    def __init__(
        self,
        factors: np.ndarray,
        factor_names: Optional[List[str]] = None,
    ) -> None:
        self.factors = factors
        T, F = factors.shape
        self.factor_names = factor_names or [f"factor_{i}" for i in range(F)]

    def estimate_betas(
        self,
        portfolio_returns: np.ndarray,
    ) -> Dict[str, float]:
        """OLS regression of portfolio on factors.

        Args:
            portfolio_returns: (T,) portfolio return series
        Returns:
            Dict with alpha, betas, r_squared, t_stats
        """
        T = len(portfolio_returns)
        # Add intercept
        X = np.column_stack([np.ones(T), self.factors])  # (T, 1+F)
        y = portfolio_returns

        # OLS: beta = (X'X)^-1 X'y
        try:
            XtX_inv = np.linalg.pinv(X.T @ X)
            beta = XtX_inv @ X.T @ y
        except np.linalg.LinAlgError:
            return {"error": "singular matrix"}

        # Residuals and stats
        y_hat = X @ beta
        resid = y - y_hat
        ss_res = (resid ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # Standard errors
        sigma2 = ss_res / max(1, T - len(beta))
        try:
            se = np.sqrt(np.diag(sigma2 * XtX_inv))
        except Exception:
            se = np.zeros(len(beta))

        t_stats = beta / (se + 1e-10)

        result = {
            "alpha": beta[0],
            "alpha_t_stat": t_stats[0],
            "r_squared": r2,
            "annualized_alpha": beta[0] * 252,
        }
        for i, name in enumerate(self.factor_names):
            result[f"beta_{name}"] = beta[i + 1]
            result[f"t_{name}"] = t_stats[i + 1]

        return result

    def tracking_error(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> Dict[str, float]:
        """Compute tracking error and information ratio.

        Args:
            portfolio_returns: (T,) portfolio returns
            benchmark_returns: (T,) benchmark returns
        Returns:
            Dict with tracking_error, information_ratio, active_return
        """
        active_returns = portfolio_returns - benchmark_returns
        te = active_returns.std() * np.sqrt(252)
        ar = active_returns.mean() * 252
        ir = ar / (te + 1e-10)
        return {
            "tracking_error": te,
            "annualized_active_return": ar,
            "information_ratio": ir,
            "active_return_t_stat": ar / (te / np.sqrt(max(1, len(active_returns))) + 1e-10),
        }


class MarketImpactModel:
    """Market impact model for realistic transaction cost estimation.

    Implements multiple market impact models:
    - Linear: impact proportional to trade size
    - Square root (Almgren): impact ~ sigma * sqrt(ADV * participation_rate)
    - Power law: impact ~ (order_size / ADV)^alpha

    Reference: Almgren et al., "Direct Estimation of Equity Market Impact"
    (2005)

    Args:
        model_type: 'linear', 'sqrt', or 'power'
        eta: Linear impact coefficient
        sigma: Volatility scaling
    """

    def __init__(
        self,
        model_type: str = "sqrt",
        eta: float = 0.1,
        sigma: float = 0.02,
        alpha: float = 0.6,
    ) -> None:
        self.model_type = model_type
        self.eta = eta
        self.sigma = sigma
        self.alpha = alpha

    def estimate_impact(
        self,
        order_size: float,
        adv: float,
        price: float,
        volatility: float,
        side: str = "buy",
    ) -> Dict[str, float]:
        """Estimate market impact for a single trade.

        Args:
            order_size: Trade size in shares
            adv: Average daily volume in shares
            price: Current price per share
            volatility: Daily return volatility
            side: 'buy' or 'sell'
        Returns:
            Dict with impact_bps, total_cost_usd, effective_spread
        """
        participation = order_size / max(1, adv)

        if self.model_type == "linear":
            impact = self.eta * participation
        elif self.model_type == "sqrt":
            # Almgren square root model
            impact = self.sigma * np.sqrt(participation) * np.sign(1 if side == "buy" else -1)
            impact = abs(impact)
        elif self.model_type == "power":
            impact = self.sigma * (participation ** self.alpha)
        else:
            impact = self.eta * participation

        impact_bps = impact * 10000
        total_cost = impact * price * order_size

        return {
            "impact_fraction": impact,
            "impact_bps": impact_bps,
            "total_cost_usd": total_cost,
            "participation_rate": participation,
        }


class RegimeDetectionBacktest:
    """Backtesting framework with regime-conditional analysis.

    Splits backtest periods by detected market regime and reports
    performance metrics separately for each regime. Enables
    understanding of strategy behavior across market conditions.

    Args:
        regime_labels: (T,) integer regime labels
        regime_names: Optional mapping from label to name
    """

    def __init__(
        self,
        regime_labels: np.ndarray,
        regime_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.regime_labels = regime_labels
        unique_regimes = sorted(set(regime_labels))
        if regime_names is None:
            regime_names = {r: f"regime_{r}" for r in unique_regimes}
        self.regime_names = regime_names

    def compute_regime_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute performance metrics within each regime.

        Args:
            returns: (T,) portfolio return series
            benchmark_returns: (T,) optional benchmark returns
        Returns:
            Dict of {regime_name: {metric: value}}
        """
        results = {}
        for regime_id, regime_name in self.regime_names.items():
            mask = self.regime_labels == regime_id
            if mask.sum() < 5:
                continue
            r = returns[mask]
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / (ann_vol + 1e-10)
            max_dd = self._max_drawdown(r)
            calmar = ann_ret / (abs(max_dd) + 1e-10)
            results[regime_name] = {
                "num_days": int(mask.sum()),
                "ann_return": ann_ret,
                "ann_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "calmar_ratio": calmar,
                "hit_rate": float((r > 0).mean()),
                "avg_return": float(r.mean()),
            }
            if benchmark_returns is not None:
                bm = benchmark_returns[mask]
                active = r - bm
                results[regime_name]["active_return"] = active.mean() * 252
                te = active.std() * np.sqrt(252)
                results[regime_name]["tracking_error"] = te
                results[regime_name]["information_ratio"] = results[regime_name]["active_return"] / (te + 1e-10)

        return results

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cum = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cum)
        drawdowns = (cum - rolling_max) / (rolling_max + 1e-10)
        return float(drawdowns.min())


class TransactionCostModel:
    """Realistic transaction cost model for backtesting.

    Components:
    - Bid-ask spread (proportional to volatility and market cap)
    - Commission: fixed per-share or percentage
    - Market impact: function of order size relative to ADV
    - Short borrow rate: for short positions

    Args:
        commission_rate: Fixed commission as fraction of trade value
        spread_model: 'fixed', 'vol_proportional', or 'cap_based'
        fixed_spread_bps: Fixed spread if spread_model='fixed'
        vol_spread_mult: Multiplier for vol-proportional spread
        short_borrow_rate: Annual short borrow rate (e.g., 0.01 = 1%)
    """

    def __init__(
        self,
        commission_rate: float = 0.0001,
        spread_model: str = "fixed",
        fixed_spread_bps: float = 10.0,
        vol_spread_mult: float = 0.5,
        short_borrow_rate: float = 0.02,
    ) -> None:
        self.commission_rate = commission_rate
        self.spread_model = spread_model
        self.fixed_spread_bps = fixed_spread_bps
        self.vol_spread_mult = vol_spread_mult
        self.short_borrow_rate = short_borrow_rate / 252  # Daily

    def estimate_spread(
        self,
        volatility: float,
        market_cap: Optional[float] = None,
    ) -> float:
        """Estimate bid-ask spread in basis points.

        Args:
            volatility: Daily return volatility
            market_cap: Optional market cap for size-based spread
        Returns:
            Estimated spread in bps
        """
        if self.spread_model == "fixed":
            return self.fixed_spread_bps
        elif self.spread_model == "vol_proportional":
            return volatility * self.vol_spread_mult * 10000
        elif self.spread_model == "cap_based" and market_cap is not None:
            # Smaller caps have wider spreads
            if market_cap > 10e9:
                return 3.0
            elif market_cap > 1e9:
                return 8.0
            else:
                return 20.0
        return self.fixed_spread_bps

    def compute_round_trip_cost(
        self,
        trade_value: float,
        volatility: float = 0.02,
        market_cap: Optional[float] = None,
        is_short: bool = False,
        holding_days: int = 1,
    ) -> Dict[str, float]:
        """Compute total round-trip transaction cost.

        Args:
            trade_value: Absolute trade value in dollars
            volatility: Daily return volatility
            market_cap: Optional for spread model
            is_short: Whether this is a short position
            holding_days: Days held (for short borrow cost)
        Returns:
            Dict with component costs and total
        """
        spread_bps = self.estimate_spread(volatility, market_cap)
        spread_cost = spread_bps / 10000 * trade_value * 2  # 2-way
        commission = self.commission_rate * trade_value * 2
        borrow_cost = 0.0
        if is_short:
            borrow_cost = self.short_borrow_rate * holding_days * trade_value
        total = spread_cost + commission + borrow_cost
        return {
            "spread_cost": spread_cost,
            "commission": commission,
            "borrow_cost": borrow_cost,
            "total": total,
            "total_bps": total / (trade_value + 1e-10) * 10000,
        }


class BootstrapMetricCalculator:
    """Bootstrap resampling for robust statistical inference on metrics.

    Computes bootstrap confidence intervals for performance metrics
    (Sharpe, Sortino, IC, etc.) to assess statistical significance.

    Args:
        num_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (e.g., 0.95)
        random_seed: Reproducibility seed
        block_size: Block size for block bootstrap (preserves autocorrelation)
    """

    def __init__(
        self,
        num_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
        block_size: int = 10,
    ) -> None:
        self.num_bootstrap = num_bootstrap
        self.confidence_level = confidence_level
        self.block_size = block_size
        np.random.seed(random_seed)

    def _block_bootstrap(self, data: np.ndarray) -> np.ndarray:
        """Generate block bootstrap sample."""
        T = len(data)
        num_blocks = T // self.block_size + 1
        block_starts = np.random.randint(0, max(1, T - self.block_size), size=num_blocks)
        blocks = [data[s:s + self.block_size] for s in block_starts]
        sample = np.concatenate(blocks)[:T]
        return sample

    def bootstrap_metric(
        self,
        returns: np.ndarray,
        metric_fn: Callable,
        use_block: bool = True,
    ) -> Dict[str, float]:
        """Bootstrap a scalar metric.

        Args:
            returns: (T,) return series
            metric_fn: Function that takes (T,) returns and returns scalar
            use_block: Use block bootstrap (True) or iid (False)
        Returns:
            Dict with mean, std, ci_lower, ci_upper, p_value_positive
        """
        bootstrap_values = []
        for _ in range(self.num_bootstrap):
            if use_block:
                sample = self._block_bootstrap(returns)
            else:
                sample = returns[np.random.randint(0, len(returns), len(returns))]
            try:
                val = metric_fn(sample)
                bootstrap_values.append(val)
            except Exception:
                continue

        if not bootstrap_values:
            return {"error": "metric computation failed"}

        bv = np.array(bootstrap_values)
        alpha = (1 - self.confidence_level) / 2
        ci_lower = np.quantile(bv, alpha)
        ci_upper = np.quantile(bv, 1 - alpha)
        p_positive = float((bv > 0).mean())

        return {
            "mean": float(bv.mean()),
            "std": float(bv.std()),
            f"ci_{int(self.confidence_level*100)}_lower": ci_lower,
            f"ci_{int(self.confidence_level*100)}_upper": ci_upper,
            "p_value_positive": p_positive,
            "original_value": float(metric_fn(returns)),
        }

    def bootstrap_sharpe(self, returns: np.ndarray) -> Dict[str, float]:
        """Bootstrap Sharpe ratio."""
        def sharpe_fn(r):
            return r.mean() * 252 / (r.std() * np.sqrt(252) + 1e-10)
        return self.bootstrap_metric(returns, sharpe_fn)

    def bootstrap_ic(
        self,
        predictions: np.ndarray,
        realized: np.ndarray,
    ) -> Dict[str, float]:
        """Bootstrap Information Coefficient (rank correlation)."""
        from scipy import stats

        def ic_fn_inner(idx):
            p, r = predictions[idx], realized[idx]
            return stats.spearmanr(p, r).correlation

        def ic_bootstrap(dummy_r):
            # Hack: resample by index
            return ic_fn_inner(slice(None))  # Just return the full IC

        # Manual bootstrap
        T = len(predictions)
        bootstrap_values = []
        for _ in range(self.num_bootstrap):
            if self.block_size > 1:
                num_blocks = T // self.block_size + 1
                starts = np.random.randint(0, max(1, T - self.block_size), size=num_blocks)
                idx = np.concatenate([np.arange(s, min(s + self.block_size, T)) for s in starts])[:T]
            else:
                idx = np.random.randint(0, T, T)
            try:
                from scipy import stats as scipy_stats
                ic = scipy_stats.spearmanr(predictions[idx], realized[idx]).correlation
                bootstrap_values.append(float(ic))
            except Exception:
                pass

        if not bootstrap_values:
            return {}
        bv = np.array(bootstrap_values)
        alpha = (1 - self.confidence_level) / 2
        return {
            "ic_mean": float(bv.mean()),
            "ic_std": float(bv.std()),
            "ic_ci_lower": float(np.quantile(bv, alpha)),
            "ic_ci_upper": float(np.quantile(bv, 1 - alpha)),
            "ic_t_stat": float(bv.mean() / (bv.std() / np.sqrt(len(bv)) + 1e-10)),
        }


class RollingSharpeAnalysis:
    """Rolling Sharpe ratio and drawdown analysis utilities.

    Provides time series of risk-adjusted performance metrics
    to identify periods of strategy degradation or improvement.

    Args:
        window: Rolling window in trading days
        annualization: Trading days per year for annualization
    """

    def __init__(self, window: int = 63, annualization: int = 252) -> None:
        self.window = window
        self.annualization = annualization

    def rolling_sharpe(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling Sharpe ratio."""
        T = len(returns)
        sharpes = np.full(T, np.nan)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            ann_ret = r.mean() * self.annualization
            ann_vol = r.std() * np.sqrt(self.annualization)
            sharpes[t] = ann_ret / (ann_vol + 1e-10)
        return sharpes

    def rolling_sortino(self, returns: np.ndarray, mar: float = 0.0) -> np.ndarray:
        """Compute rolling Sortino ratio."""
        T = len(returns)
        sortinos = np.full(T, np.nan)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            ann_ret = r.mean() * self.annualization - mar
            downside = r[r < mar / self.annualization] - mar / self.annualization
            dd_std = np.sqrt((downside ** 2).mean()) * np.sqrt(self.annualization)
            sortinos[t] = ann_ret / (dd_std + 1e-10)
        return sortinos

    def rolling_max_drawdown(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling maximum drawdown."""
        T = len(returns)
        mdd = np.full(T, np.nan)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            cum = (1 + r).cumprod()
            roll_max = np.maximum.accumulate(cum)
            dd = (cum - roll_max) / (roll_max + 1e-10)
            mdd[t] = dd.min()
        return mdd

    def rolling_calmar(self, returns: np.ndarray) -> np.ndarray:
        """Compute rolling Calmar ratio."""
        T = len(returns)
        calmar = np.full(T, np.nan)
        sharpes = self.rolling_sharpe(returns)
        mdd = self.rolling_max_drawdown(returns)
        for t in range(self.window, T):
            r = returns[t - self.window:t]
            ann_ret = r.mean() * self.annualization
            dd = abs(mdd[t])
            calmar[t] = ann_ret / (dd + 1e-10)
        return calmar

    def regime_performance_summary(
        self,
        returns: np.ndarray,
        regime_labels: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Summary statistics per regime."""
        regimes = {}
        for r_id in sorted(set(regime_labels)):
            mask = regime_labels == r_id
            r = returns[mask]
            if len(r) < 2:
                continue
            regimes[f"regime_{r_id}"] = {
                "n": int(len(r)),
                "mean": float(r.mean() * self.annualization),
                "vol": float(r.std() * np.sqrt(self.annualization)),
                "sharpe": float(r.mean() * self.annualization / (r.std() * np.sqrt(self.annualization) + 1e-10)),
                "hit_rate": float((r > 0).mean()),
                "skew": float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-10)),
                "kurt": float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-10) - 3),
            }
        return regimes


_NEW_EVALUATION_EXPORTS = [
    "AttributionAnalyzer", "FactorExposureAnalyzer", "MarketImpactModel",
    "RegimeDetectionBacktest", "TransactionCostModel",
    "BootstrapMetricCalculator", "RollingSharpeAnalysis",
]
'''

with open(PATH, "a", encoding="utf-8") as f:
    f.write(CONTENT)

import subprocess
r = subprocess.run(["wc", "-l", PATH], capture_output=True, text=True, shell=True)
print(r.stdout.strip())
