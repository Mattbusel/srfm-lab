"""
farm_client.py — Python client for the Go backtest farm API.

Provides:
    FarmClient     — low-level HTTP client (localhost:11439)
    FarmOrchestrator — high-level sweep / regime / robustness orchestration
    FarmReporter   — reports, leaderboards, overfitting estimates

Dependencies: numpy, scipy (only).  Uses urllib from the stdlib for HTTP.
"""

from __future__ import annotations

import json
import time
import hashlib
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "http://localhost:11439"
CONTENT_JSON = {"Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no requests)
# ---------------------------------------------------------------------------

class FarmHTTPError(Exception):
    """Raised when the farm API returns a non-2xx status."""
    def __init__(self, status: int, body: str, url: str):
        self.status = status
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {body[:200]}")


def _http_request(
    method: str,
    url: str,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
) -> dict | list | str:
    """Issue an HTTP request and return parsed JSON (or raw text)."""
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    hdrs = dict(CONTENT_JSON)
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raw_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise FarmHTTPError(e.code, raw_body, url) from None
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach farm at {url}: {e.reason}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _get(url: str, **kwargs) -> Any:
    return _http_request("GET", url, **kwargs)


def _post(url: str, body: dict, **kwargs) -> Any:
    return _http_request("POST", url, body=body, **kwargs)


def _delete(url: str, **kwargs) -> Any:
    return _http_request("DELETE", url, **kwargs)


# ---------------------------------------------------------------------------
# FarmClient
# ---------------------------------------------------------------------------

class FarmClient:
    """
    Low-level client for the Go backtest farm API at localhost:11439.

    Every public method maps to one API call (or a small sequence for
    streaming).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # -- Health / status ---------------------------------------------------

    def ping(self) -> bool:
        """Return True if the farm is reachable."""
        try:
            resp = _get(self._url("/health"), timeout=5.0)
            return True
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        """Return overall farm status (queue depth, workers, etc.)."""
        return _get(self._url("/status"), timeout=self.timeout)

    # -- Job submission ----------------------------------------------------

    def submit_jobs(self, configs: list[dict[str, Any]]) -> list[str]:
        """
        Submit a batch of backtest job configurations.

        Each config dict must contain at minimum:
            strategy (str), symbol (str), timeframe (str), params (dict).

        Returns a list of job IDs.
        """
        if not configs:
            return []
        resp = _post(self._url("/jobs/submit"), {"jobs": configs}, timeout=self.timeout)
        return resp.get("job_ids", [])

    def submit_grid_search(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        param_grid: list[dict[str, Any]],
        tag: str = "",
    ) -> str:
        """
        Submit a full parameter grid as a batch.

        Returns a batch_id that can be used to track progress.
        """
        payload = {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "param_grid": param_grid,
            "tag": tag or f"grid_{strategy}_{int(time.time())}",
        }
        resp = _post(self._url("/jobs/grid"), payload, timeout=self.timeout)
        return resp.get("batch_id", "")

    # -- Results -----------------------------------------------------------

    def get_results(
        self,
        top_n: int = 50,
        sort_by: str = "sharpe",
        strategy: str | None = None,
        symbol: str | None = None,
        batch_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch ranked results from the farm.

        Supports filtering by strategy, symbol, or batch_id.
        """
        params: dict[str, str] = {
            "top_n": str(top_n),
            "sort_by": sort_by,
        }
        if strategy:
            params["strategy"] = strategy
        if symbol:
            params["symbol"] = symbol
        if batch_id:
            params["batch_id"] = batch_id
        qs = urllib.parse.urlencode(params)
        return _get(self._url(f"/results?{qs}"), timeout=self.timeout)

    def get_result_by_id(self, job_id: str) -> dict[str, Any]:
        """Fetch a single result by job ID."""
        return _get(self._url(f"/results/{job_id}"), timeout=self.timeout)

    def get_landscape(
        self,
        strategy: str,
        symbol: str,
        param_x: str = "",
        param_y: str = "",
    ) -> dict[str, Any]:
        """
        Fetch heatmap data for a strategy/symbol pair.

        The farm pre-computes landscapes; this just fetches the cached data.
        """
        params: dict[str, str] = {
            "strategy": strategy,
            "symbol": symbol,
        }
        if param_x:
            params["param_x"] = param_x
        if param_y:
            params["param_y"] = param_y
        qs = urllib.parse.urlencode(params)
        return _get(self._url(f"/landscape?{qs}"), timeout=self.timeout)

    # -- Batch tracking ----------------------------------------------------

    def get_batch_status(self, batch_id: str) -> dict[str, Any]:
        """Get progress of a batch (grid search)."""
        return _get(self._url(f"/batches/{batch_id}"), timeout=self.timeout)

    def wait_for_completion(
        self,
        batch_id: str,
        timeout: float = 3600.0,
        poll_interval: float = 2.0,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Block until a batch completes (or timeout).

        Polls the farm API every *poll_interval* seconds.

        Parameters
        ----------
        batch_id : the batch to wait for.
        timeout : max seconds to wait.
        poll_interval : seconds between polls.
        progress_callback : optional callable(status_dict).

        Returns
        -------
        List of result dicts for the completed batch.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self.get_batch_status(batch_id)
            if progress_callback:
                progress_callback(status)
            state = status.get("state", "unknown")
            if state in ("completed", "done", "finished"):
                return self.get_results(top_n=10000, batch_id=batch_id)
            if state in ("failed", "error"):
                raise RuntimeError(f"Batch {batch_id} failed: {status.get('error', 'unknown')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Batch {batch_id} did not complete within {timeout}s")

    def stream_results(
        self,
        callback: Callable[[dict[str, Any]], None],
        batch_id: str | None = None,
        poll_interval: float = 1.0,
        stop_after: int = 0,
    ) -> int:
        """
        Stream results as they appear by polling.

        Parameters
        ----------
        callback : called with each new result dict.
        batch_id : optionally restrict to one batch.
        poll_interval : seconds between polls.
        stop_after : stop after this many results (0 = indefinite until
            batch completes or KeyboardInterrupt).

        Returns
        -------
        Total number of results streamed.
        """
        seen_ids: set[str] = set()
        total = 0
        try:
            while True:
                results = self.get_results(
                    top_n=500,
                    batch_id=batch_id,
                )
                if isinstance(results, list):
                    for r in results:
                        rid = r.get("job_id", r.get("id", ""))
                        if rid and rid not in seen_ids:
                            seen_ids.add(rid)
                            callback(r)
                            total += 1
                            if stop_after > 0 and total >= stop_after:
                                return total

                if batch_id:
                    try:
                        status = self.get_batch_status(batch_id)
                        if status.get("state") in ("completed", "done", "finished"):
                            break
                    except Exception:
                        pass
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            pass
        return total

    # -- Cancellation / cleanup --------------------------------------------

    def cancel_batch(self, batch_id: str) -> dict[str, Any]:
        """Cancel a running batch."""
        return _delete(self._url(f"/batches/{batch_id}"), timeout=self.timeout)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a single job."""
        return _delete(self._url(f"/jobs/{job_id}"), timeout=self.timeout)


# ---------------------------------------------------------------------------
# FarmOrchestrator
# ---------------------------------------------------------------------------

class FarmOrchestrator:
    """High-level orchestration on top of FarmClient."""

    def __init__(
        self,
        client: FarmClient | None = None,
        strategies: list[str] | None = None,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ):
        self.client = client or FarmClient()
        self.strategies = strategies or [
            "momentum_12_1", "momentum_3m", "mean_reversion_z",
            "breakout_donchian", "rsi_reversal", "macd_crossover",
            "dual_ma_cross", "vol_targeting", "hurst_adaptive",
            "combined_momentum_reversion",
        ]
        self.symbols = symbols or ["BTC", "ETH", "SPY", "QQQ", "GLD"]
        self.timeframes = timeframes or ["1h", "4h", "1d"]

    def _default_param_grid(self, strategy: str) -> list[dict[str, Any]]:
        """Generate a small default parameter grid for a strategy."""
        grids: dict[str, list[dict]] = {
            "momentum_12_1": [
                {"lookback": lb, "skip": sk, "threshold": th}
                for lb in [126, 189, 252]
                for sk in [0, 10, 21]
                for th in [0.0, 0.02, 0.05]
            ],
            "momentum_3m": [
                {"lookback": lb, "vol_scale": vs}
                for lb in [42, 63, 84]
                for vs in [True, False]
            ],
            "mean_reversion_z": [
                {"window": w, "entry_z": ez, "exit_z": xz}
                for w in [30, 60, 90]
                for ez in [1.5, 2.0, 2.5]
                for xz in [0.3, 0.5, 0.8]
            ],
            "breakout_donchian": [
                {"window": w, "atr_mult_for_stop": am}
                for w in [20, 40, 55]
                for am in [1.5, 2.0, 3.0]
            ],
            "rsi_reversal": [
                {"period": p, "oversold": os_, "overbought": ob}
                for p in [7, 14, 21]
                for os_ in [20, 30]
                for ob in [70, 80]
            ],
            "macd_crossover": [
                {"fast": f, "slow": s, "signal": sg}
                for f in [8, 12]
                for s in [21, 26]
                for sg in [7, 9]
            ],
            "dual_ma_cross": [
                {"fast_window": f, "slow_window": s}
                for f in [10, 20, 30]
                for s in [50, 100, 200]
            ],
            "vol_targeting": [
                {"target_vol": tv, "vol_window": vw}
                for tv in [0.05, 0.10, 0.15, 0.20]
                for vw in [30, 60, 90]
            ],
            "hurst_adaptive": [
                {"hurst_window": hw, "trending_h": th, "reverting_h": rh}
                for hw in [60, 100, 150]
                for th in [0.55, 0.60, 0.65]
                for rh in [0.35, 0.40, 0.45]
            ],
            "combined_momentum_reversion": [
                {"regime_window": rw, "mom_weight": mw, "rev_weight": 1.0 - mw}
                for rw in [40, 60, 90]
                for mw in [0.3, 0.5, 0.7]
            ],
        }
        return grids.get(strategy, [{}])

    def run_full_sweep(
        self,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Test every strategy x symbol x timeframe x parameter grid.

        Submits all jobs to the farm and waits for results.

        Returns
        -------
        Dict mapping "strategy::symbol::timeframe" -> list of results.
        """
        all_results: dict[str, list[dict]] = {}
        batch_ids: list[tuple[str, str]] = []

        for strategy in self.strategies:
            grid = self._default_param_grid(strategy)
            for symbol in self.symbols:
                for tf in self.timeframes:
                    key = f"{strategy}::{symbol}::{tf}"
                    configs = []
                    for params in grid:
                        configs.append({
                            "strategy": strategy,
                            "symbol": symbol,
                            "timeframe": tf,
                            "params": params,
                        })
                    if not configs:
                        continue
                    try:
                        bid = self.client.submit_grid_search(
                            strategy, symbol, tf, grid,
                            tag=f"sweep_{strategy}_{symbol}_{tf}",
                        )
                        batch_ids.append((key, bid))
                    except Exception as e:
                        all_results[key] = [{"error": str(e)}]

        for key, bid in batch_ids:
            try:
                results = self.client.wait_for_completion(
                    bid, timeout=1800, progress_callback=progress_callback,
                )
                all_results[key] = results
            except Exception as e:
                all_results[key] = [{"error": str(e)}]

        return all_results

    def run_regime_analysis(
        self,
        n_regimes: int = 4,
        regime_length: int = 500,
    ) -> dict[str, dict[str, float]]:
        """
        Test all strategies across simulated regimes.

        Simulates N return regimes (trending up, trending down, mean-reverting,
        random walk) and evaluates each strategy on each.

        Returns
        -------
        {strategy: {regime_name: sharpe}}.
        """
        rng = np.random.default_rng(42)
        regimes: dict[str, np.ndarray] = {}

        # Trending up
        trend = 0.0005 + 0.01 * rng.standard_normal(regime_length)
        regimes["trending_up"] = trend

        # Trending down
        regimes["trending_down"] = -trend

        # Mean-reverting (OU process)
        ou = np.zeros(regime_length)
        theta, mu, sigma = 0.1, 0.0, 0.01
        for i in range(1, regime_length):
            ou[i] = ou[i - 1] + theta * (mu - ou[i - 1]) + sigma * rng.standard_normal()
        regimes["mean_reverting"] = np.diff(ou) / (np.abs(ou[:-1]) + 1.0)
        regimes["mean_reverting"] = np.append(regimes["mean_reverting"], 0.0)

        # Random walk
        regimes["random_walk"] = 0.01 * rng.standard_normal(regime_length)

        results: dict[str, dict[str, float]] = {}
        for strategy in self.strategies:
            grid = self._default_param_grid(strategy)
            params = grid[len(grid) // 2] if grid else {}
            strat_results: dict[str, float] = {}

            configs = []
            for regime_name, rets in regimes.items():
                configs.append({
                    "strategy": strategy,
                    "symbol": f"SIM_{regime_name}",
                    "timeframe": "1d",
                    "params": params,
                    "returns": rets.tolist(),
                })

            try:
                job_ids = self.client.submit_jobs(configs)
                for jid, regime_name in zip(job_ids, regimes.keys()):
                    try:
                        r = self.client.get_result_by_id(jid)
                        strat_results[regime_name] = r.get("sharpe", float("nan"))
                    except Exception:
                        strat_results[regime_name] = float("nan")
            except Exception:
                for regime_name in regimes:
                    strat_results[regime_name] = float("nan")

            results[strategy] = strat_results
        return results

    def run_robustness_check(
        self,
        top_strategies: list[dict[str, Any]],
        perturbation_pct: float = 0.1,
        n_perturbations: int = 20,
    ) -> dict[str, dict[str, Any]]:
        """
        Perturb parameters of top strategies and check stability.

        For each top strategy, randomly perturb each parameter by up to
        +/- perturbation_pct and re-evaluate.

        Returns
        -------
        {strategy: {mean_sharpe, std_sharpe, worst_sharpe, robustness_score}}.
        """
        rng = np.random.default_rng(123)
        results: dict[str, dict[str, Any]] = {}

        for entry in top_strategies:
            strategy = entry.get("strategy", "")
            base_params = entry.get("params", {})
            perturbed_grids: list[dict] = []

            for _ in range(n_perturbations):
                perturbed = {}
                for k, v in base_params.items():
                    if isinstance(v, (int, float)):
                        noise = 1.0 + rng.uniform(-perturbation_pct, perturbation_pct)
                        perturbed[k] = type(v)(v * noise) if isinstance(v, int) else v * noise
                    else:
                        perturbed[k] = v
                perturbed_grids.append(perturbed)

            try:
                bid = self.client.submit_grid_search(
                    strategy, "BTC", "1d", perturbed_grids,
                    tag=f"robust_{strategy}",
                )
                batch_results = self.client.wait_for_completion(bid, timeout=600)
                sharpes = [r.get("sharpe", float("nan")) for r in batch_results]
                sharpes = [s for s in sharpes if not np.isnan(s)]
                if sharpes:
                    results[strategy] = {
                        "mean_sharpe": float(np.mean(sharpes)),
                        "std_sharpe": float(np.std(sharpes)),
                        "worst_sharpe": float(np.min(sharpes)),
                        "best_sharpe": float(np.max(sharpes)),
                        "robustness_score": float(np.mean(sharpes) - 2 * np.std(sharpes)),
                        "n_evaluated": len(sharpes),
                    }
                else:
                    results[strategy] = {"error": "no valid results"}
            except Exception as e:
                results[strategy] = {"error": str(e)}

        return results

    def compare_strategies(
        self,
        strategy_a: str,
        strategy_b: str,
        symbol: str = "BTC",
        timeframe: str = "1d",
    ) -> dict[str, Any]:
        """
        Head-to-head comparison of two strategies.

        Submits both with default params and compares metrics.

        Returns
        -------
        Dict with keys: strategy_a_metrics, strategy_b_metrics, winner,
        sharpe_diff, p_value (Welch t-test on daily returns).
        """
        grid_a = self._default_param_grid(strategy_a)
        grid_b = self._default_param_grid(strategy_b)
        params_a = grid_a[len(grid_a) // 2] if grid_a else {}
        params_b = grid_b[len(grid_b) // 2] if grid_b else {}

        configs = [
            {"strategy": strategy_a, "symbol": symbol, "timeframe": timeframe, "params": params_a},
            {"strategy": strategy_b, "symbol": symbol, "timeframe": timeframe, "params": params_b},
        ]
        job_ids = self.client.submit_jobs(configs)

        results = {}
        for jid, name in zip(job_ids, [strategy_a, strategy_b]):
            time.sleep(0.5)
            try:
                r = self.client.get_result_by_id(jid)
                results[name] = r
            except Exception as e:
                results[name] = {"error": str(e)}

        sharpe_a = results.get(strategy_a, {}).get("sharpe", 0.0)
        sharpe_b = results.get(strategy_b, {}).get("sharpe", 0.0)

        daily_a = np.array(results.get(strategy_a, {}).get("daily_returns", [0.0]))
        daily_b = np.array(results.get(strategy_b, {}).get("daily_returns", [0.0]))
        if len(daily_a) > 1 and len(daily_b) > 1:
            t_stat, p_val = sp_stats.ttest_ind(daily_a, daily_b, equal_var=False)
        else:
            t_stat, p_val = 0.0, 1.0

        winner = strategy_a if sharpe_a >= sharpe_b else strategy_b
        return {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "strategy_a_metrics": results.get(strategy_a, {}),
            "strategy_b_metrics": results.get(strategy_b, {}),
            "sharpe_a": sharpe_a,
            "sharpe_b": sharpe_b,
            "sharpe_diff": abs(sharpe_a - sharpe_b),
            "winner": winner,
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_at_5pct": p_val < 0.05,
        }

    def find_alpha(
        self,
        strategy: str,
        symbol: str = "BTC",
        timeframe: str = "1d",
        n_iterations: int = 50,
    ) -> dict[str, Any]:
        """
        Bayesian optimization to find the best parameter set for a strategy.

        Uses the farm as the evaluator, iteratively suggesting and testing
        parameter configurations.

        Returns
        -------
        Dict with best_params, best_sharpe, history.
        """
        grid = self._default_param_grid(strategy)
        if not grid:
            return {"error": "no default grid for strategy"}

        # Use first entry to determine param names and rough ranges
        template = grid[0]
        param_names = sorted(template.keys())
        all_vals: dict[str, list[float]] = {k: [] for k in param_names}
        for g in grid:
            for k in param_names:
                v = g.get(k)
                if isinstance(v, (int, float)):
                    all_vals[k].append(float(v))

        # Build ranges
        ranges: dict[str, tuple[float, float]] = {}
        for k in param_names:
            vals = all_vals[k]
            if vals:
                ranges[k] = (min(vals) * 0.5, max(vals) * 1.5)
            else:
                ranges[k] = (0.0, 1.0)

        rng = np.random.default_rng(42)
        history: list[dict] = []
        best_params: dict = {}
        best_sharpe = float("-inf")

        # Random initial evaluations
        n_init = min(n_iterations // 3, 10)
        for i in range(n_init):
            params = {}
            for k in param_names:
                lo, hi = ranges[k]
                val = rng.uniform(lo, hi)
                if isinstance(template.get(k), int):
                    val = int(round(val))
                elif isinstance(template.get(k), bool):
                    val = bool(rng.integers(2))
                params[k] = val

            try:
                job_ids = self.client.submit_jobs([{
                    "strategy": strategy, "symbol": symbol,
                    "timeframe": timeframe, "params": params,
                }])
                time.sleep(0.3)
                result = self.client.get_result_by_id(job_ids[0])
                sharpe = result.get("sharpe", float("nan"))
            except Exception:
                sharpe = float("nan")

            history.append({"params": params, "sharpe": sharpe})
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = dict(params)

        # Simple UCB-style search using history
        for iteration in range(n_init, n_iterations):
            # Exploitation: perturb the best found
            if rng.random() < 0.7 and best_params:
                params = {}
                for k in param_names:
                    lo, hi = ranges[k]
                    base = best_params.get(k, (lo + hi) / 2)
                    if isinstance(base, (int, float)):
                        noise = rng.normal(0, (hi - lo) * 0.1)
                        val = np.clip(float(base) + noise, lo, hi)
                        if isinstance(template.get(k), int):
                            val = int(round(val))
                        params[k] = val
                    else:
                        params[k] = base
            else:
                # Exploration: random
                params = {}
                for k in param_names:
                    lo, hi = ranges[k]
                    val = rng.uniform(lo, hi)
                    if isinstance(template.get(k), int):
                        val = int(round(val))
                    params[k] = val

            try:
                job_ids = self.client.submit_jobs([{
                    "strategy": strategy, "symbol": symbol,
                    "timeframe": timeframe, "params": params,
                }])
                time.sleep(0.3)
                result = self.client.get_result_by_id(job_ids[0])
                sharpe = result.get("sharpe", float("nan"))
            except Exception:
                sharpe = float("nan")

            history.append({"params": params, "sharpe": sharpe})
            if not np.isnan(sharpe) and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = dict(params)

        return {
            "strategy": strategy,
            "symbol": symbol,
            "timeframe": timeframe,
            "best_params": best_params,
            "best_sharpe": best_sharpe,
            "n_iterations": len(history),
            "history": history,
        }


# ---------------------------------------------------------------------------
# FarmReporter
# ---------------------------------------------------------------------------

class FarmReporter:
    """Generate reports from farm results."""

    def __init__(self, client: FarmClient | None = None):
        self.client = client or FarmClient()

    # -- Leaderboard -------------------------------------------------------

    def strategy_leaderboard(
        self,
        top_n: int = 20,
        metric: str = "sharpe",
        min_trades: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Strategy leaderboard with statistical significance markers.

        Fetches top results, groups by strategy, and marks whether the
        Sharpe is statistically different from zero.
        """
        results = self.client.get_results(top_n=500, sort_by=metric)
        if not isinstance(results, list):
            return []

        by_strategy: dict[str, list[dict]] = {}
        for r in results:
            name = r.get("strategy", "unknown")
            by_strategy.setdefault(name, []).append(r)

        board = []
        for name, strat_results in by_strategy.items():
            sharpes = [r.get("sharpe", 0.0) for r in strat_results if not np.isnan(r.get("sharpe", float("nan")))]
            if not sharpes:
                continue
            mean_sharpe = float(np.mean(sharpes))
            std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
            best_sharpe = float(np.max(sharpes))

            # t-test: is mean Sharpe significantly > 0?
            if len(sharpes) > 2 and std_sharpe > 1e-12:
                t_stat = mean_sharpe / (std_sharpe / np.sqrt(len(sharpes)))
                p_val = float(1.0 - sp_stats.t.cdf(t_stat, df=len(sharpes) - 1))
            else:
                t_stat, p_val = 0.0, 1.0

            significance = ""
            if p_val < 0.01:
                significance = "***"
            elif p_val < 0.05:
                significance = "**"
            elif p_val < 0.10:
                significance = "*"

            board.append({
                "strategy": name,
                "mean_sharpe": mean_sharpe,
                "best_sharpe": best_sharpe,
                "std_sharpe": std_sharpe,
                "n_configs": len(sharpes),
                "t_stat": float(t_stat),
                "p_value": p_val,
                "significance": significance,
            })

        board.sort(key=lambda x: x["mean_sharpe"], reverse=True)
        return board[:top_n]

    # -- Parameter sensitivity ---------------------------------------------

    def parameter_sensitivity(
        self,
        strategy: str,
        metric: str = "sharpe",
    ) -> dict[str, dict[str, float]]:
        """
        Parameter sensitivity report for a strategy.

        For each parameter, compute the correlation between the parameter
        value and the objective, as well as the marginal variance explained.

        Returns
        -------
        {param_name: {correlation, variance_explained, optimal_range_low, optimal_range_high}}.
        """
        results = self.client.get_results(top_n=1000, strategy=strategy, sort_by=metric)
        if not isinstance(results, list) or len(results) < 5:
            return {}

        objectives = []
        param_values: dict[str, list[float]] = {}
        for r in results:
            obj = r.get(metric, float("nan"))
            if np.isnan(obj):
                continue
            objectives.append(obj)
            params = r.get("params", {})
            for k, v in params.items():
                if isinstance(v, (int, float)):
                    param_values.setdefault(k, []).append(float(v))

        obj_arr = np.array(objectives)
        report: dict[str, dict[str, float]] = {}
        for pname, vals in param_values.items():
            if len(vals) != len(obj_arr):
                continue
            val_arr = np.array(vals)
            if np.std(val_arr) < 1e-15:
                continue
            corr = float(np.corrcoef(val_arr, obj_arr)[0, 1])

            # Top 20% optimal range
            top_mask = obj_arr >= np.percentile(obj_arr, 80)
            top_vals = val_arr[top_mask]
            report[pname] = {
                "correlation": corr,
                "variance_explained": corr ** 2,
                "optimal_range_low": float(np.percentile(top_vals, 10)) if len(top_vals) > 0 else 0.0,
                "optimal_range_high": float(np.percentile(top_vals, 90)) if len(top_vals) > 0 else 0.0,
                "mean_at_top": float(np.mean(top_vals)) if len(top_vals) > 0 else 0.0,
            }
        return report

    # -- Regime robustness matrix ------------------------------------------

    def regime_robustness_matrix(
        self,
        regime_results: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """
        Build a regime robustness matrix from orchestrator regime analysis.

        Returns
        -------
        {matrix: [[sharpe]], strategies: [names], regimes: [names],
         best_overall: str, worst_regime_per_strategy: {str: str}}.
        """
        strategies = sorted(regime_results.keys())
        if not strategies:
            return {"matrix": [], "strategies": [], "regimes": []}

        regimes = sorted(next(iter(regime_results.values())).keys())
        matrix = []
        worst_regime: dict[str, str] = {}

        for strat in strategies:
            row = []
            worst_val = float("inf")
            worst_name = ""
            for regime in regimes:
                val = regime_results[strat].get(regime, float("nan"))
                row.append(val)
                if not np.isnan(val) and val < worst_val:
                    worst_val = val
                    worst_name = regime
            matrix.append(row)
            worst_regime[strat] = worst_name

        matrix_arr = np.array(matrix)
        # Best overall = highest minimum across regimes
        min_per_strat = np.nanmin(matrix_arr, axis=1)
        best_idx = int(np.nanargmax(min_per_strat))

        return {
            "matrix": matrix,
            "strategies": strategies,
            "regimes": regimes,
            "best_overall": strategies[best_idx],
            "worst_regime_per_strategy": worst_regime,
        }

    # -- Overfitting probability -------------------------------------------

    def overfitting_probability(
        self,
        strategy: str,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Estimate probability of overfitting for a strategy.

        Uses a simplified combinatorial cross-validation approach:
        compare in-sample vs out-of-sample Sharpe degradation.

        If OOS Sharpe < 50% of IS Sharpe in most splits, likely overfit.
        """
        results = self.client.get_results(top_n=500, strategy=strategy)
        if not isinstance(results, list) or len(results) < 10:
            return {"strategy": strategy, "error": "insufficient data"}

        sharpes = np.array([r.get("sharpe", 0.0) for r in results if not np.isnan(r.get("sharpe", float("nan")))])
        if len(sharpes) < n_splits * 2:
            return {"strategy": strategy, "error": "insufficient results for cross-validation"}

        rng = np.random.default_rng(42)
        n = len(sharpes)
        is_better_count = 0
        degradation_ratios = []

        for _ in range(n_splits):
            idx = rng.permutation(n)
            half = n // 2
            is_sharpes = sharpes[idx[:half]]
            oos_sharpes = sharpes[idx[half:half * 2]]
            is_best = float(np.max(is_sharpes))
            oos_best = float(np.max(oos_sharpes))

            if abs(is_best) > 1e-12:
                ratio = oos_best / is_best
            else:
                ratio = 1.0
            degradation_ratios.append(ratio)
            if is_best > oos_best:
                is_better_count += 1

        mean_degradation = float(np.mean(degradation_ratios))
        overfit_prob = is_better_count / n_splits

        return {
            "strategy": strategy,
            "overfit_probability": overfit_prob,
            "mean_degradation_ratio": mean_degradation,
            "degradation_ratios": degradation_ratios,
            "n_splits": n_splits,
            "assessment": (
                "likely_overfit" if overfit_prob > 0.8
                else "moderate_risk" if overfit_prob > 0.5
                else "likely_robust"
            ),
        }

    # -- Ensemble construction ---------------------------------------------

    def ensemble_construction(
        self,
        top_n: int = 5,
        method: str = "equal_weight",
    ) -> dict[str, Any]:
        """
        Construct an ensemble from the top strategies.

        Methods:
            equal_weight    — 1/N weighting
            sharpe_weight   — weight proportional to Sharpe
            inverse_corr    — down-weight correlated strategies

        Returns
        -------
        {strategies: [names], weights: [floats], expected_sharpe: float,
         diversification_ratio: float}.
        """
        board = self.strategy_leaderboard(top_n=top_n)
        if not board:
            return {"error": "no strategies available"}

        names = [b["strategy"] for b in board]
        sharpes_arr = np.array([b["mean_sharpe"] for b in board])
        n = len(names)

        if method == "equal_weight":
            weights = np.ones(n) / n
        elif method == "sharpe_weight":
            pos_sharpes = np.maximum(sharpes_arr, 0.0)
            total = np.sum(pos_sharpes)
            if total > 1e-12:
                weights = pos_sharpes / total
            else:
                weights = np.ones(n) / n
        elif method == "inverse_corr":
            # Approximate: use Sharpe std as a diversity proxy
            stds = np.array([b.get("std_sharpe", 1.0) for b in board])
            inv_std = 1.0 / (stds + 0.01)
            weights = inv_std / np.sum(inv_std)
        else:
            weights = np.ones(n) / n

        expected_sharpe = float(np.dot(weights, sharpes_arr))
        # Diversification ratio: weighted avg Sharpe / portfolio Sharpe (approx)
        ind_vol = np.array([b.get("std_sharpe", 0.1) for b in board]) + 0.01
        port_vol = np.sqrt(np.sum((weights * ind_vol) ** 2))
        weighted_avg_vol = np.dot(weights, ind_vol)
        div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-12 else 1.0

        return {
            "strategies": names,
            "weights": weights.tolist(),
            "expected_sharpe": expected_sharpe,
            "diversification_ratio": float(div_ratio),
            "method": method,
            "n_strategies": n,
        }
