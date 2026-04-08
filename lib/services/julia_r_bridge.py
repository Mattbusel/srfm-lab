"""
Julia/R Statistical Services Bridge (T3-1)
gRPC-over-ZeroMQ interface to Julia and R statistical services.

Falls back to Python implementations when services are unavailable.

Julia service capabilities:
  - SDE simulation (Milstein scheme) for BH mass trajectory modeling
  - FFT spectral analysis for cycle detection
  - Tensor network correlation computations

R service capabilities:
  - Johansen cointegration test for pairs signal generation
  - ARFIMA long-memory forecasting
  - Zivot-Andrews structural break detection
  - Robust regression with outlier resilience

All calls are non-blocking with TTL-based caching.
"""
import json
import logging
import time
import math
import threading
from dataclasses import dataclass, field
from typing import Optional, Any
from functools import lru_cache

log = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    julia_host: str = "localhost"
    julia_port: int = 5559
    r_host: str = "localhost"
    r_port: int = 5560
    timeout_ms: int = 100         # max 100ms per call (non-blocking for live trading)
    cache_ttl_seconds: int = 3600  # cache results for 1 hour by default
    fast_cache_ttl_seconds: int = 60  # fast-refresh results (spectral analysis)
    enabled: bool = True

class StatisticalServiceBridge:
    """
    Bridge to Julia/R statistical microservices.

    All methods fall back to Python implementations if services are unavailable.
    This ensures the live trader never blocks waiting for external services.

    Usage:
        bridge = StatisticalServiceBridge()

        # Johansen cointegration test
        result = bridge.johansen_cointegration(["BTC", "ETH"], price_data)
        if result["cointegrated"]:
            hedge_ratio = result["hedge_ratio"]

        # SDE BH mass simulation
        paths = bridge.simulate_bh_mass_sde(mass_init=2.0, n_steps=50, n_paths=100)
    """

    def __init__(self, cfg: ServiceConfig = None):
        self.cfg = cfg or ServiceConfig()
        self._r_available = False
        self._julia_available = False
        self._cache: dict[str, tuple[Any, float]] = {}  # key → (result, expire_time)
        self._lock = threading.Lock()

        if self.cfg.enabled:
            self._probe_services()

    def _probe_services(self):
        """Try to connect to Julia/R services."""
        try:
            import zmq
            ctx = zmq.Context.instance()

            for host, port, name in [
                (self.cfg.r_host, self.cfg.r_port, "R"),
                (self.cfg.julia_host, self.cfg.julia_port, "Julia"),
            ]:
                try:
                    sock = ctx.socket(zmq.REQ)
                    sock.setsockopt(zmq.RCVTIMEO, 500)
                    sock.setsockopt(zmq.SNDTIMEO, 500)
                    sock.connect(f"tcp://{host}:{port}")
                    sock.send_json({"cmd": "ping"})
                    resp = sock.recv_json()
                    sock.close()
                    if resp.get("status") == "ok":
                        if name == "R":
                            self._r_available = True
                        else:
                            self._julia_available = True
                        log.info("StatBridge: %s service available at %s:%d", name, host, port)
                except Exception:
                    log.info("StatBridge: %s service not available (using Python fallback)", name)
        except ImportError:
            log.info("StatBridge: ZeroMQ not installed — using Python fallbacks only")

    def johansen_cointegration(
        self,
        symbols: list[str],
        price_series: dict[str, list[float]],
        max_lags: int = 5,
    ) -> dict:
        """
        Johansen cointegration test. Returns whether pair is cointegrated and hedge ratio.
        Falls back to simple correlation-based cointegration proxy.
        """
        cache_key = f"johansen_{'_'.join(sorted(symbols))}"
        cached = self._get_cache(cache_key, self.cfg.cache_ttl_seconds)
        if cached is not None:
            return cached

        if self._r_available:
            result = self._r_call("johansen_cointegration", {
                "symbols": symbols,
                "prices": price_series,
                "max_lags": max_lags,
            })
            if result:
                self._set_cache(cache_key, result)
                return result

        # Python fallback: Engle-Granger two-step cointegration proxy
        result = self._python_cointegration_proxy(symbols, price_series)
        self._set_cache(cache_key, result)
        return result

    def structural_break_detection(
        self,
        series: list[float],
        symbol: str = "",
    ) -> dict:
        """
        Zivot-Andrews structural break test. Returns break date and confidence.
        Falls back to variance-ratio changepoint detection.
        """
        cache_key = f"zivot_andrews_{symbol}_{len(series)}"
        cached = self._get_cache(cache_key, self.cfg.cache_ttl_seconds)
        if cached is not None:
            return cached

        if self._r_available:
            result = self._r_call("structural_break", {"series": series[-500:], "symbol": symbol})
            if result:
                self._set_cache(cache_key, result)
                return result

        result = self._python_changepoint_detection(series)
        self._set_cache(cache_key, result)
        return result

    def simulate_bh_mass_sde(
        self,
        mass_init: float,
        drift: float = 0.0,
        diffusion: float = 0.1,
        n_steps: int = 50,
        n_paths: int = 100,
        dt: float = 1.0,
    ) -> dict:
        """
        SDE simulation of BH mass trajectory (Milstein scheme).
        Returns distribution of future mass values.
        Falls back to GBM simulation.
        """
        cache_key = f"bh_sde_{mass_init:.3f}_{drift:.3f}_{diffusion:.3f}_{n_steps}"
        cached = self._get_cache(cache_key, self.cfg.fast_cache_ttl_seconds)
        if cached is not None:
            return cached

        if self._julia_available:
            result = self._julia_call("simulate_bh_sde", {
                "mass_init": mass_init,
                "drift": drift,
                "diffusion": diffusion,
                "n_steps": n_steps,
                "n_paths": n_paths,
                "dt": dt,
            })
            if result:
                self._set_cache(cache_key, result)
                return result

        # Python fallback: Euler-Maruyama GBM
        result = self._python_gbm_simulation(mass_init, drift, diffusion, n_steps, n_paths, dt)
        self._set_cache(cache_key, result)
        return result

    def fft_cycle_detection(
        self,
        series: list[float],
        symbol: str = "",
    ) -> dict:
        """
        FFT spectral analysis for dominant cycle detection.
        Returns list of (period_bars, amplitude) pairs.
        """
        cache_key = f"fft_{symbol}_{len(series)}"
        cached = self._get_cache(cache_key, self.cfg.fast_cache_ttl_seconds)
        if cached is not None:
            return cached

        if self._julia_available:
            result = self._julia_call("fft_analysis", {"series": series[-200:], "symbol": symbol})
            if result:
                self._set_cache(cache_key, result)
                return result

        # Python fallback: simple FFT via numpy-less DFT
        result = self._python_fft_proxy(series)
        self._set_cache(cache_key, result)
        return result

    # === Private: Python fallback implementations ===

    def _python_cointegration_proxy(self, symbols: list[str], price_series: dict) -> dict:
        """Simplified Engle-Granger cointegration test."""
        if len(symbols) < 2 or len(price_series) < 2:
            return {"cointegrated": False, "hedge_ratio": 1.0, "confidence": 0.0}

        s1, s2 = symbols[0], symbols[1]
        p1 = price_series.get(s1, [])
        p2 = price_series.get(s2, [])

        if len(p1) < 30 or len(p2) < 30:
            return {"cointegrated": False, "hedge_ratio": 1.0, "confidence": 0.0}

        n = min(len(p1), len(p2))
        x = p1[-n:]
        y = p2[-n:]

        # OLS: y = β*x + ε
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        cov_xy = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        var_x = sum((xi - x_mean)**2 for xi in x)

        if var_x < 1e-12:
            return {"cointegrated": False, "hedge_ratio": 1.0, "confidence": 0.0}

        beta = cov_xy / var_x
        residuals = [y[i] - beta * x[i] for i in range(n)]

        # ADF-like test: autocorrelation of residuals near 1 = not cointegrated
        res_mean = sum(residuals) / n
        res_lagged_cov = sum((residuals[i] - res_mean) * (residuals[i-1] - res_mean) for i in range(1, n))
        res_var = sum((r - res_mean)**2 for r in residuals)

        if res_var < 1e-12:
            return {"cointegrated": False, "hedge_ratio": beta, "confidence": 0.0}

        ar1 = res_lagged_cov / res_var
        # Cointegrated if AR(1) of residuals is significantly less than 1
        confidence = max(0.0, min(1.0, 1.0 - abs(ar1)))
        cointegrated = ar1 < 0.85 and confidence > 0.3

        return {
            "cointegrated": cointegrated,
            "hedge_ratio": float(beta),
            "ar1_residuals": float(ar1),
            "confidence": float(confidence),
            "method": "python_ols_proxy",
        }

    def _python_changepoint_detection(self, series: list[float]) -> dict:
        """Simple variance-ratio changepoint detection."""
        n = len(series)
        if n < 40:
            return {"break_detected": False, "break_index": -1, "confidence": 0.0}

        # Split series at each point and compare variance before/after
        best_idx = n // 2
        best_ratio = 0.0

        window = max(20, n // 4)
        for i in range(window, n - window):
            before = series[max(0, i-window):i]
            after = series[i:i+window]

            var_b = sum((x - sum(before)/len(before))**2 for x in before) / len(before)
            var_a = sum((x - sum(after)/len(after))**2 for x in after) / len(after)

            ratio = max(var_b, var_a) / (min(var_b, var_a) + 1e-12)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        break_detected = best_ratio > 2.0
        return {
            "break_detected": break_detected,
            "break_index": best_idx if break_detected else -1,
            "variance_ratio": float(best_ratio),
            "confidence": min(1.0, (best_ratio - 1.0) / 3.0),
            "method": "python_variance_ratio",
        }

    def _python_gbm_simulation(
        self, mass_init, drift, diffusion, n_steps, n_paths, dt
    ) -> dict:
        """Euler-Maruyama GBM simulation."""
        import random
        rng = random.Random(42)

        final_masses = []
        for _ in range(n_paths):
            m = mass_init
            for _ in range(n_steps):
                dW = rng.gauss(0, math.sqrt(dt))
                m += drift * m * dt + diffusion * m * dW
                m = max(0.0, m)  # mass can't be negative
            final_masses.append(m)

        n = len(final_masses)
        mean_m = sum(final_masses) / n
        sorted_m = sorted(final_masses)

        return {
            "mean_final_mass": mean_m,
            "p10_final_mass": sorted_m[n // 10],
            "p50_final_mass": sorted_m[n // 2],
            "p90_final_mass": sorted_m[int(n * 0.9)],
            "prob_bh_form": sum(1 for m in final_masses if m >= 1.92) / n,
            "method": "python_gbm",
        }

    def _python_fft_proxy(self, series: list[float]) -> dict:
        """Simplified DFT for dominant frequency detection."""
        n = len(series)
        if n < 8:
            return {"dominant_periods": [], "method": "python_dft"}

        # Detrend
        mean_s = sum(series) / n
        s = [x - mean_s for x in series]

        # Compute DFT magnitudes for first n/2 frequencies
        magnitudes = []
        for k in range(1, min(n // 2, 50)):
            real_part = sum(s[j] * math.cos(2 * math.pi * k * j / n) for j in range(n))
            imag_part = sum(s[j] * math.sin(2 * math.pi * k * j / n) for j in range(n))
            magnitude = math.sqrt(real_part**2 + imag_part**2) / n
            period = n / k
            magnitudes.append((period, magnitude))

        # Top 3 dominant periods
        top_periods = sorted(magnitudes, key=lambda x: -x[1])[:3]

        return {
            "dominant_periods": [{"period_bars": p, "amplitude": a} for p, a in top_periods],
            "method": "python_dft",
        }

    # === Private: ZeroMQ service calls ===

    def _r_call(self, cmd: str, params: dict) -> Optional[dict]:
        return self._zmq_call(self.cfg.r_host, self.cfg.r_port, cmd, params)

    def _julia_call(self, cmd: str, params: dict) -> Optional[dict]:
        return self._zmq_call(self.cfg.julia_host, self.cfg.julia_port, cmd, params)

    def _zmq_call(self, host: str, port: int, cmd: str, params: dict) -> Optional[dict]:
        try:
            import zmq
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self.cfg.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.cfg.timeout_ms)
            sock.connect(f"tcp://{host}:{port}")
            sock.send_json({"cmd": cmd, **params})
            result = sock.recv_json()
            sock.close()
            return result
        except Exception as e:
            log.debug("StatBridge %s:%d call failed: %s", host, port, e)
            return None

    def _get_cache(self, key: str, ttl: int) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry and time.time() < entry[1]:
                return entry[0]
        return None

    def _set_cache(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self.cfg.cache_ttl_seconds
        with self._lock:
            self._cache[key] = (value, time.time() + ttl)
            # Evict old entries
            if len(self._cache) > 1000:
                now = time.time()
                self._cache = {k: v for k, v in self._cache.items() if v[1] > now}
