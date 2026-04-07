"""
optimization/genome_bridge.py
==============================
Bridge between the Go Idea Adaptation Engine (IAE) genome evolution
system and the Python LARSA optimization/parameter management stack.

The IAE runs genetic algorithm evolution on float64 arrays (genomes).
Each genome position maps to a named LARSA parameter. This module
provides encoding/decoding, constraint enforcement, and a background
bridge thread that automatically promotes high-fitness genomes to the
Elixir coordination layer.

IAE API (Go server at :8780):
  GET  /genome/best   -- returns {genome: [float64], fitness: float}
  POST /genome/seed   -- body: {genome: [float64], fitness: float, label: str}
  GET  /genome/stats  -- returns {generation, population_size, best_fitness_history, ...}

Classes:
  GenomeDecoder  -- encodes/decodes between named params and float64 arrays
  IAEBridge      -- HTTP client with background polling thread
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import requests

_REPO_ROOT = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(_REPO_ROOT))

from config.param_schema import ParamSchema  # noqa: E402
from config.param_manager import LiveParams, ParamManager  # noqa: E402

logger = logging.getLogger(__name__)

_IAE_BASE = "http://localhost:8780"
_COORD_BASE = "http://localhost:8781"
_POLL_INTERVAL_S = 300    # 5 minutes
_FITNESS_IMPROVEMENT_THRESHOLD = 0.1   # minimum Sharpe gain to trigger proposal


# ---------------------------------------------------------------------------
# GenomeDecoder
# ---------------------------------------------------------------------------

class GenomeDecoder:
    """
    Translates between IAE float64 genome arrays and named parameter dicts.

    The IAE genome is a fixed-length array where each index corresponds
    to exactly one LARSA parameter. The ordering is determined at construction
    time from the schema and must remain stable for the lifetime of the study.

    Boolean parameters are represented as floats in [0, 1] and rounded
    to the nearest integer for decoding. list_int parameters (BLOCKED_HOURS)
    are encoded as a bitmask over the 24-hour range.

    Constraints are enforced during decode to guarantee valid parameter sets
    are always produced regardless of the raw genome values.
    """

    # Fixed ordering for reproducibility -- matches schema parameter order
    # list_int params are handled specially (BLOCKED_HOURS -> 24 bits)
    _BITMASK_PARAMS = {"BLOCKED_HOURS"}
    _BITMASK_HOURS = 24

    def __init__(self, schema: Optional[ParamSchema] = None) -> None:
        self._schema = schema or ParamSchema()
        self._param_order: list[str] = []
        self._genome_indices: dict[str, int | list[int]] = {}
        self._genome_length: int = 0
        self._build_index()

    def _build_index(self) -> None:
        """
        Build the mapping from parameter name to genome index/indices.
        list_int params (BLOCKED_HOURS) occupy multiple consecutive positions.
        """
        idx = 0
        for name in self._schema.parameter_names:
            spec = self._schema.get_spec(name)
            ptype = spec["type"]
            self._param_order.append(name)
            if ptype == "list_int" and name in self._bitmask_params():
                # Allocate one slot per possible hour (0-23)
                self._genome_indices[name] = list(range(idx, idx + self._BITMASK_HOURS))
                idx += self._BITMASK_HOURS
            else:
                self._genome_indices[name] = idx
                idx += 1
        self._genome_length = idx
        logger.debug(
            "GenomeDecoder: %d params -> genome length %d",
            len(self._param_order), self._genome_length,
        )

    def _bitmask_params(self) -> set[str]:
        return self._BITMASK_PARAMS

    @property
    def genome_length(self) -> int:
        return self._genome_length

    @property
    def param_order(self) -> list[str]:
        return list(self._param_order)

    # ------------------------------------------------------------------
    # Decode: genome -> named params
    # ------------------------------------------------------------------

    def decode(self, genome: list[float]) -> dict[str, Any]:
        """
        Map a float64 genome array to a named parameter dict.

        Clamps all values to schema ranges, rounds integers and booleans,
        and enforces cross-parameter constraints.

        Args:
            genome: list/array of float64 with length == genome_length

        Returns:
            dict of {param_name: value} ready for use as LiveParams
        """
        if len(genome) != self._genome_length:
            raise ValueError(
                f"Genome length {len(genome)} does not match expected {self._genome_length}"
            )

        result: dict[str, Any] = {}

        for name in self._param_order:
            spec = self._schema.get_spec(name)
            ptype = spec["type"]
            gene_idx = self._genome_indices[name]

            if ptype == "float":
                raw = float(genome[gene_idx])
                lo = spec.get("min", -1e9)
                hi = spec.get("max", 1e9)
                value = float(np.clip(raw, lo, hi))
                step = spec.get("step")
                if step is not None and step > 0:
                    value = round(round(value / step) * step, 10)
                result[name] = float(np.clip(value, lo, hi))

            elif ptype == "int":
                raw = float(genome[gene_idx])
                lo = int(spec.get("min", 0))
                hi = int(spec.get("max", 1000))
                step = int(spec.get("step", 1))
                value = int(round(np.clip(raw, lo, hi)))
                if step > 1:
                    value = round(value / step) * step
                result[name] = int(np.clip(value, lo, hi))

            elif ptype == "bool":
                raw = float(genome[gene_idx])
                result[name] = bool(raw >= 0.5)

            elif ptype == "list_int" and name in self._bitmask_params():
                indices = gene_idx  # list of 24 indices
                hours = []
                for hour, bit_idx in enumerate(indices):
                    if float(genome[bit_idx]) >= 0.5:
                        hours.append(hour)
                result[name] = hours

            else:
                # Fallback: treat as float
                raw = float(genome[gene_idx]) if isinstance(gene_idx, int) else 0.0
                result[name] = raw

        # Enforce cross-parameter constraints
        result = self._enforce_constraints(result)
        return result

    def _enforce_constraints(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Apply all cross-parameter constraints by adjusting values in-place.
        Repair strategy: clamp or extend the violating parameter minimally.
        """
        p = dict(params)

        # CF_BEAR_THRESH >= CF_BULL_THRESH
        bull = p.get("CF_BULL_THRESH", 1.2)
        bear = p.get("CF_BEAR_THRESH", 1.4)
        if bear < bull:
            p["CF_BEAR_THRESH"] = bull
            logger.debug("Genome constraint repair: CF_BEAR_THRESH <- %s", bull)

        # BH_MASS_EXTREME > BH_MASS_THRESH
        bh_thresh = p.get("BH_MASS_THRESH", 1.92)
        bh_extreme = p.get("BH_MASS_EXTREME", 3.5)
        if bh_extreme <= bh_thresh:
            p["BH_MASS_EXTREME"] = bh_thresh + 0.5
            logger.debug("Genome constraint repair: BH_MASS_EXTREME <- %s", p["BH_MASS_EXTREME"])

        # MAX_HOLD_BARS > MIN_HOLD_BARS
        min_hold = p.get("MIN_HOLD_BARS", 4)
        max_hold = p.get("MAX_HOLD_BARS", 96)
        if max_hold <= min_hold:
            p["MAX_HOLD_BARS"] = min_hold + 4
            logger.debug("Genome constraint repair: MAX_HOLD_BARS <- %s", p["MAX_HOLD_BARS"])

        # MAX_RISK_PCT >= BASE_RISK_PCT
        base_risk = p.get("BASE_RISK_PCT", 0.02)
        max_risk = p.get("MAX_RISK_PCT", 0.05)
        if max_risk < base_risk:
            p["MAX_RISK_PCT"] = base_risk
            logger.debug("Genome constraint repair: MAX_RISK_PCT <- %s", base_risk)

        # GARCH stationarity: alpha + beta < 1.0
        alpha = p.get("GARCH_ALPHA", 0.09)
        beta = p.get("GARCH_BETA", 0.88)
        if alpha + beta >= 1.0:
            scale = 0.97 / (alpha + beta)
            p["GARCH_ALPHA"] = round(alpha * scale, 6)
            p["GARCH_BETA"] = round(beta * scale, 6)
            logger.debug(
                "Genome constraint repair: GARCH alpha+beta scaled to %.4f + %.4f",
                p["GARCH_ALPHA"], p["GARCH_BETA"],
            )

        # OU_KAPPA_MIN < OU_KAPPA_MAX
        ou_min = p.get("OU_KAPPA_MIN", 0.05)
        ou_max = p.get("OU_KAPPA_MAX", 2.0)
        if ou_min >= ou_max:
            p["OU_KAPPA_MAX"] = ou_min + 0.1
            logger.debug("Genome constraint repair: OU_KAPPA_MAX <- %s", p["OU_KAPPA_MAX"])

        # ML suppress < ML boost
        suppress = p.get("ML_SIGNAL_SUPPRESS_THRESH", -0.30)
        boost_thresh = p.get("ML_SIGNAL_BOOST_THRESH", 0.30)
        if suppress >= boost_thresh:
            p["ML_SIGNAL_SUPPRESS_THRESH"] = boost_thresh - 0.1
            logger.debug("Genome constraint repair: ML_SIGNAL_SUPPRESS_THRESH <- %s", p["ML_SIGNAL_SUPPRESS_THRESH"])

        return p

    # ------------------------------------------------------------------
    # Encode: named params -> genome
    # ------------------------------------------------------------------

    def encode(self, params: dict[str, Any]) -> list[float]:
        """
        Map a named parameter dict to a float64 genome array.

        Missing parameters are filled with schema defaults before encoding.
        This is used to seed the IAE with known-good parameter sets.

        Args:
            params: dict of {param_name: value}

        Returns:
            list of float64 with length == genome_length
        """
        filled = self._schema.fill_defaults(params)
        genome = [0.0] * self._genome_length

        for name in self._param_order:
            spec = self._schema.get_spec(name)
            ptype = spec["type"]
            value = filled.get(name, spec.get("default", 0))
            gene_idx = self._genome_indices[name]

            if ptype == "float":
                lo = spec.get("min", -1e9)
                hi = spec.get("max", 1e9)
                genome[gene_idx] = float(np.clip(value, lo, hi))

            elif ptype == "int":
                lo = spec.get("min", 0)
                hi = spec.get("max", 1000)
                genome[gene_idx] = float(np.clip(int(value), lo, hi))

            elif ptype == "bool":
                genome[gene_idx] = 1.0 if value else 0.0

            elif ptype == "list_int" and name in self._bitmask_params():
                indices = gene_idx  # list of 24 indices
                hour_set = set(value) if isinstance(value, (list, tuple)) else set()
                for hour, bit_idx in enumerate(indices):
                    genome[bit_idx] = 1.0 if hour in hour_set else 0.0

        return genome

    def random_genome(self, rng: Optional[np.random.Generator] = None) -> list[float]:
        """
        Generate a random genome by sampling each parameter uniformly
        from its schema range. Useful for seeding the initial IAE population.
        """
        if rng is None:
            rng = np.random.default_rng()
        genome = [0.0] * self._genome_length
        for name in self._param_order:
            spec = self._schema.get_spec(name)
            ptype = spec["type"]
            gene_idx = self._genome_indices[name]
            if ptype == "float":
                lo, hi = spec.get("min", 0.0), spec.get("max", 1.0)
                if isinstance(gene_idx, int):
                    genome[gene_idx] = float(rng.uniform(lo, hi))
            elif ptype == "int":
                lo, hi = int(spec.get("min", 0)), int(spec.get("max", 100))
                if isinstance(gene_idx, int):
                    genome[gene_idx] = float(rng.integers(lo, hi + 1))
            elif ptype == "bool":
                if isinstance(gene_idx, int):
                    genome[gene_idx] = float(rng.integers(0, 2))
            elif ptype == "list_int" and name in self._bitmask_params():
                indices = gene_idx
                for bit_idx in indices:
                    genome[bit_idx] = float(rng.integers(0, 2))
        return genome


# ---------------------------------------------------------------------------
# IAEBridge
# ---------------------------------------------------------------------------

class IAEBridge:
    """
    HTTP client for the Go Idea Adaptation Engine (IAE) at :8780.

    Provides:
      - fetch_latest_genome() -- GET the current best genome
      - push_elite_params()   -- POST a known-good genome to seed the population
      - get_evolution_stats() -- GET generation/population/fitness history
      - Background polling thread that auto-promotes high-fitness genomes

    Background thread behavior:
      Every _POLL_INTERVAL_S seconds, fetches the IAE best genome.
      If the new genome's fitness exceeds the last known fitness by
      _FITNESS_IMPROVEMENT_THRESHOLD, decodes the genome and proposes
      the parameters to the Elixir coordination layer.
    """

    def __init__(
        self,
        decoder: Optional[GenomeDecoder] = None,
        manager: Optional[ParamManager] = None,
        iae_base: str = _IAE_BASE,
        timeout: float = 10.0,
        poll_interval: float = _POLL_INTERVAL_S,
        fitness_threshold: float = _FITNESS_IMPROVEMENT_THRESHOLD,
    ) -> None:
        self._decoder = decoder or GenomeDecoder()
        self._manager = manager or ParamManager()
        self._iae_base = iae_base.rstrip("/")
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._fitness_threshold = fitness_threshold
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_known_fitness: float = float("-inf")
        self._last_known_genome: Optional[list[float]] = None
        self._promotion_callbacks: list[Callable[[dict, float], None]] = []

    # ------------------------------------------------------------------
    # IAE HTTP API
    # ------------------------------------------------------------------

    def fetch_latest_genome(self) -> tuple[list[float], float]:
        """
        Fetch the current best genome from the IAE.

        GET :8780/genome/best

        Returns (genome: list[float], fitness: float).
        Returns ([], -inf) if the IAE is unreachable.
        """
        url = f"{self._iae_base}/genome/best"
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            genome = data.get("genome", [])
            fitness = float(data.get("fitness", float("-inf")))
            if not isinstance(genome, list):
                logger.warning("IAE /genome/best returned non-list genome: %s", type(genome))
                return [], float("-inf")
            logger.debug("Fetched IAE genome: fitness=%.4f, length=%d", fitness, len(genome))
            return [float(g) for g in genome], fitness
        except requests.exceptions.ConnectionError:
            logger.debug("IAE unreachable at %s", url)
            return [], float("-inf")
        except requests.exceptions.Timeout:
            logger.debug("Timeout fetching IAE genome")
            return [], float("-inf")
        except (requests.exceptions.HTTPError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("IAE genome fetch error: %s", exc)
            return [], float("-inf")

    def push_elite_params(
        self,
        params: dict[str, Any],
        label: str = "python_elite",
        fitness: Optional[float] = None,
    ) -> bool:
        """
        Encode params as a genome and POST to IAE to seed the population.

        POST :8780/genome/seed
        Body: {genome: [float64], fitness: float, label: str}

        This is used to bootstrap the IAE with known-good parameters
        from Optuna walk-forward optimization.

        Returns True on success.
        """
        url = f"{self._iae_base}/genome/seed"
        genome = self._decoder.encode(params)
        payload: dict[str, Any] = {
            "genome": genome,
            "label": label,
        }
        if fitness is not None:
            payload["fitness"] = fitness

        try:
            resp = self._session.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            result = resp.json()
            success = result.get("success", True)
            if success:
                logger.info("Pushed elite params to IAE (label=%s, genome_len=%d)", label, len(genome))
            else:
                logger.warning("IAE rejected elite genome: %s", result.get("reason"))
            return bool(success)
        except requests.exceptions.ConnectionError:
            logger.error("IAE unreachable for genome/seed push")
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout pushing elite genome to IAE")
            return False
        except (requests.exceptions.HTTPError, json.JSONDecodeError) as exc:
            logger.error("IAE genome push error: %s", exc)
            return False

    def get_evolution_stats(self) -> dict[str, Any]:
        """
        Retrieve evolution progress statistics from the IAE.

        GET :8780/genome/stats

        Returns a dict with keys:
          generation (int), population_size (int),
          best_fitness_history (list[float]), mean_fitness_history (list[float]),
          n_evaluations (int), elapsed_seconds (float)

        Returns {} on error.
        """
        url = f"{self._iae_base}/genome/stats"
        try:
            resp = self._session.get(url, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            return data
        except requests.exceptions.ConnectionError:
            logger.debug("IAE unreachable for stats")
            return {}
        except requests.exceptions.Timeout:
            logger.debug("Timeout fetching IAE stats")
            return {}
        except (requests.exceptions.HTTPError, json.JSONDecodeError) as exc:
            logger.warning("IAE stats fetch error: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Background promotion thread
    # ------------------------------------------------------------------

    def register_promotion_callback(
        self, callback: Callable[[dict, float], None]
    ) -> None:
        """
        Register a callback to be invoked when a genome is promoted.

        callback(params: dict, fitness: float) -> None
        """
        self._promotion_callbacks.append(callback)

    def start_polling(self, poll_interval: Optional[float] = None) -> None:
        """
        Start the background thread that polls the IAE and promotes
        genomes with improved fitness to the coordination layer.

        Safe to call multiple times; subsequent calls are no-ops if
        the thread is already running.
        """
        if self._poll_thread is not None and self._poll_thread.is_alive():
            logger.debug("IAEBridge poll thread already running")
            return

        interval = poll_interval if poll_interval is not None else self._poll_interval
        self._stop_event.clear()

        def _run() -> None:
            logger.info(
                "IAEBridge poll thread started (interval=%.0fs, threshold=%.3f)",
                interval, self._fitness_threshold,
            )
            while not self._stop_event.is_set():
                self._poll_once()
                self._stop_event.wait(interval)
            logger.info("IAEBridge poll thread stopped")

        self._poll_thread = threading.Thread(
            target=_run, daemon=True, name="iae-bridge-poller"
        )
        self._poll_thread.start()

    def stop_polling(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=10.0)
            self._poll_thread = None

    def _poll_once(self) -> None:
        """
        Single poll cycle: fetch best genome, compare to last known fitness,
        promote if improved by threshold.
        """
        genome, fitness = self.fetch_latest_genome()
        if not genome or not math.isfinite(fitness):
            return

        improvement = fitness - self._last_known_fitness
        if improvement <= self._fitness_threshold:
            logger.debug(
                "IAE fitness %.4f (improvement=%.4f < threshold=%.4f) -- no action",
                fitness, improvement, self._fitness_threshold,
            )
            return

        logger.info(
            "IAE fitness improved: %.4f -> %.4f (delta=+%.4f) -- promoting to coordination layer",
            self._last_known_fitness, fitness, improvement,
        )

        # Decode genome to params
        if len(genome) != self._decoder.genome_length:
            logger.warning(
                "IAE genome length %d != decoder expected %d -- skipping promotion",
                len(genome), self._decoder.genome_length,
            )
            return

        try:
            params = self._decoder.decode(genome)
        except Exception as exc:
            logger.error("Failed to decode IAE genome: %s", exc)
            return

        # Validate before proposing
        ok, reason = self._manager.validate_locally(params)
        if not ok:
            logger.error("Decoded IAE genome failed validation: %s", reason)
            return

        # Propose to coordination layer
        accepted = self._manager.propose_update(params, source="iae_genome")
        if accepted:
            self._last_known_fitness = fitness
            self._last_known_genome = genome
            logger.info("IAE genome promoted (fitness=%.4f)", fitness)
            for cb in self._promotion_callbacks:
                try:
                    cb(params, fitness)
                except Exception as cb_exc:
                    logger.error("Promotion callback raised: %s", cb_exc)
        else:
            logger.warning("Coordination layer rejected IAE genome promotion")

    # ------------------------------------------------------------------
    # Analysis utilities
    # ------------------------------------------------------------------

    def fitness_history(self) -> list[float]:
        """Return the best-fitness-per-generation history from the IAE."""
        stats = self.get_evolution_stats()
        return stats.get("best_fitness_history", [])

    def current_generation(self) -> int:
        """Return the current IAE generation number."""
        stats = self.get_evolution_stats()
        return int(stats.get("generation", 0))

    def population_size(self) -> int:
        """Return the current IAE population size."""
        stats = self.get_evolution_stats()
        return int(stats.get("population_size", 0))

    def is_converged(self, window: int = 20, tol: float = 1e-4) -> bool:
        """
        Heuristic convergence check: the best fitness has not improved
        by more than tol over the last window generations.
        """
        history = self.fitness_history()
        if len(history) < window + 1:
            return False
        recent = history[-window:]
        return (max(recent) - min(recent)) < tol

    def decode_genome(self, genome: list[float]) -> dict[str, Any]:
        """Public decode with constraint enforcement."""
        return self._decoder.decode(genome)

    def encode_params(self, params: dict[str, Any]) -> list[float]:
        """Public encode from params dict to genome."""
        return self._decoder.encode(params)

    def seed_from_live_params(self, live: LiveParams, label: str = "live_trader") -> bool:
        """Convenience: encode a LiveParams instance and push to IAE."""
        params = {
            k: v for k, v in live.to_dict().items()
            if k not in ("version", "source", "timestamp")
        }
        return self.push_elite_params(params, label=label)

    def __repr__(self) -> str:
        return (
            f"IAEBridge(iae={self._iae_base!r}, "
            f"last_fitness={self._last_known_fitness:.4f}, "
            f"polling={self._poll_thread is not None and self._poll_thread.is_alive()})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------

def create_bridge(
    schema: Optional[ParamSchema] = None,
    manager: Optional[ParamManager] = None,
) -> IAEBridge:
    """Create an IAEBridge with default configuration."""
    s = schema or ParamSchema()
    m = manager or ParamManager(s)
    decoder = GenomeDecoder(s)
    return IAEBridge(decoder=decoder, manager=m)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bridge = create_bridge()
    print("Genome length:", bridge._decoder.genome_length)
    genome, fitness = bridge.fetch_latest_genome()
    if genome:
        params = bridge.decode_genome(genome)
        print(f"IAE best genome (fitness={fitness:.4f}):")
        for k, v in params.items():
            print(f"  {k}: {v}")
    else:
        print("IAE not reachable -- generating random genome for demo")
        rng = np.random.default_rng(42)
        genome = bridge._decoder.random_genome(rng)
        params = bridge.decode_genome(genome)
        print("Random genome decoded params:")
        for k, v in params.items():
            print(f"  {k}: {v}")
