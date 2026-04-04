"""
Strategy genome encoding for genetic algorithm optimization.

Provides parameter ranges, chromosome representation, mutation operators,
and crossover operators for evolving trading strategies.
"""

from __future__ import annotations

import copy
import math
import random
import struct
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Parameter types and descriptors
# ---------------------------------------------------------------------------

class ParamType(Enum):
    CONTINUOUS = auto()   # float in [low, high]
    INTEGER = auto()      # int in [low, high]
    CATEGORICAL = auto()  # one of a set of choices
    BOOLEAN = auto()      # True / False
    LOG_SCALE = auto()    # float on log scale (e.g. learning rate)


@dataclass
class ParamRange:
    """Descriptor for a single evolvable parameter."""
    name: str
    param_type: ParamType
    low: float = 0.0
    high: float = 1.0
    choices: Optional[List[Any]] = None   # for CATEGORICAL
    step: Optional[float] = None          # for INTEGER discretization
    default: Optional[Any] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.param_type == ParamType.CATEGORICAL and not self.choices:
            raise ValueError(f"ParamRange '{self.name}': CATEGORICAL requires choices.")
        if self.param_type == ParamType.BOOLEAN:
            self.choices = [False, True]
            self.low = 0.0
            self.high = 1.0
        if self.default is None:
            self.default = self._midpoint()

    def _midpoint(self) -> Any:
        if self.param_type == ParamType.CATEGORICAL:
            return self.choices[len(self.choices) // 2]
        if self.param_type == ParamType.BOOLEAN:
            return False
        mid = (self.low + self.high) / 2.0
        if self.param_type == ParamType.INTEGER:
            return int(round(mid))
        return mid

    def clip(self, value: Any) -> Any:
        """Clip a value to the valid range for this parameter."""
        if self.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
            if value not in self.choices:
                return self.choices[0]
            return value
        if self.param_type == ParamType.LOG_SCALE:
            log_low = math.log(self.low) if self.low > 0 else -20.0
            log_high = math.log(self.high)
            log_val = math.log(max(value, 1e-300))
            log_val = max(log_low, min(log_high, log_val))
            return math.exp(log_val)
        value = max(self.low, min(self.high, value))
        if self.param_type == ParamType.INTEGER:
            return int(round(value))
        return float(value)

    def random_value(self, rng: random.Random) -> Any:
        """Sample a uniformly random value within range."""
        if self.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
            return rng.choice(self.choices)
        if self.param_type == ParamType.LOG_SCALE:
            log_low = math.log(self.low) if self.low > 0 else -10.0
            log_high = math.log(self.high)
            return math.exp(rng.uniform(log_low, log_high))
        if self.param_type == ParamType.INTEGER:
            return rng.randint(int(self.low), int(self.high))
        return rng.uniform(self.low, self.high)

    def encode_float(self, value: Any) -> float:
        """Encode value as float in [0, 1] for crossover arithmetic."""
        if self.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
            idx = self.choices.index(value) if value in self.choices else 0
            return idx / max(len(self.choices) - 1, 1)
        if self.param_type == ParamType.LOG_SCALE:
            log_low = math.log(self.low) if self.low > 0 else -10.0
            log_high = math.log(self.high)
            log_val = math.log(max(value, 1e-300))
            return (log_val - log_low) / max(log_high - log_low, 1e-10)
        span = self.high - self.low
        if span == 0:
            return 0.0
        return (value - self.low) / span

    def decode_float(self, encoded: float) -> Any:
        """Decode a [0, 1] float back to a parameter value."""
        encoded = max(0.0, min(1.0, encoded))
        if self.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
            idx = int(round(encoded * (len(self.choices) - 1)))
            idx = max(0, min(len(self.choices) - 1, idx))
            return self.choices[idx]
        if self.param_type == ParamType.LOG_SCALE:
            log_low = math.log(self.low) if self.low > 0 else -10.0
            log_high = math.log(self.high)
            log_val = log_low + encoded * (log_high - log_low)
            return math.exp(log_val)
        value = self.low + encoded * (self.high - self.low)
        return self.clip(value)


# ---------------------------------------------------------------------------
# Pre-built parameter schemas for common strategies
# ---------------------------------------------------------------------------

def momentum_strategy_params() -> List[ParamRange]:
    return [
        ParamRange("lookback_fast", ParamType.INTEGER, 5, 30, default=10,
                   description="Fast momentum lookback window"),
        ParamRange("lookback_slow", ParamType.INTEGER, 20, 200, default=60,
                   description="Slow momentum lookback window"),
        ParamRange("vol_lookback", ParamType.INTEGER, 10, 60, default=20,
                   description="Volatility estimation window"),
        ParamRange("signal_threshold", ParamType.CONTINUOUS, 0.0, 2.0, default=0.5,
                   description="Z-score entry threshold"),
        ParamRange("exit_threshold", ParamType.CONTINUOUS, 0.0, 1.0, default=0.1,
                   description="Z-score exit threshold"),
        ParamRange("max_position", ParamType.CONTINUOUS, 0.1, 1.0, default=0.5,
                   description="Maximum position size as fraction of portfolio"),
        ParamRange("vol_target", ParamType.CONTINUOUS, 0.05, 0.30, default=0.15,
                   description="Annualized volatility target"),
        ParamRange("rebalance_freq", ParamType.INTEGER, 1, 20, default=5,
                   description="Rebalancing frequency in days"),
        ParamRange("use_volume_filter", ParamType.BOOLEAN, default=True,
                   description="Apply volume filter before entering"),
        ParamRange("volume_multiplier", ParamType.CONTINUOUS, 1.0, 3.0, default=1.5,
                   description="Required volume vs 20-day average"),
        ParamRange("stop_loss", ParamType.CONTINUOUS, 0.01, 0.15, default=0.05,
                   description="Hard stop loss fraction"),
        ParamRange("take_profit", ParamType.CONTINUOUS, 0.02, 0.30, default=0.10,
                   description="Take profit fraction"),
    ]


def mean_reversion_strategy_params() -> List[ParamRange]:
    return [
        ParamRange("bb_window", ParamType.INTEGER, 10, 50, default=20,
                   description="Bollinger Band window"),
        ParamRange("bb_std", ParamType.CONTINUOUS, 1.0, 3.0, default=2.0,
                   description="Bollinger Band standard deviations"),
        ParamRange("rsi_window", ParamType.INTEGER, 5, 30, default=14,
                   description="RSI window"),
        ParamRange("rsi_oversold", ParamType.CONTINUOUS, 20.0, 40.0, default=30.0,
                   description="RSI oversold threshold"),
        ParamRange("rsi_overbought", ParamType.CONTINUOUS, 60.0, 80.0, default=70.0,
                   description="RSI overbought threshold"),
        ParamRange("mean_window", ParamType.INTEGER, 10, 100, default=30,
                   description="Mean estimation window"),
        ParamRange("entry_zscore", ParamType.CONTINUOUS, 1.0, 3.0, default=2.0,
                   description="Z-score entry threshold"),
        ParamRange("exit_zscore", ParamType.CONTINUOUS, 0.0, 1.0, default=0.2,
                   description="Z-score exit threshold"),
        ParamRange("max_holding_days", ParamType.INTEGER, 1, 30, default=10,
                   description="Maximum holding period before force exit"),
        ParamRange("position_size", ParamType.CONTINUOUS, 0.05, 0.5, default=0.2,
                   description="Fixed position size"),
        ParamRange("use_kalman_filter", ParamType.BOOLEAN, default=False,
                   description="Use Kalman filter for mean estimation"),
    ]


def pairs_trading_params() -> List[ParamRange]:
    return [
        ParamRange("coint_window", ParamType.INTEGER, 60, 252, default=120,
                   description="Cointegration estimation window"),
        ParamRange("spread_window", ParamType.INTEGER, 5, 60, default=20,
                   description="Spread z-score window"),
        ParamRange("entry_zscore", ParamType.CONTINUOUS, 1.0, 3.0, default=2.0),
        ParamRange("exit_zscore", ParamType.CONTINUOUS, 0.0, 1.0, default=0.2),
        ParamRange("stop_zscore", ParamType.CONTINUOUS, 2.0, 5.0, default=3.5,
                   description="Stop loss z-score"),
        ParamRange("hedge_ratio_method", ParamType.CATEGORICAL,
                   choices=["ols", "tls", "kalman", "rolling_ols"], default="ols"),
        ParamRange("max_holding_days", ParamType.INTEGER, 1, 60, default=20),
        ParamRange("position_size", ParamType.CONTINUOUS, 0.05, 0.5, default=0.2),
        ParamRange("transaction_cost_bps", ParamType.CONTINUOUS, 0.0, 20.0, default=5.0),
    ]


def ml_hyperparameter_params() -> List[ParamRange]:
    """Hyperparameters for an ML-based trading model."""
    return [
        ParamRange("learning_rate", ParamType.LOG_SCALE, 1e-5, 1e-1, default=3e-4,
                   description="Optimizer learning rate"),
        ParamRange("batch_size", ParamType.CATEGORICAL,
                   choices=[16, 32, 64, 128, 256, 512], default=64),
        ParamRange("hidden_dim", ParamType.CATEGORICAL,
                   choices=[64, 128, 256, 512, 1024], default=256),
        ParamRange("num_layers", ParamType.INTEGER, 1, 6, default=2),
        ParamRange("dropout", ParamType.CONTINUOUS, 0.0, 0.5, default=0.1),
        ParamRange("weight_decay", ParamType.LOG_SCALE, 1e-6, 1e-1, default=1e-4),
        ParamRange("lookback_window", ParamType.INTEGER, 5, 120, default=30),
        ParamRange("prediction_horizon", ParamType.INTEGER, 1, 22, default=5),
        ParamRange("feature_selection_pct", ParamType.CONTINUOUS, 0.3, 1.0, default=0.8),
        ParamRange("l1_penalty", ParamType.LOG_SCALE, 1e-6, 1e-1, default=1e-4),
        ParamRange("optimizer", ParamType.CATEGORICAL,
                   choices=["adam", "adamw", "sgd", "rmsprop"], default="adam"),
        ParamRange("activation", ParamType.CATEGORICAL,
                   choices=["relu", "gelu", "tanh", "elu"], default="relu"),
        ParamRange("use_batch_norm", ParamType.BOOLEAN, default=True),
        ParamRange("gradient_clip", ParamType.CONTINUOUS, 0.1, 10.0, default=1.0),
        ParamRange("warmup_steps", ParamType.INTEGER, 0, 1000, default=100),
    ]


def portfolio_weights_params(n_assets: int = 10) -> List[ParamRange]:
    """Portfolio weight parameters for n_assets."""
    params = []
    for i in range(n_assets):
        params.append(ParamRange(
            f"weight_{i}", ParamType.CONTINUOUS, 0.0, 1.0, default=1.0 / n_assets,
            description=f"Portfolio weight for asset {i}"
        ))
    params.extend([
        ParamRange("rebalance_threshold", ParamType.CONTINUOUS, 0.01, 0.20, default=0.05,
                   description="Drift threshold before rebalancing"),
        ParamRange("min_weight", ParamType.CONTINUOUS, 0.0, 0.10, default=0.02,
                   description="Minimum weight per asset"),
        ParamRange("max_weight", ParamType.CONTINUOUS, 0.10, 1.0, default=0.30,
                   description="Maximum weight per asset"),
        ParamRange("use_leverage", ParamType.BOOLEAN, default=False),
        ParamRange("leverage_limit", ParamType.CONTINUOUS, 1.0, 3.0, default=1.5),
    ])
    return params


# ---------------------------------------------------------------------------
# Gene: smallest evolvable unit
# ---------------------------------------------------------------------------

@dataclass
class Gene:
    """A single parameter gene with its descriptor."""
    param: ParamRange
    value: Any

    def clone(self) -> "Gene":
        return Gene(param=self.param, value=copy.deepcopy(self.value))

    def as_float(self) -> float:
        return self.param.encode_float(self.value)

    @classmethod
    def from_float(cls, param: ParamRange, encoded: float) -> "Gene":
        return cls(param=param, value=param.decode_float(encoded))

    def __repr__(self) -> str:
        return f"Gene({self.param.name}={self.value!r})"


# ---------------------------------------------------------------------------
# Chromosome: ordered sequence of genes
# ---------------------------------------------------------------------------

class Chromosome:
    """
    An ordered sequence of genes representing a complete set of parameters
    for a trading strategy or model configuration.
    """

    def __init__(self, genes: List[Gene], rng: Optional[random.Random] = None) -> None:
        self.genes: List[Gene] = genes
        self._rng = rng or random.Random()
        self._param_index: Dict[str, int] = {g.param.name: i for i, g in enumerate(genes)}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, params: List[ParamRange],
               rng: Optional[random.Random] = None) -> "Chromosome":
        _rng = rng or random.Random()
        genes = [Gene(p, p.random_value(_rng)) for p in params]
        return cls(genes, rng=_rng)

    @classmethod
    def from_defaults(cls, params: List[ParamRange],
                      rng: Optional[random.Random] = None) -> "Chromosome":
        genes = [Gene(p, p.default) for p in params]
        return cls(genes, rng=rng or random.Random())

    @classmethod
    def from_dict(cls, params: List[ParamRange], values: Dict[str, Any],
                  rng: Optional[random.Random] = None) -> "Chromosome":
        genes = []
        for p in params:
            v = values.get(p.name, p.default)
            genes.append(Gene(p, p.clip(v)))
        return cls(genes, rng=rng or random.Random())

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.genes)

    def __getitem__(self, key: Union[int, str]) -> Any:
        if isinstance(key, str):
            return self.genes[self._param_index[key]].value
        return self.genes[key].value

    def __setitem__(self, key: Union[int, str], value: Any) -> None:
        if isinstance(key, str):
            idx = self._param_index[key]
        else:
            idx = key
        gene = self.genes[idx]
        gene.value = gene.param.clip(value)

    def get(self, name: str, default: Any = None) -> Any:
        if name in self._param_index:
            return self.genes[self._param_index[name]].value
        return default

    def to_dict(self) -> Dict[str, Any]:
        return {g.param.name: g.value for g in self.genes}

    def to_float_vector(self) -> List[float]:
        return [g.as_float() for g in self.genes]

    @classmethod
    def from_float_vector(cls, params: List[ParamRange], vector: List[float],
                          rng: Optional[random.Random] = None) -> "Chromosome":
        genes = [Gene.from_float(p, vector[i]) for i, p in enumerate(params)]
        return cls(genes, rng=rng or random.Random())

    def clone(self) -> "Chromosome":
        new_chr = Chromosome([g.clone() for g in self.genes], rng=random.Random(self._rng.random()))
        return new_chr

    def fingerprint(self) -> str:
        """Deterministic hash of gene values for deduplication."""
        data = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.md5(data).hexdigest()[:16]

    def __repr__(self) -> str:
        params_str = ", ".join(f"{g.param.name}={g.value}" for g in self.genes[:4])
        if len(self.genes) > 4:
            params_str += f", ... ({len(self.genes)} params)"
        return f"Chromosome({params_str})"


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

class MutationOperator:
    """Collection of mutation operators for chromosomes."""

    @staticmethod
    def gaussian(chromosome: Chromosome, mutation_rate: float,
                 sigma_scale: float = 0.1, rng: Optional[random.Random] = None) -> Chromosome:
        """
        Gaussian mutation: add N(0, sigma) noise to each gene with probability mutation_rate.
        sigma is proportional to the parameter range width.
        """
        _rng = rng or random.Random()
        child = chromosome.clone()
        for gene in child.genes:
            if _rng.random() < mutation_rate:
                if gene.param.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
                    # Random reset for discrete types
                    gene.value = gene.param.random_value(_rng)
                elif gene.param.param_type == ParamType.LOG_SCALE:
                    log_low = math.log(gene.param.low) if gene.param.low > 0 else -10.0
                    log_high = math.log(gene.param.high)
                    span = log_high - log_low
                    log_val = math.log(max(gene.value, 1e-300))
                    log_val += _rng.gauss(0.0, sigma_scale * span)
                    log_val = max(log_low, min(log_high, log_val))
                    gene.value = math.exp(log_val)
                elif gene.param.param_type == ParamType.INTEGER:
                    span = gene.param.high - gene.param.low
                    delta = int(round(_rng.gauss(0.0, sigma_scale * span)))
                    gene.value = gene.param.clip(gene.value + delta)
                else:
                    span = gene.param.high - gene.param.low
                    delta = _rng.gauss(0.0, sigma_scale * span)
                    gene.value = gene.param.clip(gene.value + delta)
        return child

    @staticmethod
    def uniform(chromosome: Chromosome, mutation_rate: float,
                rng: Optional[random.Random] = None) -> Chromosome:
        """
        Uniform mutation: replace gene with random value with probability mutation_rate.
        """
        _rng = rng or random.Random()
        child = chromosome.clone()
        for gene in child.genes:
            if _rng.random() < mutation_rate:
                gene.value = gene.param.random_value(_rng)
        return child

    @staticmethod
    def swap(chromosome: Chromosome, num_swaps: int = 1,
             rng: Optional[random.Random] = None) -> Chromosome:
        """
        Swap mutation: swap the values of two genes of compatible type.
        Useful for permutation-like representations (e.g. portfolio weights).
        """
        _rng = rng or random.Random()
        child = chromosome.clone()
        continuous_indices = [
            i for i, g in enumerate(child.genes)
            if g.param.param_type in (ParamType.CONTINUOUS, ParamType.LOG_SCALE)
        ]
        for _ in range(num_swaps):
            if len(continuous_indices) >= 2:
                i, j = _rng.sample(continuous_indices, 2)
                child.genes[i].value, child.genes[j].value = (
                    child.genes[j].value, child.genes[i].value
                )
                # Clip after swap in case ranges differ
                child.genes[i].value = child.genes[i].param.clip(child.genes[i].value)
                child.genes[j].value = child.genes[j].param.clip(child.genes[j].value)
        return child

    @staticmethod
    def boundary(chromosome: Chromosome, mutation_rate: float,
                 rng: Optional[random.Random] = None) -> Chromosome:
        """
        Boundary mutation: set gene to its lower or upper bound.
        Useful for discovering boundary optima.
        """
        _rng = rng or random.Random()
        child = chromosome.clone()
        for gene in child.genes:
            if _rng.random() < mutation_rate:
                if gene.param.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
                    gene.value = _rng.choice([gene.param.choices[0], gene.param.choices[-1]])
                else:
                    gene.value = _rng.choice([gene.param.low, gene.param.high])
                    if gene.param.param_type == ParamType.INTEGER:
                        gene.value = int(gene.value)
        return child

    @staticmethod
    def polynomial(chromosome: Chromosome, mutation_rate: float,
                   eta: float = 20.0, rng: Optional[random.Random] = None) -> Chromosome:
        """
        Polynomial mutation (NSGA-II style): generates perturbation with
        distribution shaped by distribution index eta.
        Higher eta = smaller perturbations.
        """
        _rng = rng or random.Random()
        child = chromosome.clone()
        for gene in child.genes:
            if _rng.random() < mutation_rate:
                if gene.param.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
                    gene.value = gene.param.random_value(_rng)
                    continue
                # Work in encoded [0,1] space
                x = gene.as_float()
                u = _rng.random()
                if u <= 0.5:
                    delta_q = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta_q = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                x_new = max(0.0, min(1.0, x + delta_q))
                gene.value = gene.param.decode_float(x_new)
        return child

    @staticmethod
    def creep(chromosome: Chromosome, mutation_rate: float,
              step_fraction: float = 0.05, rng: Optional[random.Random] = None) -> Chromosome:
        """
        Creep mutation: add a small step (uniform random in [-step, +step]).
        """
        _rng = rng or random.Random()
        child = chromosome.clone()
        for gene in child.genes:
            if _rng.random() < mutation_rate:
                if gene.param.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
                    if _rng.random() < 0.1:
                        gene.value = gene.param.random_value(_rng)
                    continue
                x = gene.as_float()
                step = step_fraction * _rng.uniform(-1.0, 1.0)
                x_new = max(0.0, min(1.0, x + step))
                gene.value = gene.param.decode_float(x_new)
        return child

    @staticmethod
    def adaptive_gaussian(chromosome: Chromosome, mutation_rate: float,
                          strategy_params: Optional[List[float]] = None,
                          rng: Optional[random.Random] = None) -> Tuple["Chromosome", List[float]]:
        """
        Self-adaptive Gaussian mutation (Evolution Strategy style).
        Each parameter has its own step size sigma which is also evolved.
        Returns mutated chromosome and updated strategy parameters (sigmas).
        """
        _rng = rng or random.Random()
        n = len(chromosome.genes)
        if strategy_params is None:
            strategy_params = [0.1] * n

        # Update strategy parameters (log-normal update)
        tau = 1.0 / math.sqrt(2.0 * n)
        tau_prime = 1.0 / math.sqrt(2.0 * math.sqrt(n))
        global_factor = math.exp(tau_prime * _rng.gauss(0.0, 1.0))

        new_sigmas = []
        for sigma in strategy_params:
            new_sigma = sigma * global_factor * math.exp(tau * _rng.gauss(0.0, 1.0))
            new_sigma = max(1e-6, min(1.0, new_sigma))
            new_sigmas.append(new_sigma)

        child = chromosome.clone()
        for i, gene in enumerate(child.genes):
            if _rng.random() < mutation_rate:
                if gene.param.param_type in (ParamType.CATEGORICAL, ParamType.BOOLEAN):
                    gene.value = gene.param.random_value(_rng)
                else:
                    x = gene.as_float()
                    x_new = max(0.0, min(1.0, x + new_sigmas[i] * _rng.gauss(0.0, 1.0)))
                    gene.value = gene.param.decode_float(x_new)

        return child, new_sigmas


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------

class CrossoverOperator:
    """Collection of crossover operators for producing offspring chromosomes."""

    @staticmethod
    def uniform(parent1: Chromosome, parent2: Chromosome,
                swap_prob: float = 0.5,
                rng: Optional[random.Random] = None) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform crossover: each gene is inherited from either parent with probability swap_prob.
        """
        _rng = rng or random.Random()
        c1_genes = []
        c2_genes = []
        for g1, g2 in zip(parent1.genes, parent2.genes):
            if _rng.random() < swap_prob:
                c1_genes.append(Gene(g1.param, copy.deepcopy(g2.value)))
                c2_genes.append(Gene(g2.param, copy.deepcopy(g1.value)))
            else:
                c1_genes.append(g1.clone())
                c2_genes.append(g2.clone())
        return (
            Chromosome(c1_genes, rng=random.Random(_rng.random())),
            Chromosome(c2_genes, rng=random.Random(_rng.random())),
        )

    @staticmethod
    def one_point(parent1: Chromosome, parent2: Chromosome,
                  rng: Optional[random.Random] = None) -> Tuple[Chromosome, Chromosome]:
        """
        One-point crossover: split at a random point and swap tails.
        """
        _rng = rng or random.Random()
        n = len(parent1.genes)
        point = _rng.randint(1, max(1, n - 1))
        c1_genes = [g.clone() for g in parent1.genes[:point]] + \
                   [g.clone() for g in parent2.genes[point:]]
        c2_genes = [g.clone() for g in parent2.genes[:point]] + \
                   [g.clone() for g in parent1.genes[point:]]
        return (
            Chromosome(c1_genes, rng=random.Random(_rng.random())),
            Chromosome(c2_genes, rng=random.Random(_rng.random())),
        )

    @staticmethod
    def two_point(parent1: Chromosome, parent2: Chromosome,
                  rng: Optional[random.Random] = None) -> Tuple[Chromosome, Chromosome]:
        """
        Two-point crossover: swap the segment between two randomly chosen points.
        """
        _rng = rng or random.Random()
        n = len(parent1.genes)
        pts = sorted(_rng.sample(range(1, max(2, n)), min(2, n - 1)))
        if len(pts) == 1:
            pts = [pts[0], pts[0]]
        pt1, pt2 = pts[0], pts[1]
        c1_genes = (
            [g.clone() for g in parent1.genes[:pt1]] +
            [g.clone() for g in parent2.genes[pt1:pt2]] +
            [g.clone() for g in parent1.genes[pt2:]]
        )
        c2_genes = (
            [g.clone() for g in parent2.genes[:pt1]] +
            [g.clone() for g in parent1.genes[pt1:pt2]] +
            [g.clone() for g in parent2.genes[pt2:]]
        )
        return (
            Chromosome(c1_genes, rng=random.Random(_rng.random())),
            Chromosome(c2_genes, rng=random.Random(_rng.random())),
        )

    @staticmethod
    def arithmetic(parent1: Chromosome, parent2: Chromosome,
                   alpha: Optional[float] = None,
                   rng: Optional[random.Random] = None) -> Tuple[Chromosome, Chromosome]:
        """
        Arithmetic (blend) crossover: child values are linear combinations of parents.
        If alpha is None, it's sampled uniformly from [0, 1] for each pair.
        Works in encoded float space so it handles all param types consistently.
        """
        _rng = rng or random.Random()
        v1 = parent1.to_float_vector()
        v2 = parent2.to_float_vector()
        params = [g.param for g in parent1.genes]

        if alpha is not None:
            a = alpha
            c1_vec = [a * x + (1 - a) * y for x, y in zip(v1, v2)]
            c2_vec = [(1 - a) * x + a * y for x, y in zip(v1, v2)]
        else:
            c1_vec = []
            c2_vec = []
            for x, y in zip(v1, v2):
                a = _rng.random()
                c1_vec.append(a * x + (1 - a) * y)
                c2_vec.append((1 - a) * x + a * y)

        return (
            Chromosome.from_float_vector(params, c1_vec, rng=random.Random(_rng.random())),
            Chromosome.from_float_vector(params, c2_vec, rng=random.Random(_rng.random())),
        )

    @staticmethod
    def simulated_binary(parent1: Chromosome, parent2: Chromosome,
                         eta: float = 2.0,
                         rng: Optional[random.Random] = None) -> Tuple[Chromosome, Chromosome]:
        """
        Simulated Binary Crossover (SBX): mimics one-point crossover in binary encoding.
        eta controls spread: higher eta = offspring closer to parents.
        """
        _rng = rng or random.Random()
        v1 = parent1.to_float_vector()
        v2 = parent2.to_float_vector()
        params = [g.param for g in parent1.genes]

        c1_vec = []
        c2_vec = []
        for x1, x2 in zip(v1, v2):
            if abs(x1 - x2) < 1e-10:
                c1_vec.append(x1)
                c2_vec.append(x2)
                continue
            if x1 > x2:
                x1, x2 = x2, x1
            u = _rng.random()
            # Spread factor
            if u <= 0.5:
                beta = (2.0 * u) ** (1.0 / (eta + 1.0))
            else:
                beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
            c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
            c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
            c1_vec.append(max(0.0, min(1.0, c1)))
            c2_vec.append(max(0.0, min(1.0, c2)))

        return (
            Chromosome.from_float_vector(params, c1_vec, rng=random.Random(_rng.random())),
            Chromosome.from_float_vector(params, c2_vec, rng=random.Random(_rng.random())),
        )

    @staticmethod
    def uniform_order(parent1: Chromosome, parent2: Chromosome,
                      rng: Optional[random.Random] = None) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform order crossover: preserves relative order for permutation-like chromosomes.
        Useful for portfolio weight orderings.
        """
        _rng = rng or random.Random()
        # Fall back to uniform for non-order chromosomes
        return CrossoverOperator.uniform(parent1, parent2, rng=_rng)


# ---------------------------------------------------------------------------
# Strategy Genome: top-level strategy representation
# ---------------------------------------------------------------------------

@dataclass
class GenomeMetadata:
    """Metadata attached to a genome for tracking lineage."""
    genome_id: str = field(default_factory=lambda: _random_id())
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    creation_method: str = "random"   # random / crossover / mutation / elite
    age: int = 0                       # generations survived
    n_evaluations: int = 0


def _random_id() -> str:
    return hashlib.md5(struct.pack(">d", random.random())).hexdigest()[:8]


class StrategyGenome:
    """
    Complete genome for a trading strategy, combining:
    - A chromosome of strategy parameters
    - Fitness scores (potentially multi-objective)
    - Metadata (lineage, age, etc.)
    - Optional self-adaptive mutation step sizes
    """

    def __init__(
        self,
        chromosome: Chromosome,
        strategy_type: str = "generic",
        metadata: Optional[GenomeMetadata] = None,
        strategy_params_sigma: Optional[List[float]] = None,
    ) -> None:
        self.chromosome = chromosome
        self.strategy_type = strategy_type
        self.metadata = metadata or GenomeMetadata()
        self.fitness: Optional[float] = None
        self.objectives: Optional[List[float]] = None   # multi-objective
        self.constraint_violations: float = 0.0
        self.rank: Optional[int] = None          # NSGA-II rank
        self.crowding_distance: float = 0.0      # NSGA-II crowding
        self.niche_count: float = 0.0            # fitness sharing count
        self.sigma: List[float] = strategy_params_sigma or \
            [0.1] * len(chromosome.genes)        # self-adaptive step sizes

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def __getitem__(self, key: Union[int, str]) -> Any:
        return self.chromosome[key]

    def get(self, name: str, default: Any = None) -> Any:
        return self.chromosome.get(name, default)

    def to_dict(self) -> Dict[str, Any]:
        d = self.chromosome.to_dict()
        d["_fitness"] = self.fitness
        d["_objectives"] = self.objectives
        d["_strategy_type"] = self.strategy_type
        d["_genome_id"] = self.metadata.genome_id
        d["_generation"] = self.metadata.generation
        return d

    def fingerprint(self) -> str:
        return self.chromosome.fingerprint()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def mutate(self, mutation_rate: float, method: str = "gaussian",
               sigma_scale: float = 0.1, eta: float = 20.0,
               rng: Optional[random.Random] = None) -> "StrategyGenome":
        """Apply a mutation operator and return a new child genome."""
        _rng = rng or random.Random()

        if method == "gaussian":
            new_chr = MutationOperator.gaussian(
                self.chromosome, mutation_rate, sigma_scale, _rng)
            new_sigma = self.sigma[:]
        elif method == "uniform":
            new_chr = MutationOperator.uniform(self.chromosome, mutation_rate, _rng)
            new_sigma = self.sigma[:]
        elif method == "swap":
            new_chr = MutationOperator.swap(self.chromosome, rng=_rng)
            new_sigma = self.sigma[:]
        elif method == "boundary":
            new_chr = MutationOperator.boundary(self.chromosome, mutation_rate, _rng)
            new_sigma = self.sigma[:]
        elif method == "polynomial":
            new_chr = MutationOperator.polynomial(self.chromosome, mutation_rate, eta, _rng)
            new_sigma = self.sigma[:]
        elif method == "creep":
            new_chr = MutationOperator.creep(self.chromosome, mutation_rate, rng=_rng)
            new_sigma = self.sigma[:]
        elif method == "adaptive":
            new_chr, new_sigma = MutationOperator.adaptive_gaussian(
                self.chromosome, mutation_rate, self.sigma, _rng)
        else:
            raise ValueError(f"Unknown mutation method: {method}")

        child_meta = GenomeMetadata(
            generation=self.metadata.generation,
            parent_ids=[self.metadata.genome_id],
            creation_method=f"mutation_{method}",
        )
        return StrategyGenome(
            chromosome=new_chr,
            strategy_type=self.strategy_type,
            metadata=child_meta,
            strategy_params_sigma=new_sigma,
        )

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    @staticmethod
    def crossover(parent1: "StrategyGenome", parent2: "StrategyGenome",
                  method: str = "uniform",
                  rng: Optional[random.Random] = None) -> Tuple["StrategyGenome", "StrategyGenome"]:
        """Apply a crossover operator and return two child genomes."""
        _rng = rng or random.Random()

        if method == "uniform":
            c1_chr, c2_chr = CrossoverOperator.uniform(
                parent1.chromosome, parent2.chromosome, rng=_rng)
        elif method == "one_point":
            c1_chr, c2_chr = CrossoverOperator.one_point(
                parent1.chromosome, parent2.chromosome, rng=_rng)
        elif method == "two_point":
            c1_chr, c2_chr = CrossoverOperator.two_point(
                parent1.chromosome, parent2.chromosome, rng=_rng)
        elif method == "arithmetic":
            c1_chr, c2_chr = CrossoverOperator.arithmetic(
                parent1.chromosome, parent2.chromosome, rng=_rng)
        elif method == "sbx":
            c1_chr, c2_chr = CrossoverOperator.simulated_binary(
                parent1.chromosome, parent2.chromosome, rng=_rng)
        else:
            raise ValueError(f"Unknown crossover method: {method}")

        # Inherit sigma as average of parents
        avg_sigma = [
            (s1 + s2) / 2.0
            for s1, s2 in zip(parent1.sigma, parent2.sigma)
        ]

        pid1 = parent1.metadata.genome_id
        pid2 = parent2.metadata.genome_id

        meta1 = GenomeMetadata(
            generation=max(parent1.metadata.generation, parent2.metadata.generation),
            parent_ids=[pid1, pid2],
            creation_method=f"crossover_{method}",
        )
        meta2 = GenomeMetadata(
            generation=max(parent1.metadata.generation, parent2.metadata.generation),
            parent_ids=[pid1, pid2],
            creation_method=f"crossover_{method}",
        )

        child1 = StrategyGenome(c1_chr, parent1.strategy_type, meta1, avg_sigma[:])
        child2 = StrategyGenome(c2_chr, parent1.strategy_type, meta2, avg_sigma[:])
        return child1, child2

    # ------------------------------------------------------------------
    # Comparison (for sorting by fitness)
    # ------------------------------------------------------------------

    def dominates(self, other: "StrategyGenome") -> bool:
        """True if self Pareto-dominates other (all objectives >= other, at least one >)."""
        if self.objectives is None or other.objectives is None:
            if self.fitness is None:
                return False
            if other.fitness is None:
                return True
            return self.fitness > other.fitness

        if len(self.objectives) != len(other.objectives):
            return False
        at_least_one_better = False
        for a, b in zip(self.objectives, other.objectives):
            if a < b:  # worse in some objective (assuming maximization)
                return False
            if a > b:
                at_least_one_better = True
        return at_least_one_better

    def __lt__(self, other: "StrategyGenome") -> bool:
        """Lower fitness = less fit (used in min-heaps etc.)."""
        f_self = self.fitness if self.fitness is not None else float("-inf")
        f_other = other.fitness if other.fitness is not None else float("-inf")
        return f_self < f_other

    def __repr__(self) -> str:
        f = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return (f"StrategyGenome(type={self.strategy_type}, id={self.metadata.genome_id}, "
                f"fitness={f}, gen={self.metadata.generation})")


# ---------------------------------------------------------------------------
# Genome factory
# ---------------------------------------------------------------------------

class GenomeFactory:
    """Creates and manages strategy genomes for a given parameter schema."""

    STRATEGY_REGISTRY: Dict[str, Callable[[], List[ParamRange]]] = {
        "momentum": momentum_strategy_params,
        "mean_reversion": mean_reversion_strategy_params,
        "pairs_trading": pairs_trading_params,
        "ml_hyperparameter": ml_hyperparameter_params,
        "portfolio_weights": portfolio_weights_params,
    }

    def __init__(self, strategy_type: str,
                 custom_params: Optional[List[ParamRange]] = None,
                 seed: Optional[int] = None) -> None:
        self.strategy_type = strategy_type
        if custom_params is not None:
            self.params = custom_params
        elif strategy_type in self.STRATEGY_REGISTRY:
            self.params = self.STRATEGY_REGISTRY[strategy_type]()
        else:
            raise ValueError(
                f"Unknown strategy type '{strategy_type}'. "
                f"Available: {list(self.STRATEGY_REGISTRY.keys())}"
            )
        self._rng = random.Random(seed)

    @property
    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def create_random(self) -> StrategyGenome:
        chr_ = Chromosome.random(self.params, rng=random.Random(self._rng.random()))
        return StrategyGenome(chr_, self.strategy_type)

    def create_from_defaults(self) -> StrategyGenome:
        chr_ = Chromosome.from_defaults(self.params, rng=random.Random(self._rng.random()))
        return StrategyGenome(chr_, self.strategy_type)

    def create_from_dict(self, values: Dict[str, Any]) -> StrategyGenome:
        chr_ = Chromosome.from_dict(self.params, values,
                                    rng=random.Random(self._rng.random()))
        return StrategyGenome(chr_, self.strategy_type)

    def create_population(self, size: int) -> List[StrategyGenome]:
        """Create an initial random population."""
        return [self.create_random() for _ in range(size)]

    def create_seeded_population(self, size: int,
                                 seeds: Optional[List[Dict[str, Any]]] = None) -> List[StrategyGenome]:
        """
        Create population with some seeded individuals and the rest random.
        seeds: list of dicts with parameter values for known-good solutions.
        """
        population = []
        if seeds:
            for seed_dict in seeds[:size]:
                population.append(self.create_from_dict(seed_dict))
        # Fill remainder with random individuals
        while len(population) < size:
            population.append(self.create_random())
        return population

    def serialize_genome(self, genome: StrategyGenome) -> Dict[str, Any]:
        """Serialize a genome to a JSON-compatible dict."""
        return {
            "strategy_type": genome.strategy_type,
            "chromosome": genome.chromosome.to_dict(),
            "fitness": genome.fitness,
            "objectives": genome.objectives,
            "metadata": {
                "genome_id": genome.metadata.genome_id,
                "generation": genome.metadata.generation,
                "parent_ids": genome.metadata.parent_ids,
                "creation_method": genome.metadata.creation_method,
                "age": genome.metadata.age,
                "n_evaluations": genome.metadata.n_evaluations,
            },
            "sigma": genome.sigma,
        }

    def deserialize_genome(self, data: Dict[str, Any]) -> StrategyGenome:
        """Reconstruct a genome from a serialized dict."""
        chr_ = Chromosome.from_dict(self.params, data["chromosome"])
        meta = GenomeMetadata(
            genome_id=data["metadata"]["genome_id"],
            generation=data["metadata"]["generation"],
            parent_ids=data["metadata"]["parent_ids"],
            creation_method=data["metadata"]["creation_method"],
            age=data["metadata"]["age"],
            n_evaluations=data["metadata"]["n_evaluations"],
        )
        genome = StrategyGenome(chr_, data["strategy_type"], meta, data.get("sigma"))
        genome.fitness = data.get("fitness")
        genome.objectives = data.get("objectives")
        return genome


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Genome self-test ===")

    factory = GenomeFactory("momentum", seed=42)
    pop = factory.create_population(20)
    print(f"Created {len(pop)} momentum genomes")
    print(f"Example genome: {pop[0]}")
    print(f"Parameters: {pop[0].to_dict()}")

    # Test mutation
    mutated = pop[0].mutate(mutation_rate=0.3, method="gaussian")
    print(f"\nGaussian mutated: {mutated}")
    diff = {k: (pop[0].chromosome[k], mutated.chromosome[k])
            for k in factory.param_names
            if pop[0].chromosome[k] != mutated.chromosome[k]}
    print(f"Changed params ({len(diff)}): {dict(list(diff.items())[:3])}")

    # Test all mutation methods
    for method in ["uniform", "swap", "boundary", "polynomial", "creep", "adaptive"]:
        m = pop[0].mutate(0.3, method=method)
        print(f"  {method} mutation OK -> {m.metadata.creation_method}")

    # Test crossover
    c1, c2 = StrategyGenome.crossover(pop[0], pop[1], method="sbx")
    print(f"\nSBX crossover children: {c1}, {c2}")

    for method in ["uniform", "one_point", "two_point", "arithmetic"]:
        x1, x2 = StrategyGenome.crossover(pop[0], pop[1], method=method)
        print(f"  {method} crossover OK")

    # Test serialization
    serialized = factory.serialize_genome(pop[0])
    restored = factory.deserialize_genome(serialized)
    assert restored.chromosome.to_dict() == pop[0].chromosome.to_dict()
    print("\nSerialization round-trip OK")

    # Test ML hyperparameter genome
    ml_factory = GenomeFactory("ml_hyperparameter", seed=7)
    ml_pop = ml_factory.create_population(5)
    print(f"\nML hyperparameter genome: {ml_pop[0].to_dict()}")

    # Test portfolio weights
    pw_factory = GenomeFactory("portfolio_weights", seed=99)
    pw_genome = pw_factory.create_random()
    print(f"\nPortfolio weights genome params count: {len(pw_genome.chromosome.genes)}")

    print("\nAll genome tests passed.")
