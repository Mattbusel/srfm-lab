"""
category_theory_finance.py -- Category-theoretic structures in quantitative finance.

Functors, monads, natural transformations, sheaves, cohomology, fiber bundles,
gauge theory, and topos-theoretic constructs applied to pricing, arbitrage
detection, risk composition, and market data consistency.

All numerics via numpy/scipy.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, stats
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]
T = TypeVar("T")
S = TypeVar("S")
A = TypeVar("A")
B = TypeVar("B")


# ===================================================================
# 1.  Category fundamentals
# ===================================================================

@dataclass
class MarketObject:
    """Object in a market category: represents an asset class or space."""
    name: str
    dimension: int
    data: FloatArray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Morphism:
    """Morphism (arrow) between market objects."""
    source: MarketObject
    target: MarketObject
    transform: Callable[[FloatArray], FloatArray]
    name: str = ""

    def apply(self, x: FloatArray) -> FloatArray:
        return self.transform(x)

    def compose(self, other: "Morphism") -> "Morphism":
        """Compose self after other: self . other."""
        assert self.source.name == other.target.name
        return Morphism(
            source=other.source,
            target=self.target,
            transform=lambda x: self.transform(other.transform(x)),
            name=f"{self.name} . {other.name}",
        )


class Category:
    """A category with objects and morphisms."""

    def __init__(self, name: str = ""):
        self.name = name
        self.objects: Dict[str, MarketObject] = {}
        self.morphisms: List[Morphism] = []

    def add_object(self, obj: MarketObject) -> None:
        self.objects[obj.name] = obj

    def add_morphism(self, m: Morphism) -> None:
        self.morphisms.append(m)

    def identity(self, obj_name: str) -> Morphism:
        obj = self.objects[obj_name]
        return Morphism(source=obj, target=obj, transform=lambda x: x, name=f"id_{obj_name}")

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        return f.compose(g)

    def find_morphism(self, source: str, target: str) -> Optional[Morphism]:
        for m in self.morphisms:
            if m.source.name == source and m.target.name == target:
                return m
        return None


# ===================================================================
# 2.  Functor: equity -> credit via Merton model
# ===================================================================

class Functor:
    """Functor between two categories."""

    def __init__(
        self,
        source_cat: Category,
        target_cat: Category,
        object_map: Callable[[MarketObject], MarketObject],
        morphism_map: Callable[[Morphism], Morphism],
        name: str = "",
    ):
        self.source = source_cat
        self.target = target_cat
        self.object_map = object_map
        self.morphism_map = morphism_map
        self.name = name

    def apply_object(self, obj: MarketObject) -> MarketObject:
        return self.object_map(obj)

    def apply_morphism(self, m: Morphism) -> Morphism:
        return self.morphism_map(m)


def merton_equity_to_credit_functor(
    equity_cat: Category, credit_cat: Category,
    debt_level: float = 80.0,
    risk_free_rate: float = 0.02,
    maturity: float = 1.0,
) -> Functor:
    """Merton structural model: maps equity prices to credit spreads.
    Equity = Call(V, D), so V = Equity + D*exp(-r*T), credit spread derived."""

    def object_map(obj: MarketObject) -> MarketObject:
        return MarketObject(
            name=f"credit_{obj.name}",
            dimension=obj.dimension,
            metadata={"original": obj.name, "type": "credit_spread"},
        )

    def morphism_map(m: Morphism) -> Morphism:
        def credit_transform(equity_prices: FloatArray) -> FloatArray:
            # Merton: V = E + D*exp(-r*T), sigma_V from equity vol
            E = equity_prices
            D = debt_level
            V = E + D * np.exp(-risk_free_rate * maturity)
            sigma_E = np.std(np.diff(np.log(E + 1e-12))) * np.sqrt(252)
            sigma_V = sigma_E * E / (V + 1e-12)
            # Distance to default
            d2 = (np.log(V / D) + (risk_free_rate - 0.5 * sigma_V ** 2) * maturity) / (
                sigma_V * np.sqrt(maturity) + 1e-12
            )
            # Default probability
            pd = stats.norm.cdf(-d2)
            # Credit spread approximation
            spread = -np.log(1 - pd * 0.6 + 1e-12) / maturity
            return spread

        return Morphism(
            source=object_map(m.source),
            target=object_map(m.target),
            transform=credit_transform,
            name=f"merton({m.name})",
        )

    return Functor(equity_cat, credit_cat, object_map, morphism_map, "Merton")


def covariance_functor(
    returns_cat: Category, cov_cat: Category, lookback: int = 63
) -> Functor:
    """Functor mapping return series to covariance matrices."""

    def object_map(obj: MarketObject) -> MarketObject:
        return MarketObject(
            name=f"cov_{obj.name}",
            dimension=obj.dimension ** 2,
            metadata={"type": "covariance"},
        )

    def morphism_map(m: Morphism) -> Morphism:
        def cov_transform(returns: FloatArray) -> FloatArray:
            if returns.ndim == 1:
                return np.array([[returns[-lookback:].var()]])
            return np.cov(returns[-lookback:].T)

        return Morphism(
            source=object_map(m.source),
            target=object_map(m.target),
            transform=cov_transform,
            name=f"cov({m.name})",
        )

    return Functor(returns_cat, cov_cat, object_map, morphism_map, "Covariance")


# ===================================================================
# 3.  Monad: risk transformation composition
# ===================================================================

@dataclass
class RiskMonad:
    """Monad for composing risk transformations.
    unit: portfolio -> risk-adjusted portfolio
    bind: chain risk transformations (vol-target >> dd-control >> leverage-cap)."""

    def unit(self, portfolio: FloatArray) -> Tuple[FloatArray, Dict[str, float]]:
        """Lift a portfolio into the risk monad (identity transformation)."""
        return portfolio, {"leverage": float(np.abs(portfolio).sum())}

    def bind(
        self,
        m: Tuple[FloatArray, Dict[str, float]],
        f: Callable[[FloatArray], Tuple[FloatArray, Dict[str, float]]],
    ) -> Tuple[FloatArray, Dict[str, float]]:
        """Monadic bind: apply transformation f to portfolio in monad."""
        portfolio, meta = m
        new_portfolio, new_meta = f(portfolio)
        combined_meta = {**meta, **new_meta}
        return new_portfolio, combined_meta

    def compose(
        self,
        *transforms: Callable[[FloatArray], Tuple[FloatArray, Dict[str, float]]],
    ) -> Callable[[FloatArray], Tuple[FloatArray, Dict[str, float]]]:
        """Compose multiple risk transformations via monadic bind."""
        def composed(portfolio: FloatArray) -> Tuple[FloatArray, Dict[str, float]]:
            m = self.unit(portfolio)
            for t in transforms:
                m = self.bind(m, t)
            return m
        return composed


def vol_target_transform(
    target_vol: float, cov_matrix: FloatArray
) -> Callable[[FloatArray], Tuple[FloatArray, Dict[str, float]]]:
    """Risk transformation: scale portfolio to target volatility."""
    def transform(w: FloatArray) -> Tuple[FloatArray, Dict[str, float]]:
        port_var = w @ cov_matrix @ w
        port_vol = np.sqrt(max(port_var, 1e-12)) * np.sqrt(252)
        scale = target_vol / (port_vol + 1e-12)
        w_new = w * min(scale, 3.0)
        return w_new, {"vol_scale": float(scale), "target_vol": target_vol}
    return transform


def leverage_cap_transform(
    max_leverage: float,
) -> Callable[[FloatArray], Tuple[FloatArray, Dict[str, float]]]:
    """Risk transformation: cap gross leverage."""
    def transform(w: FloatArray) -> Tuple[FloatArray, Dict[str, float]]:
        gross = np.abs(w).sum()
        if gross > max_leverage:
            w = w * max_leverage / gross
        return w, {"leverage_cap_applied": float(gross > max_leverage)}
    return transform


def drawdown_control_transform(
    current_drawdown: float, threshold: float = 0.10
) -> Callable[[FloatArray], Tuple[FloatArray, Dict[str, float]]]:
    """Risk transformation: reduce exposure based on drawdown."""
    def transform(w: FloatArray) -> Tuple[FloatArray, Dict[str, float]]:
        if current_drawdown < -threshold:
            scale = max(1.0 + current_drawdown / threshold, 0.1)
            w = w * scale
            return w, {"dd_scale": float(scale), "drawdown": current_drawdown}
        return w, {"dd_scale": 1.0, "drawdown": current_drawdown}
    return transform


# ===================================================================
# 4.  Natural Transformation
# ===================================================================

class NaturalTransformation:
    """Natural transformation between two functors F, G: C -> D.
    For each object X in C, provides a morphism eta_X: F(X) -> G(X)."""

    def __init__(
        self,
        functor_f: Functor,
        functor_g: Functor,
        component_map: Dict[str, Morphism],
        name: str = "",
    ):
        self.F = functor_f
        self.G = functor_g
        self.components = component_map
        self.name = name

    def component_at(self, obj_name: str) -> Morphism:
        return self.components[obj_name]

    def is_natural(self, test_morphisms: List[Morphism]) -> bool:
        """Check naturality: G(f) . eta_X = eta_Y . F(f) for morphism f: X->Y."""
        for f in test_morphisms:
            src = f.source.name
            tgt = f.target.name
            if src in self.components and tgt in self.components:
                # Check commutative diagram (approximately)
                test_data = np.random.default_rng(0).standard_normal(f.source.dimension)
                path1 = self.G.apply_morphism(f).apply(self.components[src].apply(test_data))
                path2 = self.components[tgt].apply(self.F.apply_morphism(f).apply(test_data))
                if not np.allclose(path1, path2, atol=1e-6):
                    return False
        return True


def pricing_natural_transformation(
    bs_functor: Functor,
    mc_functor: Functor,
    objects: List[str],
) -> NaturalTransformation:
    """Natural transformation between Black-Scholes and Monte Carlo pricing.
    Each component is a correction morphism."""
    components = {}
    for obj_name in objects:
        # Correction: MC_price / BS_price ratio
        def correction(x: FloatArray) -> FloatArray:
            # In practice this would use model-specific corrections
            return x * 1.0  # identity for now
        obj = MarketObject(name=obj_name, dimension=1)
        components[obj_name] = Morphism(
            source=obj, target=obj, transform=correction,
            name=f"bs_to_mc_{obj_name}",
        )
    return NaturalTransformation(bs_functor, mc_functor, components, "BS_to_MC")


# ===================================================================
# 5.  Sheaf: local-to-global market data consistency
# ===================================================================

@dataclass
class OpenSet:
    """An open set in market space (e.g., a time window or asset subset)."""
    name: str
    indices: List[int]
    parent: str = ""


@dataclass
class SheafSection:
    """A section: assignment of data to an open set."""
    open_set: OpenSet
    data: FloatArray


class MarketSheaf:
    """Sheaf of market data: checks local-to-global consistency.

    The sheaf condition: if local sections agree on overlaps,
    there exists a unique global section.  Failure = arbitrage opportunity."""

    def __init__(self):
        self.open_sets: Dict[str, OpenSet] = {}
        self.sections: Dict[str, SheafSection] = {}
        self.restriction_maps: Dict[Tuple[str, str], Callable[[FloatArray], FloatArray]] = {}

    def add_open_set(self, os: OpenSet) -> None:
        self.open_sets[os.name] = os

    def add_section(self, section: SheafSection) -> None:
        self.sections[section.open_set.name] = section

    def add_restriction(
        self, from_set: str, to_set: str,
        restriction: Callable[[FloatArray], FloatArray],
    ) -> None:
        self.restriction_maps[(from_set, to_set)] = restriction

    def check_gluing_condition(self, tol: float = 1e-6) -> Dict[str, float]:
        """Check if local sections are compatible on overlaps.
        Returns inconsistency scores per overlap."""
        inconsistencies = {}
        names = list(self.open_sets.keys())
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i >= j:
                    continue
                # Check if restriction maps exist both ways
                key_ij = (name_i, name_j)
                key_ji = (name_j, name_i)
                if key_ij in self.restriction_maps and name_i in self.sections:
                    restricted = self.restriction_maps[key_ij](self.sections[name_i].data)
                    if name_j in self.sections:
                        overlap_j = self.sections[name_j].data
                        # Compare on overlap
                        min_len = min(len(restricted), len(overlap_j))
                        if min_len > 0:
                            diff = np.linalg.norm(restricted[:min_len] - overlap_j[:min_len])
                            inconsistencies[f"{name_i}->{name_j}"] = float(diff)
        return inconsistencies

    def detect_arbitrage(self, threshold: float = 0.01) -> List[str]:
        """Sheaf condition violation implies potential arbitrage."""
        incon = self.check_gluing_condition()
        violations = [k for k, v in incon.items() if v > threshold]
        return violations

    def global_section(self) -> FloatArray | None:
        """Attempt to construct a global section from local data.
        Uses least-squares if sections are inconsistent."""
        all_data = []
        for name, section in self.sections.items():
            all_data.append(section.data)
        if not all_data:
            return None
        # Simple: average over sections (proper sheaf would use limits)
        max_len = max(len(d) for d in all_data)
        padded = [np.pad(d, (0, max_len - len(d)), constant_values=np.nan) for d in all_data]
        stacked = np.array(padded)
        return np.nanmean(stacked, axis=0)


# ===================================================================
# 6.  Cohomology: arbitrage detection
# ===================================================================

def build_price_1_form(
    prices: FloatArray, pairs: List[Tuple[int, int]]
) -> FloatArray:
    """Build a 1-form on the asset graph from log price ratios.
    omega_{ij} = log(p_i / p_j).  If d(omega) = 0, no arbitrage."""
    n_pairs = len(pairs)
    omega = np.zeros(n_pairs)
    for k, (i, j) in enumerate(pairs):
        omega[k] = np.log(prices[i] / (prices[j] + 1e-12))
    return omega


def check_cocycle_condition(
    omega: FloatArray,
    pairs: List[Tuple[int, int]],
    triangles: List[Tuple[int, int, int]],
) -> FloatArray:
    """Check the cocycle condition: omega_{ij} + omega_{jk} + omega_{ki} = 0
    for each triangle (i,j,k).  Non-zero = arbitrage."""
    pair_dict = {}
    for idx, (i, j) in enumerate(pairs):
        pair_dict[(i, j)] = omega[idx]
        pair_dict[(j, i)] = -omega[idx]
    residuals = []
    for (i, j, k) in triangles:
        cycle = pair_dict.get((i, j), 0) + pair_dict.get((j, k), 0) + pair_dict.get((k, i), 0)
        residuals.append(cycle)
    return np.array(residuals)


def de_rham_cohomology_dimension(
    n_vertices: int, n_edges: int, n_triangles: int
) -> Tuple[int, int, int]:
    """Betti numbers for the price graph complex.
    b0 = components, b1 = independent cycles (= potential arbitrage dimension)."""
    b0 = n_vertices - n_edges + n_triangles  # Euler characteristic relation
    b1 = n_edges - n_vertices + 1  # for connected graph
    b2 = n_triangles - n_edges + n_vertices - 1  # higher
    return max(b0, 0), max(b1, 0), max(b2, 0)


class ArbitrageCohomology:
    """Detect arbitrage via cohomological analysis of the price graph."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.pairs: List[Tuple[int, int]] = []
        self.triangles: List[Tuple[int, int, int]] = []
        self._build_graph()

    def _build_graph(self) -> None:
        n = self.n_assets
        for i in range(n):
            for j in range(i + 1, n):
                self.pairs.append((i, j))
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    self.triangles.append((i, j, k))

    def boundary_operator_1(self) -> FloatArray:
        """d_0: C^0 -> C^1, boundary operator mapping functions to 1-forms."""
        n = self.n_assets
        m = len(self.pairs)
        d0 = np.zeros((m, n))
        for idx, (i, j) in enumerate(self.pairs):
            d0[idx, j] = 1.0
            d0[idx, i] = -1.0
        return d0

    def boundary_operator_2(self) -> FloatArray:
        """d_1: C^1 -> C^2, maps 1-forms to 2-forms."""
        m = len(self.pairs)
        p = len(self.triangles)
        pair_idx = {pair: i for i, pair in enumerate(self.pairs)}
        d1 = np.zeros((p, m))
        for t_idx, (i, j, k) in enumerate(self.triangles):
            if (i, j) in pair_idx:
                d1[t_idx, pair_idx[(i, j)]] = 1.0
            if (j, k) in pair_idx:
                d1[t_idx, pair_idx[(j, k)]] = 1.0
            if (i, k) in pair_idx:
                d1[t_idx, pair_idx[(i, k)]] = -1.0
        return d1

    def detect(self, prices: FloatArray, threshold: float = 1e-4) -> Dict[str, Any]:
        """Detect arbitrage opportunities in current prices."""
        omega = build_price_1_form(prices, self.pairs)
        residuals = check_cocycle_condition(omega, self.pairs, self.triangles)
        arb_triangles = []
        for idx, r in enumerate(residuals):
            if abs(r) > threshold:
                arb_triangles.append({
                    "triangle": self.triangles[idx],
                    "residual": float(r),
                    "profit_bps": float(abs(r) * 10000),
                })
        d0 = self.boundary_operator_1()
        d1 = self.boundary_operator_2()
        # H^1 = ker(d1) / im(d0)
        rank_d0 = np.linalg.matrix_rank(d0)
        rank_d1 = np.linalg.matrix_rank(d1)
        dim_ker_d1 = len(self.pairs) - rank_d1
        h1_dim = dim_ker_d1 - rank_d0
        return {
            "arbitrage_triangles": arb_triangles,
            "n_arbitrage": len(arb_triangles),
            "max_residual": float(np.max(np.abs(residuals))) if len(residuals) > 0 else 0.0,
            "h1_dimension": max(h1_dim, 0),
            "betti_numbers": de_rham_cohomology_dimension(
                self.n_assets, len(self.pairs), len(self.triangles)
            ),
        }


# ===================================================================
# 7.  Fiber Bundle: portfolio as section
# ===================================================================

@dataclass
class FiberBundle:
    """Fiber bundle over asset space.
    Base space: set of assets (or time).
    Fiber: space of possible positions/weights for that asset.
    Section: a portfolio (assignment of weight to each asset)."""
    n_assets: int
    fiber_dimension: int = 1     # 1 for scalar weights

    def section(self, weights: FloatArray) -> FloatArray:
        """A section is just a weight vector -- a smooth assignment."""
        return weights

    def connection(self, weights: FloatArray, returns: FloatArray) -> FloatArray:
        """Connection: how the portfolio changes under parallel transport.
        In finance: rebalancing needed to maintain target weights."""
        # After returns, weights drift
        new_values = weights * (1 + returns)
        new_weights = new_values / (new_values.sum() + 1e-12)
        # Rebalancing needed = difference from target
        rebalance = weights - new_weights
        return rebalance

    def curvature(self, returns_matrix: FloatArray, weights: FloatArray) -> FloatArray:
        """Curvature of the connection: measures path-dependence of rebalancing.
        Non-zero curvature = rebalancing depends on path taken."""
        n = returns_matrix.shape[0]
        curvatures = np.zeros(n - 1)
        w = weights.copy()
        for t in range(n - 1):
            r = returns_matrix[t]
            rebal = self.connection(w, r)
            w_new = w + rebal
            w_new = np.maximum(w_new, 0)
            w_new /= w_new.sum() + 1e-12
            curvatures[t] = np.linalg.norm(rebal)
            w = w_new
        return curvatures

    def holonomy(
        self, returns_matrix: FloatArray, weights: FloatArray
    ) -> FloatArray:
        """Holonomy: drift in weights after a full cycle.
        Transport weights around a closed loop in return space."""
        w = weights.copy()
        for t in range(returns_matrix.shape[0]):
            new_vals = w * (1 + returns_matrix[t])
            w = new_vals / (new_vals.sum() + 1e-12)
        # Holonomy = final - initial
        return w - weights


# ===================================================================
# 8.  Gauge Theory: no-arbitrage as gauge invariance
# ===================================================================

class GaugeTheory:
    """No-arbitrage condition as gauge invariance.

    Price process S_t is a gauge field.  Change of numeraire is a gauge transformation.
    No-arbitrage <=> existence of a gauge-invariant measure (risk-neutral)."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets

    def gauge_transform(
        self, prices: FloatArray, numeraire_idx: int
    ) -> FloatArray:
        """Change of numeraire: express all prices in terms of asset `numeraire_idx`."""
        num = prices[:, numeraire_idx : numeraire_idx + 1]
        return prices / (num + 1e-12)

    def check_gauge_invariance(
        self, prices: FloatArray, numeraire_indices: List[int]
    ) -> Dict[str, float]:
        """Check if relative prices are consistent across numeraire choices.
        Gauge invariance: conclusions shouldn't depend on numeraire choice."""
        results = {}
        base = self.gauge_transform(prices, numeraire_indices[0])
        log_ret_base = np.diff(np.log(base + 1e-12), axis=0)
        for idx in numeraire_indices[1:]:
            transformed = self.gauge_transform(prices, idx)
            log_ret = np.diff(np.log(transformed + 1e-12), axis=0)
            # Compare correlations
            for j in range(self.n_assets):
                if j in numeraire_indices:
                    continue
                corr = np.corrcoef(log_ret_base[:, j], log_ret[:, j])[0, 1]
                results[f"asset{j}_num{numeraire_indices[0]}_vs_{idx}"] = float(corr)
        return results

    def curvature_2_form(self, prices: FloatArray) -> FloatArray:
        """Gauge field strength (curvature).  Non-zero = mispricing / arbitrage.
        F_{ij,t} = d_t log(S_i/S_j) - expected drift."""
        n_t, n_a = prices.shape
        log_ratios = np.zeros((n_t, n_a, n_a))
        for i in range(n_a):
            for j in range(n_a):
                log_ratios[:, i, j] = np.log(prices[:, i] / (prices[:, j] + 1e-12) + 1e-12)
        # Curvature: derivative of log ratio - should be ~0 for no-arb
        F = np.diff(log_ratios, axis=0)
        return F

    def find_risk_neutral_measure(
        self, returns: FloatArray, risk_free_rate: float = 0.0
    ) -> FloatArray | None:
        """Attempt to find risk-neutral probabilities (Girsanov).
        Returns probability vector or None if market is incomplete."""
        n_t, n_a = returns.shape
        # Under risk-neutral measure: E^Q[r_i] = rf for all i
        # Find Q closest to uniform that satisfies this
        def objective(log_q: FloatArray) -> float:
            q = np.exp(log_q - log_q.max())
            q /= q.sum()
            # Minimize KL from uniform
            uniform = np.ones(n_t) / n_t
            kl = np.sum(q * np.log(q / uniform + 1e-30))
            # Penalize deviation from risk-neutral condition
            for j in range(n_a):
                eq_ret = np.sum(q * returns[:, j])
                kl += 100 * (eq_ret - risk_free_rate / 252) ** 2
            return kl

        from scipy.optimize import minimize
        x0 = np.zeros(n_t)
        result = minimize(objective, x0, method="L-BFGS-B")
        q = np.exp(result.x - result.x.max())
        q /= q.sum()
        return q


# ===================================================================
# 9.  Topos: logical framework for incomplete markets
# ===================================================================

@dataclass
class Proposition:
    """A proposition in the internal logic of a topos.
    In finance: 'asset X will outperform' has a truth value in [0,1]."""
    name: str
    truth_value: float          # in [0, 1], generalized truth
    evidence: FloatArray | None = None

    def __and__(self, other: "Proposition") -> "Proposition":
        return Proposition(
            f"({self.name} AND {other.name})",
            min(self.truth_value, other.truth_value),
        )

    def __or__(self, other: "Proposition") -> "Proposition":
        return Proposition(
            f"({self.name} OR {other.name})",
            max(self.truth_value, other.truth_value),
        )

    def __invert__(self) -> "Proposition":
        return Proposition(f"NOT({self.name})", 1.0 - self.truth_value)

    def implies(self, other: "Proposition") -> "Proposition":
        # Heyting algebra: a => b = sup{c : a AND c <= b}
        if self.truth_value <= other.truth_value:
            return Proposition(f"({self.name} => {other.name})", 1.0)
        return Proposition(
            f"({self.name} => {other.name})",
            other.truth_value,
        )


class MarketTopos:
    """Topos-theoretic framework for reasoning about incomplete markets.

    In complete markets, every claim has a unique price (classical logic).
    In incomplete markets, pricing is interval-valued (intuitionistic logic)."""

    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.propositions: Dict[str, Proposition] = {}

    def add_proposition(self, prop: Proposition) -> None:
        self.propositions[prop.name] = prop

    def evaluate_from_data(
        self, name: str, signal: FloatArray, threshold: float = 0.0
    ) -> Proposition:
        """Create a proposition from empirical data.
        Truth value = fraction of evidence supporting it."""
        truth = float((signal > threshold).mean())
        prop = Proposition(name, truth, signal)
        self.add_proposition(prop)
        return prop

    def price_interval(
        self,
        payoff: FloatArray,
        returns: FloatArray,
    ) -> Tuple[float, float]:
        """Compute super/sub-hedging price bounds (no-arbitrage interval).
        In incomplete markets, the price is not unique."""
        n_t, n_a = returns.shape

        # Super-hedging price: min cost of a portfolio that dominates the payoff
        def super_hedge(delta: FloatArray) -> float:
            hedge_payoff = returns @ delta
            shortfall = payoff - hedge_payoff
            return float(shortfall.max())

        # Sub-hedging price: max value of a portfolio dominated by payoff
        def sub_hedge(delta: FloatArray) -> float:
            hedge_payoff = returns @ delta
            surplus = hedge_payoff - payoff
            return float(-surplus.max())

        from scipy.optimize import minimize
        res_super = minimize(
            lambda d: super_hedge(d),
            x0=np.zeros(n_a),
            method="Nelder-Mead",
        )
        res_sub = minimize(
            lambda d: -sub_hedge(d),
            x0=np.zeros(n_a),
            method="Nelder-Mead",
        )
        upper = float(res_super.fun)
        lower = float(-res_sub.fun)
        return (lower, upper)

    def subobject_classifier(self, signal: FloatArray) -> FloatArray:
        """Subobject classifier Omega: maps each state to its 'truth degree'.
        In a topos, Omega generalizes {True, False} to a Heyting algebra."""
        # Normalize signal to [0, 1]
        s_min = signal.min()
        s_max = signal.max()
        if s_max - s_min < 1e-12:
            return np.full_like(signal, 0.5)
        return (signal - s_min) / (s_max - s_min)


# ===================================================================
# 10. Applications
# ===================================================================

def sheaf_consistency_check(
    prices_by_exchange: Dict[str, FloatArray],
    asset_names: List[str],
) -> Dict[str, float]:
    """Check price consistency across exchanges using sheaf formalism."""
    sheaf = MarketSheaf()
    for exchange, prices in prices_by_exchange.items():
        os = OpenSet(name=exchange, indices=list(range(len(asset_names))))
        sheaf.add_open_set(os)
        sheaf.add_section(SheafSection(os, prices))
        # Restriction: identity (prices should be the same)
        for other_exchange in prices_by_exchange:
            if other_exchange != exchange:
                sheaf.add_restriction(exchange, other_exchange, lambda x: x)
    return sheaf.check_gluing_condition()


def compose_risk_pipeline(
    weights: FloatArray,
    cov_matrix: FloatArray,
    current_drawdown: float = 0.0,
    vol_target: float = 0.10,
    max_leverage: float = 2.0,
) -> Tuple[FloatArray, Dict[str, float]]:
    """Full risk pipeline using monadic composition."""
    monad = RiskMonad()
    pipeline = monad.compose(
        vol_target_transform(vol_target, cov_matrix),
        leverage_cap_transform(max_leverage),
        drawdown_control_transform(current_drawdown),
    )
    return pipeline(weights)


def detect_cross_exchange_arbitrage(
    prices_a: FloatArray,
    prices_b: FloatArray,
    threshold_bps: float = 5.0,
) -> List[Dict[str, Any]]:
    """Detect arbitrage between two exchanges using cohomological analysis."""
    n_assets = len(prices_a)
    combined = np.column_stack([prices_a, prices_b])
    cohom = ArbitrageCohomology(n_assets * 2)
    result = cohom.detect(combined, threshold=threshold_bps / 10000)
    return result["arbitrage_triangles"]


def portfolio_fiber_analysis(
    returns: FloatArray,
    target_weights: FloatArray,
) -> Dict[str, Any]:
    """Analyze portfolio rebalancing via fiber bundle formalism."""
    bundle = FiberBundle(n_assets=len(target_weights))
    curv = bundle.curvature(returns, target_weights)
    hol = bundle.holonomy(returns, target_weights)
    return {
        "mean_curvature": float(curv.mean()),
        "max_curvature": float(curv.max()),
        "holonomy_norm": float(np.linalg.norm(hol)),
        "holonomy": hol.tolist(),
        "rebalancing_needed": float((curv > 0.01).mean()),
    }


# ===================================================================
# __all__
# ===================================================================

__all__ = [
    "MarketObject",
    "Morphism",
    "Category",
    "Functor",
    "merton_equity_to_credit_functor",
    "covariance_functor",
    "RiskMonad",
    "vol_target_transform",
    "leverage_cap_transform",
    "drawdown_control_transform",
    "NaturalTransformation",
    "pricing_natural_transformation",
    "MarketSheaf",
    "OpenSet",
    "SheafSection",
    "ArbitrageCohomology",
    "build_price_1_form",
    "check_cocycle_condition",
    "de_rham_cohomology_dimension",
    "FiberBundle",
    "GaugeTheory",
    "Proposition",
    "MarketTopos",
    "sheaf_consistency_check",
    "compose_risk_pipeline",
    "detect_cross_exchange_arbitrage",
    "portfolio_fiber_analysis",
]
