"""
Primordial Generator: auto-generate executable signal code from physics concepts.

Takes a physics concept description (from CrossDomainMapping or EHS templates)
and outputs syntactically valid, AST-checked Python signal functions that can
be directly injected into the backtest pipeline.

The generated code follows a strict template:
  - Input: numpy array of returns + optional auxiliary data
  - Output: numpy array of signal values (-1 to +1)
  - No side effects, no imports beyond numpy/math
  - AST-validated before execution
"""

from __future__ import annotations
import ast
import math
import hashlib
import time
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Generated Signal Template
# ---------------------------------------------------------------------------

SIGNAL_TEMPLATE = '''
def signal_{name}(returns, volume=None, spread=None, correlation_matrix=None,
                   regime_labels=None, **kwargs):
    """
    Auto-generated signal: {concept}
    Domain: {domain}
    Description: {description}
    Generated: {timestamp}

    Parameters
    ----------
    returns : np.ndarray, shape (T,)
    volume : np.ndarray, shape (T,), optional
    spread : np.ndarray, shape (T,), optional
    correlation_matrix : np.ndarray, shape (T, N, N), optional
    regime_labels : np.ndarray, shape (T,), optional

    Returns
    -------
    signal : np.ndarray, shape (T,), values in [-1, +1]
    """
    import numpy as np
    import math

    T = len(returns)
    signal = np.zeros(T)
    lookback = {lookback}

    for t in range(lookback, T):
        window = returns[t - lookback:t]
{computation_body}
        signal[t] = max(-1.0, min(1.0, raw_signal))

    return signal
'''


# ---------------------------------------------------------------------------
# Physics-to-Code Translation Rules
# ---------------------------------------------------------------------------

COMPUTATION_TEMPLATES = {
    "magnetization": textwrap.dedent("""
        # Ising Model: magnetization = mean(sign(returns))
        magnetization = np.mean(np.sign(window))
        susceptibility = np.var(np.sign(window))
        raw_signal = np.tanh(susceptibility * 5 - 2) * np.sign(magnetization)
    """),

    "condensation": textwrap.dedent("""
        # Bose-Einstein: eigenvalue concentration ratio
        if len(window) >= 5:
            centered = window - window.mean()
            var_total = np.var(window)
            # Approximate top eigenvalue contribution
            autocorr = np.corrcoef(window[1:], window[:-1])[0, 1] if len(window) > 2 else 0
            condensation = abs(autocorr)
            raw_signal = np.tanh((condensation - 0.5) * 4) * np.sign(window.mean())
        else:
            raw_signal = 0.0
    """),

    "hawking_reversal": textwrap.dedent("""
        # Hawking Radiation: reversal encoded in temperature profile during collapse
        vol = np.std(window)
        trend = np.polyfit(range(len(window)), window, 1)[0] if len(window) >= 3 else 0
        # Temperature = 1 / (8 * pi * mass); high temp = low mass = evaporating
        mass_proxy = abs(np.sum(window)) / max(vol, 1e-10)
        temperature = 1.0 / max(8 * 3.14159 * mass_proxy, 1e-6)
        # Reversal signal: high temp + declining trend = reversal
        if temperature > 0.1 and trend < -0.001:
            raw_signal = -np.sign(trend) * min(temperature * 3, 1.0)
        else:
            raw_signal = 0.0
    """),

    "casimir_force": textwrap.dedent("""
        # Casimir Effect: compressed spread -> mean reversion force ~ 1/d^4
        mean_r = window.mean()
        std_r = max(np.std(window), 1e-8)
        z_score = (window[-1] - mean_r) / std_r
        # Casimir force: inversely proportional to spread^4
        compression = max(1.0 / max(abs(z_score) ** 2, 0.01), 0)
        raw_signal = -np.sign(z_score) * np.tanh(compression * z_score)
    """),

    "kam_stability": textwrap.dedent("""
        # KAM Theorem: autocorrelation structure stability
        if len(window) >= 10:
            acf1 = np.corrcoef(window[1:], window[:-1])[0, 1] if len(window) > 2 else 0
            acf2 = np.corrcoef(window[2:], window[:-2])[0, 1] if len(window) > 3 else 0
            # Lyapunov-like: divergence of nearby trajectories
            lyapunov = abs(acf1 - acf2) * 10
            # Low stability = regime break imminent
            raw_signal = np.tanh(lyapunov - 1) * np.sign(window.mean())
        else:
            raw_signal = 0.0
    """),

    "turbulence_cascade": textwrap.dedent("""
        # Navier-Stokes: energy cascade across scales
        if len(window) >= 8:
            # Multi-scale variance
            scales = [2, 4, 8]
            energies = []
            for s in scales:
                if len(window) >= s:
                    sub = window[-s:]
                    energies.append(np.var(sub))
                else:
                    energies.append(0)
            if len(energies) >= 2 and energies[0] > 1e-10:
                # Kolmogorov slope: log(energy) vs log(scale)
                log_e = np.log(np.array(energies) + 1e-10)
                log_s = np.log(np.array(scales, dtype=float))
                slope = np.polyfit(log_s, log_e, 1)[0] if len(log_s) >= 2 else 0
                # Turbulent if slope steeper than -5/3
                turbulence = max(0, -slope - 1.67)
                raw_signal = np.tanh(turbulence * 3) * np.sign(window.mean())
            else:
                raw_signal = 0.0
        else:
            raw_signal = 0.0
    """),

    "rg_flow": textwrap.dedent("""
        # Renormalization Group: Hurst exponent at multiple scales
        hurst_estimates = []
        for scale in [10, 20]:
            if len(window) >= scale * 2:
                subseries = window[-scale * 2:]
                R = np.max(np.cumsum(subseries - subseries.mean())) - np.min(np.cumsum(subseries - subseries.mean()))
                S = max(np.std(subseries), 1e-10)
                hurst_estimates.append(math.log(max(R / S, 1e-10)) / math.log(scale))
        if hurst_estimates:
            avg_hurst = np.mean(hurst_estimates)
            # Deviation from 0.5 = exploitable inefficiency
            inefficiency = abs(avg_hurst - 0.5)
            direction = 1.0 if avg_hurst > 0.5 else -1.0  # trending vs mean-reverting
            raw_signal = direction * np.tanh(inefficiency * 5) * np.sign(window.mean())
        else:
            raw_signal = 0.0
    """),

    "generic_momentum": textwrap.dedent("""
        # Generic momentum signal
        raw_signal = np.tanh(window.mean() / max(np.std(window), 1e-8) * 2)
    """),

    "generic_mean_reversion": textwrap.dedent("""
        # Generic mean reversion signal
        z = (window[-1] - window.mean()) / max(np.std(window), 1e-8)
        raw_signal = -np.tanh(z / 2)
    """),
}


# ---------------------------------------------------------------------------
# Code Generator
# ---------------------------------------------------------------------------

@dataclass
class GeneratedSignal:
    """A generated, validated, executable signal."""
    signal_id: str
    name: str
    physics_concept: str
    domain: str
    source_code: str
    function: Optional[Callable] = None
    is_valid: bool = False
    validation_error: str = ""
    backtest_sharpe: float = 0.0
    generated_at: float = 0.0


class PrimitiveCodeGenerator:
    """
    Auto-generate executable Python signal functions from physics concepts.
    Each generated function is AST-validated and sandboxed before use.
    """

    def __init__(self):
        self._counter = 0
        self._generated: Dict[str, GeneratedSignal] = {}

    def _next_id(self) -> str:
        self._counter += 1
        return f"gen_{self._counter:04d}"

    def generate(
        self,
        concept_name: str,
        domain: str,
        description: str,
        computation_key: str,
        lookback: int = 21,
    ) -> GeneratedSignal:
        """
        Generate executable signal code from a physics concept.

        computation_key: key into COMPUTATION_TEMPLATES (e.g., "magnetization")
        """
        signal_id = self._next_id()
        func_name = concept_name.lower().replace(" ", "_").replace("-", "_")[:30]

        # Get computation body
        body = COMPUTATION_TEMPLATES.get(computation_key, COMPUTATION_TEMPLATES["generic_momentum"])
        # Indent for template insertion
        indented_body = "\n".join("        " + line for line in body.strip().split("\n"))

        source = SIGNAL_TEMPLATE.format(
            name=func_name,
            concept=concept_name,
            domain=domain,
            description=description[:200],
            timestamp=time.strftime("%Y-%m-%d %H:%M"),
            lookback=lookback,
            computation_body=indented_body,
        )

        gen = GeneratedSignal(
            signal_id=signal_id,
            name=func_name,
            physics_concept=concept_name,
            domain=domain,
            source_code=source,
            generated_at=time.time(),
        )

        # Validate
        gen = self._validate(gen)

        if gen.is_valid:
            self._generated[signal_id] = gen

        return gen

    def _validate(self, gen: GeneratedSignal) -> GeneratedSignal:
        """AST-validate the generated code and compile it."""
        try:
            # Parse AST to check syntax
            tree = ast.parse(gen.source_code)

            # Check for dangerous nodes
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Only allow numpy and math imports
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in ("numpy", "math", "np"):
                                gen.validation_error = f"Forbidden import: {alias.name}"
                                return gen
                    elif isinstance(node, ast.ImportFrom):
                        if node.module not in ("numpy", "math"):
                            gen.validation_error = f"Forbidden import from: {node.module}"
                            return gen

                # No exec, eval, open, os, sys
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ("exec", "eval", "open", "__import__", "compile"):
                            gen.validation_error = f"Forbidden call: {node.func.id}"
                            return gen

            # Compile and extract function
            code_obj = compile(gen.source_code, f"<generated:{gen.signal_id}>", "exec")
            namespace = {"np": np, "numpy": np, "math": math}
            exec(code_obj, namespace)

            # Find the signal function
            func_name = f"signal_{gen.name}"
            if func_name in namespace:
                gen.function = namespace[func_name]
                gen.is_valid = True
            else:
                gen.validation_error = f"Function {func_name} not found in generated code"

        except SyntaxError as e:
            gen.validation_error = f"Syntax error: {e}"
        except Exception as e:
            gen.validation_error = f"Validation error: {e}"

        return gen

    def backtest_signal(
        self,
        gen: GeneratedSignal,
        returns: np.ndarray,
        transaction_cost: float = 0.001,
    ) -> float:
        """Quick backtest of a generated signal."""
        if not gen.is_valid or gen.function is None:
            return 0.0

        try:
            signal = gen.function(returns)
            # Strategy returns
            strat_returns = signal[:-1] * returns[1:]
            costs = np.abs(np.diff(signal, prepend=0))[:-1] * transaction_cost
            net = strat_returns - costs

            if len(net) > 20 and net.std() > 1e-10:
                sharpe = float(net.mean() / net.std() * math.sqrt(252))
            else:
                sharpe = 0.0

            gen.backtest_sharpe = sharpe
            return sharpe

        except Exception:
            return 0.0

    def generate_all_templates(self, returns: np.ndarray) -> List[GeneratedSignal]:
        """Generate and backtest signals from all computation templates."""
        results = []
        templates = [
            ("Ising Magnetization", "statistical_mechanics", "Phase transition via spin alignment", "magnetization", 21),
            ("Bose-Einstein Condensation", "quantum_mechanics", "Single-factor dominance detection", "condensation", 42),
            ("Hawking Reversal", "quantum_gravity", "Reversal encoded in temperature profile", "hawking_reversal", 21),
            ("Casimir Force", "QFT", "Compressed spread mean reversion", "casimir_force", 21),
            ("KAM Stability", "dynamical_systems", "Orbital stability regime detection", "kam_stability", 42),
            ("Turbulence Cascade", "fluid_dynamics", "Kolmogorov energy cascade", "turbulence_cascade", 21),
            ("RG Flow", "statistical_mechanics", "Hurst convergence to efficiency", "rg_flow", 42),
        ]

        for concept, domain, desc, key, lookback in templates:
            gen = self.generate(concept, domain, desc, key, lookback)
            if gen.is_valid:
                sharpe = self.backtest_signal(gen, returns)
                results.append(gen)

        results.sort(key=lambda g: g.backtest_sharpe, reverse=True)
        return results

    def get_best(self, n: int = 5) -> List[GeneratedSignal]:
        """Get the best performing generated signals."""
        all_gen = sorted(self._generated.values(), key=lambda g: g.backtest_sharpe, reverse=True)
        return all_gen[:n]
