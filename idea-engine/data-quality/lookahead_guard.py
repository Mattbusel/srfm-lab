"""
LookaheadGuard
==============
Detects lookahead bias in strategy code and signal series.

Three modes of operation:

  1. **Static code analysis** (``scan_strategy_code``) — regex / AST patterns
     that flag the most common lookahead mistakes in Python strategy code.

  2. **Runtime signal timing validation** (``validate_signal_timing``) — verifies
     that a signal series at bar T never correlates with future price returns in
     a way that implies future data was used.

  3. **Backtest parameter checks** (``validate_backtest_params``) — reviews a
     parameter dict for obvious look-forward risks.

  4. **Safe signal marking** (``mark_safe``) — re-aligns a signal series to
     guarantee it can only see data ≤ bar T.
"""

from __future__ import annotations

import ast
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LookaheadViolation:
    """Describes a single suspected lookahead-bias incident."""

    violation_type: str      # e.g. "negative_shift", "rolling_then_slice", etc.
    severity: str            # "critical" | "warning" | "info"
    line_number: int | None
    code_snippet: str
    description: str
    fix_suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.violation_type,
            "severity": self.severity,
            "line": self.line_number,
            "snippet": self.code_snippet,
            "description": self.description,
            "fix": self.fix_suggestion,
        }


@dataclass
class SignalTimingResult:
    """Result of validate_signal_timing."""

    is_clean: bool
    correlation_with_future: float
    p_value: float
    max_lag_tested: int
    offending_lags: list[int] = field(default_factory=list)
    description: str = ""


# ---------------------------------------------------------------------------
# Regex and AST patterns
# ---------------------------------------------------------------------------

# Pattern 1 — negative shift: df['x'] = df['y'].shift(-N) or .shift(−N)
_RE_NEG_SHIFT = re.compile(
    r"""\.shift\s*\(\s*-\s*\d+""",
    re.VERBOSE,
)

# Pattern 2 — iloc[-N] usage inside loops (poor man's match)
_RE_ILOC_NEG = re.compile(
    r"""\.iloc\s*\[\s*-\s*\d+""",
)

# Pattern 3 — future column name hints
_RE_FUTURE_COLUMN = re.compile(
    r"""['"]future[_\w]*['"]|future_\w+""",
    re.IGNORECASE,
)

# Pattern 4 — rolling().mean() / rolling().std() applied to full series then sliced
# e.g. df['sma'] = df['close'].rolling(20).mean()
# The violation happens when the result is used in a vectorised backtest without shift(1)
_RE_ROLLING_NO_SHIFT = re.compile(
    r"""\.rolling\s*\([^)]+\)\s*\.\s*(?:mean|std|sum|min|max|var|median)\s*\(\s*\)(?!\s*\.\s*shift)""",
)

# Pattern 5 — resample aggregation on full series (potential future leakage)
_RE_RESAMPLE_NOFUTURE = re.compile(
    r"""\.resample\s*\([^)]+\)\s*\.\s*(?:ohlc|mean|sum|last|first)\s*\(\s*\)(?!\s*\.\s*shift)""",
)

# Pattern 6 — df.tail() / df[-N:] in what looks like signal generation context
_RE_TAIL_SLICE = re.compile(
    r"""df(?:\[['"][^'"]*['"]\])?\s*\[\s*-\s*\d+\s*:\s*\]""",
)

# Pattern 7 — pd.concat with future-labeled columns
_RE_CONCAT_FUTURE = re.compile(
    r"""pd\.concat\s*\(.*future""",
    re.DOTALL | re.IGNORECASE,
)

# Pattern 8 — double-use of apply on entire frame (the apply may see future)
_RE_APPLY_FULL = re.compile(
    r"""\.apply\s*\(.*axis\s*=\s*1""",
    re.DOTALL,
)

# Patterns that together describe lookahead signatures
_ALL_PATTERNS: list[tuple[str, re.Pattern, str, str, str]] = [
    (
        "negative_shift",
        _RE_NEG_SHIFT,
        "critical",
        "Using .shift(-N) accesses future data.",
        "Replace .shift(-N) with .shift(N) and reverse signal logic if needed.",
    ),
    (
        "iloc_negative",
        _RE_ILOC_NEG,
        "warning",
        ".iloc[-N] inside a strategy loop may reference the last real bar (future).",
        "Use explicit integer index or rewrite the loop to step forward in time.",
    ),
    (
        "future_column_name",
        _RE_FUTURE_COLUMN,
        "warning",
        "A column named 'future_...' is referenced — may contain forward-looking data.",
        "Rename and verify the column does not carry forward price information.",
    ),
    (
        "rolling_no_shift",
        _RE_ROLLING_NO_SHIFT,
        "warning",
        "Rolling statistic used without .shift(1) — the bar's own data leaks in.",
        "Append .shift(1) after .rolling(...).mean() before using as a signal.",
    ),
    (
        "resample_no_shift",
        _RE_RESAMPLE_NOFUTURE,
        "warning",
        "Resample aggregation used without .shift(1) — the current bar is included.",
        "Append .shift(1) after the resample aggregation.",
    ),
    (
        "tail_slice",
        _RE_TAIL_SLICE,
        "info",
        "Negative index slice df[-N:] — verify this is not inside a signal loop.",
        "Use explicit time-bounded slicing instead of negative index slices.",
    ),
]


# ---------------------------------------------------------------------------
# AST visitor helpers
# ---------------------------------------------------------------------------

class _LookaheadVisitor(ast.NodeVisitor):
    """
    AST-level visitor that catches more subtle patterns:
    - Assignments where the RHS references a shifted-backward column
    - ExpandingMean / cumulative stats without shift
    """

    def __init__(self) -> None:
        self.violations: list[tuple[int, str, str]] = []  # (lineno, type, snippet)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # Look for:   something = df[...].shift(-N)
        src = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if src and re.search(r"\.shift\s*\(\s*-\s*\d+", src):
            self.violations.append((
                node.lineno,
                "negative_shift_assign",
                src[:120],
            ))
        # Look for:   signal = df[...].expanding().mean()  without shift
        if src and re.search(r"\.expanding\s*\(\s*\)\s*\.\s*(?:mean|std|sum)", src):
            if ".shift" not in src:
                self.violations.append((
                    node.lineno,
                    "expanding_no_shift",
                    src[:120],
                ))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        """Flag loops that index with df.iloc[-1]."""
        src = ast.unparse(node) if hasattr(ast, "unparse") else ""
        if src and re.search(r"\.iloc\s*\[\s*-\s*1\s*\]", src):
            self.violations.append((
                node.lineno,
                "iloc_minus_one_in_loop",
                src[:120],
            ))
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# LookaheadGuard
# ---------------------------------------------------------------------------

class LookaheadGuard:
    """
    Detects and prevents lookahead bias in strategy code and signal series.

    Parameters
    ----------
    significance_level : p-value threshold for ``validate_signal_timing`` —
                         below this we flag correlation with future as suspicious.
    max_lag_test : number of forward lags to test in timing validation.
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lag_test: int = 5,
    ) -> None:
        self.significance_level = significance_level
        self.max_lag_test = max_lag_test

    # ------------------------------------------------------------------
    # 1. Static code analysis
    # ------------------------------------------------------------------

    def scan_strategy_code(self, code: str) -> list[LookaheadViolation]:
        """
        Scan Python source *code* for lookahead-bias patterns.

        Uses both regex pattern matching (for speed) and AST analysis (for
        precision).  Deduplicates by line number + type.

        Returns a list of :class:`LookaheadViolation`, sorted by severity.
        """
        violations: list[LookaheadViolation] = []
        lines = code.splitlines()

        # --- Regex pass ---
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for vtype, pattern, severity, desc, fix in _ALL_PATTERNS:
                if pattern.search(line):
                    violations.append(LookaheadViolation(
                        violation_type=vtype,
                        severity=severity,
                        line_number=line_no,
                        code_snippet=stripped[:120],
                        description=desc,
                        fix_suggestion=fix,
                    ))

        # --- AST pass ---
        try:
            tree = ast.parse(textwrap.dedent(code))
            visitor = _LookaheadVisitor()
            visitor.visit(tree)
            seen_lines = {(v.line_number, v.violation_type) for v in violations}
            for lineno, vtype, snippet in visitor.violations:
                if (lineno, vtype) not in seen_lines:
                    violations.append(LookaheadViolation(
                        violation_type=vtype,
                        severity="warning",
                        line_number=lineno,
                        code_snippet=snippet,
                        description=f"AST-detected pattern: {vtype}",
                        fix_suggestion="Review usage — potential lookahead.",
                    ))
        except SyntaxError as exc:
            logger.debug("AST parse failed (syntax error): %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("AST analysis error: %s", exc)

        # Sort: critical first, then by line number
        _sev_order = {"critical": 0, "warning": 1, "info": 2}
        violations.sort(key=lambda v: (_sev_order.get(v.severity, 3), v.line_number or 0))

        return violations

    def scan_file(self, path: str | Path) -> list[LookaheadViolation]:
        """Convenience wrapper that reads a file and calls ``scan_strategy_code``."""
        code = Path(path).read_text(encoding="utf-8")
        return self.scan_strategy_code(code)

    def scan_directory(
        self,
        directory: str | Path,
        pattern: str = "**/*.py",
    ) -> dict[str, list[LookaheadViolation]]:
        """
        Scan all Python files matching *pattern* under *directory*.

        Returns a dict mapping file path → list of violations.
        Only files with at least one violation are included.
        """
        results: dict[str, list[LookaheadViolation]] = {}
        for fpath in Path(directory).glob(pattern):
            try:
                viols = self.scan_file(fpath)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not scan %s: %s", fpath, exc)
                continue
            if viols:
                results[str(fpath)] = viols
        return results

    # ------------------------------------------------------------------
    # 2. Runtime signal timing validation
    # ------------------------------------------------------------------

    def validate_signal_timing(
        self,
        signal_series: pd.Series,
        price_series: pd.Series,
    ) -> SignalTimingResult:
        """
        Validate that *signal_series* at bar T only uses information from
        bars ≤ T.

        Method: compute the Pearson correlation of the signal with future
        returns at lags 1…max_lag_test.  If any forward lag shows a statistically
        significant positive correlation at the given significance level, we flag
        the series as potentially lookahead-contaminated.

        Parameters
        ----------
        signal_series : numeric signal (e.g. position target, -1 to 1).
        price_series  : close price series aligned to the same timestamps.

        Returns
        -------
        SignalTimingResult
        """
        # Align
        both = pd.concat(
            [signal_series.rename("signal"), price_series.rename("price")],
            axis=1,
        ).dropna()

        if len(both) < 30:
            return SignalTimingResult(
                is_clean=True,
                correlation_with_future=0.0,
                p_value=1.0,
                max_lag_tested=0,
                description="Insufficient data for timing validation (< 30 bars).",
            )

        future_returns = both["price"].pct_change().shift(-1)
        signal = both["signal"]

        offending_lags: list[int] = []
        max_corr = 0.0
        min_p = 1.0

        for lag in range(1, self.max_lag_test + 1):
            fut_ret = both["price"].pct_change().shift(-lag).dropna()
            sig_aligned = signal.reindex(fut_ret.index).dropna()
            fut_aligned = fut_ret.reindex(sig_aligned.index).dropna()
            sig_aligned = sig_aligned.reindex(fut_aligned.index)

            if len(sig_aligned) < 10:
                continue

            corr, p_val = scipy_stats.pearsonr(sig_aligned.values, fut_aligned.values)
            if abs(corr) > abs(max_corr):
                max_corr = corr
            if p_val < min_p:
                min_p = p_val

            # Suspicious: significant positive correlation with future
            if p_val < self.significance_level and corr > 0.1:
                offending_lags.append(lag)
                logger.warning(
                    "Signal has %.3f correlation with %d-bar-ahead returns "
                    "(p=%.4f) — possible lookahead bias.",
                    corr, lag, p_val,
                )

        is_clean = len(offending_lags) == 0
        description = (
            "Signal timing appears clean."
            if is_clean
            else (
                f"Suspicious forward correlation at lags {offending_lags}. "
                "Signal may incorporate future price information."
            )
        )

        return SignalTimingResult(
            is_clean=is_clean,
            correlation_with_future=float(max_corr),
            p_value=float(min_p),
            max_lag_tested=self.max_lag_test,
            offending_lags=offending_lags,
            description=description,
        )

    # ------------------------------------------------------------------
    # 3. Backtest parameter validation
    # ------------------------------------------------------------------

    def validate_backtest_params(
        self, params: dict[str, Any]
    ) -> list[dict[str, str]]:
        """
        Review backtest parameters for known lookahead-bias risks.

        Returns a list of dicts with keys: param, risk, description.
        """
        issues: list[dict[str, str]] = []

        # In-sample optimisation without out-of-sample validation
        if params.get("in_sample_only", False):
            issues.append({
                "param": "in_sample_only",
                "risk": "critical",
                "description": (
                    "Backtest is run in-sample-only mode. No OOS validation — "
                    "results will be inflated."
                ),
            })

        # Look-forward in optimisation window
        opt_end = params.get("optimize_end_date")
        bt_end = params.get("backtest_end_date")
        if opt_end and bt_end and opt_end >= bt_end:
            issues.append({
                "param": "optimize_end_date",
                "risk": "critical",
                "description": (
                    f"optimize_end_date ({opt_end}) >= backtest_end_date ({bt_end}). "
                    "Parameter optimisation uses data from the evaluation period."
                ),
            })

        # Warm-up period too short for the rolling windows used
        warmup = params.get("warmup_bars", 0)
        min_hold = params.get("min_hold_bars", 4)
        if warmup < min_hold * 2:
            issues.append({
                "param": "warmup_bars",
                "risk": "warning",
                "description": (
                    f"warmup_bars={warmup} may be insufficient for indicators "
                    f"with windows up to {min_hold * 2}. Early bars will have "
                    "look-forward in their rolling statistics."
                ),
            })

        # Future price as a parameter (unusual but possible in misconfigured params)
        for key, val in params.items():
            if "future" in key.lower():
                issues.append({
                    "param": key,
                    "risk": "warning",
                    "description": (
                        f"Parameter '{key}' has 'future' in its name — "
                        "verify it does not carry forward-looking information."
                    ),
                })

        # Negative shift values stored as params
        for key in ("signal_shift", "entry_shift", "lag"):
            val = params.get(key)
            if val is not None and float(val) < 0:
                issues.append({
                    "param": key,
                    "risk": "critical",
                    "description": (
                        f"Parameter '{key}' = {val} is negative — "
                        "this encodes a forward-looking shift."
                    ),
                })

        # Rebalance frequency mismatch — rebalancing at the open using close-based signals
        if params.get("rebalance_at") == "open" and params.get("signal_at") == "close":
            issues.append({
                "param": "rebalance_at/signal_at",
                "risk": "warning",
                "description": (
                    "Rebalancing at the open using same-bar close signals is lookahead. "
                    "Signals should come from the *previous* bar's close."
                ),
            })

        return issues

    # ------------------------------------------------------------------
    # 4. Mark safe
    # ------------------------------------------------------------------

    def mark_safe(
        self,
        df: pd.DataFrame,
        signal_col: str,
        shift_periods: int = 1,
    ) -> pd.DataFrame:
        """
        Return *df* with *signal_col* shifted forward by *shift_periods* bars,
        ensuring the signal seen at bar T was generated using only data ≤ T-1.

        A ``{signal_col}_lookahead_safe`` column is added; the original column
        is preserved.

        Parameters
        ----------
        df             : DataFrame with a signal column.
        signal_col     : name of the signal column to make safe.
        shift_periods  : number of bars to shift forward (default 1).
        """
        if signal_col not in df.columns:
            raise ValueError(f"Column '{signal_col}' not found in DataFrame.")

        safe_col = f"{signal_col}_lookahead_safe"
        df = df.copy()
        df[safe_col] = df[signal_col].shift(shift_periods)

        logger.info(
            "Column '%s' shifted forward by %d bar(s) → '%s'.",
            signal_col, shift_periods, safe_col,
        )
        return df

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    def violation_summary(self, violations: list[LookaheadViolation]) -> str:
        """Return a human-readable summary of a violation list."""
        if not violations:
            return "No lookahead violations detected."

        critical = [v for v in violations if v.severity == "critical"]
        warnings = [v for v in violations if v.severity == "warning"]
        infos = [v for v in violations if v.severity == "info"]

        lines = [
            f"Lookahead analysis: {len(violations)} violation(s) found",
            f"  Critical : {len(critical)}",
            f"  Warnings : {len(warnings)}",
            f"  Info     : {len(infos)}",
            "",
        ]
        for v in violations[:20]:  # cap output
            loc = f"L{v.line_number}" if v.line_number else "?"
            lines.append(f"  [{v.severity.upper():8s}] {loc}: {v.description}")
            if v.fix_suggestion:
                lines.append(f"           → {v.fix_suggestion}")
        if len(violations) > 20:
            lines.append(f"  ... and {len(violations) - 20} more.")
        return "\n".join(lines)

    def html_report(self, violations: list[LookaheadViolation]) -> str:
        """Produce a minimal HTML table of violations for dashboards."""
        if not violations:
            return "<p>No lookahead violations detected.</p>"

        rows = "".join(
            f"<tr>"
            f"<td>{v.severity}</td>"
            f"<td>{v.violation_type}</td>"
            f"<td>{v.line_number or '?'}</td>"
            f"<td><code>{v.code_snippet}</code></td>"
            f"<td>{v.description}</td>"
            f"<td>{v.fix_suggestion}</td>"
            f"</tr>"
            for v in violations
        )
        return (
            "<table border='1'>"
            "<thead><tr>"
            "<th>Severity</th><th>Type</th><th>Line</th>"
            "<th>Snippet</th><th>Description</th><th>Fix</th>"
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        )
