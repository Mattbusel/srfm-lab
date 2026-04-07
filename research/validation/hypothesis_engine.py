"""
hypothesis_engine.py -- manages the research hypothesis lifecycle.

Hypothesis flow: PENDING -> TESTING -> CONFIRMED | REJECTED

All hypothesis state is persisted in a SQLite database so results survive
process restarts and can be audited.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    PENDING = "PENDING"
    TESTING = "TESTING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


@dataclass
class Hypothesis:
    """
    A single research hypothesis.

    Fields
    ------
    description  : plain English description of the signal idea
    signal_code  : Python expression or module path implementing the signal
    id           : auto-generated UUID (do not set manually)
    created_at   : UTC timestamp string (ISO 8601)
    status       : current lifecycle status
    tags         : optional metadata list (e.g. ['momentum', 'crypto'])
    """

    description: str
    signal_code: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = HypothesisStatus.PENDING.value
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            description=d["description"],
            signal_code=d["signal_code"],
            id=d.get("id", str(uuid.uuid4())),
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            status=d.get("status", HypothesisStatus.PENDING.value),
            tags=d.get("tags", []),
        )


@dataclass
class HypothesisTestResult:
    """
    Full result from running a hypothesis through the HypothesisTest engine.

    Fields
    ------
    hypothesis_id    : FK to Hypothesis.id
    sharpe_is        : annualised Sharpe in the in-sample period
    sharpe_oos       : annualised Sharpe in the out-of-sample period
    ic_is            : Spearman IC in-sample
    ic_oos           : Spearman IC out-of-sample
    p_value          : two-tailed p-value on OOS mean return != 0
    deflated_sharpe  : Sharpe adjusted for multiple comparisons (Bailey et al.)
    verdict          : 'CONFIRMED' or 'REJECTED'
    notes            : free-form string with failure / warning details
    tested_at        : UTC timestamp
    """

    hypothesis_id: str
    sharpe_is: float
    sharpe_oos: float
    ic_is: float
    ic_oos: float
    p_value: float
    deflated_sharpe: float
    verdict: str
    notes: str = ""
    tested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisTestResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class HypothesisTest:
    """
    Runs a hypothesis through IS / OOS evaluation with Deflated Sharpe.

    Configuration
    -------------
    is_fraction     : fraction of data used as in-sample (default 0.70)
    min_obs         : minimum number of trading days required (default 252)
    annual_factor   : annualization factor for Sharpe (default 252)
    n_prior_tests   : number of prior hypotheses tested in the library, used
                      for the Deflated Sharpe calculation
    """

    def __init__(
        self,
        is_fraction: float = 0.70,
        min_obs: int = 252,
        annual_factor: float = 252.0,
        n_prior_tests: int = 1,
    ) -> None:
        self.is_fraction = is_fraction
        self.min_obs = min_obs
        self.annual_factor = annual_factor
        self.n_prior_tests = max(1, n_prior_tests)

    def run_test(
        self,
        hypothesis: Hypothesis,
        signal_values: np.ndarray,
        forward_returns: np.ndarray,
    ) -> HypothesisTestResult:
        """
        Evaluate a hypothesis on provided signal and return data.

        Parameters
        ----------
        hypothesis      : Hypothesis object to evaluate
        signal_values   : array of signal values, shape (T,)
        forward_returns : forward returns aligned with signal_values, shape (T,)

        Returns
        -------
        HypothesisTestResult
        """
        sig = np.asarray(signal_values, dtype=float)
        ret = np.asarray(forward_returns, dtype=float)

        mask = np.isfinite(sig) & np.isfinite(ret)
        sig, ret = sig[mask], ret[mask]
        T = len(sig)

        notes_parts: list = []

        if T < self.min_obs:
            return HypothesisTestResult(
                hypothesis_id=hypothesis.id,
                sharpe_is=np.nan,
                sharpe_oos=np.nan,
                ic_is=np.nan,
                ic_oos=np.nan,
                p_value=np.nan,
                deflated_sharpe=np.nan,
                verdict="REJECTED",
                notes=f"Insufficient data: {T} obs < minimum {self.min_obs}",
            )

        split = int(self.is_fraction * T)

        sig_is, ret_is = sig[:split], ret[:split]
        sig_oos, ret_oos = sig[split:], ret[split:]

        sharpe_is = self._annualised_sharpe(ret_is)
        sharpe_oos = self._annualised_sharpe(ret_oos)
        ic_is = float(stats.spearmanr(sig_is, ret_is)[0])
        ic_oos = float(stats.spearmanr(sig_oos, ret_oos)[0])

        # p-value on OOS mean return != 0
        t_stat, p_val = stats.ttest_1samp(ret_oos, popmean=0.0)
        p_val = float(p_val)

        # Deflated Sharpe Ratio (Bailey, Borwein, Lopez de Prado, Zhu 2014)
        deflated_sharpe = self._deflated_sharpe(
            sharpe_oos, len(ret_oos), self.n_prior_tests
        )

        # Verdict criteria
        verdict = "CONFIRMED"
        if sharpe_oos < 0.5:
            verdict = "REJECTED"
            notes_parts.append(f"OOS Sharpe={sharpe_oos:.3f} < 0.5")
        if ic_oos < 0.03:
            verdict = "REJECTED"
            notes_parts.append(f"OOS IC={ic_oos:.4f} < 0.03")
        if p_val > 0.10:
            verdict = "REJECTED"
            notes_parts.append(f"OOS mean return not significant (p={p_val:.4f})")
        if deflated_sharpe < 0.0:
            verdict = "REJECTED"
            notes_parts.append(f"Deflated Sharpe={deflated_sharpe:.3f} < 0")
        if ic_oos < 0 and ic_is > 0:
            verdict = "REJECTED"
            notes_parts.append("IC sign flipped IS->OOS")

        return HypothesisTestResult(
            hypothesis_id=hypothesis.id,
            sharpe_is=sharpe_is,
            sharpe_oos=sharpe_oos,
            ic_is=ic_is,
            ic_oos=ic_oos,
            p_value=p_val,
            deflated_sharpe=deflated_sharpe,
            verdict=verdict,
            notes="; ".join(notes_parts),
        )

    def _annualised_sharpe(self, returns: np.ndarray) -> float:
        """Annualised Sharpe ratio from a returns array."""
        if len(returns) < 2:
            return 0.0
        mu = float(np.mean(returns))
        sigma = float(np.std(returns, ddof=1))
        if sigma < 1e-12:
            return 0.0
        return float(mu / sigma * np.sqrt(self.annual_factor))

    def _deflated_sharpe(
        self, sharpe_oos: float, n_obs: int, n_trials: int
    ) -> float:
        """
        Deflated Sharpe Ratio (DSR) following Bailey & Lopez de Prado (2014).

        Corrects the Sharpe ratio for the bias introduced by selecting the
        best result from n_trials independent backtests.

        DSR = SR * (1 - gamma * SR_expected_max) / sqrt(1 + ...)
        where SR_expected_max is the expected max Sharpe from n_trials trials.

        We use the simplified formula:
          E[max SR] ~ (1 - gamma) * Z((n-1)/n) + gamma * Z(1 - 1/n)
        where Z is the standard normal quantile (approximation).
        """
        from scipy.special import gamma as gamma_fn

        if n_obs < 4 or not np.isfinite(sharpe_oos):
            return float(sharpe_oos)

        # Expected maximum Sharpe from n_trials strategies
        # Approximation: E[max] ~ stats.norm.ppf(1 - 1/n_trials)
        if n_trials <= 1:
            expected_max = 0.0
        else:
            # Euler-Mascheroni constant
            euler = 0.5772156649
            z = np.sqrt(np.log(n_trials)) / np.sqrt(2 * np.log(2 * np.log(n_trials)))
            expected_max = float(stats.norm.ppf(1.0 - 1.0 / n_trials))

        # annualise: scale expected_max to per-observation units then re-annualise
        sr_per_obs = sharpe_oos / np.sqrt(self.annual_factor)
        exp_max_per_obs = expected_max / np.sqrt(self.annual_factor)

        # Standard error of the Sharpe estimator
        se_sr = float(np.sqrt((1 + 0.5 * sr_per_obs**2) / n_obs))

        if se_sr < 1e-10:
            return sharpe_oos

        # Probability that true SR > 0 after deflation
        z_dsr = (sr_per_obs - exp_max_per_obs) / se_sr
        prob = float(stats.norm.cdf(z_dsr))
        deflated = float(stats.norm.ppf(prob)) * np.sqrt(self.annual_factor)
        return deflated


# ---------------------------------------------------------------------------
# Hypothesis library (SQLite-backed)
# ---------------------------------------------------------------------------

class HypothesisLibrary:
    """
    Persistent store for hypothesis objects and test results.

    Uses SQLite so it is embeddable, requires no server, and the database
    file can be committed to version control as a record.
    """

    SCHEMA_HYPOTHESES = """
        CREATE TABLE IF NOT EXISTS hypotheses (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            signal_code TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]'
        )
    """

    SCHEMA_RESULTS = """
        CREATE TABLE IF NOT EXISTS hypothesis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id TEXT NOT NULL,
            sharpe_is REAL,
            sharpe_oos REAL,
            ic_is REAL,
            ic_oos REAL,
            p_value REAL,
            deflated_sharpe REAL,
            verdict TEXT NOT NULL,
            notes TEXT,
            tested_at TEXT NOT NULL,
            FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
        )
    """

    SCHEMA_ICIR_LOG = """
        CREATE TABLE IF NOT EXISTS icir_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hypothesis_id TEXT NOT NULL,
            logged_at TEXT NOT NULL,
            icir REAL NOT NULL
        )
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._bootstrap_schema()

    def _bootstrap_schema(self) -> None:
        with self._con:
            self._con.execute(self.SCHEMA_HYPOTHESES)
            self._con.execute(self.SCHEMA_RESULTS)
            self._con.execute(self.SCHEMA_ICIR_LOG)

    # -- CRUD ---------------------------------------------------------------

    def add_hypothesis(self, hyp: Hypothesis) -> str:
        """Persist a new hypothesis. Returns its id."""
        with self._con:
            self._con.execute(
                """
                INSERT OR REPLACE INTO hypotheses
                    (id, description, signal_code, created_at, status, tags)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    hyp.id,
                    hyp.description,
                    hyp.signal_code,
                    hyp.created_at,
                    hyp.status,
                    json.dumps(hyp.tags),
                ),
            )
        return hyp.id

    def get_hypothesis(self, hyp_id: str) -> Optional[Hypothesis]:
        """Retrieve a hypothesis by id. Returns None if not found."""
        row = self._con.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hyp_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_hypothesis(row)

    def update_status(self, hyp_id: str, status: str) -> None:
        """Update the status of a hypothesis."""
        with self._con:
            self._con.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (status, hyp_id),
            )

    def save_result(self, result: HypothesisTestResult) -> None:
        """Persist a HypothesisTestResult and update the parent hypothesis status."""
        with self._con:
            self._con.execute(
                """
                INSERT INTO hypothesis_results
                    (hypothesis_id, sharpe_is, sharpe_oos, ic_is, ic_oos,
                     p_value, deflated_sharpe, verdict, notes, tested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.hypothesis_id,
                    result.sharpe_is,
                    result.sharpe_oos,
                    result.ic_is,
                    result.ic_oos,
                    result.p_value,
                    result.deflated_sharpe,
                    result.verdict,
                    result.notes,
                    result.tested_at,
                ),
            )
            self._con.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (result.verdict, result.hypothesis_id),
            )

    def get_result(self, hyp_id: str) -> Optional[HypothesisTestResult]:
        """Retrieve the most recent result for a hypothesis."""
        row = self._con.execute(
            """
            SELECT * FROM hypothesis_results
            WHERE hypothesis_id = ?
            ORDER BY tested_at DESC LIMIT 1
            """,
            (hyp_id,),
        ).fetchone()
        if row is None:
            return None
        return HypothesisTestResult(
            hypothesis_id=row["hypothesis_id"],
            sharpe_is=row["sharpe_is"],
            sharpe_oos=row["sharpe_oos"],
            ic_is=row["ic_is"],
            ic_oos=row["ic_oos"],
            p_value=row["p_value"],
            deflated_sharpe=row["deflated_sharpe"],
            verdict=row["verdict"],
            notes=row["notes"] or "",
            tested_at=row["tested_at"],
        )

    # -- Queries ------------------------------------------------------------

    def get_all(self, status: Optional[str] = None) -> list:
        """Return all hypotheses, optionally filtered by status."""
        if status:
            rows = self._con.execute(
                "SELECT * FROM hypotheses WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT * FROM hypotheses ORDER BY created_at"
            ).fetchall()
        return [self._row_to_hypothesis(r) for r in rows]

    def get_confirmed(self) -> list:
        """Return all CONFIRMED hypotheses ready for deployment."""
        return self.get_all(status=HypothesisStatus.CONFIRMED.value)

    def get_pending(self) -> list:
        """Return all PENDING hypotheses awaiting testing."""
        return self.get_all(status=HypothesisStatus.PENDING.value)

    # -- ICIR tracking and retirement --------------------------------------

    def log_icir(self, hyp_id: str, icir: float) -> None:
        """Record a periodic ICIR observation for a hypothesis."""
        with self._con:
            self._con.execute(
                "INSERT INTO icir_log (hypothesis_id, logged_at, icir) VALUES (?, ?, ?)",
                (hyp_id, datetime.now(timezone.utc).isoformat(), float(icir)),
            )

    def retirement_check(
        self,
        min_observations: int = 20,
        degradation_threshold: float = 0.5,
    ) -> list:
        """
        Identify CONFIRMED hypotheses whose ICIR has degraded significantly.

        A hypothesis is flagged for retirement when the ratio of its recent
        (last 25%) ICIR to its full-history mean ICIR falls below the
        degradation_threshold.

        Parameters
        ----------
        min_observations      : minimum ICIR log entries needed to evaluate
        degradation_threshold : ratio below which a hypothesis is flagged

        Returns
        -------
        list of Hypothesis objects flagged for retirement
        """
        confirmed = self.get_confirmed()
        flagged = []

        for hyp in confirmed:
            rows = self._con.execute(
                "SELECT icir FROM icir_log WHERE hypothesis_id = ? ORDER BY logged_at",
                (hyp.id,),
            ).fetchall()
            if len(rows) < min_observations:
                continue
            icir_series = np.array([r["icir"] for r in rows], dtype=float)
            n = len(icir_series)
            recent_start = int(0.75 * n)
            mean_all = float(np.mean(icir_series))
            mean_recent = float(np.mean(icir_series[recent_start:]))

            if mean_all > 1e-6:
                ratio = mean_recent / mean_all
            elif mean_recent < 0:
                ratio = 0.0
            else:
                continue

            if ratio < degradation_threshold:
                flagged.append(hyp)

        return flagged

    # -- Internals ---------------------------------------------------------

    @staticmethod
    def _row_to_hypothesis(row: sqlite3.Row) -> Hypothesis:
        return Hypothesis(
            id=row["id"],
            description=row["description"],
            signal_code=row["signal_code"],
            created_at=row["created_at"],
            status=row["status"],
            tags=json.loads(row["tags"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def __del__(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass
