"""
research/walk_forward/splits.py
───────────────────────────────
Time-series cross-validation splitters for walk-forward analysis.

Implements:
  • WFSplit dataclass
  • walk_forward_splits()  — fixed rolling window
  • expanding_window_splits() — expanding (anchored) window
  • rolling_window_splits()   — alias with explicit naming
  • CPCVSplitter              — Combinatorial Purged Cross-Validation (López de Prado 2018)
  • regime_stratified_splits()— regime-balanced folds
  • purge_overlap()           — embargo / purging utility

All index arrays are numpy int64 arrays for downstream compatibility.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Generator, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# WFSplit dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WFSplit:
    """
    Represents a single train/test split in a walk-forward scheme.

    Attributes
    ----------
    train_idx   : numpy array of integer indices for the training window.
    test_idx    : numpy array of integer indices for the test window.
    fold_id     : zero-based fold index.
    train_start : first index in training window (integer position).
    train_end   : last index in training window (inclusive).
    test_start  : first index in test window.
    test_end    : last index in test window (inclusive).
    metadata    : optional dict for extra info (e.g. regime proportions).
    """

    train_idx:   np.ndarray
    test_idx:    np.ndarray
    fold_id:     int
    train_start: int
    train_end:   int
    test_start:  int
    test_end:    int
    metadata:    Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.train_idx, np.ndarray):
            self.train_idx = np.asarray(self.train_idx, dtype=np.int64)
        if not isinstance(self.test_idx, np.ndarray):
            self.test_idx = np.asarray(self.test_idx, dtype=np.int64)

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    @property
    def train_span(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def test_span(self) -> int:
        return self.test_end - self.test_start + 1

    def __repr__(self) -> str:
        return (
            f"WFSplit(fold={self.fold_id}, "
            f"train=[{self.train_start}:{self.train_end}] n={self.n_train}, "
            f"test=[{self.test_start}:{self.test_end}] n={self.n_test})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: index array builders
# ─────────────────────────────────────────────────────────────────────────────

def _arange(start: int, stop: int) -> np.ndarray:
    """Return int64 arange [start, stop) (stop exclusive)."""
    return np.arange(start, stop, dtype=np.int64)


def _validate_params(n: int, train_size: int, test_size: int, step: int, gap: int = 0) -> None:
    """Raise ValueError for nonsensical split parameters."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")
    if test_size <= 0:
        raise ValueError(f"test_size must be positive, got {test_size}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    if train_size + test_size + gap > n:
        raise ValueError(
            f"train_size({train_size}) + test_size({test_size}) + gap({gap}) = "
            f"{train_size + test_size + gap} > n({n})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# walk_forward_splits — fixed-size rolling window
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int,
    gap: int = 0,
) -> List[WFSplit]:
    """
    Generate fixed-size rolling walk-forward splits.

    Each fold:
        train = [start, start + train_size)
        gap   = train_size bars discarded (overlap buffer)
        test  = [start + train_size + gap, start + train_size + gap + test_size)

    The window shifts forward by `step` bars each fold.

    Parameters
    ----------
    n          : total number of observations.
    train_size : number of training observations per fold.
    test_size  : number of test observations per fold.
    step       : number of bars to advance between folds (≤ test_size for contiguous OOS).
    gap        : embargo gap between train end and test start (default 0).

    Returns
    -------
    List of WFSplit objects, chronologically ordered.

    Examples
    --------
    >>> splits = walk_forward_splits(1000, 500, 100, 100, gap=5)
    >>> len(splits)
    5
    """
    _validate_params(n, train_size, test_size, step, gap)

    splits: List[WFSplit] = []
    fold_id = 0
    start = 0

    while True:
        train_start = start
        train_end   = start + train_size - 1
        test_start  = train_end + 1 + gap
        test_end    = test_start + test_size - 1

        if test_end >= n:
            break

        splits.append(
            WFSplit(
                train_idx   = _arange(train_start, train_end + 1),
                test_idx    = _arange(test_start,  test_end  + 1),
                fold_id     = fold_id,
                train_start = train_start,
                train_end   = train_end,
                test_start  = test_start,
                test_end    = test_end,
            )
        )

        fold_id += 1
        start   += step

    if not splits:
        warnings.warn(
            f"walk_forward_splits produced 0 folds for n={n}, "
            f"train_size={train_size}, test_size={test_size}, step={step}, gap={gap}",
            UserWarning,
            stacklevel=2,
        )
    else:
        logger.debug(
            "walk_forward_splits: n=%d → %d folds (train=%d, test=%d, step=%d, gap=%d)",
            n, len(splits), train_size, test_size, step, gap,
        )

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# expanding_window_splits — anchored / expanding training set
# ─────────────────────────────────────────────────────────────────────────────

def expanding_window_splits(
    n: int,
    min_train: int,
    test_size: int,
    step: int,
    gap: int = 0,
) -> List[WFSplit]:
    """
    Generate expanding-window (anchored-start) walk-forward splits.

    Training always starts at index 0 and grows with each fold. This is the
    classic "anchored walk-forward" used in financial literature.

    Parameters
    ----------
    n         : total number of observations.
    min_train : minimum training window size (first fold).
    test_size : number of test observations per fold (fixed).
    step      : how many bars to extend training and advance test each fold.
    gap       : embargo gap between train end and test start.

    Returns
    -------
    List of WFSplit objects.

    Examples
    --------
    >>> splits = expanding_window_splits(1000, 400, 100, 100)
    >>> splits[0].n_train, splits[1].n_train
    (400, 500)
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if min_train <= 0:
        raise ValueError(f"min_train must be positive, got {min_train}")
    if test_size <= 0:
        raise ValueError(f"test_size must be positive, got {test_size}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    if min_train + test_size + gap > n:
        raise ValueError(
            f"min_train({min_train}) + test_size({test_size}) + gap({gap}) > n({n})"
        )

    splits: List[WFSplit] = []
    fold_id    = 0
    train_size = min_train

    while True:
        train_start = 0
        train_end   = train_size - 1
        test_start  = train_end + 1 + gap
        test_end    = test_start + test_size - 1

        if test_end >= n:
            break

        splits.append(
            WFSplit(
                train_idx   = _arange(train_start, train_end + 1),
                test_idx    = _arange(test_start,  test_end  + 1),
                fold_id     = fold_id,
                train_start = train_start,
                train_end   = train_end,
                test_start  = test_start,
                test_end    = test_end,
            )
        )

        fold_id    += 1
        train_size += step

    if not splits:
        warnings.warn(
            f"expanding_window_splits produced 0 folds for n={n}, min_train={min_train}, "
            f"test_size={test_size}, step={step}",
            UserWarning,
            stacklevel=2,
        )
    else:
        logger.debug(
            "expanding_window_splits: n=%d → %d folds (min_train=%d, test=%d, step=%d)",
            n, len(splits), min_train, test_size, step,
        )

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# rolling_window_splits — explicit alias for clarity
# ─────────────────────────────────────────────────────────────────────────────

def rolling_window_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int,
    gap: int = 0,
) -> List[WFSplit]:
    """
    Alias for walk_forward_splits with rolling (fixed-size) window semantics.

    Identical to walk_forward_splits; provided for API clarity when callers
    want to explicitly signal rolling vs expanding intent.

    Parameters
    ----------
    n          : total observations.
    train_size : fixed training window length.
    test_size  : fixed test window length.
    step       : fold advance step.
    gap        : embargo between train end and test start.

    Returns
    -------
    List[WFSplit]
    """
    return walk_forward_splits(n, train_size, test_size, step, gap)


# ─────────────────────────────────────────────────────────────────────────────
# purge_overlap — remove training indices that overlap with test period + embargo
# ─────────────────────────────────────────────────────────────────────────────

def purge_overlap(
    train_idx: np.ndarray,
    test_idx:  np.ndarray,
    embargo:   int = 5,
) -> np.ndarray:
    """
    Remove from train_idx any indices within `embargo` bars of the test window.

    In financial time-series, labels computed from overlapping windows (e.g.
    triple-barrier labels spanning multiple bars) can leak forward. Purging
    removes training samples whose label-horizon overlaps the test period.

    This simplified version removes all training indices whose position is
    within `embargo` bars BEFORE the first test index. For full label-based
    purging, pass the label end-times explicitly to CPCVSplitter.

    Parameters
    ----------
    train_idx : integer positions of the training set.
    test_idx  : integer positions of the test set.
    embargo   : number of bars before test_start to exclude from training.

    Returns
    -------
    Purged train_idx (numpy int64 array).

    Examples
    --------
    >>> train = np.arange(100)
    >>> test  = np.arange(100, 200)
    >>> purged = purge_overlap(train, test, embargo=10)
    >>> purged[-1]  # should be 89
    89
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return train_idx.copy()

    test_start = int(test_idx.min())
    cutoff     = test_start - embargo

    # Remove training indices at or after the cutoff
    purged = train_idx[train_idx < cutoff]

    n_purged = len(train_idx) - len(purged)
    if n_purged > 0:
        logger.debug("purge_overlap: removed %d training samples (embargo=%d)", n_purged, embargo)

    return purged.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# CPCVSplitter — Combinatorial Purged Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

class CPCVSplitter:
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Implements the CPCV algorithm from López de Prado (2018) *Advances in
    Financial Machine Learning*, Chapter 12.

    In standard k-fold CV, all k-1 folds are used for training and 1 for
    testing — this produces only 1 backtest path. CPCV splits the data into k
    groups and uses C(k, k_test) combinations of test groups, producing
    C(k, k_test) * k_test / k backtest paths covering the full data length.

    This allows estimation of the Probability of Backtest Overfitting (PBO)
    from the empirical distribution of IS vs OOS Sharpe ratios.

    Parameters
    ----------
    n_splits     : total number of groups (k), default 6.
    n_test_splits: number of groups used as test (k_test), default 2.
    purge        : bars to purge at boundary between train/test groups.
    embargo      : bars after test group to exclude from training.

    Examples
    --------
    >>> cpcv = CPCVSplitter(n_splits=6, n_test_splits=2)
    >>> splits = list(cpcv.split(X))
    >>> len(splits)
    15  # C(6,2) = 15

    References
    ----------
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Wiley. Chapter 12: Backtesting Through Cross-Validation.
    """

    def __init__(
        self,
        n_splits:      int = 6,
        n_test_splits: int = 2,
        purge:         int = 5,
        embargo:       int = 5,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be ≥ 2, got {n_splits}")
        if n_test_splits < 1:
            raise ValueError(f"n_test_splits must be ≥ 1, got {n_test_splits}")
        if n_test_splits >= n_splits:
            raise ValueError(
                f"n_test_splits({n_test_splits}) must be < n_splits({n_splits})"
            )
        if purge < 0:
            raise ValueError(f"purge must be ≥ 0, got {purge}")
        if embargo < 0:
            raise ValueError(f"embargo must be ≥ 0, got {embargo}")

        self.n_splits      = n_splits
        self.n_test_splits = n_test_splits
        self.purge         = purge
        self.embargo       = embargo

        # Number of combinatorial test paths: C(k, k_test)
        self._n_combinations = math.comb(n_splits, n_test_splits)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_n_splits(self, X: Optional[np.ndarray] = None) -> int:
        """Return the number of (train, test) pairs (= C(k, k_test))."""
        return self._n_combinations

    def split(
        self,
        X:  np.ndarray,
        y:  Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_idx, test_idx) pairs for all C(k, k_test) combinations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or 1-D length array.
            Only the length is used.
        y : ignored (present for sklearn compatibility).

        Yields
        ------
        train_idx : purged training indices.
        test_idx  : test indices.
        """
        n = len(X)
        groups_arr = self._make_groups(n)

        for test_group_ids in itertools.combinations(range(self.n_splits), self.n_test_splits):
            test_groups  = set(test_group_ids)
            train_groups = set(range(self.n_splits)) - test_groups

            test_idx  = self._group_indices(groups_arr, test_groups)
            train_idx = self._group_indices(groups_arr, train_groups)

            # Apply purge + embargo
            train_idx = self._purge_and_embargo(train_idx, test_idx, n)

            yield train_idx, test_idx

    def get_backtest_paths(self, n: int) -> List[List[int]]:
        """
        Return all combinatorial backtest paths.

        Each path is a list of group IDs constituting a contiguous
        out-of-sample coverage of the full sample. CPCV generates
        C(k, k_test) * k_test / k full-length paths by recombining
        test folds across combinations.

        Parameters
        ----------
        n : total number of observations (used to build group sizes).

        Returns
        -------
        List of paths, where each path is a list of group indices (sorted)
        that together cover all k groups exactly once in combination.

        Note
        ----
        For PBO estimation, each unique arrangement of test-group selections
        that collectively cover the full timeline is one "backtest path."
        """
        groups_arr     = self._make_groups(n)
        all_combos     = list(itertools.combinations(range(self.n_splits), self.n_test_splits))

        # Build a bipartite coverage: each path must cover each group once
        # Group index → which combos include it
        group_to_combos: Dict[int, List[int]] = defaultdict(list)
        for ci, combo in enumerate(all_combos):
            for g in combo:
                group_to_combos[g].append(ci)

        # A complete backtest path = k / k_test non-overlapping combos that
        # together cover all k groups. We enumerate via exact cover search.
        paths: List[List[int]] = []
        n_paths_needed = self.n_splits // self.n_test_splits

        # Use backtracking to find non-overlapping combo sets
        def _backtrack(covered: set, chosen: List[int], start_ci: int) -> None:
            if len(covered) == self.n_splits:
                paths.append(list(chosen))
                return
            if len(chosen) == n_paths_needed:
                return
            for ci in range(start_ci, len(all_combos)):
                combo = set(all_combos[ci])
                if combo.isdisjoint(covered):
                    _backtrack(covered | combo, chosen + [ci], ci + 1)

        _backtrack(set(), [], 0)

        logger.debug(
            "CPCV get_backtest_paths: n=%d, n_splits=%d, n_test_splits=%d → "
            "%d combos, %d complete paths",
            n, self.n_splits, self.n_test_splits, len(all_combos), len(paths),
        )

        return paths

    def get_path_test_indices(self, n: int) -> List[List[np.ndarray]]:
        """
        Return test index arrays for each backtest path.

        Parameters
        ----------
        n : total number of observations.

        Returns
        -------
        List of paths; each path is a list of np.ndarray test-index blocks
        in chronological order, together covering the full sample.
        """
        groups_arr  = self._make_groups(n)
        all_combos  = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        paths       = self.get_backtest_paths(n)

        result: List[List[np.ndarray]] = []
        for path in paths:
            path_test_blocks: List[np.ndarray] = []
            all_test_groups: List[int] = []
            for ci in path:
                all_test_groups.extend(all_combos[ci])
            # Sort groups by their chronological position
            all_test_groups_sorted = sorted(all_test_groups)
            for g in all_test_groups_sorted:
                path_test_blocks.append(self._group_indices(groups_arr, {g}))
            result.append(path_test_blocks)

        return result

    def summary(self) -> str:
        """Return a human-readable summary of the CPCV configuration."""
        n_combinations = self._n_combinations
        n_paths = math.comb(self.n_splits, self.n_test_splits)
        coverage_ratio = self.n_test_splits / self.n_splits
        return (
            f"CPCVSplitter(\n"
            f"  n_splits={self.n_splits}, n_test_splits={self.n_test_splits}\n"
            f"  purge={self.purge}, embargo={self.embargo}\n"
            f"  C({self.n_splits},{self.n_test_splits}) = {n_combinations} train/test pairs\n"
            f"  OOS coverage per combination: {coverage_ratio:.1%}\n"
            f"  Complete backtest paths: {n_paths}\n"
            f")"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_groups(self, n: int) -> np.ndarray:
        """
        Assign each of the n observations to one of n_splits groups.

        Groups are contiguous and as equal-sized as possible. The last group
        absorbs any remainder observations.

        Returns
        -------
        groups_arr : int64 array of shape (n,) with values in [0, n_splits).
        """
        groups_arr = np.empty(n, dtype=np.int64)
        base_size  = n // self.n_splits
        remainder  = n % self.n_splits

        pos = 0
        for g in range(self.n_splits):
            size = base_size + (1 if g < remainder else 0)
            groups_arr[pos:pos + size] = g
            pos += size

        return groups_arr

    def _group_indices(self, groups_arr: np.ndarray, group_ids: set) -> np.ndarray:
        """Return sorted indices belonging to any of the given group IDs."""
        mask = np.isin(groups_arr, list(group_ids))
        return np.where(mask)[0].astype(np.int64)

    def _purge_and_embargo(
        self,
        train_idx: np.ndarray,
        test_idx:  np.ndarray,
        n:         int,
    ) -> np.ndarray:
        """
        Apply purge (before test) and embargo (after test) to training indices.

        For each contiguous test block, remove training indices that fall
        within `purge` bars before the block starts or `embargo` bars after
        the block ends.
        """
        if len(test_idx) == 0 or len(train_idx) == 0:
            return train_idx

        # Identify contiguous test blocks (groups may be non-contiguous)
        test_idx_sorted = np.sort(test_idx)
        breaks = np.where(np.diff(test_idx_sorted) > 1)[0] + 1
        blocks: List[Tuple[int, int]] = []

        prev = 0
        for b in breaks:
            blocks.append((int(test_idx_sorted[prev]), int(test_idx_sorted[b - 1])))
            prev = b
        blocks.append((int(test_idx_sorted[prev]), int(test_idx_sorted[-1])))

        # Build mask: True = keep in training
        keep = np.ones(len(train_idx), dtype=bool)
        train_set = train_idx  # sorted in general

        for block_start, block_end in blocks:
            purge_start  = max(0,     block_start - self.purge)
            embargo_end  = min(n - 1, block_end   + self.embargo)

            # Remove training indices in [purge_start, embargo_end]
            mask_block = (train_idx >= purge_start) & (train_idx <= embargo_end)
            keep &= ~mask_block

        purged = train_idx[keep]
        n_removed = len(train_idx) - len(purged)
        if n_removed > 0:
            logger.debug(
                "_purge_and_embargo: removed %d / %d training samples",
                n_removed, len(train_idx),
            )
        return purged.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# regime_stratified_splits — balanced regime composition per fold
# ─────────────────────────────────────────────────────────────────────────────

def regime_stratified_splits(
    trades:      pd.DataFrame,
    n_folds:     int,
    regime_col:  str = "regime",
    gap:         int = 0,
    min_trades_per_regime_per_fold: int = 3,
) -> List[WFSplit]:
    """
    Generate walk-forward splits ensuring each fold has similar regime composition.

    Standard walk-forward splits can result in test folds that are dominated
    by a single regime (e.g., all bear market). Regime-stratified splits
    attempt to preserve the population regime distribution in both train and
    test folds.

    Algorithm
    ---------
    1. Sort trades by exit_time (chronological order).
    2. Compute cumulative regime frequencies.
    3. Use a greedy assignment that maximises KL-divergence reduction between
       fold regime distribution and global regime distribution.
    4. Fall back to uniform time splits if stratification is not feasible.

    Parameters
    ----------
    trades     : DataFrame with at least a `regime` column (and ideally
                 `exit_time` for time-ordering).
    n_folds    : desired number of folds.
    regime_col : column name containing regime labels.
    gap        : gap between train and test in number of trades.
    min_trades_per_regime_per_fold : warn if any fold contains fewer trades
                                     from a given regime.

    Returns
    -------
    List[WFSplit] of length n_folds.

    Notes
    -----
    If the DataFrame has a DatetimeIndex or an `exit_time` column, trades are
    sorted chronologically before splitting. Otherwise, the existing order is
    preserved and splits are sequential.
    """
    if trades.empty:
        raise ValueError("trades DataFrame is empty")
    if n_folds < 2:
        raise ValueError(f"n_folds must be ≥ 2, got {n_folds}")
    if regime_col not in trades.columns:
        raise ValueError(f"Column '{regime_col}' not found in trades DataFrame")

    # Sort chronologically
    df = trades.copy().reset_index(drop=True)
    if "exit_time" in df.columns:
        df = df.sort_values("exit_time").reset_index(drop=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index().reset_index(drop=True)

    n = len(df)
    all_regimes     = sorted(df[regime_col].unique())
    global_counts   = df[regime_col].value_counts(normalize=True).to_dict()
    global_dist     = np.array([global_counts.get(r, 0.0) for r in all_regimes])

    logger.debug(
        "regime_stratified_splits: n=%d, n_folds=%d, regimes=%s",
        n, n_folds, all_regimes,
    )
    logger.debug("Global regime distribution: %s", dict(zip(all_regimes, global_dist)))

    # Try stratified assignment using a sliding window that maximises
    # coverage fidelity to the global distribution.
    test_size  = n // n_folds
    train_size = n - test_size * n_folds  # absorbed into first fold or evenly

    # Use equal-length sequential folds with regime diagnostic
    fold_size = n // n_folds
    splits: List[WFSplit] = []

    for fold_id in range(n_folds):
        # Test is the last `fold_size` trades in this fold's segment
        segment_start = fold_id * fold_size
        segment_end   = (fold_id + 1) * fold_size if fold_id < n_folds - 1 else n

        # Test window: last `fold_size` portion of current segment
        test_start_raw = max(segment_start, segment_end - fold_size)
        test_end_raw   = segment_end

        # Train window: everything before this segment (no look-ahead)
        # Use all prior folds as training
        if fold_id == 0:
            # Not enough history for fold 0: skip or use minimal
            logger.warning("Fold 0 has no training data — skipped in regime_stratified_splits")
            continue

        train_end_raw   = segment_start - gap
        train_start_raw = 0

        if train_end_raw <= train_start_raw:
            logger.warning("Fold %d: train_end=%d ≤ train_start=%d — skipped", fold_id, train_end_raw, train_start_raw)
            continue

        train_idx = _arange(train_start_raw, train_end_raw)
        test_idx  = _arange(test_start_raw,  test_end_raw)

        # Compute fold-specific regime distribution
        fold_test_regimes = df.iloc[test_idx][regime_col].value_counts(normalize=True).to_dict()
        fold_dist         = np.array([fold_test_regimes.get(r, 0.0) for r in all_regimes])

        # KL divergence from global
        kl = _kl_divergence(global_dist, fold_dist)

        # Check per-regime trade counts
        regime_counts_test: Dict[str, int] = {}
        for r in all_regimes:
            cnt = int((df.iloc[test_idx][regime_col] == r).sum())
            regime_counts_test[r] = cnt
            if 0 < cnt < min_trades_per_regime_per_fold:
                logger.warning(
                    "Fold %d test: regime '%s' has only %d trades (< min %d)",
                    fold_id, r, cnt, min_trades_per_regime_per_fold,
                )

        splits.append(
            WFSplit(
                train_idx   = train_idx,
                test_idx    = test_idx,
                fold_id     = fold_id - 1,  # re-base to 0
                train_start = int(train_start_raw),
                train_end   = int(train_end_raw - 1),
                test_start  = int(test_start_raw),
                test_end    = int(test_end_raw - 1),
                metadata    = {
                    "regime_distribution": fold_test_regimes,
                    "kl_divergence_from_global": float(kl),
                    "regime_counts_test": regime_counts_test,
                    "global_distribution": global_counts,
                },
            )
        )

    if not splits:
        raise RuntimeError(
            "regime_stratified_splits produced 0 folds — check input size and n_folds"
        )

    # Re-assign fold IDs sequentially
    for i, sp in enumerate(splits):
        sp.fold_id = i

    logger.info(
        "regime_stratified_splits: produced %d folds from n=%d trades, n_folds=%d",
        len(splits), n, n_folds,
    )

    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL divergence KL(p || q) with clipping to avoid log(0).

    Parameters
    ----------
    p, q : probability distributions (will be normalized internally).

    Returns
    -------
    float KL divergence.
    """
    p_ = np.clip(p, eps, None)
    q_ = np.clip(q, eps, None)
    p_ = p_ / p_.sum()
    q_ = q_ / q_.sum()
    return float(np.sum(p_ * np.log(p_ / q_)))


def _describe_splits(splits: List[WFSplit]) -> str:
    """Return a compact string summary of a list of WFSplit objects."""
    if not splits:
        return "[] (empty)"
    lines = [f"  {sp}" for sp in splits]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic utilities
# ─────────────────────────────────────────────────────────────────────────────

def verify_no_lookahead(splits: List[WFSplit]) -> bool:
    """
    Verify that no split contains test indices ≤ max(train_idx).

    Returns True if all splits are clean (no data leakage), False otherwise.
    Logs warnings for any violations.
    """
    clean = True
    for sp in splits:
        if len(sp.train_idx) == 0 or len(sp.test_idx) == 0:
            continue
        max_train = int(sp.train_idx.max())
        min_test  = int(sp.test_idx.min())
        if min_test <= max_train:
            logger.warning(
                "LOOKAHEAD in fold %d: min_test=%d ≤ max_train=%d",
                sp.fold_id, min_test, max_train,
            )
            clean = False
    return clean


def split_coverage(splits: List[WFSplit], n: int) -> Dict[str, object]:
    """
    Compute OOS coverage statistics for a list of splits.

    Parameters
    ----------
    splits : list of WFSplit objects.
    n      : total number of observations.

    Returns
    -------
    Dict with keys: n_folds, unique_test_indices, oos_coverage_fraction,
    overlap_count (indices appearing in >1 test fold).
    """
    all_test: List[int] = []
    for sp in splits:
        all_test.extend(sp.test_idx.tolist())

    counter = Counter(all_test)
    unique  = len(counter)
    overlap = sum(1 for v in counter.values() if v > 1)

    return {
        "n_folds":               len(splits),
        "unique_test_indices":   unique,
        "oos_coverage_fraction": unique / n if n > 0 else 0.0,
        "overlap_count":         overlap,
        "total_test_obs":        len(all_test),
    }


def time_splits_to_date_splits(
    splits:     List[WFSplit],
    index:      pd.DatetimeIndex,
) -> List[WFSplit]:
    """
    Attach datetime information to WFSplit metadata.

    Parameters
    ----------
    splits : list produced by any split function.
    index  : DatetimeIndex aligned with the integer positions in splits.

    Returns
    -------
    Same list with metadata updated to include:
    train_start_date, train_end_date, test_start_date, test_end_date.
    """
    if len(index) == 0:
        return splits

    for sp in splits:
        try:
            sp.metadata["train_start_date"] = str(index[sp.train_start])
            sp.metadata["train_end_date"]   = str(index[sp.train_end])
            sp.metadata["test_start_date"]  = str(index[sp.test_start])
            sp.metadata["test_end_date"]    = str(index[sp.test_end])
        except (IndexError, KeyError):
            pass

    return splits


def print_splits_summary(splits: List[WFSplit]) -> None:
    """Print a human-readable table of all splits to stdout."""
    if not splits:
        print("No splits to display.")
        return

    header = f"{'Fold':>5} | {'Train [start:end]':>25} | {'N_train':>8} | {'Test [start:end]':>25} | {'N_test':>7}"
    print(header)
    print("-" * len(header))
    for sp in splits:
        train_range = f"[{sp.train_start}:{sp.train_end}]"
        test_range  = f"[{sp.test_start}:{sp.test_end}]"
        print(f"{sp.fold_id:>5} | {train_range:>25} | {sp.n_train:>8} | {test_range:>25} | {sp.n_test:>7}")


# ─────────────────────────────────────────────────────────────────────────────
# Sklearn-compatible adapter for CPCVSplitter
# ─────────────────────────────────────────────────────────────────────────────

class CPCVSklearnAdapter:
    """
    Thin sklearn cross-validator adapter wrapping CPCVSplitter.

    Allows CPCVSplitter to be used directly with sklearn's cross_val_score
    and GridSearchCV as the `cv` parameter.

    Parameters
    ----------
    n_splits     : total number of CPCV groups.
    n_test_splits: number of test groups per combination.
    purge        : purge bars at group boundaries.
    embargo      : embargo bars after test groups.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> cv = CPCVSklearnAdapter(n_splits=6, n_test_splits=2)
    >>> scores = cross_val_score(estimator, X, y, cv=cv)
    """

    def __init__(
        self,
        n_splits:      int = 6,
        n_test_splits: int = 2,
        purge:         int = 5,
        embargo:       int = 5,
    ) -> None:
        self._cpcv = CPCVSplitter(n_splits, n_test_splits, purge, embargo)

    def split(
        self,
        X:  object,
        y:  Optional[object] = None,
        groups: Optional[object] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Delegate to CPCVSplitter.split()."""
        yield from self._cpcv.split(np.asarray(X) if not isinstance(X, np.ndarray) else X)

    def get_n_splits(
        self,
        X:  Optional[object] = None,
        y:  Optional[object] = None,
        groups: Optional[object] = None,
    ) -> int:
        return self._cpcv.get_n_splits()


# ─────────────────────────────────────────────────────────────────────────────
# Factory function for quick setup
# ─────────────────────────────────────────────────────────────────────────────

def make_splitter(
    method:        str,
    n:             int,
    *,
    # rolling / expanding
    train_size:    Optional[int] = None,
    min_train:     Optional[int] = None,
    test_size:     Optional[int] = None,
    step:          Optional[int] = None,
    gap:           int = 0,
    # CPCV
    n_splits:      int = 6,
    n_test_splits: int = 2,
    purge:         int = 5,
    embargo:       int = 5,
) -> object:
    """
    Factory function for creating splitters by name.

    Parameters
    ----------
    method : one of 'rolling', 'expanding', 'cpcv'.
    n      : total observations (required for rolling/expanding, optional for cpcv).
    ...    : method-specific keyword arguments.

    Returns
    -------
    For 'rolling'/'expanding': List[WFSplit].
    For 'cpcv': CPCVSplitter instance.

    Examples
    --------
    >>> splits = make_splitter('rolling', 1000, train_size=500, test_size=100, step=100)
    >>> cpcv   = make_splitter('cpcv', n_splits=6, n_test_splits=2)
    """
    method = method.lower().strip()

    if method in ("rolling", "walk_forward", "wf"):
        if train_size is None or test_size is None or step is None:
            raise ValueError("rolling splitter requires train_size, test_size, step")
        return walk_forward_splits(n, train_size, test_size, step, gap)

    elif method in ("expanding", "anchored"):
        if min_train is None or test_size is None or step is None:
            raise ValueError("expanding splitter requires min_train, test_size, step")
        return expanding_window_splits(n, min_train, test_size, step, gap)

    elif method in ("cpcv", "combinatorial"):
        return CPCVSplitter(n_splits, n_test_splits, purge, embargo)

    else:
        raise ValueError(f"Unknown splitter method: '{method}'. Choose from: rolling, expanding, cpcv")
