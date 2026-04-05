"""
training/hyperopt.py
=====================
Random search + successive halving for hyperparameter optimisation.

Financial rationale
-------------------
Hyperparameter optimisation in finance is more dangerous than in vision
/ NLP because:
1. The evaluation metric (out-of-sample IC) is very noisy.
2. Overfitting to a specific historical period ("backtest fitting") is
   the primary cause of live signal decay.

Two safeguards are built in:
a. Successive halving: allocate a small training budget to many configs,
   then increase budget only for the promising survivors.  This prevents
   committing full compute to bad configs.
b. Holdout set: 20 % of data is withheld entirely and only used once
   at the very end to evaluate the chosen configuration.  This simulates
   live performance and detects in-sample overfitting.

Search spaces
-------------
LSTM        : hidden_size, n_layers, seq_len, lr, dropout (not used in
              weights directly but controls gradient scale)
Transformer : n_heads, d_model, n_layers, dropout, lr
XGBoost     : n_trees, lr, max_depth (stump only, so max_depth=1 always),
              row_sub, feat_sub
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------

LSTM_SPACE: Dict[str, List] = {
    "hidden_size": [32, 64, 128],
    "n_layers":    [1, 2, 3],
    "seq_len":     [10, 20, 30, 50],
    "lr":          [1e-4, 5e-4, 1e-3, 3e-3],
    "epochs":      [20, 40, 60],
}

TRANSFORMER_SPACE: Dict[str, List] = {
    "n_heads":    [2, 4, 8],
    "d_model":    [64, 128, 256],
    "n_layers":   [1, 2, 3],
    "lr":         [1e-4, 3e-4, 1e-3],
    "epochs":     [20, 40],
    "d_ff":       [128, 256, 512],
}

XGBOOST_SPACE: Dict[str, List] = {
    "n_trees":  [50, 100, 200, 300],
    "lr":       [0.01, 0.05, 0.1, 0.2],
    "row_sub":  [0.6, 0.7, 0.8, 0.9],
    "feat_sub": [0.5, 0.6, 0.8, 1.0],
}

SPACE_MAP: Dict[str, Dict] = {
    "LSTMSignal":        LSTM_SPACE,
    "TransformerSignal": TRANSFORMER_SPACE,
    "XGBoostSignal":     XGBOOST_SPACE,
}


@dataclass
class HyperoptResult:
    """Result of one hyperparameter trial."""
    trial_id:     int
    config:       Dict[str, Any]
    val_ic:       float
    val_icir:     float
    holdout_ic:   float = 0.0
    train_sec:    float = 0.0
    n_train:      int   = 0
    survived_sh:  bool  = False  # made it through successive halving
    extra:        Dict  = field(default_factory=dict)

    def __lt__(self, other: "HyperoptResult") -> bool:
        return self.val_ic < other.val_ic


# ---------------------------------------------------------------------------
# HyperoptSearch
# ---------------------------------------------------------------------------

class HyperoptSearch:
    """Random search + successive halving over model hyperparameters.

    Parameters
    ----------
    model_class : type
        One of LSTMSignal, TransformerSignal, XGBoostSignal.
    n_trials : int
        Total number of random configurations to draw.
    halving_rounds : int
        Number of successive halving rounds.  After each round, half the
        configurations are discarded and the survivors' training budget
        is doubled.
    val_frac : float
        Fraction of data to use as validation in each trial.
    holdout_frac : float
        Fraction of data reserved as a final holdout (never seen during search).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        model_class,
        n_trials:       int   = 50,
        halving_rounds: int   = 3,
        val_frac:       float = 0.2,
        holdout_frac:   float = 0.2,
        seed:           int   = 0,
    ) -> None:
        self.model_class    = model_class
        self.n_trials       = n_trials
        self.halving_rounds = halving_rounds
        self.val_frac       = val_frac
        self.holdout_frac   = holdout_frac
        self._rng           = np.random.default_rng(seed)
        self._results:      List[HyperoptResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, df: pd.DataFrame) -> HyperoptResult:
        """Run full hyperparameter search.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame with ``target`` column.

        Returns
        -------
        HyperoptResult  for the best configuration found on the holdout set.
        """
        n = len(df)
        holdout_start = int(n * (1.0 - self.holdout_frac))
        dev_df     = df.iloc[:holdout_start].copy()
        holdout_df = df.iloc[holdout_start:].copy()

        space = SPACE_MAP.get(self.model_class.__name__, {})
        configs = self._sample_configs(space, self.n_trials)

        # Phase 1: quick evaluation (1 epoch / small budget)
        survivors = self._evaluate_configs(configs, dev_df, budget=1)

        # Successive halving
        n_survive = max(1, len(survivors) // 2)
        for sh_round in range(self.halving_rounds):
            budget    = 2 ** (sh_round + 2)  # doubling budget
            survivors.sort(key=lambda r: -r.val_ic)
            survivors = survivors[:n_survive]
            survivors = self._evaluate_configs(
                [r.config for r in survivors], dev_df, budget=budget
            )
            n_survive = max(1, n_survive // 2)

        # Mark survivors
        for r in survivors:
            r.survived_sh = True
        self._results.extend(survivors)

        # Final evaluation on holdout for top survivor
        survivors.sort(key=lambda r: -r.val_ic)
        best_config = survivors[0].config if survivors else {}

        holdout_ic = self._holdout_eval(best_config, dev_df, holdout_df)
        best_result = survivors[0] if survivors else HyperoptResult(
            trial_id=0, config={}, val_ic=0.0, val_icir=0.0)
        best_result.holdout_ic = holdout_ic

        return best_result

    def best_config(self) -> Dict[str, Any]:
        """Return config with highest val_ic seen so far."""
        if not self._results:
            return {}
        return max(self._results, key=lambda r: r.val_ic).config

    @property
    def results(self) -> List[HyperoptResult]:
        return list(self._results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_configs(
        self, space: Dict[str, List], n: int
    ) -> List[Dict[str, Any]]:
        configs = []
        for _ in range(n):
            cfg = {k: self._rng.choice(v) for k, v in space.items()}
            # Convert numpy types to Python natives for JSON compatibility
            cfg = {k: (int(v) if isinstance(v, np.integer) else
                        float(v) if isinstance(v, np.floating) else v)
                   for k, v in cfg.items()}
            configs.append(cfg)
        return configs

    def _evaluate_configs(
        self,
        configs: List[Dict[str, Any]],
        dev_df:  pd.DataFrame,
        budget:  int,
    ) -> List[HyperoptResult]:
        n       = len(dev_df)
        val_n   = int(n * self.val_frac)
        train_df = dev_df.iloc[: n - val_n]
        val_df   = dev_df.iloc[n - val_n :]

        results = []
        for trial_i, cfg in enumerate(configs):
            t0 = time.perf_counter()
            try:
                kwargs = dict(cfg)
                # Override epochs with budget
                if "epochs" in kwargs:
                    kwargs["epochs"] = min(int(kwargs["epochs"]), budget * 5)
                elif hasattr(self.model_class, "epochs"):
                    kwargs["epochs"] = budget * 5

                model = self.model_class(**{
                    k: v for k, v in kwargs.items()
                    if k in self.model_class.__init__.__code__.co_varnames
                })
                model.fit(train_df)
                preds   = self._predict_series(model, val_df)
                actuals = val_df["target"].values[:len(preds)]

                from scipy.stats import spearmanr
                ic, _  = spearmanr(preds, actuals)
                ic      = float(ic) if not np.isnan(ic) else 0.0

                window = min(20, len(preds) // 2)
                ics = []
                for i in range(window, len(preds)):
                    ic_w, _ = spearmanr(preds[i - window:i], actuals[i - window:i])
                    if not np.isnan(ic_w):
                        ics.append(ic_w)
                ics_arr = np.array(ics) if ics else np.array([ic])
                icir    = float(ics_arr.mean() / (ics_arr.std() + 1e-9))

                results.append(HyperoptResult(
                    trial_id  = trial_i,
                    config    = cfg,
                    val_ic    = ic,
                    val_icir  = icir,
                    train_sec = time.perf_counter() - t0,
                    n_train   = len(train_df),
                ))
            except Exception:
                results.append(HyperoptResult(
                    trial_id=trial_i, config=cfg, val_ic=-1.0, val_icir=0.0))

        return results

    def _predict_series(self, model, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for i in range(len(df)):
            try:
                p = model.predict(df.iloc[:i + 1])
            except Exception:
                p = 0.0
            preds.append(float(p))
        return np.array(preds)

    def _holdout_eval(
        self,
        config:     Dict[str, Any],
        dev_df:     pd.DataFrame,
        holdout_df: pd.DataFrame,
    ) -> float:
        if not config or holdout_df.empty:
            return 0.0
        try:
            model = self.model_class(**{
                k: v for k, v in config.items()
                if k in self.model_class.__init__.__code__.co_varnames
            })
            model.fit(dev_df)
            preds   = self._predict_series(model, holdout_df)
            actuals = holdout_df["target"].values[:len(preds)]
            from scipy.stats import spearmanr
            ic, _   = spearmanr(preds, actuals)
            return float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0

    def summary_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising all evaluated configurations."""
        if not self._results:
            return pd.DataFrame()
        rows = []
        for r in sorted(self._results, key=lambda x: -x.val_ic):
            row = {"trial_id": r.trial_id, "val_ic": r.val_ic,
                   "val_icir": r.val_icir, "holdout_ic": r.holdout_ic,
                   "survived_sh": r.survived_sh, "train_sec": r.train_sec}
            row.update(r.config)
            rows.append(row)
        return pd.DataFrame(rows)
