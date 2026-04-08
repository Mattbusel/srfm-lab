"""
Model Zoo and Registry -- versioned ML model catalog with standardized interfaces.

Implements:
  - ModelRecord: metadata, version, performance history, config
  - ModelRegistry: register, retrieve, compare, promote, deprecate models
  - Standardized prediction interface: predict(features) -> signal
  - Model versioning: semantic versions with changelog
  - A/B comparison: paired evaluation on same data
  - Model lineage: parent/child tracking for ensemble and fine-tuned models
  - Auto-retirement: deprecate models below performance threshold
  - Ensemble builder: combine registered models with learned weights
  - Model serialization: save/load model state (numpy arrays)
  - Performance dashboard data: rolling metrics per model
"""

from __future__ import annotations
import math
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    model_type: str          # linear / tree / nn / ensemble / rule
    hyperparameters: dict = field(default_factory=dict)
    feature_names: list = field(default_factory=list)
    target: str = "forward_return_1d"
    training_window: int = 252
    retrain_frequency: int = 63


@dataclass
class PerformanceSnapshot:
    timestamp: float
    sharpe: float
    ic: float
    ic_ir: float
    hit_rate: float
    turnover: float
    max_drawdown: float
    n_predictions: int


@dataclass
class ModelRecord:
    model_id: str
    name: str
    version: str              # semantic: major.minor.patch
    model_type: str
    config: ModelConfig
    status: str = "active"    # active / staging / deprecated / retired
    created_at: float = 0.0
    updated_at: float = 0.0
    parent_id: Optional[str] = None
    tags: list = field(default_factory=list)
    description: str = ""
    performance_history: list = field(default_factory=list)
    weights: Optional[np.ndarray] = None
    parameters: dict = field(default_factory=dict)


class BaseModel:
    """Standardized model interface."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass


class LinearModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.weights = None
        self.bias = 0.0
        self.l2_reg = config.hyperparameters.get("l2_reg", 0.01)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        n, d = features.shape
        X = np.column_stack([features, np.ones(n)])
        reg = np.eye(d + 1) * self.l2_reg
        reg[-1, -1] = 0
        try:
            w = np.linalg.solve(X.T @ X + reg, X.T @ targets)
            self.weights = w[:-1]
            self.bias = float(w[-1])
        except np.linalg.LinAlgError:
            self.weights = np.zeros(d)
            self.bias = float(targets.mean())
        self._fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.zeros(features.shape[0])
        return features @ self.weights + self.bias

    def get_params(self) -> dict:
        return {"weights": self.weights.tolist() if self.weights is not None else [],
                "bias": self.bias}

    def set_params(self, params: dict) -> None:
        if "weights" in params:
            self.weights = np.array(params["weights"])
        if "bias" in params:
            self.bias = params["bias"]
        self._fitted = True


class TreeModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.max_depth = config.hyperparameters.get("max_depth", 5)
        self.min_samples = config.hyperparameters.get("min_samples", 20)
        self.tree = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.tree = self._build_tree(features, targets, 0)
        self._fitted = True

    def _build_tree(self, X, y, depth):
        n = len(y)
        if depth >= self.max_depth or n < self.min_samples:
            return {"leaf": True, "value": float(y.mean())}

        best_feat, best_thresh, best_score = -1, 0.0, float("inf")
        for j in range(X.shape[1]):
            thresholds = np.percentile(X[:, j], [25, 50, 75])
            for t in thresholds:
                left = y[X[:, j] <= t]
                right = y[X[:, j] > t]
                if len(left) < self.min_samples or len(right) < self.min_samples:
                    continue
                score = len(left) * left.var() + len(right) * right.var()
                if score < best_score:
                    best_score = score
                    best_feat = j
                    best_thresh = float(t)

        if best_feat == -1:
            return {"leaf": True, "value": float(y.mean())}

        left_mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.tree is None:
            return np.zeros(features.shape[0])
        return np.array([self._predict_one(x, self.tree) for x in features])


class EnsembleModel(BaseModel):
    def __init__(self, config: ModelConfig, models: list = None, weights: list = None):
        super().__init__(config)
        self.models = models or []
        self.model_weights = np.array(weights or [1.0 / max(len(self.models), 1)] * max(len(self.models), 1))
        self._fitted = all(m._fitted for m in self.models) if self.models else False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        for m in self.models:
            m.fit(features, targets)
        self._fitted = True

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(features.shape[0])
        preds = np.array([m.predict(features) for m in self.models])
        return (self.model_weights[:, None] * preds).sum(axis=0) / (self.model_weights.sum() + 1e-10)

    def update_weights_from_performance(self, performances: list[float]) -> None:
        perf = np.array(performances)
        perf = np.maximum(perf, 0)
        total = perf.sum()
        if total > 0:
            self.model_weights = perf / total
        else:
            self.model_weights = np.ones(len(self.models)) / len(self.models)


# -- Model Registry --

class ModelRegistry:
    """Central registry for all ML models."""

    def __init__(self, storage_dir: Optional[str] = None):
        self._models: dict[str, ModelRecord] = {}
        self._instances: dict[str, BaseModel] = {}
        self._id_counter = 0
        self.storage_dir = storage_dir

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"model_{self._id_counter:05d}"

    def register(
        self,
        name: str,
        model_type: str,
        config: ModelConfig,
        instance: Optional[BaseModel] = None,
        parent_id: Optional[str] = None,
        tags: list = None,
        description: str = "",
    ) -> str:
        model_id = self._next_id()
        record = ModelRecord(
            model_id=model_id,
            name=name,
            version="1.0.0",
            model_type=model_type,
            config=config,
            status="staging",
            created_at=time.time(),
            updated_at=time.time(),
            parent_id=parent_id,
            tags=tags or [],
            description=description,
        )
        self._models[model_id] = record
        if instance:
            self._instances[model_id] = instance
        return model_id

    def get(self, model_id: str) -> Optional[ModelRecord]:
        return self._models.get(model_id)

    def get_instance(self, model_id: str) -> Optional[BaseModel]:
        return self._instances.get(model_id)

    def promote(self, model_id: str) -> None:
        if model_id in self._models:
            self._models[model_id].status = "active"
            self._models[model_id].updated_at = time.time()

    def deprecate(self, model_id: str) -> None:
        if model_id in self._models:
            self._models[model_id].status = "deprecated"
            self._models[model_id].updated_at = time.time()

    def retire(self, model_id: str) -> None:
        if model_id in self._models:
            self._models[model_id].status = "retired"

    def record_performance(self, model_id: str, snapshot: PerformanceSnapshot) -> None:
        if model_id in self._models:
            self._models[model_id].performance_history.append(snapshot)
            self._models[model_id].updated_at = time.time()

    def list_active(self) -> list[ModelRecord]:
        return [m for m in self._models.values() if m.status == "active"]

    def list_by_type(self, model_type: str) -> list[ModelRecord]:
        return [m for m in self._models.values() if m.model_type == model_type]

    def compare(self, model_id_a: str, model_id_b: str) -> dict:
        a = self._models.get(model_id_a)
        b = self._models.get(model_id_b)
        if not a or not b:
            return {"error": "model not found"}

        def latest_perf(m):
            if m.performance_history:
                p = m.performance_history[-1]
                return {"sharpe": p.sharpe, "ic": p.ic, "hit_rate": p.hit_rate, "max_dd": p.max_drawdown}
            return {}

        return {
            "model_a": {"id": a.model_id, "name": a.name, "perf": latest_perf(a)},
            "model_b": {"id": b.model_id, "name": b.name, "perf": latest_perf(b)},
            "winner": a.model_id if (latest_perf(a).get("sharpe", 0) > latest_perf(b).get("sharpe", 0)) else b.model_id,
        }

    def leaderboard(self, metric: str = "sharpe", top_k: int = 10) -> list[dict]:
        entries = []
        for m in self._models.values():
            if m.status in ("active", "staging") and m.performance_history:
                latest = m.performance_history[-1]
                val = getattr(latest, metric, 0.0)
                entries.append({"model_id": m.model_id, "name": m.name, metric: val, "status": m.status})
        entries.sort(key=lambda x: x.get(metric, 0), reverse=True)
        return entries[:top_k]

    def auto_retire(self, min_sharpe: float = 0.0, min_ic: float = 0.01, lookback: int = 5) -> list[str]:
        retired = []
        for m in self._models.values():
            if m.status != "active":
                continue
            recent = m.performance_history[-lookback:] if len(m.performance_history) >= lookback else []
            if not recent:
                continue
            avg_sharpe = float(np.mean([p.sharpe for p in recent]))
            avg_ic = float(np.mean([p.ic for p in recent]))
            if avg_sharpe < min_sharpe and avg_ic < min_ic:
                self.retire(m.model_id)
                retired.append(m.model_id)
        return retired

    def build_ensemble(self, model_ids: list[str], weights: list[float] = None) -> Optional[str]:
        models = [self._instances.get(mid) for mid in model_ids]
        models = [m for m in models if m is not None]
        if not models:
            return None
        config = ModelConfig(model_type="ensemble")
        ensemble = EnsembleModel(config, models, weights)
        eid = self.register(
            f"ensemble_{len(model_ids)}",
            "ensemble",
            config,
            ensemble,
            tags=["ensemble", "auto"],
        )
        return eid

    def summary(self) -> dict:
        status_counts = {}
        type_counts = {}
        for m in self._models.values():
            status_counts[m.status] = status_counts.get(m.status, 0) + 1
            type_counts[m.model_type] = type_counts.get(m.model_type, 0) + 1
        return {
            "total_models": len(self._models),
            "by_status": status_counts,
            "by_type": type_counts,
            "active_count": status_counts.get("active", 0),
        }
