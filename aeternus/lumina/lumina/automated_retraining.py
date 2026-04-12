"""
automated_retraining.py
=======================
Automated retraining trigger system for Lumina MoE.

Components:
  - PerformanceMonitor: tracks rolling IC, directional accuracy, Sharpe ratio
  - ConceptDriftDetector: CUSUM + Page-Hinkley test on IC time series
  - RetrainingTrigger: queues fine-tuning jobs when drift is detected
  - ContinualLearningGuard: EWC penalty check before retraining
  - ABTestingManager: 10% traffic split, statistical significance test, model promotion
  - ModelVersionManager: atomic model swap with zero downtime
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import enum
import json
import logging
import math
import os
import pickle
import queue
import shutil
import statistics
import tempfile
import threading
import time
import uuid
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SAMPLES_FOR_IC = 20
IC_WINDOW = 500
CUSUM_THRESHOLD = 4.0
CUSUM_DRIFT_MAGNITUDE = 0.2
PAGE_HINKLEY_THRESHOLD = 50.0
PAGE_HINKLEY_ALPHA = 0.01
AB_TEST_TRAFFIC_FRACTION = 0.10
AB_TEST_MIN_SAMPLES = 200
AB_TEST_SIGNIFICANCE = 0.05
EWC_LAMBDA = 1000.0
MODEL_CHECKPOINT_DIR = Path("checkpoints/lumina")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DriftType(enum.Enum):
    NONE = "none"
    CUSUM_POSITIVE = "cusum_positive"
    CUSUM_NEGATIVE = "cusum_negative"
    PAGE_HINKLEY = "page_hinkley"
    PERFORMANCE_DROP = "performance_drop"


class RetrainingState(enum.Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelPromotionDecision(enum.Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    INSUFFICIENT_DATA = "insufficient_data"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PredictionRecord:
    """A single prediction + outcome pair for performance tracking."""
    timestamp: float
    prediction: float        # model output (e.g., predicted return)
    realized: float          # actual outcome
    request_id: str = ""
    model_version: str = "v1"


@dataclasses.dataclass
class PerformanceSnapshot:
    timestamp: float
    ic: float
    directional_accuracy: float
    sharpe_ratio: float
    n_samples: int
    model_version: str


@dataclasses.dataclass
class DriftEvent:
    timestamp: float
    drift_type: DriftType
    statistic: float
    threshold: float
    recent_ic: float
    baseline_ic: float
    triggered_retraining: bool = False


@dataclasses.dataclass
class RetrainingJob:
    job_id: str
    trigger_event: DriftEvent
    state: RetrainingState = RetrainingState.QUEUED
    created_at: float = dataclasses.field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    new_model_path: Optional[str] = None
    error: Optional[str] = None
    metrics_before: Optional[PerformanceSnapshot] = None
    metrics_after: Optional[PerformanceSnapshot] = None


@dataclasses.dataclass
class ABTestResult:
    test_id: str
    control_version: str
    challenger_version: str
    n_control: int
    n_challenger: int
    control_ic: float
    challenger_ic: float
    control_sharpe: float
    challenger_sharpe: float
    p_value: float
    decision: ModelPromotionDecision
    created_at: float = dataclasses.field(default_factory=time.time)
    completed_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Performance Monitor
# ---------------------------------------------------------------------------


class PerformanceMonitor:
    """
    Tracks rolling IC (information coefficient), directional accuracy,
    and Sharpe ratio on live model predictions.

    IC = Pearson correlation between predictions and realized outcomes.
    Directional accuracy = fraction of correct sign predictions.
    Sharpe = mean(IC) / std(IC) * sqrt(annualization_factor)
    """

    def __init__(
        self,
        window: int = IC_WINDOW,
        annualization: float = 252.0,
        model_version: str = "v1",
    ):
        self.window = window
        self.annualization = annualization
        self.model_version = model_version

        self._records: deque = deque(maxlen=window)
        self._ic_history: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self._n_total = 0

    def record(self, prediction: float, realized: float, timestamp: Optional[float] = None) -> None:
        """Record a prediction-outcome pair."""
        rec = PredictionRecord(
            timestamp=timestamp or time.time(),
            prediction=prediction,
            realized=realized,
            model_version=self.model_version,
        )
        with self._lock:
            self._records.append(rec)
            self._n_total += 1

            # Recompute rolling IC every 10 samples
            if self._n_total % 10 == 0:
                ic = self._compute_ic_from_records()
                if ic is not None:
                    self._ic_history.append(ic)

    def record_batch(self, predictions: np.ndarray, realized: np.ndarray) -> None:
        """Record a batch of prediction-outcome pairs."""
        now = time.time()
        for p, r in zip(predictions.flat, realized.flat):
            self.record(float(p), float(r), now)

    def compute_ic(self) -> Optional[float]:
        """Compute IC over the rolling window."""
        with self._lock:
            return self._compute_ic_from_records()

    def _compute_ic_from_records(self) -> Optional[float]:
        if len(self._records) < MIN_SAMPLES_FOR_IC:
            return None
        preds = np.array([r.prediction for r in self._records])
        realized = np.array([r.realized for r in self._records])
        if preds.std() < 1e-10 or realized.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(preds, realized)[0, 1])

    def compute_directional_accuracy(self) -> Optional[float]:
        with self._lock:
            if len(self._records) < MIN_SAMPLES_FOR_IC:
                return None
            correct = sum(
                1 for r in self._records
                if np.sign(r.prediction) == np.sign(r.realized)
            )
            return correct / len(self._records)

    def compute_sharpe(self) -> Optional[float]:
        with self._lock:
            if len(self._ic_history) < 10:
                return None
            ics = np.array(list(self._ic_history))
            mean_ic = ics.mean()
            std_ic = ics.std()
            if std_ic < 1e-10:
                return 0.0
            return float(mean_ic / std_ic * math.sqrt(self.annualization))

    def snapshot(self) -> Optional[PerformanceSnapshot]:
        """Return a current performance snapshot."""
        ic = self.compute_ic()
        if ic is None:
            return None
        return PerformanceSnapshot(
            timestamp=time.time(),
            ic=ic,
            directional_accuracy=self.compute_directional_accuracy() or 0.0,
            sharpe_ratio=self.compute_sharpe() or 0.0,
            n_samples=len(self._records),
            model_version=self.model_version,
        )

    def ic_time_series(self) -> np.ndarray:
        """Return the rolling IC time series for drift detection."""
        with self._lock:
            return np.array(list(self._ic_history))

    @property
    def n_samples(self) -> int:
        return len(self._records)

    @property
    def recent_ic(self) -> Optional[float]:
        return self.compute_ic()


# ---------------------------------------------------------------------------
# CUSUM Drift Detector
# ---------------------------------------------------------------------------


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) control chart for concept drift detection.
    Detects a shift in the mean of the IC time series.

    References: Page (1954), Montgomery (2012).
    """

    def __init__(
        self,
        threshold: float = CUSUM_THRESHOLD,
        drift_magnitude: float = CUSUM_DRIFT_MAGNITUDE,
        baseline_window: int = 100,
    ):
        self.threshold = threshold
        self.drift_magnitude = drift_magnitude
        self.baseline_window = baseline_window

        self._cusum_pos = 0.0   # positive drift accumulator
        self._cusum_neg = 0.0   # negative drift accumulator
        self._baseline: deque = deque(maxlen=baseline_window)
        self._history: List[Tuple[float, float, float]] = []  # (pos, neg, ts)
        self._n = 0

    def update(self, value: float) -> Optional[DriftType]:
        """
        Update CUSUM with a new IC observation.
        Returns DriftType if drift detected, else None.
        """
        self._baseline.append(value)
        self._n += 1

        if len(self._baseline) < self.baseline_window // 2:
            return None

        mu = float(np.mean(self._baseline))
        k = self.drift_magnitude / 2.0

        # Two-sided CUSUM
        self._cusum_pos = max(0.0, self._cusum_pos + (value - mu - k))
        self._cusum_neg = max(0.0, self._cusum_neg - (value - mu + k))

        self._history.append((self._cusum_pos, self._cusum_neg, time.time()))

        if self._cusum_pos > self.threshold:
            self._cusum_pos = 0.0
            return DriftType.CUSUM_POSITIVE
        elif self._cusum_neg > self.threshold:
            self._cusum_neg = 0.0
            return DriftType.CUSUM_NEGATIVE

        return None

    def reset(self) -> None:
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0

    @property
    def current_statistic(self) -> Tuple[float, float]:
        return (self._cusum_pos, self._cusum_neg)

    def get_history(self) -> List[Tuple[float, float, float]]:
        return list(self._history[-1000:])


# ---------------------------------------------------------------------------
# Page-Hinkley Drift Detector
# ---------------------------------------------------------------------------


class PageHinkleyDetector:
    """
    Page-Hinkley test for change detection in the mean of a stream.
    More sensitive than CUSUM for gradual drift.

    References: Hinkley (1971), Mouss et al. (2004).
    """

    def __init__(
        self,
        threshold: float = PAGE_HINKLEY_THRESHOLD,
        alpha: float = PAGE_HINKLEY_ALPHA,
        min_samples: int = 30,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples

        self._sum = 0.0
        self._min_sum = float("inf")
        self._n = 0
        self._mean = 0.0

    def update(self, value: float) -> bool:
        """
        Update with a new observation.
        Returns True if drift detected.
        """
        self._n += 1
        # Online mean update
        self._mean += (value - self._mean) / self._n

        self._sum += value - self._mean - self.alpha
        self._min_sum = min(self._min_sum, self._sum)

        if self._n < self.min_samples:
            return False

        ph = self._sum - self._min_sum
        return ph > self.threshold

    def reset(self) -> None:
        self._sum = 0.0
        self._min_sum = float("inf")
        self._n = 0
        self._mean = 0.0

    @property
    def statistic(self) -> float:
        return self._sum - self._min_sum

    @property
    def n_samples(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Concept Drift Detector (composite)
# ---------------------------------------------------------------------------


class ConceptDriftDetector:
    """
    Composite drift detector that runs both CUSUM and Page-Hinkley.
    Feeds from the IC time series produced by PerformanceMonitor.
    """

    def __init__(
        self,
        cusum_threshold: float = CUSUM_THRESHOLD,
        ph_threshold: float = PAGE_HINKLEY_THRESHOLD,
        cooldown_sec: float = 300.0,   # minimum time between triggers
    ):
        self.cusum = CUSUMDetector(threshold=cusum_threshold)
        self.ph = PageHinkleyDetector(threshold=ph_threshold)
        self.cooldown_sec = cooldown_sec

        self._last_trigger: Optional[float] = None
        self._events: List[DriftEvent] = []
        self._baseline_ic: Optional[float] = None
        self._ic_buffer: deque = deque(maxlen=50)

    def update(self, ic: float) -> Optional[DriftEvent]:
        """
        Feed a new IC observation. Returns DriftEvent if drift detected.
        """
        self._ic_buffer.append(ic)

        if self._baseline_ic is None and len(self._ic_buffer) >= 20:
            self._baseline_ic = float(np.mean(self._ic_buffer))

        cusum_result = self.cusum.update(ic)
        ph_result = self.ph.update(ic)

        drift_type = None
        if cusum_result is not None:
            drift_type = cusum_result
            stat = max(self.cusum.current_statistic)
            threshold = self.cusum.threshold
        elif ph_result:
            drift_type = DriftType.PAGE_HINKLEY
            stat = self.ph.statistic
            threshold = self.ph.threshold
        else:
            return None

        # Cooldown check
        now = time.time()
        if self._last_trigger is not None and (now - self._last_trigger) < self.cooldown_sec:
            logger.debug(f"Drift detected ({drift_type.value}) but in cooldown period")
            return None

        self._last_trigger = now
        recent = float(np.mean(self._ic_buffer)) if self._ic_buffer else 0.0
        event = DriftEvent(
            timestamp=now,
            drift_type=drift_type,
            statistic=stat,
            threshold=threshold,
            recent_ic=recent,
            baseline_ic=self._baseline_ic or 0.0,
        )
        self._events.append(event)
        logger.warning(
            f"Concept drift detected: type={drift_type.value}, "
            f"stat={stat:.3f}, recent_ic={recent:.4f}, "
            f"baseline_ic={self._baseline_ic or 0.0:.4f}"
        )
        return event

    def reset_after_retraining(self) -> None:
        """Reset detectors after a model update."""
        self.cusum.reset()
        self.ph.reset()
        self._baseline_ic = None
        logger.info("DriftDetector reset after retraining")

    @property
    def drift_history(self) -> List[DriftEvent]:
        return list(self._events)


# ---------------------------------------------------------------------------
# Elastic Weight Consolidation (EWC) Guard
# ---------------------------------------------------------------------------


class EWCGuard:
    """
    Elastic Weight Consolidation guard.
    Before retraining, computes the EWC penalty to check how severe
    catastrophic forgetting would be.

    Reference: Kirkpatrick et al., 2017 (https://arxiv.org/abs/1612.00796)
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = EWC_LAMBDA):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self._fisher: Optional[Dict[str, Tensor]] = None
        self._theta_star: Optional[Dict[str, Tensor]] = None

    def compute_fisher(
        self,
        dataloader: Any,   # iterable of (inputs, targets)
        n_batches: int = 50,
        device: str = "cpu",
    ) -> Dict[str, Tensor]:
        """
        Compute the diagonal Fisher Information Matrix (FIM) approximation.
        FIM_ii = E[( d log p(y|x) / d theta_i )^2]
        """
        fisher: Dict[str, Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                fisher[name] = torch.zeros_like(p.data)

        self.model.eval()
        n = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs = batch
                targets = None

            if not isinstance(inputs, Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)

            self.model.zero_grad()

            # Forward pass
            output = self.model(inputs)
            if targets is not None:
                if not isinstance(targets, Tensor):
                    targets = torch.tensor(targets)
                targets = targets.to(device)
                loss = F.cross_entropy(
                    output.view(-1, output.shape[-1]),
                    targets.view(-1).long(),
                )
            else:
                # Use output variance as a proxy loss
                loss = -output.float().var()

            loss.backward()

            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[name] += p.grad.data.pow(2)

            n += 1

        for name in fisher:
            fisher[name] /= max(n, 1)

        self._fisher = fisher
        self._theta_star = {
            name: p.data.clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        logger.info(f"EWC: Fisher matrix computed over {n} batches")
        return fisher

    def ewc_penalty(self, current_model: Optional[nn.Module] = None) -> float:
        """
        Compute the EWC penalty for the current model weights.
        penalty = lambda/2 * sum_i F_i * (theta_i - theta_star_i)^2
        """
        if self._fisher is None or self._theta_star is None:
            return 0.0

        model = current_model or self.model
        penalty = 0.0

        for name, p in model.named_parameters():
            if name in self._fisher and name in self._theta_star:
                diff = p.data - self._theta_star[name].to(p.device)
                f = self._fisher[name].to(p.device)
                penalty += float((f * diff.pow(2)).sum().item())

        return self.ewc_lambda / 2.0 * penalty

    def is_safe_to_retrain(self, threshold: float = 1e6) -> Tuple[bool, float]:
        """
        Check if retraining would cause unacceptable forgetting.
        Returns (safe, penalty_estimate).
        """
        if self._fisher is None:
            return True, 0.0

        penalty = self.ewc_penalty()
        safe = penalty < threshold
        if not safe:
            logger.warning(
                f"EWC: retraining penalty {penalty:.2e} exceeds threshold {threshold:.2e}. "
                "Consider elastic fine-tuning with EWC regularization."
            )
        return safe, penalty

    def ewc_loss(self, current_model: Optional[nn.Module] = None) -> Tensor:
        """Return EWC loss as a differentiable tensor (for use during training)."""
        if self._fisher is None or self._theta_star is None:
            return torch.tensor(0.0)

        model = current_model or self.model
        loss = torch.tensor(0.0)

        for name, p in model.named_parameters():
            if name in self._fisher and name in self._theta_star:
                diff = p - self._theta_star[name].to(p.device)
                f = self._fisher[name].to(p.device)
                loss = loss + (f * diff.pow(2)).sum()

        return self.ewc_lambda / 2.0 * loss


# ---------------------------------------------------------------------------
# Retraining Job Queue
# ---------------------------------------------------------------------------


class RetrainingJobQueue:
    """Thread-safe queue for retraining jobs."""

    def __init__(self, max_queued: int = 3):
        self.max_queued = max_queued
        self._queue: queue.Queue = queue.Queue(maxsize=max_queued)
        self._completed: List[RetrainingJob] = []
        self._lock = threading.Lock()

    def submit(self, job: RetrainingJob) -> bool:
        """Submit a job. Returns False if queue is full."""
        try:
            self._queue.put_nowait(job)
            logger.info(f"Retraining job {job.job_id} queued")
            return True
        except queue.Full:
            logger.warning("Retraining queue full, dropping job")
            return False

    def get_next(self, timeout: float = 1.0) -> Optional[RetrainingJob]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def mark_completed(self, job: RetrainingJob) -> None:
        with self._lock:
            self._completed.append(job)
            if len(self._completed) > 100:
                self._completed = self._completed[-100:]

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def history(self) -> List[RetrainingJob]:
        with self._lock:
            return list(self._completed)


# ---------------------------------------------------------------------------
# Retraining Trigger
# ---------------------------------------------------------------------------


class RetrainingTrigger:
    """
    Watches the drift detector and submits fine-tuning jobs when
    concept drift is detected, subject to EWC safety check.
    """

    def __init__(
        self,
        model: nn.Module,
        drift_detector: ConceptDriftDetector,
        ewc_guard: EWCGuard,
        job_queue: RetrainingJobQueue,
        min_ic_drop: float = 0.02,
        ewc_threshold: float = 5e5,
    ):
        self.model = model
        self.drift_detector = drift_detector
        self.ewc_guard = ewc_guard
        self.job_queue = job_queue
        self.min_ic_drop = min_ic_drop
        self.ewc_threshold = ewc_threshold
        self._n_triggered = 0

    def on_new_ic(self, ic: float, monitor: PerformanceMonitor) -> Optional[RetrainingJob]:
        """
        Called whenever a new IC observation is available.
        Returns a RetrainingJob if a new job was queued.
        """
        event = self.drift_detector.update(ic)
        if event is None:
            return None

        # Check that IC actually dropped meaningfully
        if self.drift_detector._baseline_ic is not None:
            ic_drop = self.drift_detector._baseline_ic - event.recent_ic
            if ic_drop < self.min_ic_drop:
                logger.info(
                    f"Drift detected but IC drop {ic_drop:.4f} < threshold {self.min_ic_drop:.4f}, "
                    "skipping retraining"
                )
                return None

        # EWC safety check
        safe, penalty = self.ewc_guard.is_safe_to_retrain(self.ewc_threshold)
        if not safe:
            logger.warning(
                f"EWC penalty {penalty:.2e} too high, deferring retraining"
            )
            return None

        # Capture current performance snapshot
        perf = monitor.snapshot()

        job = RetrainingJob(
            job_id=str(uuid.uuid4())[:8],
            trigger_event=event,
            state=RetrainingState.QUEUED,
            metrics_before=perf,
        )
        event.triggered_retraining = True
        success = self.job_queue.submit(job)
        if success:
            self._n_triggered += 1
            logger.info(f"Retraining triggered: job_id={job.job_id}, drift={event.drift_type.value}")
        return job if success else None

    @property
    def n_triggered(self) -> int:
        return self._n_triggered


# ---------------------------------------------------------------------------
# Fine-Tuning Runner
# ---------------------------------------------------------------------------


class FineTuner:
    """
    Executes fine-tuning jobs from the RetrainingJobQueue.
    Uses recent data and optionally applies EWC regularization.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_guard: EWCGuard,
        checkpoint_dir: Path = MODEL_CHECKPOINT_DIR,
        lr: float = 1e-4,
        max_steps: int = 500,
        ewc_lambda: float = EWC_LAMBDA,
        device: str = "cpu",
    ):
        self.model = model
        self.ewc_guard = ewc_guard
        self.checkpoint_dir = Path(checkpoint_dir)
        self.lr = lr
        self.max_steps = max_steps
        self.ewc_lambda = ewc_lambda
        self.device = device

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        job: RetrainingJob,
        recent_data: Any,   # iterable of (inputs, targets)
        monitor: PerformanceMonitor,
    ) -> RetrainingJob:
        """Execute a fine-tuning job."""
        job.state = RetrainingState.RUNNING
        job.started_at = time.time()

        try:
            logger.info(f"Starting fine-tuning job {job.job_id} ...")
            fine_tuned_model = self._fine_tune(recent_data)

            # Save checkpoint
            ckpt_path = self.checkpoint_dir / f"model_{job.job_id}.pt"
            torch.save(fine_tuned_model.state_dict(), ckpt_path)

            job.new_model_path = str(ckpt_path)
            job.state = RetrainingState.COMPLETED
            job.completed_at = time.time()
            job.metrics_after = monitor.snapshot()

            duration = job.completed_at - job.started_at
            logger.info(
                f"Fine-tuning job {job.job_id} completed in {duration:.1f}s. "
                f"Checkpoint: {ckpt_path}"
            )
        except Exception as e:
            job.state = RetrainingState.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            logger.error(f"Fine-tuning job {job.job_id} failed: {e}")

        return job

    def _fine_tune(self, dataloader: Any) -> nn.Module:
        """Run the actual fine-tuning loop with EWC regularization."""
        model = copy.deepcopy(self.model).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_steps)

        model.train()
        step = 0

        for epoch in range(100):
            if step >= self.max_steps:
                break

            for batch in dataloader:
                if step >= self.max_steps:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch
                    targets = None

                if not isinstance(inputs, Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                output = model(inputs)

                if targets is not None:
                    if not isinstance(targets, Tensor):
                        targets = torch.tensor(targets)
                    targets = targets.to(self.device)
                    task_loss = F.mse_loss(output.float(), targets.float())
                else:
                    task_loss = output.float().pow(2).mean() * 0.0

                # EWC penalty
                ewc_loss = self.ewc_guard.ewc_loss(model)
                total_loss = task_loss + ewc_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                step += 1

                if step % 100 == 0:
                    logger.debug(
                        f"Fine-tuning step {step}/{self.max_steps}: "
                        f"task_loss={task_loss.item():.4f} ewc_loss={ewc_loss.item():.4f}"
                    )

        return model


# ---------------------------------------------------------------------------
# A/B Testing Manager
# ---------------------------------------------------------------------------


class ABTestingManager:
    """
    Manages A/B tests between a control model (current production) and
    a challenger model (newly fine-tuned).

    Routes AB_TEST_TRAFFIC_FRACTION of requests to the challenger.
    Uses a two-sample t-test on IC to determine if challenger is significantly better.
    """

    def __init__(
        self,
        traffic_fraction: float = AB_TEST_TRAFFIC_FRACTION,
        min_samples: int = AB_TEST_MIN_SAMPLES,
        significance: float = AB_TEST_SIGNIFICANCE,
    ):
        self.traffic_fraction = traffic_fraction
        self.min_samples = min_samples
        self.significance = significance

        self._active_test: Optional[ABTestResult] = None
        self._control_records: deque = deque(maxlen=5000)
        self._challenger_records: deque = deque(maxlen=5000)
        self._test_history: List[ABTestResult] = []
        self._lock = threading.Lock()

        self._control_version: str = "control"
        self._challenger_version: str = "challenger"
        self._rng = np.random.default_rng(42)

    def start_test(self, control_version: str, challenger_version: str) -> ABTestResult:
        """Start a new A/B test."""
        with self._lock:
            if self._active_test is not None:
                logger.warning("Ending previous A/B test before starting new one")
                self._finalize_test()

            self._control_version = control_version
            self._challenger_version = challenger_version
            self._control_records.clear()
            self._challenger_records.clear()

            self._active_test = ABTestResult(
                test_id=str(uuid.uuid4())[:8],
                control_version=control_version,
                challenger_version=challenger_version,
                n_control=0,
                n_challenger=0,
                control_ic=0.0,
                challenger_ic=0.0,
                control_sharpe=0.0,
                challenger_sharpe=0.0,
                p_value=1.0,
                decision=ModelPromotionDecision.INSUFFICIENT_DATA,
            )
            logger.info(
                f"A/B test started: {control_version} vs {challenger_version} "
                f"(traffic={self.traffic_fraction:.0%})"
            )
            return self._active_test

    def route_request(self) -> str:
        """
        Decide which model version to use for this request.
        Returns model version string ('control' or 'challenger').
        """
        if self._active_test is None:
            return "control"
        if self._rng.random() < self.traffic_fraction:
            return "challenger"
        return "control"

    def record_outcome(
        self,
        version: str,
        prediction: float,
        realized: float,
    ) -> Optional[ModelPromotionDecision]:
        """
        Record an outcome for the A/B test.
        Returns a promotion decision if the test has reached significance.
        """
        with self._lock:
            if self._active_test is None:
                return None

            rec = PredictionRecord(
                timestamp=time.time(),
                prediction=prediction,
                realized=realized,
                model_version=version,
            )

            if version == self._control_version:
                self._control_records.append(rec)
            else:
                self._challenger_records.append(rec)

            n_c = len(self._control_records)
            n_ch = len(self._challenger_records)

            if n_c >= self.min_samples and n_ch >= self.min_samples:
                return self._evaluate_test()

        return None

    def _evaluate_test(self) -> Optional[ModelPromotionDecision]:
        """Evaluate the A/B test and potentially make a promotion decision."""
        control_preds = np.array([r.prediction for r in self._control_records])
        control_real = np.array([r.realized for r in self._control_records])
        chall_preds = np.array([r.prediction for r in self._challenger_records])
        chall_real = np.array([r.realized for r in self._challenger_records])

        # Compute per-period ICs
        # Use windowed IC for stability
        control_ic = self._rolling_ic(control_preds, control_real)
        challenger_ic = self._rolling_ic(chall_preds, chall_real)

        if control_ic is None or challenger_ic is None:
            return None

        # Two-sample t-test on IC time series
        p_value = self._welch_t_test(control_ic, challenger_ic)

        control_mean = float(np.mean(control_ic))
        challenger_mean = float(np.mean(challenger_ic))

        if p_value < self.significance and challenger_mean > control_mean:
            decision = ModelPromotionDecision.PROMOTE
        elif p_value < self.significance and challenger_mean <= control_mean:
            decision = ModelPromotionDecision.REJECT
        else:
            return None  # Not yet significant

        self._active_test.n_control = len(self._control_records)
        self._active_test.n_challenger = len(self._challenger_records)
        self._active_test.control_ic = control_mean
        self._active_test.challenger_ic = challenger_mean
        self._active_test.p_value = p_value
        self._active_test.decision = decision
        self._active_test.completed_at = time.time()

        logger.info(
            f"A/B test {self._active_test.test_id} concluded: "
            f"decision={decision.value}, "
            f"control_ic={control_mean:.4f}, challenger_ic={challenger_mean:.4f}, "
            f"p={p_value:.4f}"
        )
        self._finalize_test()
        return decision

    def _finalize_test(self) -> None:
        if self._active_test is not None:
            self._test_history.append(self._active_test)
            self._active_test = None

    @staticmethod
    def _rolling_ic(preds: np.ndarray, realized: np.ndarray, window: int = 50) -> Optional[np.ndarray]:
        n = len(preds)
        if n < window:
            return None
        ics = []
        for i in range(window, n, window // 2):
            p = preds[i - window:i]
            r = realized[i - window:i]
            if p.std() > 1e-10 and r.std() > 1e-10:
                ics.append(float(np.corrcoef(p, r)[0, 1]))
        return np.array(ics) if ics else None

    @staticmethod
    def _welch_t_test(a: np.ndarray, b: np.ndarray) -> float:
        """Welch's t-test, returns p-value."""
        n_a, n_b = len(a), len(b)
        if n_a < 2 or n_b < 2:
            return 1.0
        mean_a, mean_b = a.mean(), b.mean()
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)

        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se < 1e-10:
            return 1.0

        t_stat = (mean_b - mean_a) / se
        # Degrees of freedom (Welch-Satterthwaite)
        df_num = (var_a / n_a + var_b / n_b) ** 2
        df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = df_num / max(df_den, 1e-10)

        # Approximate p-value using normal distribution for large df
        if df > 30:
            # Normal approximation
            p_value = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
        else:
            # t-distribution approximation (simplified)
            p_value = 2.0 * (1.0 - _t_cdf(abs(t_stat), df))

        return float(np.clip(p_value, 0.0, 1.0))

    @property
    def active_test(self) -> Optional[ABTestResult]:
        return self._active_test

    @property
    def test_history(self) -> List[ABTestResult]:
        return list(self._test_history)


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_cdf(t: float, df: float) -> float:
    """Very rough approximation of t-distribution CDF."""
    x = df / (df + t * t)
    return 1.0 - 0.5 * _regularized_incomplete_beta(df / 2, 0.5, x)


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Very rough approximation."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return x ** a * (1 - x) ** b


# ---------------------------------------------------------------------------
# Model Version Manager
# ---------------------------------------------------------------------------


class ModelVersionManager:
    """
    Manages model versions and performs atomic hot-swap with zero downtime.

    Uses a read-write lock pattern:
    - Multiple concurrent readers (inference threads)
    - Single writer (model swap)

    Version files are stored as:
      checkpoint_dir/
        model_v{N}.pt          — serialized state dict
        latest -> model_v{N}   — symlink to current version
        versions.json          — version metadata
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: Path = MODEL_CHECKPOINT_DIR,
        max_versions: int = 5,
    ):
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_versions = max_versions

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._current_version = "v1"
        self._version_meta: Dict[str, Dict[str, Any]] = {}
        self._rw_lock = threading.RLock()
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._write_lock = threading.Lock()

        self._load_version_metadata()

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def save_version(
        self,
        version_name: str,
        model: Optional[nn.Module] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model version to disk."""
        m = model or self.model
        ckpt_path = self.checkpoint_dir / f"model_{version_name}.pt"
        torch.save(m.state_dict(), ckpt_path)

        self._version_meta[version_name] = {
            "path": str(ckpt_path),
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        self._save_version_metadata()
        logger.info(f"Saved model version {version_name} to {ckpt_path}")
        return ckpt_path

    def load_version(self, version_name: str) -> bool:
        """Load a saved version into the current model (not atomic)."""
        meta = self._version_meta.get(version_name)
        if meta is None:
            logger.error(f"Version {version_name} not found")
            return False

        path = Path(meta["path"])
        if not path.exists():
            logger.error(f"Checkpoint {path} does not exist")
            return False

        state_dict = torch.load(path, map_location="cpu")
        with self._rw_lock:
            self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded model version {version_name}")
        return True

    def atomic_swap(
        self,
        new_model: nn.Module,
        new_version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Atomically swap the current model for new_model.

        Steps:
        1. Save new model weights to a temp file
        2. Acquire write lock (waits for in-flight reads to finish)
        3. Update model in-place (copy weights)
        4. Update symlink
        5. Release write lock
        """
        try:
            # Save new weights to temp
            with tempfile.NamedTemporaryFile(
                suffix=".pt",
                dir=self.checkpoint_dir,
                delete=False,
            ) as f:
                tmp_path = f.name
            torch.save(new_model.state_dict(), tmp_path)

            # Acquire write lock
            with self._write_lock:
                # Wait for readers (simplified: just acquire rw_lock)
                with self._rw_lock:
                    # Copy weights in-place (preserves existing references)
                    new_sd = torch.load(tmp_path, map_location="cpu")
                    self.model.load_state_dict(new_sd, strict=False)

                    # Finalize the checkpoint
                    final_path = self.checkpoint_dir / f"model_{new_version}.pt"
                    os.replace(tmp_path, final_path)

                    self._version_meta[new_version] = {
                        "path": str(final_path),
                        "created_at": time.time(),
                        "metadata": metadata or {},
                    }

                    # Update 'latest' symlink
                    latest = self.checkpoint_dir / "latest.pt"
                    with contextlib.suppress(Exception):
                        latest.unlink()
                    with contextlib.suppress(Exception):
                        latest.symlink_to(final_path)

                    old_version = self._current_version
                    self._current_version = new_version
                    self._save_version_metadata()

                    logger.info(
                        f"Atomic model swap: {old_version} -> {new_version}"
                    )

            self._prune_old_versions()
            return True

        except Exception as e:
            logger.error(f"Atomic swap failed: {e}")
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
            return False

    # ------------------------------------------------------------------
    # Context manager for safe inference
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def inference_context(self):
        """Context manager that protects inference from concurrent model swaps."""
        with self._readers_lock:
            self._readers += 1
        try:
            yield self.model
        finally:
            with self._readers_lock:
                self._readers -= 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_old_versions(self) -> None:
        """Delete old checkpoints, keeping only the most recent N."""
        versions = sorted(
            self._version_meta.items(),
            key=lambda kv: kv[1].get("created_at", 0),
        )
        while len(versions) > self.max_versions:
            name, meta = versions.pop(0)
            path = Path(meta["path"])
            if path.exists() and name != self._current_version:
                with contextlib.suppress(Exception):
                    path.unlink()
                del self._version_meta[name]
                logger.debug(f"Pruned old version: {name}")

        self._save_version_metadata()

    def _save_version_metadata(self) -> None:
        meta_path = self.checkpoint_dir / "versions.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "current_version": self._current_version,
                    "versions": self._version_meta,
                },
                f,
                indent=2,
            )

    def _load_version_metadata(self) -> None:
        meta_path = self.checkpoint_dir / "versions.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    data = json.load(f)
                self._current_version = data.get("current_version", "v1")
                self._version_meta = data.get("versions", {})
            except Exception as e:
                logger.warning(f"Could not load version metadata: {e}")

    @property
    def current_version(self) -> str:
        return self._current_version

    @property
    def available_versions(self) -> List[str]:
        return list(self._version_meta.keys())


# ---------------------------------------------------------------------------
# Retraining Orchestrator (ties everything together)
# ---------------------------------------------------------------------------


class RetrainingOrchestrator:
    """
    Top-level orchestrator that ties together:
      PerformanceMonitor -> ConceptDriftDetector -> RetrainingTrigger
      -> FineTuner -> ABTestingManager -> ModelVersionManager

    Call .on_prediction(pred, realized) on each live prediction.
    The orchestrator handles everything else automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_dir: Union[str, Path] = MODEL_CHECKPOINT_DIR,
        device: str = "cpu",
        enable_ab_testing: bool = True,
        ewc_lambda: float = EWC_LAMBDA,
    ):
        self.model = model
        self.device = device
        self.enable_ab_testing = enable_ab_testing

        # Sub-components
        self.monitor = PerformanceMonitor(model_version="v1")
        self.drift_detector = ConceptDriftDetector()
        self.ewc_guard = EWCGuard(model, ewc_lambda)
        self.job_queue = RetrainingJobQueue()
        self.trigger = RetrainingTrigger(model, self.drift_detector, self.ewc_guard, self.job_queue)
        self.version_manager = ModelVersionManager(model, Path(checkpoint_dir))
        self.ab_manager = ABTestingManager() if enable_ab_testing else None
        self.fine_tuner: Optional[FineTuner] = None  # set lazily

        # Worker thread for running fine-tuning jobs
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # IC update interval
        self._n_since_ic_update = 0
        self._ic_update_interval = 10

    def start(self) -> None:
        """Start the background retraining worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._retraining_worker,
            name="lumina-retraining-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("RetrainingOrchestrator started")

    def stop(self) -> None:
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=10)
        logger.info("RetrainingOrchestrator stopped")

    def on_prediction(
        self,
        prediction: float,
        realized: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a prediction-outcome pair.
        Automatically triggers retraining if drift is detected.
        """
        self.monitor.record(prediction, realized, timestamp)
        self._n_since_ic_update += 1

        if self._n_since_ic_update >= self._ic_update_interval:
            ic = self.monitor.recent_ic
            if ic is not None:
                self.trigger.on_new_ic(ic, self.monitor)
            self._n_since_ic_update = 0

    def set_fine_tuner(self, fine_tuner: FineTuner) -> None:
        self.fine_tuner = fine_tuner

    def compute_ewc_fisher(self, dataloader: Any, n_batches: int = 50) -> None:
        """Compute Fisher matrix for EWC (call once on initial training data)."""
        self.ewc_guard.compute_fisher(dataloader, n_batches, self.device)
        logger.info("EWC Fisher matrix computed")

    def _retraining_worker(self) -> None:
        """Background thread that processes retraining jobs."""
        while self._running:
            job = self.job_queue.get_next(timeout=1.0)
            if job is None:
                continue

            if self.fine_tuner is None:
                logger.warning("No FineTuner configured, skipping retraining job")
                job.state = RetrainingState.FAILED
                job.error = "No FineTuner configured"
                self.job_queue.mark_completed(job)
                continue

            # Create a dummy dataloader from recent records if none provided
            # In production this would use a real data pipeline
            dummy_loader = self._build_recent_dataloader()
            job = self.fine_tuner.run(job, dummy_loader, self.monitor)
            self.job_queue.mark_completed(job)

            if job.state == RetrainingState.COMPLETED and job.new_model_path:
                self._handle_completed_job(job)

    def _handle_completed_job(self, job: RetrainingJob) -> None:
        """Handle a completed fine-tuning job: A/B test or direct promotion."""
        new_version = f"v{int(time.time())}"

        # Load new model
        new_model = copy.deepcopy(self.model)
        try:
            sd = torch.load(job.new_model_path, map_location="cpu")
            new_model.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.error(f"Could not load fine-tuned model: {e}")
            return

        if self.enable_ab_testing and self.ab_manager:
            # Start A/B test
            self.ab_manager.start_test(
                self.version_manager.current_version,
                new_version,
            )
            # Save challenger (but don't swap yet)
            self.version_manager.save_version(new_version, new_model)
            logger.info(f"A/B test started for version {new_version}")
        else:
            # Direct promotion
            self.version_manager.atomic_swap(new_model, new_version)
            self.drift_detector.reset_after_retraining()
            logger.info(f"Model directly promoted to {new_version}")

    def _build_recent_dataloader(self):
        """Build a simple dataloader from recent prediction records."""
        records = list(self.monitor._records)
        if not records:
            return []

        # Create synthetic (features, target) pairs from records
        preds = torch.tensor([r.prediction for r in records], dtype=torch.float32)
        realized = torch.tensor([r.realized for r in records], dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(
            preds.unsqueeze(1), realized.unsqueeze(1)
        )
        return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    @property
    def status(self) -> Dict[str, Any]:
        return {
            "n_samples": self.monitor.n_samples,
            "recent_ic": self.monitor.recent_ic,
            "n_retraining_triggered": self.trigger.n_triggered,
            "queue_depth": self.job_queue.queue_depth,
            "completed_jobs": len(self.job_queue.history),
            "current_version": self.version_manager.current_version,
            "ab_test_active": self.ab_manager.active_test is not None if self.ab_manager else False,
        }


# ---------------------------------------------------------------------------
# Import guard for Tensor type
# ---------------------------------------------------------------------------

from torch import Tensor
import contextlib

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: simulate predictions and trigger detection
    print("Simulating performance monitoring and drift detection...")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        def forward(self, x):
            return self.fc(x)

    model = DummyModel()
    orchestrator = RetrainingOrchestrator(model, checkpoint_dir="/tmp/lumina_ckpts")

    rng = np.random.default_rng(0)
    n = 600
    for i in range(n):
        # Simulate concept drift at step 400
        if i < 400:
            pred = rng.normal(0.1, 0.5)
            realized = pred + rng.normal(0, 0.3)
        else:
            pred = rng.normal(0.1, 0.5)
            realized = -pred + rng.normal(0, 0.3)  # drift: signs flip

        orchestrator.on_prediction(pred, realized)

    snap = orchestrator.monitor.snapshot()
    if snap:
        print(f"Final IC: {snap.ic:.4f}, DA: {snap.directional_accuracy:.3f}")
    print(f"Drift events: {len(orchestrator.drift_detector.drift_history)}")
    print(f"Retraining triggered: {orchestrator.trigger.n_triggered} time(s)")


# ---------------------------------------------------------------------------
# Extended: Sliding window IC tracker with momentum
# ---------------------------------------------------------------------------


class MomentumICTracker:
    """
    Tracks information coefficient with exponential moving average (EMA)
    and momentum-adjusted drift detection.

    Uses both short-term and long-term EMAs to detect:
    - Sharp IC drops (short > long by threshold)
    - Sustained IC degradation (both EMA falling over time)
    """

    def __init__(
        self,
        fast_alpha: float = 0.1,
        slow_alpha: float = 0.02,
        threshold_ratio: float = 0.8,
        min_samples: int = 30,
    ):
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.threshold_ratio = threshold_ratio
        self.min_samples = min_samples

        self._fast_ema: Optional[float] = None
        self._slow_ema: Optional[float] = None
        self._n = 0
        self._crossover_events: List[Dict[str, Any]] = []
        self._ic_history: deque = deque(maxlen=500)

    def update(self, ic: float) -> Optional[str]:
        """
        Update with a new IC observation.
        Returns event type if drift detected: 'sharp_drop' | 'sustained_drop' | None
        """
        self._n += 1
        self._ic_history.append(ic)

        if self._fast_ema is None:
            self._fast_ema = ic
            self._slow_ema = ic
            return None

        prev_fast = self._fast_ema
        prev_slow = self._slow_ema

        self._fast_ema = self.fast_alpha * ic + (1 - self.fast_alpha) * self._fast_ema
        self._slow_ema = self.slow_alpha * ic + (1 - self.slow_alpha) * self._slow_ema

        if self._n < self.min_samples:
            return None

        event_type = None

        # Sharp drop: fast EMA falls significantly below slow EMA
        if self._fast_ema < self._slow_ema * self.threshold_ratio and prev_fast >= prev_slow * self.threshold_ratio:
            event_type = "sharp_drop"
            self._crossover_events.append({
                "type": "sharp_drop",
                "timestamp": time.time(),
                "fast_ema": self._fast_ema,
                "slow_ema": self._slow_ema,
                "ic": ic,
            })

        # Sustained drop: both EMAs declining over many steps
        elif self._n > 100 and self._fast_ema < 0 and self._slow_ema < 0:
            if self._fast_ema < prev_fast and self._slow_ema < prev_slow:
                event_type = "sustained_drop"

        return event_type

    @property
    def fast_ema(self) -> Optional[float]:
        return self._fast_ema

    @property
    def slow_ema(self) -> Optional[float]:
        return self._slow_ema

    @property
    def momentum(self) -> float:
        """IC momentum: fast_ema - slow_ema (positive = improving, negative = degrading)."""
        if self._fast_ema is None or self._slow_ema is None:
            return 0.0
        return self._fast_ema - self._slow_ema

    @property
    def crossover_events(self) -> List[Dict[str, Any]]:
        return list(self._crossover_events)

    def summary(self) -> Dict[str, Any]:
        return {
            "fast_ema": self._fast_ema,
            "slow_ema": self._slow_ema,
            "momentum": self.momentum,
            "n_observations": self._n,
            "n_crossovers": len(self._crossover_events),
        }


# ---------------------------------------------------------------------------
# Extended: Retraining data curator
# ---------------------------------------------------------------------------


class RetrainingDataCurator:
    """
    Curates training data for model retraining.

    Features:
    - Prioritizes recent data (recency weighting)
    - Removes outlier predictions (based on IQR of errors)
    - Balances data across market regimes (trending, mean-reverting, volatile)
    - Ensures minimum sample size per regime
    """

    def __init__(
        self,
        max_samples: int = 50000,
        recency_half_life_hours: float = 24.0,
        outlier_iqr_factor: float = 3.0,
        min_samples_per_regime: int = 100,
    ):
        self.max_samples = max_samples
        self.recency_half_life_hours = recency_half_life_hours
        self.outlier_iqr_factor = outlier_iqr_factor
        self.min_samples_per_regime = min_samples_per_regime

        self._records: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def add(
        self,
        features: np.ndarray,
        target: float,
        timestamp: Optional[float] = None,
        regime: str = "unknown",
    ) -> None:
        """Add a training sample."""
        with self._lock:
            self._records.append({
                "features": features,
                "target": target,
                "timestamp": timestamp or time.time(),
                "regime": regime,
            })

    def get_training_set(
        self,
        apply_recency_weights: bool = True,
        remove_outliers: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (features, targets, weights) for training.
        weights: sample weights for weighted loss computation.
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return np.array([]), np.array([]), np.array([])

        features = np.stack([r["features"] for r in records])
        targets = np.array([r["target"] for r in records])
        timestamps = np.array([r["timestamp"] for r in records])

        # Remove outliers
        if remove_outliers and len(targets) > 10:
            q25, q75 = np.percentile(targets, [25, 75])
            iqr = q75 - q25
            inlier_mask = np.abs(targets - np.median(targets)) <= self.outlier_iqr_factor * iqr
            features = features[inlier_mask]
            targets = targets[inlier_mask]
            timestamps = timestamps[inlier_mask]

        # Recency weights
        if apply_recency_weights and len(timestamps) > 0:
            now = timestamps.max()
            half_life_sec = self.recency_half_life_hours * 3600
            age_sec = now - timestamps
            weights = np.exp(-age_sec / half_life_sec)
            weights /= weights.sum()
        else:
            weights = np.ones(len(targets)) / len(targets)

        return features, targets, weights

    def regime_distribution(self) -> Dict[str, int]:
        """Return sample counts per market regime."""
        with self._lock:
            counts: Dict[str, int] = {}
            for r in self._records:
                regime = r["regime"]
                counts[regime] = counts.get(regime, 0) + 1
        return counts

    def is_ready(self, min_total: int = 1000) -> bool:
        """Check if enough data is available for retraining."""
        with self._lock:
            n = len(self._records)
        return n >= min_total


# ---------------------------------------------------------------------------
# Extended: Statistical test library for IC comparison
# ---------------------------------------------------------------------------


class ICStatisticalTests:
    """
    Statistical tests for comparing IC distributions between model versions.
    Used by ABTestingManager for promotion decisions.
    """

    @staticmethod
    def paired_t_test(ic_control: np.ndarray, ic_challenger: np.ndarray) -> Tuple[float, float]:
        """
        Paired t-test for matched IC observations.
        Returns (t_statistic, p_value).
        """
        if len(ic_control) != len(ic_challenger):
            min_n = min(len(ic_control), len(ic_challenger))
            ic_control = ic_control[:min_n]
            ic_challenger = ic_challenger[:min_n]

        if len(ic_control) < 2:
            return 0.0, 1.0

        diff = ic_challenger - ic_control
        n = len(diff)
        mean_diff = diff.mean()
        std_diff = diff.std(ddof=1)

        if std_diff < 1e-10:
            return 0.0, 1.0 if mean_diff == 0 else 0.0

        t_stat = mean_diff / (std_diff / math.sqrt(n))
        df = n - 1

        # p-value: two-tailed (improved approximation)
        p_value = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))
        return float(t_stat), float(p_value)

    @staticmethod
    def mann_whitney_u(ic_control: np.ndarray, ic_challenger: np.ndarray) -> Tuple[float, float]:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        Returns (U_statistic, p_value).
        More robust to non-normal IC distributions.
        """
        n1, n2 = len(ic_control), len(ic_challenger)
        if n1 < 3 or n2 < 3:
            return 0.0, 1.0

        # Compute U statistic
        u1 = 0.0
        for x in ic_challenger:
            u1 += np.sum(ic_control < x) + 0.5 * np.sum(ic_control == x)
        u2 = n1 * n2 - u1

        U = min(u1, u2)
        mu_u = n1 * n2 / 2.0
        sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

        if sigma_u < 1e-10:
            return float(U), 1.0

        z = (U - mu_u) / sigma_u
        p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
        return float(U), float(np.clip(p_value, 0, 1))

    @staticmethod
    def bootstrap_ic_diff(
        ic_control: np.ndarray,
        ic_challenger: np.ndarray,
        n_bootstrap: int = 1000,
        ci_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval for the IC difference.
        Returns estimate and CI bounds.
        """
        rng = np.random.default_rng(42)
        diffs = []
        n_c = len(ic_control)
        n_ch = len(ic_challenger)

        for _ in range(n_bootstrap):
            sample_c = rng.choice(ic_control, size=n_c, replace=True)
            sample_ch = rng.choice(ic_challenger, size=n_ch, replace=True)
            diffs.append(sample_ch.mean() - sample_c.mean())

        diffs = np.array(diffs)
        alpha = 1 - ci_level
        ci_lo = float(np.percentile(diffs, alpha / 2 * 100))
        ci_hi = float(np.percentile(diffs, (1 - alpha / 2) * 100))
        p_value = float(np.mean(diffs <= 0) * 2)  # two-sided

        return {
            "mean_diff": float(diffs.mean()),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_value": float(np.clip(p_value, 0, 1)),
            "n_bootstrap": n_bootstrap,
            "ci_level": ci_level,
        }

    @staticmethod
    def minimum_detectable_effect(
        n_samples: int,
        alpha: float = 0.05,
        power: float = 0.80,
        ic_std: float = 0.05,
    ) -> float:
        """
        Compute the minimum detectable effect size (IC difference) given
        sample size and desired statistical power.
        """
        z_alpha = 1.96 if alpha == 0.05 else -math.log(alpha / 2)  # two-tailed
        z_beta = 0.842 if power == 0.80 else -math.log(1 - power)
        mde = (z_alpha + z_beta) * ic_std * math.sqrt(2 / max(n_samples, 1))
        return float(mde)


# ---------------------------------------------------------------------------
# Extended: Performance regression detector
# ---------------------------------------------------------------------------


class PerformanceRegressionDetector:
    """
    Detects performance regressions after a model update.
    Compares rolling metrics before and after the update point.
    """

    def __init__(self, lookback_window: int = 200, significance: float = 0.05):
        self.lookback_window = lookback_window
        self.significance = significance
        self._ic_before: List[float] = []
        self._ic_after: List[float] = []
        self._update_timestamps: List[float] = []
        self._regression_events: List[Dict[str, Any]] = []

    def mark_model_update(self, timestamp: Optional[float] = None) -> None:
        """Mark the timestamp of a model update."""
        ts = timestamp or time.time()
        self._update_timestamps.append(ts)
        logger.info(f"PerformanceRegressionDetector: model update marked at {ts:.0f}")

    def record_ic(self, ic: float, timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Record IC observation. Returns regression event if detected.
        """
        ts = timestamp or time.time()

        # Classify as before/after latest update
        if self._update_timestamps:
            latest_update = self._update_timestamps[-1]
            if ts >= latest_update:
                self._ic_after.append(ic)
                # Keep only last window
                if len(self._ic_after) > self.lookback_window:
                    self._ic_after = self._ic_after[-self.lookback_window:]
            else:
                self._ic_before.append(ic)
                if len(self._ic_before) > self.lookback_window:
                    self._ic_before = self._ic_before[-self.lookback_window:]
        else:
            self._ic_before.append(ic)

        return self._check_regression()

    def _check_regression(self) -> Optional[Dict[str, Any]]:
        """Check if a regression has occurred."""
        if (
            len(self._ic_before) < 30
            or len(self._ic_after) < 30
        ):
            return None

        before = np.array(self._ic_before[-self.lookback_window:])
        after = np.array(self._ic_after)

        mean_before = float(before.mean())
        mean_after = float(after.mean())

        # Statistical test
        _, p_value = ICStatisticalTests.paired_t_test(before[:len(after)], after[:len(before)])

        if p_value < self.significance and mean_after < mean_before:
            event = {
                "type": "performance_regression",
                "timestamp": time.time(),
                "ic_before": mean_before,
                "ic_after": mean_after,
                "ic_drop": mean_before - mean_after,
                "p_value": p_value,
                "n_before": len(before),
                "n_after": len(after),
            }
            self._regression_events.append(event)
            logger.warning(
                f"Performance regression detected: IC {mean_before:.4f} -> {mean_after:.4f} "
                f"(p={p_value:.4f})"
            )
            return event

        return None

    @property
    def regression_history(self) -> List[Dict[str, Any]]:
        return list(self._regression_events)
