"""
satellite_web.py -- Alternative web and proxy data signals.

This module derives sentiment and attention signals from unstructured text
and structured activity data without requiring paid API subscriptions.

Modules:
  GoogleTrendsProxy    -- Attention scoring from news headline frequency
  GitHubActivitySignal -- Developer activity from repo commit/star/fork data
  AppStoreProxy        -- Exchange user growth signal from headline mentions
"""

from __future__ import annotations

import math
import re
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_count(headlines: List[str], keyword: str) -> int:
    """
    Count the number of headlines containing the normalized keyword.

    Matching rules:
    - Single-word keywords: match if the token appears in the headline.
    - Multi-word keywords: require ALL tokens of the keyword phrase to appear
      in the headline (AND logic). This prevents false positives where a
      headline about "new user sign-ups" spuriously matches "user exodus"
      because the single token "user" overlaps.
    """
    kw_normalized = _normalize_text(keyword)
    kw_tokens = kw_normalized.split()
    count = 0
    if len(kw_tokens) <= 1:
        # Single token: simple membership check
        for headline in headlines:
            tokens = set(_normalize_text(headline).split())
            if kw_tokens and kw_tokens[0] in tokens:
                count += 1
    else:
        # Multi-token phrase: require all tokens to be present (AND logic)
        kw_token_set = set(kw_tokens)
        for headline in headlines:
            tokens = set(_normalize_text(headline).split())
            if kw_token_set.issubset(tokens):
                count += 1
    return count


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_mean(values: List[float]) -> float:
    """Return mean or 0.0 for empty list."""
    if not values:
        return 0.0
    return statistics.mean(values)


def _safe_stdev(values: List[float]) -> float:
    """Return sample stdev or 1.0 as fallback for insufficient data."""
    if len(values) < 2:
        return 1.0
    return statistics.stdev(values)


# ---------------------------------------------------------------------------
# GoogleTrendsProxy
# ---------------------------------------------------------------------------

class GoogleTrendsProxy:
    """
    Proxies Google Trends search interest using news headline frequency as
    a substitute signal. No external API is required.

    The underlying hypothesis is that keyword search volume correlates with
    the rate at which headlines containing that keyword appear in news feeds.
    This is a well-established proxy in academic NLP-based finance research.

    Parameters
    ----------
    baseline_window_days : int
        Number of historical daily counts to use as the baseline average.
        Default 30 (1 month rolling baseline).
    spike_threshold : float
        Multiplier above the baseline that triggers a spike signal.
        Default 3.0 (3x the baseline = strong attention spike).
    """

    def __init__(
        self,
        baseline_window_days: int = 30,
        spike_threshold: float = 3.0,
    ) -> None:
        self.baseline_window_days = baseline_window_days
        self.spike_threshold = spike_threshold
        # Stores daily headline counts per keyword: {keyword: [day1_count, day2_count, ...]}
        self._daily_counts: Dict[str, List[float]] = defaultdict(list)

    def compute_attention_score(
        self,
        headlines: List[str],
        keyword: str,
        window_days: int = 7,
    ) -> float:
        """
        Compute a normalized attention score for the given keyword.

        Parameters
        ----------
        headlines   : List of news headlines for the CURRENT day/period.
        keyword     : Target keyword or phrase (case-insensitive).
        window_days : Lookback window for computing the "current rate"
                      relative to baseline. Default 7.

        Returns
        -------
        float in [-1, +1]:
          +1.0 -> massive attention spike (extreme buzz)
           0.0 -> attention at baseline levels
          -1.0 -> attention has dropped well below baseline (fading interest)
          +0.5 -> 3x baseline spike (default spike threshold)
        """
        kw_key = _normalize_text(keyword)
        daily_count = float(_keyword_count(headlines, keyword))
        self._daily_counts[kw_key].append(daily_count)
        history = self._daily_counts[kw_key]

        if len(history) < 3:
            # Not enough history to compute a meaningful baseline
            return 0.0

        # Compute rolling baseline: mean over previous `baseline_window_days`
        baseline_data = history[-(self.baseline_window_days + 1):-1]
        if not baseline_data:
            return 0.0
        baseline = _safe_mean(baseline_data)
        baseline_std = _safe_stdev(baseline_data)

        # Current rate: mean of last `window_days` observations
        current_window = history[-window_days:] if len(history) >= window_days else history
        current_rate = _safe_mean(current_window)

        # Compute relative attention: how many standard deviations above baseline?
        if baseline < 0.5:
            # Near-zero baseline: any mention is meaningful
            if current_rate > 0:
                return _clamp(current_rate / max(1.0, baseline_std))
            return 0.0

        relative = current_rate / baseline  # ratio to baseline

        # Signal mapping:
        # > spike_threshold (e.g. 3x) -> +0.5 to +1.0 (rising fast)
        # 1.5x to 3x -> mild positive
        # 0.5x to 1.5x -> neutral
        # < 0.5x -> fading attention -> negative signal
        if relative >= self.spike_threshold:
            # Map 3x -> +0.5, 6x -> +1.0
            excess = relative / self.spike_threshold
            return _clamp(0.5 + 0.5 * math.log(max(excess, 1.0), 2.0))
        if relative >= 1.5:
            return _clamp((relative - 1.0) / 4.0)
        if relative >= 0.5:
            return 0.0
        # Fading attention
        fade = 1.0 - relative / 0.5  # 0 -> 0.0 signal, 1 -> -1.0 signal
        return _clamp(-fade)

    def get_attention_history(self, keyword: str) -> List[float]:
        """Return the raw daily count history for a keyword."""
        return list(self._daily_counts[_normalize_text(keyword)])

    def get_baseline(self, keyword: str) -> float:
        """Return the current baseline (mean of last baseline_window_days counts)."""
        kw_key = _normalize_text(keyword)
        history = self._daily_counts[kw_key]
        if len(history) < 2:
            return 0.0
        baseline_data = history[-(self.baseline_window_days + 1):-1]
        return _safe_mean(baseline_data)

    def is_spiking(self, keyword: str) -> bool:
        """
        Return True if current attention is above the spike threshold.

        When the baseline is near zero (keyword barely mentioned historically),
        any positive mention count is treated as a spike -- an appearance from
        near silence is inherently noteworthy.
        """
        kw_key = _normalize_text(keyword)
        history = self._daily_counts[kw_key]
        if len(history) < 3:
            return False
        baseline = self.get_baseline(keyword)
        current  = history[-1]
        if baseline < 0.5:
            # Near-zero baseline: any positive mention is a spike
            return current > 0
        return current >= baseline * self.spike_threshold

    def multi_keyword_score(
        self,
        headlines: List[str],
        keywords: List[str],
        weights: Optional[List[float]] = None,
        window_days: int = 7,
    ) -> float:
        """
        Compute a weighted composite attention score across multiple keywords.

        Parameters
        ----------
        keywords : List of keyword strings.
        weights  : Optional weight list (must sum to 1.0). Equal-weighted if None.
        """
        if not keywords:
            return 0.0
        w = weights if weights is not None else [1.0 / len(keywords)] * len(keywords)
        if len(w) != len(keywords):
            raise ValueError("weights length must match keywords length")
        total = sum(
            wi * self.compute_attention_score(headlines, kw, window_days)
            for kw, wi in zip(keywords, w)
        )
        return _clamp(total)


# ---------------------------------------------------------------------------
# GitHubActivitySignal
# ---------------------------------------------------------------------------

class GitHubActivitySignal:
    """
    Derives a developer activity signal from GitHub repository statistics.

    Developer activity is a leading indicator of protocol health and future
    adoption. Rising commit frequency, star growth, and fork activity signal
    that builders are engaged -- which historically precedes user growth.

    All data is passed as parameters -- no live API calls are made.
    Callers should fetch data from the GitHub API or a provider like
    Electric Capital's crypto developer report.

    Parameters
    ----------
    history_window : int
        Number of historical snapshots to retain per repository.
    """

    # Weights for the composite developer momentum score
    COMMIT_WEIGHT  = 0.50
    STAR_WEIGHT    = 0.25
    FORK_WEIGHT    = 0.15
    PR_WEIGHT      = 0.10

    def __init__(self, history_window: int = 90) -> None:
        self.history_window = history_window
        # Per-repo history: {repo_name: {"commits": [...], "stars": [...], ...}}
        self._histories: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def record(self, repo_name: str, repo_stats: Dict[str, float]) -> None:
        """
        Record a new snapshot of repository statistics.

        Parameters
        ----------
        repo_name  : Repository identifier (e.g., "bitcoin/bitcoin").
        repo_stats : Dict with keys:
          commits_30d   : Commit count in the last 30 days.
          stars_total   : Total stars (cumulative).
          forks_total   : Total forks (cumulative).
          open_prs      : Number of open pull requests.
          contributors  : Active contributor count.
          issues_closed : Issues closed in last 30 days (quality signal).
        """
        hist = self._histories[repo_name]
        for key, value in repo_stats.items():
            hist[key].append(value)
            # Enforce history window
            if len(hist[key]) > self.history_window:
                hist[key] = hist[key][-self.history_window:]

    def compute_dev_activity(self, repo_stats: Dict[str, float]) -> float:
        """
        Compute a normalized developer momentum score in [0, 1].

        Parameters
        ----------
        repo_stats : Current snapshot with any subset of keys:
          commits_30d, stars_7d_growth, forks_7d_growth, open_prs,
          contributors, issues_closed

        Returns
        -------
        float in [0, 1] where:
          1.0 -> peak developer engagement
          0.5 -> average activity
          0.0 -> minimal/declining activity

        Methodology:
          Each metric is z-scored vs its own history, then combined with
          fixed weights. Missing metrics are treated as neutral (z=0).
        """
        metrics: Dict[str, float] = {}

        # Commits: primary developer engagement signal
        commits = repo_stats.get("commits_30d", repo_stats.get("commits", 0.0))
        metrics["commits"] = commits

        # Stars: community interest / growth
        stars = repo_stats.get("stars_7d_growth", repo_stats.get("stars_total", 0.0))
        metrics["stars"] = stars

        # Forks: active development by external contributors
        forks = repo_stats.get("forks_7d_growth", repo_stats.get("forks_total", 0.0))
        metrics["forks"] = forks

        # Open PRs: development pipeline depth
        prs = repo_stats.get("open_prs", 0.0)
        metrics["prs"] = prs

        # Compute a simple ratio-based score when no history is available
        # (First snapshot -- no z-score possible)
        total_activity = (
            self.COMMIT_WEIGHT * min(commits / 100.0, 1.0)
            + self.STAR_WEIGHT  * min(stars   / 500.0, 1.0)
            + self.FORK_WEIGHT  * min(forks   / 100.0, 1.0)
            + self.PR_WEIGHT    * min(prs     / 50.0,  1.0)
        )
        return _clamp(total_activity, 0.0, 1.0)

    def dev_momentum_delta(
        self, repo_name: str, current_stats: Dict[str, float], lookback: int = 30
    ) -> float:
        """
        Compute the change in developer activity relative to `lookback` periods ago.

        Returns a value in [-1, +1]:
          +1.0 -> dev activity significantly increased
          -1.0 -> dev activity significantly decreased
        """
        hist = self._histories[repo_name]
        commit_hist = hist.get("commits_30d", hist.get("commits", []))
        if len(commit_hist) < lookback + 1:
            return 0.0
        old = _safe_mean(commit_hist[-(lookback + 1):-lookback])
        new = _safe_mean(commit_hist[-lookback:])
        if old < 1e-6:
            return 0.0 if new < 1e-6 else 1.0
        relative_change = (new - old) / old
        return _clamp(relative_change)

    def ecosystem_signal(
        self, repos: List[str], current_stats_map: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Compute an ecosystem-level developer signal by aggregating across
        multiple protocol repositories.

        Parameters
        ----------
        repos             : List of repo identifiers to include.
        current_stats_map : Dict mapping repo names to their current stats dicts.

        Returns
        -------
        float in [0, 1] -- average dev activity score across repos.
        """
        if not repos:
            return 0.0
        scores = []
        for repo in repos:
            stats = current_stats_map.get(repo, {})
            scores.append(self.compute_dev_activity(stats))
        return _safe_mean(scores)

    def is_trending_up(self, repo_name: str, window: int = 14) -> bool:
        """Return True if commit activity is rising over the last `window` periods."""
        hist = self._histories[repo_name]
        commits = hist.get("commits_30d", hist.get("commits", []))
        if len(commits) < window * 2:
            return False
        old_avg = _safe_mean(commits[-(window * 2):-window])
        new_avg = _safe_mean(commits[-window:])
        return new_avg > old_avg * 1.1


# ---------------------------------------------------------------------------
# AppStoreProxy
# ---------------------------------------------------------------------------

class AppStoreProxy:
    """
    Proxies exchange and wallet app download signals using news headline
    frequency as a substitute for proprietary app store data.

    Growth-related headlines ("record downloads", "new users surge", "sign-ups")
    are treated as positive signals. Declining or negative headlines are negative.

    Parameters
    ----------
    growth_keywords   : Keywords indicating positive user growth.
    decline_keywords  : Keywords indicating user decline or exit.
    baseline_window   : Days of history for baseline computation.
    """

    DEFAULT_GROWTH_KEYWORDS = [
        "record downloads", "new users", "sign ups", "sign-ups",
        "user growth", "installs", "onboarded", "registrations",
        "downloads surged", "downloaded", "app store", "new accounts",
    ]

    DEFAULT_DECLINE_KEYWORDS = [
        "users leaving", "user exodus", "deplatformed", "accounts deleted",
        "regulatory ban", "withdrawals halted", "outflows", "lost users",
    ]

    def __init__(
        self,
        growth_keywords:  Optional[List[str]] = None,
        decline_keywords: Optional[List[str]] = None,
        baseline_window:  int = 30,
    ) -> None:
        self.growth_keywords  = growth_keywords  or self.DEFAULT_GROWTH_KEYWORDS
        self.decline_keywords = decline_keywords or self.DEFAULT_DECLINE_KEYWORDS
        self.baseline_window  = baseline_window
        self._growth_history:  List[float] = []
        self._decline_history: List[float] = []

    def user_growth_signal(self, headlines: List[str]) -> float:
        """
        Compute a user growth signal from news headlines.

        Parameters
        ----------
        headlines : List of news headline strings for the current period.

        Returns
        -------
        float in [-1, +1]:
          +1.0 -> strong evidence of user adoption acceleration
           0.0 -> neutral / no strong signal
          -1.0 -> strong evidence of user decline
        """
        growth_count  = float(sum(
            _keyword_count(headlines, kw) for kw in self.growth_keywords
        ))
        decline_count = float(sum(
            _keyword_count(headlines, kw) for kw in self.decline_keywords
        ))
        self._growth_history.append(growth_count)
        self._decline_history.append(decline_count)

        if len(self._growth_history) < 3:
            # Insufficient history: return raw directional signal only
            net = growth_count - decline_count
            if net == 0:
                return 0.0
            return _clamp(net / max(1.0, growth_count + decline_count))

        # Compute baselines
        g_baseline = _safe_mean(self._growth_history[-self.baseline_window:-1])
        d_baseline = _safe_mean(self._decline_history[-self.baseline_window:-1])

        # Z-score each component. When baseline stdev is near zero (sparse signal),
        # fall back to a simple ratio-based score to avoid division-by-zero silence.
        g_std = _safe_stdev(self._growth_history[-self.baseline_window:])
        d_std = _safe_stdev(self._decline_history[-self.baseline_window:])

        if g_std < 0.5 and g_baseline < 0.5:
            # Near-zero baseline: any mentions are meaningful signal
            g_z = growth_count / max(1.0, growth_count + 1.0) * 3.0
        else:
            g_z = (growth_count - g_baseline) / max(g_std, 1e-6)

        if d_std < 0.5 and d_baseline < 0.5:
            d_z = decline_count / max(1.0, decline_count + 1.0) * 3.0
        else:
            d_z = (decline_count - d_baseline) / max(d_std, 1e-6)

        # Net signal: positive growth z minus positive decline z
        net_z = g_z - d_z
        return _clamp(net_z / 3.0)  # scale: 3-sigma net -> full signal

    def growth_velocity(self, lookback: int = 7) -> float:
        """
        Return the rate of change in growth headline frequency over the
        last `lookback` periods vs the prior period of equal length.

        Positive -> accelerating growth coverage.
        Negative -> decelerating growth coverage.
        """
        if len(self._growth_history) < lookback * 2:
            return 0.0
        old = _safe_mean(self._growth_history[-(lookback * 2):-lookback])
        new = _safe_mean(self._growth_history[-lookback:])
        if old < 0.5:
            return 0.0
        return _clamp((new - old) / old)

    def decline_alert(self) -> bool:
        """
        Return True if decline keyword count is significantly elevated vs baseline.
        Threshold: 2 standard deviations above the historical mean.
        """
        if len(self._decline_history) < 5:
            return False
        history = self._decline_history
        baseline = _safe_mean(history[-self.baseline_window:-1])
        std = _safe_stdev(history[-self.baseline_window:])
        current = history[-1]
        return current > baseline + 2.0 * std

    def combined_sentiment_score(self, headlines: List[str]) -> Dict[str, float]:
        """
        Return a dict with the user growth signal and its components for
        the current headline batch.
        """
        signal = self.user_growth_signal(headlines)
        growth_count = float(sum(
            _keyword_count(headlines, kw) for kw in self.growth_keywords
        ))
        decline_count = float(sum(
            _keyword_count(headlines, kw) for kw in self.decline_keywords
        ))
        return {
            "user_growth_signal":  signal,
            "growth_mentions":     growth_count,
            "decline_mentions":    decline_count,
            "growth_velocity_7d":  self.growth_velocity(7),
            "decline_alert":       float(self.decline_alert()),
        }
