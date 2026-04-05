"""
debate-system/promoter.py

HypothesisPromoter: acts on the DebateResult to move hypotheses through the
IAE lifecycle.

Actions
-------
APPROVED      → POST /hypotheses/{id}/approve to IAE API
                Sets hypothesis status to PROMOTED in local DB.
REJECTED      → Archive with rejection reason (debate_id + summary concerns).
                Sets hypothesis status to REJECTED.
NEEDS_MORE_DATA → Schedule re-debate after 7 days via IAE scheduler table.
                  Sets hypothesis status to PENDING with scheduled_redebate_at.

Retry policy
------------
IAE API calls are retried up to 3 times with exponential backoff before
giving up and logging the failure. A failed promote does NOT mark the
hypothesis as REJECTED — it stays at QUEUED and is retried on next run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from debate_system.debate.chamber import DebateOutcome, DebateResult

logger = logging.getLogger(__name__)

IAE_API_BASE = "http://localhost:8767"
DB_PATH = Path("C:/Users/Matthew/srfm-lab/idea-engine/idea_engine.db")


class HypothesisPromoter:
    """
    Routes debate outcomes to the appropriate IAE lifecycle action.

    Parameters
    ----------
    api_base  : Base URL of the IAE idea-api service.
    db_path   : Path to idea_engine.db for local status updates.
    dry_run   : If True, skip actual API calls (for testing).
    """

    REDEBATE_DELAY_DAYS = 7
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0    # seconds; doubles each retry

    def __init__(
        self,
        api_base: str = IAE_API_BASE,
        db_path: Path = DB_PATH,
        dry_run: bool = False,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.db_path = db_path
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, result: DebateResult) -> None:
        """
        Route a DebateResult to the appropriate action.
        This is the main entry point called by the debate orchestrator.
        """
        logger.info(
            "Processing debate result: hypothesis=%s outcome=%s",
            result.hypothesis_id[:8],
            result.outcome.value,
        )

        if result.outcome == DebateOutcome.APPROVED:
            self._handle_approved(result)
        elif result.outcome == DebateOutcome.REJECTED:
            self._handle_rejected(result)
        elif result.outcome == DebateOutcome.NEEDS_MORE_DATA:
            self._handle_needs_more_data(result)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_approved(self, result: DebateResult) -> None:
        """Promote approved hypothesis to backtest queue."""
        hypothesis_id = result.hypothesis_id
        logger.info("Promoting hypothesis %s to backtest queue.", hypothesis_id)

        payload = {
            "debate_id": result.debate_id,
            "weighted_for": round(result.weighted_for, 4),
            "consensus_score": round(result.consensus_score, 4),
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }

        success = self._post_with_retry(
            endpoint=f"/hypotheses/{hypothesis_id}/approve",
            payload=payload,
        )

        if success:
            self._update_hypothesis_status(hypothesis_id, "promoted", {
                "debate_id": result.debate_id,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("Hypothesis %s promoted successfully.", hypothesis_id)
        else:
            logger.error(
                "Failed to promote hypothesis %s after %d retries. "
                "Leaving as QUEUED for next run.",
                hypothesis_id,
                self.MAX_RETRIES,
            )

    def _handle_rejected(self, result: DebateResult) -> None:
        """Archive rejected hypothesis with full debate reasoning."""
        hypothesis_id = result.hypothesis_id

        # Collect the top concerns from all AGAINST votes
        top_concerns: list[str] = []
        for verdict in result.verdicts:
            from debate_system.agents.base_agent import Vote
            if verdict.vote == Vote.AGAINST:
                top_concerns.extend(verdict.key_concerns[:2])
        top_concerns = top_concerns[:6]   # cap at 6 concerns for storage

        rejection_summary = {
            "debate_id": result.debate_id,
            "rejected_at": datetime.now(timezone.utc).isoformat(),
            "weighted_for": round(result.weighted_for, 4),
            "veto_issued": result.veto_issued,
            "top_concerns": top_concerns,
            "verdict_summary": [
                {
                    "agent": v.agent_name,
                    "vote": v.vote.value,
                    "confidence": round(v.confidence, 3),
                }
                for v in result.verdicts
            ],
        }

        self._update_hypothesis_status(hypothesis_id, "rejected", rejection_summary)
        logger.info(
            "Hypothesis %s archived as REJECTED. Veto=%s. Top concern: %s",
            hypothesis_id,
            result.veto_issued,
            top_concerns[0] if top_concerns else "none",
        )

    def _handle_needs_more_data(self, result: DebateResult) -> None:
        """Schedule the hypothesis for re-debate in 7 days."""
        hypothesis_id = result.hypothesis_id
        redebate_at = (
            datetime.now(timezone.utc) + timedelta(days=self.REDEBATE_DELAY_DAYS)
        ).isoformat()

        schedule_data = {
            "debate_id": result.debate_id,
            "scheduled_redebate_at": redebate_at,
            "last_weighted_for": round(result.weighted_for, 4),
            "last_debate_concerns": [
                c
                for v in result.verdicts
                for c in v.key_concerns[:1]
            ][:4],
        }

        self._update_hypothesis_status(hypothesis_id, "pending", schedule_data)

        # Also notify the scheduler service if available
        self._post_with_retry(
            endpoint=f"/hypotheses/{hypothesis_id}/schedule-redebate",
            payload={"redebate_at": redebate_at, "debate_id": result.debate_id},
            fail_ok=True,    # scheduler is optional — don't hard-fail
        )

        logger.info(
            "Hypothesis %s scheduled for re-debate at %s.",
            hypothesis_id,
            redebate_at,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_with_retry(
        self,
        endpoint: str,
        payload: dict[str, Any],
        fail_ok: bool = False,
    ) -> bool:
        """POST JSON payload to IAE API with exponential backoff retry."""
        if self.dry_run:
            logger.debug("[DRY RUN] POST %s %s", endpoint, payload)
            return True

        url = self.api_base + endpoint
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if 200 <= resp.status < 300:
                        return True
                    logger.warning(
                        "POST %s returned HTTP %d (attempt %d/%d)",
                        url,
                        resp.status,
                        attempt,
                        self.MAX_RETRIES,
                    )
            except urllib.error.URLError as exc:
                logger.warning(
                    "POST %s failed (attempt %d/%d): %s",
                    url,
                    attempt,
                    self.MAX_RETRIES,
                    exc,
                )
            except Exception as exc:
                logger.error("Unexpected error posting to %s: %s", url, exc)

            if attempt < self.MAX_RETRIES:
                sleep_sec = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(sleep_sec)

        if not fail_ok:
            logger.error("All %d retries exhausted for %s", self.MAX_RETRIES, url)
        return False

    def _update_hypothesis_status(
        self,
        hypothesis_id: str,
        new_status: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        Write status update and metadata JSON to the hypotheses table.
        Uses the existing schema from hypothesis/hypothesis_store.py.
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Store debate metadata in the parameters JSON blob
                # (avoids schema migration while keeping data together)
                row = conn.execute(
                    "SELECT parameters FROM hypotheses WHERE hypothesis_id = ?",
                    (hypothesis_id,),
                ).fetchone()

                if row is None:
                    logger.warning(
                        "Hypothesis %s not found in DB — cannot update status.",
                        hypothesis_id,
                    )
                    return

                params = json.loads(row[0]) if row[0] else {}
                params["_debate_metadata"] = metadata

                conn.execute(
                    """UPDATE hypotheses
                       SET status = ?, parameters = ?
                       WHERE hypothesis_id = ?""",
                    (new_status, json.dumps(params), hypothesis_id),
                )
                conn.commit()
        except sqlite3.Error as exc:
            logger.error(
                "DB error updating hypothesis %s to status %s: %s",
                hypothesis_id,
                new_status,
                exc,
            )
