"""
Narrative Intelligence — Component of the Idea Automation Engine (IAE)
=======================================================================
Takes all the data — sim runs, hypotheses, counterfactuals, shadow
comparisons — and writes human-readable research narratives, weekly
reports, and idea summaries.

Public API
----------
    from narrative import NarrativeWriter, TemplateEngine, AlertWriter

Typical usage::

    from narrative import NarrativeWriter, AlertWriter

    writer = NarrativeWriter(db_path="idea_engine.db")
    report = writer.weekly_report()
    print(report)

    alerts = AlertWriter(db_path="idea_engine.db")
    for alert in alerts.check_alerts():
        print(alerts.format_alert(alert))
"""

from .report_writer   import NarrativeWriter
from .template_engine import TemplateEngine
from .alert_writer    import AlertWriter, Alert

__all__ = [
    "NarrativeWriter",
    "TemplateEngine",
    "AlertWriter",
    "Alert",
]

__version__ = "0.1.0"
