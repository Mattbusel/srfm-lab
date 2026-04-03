"""
LARSA v2 — Next iteration workspace.

Starting point: larsa-v1 (274% baseline).
Current objective: multi-asset calibration (ZB, GC) + regime-aware position sizing.

Experiment log:
    [DATE] — Created from larsa-v1 baseline
    [DATE] — Add ZB (30Y T-Bond) as second instrument
    [DATE] — Calibrate BH parameters independently per asset

To run:
    make backtest s=larsa-v2
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "lib"))

# TODO: start from larsa-v1 and modify one thing at a time.
# Use the template in strategies/templates/main.py as the scaffold.
raise NotImplementedError("Build larsa-v2 from larsa-v1 baseline.")
