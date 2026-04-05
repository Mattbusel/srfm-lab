"""Experiments sub-package: A/B testing, significance testing, promotion, shadow trading."""

from .ab_test import ABTest, ABTestStatus, ABTestRunner
from .significance_tester import SignificanceTester, SignificanceResult
from .promoter import Promoter, PromotionEvent
from .shadow_trader import ShadowTrader, ShadowResult

__all__ = [
    "ABTest", "ABTestStatus", "ABTestRunner",
    "SignificanceTester", "SignificanceResult",
    "Promoter", "PromotionEvent",
    "ShadowTrader", "ShadowResult",
]
