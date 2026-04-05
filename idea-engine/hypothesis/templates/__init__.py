"""
hypothesis/templates/__init__.py

Template registry for the hypothesis generator.
`get_template_for_pattern` returns the best-fit template for a given MinedPattern.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hypothesis.types import Hypothesis, MinedPattern

from .cross_asset_template import CrossAssetTemplate
from .entry_timing_template import EntryTimingTemplate
from .exit_rule_template import ExitRuleTemplate
from .parameter_tweak_template import ParameterTweakTemplate
from .regime_filter_template import RegimeFilterTemplate


@runtime_checkable
class BaseTemplate(Protocol):
    """Duck-type protocol that all templates must satisfy."""

    def can_handle(self, pattern: MinedPattern) -> bool: ...
    def generate(self, pattern: MinedPattern) -> list[Hypothesis]: ...


# Ordered by specificity: more specific patterns first.
_REGISTRY: list[BaseTemplate] = [
    EntryTimingTemplate(),
    ExitRuleTemplate(),
    RegimeFilterTemplate(),
    CrossAssetTemplate(),
    ParameterTweakTemplate(),   # catch-all for any pattern with a detectable param
]


def get_template_for_pattern(pattern: MinedPattern) -> BaseTemplate:
    """
    Return the first template that can handle the given pattern.
    Falls back to ParameterTweakTemplate as last resort.
    Raises ValueError if no template can handle the pattern.
    """
    for template in _REGISTRY:
        if template.can_handle(pattern):
            return template

    raise ValueError(
        f"No template found for pattern type '{pattern.pattern_type}' "
        f"(id={pattern.pattern_id})"
    )


def get_all_applicable_templates(pattern: MinedPattern) -> list[BaseTemplate]:
    """
    Return ALL templates that claim to handle the pattern.
    Used by the generator when compound hypothesis generation is enabled.
    """
    return [t for t in _REGISTRY if t.can_handle(pattern)]


__all__ = [
    "BaseTemplate",
    "EntryTimingTemplate",
    "ExitRuleTemplate",
    "RegimeFilterTemplate",
    "CrossAssetTemplate",
    "ParameterTweakTemplate",
    "get_template_for_pattern",
    "get_all_applicable_templates",
]
