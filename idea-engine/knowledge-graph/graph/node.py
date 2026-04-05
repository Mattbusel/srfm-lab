"""
Knowledge graph node definitions.

All domain objects in the trading knowledge graph are represented as typed
nodes.  Each node carries a confidence score, a property dict, and a
creation timestamp.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class NodeType(str, Enum):
    INSTRUMENT = "instrument"
    SIGNAL     = "signal"
    HYPOTHESIS = "hypothesis"
    REGIME     = "regime"
    PATTERN    = "pattern"
    EVENT      = "event"
    PARAMETER  = "parameter"


@dataclass
class BaseNode:
    """Base class for all knowledge graph nodes."""

    node_id: str
    node_type: NodeType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.5        # 0-1 evidential confidence
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.node_id:
            self.node_id = str(uuid.uuid4())

    def update_confidence(self, new_confidence: float) -> None:
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.updated_at = datetime.now(timezone.utc)

    def set_property(self, key: str, value: Any) -> None:
        self.properties[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def _make_id(cls, prefix: str, label: str) -> str:
        safe = label.lower().replace(" ", "_").replace("/", "_")[:40]
        return f"{prefix}_{safe}"


@dataclass
class InstrumentNode(BaseNode):
    """A tradeable instrument (BTC, ETH, SOL, etc.)."""

    def __init__(
        self,
        symbol: str,
        exchange: str = "binance",
        asset_class: str = "crypto",
        confidence: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("instr", symbol),
            node_type=NodeType.INSTRUMENT,
            label=symbol.upper(),
            confidence=confidence,
            properties={
                "symbol": symbol.upper(),
                "exchange": exchange,
                "asset_class": asset_class,
                **kwargs,
            },
        )


@dataclass
class SignalNode(BaseNode):
    """A trading signal or indicator."""

    def __init__(
        self,
        signal_name: str,
        signal_family: str = "unknown",
        description: str = "",
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("sig", signal_name),
            node_type=NodeType.SIGNAL,
            label=signal_name,
            confidence=confidence,
            properties={
                "signal_family": signal_family,
                "description": description,
                **kwargs,
            },
        )


@dataclass
class HypothesisNode(BaseNode):
    """
    An IAE hypothesis — a proposed change to trading parameters or strategy.
    """

    def __init__(
        self,
        hypothesis_id: str,
        hypothesis_text: str,
        parameter: str = "",
        direction: str = "",   # increase | decrease | remove | add
        confidence: float = 0.5,
        evidence_count: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("hyp", hypothesis_id),
            node_type=NodeType.HYPOTHESIS,
            label=hypothesis_id,
            confidence=confidence,
            properties={
                "hypothesis_text": hypothesis_text,
                "parameter": parameter,
                "direction": direction,
                "evidence_count": evidence_count,
                **kwargs,
            },
        )


@dataclass
class RegimeNode(BaseNode):
    """A market regime (high-vol, trending, ranging, bear, bull, etc.)."""

    def __init__(
        self,
        regime_name: str,
        regime_type: str = "volatility",  # volatility | trend | sentiment
        definition: str = "",
        confidence: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("regime", regime_name),
            node_type=NodeType.REGIME,
            label=regime_name,
            confidence=confidence,
            properties={
                "regime_type": regime_type,
                "definition": definition,
                **kwargs,
            },
        )


@dataclass
class PatternNode(BaseNode):
    """
    A known market pattern with its causal explanation and conditions.
    """

    def __init__(
        self,
        pattern_name: str,
        pattern_type: str = "price",   # price | on_chain | macro
        description: str = "",
        causal_explanation: str = "",
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("pat", pattern_name),
            node_type=NodeType.PATTERN,
            label=pattern_name,
            confidence=confidence,
            properties={
                "pattern_type": pattern_type,
                "description": description,
                "causal_explanation": causal_explanation,
                **kwargs,
            },
        )


@dataclass
class EventNode(BaseNode):
    """A market-moving event (macro, listing, unlock, upgrade)."""

    def __init__(
        self,
        event_name: str,
        event_type: str = "crypto",   # crypto | macro | on_chain
        expected_impact_pct: float = 0.0,
        direction: str = "NEUTRAL",
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("evt", event_name),
            node_type=NodeType.EVENT,
            label=event_name,
            confidence=confidence,
            properties={
                "event_type": event_type,
                "expected_impact_pct": expected_impact_pct,
                "direction": direction,
                **kwargs,
            },
        )


@dataclass
class ParameterNode(BaseNode):
    """A strategy parameter (min_hold_bars, entry_hour_filter, etc.)."""

    def __init__(
        self,
        param_name: str,
        current_value: Any = None,
        value_type: str = "float",
        valid_range: Optional[tuple] = None,
        confidence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            node_id=self._make_id("param", param_name),
            node_type=NodeType.PARAMETER,
            label=param_name,
            confidence=confidence,
            properties={
                "current_value": current_value,
                "value_type": value_type,
                "valid_range": list(valid_range) if valid_range else None,
                **kwargs,
            },
        )
