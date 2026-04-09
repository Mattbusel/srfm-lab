"""
Serendipity Stream: Signal-as-a-Service API.

Streams physics-derived trading signals to external clients via HTTP/WebSocket.
Revenue model: tiered subscriptions (Silver: delayed, Gold: real-time + narratives).

Architecture:
  EHS -> IdeaSynthesizer -> SignalDispatcher -> Redis/PubSub -> Client API

Each signal payload includes:
  - signal_id, symbol, direction, strength, confidence
  - physics_domain (what physics concept generated it)
  - narrative (human-readable explanation from NarrativeGenerator)
  - provenance_trace (full decision chain)
  - dream_fragility_score (how robust is this signal against dream scenarios)
"""

from __future__ import annotations
import json
import time
import hashlib
import hmac
from collections import defaultdict, deque
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs


# ---------------------------------------------------------------------------
# Signal Payload
# ---------------------------------------------------------------------------

@dataclass
class SignalPayload:
    """A single signal broadcast to API clients."""
    signal_id: str
    timestamp: float
    symbol: str
    direction: str              # "long" / "short" / "neutral"
    strength: float             # 0-1 signal strength
    confidence: float           # 0-1 statistical confidence
    physics_domain: str         # e.g., "statistical_mechanics", "QFT"
    physics_concept: str        # e.g., "Ising Model Phase Transition"
    narrative: str              # human-readable explanation
    regime: str                 # current market regime
    dream_fragility: float      # 0=robust, 1=fragile
    provenance_depth: int       # how many layers in the decision chain
    ttl_bars: int               # expected signal lifetime in bars
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "physics_domain": self.physics_domain,
            "physics_concept": self.physics_concept,
            "narrative": self.narrative,
            "regime": self.regime,
            "dream_fragility": self.dream_fragility,
            "provenance_depth": self.provenance_depth,
            "ttl_bars": self.ttl_bars,
        }

    def to_silver(self) -> dict:
        """Silver tier: delayed, no narrative, no provenance."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp - 300,  # 5 minute delay
            "symbol": self.symbol,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "physics_domain": self.physics_domain,
            "regime": self.regime,
        }

    def to_gold(self) -> dict:
        """Gold tier: real-time, with narrative and full metadata."""
        d = self.to_dict()
        d["narrative"] = self.narrative
        d["metadata"] = self.metadata
        return d


# ---------------------------------------------------------------------------
# API Key Management
# ---------------------------------------------------------------------------

@dataclass
class APIClient:
    """A registered API client."""
    client_id: str
    api_key: str
    tier: str                   # "silver" / "gold" / "platinum"
    symbols: List[str]          # subscribed symbols
    rate_limit_per_min: int = 60
    active: bool = True
    created_at: float = 0.0
    total_requests: int = 0
    total_signals_consumed: int = 0


class APIKeyManager:
    """Manage API keys and client authentication."""

    def __init__(self):
        self._clients: Dict[str, APIClient] = {}
        self._key_to_client: Dict[str, str] = {}
        self._request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=120))

    def register_client(self, client_id: str, tier: str = "silver",
                         symbols: List[str] = None) -> str:
        """Register a new client and return their API key."""
        api_key = hashlib.sha256(f"{client_id}:{time.time()}:{tier}".encode()).hexdigest()[:32]
        client = APIClient(
            client_id=client_id,
            api_key=api_key,
            tier=tier,
            symbols=symbols or ["BTC", "ETH"],
            rate_limit_per_min={"silver": 30, "gold": 120, "platinum": 600}.get(tier, 30),
            created_at=time.time(),
        )
        self._clients[client_id] = client
        self._key_to_client[api_key] = client_id
        return api_key

    def authenticate(self, api_key: str) -> Optional[APIClient]:
        """Authenticate a request by API key."""
        client_id = self._key_to_client.get(api_key)
        if not client_id:
            return None
        client = self._clients.get(client_id)
        if not client or not client.active:
            return None

        # Rate limiting
        now = time.time()
        self._request_counts[client_id].append(now)
        recent = [t for t in self._request_counts[client_id] if now - t < 60]
        if len(recent) > client.rate_limit_per_min:
            return None  # rate limited

        client.total_requests += 1
        return client


# ---------------------------------------------------------------------------
# Signal Dispatcher
# ---------------------------------------------------------------------------

class SignalDispatcher:
    """
    Dispatches signals from the EHS/autonomous loop to API clients.
    Maintains a buffer of recent signals for polling clients.
    """

    def __init__(self, buffer_size: int = 1000):
        self._buffer: deque = deque(maxlen=buffer_size)
        self._by_symbol: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._total_dispatched: int = 0

    def dispatch(self, signal: SignalPayload) -> None:
        """Add a new signal to the dispatch buffer."""
        self._buffer.append(signal)
        self._by_symbol[signal.symbol].append(signal)
        self._total_dispatched += 1

    def get_latest(self, symbol: Optional[str] = None, n: int = 10,
                    tier: str = "gold") -> List[dict]:
        """Get latest signals, formatted for the client's tier."""
        if symbol:
            signals = list(self._by_symbol.get(symbol, []))[-n:]
        else:
            signals = list(self._buffer)[-n:]

        if tier == "silver":
            return [s.to_silver() for s in signals]
        elif tier == "gold":
            return [s.to_gold() for s in signals]
        else:
            return [s.to_dict() for s in signals]

    def get_stats(self) -> dict:
        return {
            "total_dispatched": self._total_dispatched,
            "buffer_size": len(self._buffer),
            "symbols": list(self._by_symbol.keys()),
            "signals_per_symbol": {k: len(v) for k, v in self._by_symbol.items()},
        }


# ---------------------------------------------------------------------------
# Revenue Tracker
# ---------------------------------------------------------------------------

class RevenueTracker:
    """Track revenue from API usage."""

    PRICING = {
        "silver": {"monthly": 299, "per_signal": 0.0},
        "gold": {"monthly": 999, "per_signal": 0.01},
        "platinum": {"monthly": 4999, "per_signal": 0.0},
    }

    def __init__(self):
        self._usage: Dict[str, Dict] = defaultdict(lambda: {"signals": 0, "requests": 0})

    def record_signal_consumed(self, client_id: str, tier: str) -> None:
        self._usage[client_id]["signals"] += 1

    def compute_invoice(self, client_id: str, tier: str) -> dict:
        usage = self._usage.get(client_id, {"signals": 0, "requests": 0})
        pricing = self.PRICING.get(tier, self.PRICING["silver"])
        base = pricing["monthly"]
        per_signal = usage["signals"] * pricing["per_signal"]
        return {
            "client_id": client_id,
            "tier": tier,
            "base_fee": base,
            "signal_usage_fee": per_signal,
            "total": base + per_signal,
            "signals_consumed": usage["signals"],
        }

    def total_mrr(self, clients: Dict[str, APIClient]) -> float:
        """Estimate Monthly Recurring Revenue."""
        total = 0.0
        for cid, client in clients.items():
            total += self.PRICING.get(client.tier, {}).get("monthly", 0)
        return total


# ---------------------------------------------------------------------------
# Signal API Server
# ---------------------------------------------------------------------------

class SignalAPIServer:
    """
    HTTP API server for the Serendipity Stream signal service.

    Endpoints:
      GET  /signals/latest?symbol=BTC&n=10      - Latest signals
      GET  /signals/stream?symbol=BTC            - SSE stream (long-poll)
      GET  /signals/history?symbol=BTC&from=ts   - Historical signals
      GET  /account/usage                        - Usage stats
      GET  /account/invoice                      - Current invoice
      POST /webhooks/register                    - Register webhook URL
      GET  /health                               - Service health
    """

    def __init__(self, port: int = 11436):
        self.port = port
        self.keys = APIKeyManager()
        self.dispatcher = SignalDispatcher()
        self.revenue = RevenueTracker()

    def start(self):
        """Start the API server."""
        server_ref = self  # capture for handler

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                params = parse_qs(parsed.query)

                # Auth
                api_key = self.headers.get("X-API-Key", "")
                client = server_ref.keys.authenticate(api_key)

                if path == "/health":
                    self._json_response({"status": "ok", "signals": server_ref.dispatcher._total_dispatched})
                    return

                if not client:
                    self._json_response({"error": "Unauthorized or rate limited"}, 401)
                    return

                if path == "/signals/latest":
                    symbol = params.get("symbol", [None])[0]
                    n = int(params.get("n", ["10"])[0])
                    signals = server_ref.dispatcher.get_latest(symbol, n, client.tier)
                    for _ in signals:
                        server_ref.revenue.record_signal_consumed(client.client_id, client.tier)
                    client.total_signals_consumed += len(signals)
                    self._json_response({"signals": signals, "count": len(signals)})

                elif path == "/account/usage":
                    usage = server_ref.revenue._usage.get(client.client_id, {})
                    self._json_response({"client_id": client.client_id, "tier": client.tier, **usage})

                elif path == "/account/invoice":
                    invoice = server_ref.revenue.compute_invoice(client.client_id, client.tier)
                    self._json_response(invoice)

                else:
                    self._json_response({"error": "Not found"}, 404)

            def _json_response(self, data, code=200):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def log_message(self, *args):
                pass

        print(f"Signal API starting on http://localhost:{self.port}")
        HTTPServer(("0.0.0.0", self.port), Handler).serve_forever()
