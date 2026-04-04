"""
trade_classification.py — Trade direction classification algorithms.

Covers:
  - Lee-Ready (tick rule + quote rule combined)
  - Ellis-Michaely-O'Hara (EMO) — uses quote midpoint
  - Bulk volume classification (BVC) for high-frequency data
  - Accuracy metrics vs signed trade data
  - Misclassification analysis
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TradeSide(Enum):
    BUY  = 1
    SELL = -1
    UNKNOWN = 0


class ClassificationMethod(Enum):
    LEE_READY = "lee_ready"
    EMO       = "emo"
    BVC       = "bvc"
    TICK      = "tick"
    QUOTE     = "quote"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    timestamp: datetime
    price: float
    size: float
    bid: float
    ask: float
    exchange: str = ""
    true_side: Optional[TradeSide] = None  # for accuracy testing

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def relative_to_mid(self) -> float:
        """Position within bid-ask spread: +1=at ask, -1=at bid."""
        if self.spread == 0:
            return 0.0
        return 2 * (self.price - self.midpoint) / self.spread


@dataclass
class ClassifiedTrade:
    trade: Trade
    side: TradeSide
    method: ClassificationMethod
    confidence: float   # 0-1, higher = more confident


@dataclass
class ClassificationAccuracy:
    method: ClassificationMethod
    n_trades: int
    n_correct: int
    accuracy: float
    n_buy_correct: int
    n_sell_correct: int
    buy_accuracy: float
    sell_accuracy: float
    confusion_matrix: Dict   # {"TP":, "TN":, "FP":, "FN":}

    @property
    def balanced_accuracy(self) -> float:
        return (self.buy_accuracy + self.sell_accuracy) / 2


# ---------------------------------------------------------------------------
# Tick rule
# ---------------------------------------------------------------------------

class TickRuleClassifier:
    """
    Tick rule: sign of price change.
    Up-tick (+Δp) → buy, down-tick (−Δp) → sell, zero-tick → last signed tick.
    """

    def classify(self, trades: List[Trade]) -> List[ClassifiedTrade]:
        result = []
        last_direction = TradeSide.BUY   # initialization convention
        last_price = trades[0].price if trades else 0.0

        for trade in trades:
            dp = trade.price - last_price
            if dp > 0:
                side = TradeSide.BUY
            elif dp < 0:
                side = TradeSide.SELL
            else:
                side = last_direction   # zero-tick: use last signed direction

            conf = 0.6 if dp != 0 else 0.4
            result.append(ClassifiedTrade(trade, side, ClassificationMethod.TICK, conf))
            last_direction = side
            last_price = trade.price

        return result


# ---------------------------------------------------------------------------
# Quote rule
# ---------------------------------------------------------------------------

class QuoteRuleClassifier:
    """
    Quote rule: compare trade price to midpoint.
    price > mid → buy, price < mid → sell.
    price = mid → unclassified / use tick rule.
    """

    EPSILON = 1e-6   # tolerance for midpoint comparison

    def classify_one(self, trade: Trade, last_side: TradeSide) -> ClassifiedTrade:
        mid = trade.midpoint
        if trade.price > mid + self.EPSILON:
            side = TradeSide.BUY
            conf = min(1.0, 0.5 + (trade.price - mid) / max(trade.spread, 0.01))
        elif trade.price < mid - self.EPSILON:
            side = TradeSide.SELL
            conf = min(1.0, 0.5 + (mid - trade.price) / max(trade.spread, 0.01))
        else:
            side = last_side   # at mid: use prior
            conf = 0.3
        return ClassifiedTrade(trade, side, ClassificationMethod.QUOTE, conf)

    def classify(self, trades: List[Trade]) -> List[ClassifiedTrade]:
        result = []
        last_side = TradeSide.BUY
        for trade in trades:
            ct = self.classify_one(trade, last_side)
            result.append(ct)
            last_side = ct.side
        return result


# ---------------------------------------------------------------------------
# Lee-Ready classifier
# ---------------------------------------------------------------------------

class LeeReadyClassifier:
    """
    Lee-Ready (1991): combine quote rule and tick rule.
    1. Apply quote rule (price vs mid).
    2. For trades at the midpoint, apply tick rule.
    The standard implementation with a 5-tick lookback for zero-ticks.
    """

    EPSILON = 1e-6

    def __init__(self):
        self.tick_rule = TickRuleClassifier()
        self.quote_rule = QuoteRuleClassifier()

    def classify(self, trades: List[Trade]) -> List[ClassifiedTrade]:
        # First pass: tick rule for all
        tick_classified = self.tick_rule.classify(trades)
        tick_map = {i: ct.side for i, ct in enumerate(tick_classified)}

        result = []
        last_quote_side = TradeSide.BUY

        for i, trade in enumerate(trades):
            mid = trade.midpoint
            diff = trade.price - mid

            if diff > self.EPSILON:
                # Clear buyer: at or above ask
                side = TradeSide.BUY
                conf = 0.90
            elif diff < -self.EPSILON:
                # Clear seller: at or below bid
                side = TradeSide.SELL
                conf = 0.90
            else:
                # Midpoint: use tick rule
                side = tick_map.get(i, last_quote_side)
                conf = 0.65

            ct = ClassifiedTrade(trade, side, ClassificationMethod.LEE_READY, conf)
            result.append(ct)
            last_quote_side = side

        return result


# ---------------------------------------------------------------------------
# Ellis-Michaely-O'Hara classifier
# ---------------------------------------------------------------------------

class EMOClassifier:
    """
    EMO (2000): improved rule that uses ask/bid directly.
    - price >= ask → buy
    - price <= bid → sell
    - bid < price < ask → tick rule
    Typically more accurate than Lee-Ready for NASDAQ-style markets.
    """

    def __init__(self):
        self.tick_rule = TickRuleClassifier()

    def classify(self, trades: List[Trade]) -> List[ClassifiedTrade]:
        tick_classified = self.tick_rule.classify(trades)
        tick_map = {i: ct.side for i, ct in enumerate(tick_classified)}

        result = []
        for i, trade in enumerate(trades):
            if trade.price >= trade.ask - 1e-8:
                side = TradeSide.BUY
                conf = 0.92
            elif trade.price <= trade.bid + 1e-8:
                side = TradeSide.SELL
                conf = 0.92
            else:
                # Inside spread: use tick rule
                side = tick_map.get(i, TradeSide.BUY)
                conf = 0.62

            result.append(ClassifiedTrade(trade, side, ClassificationMethod.EMO, conf))
        return result


# ---------------------------------------------------------------------------
# Bulk volume classification (BVC)
# ---------------------------------------------------------------------------

class BVCClassifier:
    """
    Bulk Volume Classification (Easley et al. 2012).
    Classifies trades in aggregate bars rather than tick-by-tick.
    Uses the price return to allocate volume between buy/sell.
    BVC uses Z(ΔP/σ(ΔP)) as probability that volume is buy-initiated.
    """

    def __init__(self, window: int = 50):
        self.window = window

    def classify_bars(
        self,
        prices: List[float],
        volumes: List[float],
    ) -> List[Dict]:
        """
        Given OHLCV-style bar data, classify each bar's volume.
        Returns list of {price, volume, buy_vol, sell_vol, p_buy}.
        """
        if len(prices) < 2:
            return []

        returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]

        # Rolling vol of returns
        results = []
        for i, (ret, vol) in enumerate(zip(returns, volumes[1:])):
            window_rets = returns[max(0, i - self.window + 1) : i + 1]
            if len(window_rets) < 2:
                sigma = 0.01
            else:
                sigma = float(np.std(window_rets))
            sigma = max(sigma, 1e-6)

            z = ret / sigma
            p_buy = self._norm_cdf(z)   # probability of buy given return

            buy_vol  = vol * p_buy
            sell_vol = vol * (1 - p_buy)

            results.append({
                "price": prices[i+1],
                "return": ret,
                "total_vol": vol,
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "p_buy": p_buy,
                "order_imbalance": (buy_vol - sell_vol) / vol if vol > 0 else 0.0,
            })

        return results

    def classify_trades_bulk(
        self,
        trades: List[Trade],
        bar_size: int = 100,
    ) -> List[ClassifiedTrade]:
        """Group trades into bars and classify by BVC."""
        classified = []
        for i in range(0, len(trades), bar_size):
            bar = trades[i : i + bar_size]
            if len(bar) < 2:
                for t in bar:
                    classified.append(ClassifiedTrade(t, TradeSide.UNKNOWN, ClassificationMethod.BVC, 0.5))
                continue

            prices = [bar[0].price] + [t.price for t in bar]
            volumes = [0.0] + [t.size for t in bar]
            bar_results = self.classify_bars(prices, volumes)

            for trade, res in zip(bar, bar_results):
                side = TradeSide.BUY if res["p_buy"] >= 0.5 else TradeSide.SELL
                classified.append(ClassifiedTrade(
                    trade, side, ClassificationMethod.BVC, abs(res["p_buy"] - 0.5) * 2
                ))

        return classified

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# Accuracy evaluator
# ---------------------------------------------------------------------------

class ClassificationAccuracyEvaluator:
    """Computes accuracy metrics for classifiers with known true labels."""

    def evaluate(
        self,
        classified: List[ClassifiedTrade],
        method: ClassificationMethod,
    ) -> ClassificationAccuracy:
        labeled = [ct for ct in classified if ct.trade.true_side is not None]
        if not labeled:
            return ClassificationAccuracy(
                method=method, n_trades=0, n_correct=0, accuracy=0.0,
                n_buy_correct=0, n_sell_correct=0, buy_accuracy=0.0, sell_accuracy=0.0,
                confusion_matrix={},
            )

        tp = fp = tn = fn = 0
        buy_correct = buy_total = sell_correct = sell_total = 0

        for ct in labeled:
            true = ct.trade.true_side
            pred = ct.side

            if true == TradeSide.BUY:
                buy_total += 1
                if pred == TradeSide.BUY:
                    tp += 1
                    buy_correct += 1
                else:
                    fn += 1
            elif true == TradeSide.SELL:
                sell_total += 1
                if pred == TradeSide.SELL:
                    tn += 1
                    sell_correct += 1
                else:
                    fp += 1

        n = len(labeled)
        correct = tp + tn
        return ClassificationAccuracy(
            method=method,
            n_trades=n,
            n_correct=correct,
            accuracy=correct / n if n > 0 else 0.0,
            n_buy_correct=buy_correct,
            n_sell_correct=sell_correct,
            buy_accuracy=buy_correct / buy_total if buy_total > 0 else 0.0,
            sell_accuracy=sell_correct / sell_total if sell_total > 0 else 0.0,
            confusion_matrix={"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        )

    def compare_methods(
        self,
        trades: List[Trade],
        methods: List[ClassificationMethod] = None,
    ) -> List[ClassificationAccuracy]:
        """Compare all classifiers on the same trade set."""
        if methods is None:
            methods = list(ClassificationMethod)

        lr = LeeReadyClassifier()
        emo = EMOClassifier()
        bvc = BVCClassifier()
        tick = TickRuleClassifier()
        quote = QuoteRuleClassifier()

        results = []
        for method in methods:
            if method == ClassificationMethod.LEE_READY:
                classified = lr.classify(trades)
            elif method == ClassificationMethod.EMO:
                classified = emo.classify(trades)
            elif method == ClassificationMethod.BVC:
                classified = bvc.classify_trades_bulk(trades)
            elif method == ClassificationMethod.TICK:
                classified = tick.classify(trades)
            elif method == ClassificationMethod.QUOTE:
                classified = quote.classify(trades)
            else:
                continue
            acc = self.evaluate(classified, method)
            results.append(acc)

        return sorted(results, key=lambda a: a.accuracy, reverse=True)


# ---------------------------------------------------------------------------
# Synthetic trade generator (for testing)
# ---------------------------------------------------------------------------

class SyntheticTradeGenerator:
    """Generates synthetic tick data with known true sides for testing."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(
        self,
        n: int = 1000,
        initial_price: float = 100.0,
        spread_pct: float = 0.0005,
        price_vol: float = 0.0002,
    ) -> List[Trade]:
        """Generate synthetic trades with true sides."""
        from datetime import timezone
        trades = []
        price = initial_price
        spread = initial_price * spread_pct

        for i in range(n):
            # Random walk price
            dp = self.rng.normal(0, price * price_vol)
            price = max(1.0, price + dp)

            bid = price - spread / 2
            ask = price + spread / 2
            mid = price

            true_side = TradeSide.BUY if self.rng.random() > 0.5 else TradeSide.SELL

            # Trade price: 60% at bid or ask, 40% in spread
            if self.rng.random() < 0.60:
                trade_price = ask if true_side == TradeSide.BUY else bid
            else:
                trade_price = mid + self.rng.uniform(-spread/4, spread/4)

            # Occasionally trade inside spread or at wrong side
            if self.rng.random() < 0.05:
                trade_price = mid   # 5% at midpoint

            size = abs(self.rng.lognormal(5, 1))

            ts = datetime.now(timezone.utc).replace(microsecond=0)
            trades.append(Trade(
                timestamp=ts,
                price=round(trade_price, 4),
                size=round(size, 0),
                bid=round(bid, 4),
                ask=round(ask, 4),
                true_side=true_side,
            ))

        return trades


# ---------------------------------------------------------------------------
# Misclassification analyzer
# ---------------------------------------------------------------------------

class MisclassificationAnalyzer:
    """Analyzes patterns in classification errors."""

    def analyze(
        self,
        classified: List[ClassifiedTrade],
    ) -> Dict:
        labeled = [ct for ct in classified if ct.trade.true_side is not None]
        if not labeled:
            return {}

        errors = [ct for ct in labeled if ct.side != ct.trade.true_side]
        correct = [ct for ct in labeled if ct.side == ct.trade.true_side]

        if not errors:
            return {"n_errors": 0, "error_rate": 0.0}

        # Error by position in spread
        def spread_pos(ct):
            mid = ct.trade.midpoint
            sp = ct.trade.spread
            if sp == 0: return 0.0
            return (ct.trade.price - mid) / (sp / 2)

        err_pos = [spread_pos(ct) for ct in errors]
        ok_pos  = [spread_pos(ct) for ct in correct]

        # Error by price change magnitude
        all_prices = [ct.trade.price for ct in classified]
        err_dps = []
        ok_dps = []
        for i in range(1, len(classified)):
            dp = abs(classified[i].trade.price - classified[i-1].trade.price)
            if classified[i] in errors:
                err_dps.append(dp)
            else:
                ok_dps.append(dp)

        return {
            "n_errors": len(errors),
            "error_rate": len(errors) / len(labeled),
            "avg_spread_pos_errors": float(np.mean(err_pos)) if err_pos else 0.0,
            "avg_spread_pos_correct": float(np.mean(ok_pos)) if ok_pos else 0.0,
            "pct_errors_at_midpoint": sum(1 for p in err_pos if abs(p) < 0.1) / len(errors),
            "pct_errors_large_dp": sum(1 for d in err_dps if d > 0.01) / len(err_dps) if err_dps else 0.0,
        }


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

class TradeClassificationEngine:
    """Unified API for all classification methods."""

    def __init__(self, default_method: ClassificationMethod = ClassificationMethod.LEE_READY):
        self.default_method = default_method
        self._classifiers = {
            ClassificationMethod.LEE_READY: LeeReadyClassifier(),
            ClassificationMethod.EMO:       EMOClassifier(),
            ClassificationMethod.BVC:       BVCClassifier(),
            ClassificationMethod.TICK:      TickRuleClassifier(),
            ClassificationMethod.QUOTE:     QuoteRuleClassifier(),
        }
        self.evaluator = ClassificationAccuracyEvaluator()
        self.mis_analyzer = MisclassificationAnalyzer()

    def classify(
        self,
        trades: List[Trade],
        method: ClassificationMethod = None,
    ) -> List[ClassifiedTrade]:
        method = method or self.default_method
        clf = self._classifiers[method]
        if method == ClassificationMethod.BVC:
            return clf.classify_trades_bulk(trades)
        return clf.classify(trades)

    def classify_all(self, trades: List[Trade]) -> Dict[str, List[ClassifiedTrade]]:
        return {m.value: self.classify(trades, m) for m in ClassificationMethod}

    def benchmark(self, trades: List[Trade] = None) -> str:
        """Generate synthetic data and benchmark all methods."""
        gen = SyntheticTradeGenerator()
        test_trades = trades or gen.generate(2000)
        accuracies = self.evaluator.compare_methods(test_trades)

        lines = [
            "=== Trade Classification Benchmark ===",
            f"{'Method':>15} {'N':>6} {'Acc':>7} {'BuyAcc':>8} {'SellAcc':>9} {'BalAcc':>8}",
            "-" * 55,
        ]
        for acc in accuracies:
            lines.append(
                f"{acc.method.value:>15} {acc.n_trades:>6} {acc.accuracy:>7.1%} "
                f"{acc.buy_accuracy:>8.1%} {acc.sell_accuracy:>9.1%} {acc.balanced_accuracy:>8.1%}"
            )
        return "\n".join(lines)

    def signed_volume(self, classified: List[ClassifiedTrade]) -> float:
        """Net signed volume: buy - sell."""
        total = 0.0
        for ct in classified:
            if ct.side == TradeSide.BUY:
                total += ct.trade.size
            elif ct.side == TradeSide.SELL:
                total -= ct.trade.size
        return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trade classification CLI")
    parser.add_argument("--action", choices=["benchmark", "classify"], default="benchmark")
    parser.add_argument("--method", default="lee_ready")
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    engine = TradeClassificationEngine()

    if args.action == "benchmark":
        print(engine.benchmark())
    elif args.action == "classify":
        gen = SyntheticTradeGenerator()
        trades = gen.generate(args.n)
        method = ClassificationMethod(args.method)
        classified = engine.classify(trades, method)
        signed_vol = engine.signed_volume(classified)
        buy_count = sum(1 for ct in classified if ct.side == TradeSide.BUY)
        sell_count = sum(1 for ct in classified if ct.side == TradeSide.SELL)
        print(f"Classified {args.n} trades using {method.value}")
        print(f"Buys: {buy_count} ({buy_count/args.n:.1%}), Sells: {sell_count} ({sell_count/args.n:.1%})")
        print(f"Signed volume: {signed_vol:+,.0f}")
