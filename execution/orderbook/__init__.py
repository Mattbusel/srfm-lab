"""
execution/orderbook
====================
L2 order-book data feeds and the BookManager that wires them to the
execution layer.

Public surface
--------------
.. code-block:: python

    from execution.orderbook import (
        OrderBook,
        AlpacaL2Feed,
        BinanceL2Feed,
        BookManager,
        FeedMonitor,
        InsufficientLiquidityError,
    )

Typical setup in an async main
-------------------------------
.. code-block:: python

    import asyncio
    from execution.orderbook import BookManager, FeedMonitor

    async def main():
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

        bm = BookManager(symbols=symbols)
        monitor = FeedMonitor(bm, symbols)

        await bm.start()
        await monitor.start()

        # Wire to SmartRouter
        from execution.routing.smart_router import SmartRouter
        router = SmartRouter(
            broker=broker,
            spread_feed=lambda sym: bm.get_bid_ask(sym),
            book_manager=bm,
        )

        await asyncio.sleep(3600)  # run for an hour

    asyncio.run(main())
"""

from .orderbook import OrderBook, InsufficientLiquidityError
from .alpaca_l2_feed import AlpacaL2Feed
from .binance_l2_feed import BinanceL2Feed
from .book_manager import BookManager
from .feed_monitor import FeedMonitor

__all__ = [
    "OrderBook",
    "InsufficientLiquidityError",
    "AlpacaL2Feed",
    "BinanceL2Feed",
    "BookManager",
    "FeedMonitor",
]
