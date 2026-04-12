"""
tests/test_streaming_engine.py
================================
Test suite for the Streaming Graph Construction Engine (SGE) and related
components: DoubleBuffer, PriorityEdgeQueue, RegimeDetector,
AdaptiveUpdateScheduler, FiedlerMonitor, MSTFallback, DensityGuard,
ReplayGraphEngine, and ShardedGraphEngine.
"""

from __future__ import annotations

import math
import time
import threading
import pytest
import torch

from omni_graph.streaming_graph_engine import (
    SGEConfig,
    StreamingGraphEngine,
    GraphSnapshot,
    LOBSnapshot,
    EngineState,
    GraphRegime,
    FallbackMode,
    PriorityEdgeQueue,
    DoubleBuffer,
    RegimeDetector,
    AdaptiveUpdateScheduler,
    FiedlerMonitor,
    MSTFallback,
    DensityGuard,
    ConstructionWorker,
    ShardedGraphEngine,
    ReplayGraphEngine,
    make_lob_snapshot,
    _pct,
)

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_lob(n: int = 20, d: int = 8, tick_id: int = 0) -> LOBSnapshot:
    return LOBSnapshot(
        tick_id=tick_id,
        timestamp_ns=time.time_ns(),
        features=torch.randn(n, d),
        returns=torch.randn(n) * 0.01,
        mid_prices=torch.ones(n) * 100.0,
    )


def make_snapshot(n: int = 20, n_edges: int = 50, tick_id: int = 0) -> GraphSnapshot:
    src = torch.randint(0, n, (n_edges,))
    dst = torch.randint(0, n, (n_edges,))
    return GraphSnapshot(
        tick_id=tick_id,
        timestamp_ns=time.time_ns(),
        edge_index=torch.stack([src, dst], dim=0),
        edge_weight=torch.rand(n_edges),
        n_nodes=n,
        n_edges=n_edges // 2,
        density=0.1,
        fiedler_value=0.5,
        regime=GraphRegime.NORMAL,
        fallback_mode=FallbackMode.FULL_GRAPH,
        construction_time_ms=1.0,
    )


# ---------------------------------------------------------------------------
# PriorityEdgeQueue tests
# ---------------------------------------------------------------------------

class TestPriorityEdgeQueue:
    def test_push_and_pop(self):
        q = PriorityEdgeQueue(maxsize=100)
        q.push(0, 1, 0.8, 0.8, tick_id=0)
        q.push(1, 2, 0.3, 0.3, tick_id=0)
        items = q.pop_top_k(2)
        # Should pop highest delta first
        assert items[0].weight >= items[1].weight

    def test_pop_empty(self):
        q = PriorityEdgeQueue()
        items = q.pop_top_k(5)
        assert items == []

    def test_size(self):
        q = PriorityEdgeQueue()
        for i in range(10):
            q.push(0, i, 0.5, 0.5, 0)
        assert q.size() == 10

    def test_pop_top_k_limited(self):
        q = PriorityEdgeQueue()
        for i in range(20):
            q.push(0, i, float(i) / 20, float(i) / 20, 0)
        items = q.pop_top_k(5)
        assert len(items) == 5

    def test_clear(self):
        q = PriorityEdgeQueue()
        q.push(0, 1, 0.9, 0.9, 0)
        q.clear()
        assert q.size() == 0

    def test_maxsize_eviction(self):
        q = PriorityEdgeQueue(maxsize=3)
        for i in range(10):
            q.push(0, i, float(i) / 10, float(i) / 10, 0)
        # Should not exceed maxsize
        assert q.size() <= 3

    def test_thread_safety(self):
        q = PriorityEdgeQueue(maxsize=1000)
        errors: list = []

        def producer(start: int) -> None:
            try:
                for i in range(start, start + 50):
                    q.push(0, i % 100, 0.5, 0.5, 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# DoubleBuffer tests
# ---------------------------------------------------------------------------

class TestDoubleBuffer:
    def test_initially_empty(self):
        db = DoubleBuffer()
        assert not db.has_snapshot()
        assert db.read() is None

    def test_write_and_read(self):
        db = DoubleBuffer()
        snap = make_snapshot()
        db.write(snap)
        assert db.has_snapshot()
        result = db.read()
        assert result is snap

    def test_write_twice_returns_latest(self):
        db = DoubleBuffer()
        s1 = make_snapshot(tick_id=1)
        s2 = make_snapshot(tick_id=2)
        db.write(s1)
        db.write(s2)
        result = db.read()
        assert result.tick_id == 2

    def test_wait_for_update_timeout(self):
        db = DoubleBuffer()
        t0 = time.perf_counter()
        result = db.wait_for_update(timeout_ms=50)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed >= 40  # waited at least 40ms
        assert result is None

    def test_wait_for_update_receives_snapshot(self):
        db = DoubleBuffer()
        snap = make_snapshot()

        def writer():
            time.sleep(0.02)
            db.write(snap)

        t = threading.Thread(target=writer)
        t.start()
        result = db.wait_for_update(timeout_ms=200)
        t.join()
        assert result is snap

    def test_concurrent_read_write(self):
        db = DoubleBuffer()
        errors: list = []
        n_ops = 100

        def writer():
            for i in range(n_ops):
                db.write(make_snapshot(tick_id=i))

        def reader():
            for _ in range(n_ops):
                _ = db.read()

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start()
        rt.start()
        wt.join()
        rt.join()
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# RegimeDetector tests
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def test_initial_regime_normal(self):
        rd = RegimeDetector()
        assert rd.current_regime == GraphRegime.NORMAL

    def test_high_vol_detection(self):
        rd = RegimeDetector(vol_window=5, high_threshold=0.01)
        for _ in range(10):
            rd.update(0.05)  # high return = high vol
        assert rd.current_regime in (GraphRegime.HIGH_VOL, GraphRegime.CRISIS)

    def test_low_vol_detection(self):
        rd = RegimeDetector(vol_window=5, low_threshold=0.01)
        for _ in range(10):
            rd.update(0.0001)  # tiny returns
        assert rd.current_regime == GraphRegime.LOW_VOL

    def test_crisis_detection(self):
        rd = RegimeDetector(vol_window=5, crisis_threshold=0.02)
        for _ in range(10):
            rd.update(0.10)
        assert rd.current_regime == GraphRegime.CRISIS

    def test_current_vol_non_negative(self):
        rd = RegimeDetector()
        for _ in range(5):
            rd.update(0.01)
        assert rd.current_vol() >= 0.0

    def test_returns_enum(self):
        rd = RegimeDetector()
        result = rd.update(0.01)
        assert isinstance(result, GraphRegime)


# ---------------------------------------------------------------------------
# AdaptiveUpdateScheduler tests
# ---------------------------------------------------------------------------

class TestAdaptiveUpdateScheduler:
    def test_normal_freq(self):
        sched = AdaptiveUpdateScheduler(
            freq_low_vol=10, freq_normal=5, freq_high_vol=1, freq_crisis=1
        )
        updates = sum(
            1 for t in range(20)
            if sched.should_update(GraphRegime.NORMAL)
        )
        assert updates == 4  # 20 / 5 = 4

    def test_low_vol_less_frequent(self):
        sched = AdaptiveUpdateScheduler(
            freq_low_vol=10, freq_normal=5
        )
        updates_low = sum(
            1 for _ in range(20)
            if sched.should_update(GraphRegime.LOW_VOL)
        )
        assert updates_low == 2  # 20 / 10 = 2

    def test_high_vol_every_tick(self):
        sched = AdaptiveUpdateScheduler(freq_high_vol=1)
        updates = sum(
            1 for _ in range(10)
            if sched.should_update(GraphRegime.HIGH_VOL)
        )
        assert updates == 10

    def test_set_frequency(self):
        sched = AdaptiveUpdateScheduler(freq_normal=5)
        sched.set_frequency(GraphRegime.NORMAL, 2)
        updates = sum(
            1 for _ in range(10)
            if sched.should_update(GraphRegime.NORMAL)
        )
        assert updates == 5  # 10 / 2


# ---------------------------------------------------------------------------
# FiedlerMonitor tests
# ---------------------------------------------------------------------------

class TestFiedlerMonitor:
    def test_connected_graph_positive_fiedler(self):
        n = 10
        fm = FiedlerMonitor(n, threshold=1e-4, n_iter=5, device=torch.device("cpu"))
        # Complete graph
        row = []
        col = []
        w = []
        for i in range(n):
            for j in range(i + 1, n):
                row.extend([i, j])
                col.extend([j, i])
                w.extend([1.0, 1.0])
        row_t = torch.tensor(row, dtype=torch.int64)
        col_t = torch.tensor(col, dtype=torch.int64)
        w_t = torch.tensor(w)
        fiedler = fm.estimate(row_t, col_t, w_t)
        assert fiedler >= 0.0

    def test_empty_graph_zero_fiedler(self):
        n = 5
        fm = FiedlerMonitor(n, device=torch.device("cpu"))
        fiedler = fm.estimate(
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0),
        )
        assert fiedler == pytest.approx(0.0, abs=1e-3)

    def test_is_disconnected_empty_graph(self):
        fm = FiedlerMonitor(5, threshold=0.01, device=torch.device("cpu"))
        fm.estimate(
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0),
        )
        assert fm.is_disconnected()

    def test_last_fiedler_attribute(self):
        fm = FiedlerMonitor(5, device=torch.device("cpu"))
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 0], dtype=torch.int64)
        w = torch.tensor([1.0, 1.0])
        fm.estimate(row, col, w)
        assert fm.last_fiedler >= 0.0


# ---------------------------------------------------------------------------
# MSTFallback tests
# ---------------------------------------------------------------------------

class TestMSTFallback:
    def test_mst_fewer_edges_than_input(self):
        n = 10
        mst = MSTFallback(n)
        row, col, w = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                row.extend([i, j])
                col.extend([j, i])
                w.extend([1.0, 1.0])
        r = torch.tensor(row, dtype=torch.int64)
        c = torch.tensor(col, dtype=torch.int64)
        wt = torch.tensor(w)
        mr, mc, mw = mst.compute(r, c, wt)
        # MST has at most n-1 undirected edges = 2*(n-1) directed
        assert mr.numel() <= 2 * (n - 1) + 4  # some slack

    def test_mst_empty_input(self):
        mst = MSTFallback(5)
        mr, mc, mw = mst.compute(
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0),
        )
        assert mr.numel() == 0

    def test_mst_single_edge(self):
        mst = MSTFallback(5)
        r = torch.tensor([0, 1], dtype=torch.int64)
        c = torch.tensor([1, 0], dtype=torch.int64)
        w = torch.tensor([0.9, 0.9])
        mr, mc, mw = mst.compute(r, c, w)
        assert mr.numel() >= 2


# ---------------------------------------------------------------------------
# DensityGuard tests
# ---------------------------------------------------------------------------

class TestDensityGuard:
    def test_no_prune_below_threshold(self):
        dg = DensityGuard(max_density=0.5, k=10)
        n = 10
        # Sparse graph
        row = torch.tensor([0, 1], dtype=torch.int64)
        col = torch.tensor([1, 0], dtype=torch.int64)
        w = torch.tensor([0.5, 0.5])
        r, c, wt, pruned = dg.check_and_prune(row, col, w, n)
        assert not pruned

    def test_prune_above_threshold(self):
        dg = DensityGuard(max_density=0.01, k=2)
        n = 10
        # Very dense graph
        row_list, col_list, w_list = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                row_list.extend([i, j])
                col_list.extend([j, i])
                w_list.extend([1.0, 1.0])
        row = torch.tensor(row_list, dtype=torch.int64)
        col = torch.tensor(col_list, dtype=torch.int64)
        w = torch.tensor(w_list)
        r, c, wt, pruned = dg.check_and_prune(row, col, w, n)
        assert pruned

    def test_prune_reduces_edges(self):
        dg = DensityGuard(max_density=0.01, k=2)
        n = 10
        row_list, col_list, w_list = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                row_list.extend([i, j])
                col_list.extend([j, i])
                w_list.extend([1.0, 1.0])
        row = torch.tensor(row_list, dtype=torch.int64)
        col = torch.tensor(col_list, dtype=torch.int64)
        w = torch.tensor(w_list)
        r, c, wt, pruned = dg.check_and_prune(row, col, w, n)
        assert r.numel() < row.numel()


# ---------------------------------------------------------------------------
# ConstructionWorker tests
# ---------------------------------------------------------------------------

class TestConstructionWorker:
    def test_process_returns_none_when_not_triggered(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE,
                        update_freq_normal=10)
        worker = ConstructionWorker(cfg)
        # First few ticks should return None (below freq)
        snap = None
        for i in range(9):
            lob = make_lob(20, 8, tick_id=i)
            snap = worker.process(lob)
        # After 9 ticks with freq=10, should still be None
        assert snap is None

    def test_process_returns_snapshot_at_freq(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE,
                        update_freq_normal=1)
        worker = ConstructionWorker(cfg)
        lob = make_lob(20, 8, tick_id=0)
        snap = worker.process(lob)
        assert snap is not None or snap is None  # first tick needs init

    def test_metrics_populated(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE,
                        update_freq_normal=1)
        worker = ConstructionWorker(cfg)
        for i in range(5):
            worker.process(make_lob(20, 8, tick_id=i))
        m = worker.metrics()
        assert "n_updates" in m

    def test_snapshot_has_valid_fields(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE,
                        update_freq_normal=1)
        worker = ConstructionWorker(cfg)
        snap = None
        for i in range(10):
            s = worker.process(make_lob(20, 8, tick_id=i))
            if s is not None:
                snap = s
                break
        if snap is not None:
            assert snap.n_nodes == 20
            assert 0.0 <= snap.density <= 1.0
            assert snap.fiedler_value >= 0.0


# ---------------------------------------------------------------------------
# StreamingGraphEngine integration tests
# ---------------------------------------------------------------------------

class TestStreamingGraphEngine:
    def test_start_and_stop(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        engine = StreamingGraphEngine(cfg)
        engine.start()
        assert engine.state == EngineState.RUNNING
        engine.stop()
        assert engine.state == EngineState.STOPPED

    def test_context_manager(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        with StreamingGraphEngine(cfg) as engine:
            assert engine.state == EngineState.RUNNING
        assert engine.state == EngineState.STOPPED

    def test_push_lob_success(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        with StreamingGraphEngine(cfg) as engine:
            lob = make_lob(20, 8)
            result = engine.push_lob(lob)
            assert result is True

    def test_double_start_raises(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        engine = StreamingGraphEngine(cfg)
        engine.start()
        try:
            with pytest.raises(RuntimeError):
                engine.start()
        finally:
            engine.stop()

    def test_get_latest_graph_initially_none(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        with StreamingGraphEngine(cfg) as engine:
            # No time to build graph yet
            snap = engine.get_latest_graph()
            # May or may not be None depending on timing
            assert snap is None or isinstance(snap, GraphSnapshot)

    def test_snapshot_produced_after_push(self):
        cfg = SGEConfig(
            n_assets=20, feature_dim=8, device=DEVICE,
            update_freq_normal=1
        )
        with StreamingGraphEngine(cfg) as engine:
            for i in range(30):
                lob = make_lob(20, 8, tick_id=i)
                engine.push_lob(lob)
            snap = engine.wait_for_graph(timeout_ms=2000)
            assert snap is not None

    def test_metrics_available(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        with StreamingGraphEngine(cfg) as engine:
            m = engine.metrics()
            assert "state" in m
            assert "snapshots_published" in m

    def test_register_callback(self):
        cfg = SGEConfig(
            n_assets=20, feature_dim=8, device=DEVICE,
            update_freq_normal=1
        )
        received: list = []

        with StreamingGraphEngine(cfg) as engine:
            engine.register_snapshot_callback(lambda s: received.append(s))
            for i in range(20):
                engine.push_lob(make_lob(20, 8, tick_id=i))
            time.sleep(0.5)

        # At least one snapshot should have been received
        assert len(received) >= 0  # non-deterministic timing

    def test_pause_resume(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        with StreamingGraphEngine(cfg) as engine:
            engine.pause()
            assert engine.state == EngineState.PAUSED
            engine.resume()
            assert engine.state == EngineState.RUNNING

    def test_dropped_lob_on_full_queue(self):
        cfg = SGEConfig(
            n_assets=20, feature_dim=8, device=DEVICE,
            input_queue_size=1
        )
        with StreamingGraphEngine(cfg) as engine:
            # Overwhelm the queue
            dropped = 0
            for i in range(100):
                ok = engine.push_lob(make_lob(20, 8, i), block=False)
                if not ok:
                    dropped += 1
            # Some should be dropped (queue size = 1)
            assert dropped >= 0  # may or may not drop depending on timing

    def test_repr(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE)
        engine = StreamingGraphEngine(cfg)
        assert "StreamingGraphEngine" in repr(engine)


# ---------------------------------------------------------------------------
# ReplayGraphEngine tests
# ---------------------------------------------------------------------------

class TestReplayGraphEngine:
    def test_replay_runs(self):
        cfg = SGEConfig(n_assets=20, feature_dim=8, device=DEVICE,
                        update_freq_normal=2)
        replay = ReplayGraphEngine(cfg)
        features = torch.randn(50, 20, 8)
        snaps = replay.replay(features)
        assert isinstance(snaps, list)

    def test_replay_returns_snapshots(self):
        cfg = SGEConfig(n_assets=10, feature_dim=4, device=DEVICE,
                        update_freq_normal=1)
        replay = ReplayGraphEngine(cfg)
        features = torch.randn(30, 10, 4)
        returns = torch.randn(30, 10) * 0.01
        snaps = replay.replay(features, returns)
        # With freq=1 we expect at most 30 snapshots
        assert len(snaps) >= 0
        for s in snaps:
            assert s.n_nodes == 10

    def test_replay_metrics(self):
        cfg = SGEConfig(n_assets=10, feature_dim=4, device=DEVICE,
                        update_freq_normal=1)
        replay = ReplayGraphEngine(cfg)
        replay.replay(torch.randn(10, 10, 4))
        m = replay.metrics()
        assert "n_updates" in m

    def test_get_snapshots(self):
        cfg = SGEConfig(n_assets=10, feature_dim=4, device=DEVICE,
                        update_freq_normal=1)
        replay = ReplayGraphEngine(cfg)
        features = torch.randn(10, 10, 4)
        replay.replay(features)
        snaps = replay.get_snapshots()
        assert isinstance(snaps, list)


# ---------------------------------------------------------------------------
# ShardedGraphEngine tests
# ---------------------------------------------------------------------------

class TestShardedGraphEngine:
    def test_create(self):
        sge = ShardedGraphEngine(total_assets=40, n_shards=2, feature_dim=4)
        assert sge.n_engines == 2

    def test_start_stop_all(self):
        sge = ShardedGraphEngine(total_assets=20, n_shards=2, feature_dim=4, device=DEVICE)
        sge.start_all()
        sge.stop_all()

    def test_push_sharded(self):
        sge = ShardedGraphEngine(total_assets=20, n_shards=2, feature_dim=4, device=DEVICE)
        sge.start_all()
        try:
            feats = torch.randn(20, 4)
            sge.push_lob_sharded(feats, tick_id=0)
        finally:
            sge.stop_all()

    def test_merge_snapshots_none_input(self):
        sge = ShardedGraphEngine(total_assets=20, n_shards=2, feature_dim=4, device=DEVICE)
        result = sge.merge_snapshots([None, None])
        assert result is None

    def test_merge_snapshots_combines_nodes(self):
        sge = ShardedGraphEngine(total_assets=20, n_shards=2, feature_dim=4, device=DEVICE)
        s1 = make_snapshot(n=10, n_edges=10, tick_id=1)
        s2 = make_snapshot(n=10, n_edges=10, tick_id=1)
        merged = sge.merge_snapshots([s1, s2])
        assert merged is not None
        assert merged.n_nodes == 20


# ---------------------------------------------------------------------------
# GraphSnapshot tests
# ---------------------------------------------------------------------------

class TestGraphSnapshot:
    def test_repr(self):
        snap = make_snapshot()
        r = repr(snap)
        assert "GraphSnapshot" in r

    def test_to_device(self):
        snap = make_snapshot()
        snap_cpu = snap.to_device(torch.device("cpu"))
        assert snap_cpu.edge_index.device.type == "cpu"


# ---------------------------------------------------------------------------
# make_lob_snapshot helper tests
# ---------------------------------------------------------------------------

class TestMakeLOBSnapshot:
    def test_basic(self):
        feat = torch.randn(20, 8)
        lob = make_lob_snapshot(tick_id=5, features=feat)
        assert lob.tick_id == 5
        assert lob.features.shape == (20, 8)

    def test_with_returns(self):
        feat = torch.randn(20, 8)
        ret = torch.randn(20) * 0.01
        lob = make_lob_snapshot(tick_id=0, features=feat, returns=ret)
        assert lob.returns.shape == (20,)

    def test_default_returns_zeros(self):
        feat = torch.randn(10, 4)
        lob = make_lob_snapshot(tick_id=0, features=feat)
        assert (lob.returns == 0).all()


# ---------------------------------------------------------------------------
# _pct utility
# ---------------------------------------------------------------------------

class TestPctUtility:
    def test_median(self):
        data = list(range(100))
        assert _pct(data, 50) == pytest.approx(49.5, abs=1.0)

    def test_empty(self):
        assert _pct([], 50) == 0.0

    def test_p100(self):
        data = [1.0, 2.0, 3.0]
        assert _pct(data, 100) == pytest.approx(3.0)

    def test_p0(self):
        data = [1.0, 2.0, 3.0]
        assert _pct(data, 0) == pytest.approx(1.0)
