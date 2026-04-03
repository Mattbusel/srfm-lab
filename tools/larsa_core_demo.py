"""
larsa_core_demo.py -- Demonstrate Rust-accelerated SRFM performance.

Usage:
    python tools/larsa_core_demo.py
    python tools/larsa_core_demo.py --benchmark  # detailed timing
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))


def demo():
    try:
        import larsa_core
        print(f"larsa_core {larsa_core.__version__} loaded (Rust)")
        rust_available = True
    except ImportError:
        print("larsa_core not built -- run: cd crates/larsa-core && maturin build --release && pip install target/wheels/*.whl")
        print("Falling back to Python simulation for demo...")
        rust_available = False

    # Generate test data
    import numpy as np
    np.random.seed(42)
    n = 50000
    returns = np.random.normal(0, 0.001, n)
    closes = 5000.0 * np.exp(np.cumsum(returns))

    if rust_available:
        # Rust benchmark
        t0 = time.perf_counter()
        for _ in range(10):
            equity, positions, trades = larsa_core.simulate(
                closes.tolist(), 0.005, 1.5, 0.95, 1.0, 0.65
            )
        rust_time = (time.perf_counter() - t0) / 10

        sh = larsa_core.sharpe(equity)
        dd = larsa_core.max_drawdown(equity)
        ret = (equity[-1] - 1.0) * 100

        print(f"\nRust simulate({n} bars):")
        print(f"  Time:    {rust_time*1000:.2f}ms per run")
        print(f"  Sharpe:  {sh:.3f}")
        print(f"  Return:  {ret:+.2f}%")
        print(f"  Max DD:  {dd*100:.1f}%")
        print(f"  Trades:  {trades}")

        # Sweep benchmark
        print(f"\nRust sweep (5x5=25 combinations):")
        t0 = time.perf_counter()
        results = larsa_core.sweep(
            closes.tolist(),
            [0.003, 0.005, 0.007, 0.009, 0.011],
            [0.40, 0.50, 0.60, 0.70, 0.80]
        )
        sweep_time = time.perf_counter() - t0
        print(f"  Time:    {sweep_time:.3f}s for 25 combos")
        best = max(results, key=lambda r: r[2])
        print(f"  Best:    cf={best[0]:.3f} lev={best[1]:.2f} sharpe={best[2]:.3f}")

        # Beta series check
        betas = larsa_core.beta_series(closes.tolist(), 0.005)
        timelike_pct = sum(1 for b in betas if b < 1.0) / len(betas) * 100
        print(f"\nBeta series ({len(betas)} values):")
        print(f"  TIMELIKE: {timelike_pct:.1f}%  SPACELIKE: {100-timelike_pct:.1f}%")


if __name__ == "__main__":
    demo()
