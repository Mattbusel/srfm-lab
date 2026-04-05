# ============================================================
# Notebook 26: DeFi Protocol Analysis & AMM Dynamics
# ============================================================
# Topics:
#   1. AMM constant-product mechanics
#   2. Impermanent loss deep dive
#   3. Concentrated liquidity (Uniswap v3)
#   4. StableSwap invariant
#   5. Yield optimization across protocols
#   6. LP return simulation
#   7. MEV and sandwich attack analysis
#   8. Capital efficiency comparison
# ============================================================

using Statistics, LinearAlgebra

println("="^60)
println("Notebook 26: DeFi Protocol Analysis")
println("="^60)

# ── Section 1: AMM Constant-Product Mechanics ─────────────

println("\n--- Section 1: Constant-Product AMM ---")

# Initialize a USDC/ETH pool (x=USDC, y=ETH)
# Represents roughly $100M TVL with ETH at $2000
reserve_x = 50_000_000.0   # 50M USDC
reserve_y = 25_000.0        # 25k ETH
fee_rate = 0.003            # 30 bps
k = reserve_x * reserve_y   # invariant

println("Initial pool state:")
println("  USDC reserve: \$$(round(reserve_x/1e6, digits=1))M")
println("  ETH reserve:  $(reserve_y) ETH")
println("  ETH price:    \$$(round(reserve_x/reserve_y, digits=2))")
println("  Invariant k:  $(round(k, digits=0))")

function amm_swap(rx, ry, dx, fee)
    """Buy ry (sell rx). Returns (dy, new_rx, new_ry)."""
    k_local = rx * ry
    dx_eff = dx * (1.0 - fee)
    new_rx = rx + dx_eff
    new_ry = k_local / new_rx
    dy = ry - new_ry
    return dy, rx + dx, new_ry
end

# Simulate a large buy of ETH with USDC
buy_sizes = [100_000, 500_000, 1_000_000, 5_000_000]
println("\nPrice impact of ETH purchases:")
println("  Amount USDC | ETH Received | Avg Price  | Impact (bps)")
println("  " * "-"^58)
for size in buy_sizes
    dy, new_rx, new_ry = amm_swap(reserve_x, reserve_y, Float64(size), fee_rate)
    avg_price = size / dy
    mid_price = reserve_x / reserve_y
    impact_bps = (avg_price - mid_price) / mid_price * 10_000
    @printf "  \$%9.0f  | %9.4f ETH | \$%9.2f | %7.1f bps\n" size dy avg_price impact_bps
end

# ── Section 2: Impermanent Loss Deep Dive ─────────────────

println("\n--- Section 2: Impermanent Loss Analysis ---")

function impermanent_loss(price_ratio)
    r = price_ratio
    return 2.0 * sqrt(r) / (1.0 + r) - 1.0
end

price_ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
println("IL vs Price Ratio (ETH/initial):")
println("  Price Ratio | IL %    | Fee to break-even (30d, 10% vol/day)")
println("  " * "-"^55)
for r in price_ratios
    il = impermanent_loss(r)
    # Break-even: fee_income = |IL| * TVL
    # fee_income = fee_rate * volume_daily * 30
    # volume_daily ≈ alpha * TVL (assume 50% turnover)
    breakeven_fee = abs(il) / (0.5 * 30.0)  # fraction per unit volume
    @printf "  %10.2fx  | %6.2f%% | %5.2f%%\n" r il*100 breakeven_fee*100
end

# Monte Carlo IL distribution
println("\nMonte Carlo IL distribution (ETH, 30-day horizon, 80% annualized vol):")
sigma = 0.80
T = 30.0 / 365.0
n_paths = 10_000
state = UInt64(42)
ils = zeros(n_paths)
for i in 1:n_paths
    state = state * 6364136223846793005 + 1442695040888963407
    u1 = max((state >> 11) / Float64(2^53), 1e-15)
    state = state * 6364136223846793005 + 1442695040888963407
    u2 = (state >> 11) / Float64(2^53)
    z = sqrt(-2.0 * log(u1)) * cos(2π * u2)
    price_ratio = exp((0.0 - 0.5*sigma^2)*T + sigma*sqrt(T)*z)
    ils[i] = impermanent_loss(price_ratio)
end
println("  Mean IL: $(round(mean(ils)*100, digits=2))%")
println("  5th pct: $(round(quantile(sort(ils), 0.05)*100, digits=2))%")
println("  25th pct: $(round(quantile(sort(ils), 0.25)*100, digits=2))%")
println("  Probability IL > 1%: $(round(mean(ils .< -0.01)*100, digits=1))%")
println("  Probability IL > 5%: $(round(mean(ils .< -0.05)*100, digits=1))%")

# ── Section 3: Concentrated Liquidity (v3) ───────────────

println("\n--- Section 3: Uniswap v3 Concentrated Liquidity ---")

const TICK_BASE = 1.0001
tick_to_price(t) = TICK_BASE^t
price_to_tick(p) = round(Int, log(p) / log(TICK_BASE))

current_eth_price = 2000.0
println("Current ETH price: \$$(current_eth_price)")

# Different range widths
ranges = [
    (0.50, 2.0, "±50% range"),
    (0.80, 1.25, "±20% range"),
    (0.90, 1.10, "±10% range"),
    (0.95, 1.05, "±5% range"),
]

println("\nCapital efficiency vs range width:")
println("  Range           | pa      | pb      | Cap. Efficiency")
println("  " * "-"^55)
for (lo, hi, label) in ranges
    pa = current_eth_price * lo
    pb = current_eth_price * hi
    # CE = 1 / (1 - (pa/pb)^(1/4))
    r_ratio = (pa/pb)^0.25
    ce = 1.0 / max(1.0 - r_ratio, 0.001)
    @printf "  %-16s | \$%6.0f  | \$%6.0f  | %7.1fx\n" label pa pb ce
end

# Token amounts for a $1M position in each range
println("\nToken amounts for \$1M position:")
for (lo, hi, label) in ranges
    pa = current_eth_price * lo
    pb = current_eth_price * hi
    p = current_eth_price
    L = 1_000_000.0  # notional liquidity
    if p <= pa
        ax = L * (1.0/sqrt(pa) - 1.0/sqrt(pb))
        ay = 0.0
    elseif p >= pb
        ax = 0.0
        ay = L * (sqrt(pb) - sqrt(pa))
    else
        ax = L * (1.0/sqrt(p) - 1.0/sqrt(pb))
        ay = L * (sqrt(p) - sqrt(pa))
    end
    # Normalize to $1M
    total_usd = ax * p + ay
    scale = 1_000_000.0 / max(total_usd, 1.0)
    ax *= scale; ay *= scale
    @printf "  %-16s | %7.2f ETH + \$%9.0f USDC\n" label ax ay
end

# ── Section 4: StableSwap Invariant ──────────────────────

println("\n--- Section 4: StableSwap (Curve) ---")

function stableswap_D(reserves, A)
    n = length(reserves)
    S = sum(reserves)
    D = S
    Ann = A * n^n
    for _ in 1:256
        D_P = D
        for xi in reserves
            D_P = D_P * D / (n * xi + 1e-18)
        end
        D_prev = D
        D = (Ann * S + n * D_P) * D / ((Ann - 1) * D + (n + 1) * D_P)
        if abs(D - D_prev) < 1e-12; break; end
    end
    return D
end

function stableswap_out(reserves, i, j, dx, A, fee)
    n = length(reserves)
    Ann = A * n^n
    dx_eff = dx * (1.0 - fee)
    new_res = copy(reserves)
    new_res[i] += dx_eff
    D = stableswap_D(reserves, A)
    S_ = sum(new_res) - new_res[j]
    c = D
    for k in 1:n
        if k == j; continue; end
        c = c * D / (n * new_res[k])
    end
    c = c * D / (n * Ann)
    b = S_ + D / Ann
    y = D
    for _ in 1:256
        y_prev = y
        y = (y^2 + c) / (2*y + b - D)
        if abs(y - y_prev) < 1e-12; break; end
    end
    return reserves[j] - y
end

# 3pool: USDC/USDT/DAI with $100M each
reserves_3pool = [100e6, 100e6, 100e6]
A_values = [10.0, 100.0, 500.0, 2000.0]

println("StableSwap price impact (\$1M trade USDC→USDT):")
println("  A param | Output USDT    | Slippage (bps)")
println("  " * "-"^40)
for A in A_values
    dy = stableswap_out(reserves_3pool, 1, 2, 1e6, A, 0.0004)
    slip = (1e6 - dy) / 1e6 * 10_000
    @printf "  %7.0f | %14.2f | %9.2f\n" A dy slip
end

# ── Section 5: Yield Optimization ────────────────────────

println("\n--- Section 5: Yield Optimization ---")

# Compare protocols: APY, TVL (proxy for safety), IL risk
protocols = [
    ("USDC/ETH 0.3% v2",     0.35, 500e6, 0.80, true),
    ("USDC/ETH 0.05% v3",    0.45, 200e6, 0.60, true),
    ("ETH/WBTC 0.3%",        0.20, 150e6, 0.30, true),
    ("USDC/USDT Curve",      0.05,  2e9,  0.01, false),
    ("ETH Staking",          0.04,  5e9,  0.00, false),
    ("ETH/STETH Curve",      0.08,  1e9,  0.02, false),
]

println("Protocol comparison:")
println("  Protocol                | APY    | TVL      | IL Risk | Risk-adj APY")
println("  " * "-"^65)
for (name, apy, tvl, il_risk, has_il) in protocols
    # Risk-adjusted APY: discount for IL risk
    risk_adj = has_il ? apy - 0.5 * il_risk : apy
    @printf "  %-23s | %5.1f%% | \$%6.0fM | %5.1f%%  | %6.2f%%\n" name apy*100 tvl/1e6 il_risk*100 risk_adj*100
end

# Optimal allocation using simplified mean-variance
apys_vec = [p[2] for p in protocols]
risks_vec = [p[4] for p in protocols]
lambda = 2.0
weights = max.(apys_vec .- lambda .* risks_vec, 0.0)
weights ./= sum(weights) + 1e-12
println("\nOptimal allocation (lambda=2.0):")
for (i, (name, _, _, _, _)) in enumerate(protocols)
    if weights[i] > 0.01
        @printf "  %-23s: %5.1f%%\n" name weights[i]*100
    end
end
portfolio_apy = dot(weights, apys_vec) * 100
portfolio_risk = dot(weights, risks_vec) * 100
println("Portfolio APY: $(round(portfolio_apy, digits=2))%, Risk: $(round(portfolio_risk, digits=2))%")

# ── Section 6: LP Return Simulation ──────────────────────

println("\n--- Section 6: LP Return Simulation ---")

# Simulate 1 year of LP returns vs HODL for USDC/ETH pool
function simulate_lp_vs_hodl(S0, mu, sigma, T_days, fee_rate, vol_ratio, n_paths)
    dt = 1.0/365.0
    state = UInt64(777)
    lp_rets = zeros(n_paths)
    hodl_rets = zeros(n_paths)
    for i in 1:n_paths
        S = S0
        cum_fee = 0.0
        initial_val = 2 * S0  # 1 unit ETH + S0 USDC
        for _ in 1:T_days
            state = state * 6364136223846793005 + 1442695040888963407
            u1 = max((state >> 11) / Float64(2^53), 1e-15)
            state = state * 6364136223846793005 + 1442695040888963407
            u2 = (state >> 11) / Float64(2^53)
            z = sqrt(-2.0 * log(u1)) * cos(2π * u2)
            S *= exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            tvl = initial_val * 2.0 * sqrt(S/S0)  # LP value
            cum_fee += tvl * vol_ratio * fee_rate
        end
        r = S / S0
        lp_val = initial_val * (2*sqrt(r)/(1+r)) + cum_fee
        hodl_val = 1.0 * S + S0  # 1 ETH + S0 USDC
        lp_rets[i] = (lp_val - initial_val) / initial_val
        hodl_rets[i] = (hodl_val - initial_val) / initial_val
    end
    return lp_rets, hodl_rets
end

println("Simulating 1-year LP vs HODL (1000 paths, ETH vol=80%, vol_ratio=0.5):")
lp_r, hodl_r = simulate_lp_vs_hodl(2000.0, 0.0, 0.80, 365, 0.003, 0.5, 1000)
println("  LP Return:   mean=$(round(mean(lp_r)*100,digits=1))%, std=$(round(std(lp_r)*100,digits=1))%")
println("  HODL Return: mean=$(round(mean(hodl_r)*100,digits=1))%, std=$(round(std(hodl_r)*100,digits=1))%")
println("  LP > HODL:   $(round(mean(lp_r .> hodl_r)*100, digits=1))% of paths")
lp_sharpe = mean(lp_r) / std(lp_r)
hodl_sharpe = mean(hodl_r) / std(hodl_r)
println("  LP Sharpe:   $(round(lp_sharpe, digits=3))")
println("  HODL Sharpe: $(round(hodl_sharpe, digits=3))")

# ── Section 7: MEV / Sandwich Attacks ────────────────────

println("\n--- Section 7: MEV & Sandwich Attacks ---")

function sandwich_profit(rx, ry, fee, victim_dx, front_dx)
    k_inv = rx * ry
    # Front run: attacker buys ETH with front_dx USDC
    dx_eff_f = front_dx * (1.0 - fee)
    new_rx_f = rx + dx_eff_f
    new_ry_f = k_inv / new_rx_f
    dy_f = ry - new_ry_f  # ETH bought

    # Victim swap
    rx2 = rx + front_dx
    ry2 = new_ry_f
    k2 = rx2 * ry2
    dx_eff_v = victim_dx * (1.0 - fee)
    new_rx2 = rx2 + dx_eff_v
    new_ry2 = k2 / new_rx2

    # Back run: attacker sells dy_f ETH back
    rx3 = new_rx2; ry3 = new_ry2
    k3 = rx3 * ry3
    dy_eff_b = dy_f * (1.0 - fee)
    new_ry3 = ry3 + dy_eff_b
    new_rx3 = k3 / new_ry3
    dx_back = rx3 - new_rx3  # USDC received

    return max(dx_back - front_dx, 0.0)
end

victim_sizes = [10_000.0, 50_000.0, 100_000.0, 500_000.0]
attacker_capital = 1_000_000.0
println("Sandwich attack profitability (attacker capital: \$1M):")
println("  Victim Order | Profit     | ROI (bps)")
println("  " * "-"^40)
for vs in victim_sizes
    profit = sandwich_profit(reserve_x, reserve_y, fee_rate, vs, attacker_capital)
    roi = profit / attacker_capital * 10_000
    @printf "  \$%10.0f | \$%9.2f | %7.1f bps\n" vs profit roi
end

# ── Section 8: Capital Efficiency Summary ─────────────────

println("\n--- Section 8: Capital Efficiency Summary ---")

println("Comparing v2 vs v3 LP efficiency:")
println("")
scenarios = [
    ("Stable range ±10%", 0.90, 1.10),
    ("Medium range ±25%", 0.75, 1.25),
    ("Wide range ±50%",   0.50, 1.50),
    ("Full range",         0.01, 100.0),
]

for (label, lo, hi) in scenarios
    pa = 2000.0 * lo
    pb = 2000.0 * hi
    r = (pa/pb)^0.25
    ce = r < 1.0 ? 1.0 / max(1.0 - r, 0.001) : 1.0
    # For v3, fee yield is amplified proportionally
    v2_fee_apy = 0.35  # example 35% APY for v2
    v3_fee_apy = v2_fee_apy * ce
    # But IL is also amplified for narrow ranges
    il_amplifier = ce  # rough approximation
    @printf "  %-20s: CE=%6.1fx, v3 Fee APY=%5.0f%%, IL amplifier=%5.1fx\n" label ce v3_fee_apy*100 il_amplifier
end

println("\n✓ Notebook 26 complete")

macro printf(fmt, args...)
    :(Printf.@printf($fmt, $(args...)))
end

# Re-implement simple formatted output
function @sprintf(fmt, args...)
    return string(args...)
end
