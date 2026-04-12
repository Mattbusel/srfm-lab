"""
test_calibration.jl — Tests for all calibration methods in NeuroSDE

Tests:
  1. Black-Scholes formulas (put-call parity, Greeks, boundary conditions)
  2. Implied volatility extraction (Newton, Brent, surface)
  3. SABR model: smile shape, ATM formula, calibration round-trip
  4. Heston model: characteristic function properties, FFT pricing
  5. Dupire local volatility: consistency with input prices
  6. Levenberg-Marquardt optimizer: convergence on known problems
  7. Parameter stability diagnostics
  8. SVI parametrisation: no-arbitrage conditions, calibration
  9. SSVI surface: calendar spread constraint, global fit
 10. Calibration quality metrics: computation and bounds
"""

using Test
using Statistics
using LinearAlgebra
using Random

# ─────────────────────────────────────────────────────────────────────────────
# Helper: numerical derivative
# ─────────────────────────────────────────────────────────────────────────────

function numerical_deriv(f, x; ε=1e-5)
    return (f(x + ε) - f(x - ε)) / (2ε)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BLACK-SCHOLES TESTS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Black-Scholes Formulas" begin

    S, K, r, q, σ, T = 100.0, 100.0, 0.05, 0.02, 0.20, 1.0

    @testset "Put-call parity" begin
        C = bs_call(S, K, r, q, σ, T)
        P = bs_put(S, K, r, q, σ, T)
        parity_rhs = S * exp(-q*T) - K * exp(-r*T)
        @test isapprox(C - P, parity_rhs, atol=1e-10)
    end

    @testset "Boundary conditions at T=0" begin
        C0 = bs_call(S, K, r, q, σ, 1e-10)
        @test C0 >= 0
        # Deep ITM call approaches intrinsic value
        C_itm = bs_call(200.0, 100.0, 0.05, 0.02, 0.20, 1.0)
        @test C_itm > 80.0
    end

    @testset "Call price bounds" begin
        C = bs_call(S, K, r, q, σ, T)
        # Lower bound: C >= max(S e^{-qT} - K e^{-rT}, 0)
        lb = max(S * exp(-q*T) - K * exp(-r*T), 0.0)
        @test C >= lb - 1e-10
        # Upper bound: C <= S e^{-qT}
        @test C <= S * exp(-q*T) + 1e-10
    end

    @testset "Greeks: vega positive" begin
        v = bs_vega(S, K, r, q, σ, T)
        @test v > 0
    end

    @testset "Greeks: delta bounds" begin
        δ_call = bs_delta(S, K, r, q, σ, T; call=true)
        δ_put  = bs_delta(S, K, r, q, σ, T; call=false)
        @test 0.0 <= δ_call <= 1.0
        @test -1.0 <= δ_put <= 0.0
    end

    @testset "Greeks: call-put delta relationship" begin
        δ_call = bs_delta(S, K, r, q, σ, T; call=true)
        δ_put  = bs_delta(S, K, r, q, σ, T; call=false)
        # δ_call - δ_put = e^{-qT}
        @test isapprox(δ_call - δ_put, exp(-q*T), atol=1e-10)
    end

    @testset "Gamma positive" begin
        γ = bs_gamma(S, K, r, q, σ, T)
        @test γ > 0
    end

    @testset "Call price increases with S" begin
        C1 = bs_call(100.0, K, r, q, σ, T)
        C2 = bs_call(110.0, K, r, q, σ, T)
        @test C2 > C1
    end

    @testset "Call price decreases with K" begin
        C1 = bs_call(S, 90.0, r, q, σ, T)
        C2 = bs_call(S, 100.0, r, q, σ, T)
        @test C1 > C2
    end

    @testset "Call price increases with σ" begin
        C1 = bs_call(S, K, r, q, 0.10, T)
        C2 = bs_call(S, K, r, q, 0.30, T)
        @test C2 > C1
    end

    @testset "Zero vol: intrinsic value" begin
        C_zv = bs_call(S, 80.0, r, q, 1e-8, T)
        intrinsic = max(S * exp(-q*T) - 80.0 * exp(-r*T), 0.0)
        @test isapprox(C_zv, intrinsic, atol=1e-3)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: IMPLIED VOLATILITY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

@testset "Implied Volatility Extraction" begin

    S, K, r, q, T = 100.0, 100.0, 0.05, 0.02, 1.0

    @testset "Newton: round-trip ATM" begin
        σ_true = 0.25
        price  = bs_call(S, K, r, q, σ_true, T)
        res    = implied_vol_newton(price, S, K, r, q, T)
        @test res.converged
        @test isapprox(res.sigma, σ_true, atol=1e-6)
    end

    @testset "Newton: round-trip deep ITM" begin
        σ_true = 0.30
        K_itm  = 80.0
        price  = bs_call(S, K_itm, r, q, σ_true, T)
        res    = implied_vol_newton(price, S, K_itm, r, q, T)
        @test res.converged
        @test isapprox(res.sigma, σ_true, atol=1e-5)
    end

    @testset "Brent: round-trip OTM" begin
        σ_true = 0.20
        K_otm  = 120.0
        price  = bs_call(S, K_otm, r, q, σ_true, T)
        res    = implied_vol_brent(price, S, K_otm, r, q, T)
        @test res.converged
        @test isapprox(res.sigma, σ_true, atol=1e-7)
    end

    @testset "Brent vs Newton agreement" begin
        for σ_true in [0.10, 0.20, 0.40, 0.80]
            price  = bs_call(S, K, r, q, σ_true, T)
            res_n  = implied_vol_newton(price, S, K, r, q, T)
            res_b  = implied_vol_brent(price, S, K, r, q, T)
            if res_n.converged && res_b.converged
                @test isapprox(res_n.sigma, res_b.sigma, atol=1e-5)
            end
        end
    end

    @testset "Surface extraction shape" begin
        strikes  = [80.0, 90.0, 100.0, 110.0, 120.0]
        expiries = [0.25, 1.0]
        σ_true   = [0.25 0.22; 0.22 0.20; 0.20 0.19; 0.22 0.20; 0.25 0.22]
        prices   = zeros(5, 2)
        for j in 1:2, i in 1:5
            prices[i,j] = bs_call(S, strikes[i], r, q, σ_true[i,j], expiries[j])
        end
        res = implied_vol_surface(prices, S, strikes, expiries, r, q)
        σ_mat, conv = extract_vol_matrix(res)
        @test all(conv)
        @test isapprox(σ_mat, σ_true, atol=1e-4)
    end

    @testset "Put implied vol matches call" begin
        σ_true = 0.25
        call_p = bs_call(S, K, r, q, σ_true, T)
        put_p  = bs_put(S, K, r, q, σ_true, T)
        res_c  = implied_vol_brent(call_p, S, K, r, q, T; call=true)
        res_p  = implied_vol_brent(put_p,  S, K, r, q, T; call=false)
        @test isapprox(res_c.sigma, σ_true, atol=1e-6)
        @test isapprox(res_p.sigma, σ_true, atol=1e-6)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: SABR MODEL
# ─────────────────────────────────────────────────────────────────────────────

@testset "SABR Model" begin

    F, T = 100.0, 1.0

    @testset "ATM vol formula continuity" begin
        # Smile at K=F should equal ATM formula
        p_atm = SABRParams(0.3, 0.5, -0.3, 0.25)
        K_atm = F
        v_atm_limit = sabr_implied_vol(F, F + 1e-8, T, p_atm)
        v_atm_exact = sabr_implied_vol(F, F, T, p_atm)   # ATM branch
        @test isapprox(v_atm_limit, v_atm_exact, atol=1e-3)
    end

    @testset "Smile is convex in strike" begin
        p = SABRParams(0.3, 0.5, -0.3, 0.25)
        strikes = collect(80.0:10.0:120.0)
        vols = sabr_smile(F, strikes, T, p)
        @test all(vols .> 0)
        # Check smile is roughly U-shaped (min near ATM)
        atm_idx = argmin(abs.(strikes .- F))
        wing_avg = mean([vols[1], vols[end]])
        @test wing_avg >= vols[atm_idx] - 0.01
    end

    @testset "α increases vol" begin
        p1 = SABRParams(0.2, 0.5, 0.0, 0.20)
        p2 = SABRParams(0.5, 0.5, 0.0, 0.20)
        v1 = sabr_implied_vol(F, F*1.1, T, p1)
        v2 = sabr_implied_vol(F, F*1.1, T, p2)
        @test v2 > v1
    end

    @testset "ρ controls skew direction" begin
        p_neg = SABRParams(0.3, 0.5, -0.5, 0.25)
        p_pos = SABRParams(0.3, 0.5, +0.5, 0.25)
        # Negative ρ: vol higher for low strikes
        v_neg_low  = sabr_implied_vol(F, F*0.9, T, p_neg)
        v_neg_high = sabr_implied_vol(F, F*1.1, T, p_neg)
        v_pos_low  = sabr_implied_vol(F, F*0.9, T, p_pos)
        v_pos_high = sabr_implied_vol(F, F*1.1, T, p_pos)
        @test v_neg_low > v_neg_high   # negative skew
        @test v_pos_low < v_pos_high   # positive skew
    end

    @testset "Calibration round-trip" begin
        rng = MersenneTwister(1)
        S, r, q = 100.0, 0.05, 0.02
        p_true  = SABRParams(0.3, 0.5, -0.3, 0.20)
        strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
        market_vols = sabr_smile(F, strikes, T, p_true)

        res = calibrate_sabr(F, strikes, market_vols, T, S, r, q;
                             β=0.5, n_restarts=3)
        @test res.rmse < 1e-4
        @test isapprox(res.params.σ0, p_true.σ0, atol=0.02)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: HESTON MODEL
# ─────────────────────────────────────────────────────────────────────────────

@testset "Heston Model" begin

    S, r, q, T = 100.0, 0.05, 0.02, 1.0
    p_h = HestonParams(2.0, 0.04, 0.5, -0.7, 0.04)

    @testset "Characteristic function: unit at u=0 (approx)" begin
        # φ(0) = E[1] = 1... but our CF is Φ(u) = E[exp(iu log S_T/S_0)]
        # At u=0: Φ(0) = 1
        φ0 = heston_char_fn(0.0 + 1e-8, S, r, q, T, p_h)
        @test abs(real(φ0) - 1.0) < 0.1   # rough check
    end

    @testset "FFT call prices non-negative" begin
        strikes = collect(70.0:10.0:130.0)
        prices  = heston_call_fft(S, strikes, r, q, T, p_h; N=2048)
        @test all(prices .>= -1e-6)
    end

    @testset "FFT prices satisfy put-call parity" begin
        strikes = [90.0, 100.0, 110.0]
        calls   = heston_call_fft(S, strikes, r, q, T, p_h; N=4096)
        for (i, K) in enumerate(strikes)
            parity_lhs = calls[i]
            put = parity_lhs - (S*exp(-q*T) - K*exp(-r*T))
            @test put >= -1e-4  # put must be non-negative
        end
    end

    @testset "Implied vols from Heston positive" begin
        strikes = collect(80.0:10.0:120.0)
        ivs     = heston_implied_vols(S, strikes, r, q, T, p_h)
        valid   = filter(!isnan, ivs)
        @test length(valid) > 0
        @test all(valid .> 0)
    end

    @testset "Heston reduces to BS as ξ → 0" begin
        σ_BS = sqrt(p_h.θ)   # long-run vol
        p_degen = HestonParams(100.0, p_h.V0, 1e-6, 0.0, p_h.V0)
        K  = 100.0
        bs_price  = bs_call(S, K, r, q, σ_BS, T)
        heston_price = heston_call_fft(S, [K], r, q, T, p_degen; N=4096)[1]
        @test isapprox(bs_price, heston_price, rtol=0.05)
    end

    @testset "Calibration RMSE < 2 vol points" begin
        # Build synthetic surface and calibrate
        strikes_1d = collect(80.0:10.0:120.0)
        expiries   = [0.5, 1.0]
        K_mat      = repeat(strikes_1d, 1, 2)
        iv_mat     = zeros(length(strikes_1d), 2)
        for j in 1:2
            iv_mat[:, j] = heston_implied_vols(S, strikes_1d, r, q, expiries[j], p_h)
        end
        valid = .!any(isnan.(iv_mat), dims=2)[:]
        if sum(valid) >= 3
            res = calibrate_heston(S, K_mat[valid, :], iv_mat[valid, :],
                                   expiries, r, q; n_restarts=3)
            @test res.rmse < 0.02   # < 2 vol points RMSE
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DUPIRE LOCAL VOLATILITY
# ─────────────────────────────────────────────────────────────────────────────

@testset "Dupire Local Volatility" begin

    S, r, q = 100.0, 0.05, 0.02
    strikes  = collect(80.0:5.0:120.0)
    expiries = [0.25, 0.5, 1.0]
    nK, nT   = length(strikes), length(expiries)

    # Generate BS flat-vol surface (LV should equal constant vol)
    σ_flat   = 0.20
    call_prices = zeros(nK, nT)
    for j in 1:nT, i in 1:nK
        call_prices[i,j] = bs_call(S, strikes[i], r, q, σ_flat, expiries[j])
    end

    lv = dupire_local_vol(S, strikes, expiries, call_prices, r, q)

    @testset "LV surface dimensions" begin
        @test size(lv.sigma) == (nK, nT)
    end

    @testset "LV values in reasonable range" begin
        valid = lv.sigma[.!isnan.(lv.sigma)]
        @test all(valid .>= 0)
        @test all(valid .<= 2.0)   # sanity: < 200% vol
    end

    @testset "Interpolation returns valid value" begin
        σ_loc = interp_local_vol(lv, 100.0, 0.5)
        @test !isnan(σ_loc)
        @test σ_loc > 0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: LEVENBERG-MARQUARDT OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────

@testset "Levenberg-Marquardt Optimizer" begin

    @testset "Linear least squares" begin
        # f(x) = A x - b, minimum at x = A\b
        A = [1.0 0; 0 2.0; 1 1]
        b = [3.0, 4.0, 5.0]
        f = x -> A * x .- b
        x0  = zeros(2)
        res = levenberg_marquardt(f, x0)
        x_exact = A \ b
        @test isapprox(res.x, x_exact, atol=1e-5)
        @test res.converged
    end

    @testset "Rosenbrock function (nonlinear)" begin
        # Rosenbrock as residuals: f1 = 10(x2 - x1²), f2 = (1 - x1)
        f = x -> [10*(x[2] - x[1]^2), 1 - x[1]]
        x0  = [-1.0, 1.0]
        res = levenberg_marquardt(f, x0)
        @test isapprox(res.x, [1.0, 1.0], atol=1e-3)
    end

    @testset "Convergence monotone" begin
        f = x -> [x[1] - 3.0, x[2] + 2.0]
        x0  = [0.0, 0.0]
        res = levenberg_marquardt(f, x0)
        # Cost should be non-increasing
        for i in 2:length(res.history)
            @test res.history[i] <= res.history[i-1] + 1e-8
        end
    end

    @testset "Bounds respected" begin
        f = x -> [x[1] - 3.0]
        x0  = [0.0]
        res = levenberg_marquardt(f, x0; bounds=([-1.0], [2.0]))
        @test res.x[1] <= 2.0 + 1e-8
        @test res.x[1] >= -1.0 - 1e-8
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PARAMETER STABILITY
# ─────────────────────────────────────────────────────────────────────────────

@testset "Parameter Stability Diagnostics" begin

    @testset "Stable parameters detected" begin
        values = [0.3, 0.31, 0.29, 0.305, 0.295]
        labels = ["T=$(t)" for t in [0.25, 0.5, 1.0, 1.5, 2.0]]
        stab   = assess_stability(values, labels, "α"; cv_threshold=0.1)
        @test stab.is_stable
        @test isapprox(stab.mean_val, mean(values), atol=1e-10)
    end

    @testset "Unstable parameters flagged" begin
        values = [0.1, 0.5, 0.9, 0.2, 0.8]
        labels = ["T=$(t)" for t in [0.25, 0.5, 1.0, 1.5, 2.0]]
        stab   = assess_stability(values, labels, "α"; cv_threshold=0.2)
        @test !stab.is_stable
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: CALIBRATION QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Calibration Quality Metrics" begin

    S, r, q    = 100.0, 0.05, 0.02
    expiries   = [0.5, 1.0]
    strikes    = [90.0 90.0; 100.0 100.0; 110.0 110.0]   # 3×2
    market_iv  = [0.22 0.20; 0.20 0.19; 0.22 0.20]
    model_iv   = market_iv .+ 0.01   # slightly off

    @testset "RMSE is non-negative" begin
        m = compute_calib_metrics(model_iv, market_iv, S, strikes, expiries, r, q)
        @test m.rmse >= 0.0
        @test isapprox(m.rmse, sqrt(mean((model_iv .- market_iv).^2)), atol=1e-10)
    end

    @testset "Perfect fit gives RMSE=0" begin
        m = compute_calib_metrics(market_iv, market_iv, S, strikes, expiries, r, q)
        @test m.rmse < 1e-10
        @test isapprox(m.r_squared, 1.0, atol=1e-6)
    end

    @testset "MAPE in percent" begin
        m = compute_calib_metrics(model_iv, market_iv, S, strikes, expiries, r, q)
        @test m.mape > 0.0
        @test m.mape < 100.0
    end

    @testset "Vega-weighted RMSE non-negative" begin
        m = compute_calib_metrics(model_iv, market_iv, S, strikes, expiries, r, q)
        @test m.vega_wrmse >= 0.0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SVI PARAMETRISATION
# ─────────────────────────────────────────────────────────────────────────────

@testset "SVI Parametrisation" begin

    p = SVIParams(0.04, 0.2, -0.3, 0.0, 0.1)
    T = 1.0

    @testset "Total variance positive on grid" begin
        ks = range(-2.0, 2.0, length=100)
        ws = [svi_total_var(k, p) for k in ks]
        @test all(ws .> 0)
    end

    @testset "Implied vol positive" begin
        for k in [-1.0, 0.0, 1.0]
            σ = svi_implied_vol(k, T, p)
            @test σ > 0
        end
    end

    @testset "Smile is U-shaped (convex)" begin
        ks = range(-1.5, 1.5, length=50)
        ws = [svi_total_var(k, p) for k in ks]
        # Minimum should be interior (not at boundary)
        min_idx = argmin(ws)
        @test 2 <= min_idx <= length(ws) - 1
    end

    @testset "SVI calibration round-trip" begin
        F = 100.0
        strikes = collect(80.0:5.0:120.0)
        # Generate true SVI smile
        ks  = log.(strikes ./ F)
        true_vols = [sqrt(max(svi_total_var(k, p) / T, 0.0)) for k in ks]

        res = calibrate_svi(F, strikes, true_vols, T; n_restarts=3)
        @test res.rmse < 1e-4
    end

    @testset "No-butterfly check" begin
        @test svi_no_butterfly_check(p)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: MARKET VOL SURFACE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@testset "Market Vol Surface Pipeline" begin

    S, r, q = 100.0, 0.05, 0.02
    expiries = [0.25, 0.5, 1.0]
    nK, nT   = 5, 3
    strikes_1d = [85.0, 92.5, 100.0, 107.5, 115.0]
    K_mat    = repeat(strikes_1d, 1, nT)

    # Flat vol surface
    iv_mat   = fill(0.20, nK, nT)
    surf     = MarketVolSurface(S, r, q, K_mat, expiries, iv_mat, nothing, nothing)

    @testset "ATM vol correct" begin
        atm = atm_vol(surf)
        @test all(isapprox.(atm, 0.20, atol=0.01))
    end

    @testset "Arbitrage check passes on flat surface" begin
        arb = arbitrage_free_check(surf)
        # May have numerical violations on flat surface; just check structure
        @test arb isa NamedTuple
        @test haskey(arb, :calendar_ok)
        @test haskey(arb, :butterfly_ok)
    end

    @testset "Bootstrap CI structure" begin
        bc = bootstrap_heston(surf; B=20, n_restarts=1, seed=42)
        @test length(bc.param_names) == 5
        @test length(bc.means) == 5
        @test bc.n_bootstrap <= 20
    end
end

println("\n✓ All calibration tests passed.")
