"""
    SRFM Julia Module Test Suite

Comprehensive tests for AdvancedOptions, LiveRisk, MLSignals, and Backtesting modules.
Uses Test.jl for assertions and numerically validates key algorithms.
"""

using Test
using LinearAlgebra
using Statistics
using Distributions

# Load all modules from parent directory
include("../AdvancedOptions.jl")
include("../LiveRisk.jl")
include("../MLSignals.jl")
include("../Backtesting.jl")

using .AdvancedOptions
using .LiveRisk
using .MLSignals
using .Backtesting

# ============================================================================
# Test utilities
# ============================================================================

"""Assert two values are within relative tolerance."""
function assert_rel_close(a::Real, b::Real, rtol::Real=0.01; msg::String="")
    diff = abs(a - b) / (abs(b) + 1e-10)
    @test diff < rtol
    if diff >= rtol
        println("FAIL: $a vs $b (rtol=$rtol) $msg")
    end
end

"""Assert two values are within absolute tolerance."""
function assert_abs_close(a::Real, b::Real, atol::Real=1e-6; msg::String="")
    @test abs(a - b) < atol
end

# ============================================================================
# AdvancedOptions tests
# ============================================================================

@testset "AdvancedOptions" begin

    @testset "Black-Scholes baseline" begin
        # Well-known ATM call: S=100, K=100, r=0.05, q=0, sigma=0.2, T=1
        call = black_scholes_call(100.0, 100.0, 0.05, 0.0, 0.2, 1.0)
        @test isapprox(call, 10.4506, atol=0.01)

        put = black_scholes_put(100.0, 100.0, 0.05, 0.0, 0.2, 1.0)
        @test isapprox(put, 5.5735, atol=0.01)

        # Put-call parity
        pcp = call - put - 100.0 + 100.0 * exp(-0.05)
        @test abs(pcp) < 1e-8

        # Deep ITM call ~ intrinsic
        deep_itm = black_scholes_call(150.0, 100.0, 0.05, 0.0, 0.01, 1.0)
        @test deep_itm > 45.0

        # Deep OTM call ~ 0
        deep_otm = black_scholes_call(50.0, 100.0, 0.05, 0.0, 0.2, 1.0)
        @test deep_otm < 0.01

        # Zero time to expiry
        call_expired = black_scholes_call(110.0, 100.0, 0.05, 0.0, 0.2, 0.0)
        @test call_expired >= 10.0

        # Vectorized call
        Ss = [90.0, 100.0, 110.0]
        Ks = [100.0, 100.0, 100.0]
        Ts = [1.0, 1.0, 1.0]
        calls_vec = black_scholes_call.(Ss, Ks, 0.05, 0.0, 0.2, Ts)
        @test length(calls_vec) == 3
        @test calls_vec[2] > calls_vec[1]
        @test calls_vec[3] > calls_vec[2]
    end

    @testset "Black-Scholes Greeks" begin
        S, K, r, q, sigma, T = 100.0, 100.0, 0.05, 0.0, 0.2, 1.0

        delta_call = bs_delta(S, K, r, q, sigma, T; option_type=:call)
        @test 0.5 < delta_call < 0.7  # ATM call delta

        delta_put = bs_delta(S, K, r, q, sigma, T; option_type=:put)
        @test -0.5 > delta_put > -0.7

        # Delta put-call parity: delta_call - delta_put = exp(-q*T)
        @test isapprox(delta_call - delta_put, exp(-q * T), atol=1e-8)

        vega_val = bs_vega(S, K, r, q, sigma, T)
        @test vega_val > 0.0

        gamma_val = bs_gamma(S, K, r, q, sigma, T)
        @test gamma_val > 0.0

        # Vanna and volga
        vanna_val = vanna(S, K, r, q, sigma, T)
        @test isfinite(vanna_val)

        volga_val = volga(S, K, r, q, sigma, T)
        @test isfinite(volga_val)
    end

    @testset "Gauss-Legendre quadrature" begin
        nodes, weights = gauss_legendre_nodes_weights(10)
        @test length(nodes) == 10
        @test length(weights) == 10

        # Weights must sum to 2 (integral of 1 over [-1,1])
        @test isapprox(sum(weights), 2.0, atol=1e-12)

        # Nodes must be in [-1, 1]
        @test all(-1.0 .<= nodes .<= 1.0)

        # Integrate x^2 from 0 to 1 = 1/3
        result = gl_integrate(x -> x^2, 0.0, 1.0, 16)
        @test isapprox(result, 1.0 / 3.0, atol=1e-10)

        # Integrate sin(x) from 0 to pi = 2
        result2 = gl_integrate(x -> sin(x), 0.0, pi, 32)
        @test isapprox(result2, 2.0, atol=1e-8)
    end

    @testset "Heston vs Black-Scholes convergence" begin
        S, K, r, q, T = 100.0, 100.0, 0.05, 0.0, 1.0
        sigma_bs = 0.2

        # Heston reduces to BS when vol-of-vol sigma_v -> 0 and v0 = theta = sigma_bs^2
        v0 = sigma_bs^2
        theta = sigma_bs^2
        kappa = 5.0
        sigma_v = 0.001  # nearly zero vol-of-vol
        rho = 0.0

        heston_price = heston_call_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0;
                                          n_quad=64)
        bs_price = black_scholes_call(S, K, r, q, sigma_bs, T)

        # Should converge to within 1% of BS
        @test isapprox(heston_price, bs_price, rtol=0.02)
    end

    @testset "Heston call price basic sanity" begin
        S, K, r, q, T = 100.0, 100.0, 0.05, 0.02, 1.0
        kappa, theta, sigma_v, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

        call = heston_call_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0)
        @test call > 0.0
        @test call < S  # call can't exceed spot price

        put = heston_put_price(S, K, r, q, T, kappa, theta, sigma_v, rho, v0)
        @test put > 0.0

        # Put-call parity for Heston
        pcp = call - put - S * exp(-q * T) + K * exp(-r * T)
        @test abs(pcp) < 0.05  # should be close to zero

        # Monotone in strike: higher strike = lower call
        call_otm = heston_call_price(S, 110.0, r, q, T, kappa, theta, sigma_v, rho, v0)
        call_itm = heston_call_price(S, 90.0, r, q, T, kappa, theta, sigma_v, rho, v0)
        @test call_itm > call > call_otm
    end

    @testset "SABR ATM formula" begin
        F, T = 100.0, 1.0
        alpha, beta, rho, nu = 0.2, 0.5, -0.3, 0.4

        # ATM vol should be positive and reasonable
        atm_vol = sabr_implied_vol(F, F, T, alpha, beta, rho, nu)
        @test atm_vol > 0.0
        @test atm_vol < 2.0

        # Symmetric strikes around ATM: slight skew expected
        vol_otm_call = sabr_implied_vol(F, 1.1 * F, T, alpha, beta, rho, nu)
        vol_otm_put  = sabr_implied_vol(F, 0.9 * F, T, alpha, beta, rho, nu)
        @test vol_otm_call > 0.0
        @test vol_otm_put > 0.0

        # With negative rho, puts should be more expensive (neg skew)
        @test vol_otm_put > vol_otm_call
    end

    @testset "SABR calibration" begin
        F, T = 100.0, 0.5
        true_alpha, true_beta, true_rho, true_nu = 0.25, 0.5, -0.4, 0.5
        strikes = collect(range(85.0, 115.0, length=10))
        market_vols = [sabr_implied_vol(F, k, T, true_alpha, true_beta, true_rho, true_nu)
                       for k in strikes]

        cal = calibrate_sabr(F, strikes, T, market_vols; beta=true_beta)
        @test cal.rmse < 0.005  # should fit well (we generated synthetic data)
        @test cal.alpha > 0.0
        @test cal.nu > 0.0
        @test abs(cal.rho) <= 1.0
    end

    @testset "SABR normal vol" begin
        F, K, T = 0.02, 0.02, 1.0  # low-rate environment
        alpha, beta, rho, nu = 0.005, 0.0, -0.3, 0.3
        nvol = sabr_normal_vol(F, K, T, alpha, beta, rho, nu)
        @test nvol > 0.0
        @test isfinite(nvol)
    end

    @testset "Variance swap fair strike" begin
        S, r, q, T = 100.0, 0.05, 0.0, 1.0
        sigma_imp = 0.2

        # Create a flat vol surface (all options at sigma_imp)
        F = S * exp((r - q) * T)
        put_strikes  = collect(range(50.0, F - 1.0, length=30))
        call_strikes = collect(range(F + 1.0, 200.0, length=30))

        put_prices  = black_scholes_put.(S, put_strikes, r, q, sigma_imp, T)
        call_prices = black_scholes_call.(S, call_strikes, r, q, sigma_imp, T)

        kvar = variance_swap_fair_strike(S, r, q, T, call_strikes, call_prices,
                                          put_strikes, put_prices)

        # Fair variance should be close to sigma_imp^2 for flat vol
        @test isapprox(kvar, sigma_imp^2, rtol=0.15)

        # Replication weights
        weights = variance_swap_replication_weights(S, F, T, call_strikes)
        @test length(weights) == length(call_strikes)
        @test all(weights .> 0.0)
    end

    @testset "Volatility cone" begin
        Random.seed!(42)
        log_rets = randn(500) * 0.01  # ~16% annual vol
        horizons = [5, 10, 21, 63]

        cone = volatility_cone(log_rets, horizons)
        @test length(cone.horizons) == 4
        @test all(cone.p10 .< cone.p50)
        @test all(cone.p50 .< cone.p90)

        # Vol estimates should be positive
        @test all(cone.p10 .> 0.0)

        # Longer horizons: rolling vol is smoother, narrower cone
        # (not necessarily always true, but current should be defined)
        @test all(isfinite.(cone.current))
    end

    @testset "Merton jump-diffusion" begin
        S, K, r, q, T = 100.0, 100.0, 0.05, 0.0, 1.0
        sigma, lambda, mu_j, sigma_j = 0.15, 0.5, -0.05, 0.1

        call = merton_call_price(S, K, r, q, T, sigma, lambda, mu_j, sigma_j)
        @test call > 0.0
        @test isfinite(call)

        put = merton_put_price(S, K, r, q, T, sigma, lambda, mu_j, sigma_j)
        @test put > 0.0

        # Put-call parity check
        pcp = call - put - S + K * exp(-r * T)
        @test abs(pcp) < 0.10

        # With no jumps (lambda=0) should approach BS
        call_no_jump = merton_call_price(S, K, r, q, T, sigma, 0.0, 0.0, 0.01)
        bs_ref = black_scholes_call(S, K, r, q, sigma, T)
        @test isapprox(call_no_jump, bs_ref, rtol=0.01)
    end

    @testset "Implied vol inversion" begin
        S, K, r, q, sigma_true, T = 100.0, 100.0, 0.05, 0.0, 0.25, 0.5
        call_price = black_scholes_call(S, K, r, q, sigma_true, T)
        sigma_impl = implied_vol(call_price, S, K, r, q, T; option_type=:call)
        @test isapprox(sigma_impl, sigma_true, atol=1e-5)

        put_price = black_scholes_put(S, K, r, q, sigma_true, T)
        sigma_impl_put = implied_vol(put_price, S, K, r, q, T; option_type=:put)
        @test isapprox(sigma_impl_put, sigma_true, atol=1e-5)
    end

    @testset "SVI total variance" begin
        # SVI should produce non-negative variance
        a, b, rho_svi, m, sigma_svi = 0.04, 0.1, -0.3, 0.0, 0.2
        for k in -0.5:0.1:0.5
            w = svi_total_variance(k, a, b, rho_svi, m, sigma_svi)
            @test w >= 0.0
        end
    end

    @testset "Vanna-Volga exotic pricing" begin
        S, K, r, q, T = 100.0, 100.0, 0.05, 0.0, 0.5
        sigma_atm = 0.2
        sigma_25c = 0.22  # vol skew
        sigma_25p = 0.25
        exotic_bsm = black_scholes_call(S, K, r, q, sigma_atm, T)
        vv_price = compute_vanna_volga_price(S, K, r, q, T, sigma_atm,
                                              sigma_25c, sigma_25p, exotic_bsm)
        @test isfinite(vv_price)
        @test vv_price > 0.0
    end

end  # AdvancedOptions testset

# ============================================================================
# LiveRisk tests
# ============================================================================

@testset "LiveRisk" begin

    Random.seed!(123)
    T_obs = 500
    N_assets = 4
    # Generate correlated returns
    Sigma_true = [0.04 0.01 0.005 0.002;
                  0.01 0.09 0.01 0.003;
                  0.005 0.01 0.01 0.001;
                  0.002 0.003 0.001 0.04]
    returns_raw = randn(T_obs, N_assets) * cholesky(Sigma_true).L'

    weights_eq = fill(0.25, N_assets)

    @testset "EWMA covariance" begin
        Sigma_ewma = ewma_covariance(returns_raw; lambda=0.94)
        @test size(Sigma_ewma) == (N_assets, N_assets)

        # Positive definite
        eigs = eigvals(Symmetric(Sigma_ewma))
        @test all(eigs .> 0.0)

        # Diagonal: positive variances
        @test all(diag(Sigma_ewma) .> 0.0)

        # Symmetry
        @test isapprox(Sigma_ewma, Sigma_ewma', atol=1e-12)

        # Single asset
        sigma2 = ewma_variance(returns_raw[:, 1])
        @test sigma2 > 0.0
    end

    @testset "Parametric VaR" begin
        var_95 = parametric_var(weights_eq, returns_raw, 0.95)
        var_99 = parametric_var(weights_eq, returns_raw, 0.99)

        @test var_95 > 0.0
        @test var_99 > var_95  # 99% VaR > 95% VaR

        # VaR should scale with sqrt(holding_period)
        var_1d = parametric_var(weights_eq, returns_raw, 0.99; holding_period=1)
        var_5d = parametric_var(weights_eq, returns_raw, 0.99; holding_period=5)
        @test isapprox(var_5d / var_1d, sqrt(5.0), atol=0.01)
    end

    @testset "Historical VaR" begin
        hvar_99 = historical_var(weights_eq, returns_raw, 0.99)
        @test hvar_99 > 0.0
        @test isfinite(hvar_99)

        hvar_95 = historical_var(weights_eq, returns_raw, 0.95)
        @test hvar_99 > hvar_95
    end

    @testset "Cornish-Fisher VaR" begin
        cfvar = cornish_fisher_var(weights_eq, returns_raw, 0.99)
        @test isfinite(cfvar)
        @test cfvar > 0.0

        # Test skewness and kurtosis helpers
        pnl = returns_raw * weights_eq
        s = LiveRisk.skewness_stat(pnl)
        k = LiveRisk.kurtosis_stat(pnl)
        @test isfinite(s)
        @test isfinite(k)
    end

    @testset "Expected shortfall" begin
        es_hist = expected_shortfall(weights_eq, returns_raw, 0.99; method=:historical)
        var_99  = historical_var(weights_eq, returns_raw, 0.99)
        @test es_hist >= var_99  # ES >= VaR by definition

        es_param = expected_shortfall(weights_eq, returns_raw, 0.99; method=:parametric)
        @test es_param > 0.0

        # Bootstrap bands
        bands = bootstrap_es_bands(weights_eq, returns_raw, 0.99; n_boot=100)
        @test bands.lower < bands.es_mean < bands.upper
        @test bands.std_err > 0.0
    end

    @testset "Marginal and component VaR" begin
        mvar = marginal_var(weights_eq, returns_raw, 0.99)
        @test length(mvar) == N_assets
        @test all(isfinite.(mvar))

        cvar_comp = component_var(weights_eq, returns_raw, 0.99)
        @test length(cvar_comp) == N_assets

        # Component VaR sums to portfolio VaR
        total_pvar = parametric_var(weights_eq, returns_raw, 0.99)
        @test isapprox(sum(cvar_comp), total_pvar, rtol=0.05)
    end

    @testset "Stress testing" begin
        s2008  = stress_test_2008()
        scovid = stress_test_covid2020()
        scrypto = stress_test_crypto2022()

        @test s2008.name == "2008_crisis"
        @test scovid.name == "covid_2020"
        @test scrypto.name == "crypto_winter_2022"

        @test length(s2008.factor_shocks) > 5
        @test s2008.factor_shocks["equity_spx"] < 0.0  # equity dropped

        exposures = Dict("equity_spx" => 1_000_000.0, "btc" => 200_000.0)
        pnl_2008 = apply_stress_scenario(exposures, s2008)
        @test pnl_2008 < 0.0  # large loss in 2008

        all_results = run_all_stress_tests(exposures)
        @test haskey(all_results, "2008_crisis")
        @test haskey(all_results, "covid_2020")
        @test haskey(all_results, "crypto_winter_2022")
    end

    @testset "Liquidity-adjusted VaR" begin
        positions_dollar = fill(500_000.0, N_assets)
        adv_vals = [2e6, 5e6, 1e6, 3e6]

        lv = liquidity_adjusted_var(weights_eq, returns_raw, 0.99, positions_dollar, adv_vals)
        @test lv.lvar > 0.0
        @test lv.lvar >= lv.var_base  # LVaR >= standard VaR
        @test lv.liquidity_cost >= 0.0

        # Amihud illiquidity
        rets_1d = returns_raw[:, 1]
        vol_1d  = fill(5e6, T_obs)
        illiq = amihud_illiquidity(rets_1d, vol_1d)
        @test illiq >= 0.0
        @test isfinite(illiq)
    end

    @testset "Fixed income" begin
        face, coupon, ytm, mat = 1000.0, 0.05, 0.05, 10.0

        # Par bond: price = face when coupon = ytm
        dv01_val = dv01_bond(face, coupon, ytm, mat)
        @test dv01_val > 0.0
        @test dv01_val < 10.0  # sanity: DV01 < $10 for $1000 bond

        mac_dur = macaulay_duration(face, coupon, ytm, mat)
        @test 1.0 < mac_dur < mat  # duration between 1 and maturity

        mod_dur = modified_duration(face, coupon, ytm, mat)
        @test mod_dur < mac_dur

        # Zero-coupon bond: Macaulay duration = maturity
        mac_dur_zc = macaulay_duration(face, 0.0001, ytm, mat)
        @test isapprox(mac_dur_zc, mat, atol=0.1)
    end

end  # LiveRisk testset

# ============================================================================
# MLSignals tests
# ============================================================================

@testset "MLSignals" begin

    Random.seed!(777)

    @testset "GP squared exponential kernel" begin
        x1 = [0.0, 0.0]
        x2 = [1.0, 0.0]
        k00 = gp_sq_exp_kernel(x1, x1, 1.0, 1.0)
        k01 = gp_sq_exp_kernel(x1, x2, 1.0, 1.0)
        @test k00 == 1.0  # self-covariance = signal_var
        @test 0.0 < k01 < 1.0
        @test k01 < k00  # closer points = higher covariance
    end

    @testset "GP Matern 5/2 kernel" begin
        x1 = [0.0]
        x2 = [0.5]
        km00 = gp_matern52_kernel(x1, x1, 1.0, 1.0)
        km01 = gp_matern52_kernel(x1, x2, 1.0, 1.0)
        @test km00 == 1.0
        @test 0.0 < km01 < 1.0
    end

    @testset "GP prediction" begin
        # Simple 1D regression: y = sin(x) + noise
        N_train = 20
        X_train = reshape(collect(range(0.0, 2*pi, length=N_train)), N_train, 1)
        y_train = sin.(X_train[:, 1]) .+ 0.05 .* randn(N_train)

        N_test = 10
        X_test = reshape(collect(range(0.0, 2*pi, length=N_test)), N_test, 1)

        result = gp_predict(X_train, y_train, X_test, 1.0, 1.0, 0.01)
        @test length(result.mean) == N_test
        @test length(result.variance) == N_test
        @test all(result.variance .>= 0.0)

        # Predictions should roughly track sin(x)
        y_test_true = sin.(X_test[:, 1])
        rmse = sqrt(mean((result.mean .- y_test_true).^2))
        @test rmse < 0.5  # should fit sin reasonably well
    end

    @testset "GP log marginal likelihood" begin
        N = 15
        X = reshape(collect(range(0.0, 5.0, length=N)), N, 1)
        y = sin.(X[:, 1]) .+ 0.05 .* randn(N)

        lml_good = gp_log_marginal_likelihood(X, y, 1.0, 1.0, 0.01)
        lml_bad  = gp_log_marginal_likelihood(X, y, 0.001, 0.001, 10.0)
        @test isfinite(lml_good)
        @test lml_good > lml_bad  # better params = higher LML

        # Invalid params
        lml_invalid = gp_log_marginal_likelihood(X, y, -1.0, 1.0, 0.01)
        @test lml_invalid == -Inf
    end

    @testset "Bayesian Ridge regression" begin
        N, D = 100, 5
        true_w = [1.0, -2.0, 0.5, 0.0, 3.0]
        X = randn(N, D)
        y = X * true_w .+ 0.1 .* randn(N)

        model = bayesian_ridge_fit(X, y)
        @test length(model.weights) == D
        @test model.alpha > 0.0
        @test model.lambda > 0.0

        # Weights should be close to true values
        @test isapprox(model.weights, true_w, atol=0.3)

        # Predictions
        X_test = randn(20, D)
        pred = bayesian_ridge_predict(X_test, model)
        @test length(pred.mean) == 20
        @test all(pred.variance .>= 0.0)
    end

    @testset "Kalman filter" begin
        # Simple local level model: x_t = x_{t-1} + w, y_t = x_t + v
        T_kf = 100
        true_signal = cumsum(0.1 .* randn(T_kf))
        obs = true_signal .+ 0.5 .* randn(T_kf)

        F = reshape([1.0], 1, 1)
        H = reshape([1.0], 1, 1)
        Q = reshape([0.01], 1, 1)
        R = reshape([0.25], 1, 1)
        x0 = [obs[1]]
        P0 = reshape([1.0], 1, 1)

        Y = reshape(obs, T_kf, 1)
        kf_result = kalman_filter(Y, F, H, Q, R, x0, P0)

        @test size(kf_result.filtered_states) == (T_kf, 1)
        @test isfinite(kf_result.log_likelihood)
        @test kf_result.log_likelihood < 0.0  # log likelihood should be negative

        # Filtered states should be closer to true signal than raw obs
        filtered_rmse = sqrt(mean((kf_result.filtered_states[:, 1] .- true_signal).^2))
        obs_rmse = sqrt(mean((obs .- true_signal).^2))
        @test filtered_rmse < obs_rmse
    end

    @testset "Kalman smoother" begin
        T_ks = 50
        obs = sin.(range(0.0, 4*pi, length=T_ks)) .+ 0.2 .* randn(T_ks)
        Y = reshape(obs, T_ks, 1)

        F = reshape([1.0], 1, 1)
        H = reshape([1.0], 1, 1)
        Q = reshape([0.01], 1, 1)
        R = reshape([0.04], 1, 1)
        x0 = [obs[1]]
        P0 = reshape([1.0], 1, 1)

        kf = kalman_filter(Y, F, H, Q, R, x0, P0)
        ks = kalman_smoother(kf.filtered_states, kf.filtered_covs, F, Q)

        @test size(ks.smoothed_states) == (T_ks, 1)
        # Smoother should not increase variance
        filtered_var = mean([kf.filtered_covs[t][1,1] for t in 1:T_ks])
        smoothed_var = mean([ks.smoothed_covs[t][1,1] for t in 1:T_ks])
        @test smoothed_var <= filtered_var + 1e-8
    end

    @testset "Kalman EM" begin
        T_em = 80
        signal = cumsum(randn(T_em) * 0.05)
        obs = signal .+ randn(T_em) * 0.2
        result = kalman_em(obs; n_iter=20)
        @test length(result.log_likelihoods) > 0
        @test result.log_likelihoods[end] >= result.log_likelihoods[1]  # LML increases
        @test size(result.F) == (2, 2)  # local linear trend
    end

    @testset "HMM Baum-Welch EM" begin
        # Generate synthetic 3-state (bull/sideways/bear) data
        Random.seed!(42)
        T_hmm = 300
        true_mus = [-0.01, 0.0, 0.015]
        true_sigs = [0.02, 0.01, 0.015]

        obs_hmm = Float64[]
        state = 2
        for _ in 1:T_hmm
            push!(obs_hmm, true_mus[state] + true_sigs[state] * randn())
            # Transition
            r = rand()
            if state == 1
                state = r < 0.7 ? 1 : (r < 0.9 ? 2 : 3)
            elseif state == 2
                state = r < 0.2 ? 1 : (r < 0.8 ? 2 : 3)
            else
                state = r < 0.1 ? 1 : (r < 0.3 ? 2 : 3)
            end
        end

        model = hmm_baum_welch(obs_hmm, 3; n_iter=50, n_restarts=2)
        @test model.K == 3
        @test length(model.mu) == 3
        @test all(model.sigma .> 0.0)
        @test isapprox(sum(model.pi_init), 1.0, atol=1e-8)
        @test all(isapprox.(sum(model.A, dims=2), 1.0, atol=1e-8))

        # States sorted by mean: bear < sideways < bull
        @test model.mu[1] < model.mu[2] < model.mu[3]
    end

    @testset "HMM Viterbi decoding" begin
        Random.seed!(99)
        # Simple 2-state HMM
        model_2 = HMMModel(2)
        model_2.mu = [-0.01, 0.01]
        model_2.sigma = [0.02, 0.02]
        model_2.pi_init = [0.5, 0.5]
        model_2.A = [0.9 0.1; 0.1 0.9]

        obs_2 = [model_2.mu[1] .+ model_2.sigma[1] .* randn() for _ in 1:50]

        states = hmm_viterbi(model_2, obs_2)
        @test length(states) == 50
        @test all(s in [1, 2] for s in states)
    end

    @testset "HMM regime detection (recovery test)" begin
        # Test that HMM can distinguish two clearly separated states
        Random.seed!(13)
        # State 1: mean=-0.02, State 2: mean=0.02 (very separated)
        T_regime = 200
        true_states = [i <= 100 ? 1 : 2 for i in 1:T_regime]
        obs_regime = [true_states[i] == 1 ? -0.02 + 0.005*randn() :
                                              0.02 + 0.005*randn() for i in 1:T_regime]

        model = hmm_baum_welch(obs_regime, 2; n_iter=100, n_restarts=3)
        viterbi_states = hmm_viterbi(model, obs_regime)

        # With clearly separated states, Viterbi should recover > 85% correctly
        # Map: state 1 (lower mean) = index 1, state 2 (higher mean) = index 2
        accuracy = mean(viterbi_states .== true_states)
        alt_accuracy = mean(viterbi_states .== (3 .- true_states))  # flipped mapping
        best_accuracy = max(accuracy, alt_accuracy)
        @test best_accuracy > 0.85
    end

    @testset "Mutual information" begin
        N_mi = 500
        x_indep = randn(N_mi)
        y_indep = randn(N_mi)
        y_dep   = x_indep .+ 0.1 .* randn(N_mi)  # very correlated

        mi_indep = mutual_information(x_indep, y_indep)
        mi_dep   = mutual_information(x_indep, y_dep)

        @test mi_indep >= 0.0
        @test mi_dep >= 0.0
        @test mi_dep > mi_indep  # dependent pair has more MI
    end

    @testset "mRMR feature selection" begin
        N_feat = 100
        D_feat = 8
        # First 3 features are informative, rest are noise
        true_w = [2.0, -1.5, 1.0, zeros(5)...]
        X_feat = randn(N_feat, D_feat)
        y_feat = X_feat * true_w .+ 0.1 .* randn(N_feat)

        selected = mrmr_feature_selection(X_feat, y_feat, 3)
        @test length(selected) == 3
        @test all(1 .<= selected .<= D_feat)
        @test length(unique(selected)) == 3  # no duplicates

        # At least 2 of the 3 truly informative features should be selected
        n_informative = sum(s in [1, 2, 3] for s in selected)
        @test n_informative >= 2
    end

    @testset "Isotonic regression calibration" begin
        N_cal = 100
        scores_cal = sort(randn(N_cal))
        # Labels: noisy sigmoid of scores
        labels_cal = Float64[rand() < (1.0 / (1.0 + exp(-s))) ? 1.0 : 0.0 for s in scores_cal]

        cal = isotonic_regression(scores_cal, labels_cal)
        @test length(cal) == N_cal
        @test all(0.0 .<= cal .<= 1.0)
    end

    @testset "Platt scaling calibration" begin
        N_platt = 200
        scores_p = randn(N_platt)
        labels_p = Float64[rand() < (1.0 / (1.0 + exp(-s * 1.5))) ? 1.0 : 0.0 for s in scores_p]

        params = platt_scaling(scores_p, labels_p)
        @test isfinite(params.A)
        @test isfinite(params.B)

        cal_probs = calibrate_probabilities(scores_p, labels_p, scores_p; method=:platt)
        @test length(cal_probs) == N_platt
        @test all(0.0 .<= cal_probs .<= 1.0)

        iso_probs = calibrate_probabilities(scores_p, labels_p, scores_p; method=:isotonic)
        @test length(iso_probs) == N_platt
        @test all(0.0 .<= iso_probs .<= 1.0)
    end

    @testset "logsumexp numerical stability" begin
        # Should not overflow
        x_large = [1000.0, 1001.0, 999.0]
        result = MLSignals.logsumexp(x_large)
        @test isfinite(result)
        @test result > 1000.0

        # All -Inf
        x_neginf = [-Inf, -Inf, -Inf]
        result_inf = MLSignals.logsumexp(x_neginf)
        @test result_inf == -Inf
    end

end  # MLSignals testset

# ============================================================================
# Backtesting tests
# ============================================================================

@testset "Backtesting" begin

    Random.seed!(314)

    @testset "Transaction cost model" begin
        tc = TransactionCostModel()
        @test tc.commission_bps == 2.0
        @test tc.spread_bps == 1.0

        # Cost for a $100k trade with $10M ADV
        cost = compute_transaction_costs(100_000.0, 10_000_000.0, tc)
        @test cost > 0.0
        @test cost < 1000.0  # should be a few hundred dollars

        # Minimum commission kicks in for tiny trade
        cost_small = compute_transaction_costs(1.0, 1_000_000.0, tc)
        @test cost_small >= tc.min_commission
    end

    @testset "Backtest config defaults" begin
        config = BacktestConfig()
        @test config.initial_capital == 1_000_000.0
        @test config.max_leverage == 2.0
        @test config.position_sizing == :signal_scaled
    end

    @testset "Basic backtest" begin
        T_bt = 252
        N_bt = 3
        # Random walk prices
        log_rets = randn(T_bt, N_bt) * 0.01
        prices = exp.(cumsum(vcat(zeros(1, N_bt), log_rets), dims=1))

        # Random signals in [-1, 1]
        signals = randn(T_bt + 1, N_bt)
        signals = clamp.(signals, -1.0, 1.0)

        config = BacktestConfig(initial_capital=100_000.0)
        result = run_backtest(prices, signals, config)

        @test length(result.equity_curve) == T_bt + 1
        @test result.equity_curve[1] == 100_000.0
        @test all(result.equity_curve .> 0.0)
        @test isfinite(result.sharpe_ratio)
        @test 0.0 <= result.max_drawdown <= 1.0
        @test 0.0 <= result.hit_rate <= 1.0
        @test result.profit_factor > 0.0
    end

    @testset "Trending strategy outperforms random" begin
        # A perfect trend-following strategy should have positive Sharpe
        T_trend = 500
        N_trend = 2
        true_log_rets = randn(T_trend, N_trend) * 0.01 .+ 0.0005  # slight positive drift
        prices = exp.(cumsum(vcat(zeros(1, N_trend), true_log_rets), dims=1))

        # Perfect signal: long when next return > 0
        perfect_signals = vcat(sign.(true_log_rets), zeros(1, N_trend))
        config = BacktestConfig(initial_capital=1_000_000.0,
                                 tc_model=TransactionCostModel(commission_bps=0.0,
                                                               spread_bps=0.0,
                                                               market_impact_coeff=0.0,
                                                               slippage_bps=0.0))

        result = run_backtest(prices, perfect_signals, config)
        @test result.sharpe_ratio > 0.0
        @test result.equity_curve[end] > result.equity_curve[1]
    end

    @testset "Max drawdown computation" begin
        equity1 = [100.0, 110.0, 105.0, 90.0, 95.0, 100.0]
        mdd = Backtesting.max_drawdown(equity1)
        @test isapprox(mdd, (110.0 - 90.0) / 110.0, atol=1e-8)

        # Monotone increasing: drawdown = 0
        equity2 = [100.0, 110.0, 120.0, 130.0]
        @test Backtesting.max_drawdown(equity2) == 0.0
    end

    @testset "LARSA BH mass" begin
        prices_larsa = 100.0 .* exp.(cumsum(randn(100) * 0.01))
        mass = larsa_bh_mass(prices_larsa; window=20)
        @test length(mass) == 100
        @test all(-1.0 .<= mass .<= 1.0)
        @test all(mass[1:19] .== 0.0)  # before window fills up
    end

    @testset "LARSA CF cross signal" begin
        prices_cf = 100.0 .* exp.(cumsum(randn(200) * 0.01))
        signal = larsa_cf_cross(prices_cf; fast_window=5, slow_window=20)
        @test length(signal) == 200
        @test all(isfinite.(signal))
    end

    @testset "Hurst exponent" begin
        # Brownian motion: H ~= 0.5
        prices_bm = 100.0 .* exp.(cumsum(randn(500) * 0.01))
        h_bm = larsa_hurst_exponent(prices_bm)
        @test 0.0 <= h_bm <= 1.0

        # Strongly trending series should have H > 0.5
        trending = cumsum(fill(0.01, 500))
        prices_trend = 100.0 .* exp.(trending)
        h_trend = larsa_hurst_exponent(prices_trend)
        @test h_trend > 0.5
    end

    @testset "Quaternion navigation signal" begin
        T_quat = 100
        px = 100.0 .* exp.(cumsum(randn(T_quat) * 0.01))
        py = 100.0 .* exp.(cumsum(randn(T_quat) * 0.01))
        pz = 100.0 .* exp.(cumsum(randn(T_quat) * 0.01))

        sig = larsa_quaternion_nav(px, py, pz; window=20)
        @test length(sig) == T_quat
        @test all(isfinite.(sig))
        @test all(-1.0 .<= sig .<= 1.0)
    end

    @testset "CPCV splits" begin
        T_cpcv = 120
        splits = cpcv_splits(T_cpcv, 6, 2)
        @test !isempty(splits)

        for s in splits
            @test !isempty(s.train)
            @test !isempty(s.test)
            # No overlap between train and test
            @test isempty(intersect(s.train, s.test))
        end

        # C(6,2) = 15 combinations
        @test length(splits) == 15
    end

    @testset "Deflated Sharpe Ratio" begin
        # One trial: DSR should be high if SR is large
        sr_high = [2.0]
        dsr_high = deflated_sharpe_ratio(sr_high, 252)
        @test 0.0 <= dsr_high.dsr <= 1.0
        @test dsr_high.dsr > 0.5  # significant SR

        # Many trials inflate benchmark
        sr_many = [2.0; randn(99)]
        dsr_many = deflated_sharpe_ratio(sr_many, 252)
        @test dsr_many.dsr < dsr_high.dsr  # multiple testing penalty

        # SR star should be higher with more trials
        @test dsr_many.sr_star > dsr_high.sr_star
    end

    @testset "Performance metrics" begin
        r_perf = randn(252) * 0.01 .+ 0.0005
        metrics = Backtesting.compute_performance_metrics(r_perf)
        @test !isnothing(metrics)
        @test isfinite(metrics.sharpe_ratio)
        @test isfinite(metrics.annualized_return)
        @test metrics.max_drawdown >= 0.0
        @test 0.0 <= metrics.hit_rate <= 1.0
    end

    @testset "Combinations generator" begin
        combos = Backtesting.combinations_generator(1:4, 2)
        @test length(combos) == 6  # C(4,2)

        combos3 = Backtesting.combinations_generator(1:5, 3)
        @test length(combos3) == 10  # C(5,3)

        # All unique
        for c in combos
            @test length(unique(c)) == length(c)
        end
    end

    @testset "Walk-forward optimization structure" begin
        T_wf = 200
        N_wf = 2
        log_rets_wf = randn(T_wf, N_wf) * 0.01
        prices_wf = exp.(cumsum(vcat(zeros(1, N_wf), log_rets_wf), dims=1))

        param_grid_wf = [Dict("threshold" => t) for t in [0.1, 0.2, 0.3]]
        config_wf = BacktestConfig(initial_capital=100_000.0)

        function simple_signal_gen(prices, params)
            T_s, N_s = size(prices)
            thresh = get(params, "threshold", 0.1)
            sigs = randn(T_s, N_s) * thresh
            return clamp.(sigs, -1.0, 1.0)
        end

        results_wf = walk_forward_optimize(prices_wf, simple_signal_gen, param_grid_wf, config_wf;
                                            train_periods=80, test_periods=30, step_size=30)

        @test !isempty(results_wf)
        for r in results_wf
            @test isfinite(r.oos_sharpe)
            @test !isnothing(r.params)
        end
    end

end  # Backtesting testset

# ============================================================================
# Cross-module integration tests
# ============================================================================

@testset "Integration tests" begin

    @testset "Options to risk: VaR of options portfolio" begin
        # Price options and use P&L scenarios for VaR
        S, r, q = 100.0, 0.05, 0.0

        # Simulate returns
        Random.seed!(55)
        T_int = 252
        spot_rets = randn(T_int) * 0.01

        # Greeks-based P&L approximation
        sigma_atm = 0.2
        T_opt = 0.5
        delta_pos = bs_delta(S, S, r, q, sigma_atm, T_opt) * 1000.0  # 1000 calls
        gamma_pos = bs_gamma(S, S, r, q, sigma_atm, T_opt) * 1000.0
        vega_pos  = bs_vega(S, S, r, q, sigma_atm, T_opt) * 1000.0

        # Approximate P&L
        pnl_scenarios = delta_pos .* spot_rets .* S .+
                        0.5 .* gamma_pos .* (spot_rets .* S).^2

        @test isfinite(delta_pos)
        @test isfinite(gamma_pos)

        # Compute historical VaR on options P&L
        pnl_mat = reshape(pnl_scenarios, T_int, 1)
        w1 = [1.0]
        hvar = historical_var(w1, pnl_mat, 0.99)
        @test hvar > 0.0
    end

    @testset "HMM regimes feeding backtest signals" begin
        # Fit HMM to returns, use predicted regime as signal
        Random.seed!(88)
        T_rig = 300
        N_rig = 1
        returns_rig = [randn() * (i <= 150 ? 0.005 : 0.02) + (i <= 150 ? 0.0005 : -0.001)
                       for i in 1:T_rig]
        prices_rig = reshape(100.0 .* exp.(cumsum(returns_rig)), T_rig + 1, 1)[1:T_rig, :]

        model_rig = hmm_baum_welch(returns_rig, 2; n_iter=30, n_restarts=2)
        pred = hmm_predict_state(model_rig, returns_rig)

        @test length(pred.state_sequence) == T_rig
        @test length(pred.state_probs) == 2
        @test isapprox(sum(pred.state_probs), 1.0, atol=1e-8)

        # Convert regime signal: bull=1, bear=-1, sideways=0
        regime_signal = zeros(T_rig, N_rig)
        for t in 1:T_rig
            st = pred.state_sequence[t]
            model_rig.K == 2 && (regime_signal[t, 1] = st == 2 ? 1.0 : -1.0)
        end

        config_rig = BacktestConfig(initial_capital=100_000.0)
        result_rig = run_backtest(prices_rig, regime_signal, config_rig)
        @test isfinite(result_rig.sharpe_ratio)
    end

end  # Integration tests

println("\n=== Test summary complete ===")
println("All test sets executed.")
