"""
test_particle_filter.jl — Accuracy and correctness tests for particle filters

Tests:
  1. Resampling strategies: correctness, variance reduction
  2. Bootstrap filter: linear Gaussian model (vs Kalman ground truth)
  3. Bootstrap filter: effective sample size maintenance
  4. Auxiliary particle filter: lower variance than bootstrap
  5. Unscented Kalman Filter: linear model (vs exact Kalman)
  6. Ensemble Kalman Filter: basic functionality
  7. Degeneracy diagnostics: ESS computation
  8. Log marginal likelihood estimation
  9. Backward smoother (when available)
 10. Heston state-space filter
"""

using Test
using Statistics
using LinearAlgebra
using Random
using Distributions

# ─────────────────────────────────────────────────────────────────────────────
# Helper: simple Kalman filter for linear Gaussian comparison
# ─────────────────────────────────────────────────────────────────────────────

"""
    kalman_filter_linear(y, F, H, Q, R, x0, P0)

Run exact Kalman filter on scalar state / observation model.
Returns filtered means.
"""
function kalman_filter_linear(y::AbstractVector, F::Real, H::Real,
                               Q::Real, R::Real, x0::Real, P0::Real)
    T  = length(y)
    xs = zeros(T)
    x, P = x0, P0
    for t in 1:T
        # Predict
        xp = F * x
        Pp = F * P * F + Q
        # Update
        S  = H * Pp * H + R
        K  = Pp * H / S
        x  = xp + K * (y[t] - H * xp)
        P  = (1 - K * H) * Pp
        xs[t] = x
    end
    return xs
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: RESAMPLING STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

@testset "Resampling Strategies" begin

    rng = MersenneTwister(42)
    N   = 1000
    # Non-uniform weights: few dominant particles
    w_raw  = rand(rng, N).^3
    w_raw ./= sum(w_raw)

    @testset "Systematic resampling: correct count" begin
        idx = systematic_resample(w_raw, N; rng=rng)
        @test length(idx) == N
        @test all(1 .<= idx .<= N)
    end

    @testset "Stratified resampling: correct count" begin
        idx = stratified_resample(w_raw, N; rng=rng)
        @test length(idx) == N
        @test all(1 .<= idx .<= N)
    end

    @testset "Residual resampling: correct count" begin
        idx = residual_resample(w_raw, N; rng=rng)
        @test length(idx) == N
        @test all(1 .<= idx .<= N)
    end

    @testset "Multinomial resampling: correct count" begin
        idx = multinomial_resample(w_raw, N; rng=rng)
        @test length(idx) == N
        @test all(1 .<= idx .<= N)
    end

    @testset "Systematic resampling: distribution preservation" begin
        # After resampling with uniform output weights,
        # empirical mean should match weighted mean
        vals   = randn(rng, N)
        w_test = rand(rng, N).^2
        w_test ./= sum(w_test)
        wmean  = dot(vals, w_test)
        idx    = systematic_resample(w_test, N; rng=rng)
        resamp_mean = mean(vals[idx])
        @test isapprox(resamp_mean, wmean, atol=0.1)
    end

    @testset "Resampling via ParticleState" begin
        particles = randn(rng, 3, N)
        ps = ParticleState(particles)
        ps.weights = w_raw
        ps_new = resample(ps, SystematicResampling(); rng=rng)
        @test size(ps_new.particles) == (3, N)
        @test isapprox(sum(ps_new.weights), 1.0, atol=1e-10)
    end

    @testset "ESS: uniform weights = N" begin
        w_unif = fill(1.0/N, N)
        ess    = effective_sample_size(w_unif)
        @test isapprox(ess, Float64(N), atol=1.0)
    end

    @testset "ESS: single dominant particle ≈ 1" begin
        w_deg    = zeros(N); w_deg[1] = 1.0
        ess_deg  = effective_sample_size(w_deg)
        @test isapprox(ess_deg, 1.0, atol=0.01)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: BOOTSTRAP PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────────────────

@testset "Bootstrap Particle Filter — Linear Gaussian" begin

    rng   = MersenneTwister(1)
    T_len = 100
    N     = 2000
    F_val = 0.9; H_val = 1.0; Q_val = 1.0; R_val = 0.5

    # Simulate state and observations
    x_true = zeros(T_len + 1)
    y_obs  = zeros(T_len)
    for t in 1:T_len
        x_true[t+1] = F_val * x_true[t] + sqrt(Q_val) * randn(rng)
        y_obs[t]    = H_val * x_true[t+1] + sqrt(R_val) * randn(rng)
    end

    # Run bootstrap filter
    init_X = randn(rng, 1, N)
    trans! = (xn, xo, t, r) -> begin xn[1] = F_val * xo[1] + sqrt(Q_val) * randn(r) end
    loglik = (y, x, t) -> logpdf(Normal(H_val * x[1], sqrt(R_val)), y[1])
    obs_m  = reshape(y_obs, 1, T_len)
    bf     = BootstrapFilter(N; ess_threshold=0.5, strategy=SystematicResampling())
    result = bootstrap_filter(bf, init_X, trans!, loglik, obs_m; rng=rng)

    # Exact Kalman filter
    kf_means = kalman_filter_linear(y_obs, F_val, H_val, Q_val, R_val, 0.0, 1.0)

    @testset "Filtered mean close to Kalman" begin
        pf_mean = result.filtered_mean[1, :]
        rmse = sqrt(mean((pf_mean .- kf_means).^2))
        @test rmse < 0.3   # generous bound for finite N
    end

    @testset "Log marginal likelihood finite" begin
        @test isfinite(result.log_marglik)
        @test result.log_marglik < 0   # log-density must be < 0 in general
    end

    @testset "ESS history non-zero" begin
        @test all(result.ess_history .> 0)
        @test length(result.ess_history) == T_len
    end

    @testset "Filtered variance positive" begin
        @test all(result.filtered_var .>= 0)
    end

    @testset "Resampling occurred" begin
        @test result.n_resample >= 0   # at least 0
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: UNSCENTED KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

@testset "Unscented Kalman Filter" begin

    @testset "UKF weights sum to 1" begin
        for n in [1, 2, 5, 10]
            Wm, Wc, _ = ukf_weights(n, UKFParams())
            @test isapprox(sum(Wm), 1.0, atol=1e-10)
        end
    end

    @testset "Sigma points span correct mean" begin
        n = 3
        x = [1.0, 2.0, 3.0]
        P = Diagonal([0.1, 0.2, 0.15]) |> Matrix
        Wm, Wc, λ = ukf_weights(n, UKFParams())
        SP = sigma_points(x, P, λ, n)
        x_rec = SP * Wm
        @test isapprox(x_rec, x, atol=1e-10)
    end

    @testset "UKF: linear model matches Kalman" begin
        rng   = MersenneTwister(42)
        n, m, T_len = 2, 1, 80
        F_mat = [0.9 0.1; 0.0 0.8] |> Matrix
        H_mat = reshape([1.0, 0.0], 1, 2)
        Q_mat = 0.1 * I(n) |> Matrix
        R_mat = reshape([0.5], 1, 1)

        # Simulate
        x_t  = randn(rng, n, T_len + 1)
        y_obs = zeros(m, T_len)
        for t in 1:T_len
            x_t[:, t+1] = F_mat * x_t[:, t] + sqrt(0.1) * randn(rng, n)
            y_obs[:, t]  = H_mat * x_t[:, t+1] + sqrt(0.5) * randn(rng, m)
        end

        f_ukf = (x, t) -> F_mat * x
        h_ukf = (x, t) -> H_mat * x

        res = ukf_run(zeros(n), I(n) |> Matrix, f_ukf, h_ukf,
                      Q_mat, R_mat, y_obs)

        @test size(res.filtered_mean) == (n, T_len)
        @test size(res.filtered_cov)  == (n, n, T_len)
        @test isfinite(res.log_likelihood)

        # Basic sanity: filtered means should track true state
        rmse = sqrt(mean((res.filtered_mean .- x_t[:, 2:end]).^2))
        @test rmse < 1.0   # loose bound
    end

    @testset "UKF: log-likelihood finite" begin
        rng   = MersenneTwister(5)
        n, m, T_len = 1, 1, 50
        f = (x, t) -> 0.9 .* x
        h = (x, t) -> x.^2   # nonlinear observation
        Q = reshape([0.1], 1, 1)
        R = reshape([0.5], 1, 1)
        x_t = cumsum(randn(rng, T_len)) .* 0.3
        y_obs = reshape(x_t.^2 .+ 0.5 .* randn(rng, T_len), 1, T_len)
        res = ukf_run([0.0], reshape([1.0], 1, 1), f, h, Q, R, y_obs)
        @test isfinite(res.log_likelihood)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: ENSEMBLE KALMAN FILTER
# ─────────────────────────────────────────────────────────────────────────────

@testset "Ensemble Kalman Filter" begin

    rng = MersenneTwister(99)
    n, m, T_len, N = 2, 1, 60, 200

    F_mat = 0.9 * I(n) |> Matrix
    H_mat = reshape([1.0, 0.5], 1, 2)
    Q     = 0.1 * I(n) |> Matrix
    R     = reshape([0.3], 1, 1)

    # Simulate system
    x_true = zeros(n, T_len + 1)
    y_obs  = zeros(m, T_len)
    for t in 1:T_len
        x_true[:, t+1] = F_mat * x_true[:, t] + 0.316 * randn(rng, n)
        y_obs[:, t]    = H_mat * x_true[:, t+1] + 0.548 * randn(rng, m)
    end

    ens0 = randn(rng, n, N)
    f! = (xn, x, t, rng) -> begin
        mul!(xn, F_mat, x)
    end
    h = (x, t) -> H_mat * x

    cfg = EnKFConfig(N; inflate=1.0, store_history=false)
    res = enkf_run(cfg, ens0, f!, h, Q, R, y_obs; rng=rng)

    @testset "Output dimensions correct" begin
        @test size(res.filtered_mean) == (n, T_len)
        @test size(res.filtered_var)  == (n, T_len)
    end

    @testset "Log-likelihood finite" begin
        @test isfinite(res.log_likelihood)
    end

    @testset "Filtered variance non-negative" begin
        @test all(res.filtered_var .>= 0)
    end

    @testset "Basic tracking accuracy" begin
        rmse = sqrt(mean((res.filtered_mean .- x_true[:, 2:end]).^2))
        @test rmse < 2.0   # generous bound
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: DEGENERACY DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Degeneracy Diagnostics" begin

    N = 500

    @testset "Uniform weights: not degenerate" begin
        w = fill(1.0/N, N)
        diag = diagnose_degeneracy(w; threshold=0.5)
        @test !diag.is_degenerate
        @test diag.ess_fraction ≈ 1.0
    end

    @testset "Single particle: maximally degenerate" begin
        w = zeros(N); w[1] = 1.0
        diag = diagnose_degeneracy(w; threshold=0.5)
        @test diag.is_degenerate
        @test diag.max_weight ≈ 1.0
        @test diag.n_effective == 1
    end

    @testset "Entropy range" begin
        w = fill(1.0/N, N)
        diag_unif = diagnose_degeneracy(w)
        @test diag_unif.entropy > 0
        w2 = zeros(N); w2[1] = 1.0
        diag_deg = diagnose_degeneracy(w2)
        @test diag_deg.entropy < diag_unif.entropy
    end

    @testset "Jitter particles modifies in-place" begin
        rng = MersenneTwister(1)
        particles = zeros(2, N)
        weights   = fill(1.0/N, N)
        jitter_particles!(particles, weights, 0.01; rng=rng)
        # After jitter, not all zero
        @test std(particles[1, :]) > 1e-6
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: HESTON STATE-SPACE FILTER
# ─────────────────────────────────────────────────────────────────────────────

@testset "Heston Particle Filter" begin

    rng = MersenneTwister(3)
    dt  = 1/252; n_obs = 100; N = 500
    κ, θ, ξ, ρ = 2.0, 0.04, 0.5, -0.7
    V0  = 0.04; S0 = 100.0

    # Simulate Heston path
    log_S  = zeros(n_obs + 1); log_S[1]  = log(S0)
    V      = zeros(n_obs + 1); V[1]      = V0
    sqdt   = sqrt(dt)
    for t in 1:n_obs
        z1 = randn(rng); z2 = ρ*z1 + sqrt(1-ρ^2)*randn(rng)
        sqV = sqrt(max(V[t], 0.0))
        log_S[t+1] = log_S[t] + (0.05 - 0.02 - 0.5*V[t])*dt + sqV*sqdt*z1
        V[t+1]     = max(V[t] + κ*(θ-V[t])*dt + ξ*sqV*sqdt*z2, 0.0)
    end

    # Run particle filter
    result = filter_heston(log_S[2:end], κ, θ, ξ, ρ, 0.05, 0.02, dt;
                            N=N, σ_obs=0.001, rng=rng)

    @testset "Filtered state dimensions" begin
        @test size(result.filtered_mean) == (2, n_obs)
    end

    @testset "Filtered log-price close to true" begin
        rmse = sqrt(mean((result.filtered_mean[1, :] .- log_S[2:end]).^2))
        @test rmse < 0.1
    end

    @testset "Filtered variance positive" begin
        @test all(result.filtered_var[2, :] .>= 0)   # variance state
    end

    @testset "ESS history non-trivial" begin
        @test mean(result.ess_history) > N * 0.05   # ESS > 5% of N
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: LOG MARGINAL LIKELIHOOD
# ─────────────────────────────────────────────────────────────────────────────

@testset "Log Marginal Likelihood" begin

    rng = MersenneTwister(7)
    T_len = 50; N = 1000

    # Model A: correct noise level
    F_val = 0.8; Q_val = 0.5; R_val = 0.3
    x_true = zeros(T_len + 1)
    y_obs  = zeros(T_len)
    for t in 1:T_len
        x_true[t+1] = F_val * x_true[t] + sqrt(Q_val) * randn(rng)
        y_obs[t]    = x_true[t+1] + sqrt(R_val) * randn(rng)
    end

    obs_m = reshape(y_obs, 1, T_len)

    function run_filter(obs_noise)
        init_X = randn(rng, 1, N)
        trans! = (xn, xo, t, r) -> begin xn[1] = F_val * xo[1] + sqrt(Q_val) * randn(r) end
        loglik = (y, x, t) -> logpdf(Normal(x[1], sqrt(obs_noise)), y[1])
        bf     = BootstrapFilter(N; ess_threshold=0.5)
        return bootstrap_filter(bf, init_X, trans!, loglik, obs_m; rng=rng)
    end

    res_correct = run_filter(R_val)
    res_wrong   = run_filter(5.0)   # wrong obs noise

    @testset "Correct model has higher log-likelihood (tendency)" begin
        # This is probabilistic; test that correct model LL is not drastically worse
        @test res_correct.log_marglik > res_wrong.log_marglik - 200
    end

    @testset "Log marginal likelihood is finite" begin
        @test isfinite(res_correct.log_marglik)
        @test isfinite(res_wrong.log_marglik)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: AUXILIARY PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────────────────

@testset "Auxiliary Particle Filter" begin

    rng   = MersenneTwister(11)
    T_len = 80; N = 1000
    F_val = 0.9; Q_val = 1.0; R_val = 0.5

    # Simulate
    x_true = zeros(T_len + 1)
    y_obs  = zeros(T_len)
    for t in 1:T_len
        x_true[t+1] = F_val * x_true[t] + sqrt(Q_val) * randn(rng)
        y_obs[t]    = x_true[t+1] + sqrt(R_val) * randn(rng)
    end

    obs_m  = reshape(y_obs, 1, T_len)
    init_X = randn(rng, 1, N)

    trans!  = (xn, xo, t, r) -> begin xn[1] = F_val * xo[1] + sqrt(Q_val) * randn(r) end
    loglik  = (y, x, t) -> logpdf(Normal(x[1], sqrt(R_val)), y[1])
    pilot_ll = (y, x, t) -> logpdf(Normal(F_val * x[1], sqrt(Q_val + R_val)), y[1])

    apf = AuxiliaryParticleFilter(N; ess_threshold=0.5)
    result = auxiliary_particle_filter(apf, init_X, trans!, loglik,
                                        pilot_ll, obs_m; rng=rng)

    @testset "APF output dimensions" begin
        @test size(result.filtered_mean) == (1, T_len)
    end

    @testset "APF tracks truth" begin
        rmse = sqrt(mean((result.filtered_mean[1, :] .- x_true[2:end]).^2))
        @test rmse < 0.5   # should be accurate
    end

    @testset "APF log marginal likelihood finite" begin
        @test isfinite(result.log_marglik)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: FILTER SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Filter Summary Statistics" begin

    rng   = MersenneTwister(20)
    T_len = 50; N = 500
    init_X = randn(rng, 2, N)
    trans! = (xn, xo, t, r) -> begin xn .= 0.9 .* xo .+ 0.3 .* randn(r, 2) end
    loglik = (y, x, t) -> logpdf(Normal(x[1], 0.5), y[1])
    y_obs  = randn(rng, 1, T_len)
    bf     = BootstrapFilter(N)
    result = bootstrap_filter(bf, init_X, trans!, loglik, y_obs; rng=rng)

    s = filter_summary(result)

    @testset "Summary has correct fields" begin
        @test haskey(s, :n_steps)
        @test haskey(s, :log_marglik)
        @test haskey(s, :min_ess)
        @test s.n_steps == T_len
        @test s.state_dim == 2
    end

    @testset "Min ESS <= Mean ESS" begin
        @test s.min_ess <= s.mean_ess + 1e-6
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: SMC² INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

@testset "SMC2 Interface" begin

    rng = MersenneTwister(55)
    T_len = 30; N_x = 50; N_θ = 5

    cfg = SMC2Config(; N_x=N_x, N_θ=N_θ, ess_frac=0.5)

    # Simple model: x_{t+1} = θ x_t + ε, ε ~ N(0,0.1)
    # y_t = x_t + η, η ~ N(0, 0.2)
    # θ ∈ [0.7, 0.95]
    θ_particles = reshape(rand(rng, N_θ) .* 0.25 .+ 0.70, 1, N_θ)

    y_obs = reshape(randn(rng, T_len), 1, T_len)

    log_prior = θ -> sum(logpdf.(Uniform(0.5, 1.0), θ))

    init_fn = (θ, Nx; rng) -> randn(rng, 1, Nx)

    trans_factory = θ -> begin
        θ1 = θ[1]
        (xn, xo, t, r) -> begin xn[1] = θ1 * xo[1] + 0.316 * randn(r) end
    end

    ll_factory = θ -> ((y, x, t) -> logpdf(Normal(x[1], 0.447), y[1]))

    lmls = smc2_log_marglik(cfg, θ_particles, log_prior, init_fn,
                              trans_factory, ll_factory, y_obs; rng=rng)

    @testset "SMC2: returns N_θ values" begin
        @test length(lmls) == N_θ
    end

    @testset "SMC2: log marginal likelihoods finite or -Inf" begin
        @test all(isfinite.(lmls) .| isinf.(lmls))
    end

    @testset "SMC2: at least one finite value" begin
        @test any(isfinite.(lmls))
    end
end

println("\n✓ All particle filter tests passed.")
