"""
benchmarks.jl — Performance benchmarks for NeuroSDE solvers

Implements:
  1. Timing benchmarks for all SDE solvers (EM, Milstein, RK4.5)
  2. Particle filter throughput benchmarks
  3. Calibration speed benchmarks (SABR, Heston FFT)
  4. Memory profiling utilities
  5. Accuracy vs speed tradeoffs
  6. Comparison to DifferentialEquations.jl baselines (via explicit timing)
  7. GPU vs CPU throughput comparison (placeholder)
  8. Scaling benchmarks (N paths, step size, state dimension)
  9. Report generation

All benchmarks use wall-clock time via `time()` and memory via `Base.gc_live_bytes()`.
"""

using Statistics
using LinearAlgebra
using Random
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BENCHMARK UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

"""
    BenchmarkResult

Result of a single benchmark run.
"""
struct BenchmarkResult
    name        :: String
    n_trials    :: Int
    times_sec   :: Vector{Float64}   # wall time per trial
    allocs_mb   :: Vector{Float64}   # memory per trial (MB)
    result_ref  :: Any               # reference output for correctness
    mean_time   :: Float64
    std_time    :: Float64
    min_time    :: Float64
    median_time :: Float64
    mean_alloc  :: Float64
end

function BenchmarkResult(name, n_trials, times, allocs, result_ref)
    BenchmarkResult(name, n_trials, times, allocs, result_ref,
                    mean(times), std(times), minimum(times),
                    median(times), mean(allocs))
end

"""
    @benchmark_it(name, n_trials, expr)

Macro-like helper to time a Julia expression.
Usage: benchmark_fn("My bench", 10, () -> my_function(args))
"""
function benchmark_fn(name::String, n_trials::Int, fn::Function;
                       warmup::Int = 2)
    # Warmup
    result_ref = nothing
    for _ in 1:warmup
        result_ref = fn()
    end
    GC.gc()

    times  = zeros(n_trials)
    allocs = zeros(n_trials)
    for i in 1:n_trials
        GC.gc()
        mem_before = Base.gc_num().malloc
        t0 = time()
        fn()
        times[i] = time() - t0
        mem_after = Base.gc_num().malloc
        allocs[i] = max(mem_after - mem_before, 0) / (1024^2)
    end
    return BenchmarkResult(name, n_trials, times, allocs, result_ref)
end

"""
    print_benchmark(br::BenchmarkResult)
"""
function print_benchmark(br::BenchmarkResult)
    @printf "  %-35s  mean=%7.4fs  min=%7.4fs  median=%7.4fs  alloc=%7.2fMB\n" \
            br.name br.mean_time br.min_time br.median_time br.mean_alloc
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: SDE SOLVER BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    BenchmarkSuiteSDESolvers

Suite of benchmarks for SDE numerical solvers.
"""
struct BenchmarkSuiteSDESolvers
    results :: Vector{BenchmarkResult}
end

"""
    euler_maruyama_basic(N, n_steps, d; seed=1) → Matrix{Float64}

Baseline Euler-Maruyama implementation for benchmarking.
GBM: dS = μ S dt + σ S dW
"""
function euler_maruyama_basic(N::Int, n_steps::Int, d::Int;
                               seed::Int   = 1,
                               μ::Float64  = 0.05,
                               σ::Float64  = 0.20,
                               dt::Float64 = 1/252)
    rng    = MersenneTwister(seed)
    paths  = ones(d, N)
    sqdt   = sqrt(dt)
    drift  = (μ - 0.5 * σ^2) * dt
    for _ in 1:n_steps
        z     = randn(rng, d, N)
        paths = paths .* exp.(drift .+ σ .* sqdt .* z)
    end
    return paths
end

"""
    milstein_basic(N, n_steps; seed=1) → Matrix{Float64}

Milstein scheme for scalar GBM (d=1):
ΔS = μ S Δt + σ S ΔW + ½ σ² S (ΔW² - Δt)
"""
function milstein_basic(N::Int, n_steps::Int;
                         seed::Int   = 1,
                         μ::Float64  = 0.05,
                         σ::Float64  = 0.20,
                         dt::Float64 = 1/252)
    rng  = MersenneTwister(seed)
    S    = ones(N)
    sqdt = sqrt(dt)
    for _ in 1:n_steps
        z = randn(rng, N)
        W = sqdt .* z
        S = S .* (1 .+ μ .* dt .+ σ .* W .+ 0.5 .* σ^2 .* (W.^2 .- dt))
    end
    return reshape(S, 1, N)
end

"""
    heston_euler_basic(N, n_steps; seed=1) → (S_paths, V_paths)

Euler-Maruyama for Heston model (benchmark version).
"""
function heston_euler_basic(N::Int, n_steps::Int;
                              seed::Int    = 1,
                              κ::Float64   = 2.0,
                              θ::Float64   = 0.04,
                              ξ::Float64   = 0.5,
                              ρ::Float64   = -0.7,
                              r::Float64   = 0.05,
                              q::Float64   = 0.02,
                              V0::Float64  = 0.04,
                              S0::Float64  = 100.0,
                              dt::Float64  = 1/252)
    rng  = MersenneTwister(seed)
    S    = fill(S0, N)
    V    = fill(V0, N)
    sqdt = sqrt(dt)
    for _ in 1:n_steps
        z1 = randn(rng, N)
        z2 = ρ .* z1 .+ sqrt(1 - ρ^2) .* randn(rng, N)
        sqV  = sqrt.(max.(V, 0.0))
        S  = S .* exp.((r - q .- 0.5 .* V) .* dt .+ sqV .* sqdt .* z1)
        V  = max.(V .+ κ .* (θ .- V) .* dt .+ ξ .* sqV .* sqdt .* z2, 0.0)
    end
    return S, V
end

"""
    bench_sde_solvers(; n_trials=5, N_vals=[100,1000,10000],
                        n_steps=252) → BenchmarkSuiteSDESolvers

Run timing benchmarks for SDE solvers across path counts.
"""
function bench_sde_solvers(; n_trials::Int = 5,
                             N_vals::Vector{Int} = [100, 1000, 10000],
                             n_steps::Int = 252)
    results = BenchmarkResult[]

    for N in N_vals
        # Euler-Maruyama (1D)
        br = benchmark_fn("EM-GBM N=$N T=$n_steps", n_trials,
                           () -> euler_maruyama_basic(N, n_steps, 1);
                           warmup=1)
        push!(results, br)

        # Milstein (1D)
        br = benchmark_fn("Milstein-GBM N=$N T=$n_steps", n_trials,
                           () -> milstein_basic(N, n_steps); warmup=1)
        push!(results, br)

        # Heston EM
        br = benchmark_fn("Heston-EM N=$N T=$n_steps", n_trials,
                           () -> heston_euler_basic(N, n_steps); warmup=1)
        push!(results, br)
    end

    return BenchmarkSuiteSDESolvers(results)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PARTICLE FILTER BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    bootstrap_filter_simple(N, T; seed=1) → Vector{Float64}

Simple bootstrap filter on linear Gaussian model for benchmarking.
"""
function bootstrap_filter_simple(N::Int, T::Int;
                                  seed::Int   = 1,
                                  σ_proc::Float64 = 1.0,
                                  σ_obs::Float64  = 0.5)
    rng   = MersenneTwister(seed)
    # Simulate observations
    x_true = cumsum(randn(rng, T))
    y_obs  = x_true .+ σ_obs .* randn(rng, T)

    # Bootstrap filter
    particles = randn(rng, N)
    weights   = fill(1.0/N, N)
    fil_mean  = zeros(T)

    cumw = zeros(N)
    for t in 1:T
        # Propagate
        particles .+= σ_proc .* randn(rng, N)
        # Weight
        for i in 1:N
            weights[i] = exp(-0.5 * ((y_obs[t] - particles[i]) / σ_obs)^2)
        end
        wsum = sum(weights)
        weights ./= wsum
        fil_mean[t] = dot(particles, weights)

        # Resample (systematic)
        ess = 1.0 / sum(weights.^2)
        if ess < 0.5 * N
            cumsum!(cumw, weights)
            u0  = rand(rng) / N
            j   = 1
            new_p = similar(particles)
            for i in 1:N
                u = u0 + (i-1)/N
                while j < N && cumw[j] < u; j += 1; end
                new_p[i] = particles[j]
            end
            particles = new_p
            fill!(weights, 1.0/N)
        end
    end
    return fil_mean
end

"""
    bench_particle_filters(; n_trials=5, N_vals=[100,500,2000,10000], T=500)
"""
function bench_particle_filters(; n_trials::Int = 5,
                                  N_vals::Vector{Int} = [100, 500, 2000, 10000],
                                  T::Int = 500)
    results = BenchmarkResult[]
    for N in N_vals
        br = benchmark_fn("BootstrapFilter N=$N T=$T", n_trials,
                           () -> bootstrap_filter_simple(N, T); warmup=1)
        push!(results, br)
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CALIBRATION BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    sabr_smile_bench(n_strikes; seed=1) → Vector{Float64}

Benchmark SABR smile computation.
"""
function sabr_smile_bench(n_strikes::Int; seed::Int=1)
    rng = MersenneTwister(seed)
    F   = 100.0; T = 1.0
    strikes = collect(range(80.0, 120.0, length=n_strikes))
    α, β, ρ, σ0 = 0.3, 0.5, -0.3, 0.2
    vols = Float64[]
    for K in strikes
        # Inline SABR formula
        if abs(F - K) < 1e-10 * F
            FKβ = F^(1-β)
            v   = σ0/FKβ * (1 + ((1-β)^2/24*σ0^2/FKβ^2 + ρ*β*α*σ0/(4*FKβ) + (2-3ρ^2)/24*α^2)*T)
        else
            FK    = F * K
            FKm   = FK^((1-β)/2)
            logFK = log(F/K)
            z     = α/σ0 * FKm * logFK
            χ     = log((sqrt(1 - 2ρ*z + z^2) + z - ρ) / (1-ρ))
            zχ    = abs(χ) < 1e-10 ? 1.0 : z/χ
            denom = FKm * (1 + (1-β)^2/24*logFK^2 + (1-β)^4/1920*logFK^4)
            corr  = 1 + ((1-β)^2/24*σ0^2/FK^(1-β) + ρ*β*α*σ0/(4*FK^((1-β)/2)) + (2-3ρ^2)/24*α^2)*T
            v     = σ0/denom * zχ * corr
        end
        push!(vols, v)
    end
    return vols
end

"""
    heston_fft_bench(n_strikes; N_fft=4096) → Vector{Float64}

Benchmark Heston FFT pricing.
"""
function heston_fft_bench(n_strikes::Int; N_fft::Int=4096)
    S, r, q = 100.0, 0.05, 0.02; T = 1.0
    strikes = collect(range(80.0, 120.0, length=n_strikes))
    κ, θ, ξ, ρ, V0 = 2.0, 0.04, 0.5, -0.7, 0.04
    # Simplified FFT pricing (copy of core logic from calibration_engine.jl)
    η    = 0.25; α = 1.5
    λ_f  = 2π / (N_fft * η)
    b    = N_fft * λ_f / 2
    ku   = [-b + λ_f * (j-1) for j in 1:N_fft]
    νs   = [η * (j-1) for j in 1:N_fft]

    function char_fn(u)
        iu = im * u
        d  = sqrt((ρ*ξ*iu - κ)^2 + ξ^2*(iu + u^2))
        g  = (κ - ρ*ξ*iu - d) / (κ - ρ*ξ*iu + d)
        ed = exp(-d*T)
        B  = (κ - ρ*ξ*iu - d) * (1-ed) / (ξ^2 * (1 - g*ed))
        A  = iu*(r-q)*T + κ*θ/ξ^2*((κ - ρ*ξ*iu - d)*T - 2*log((1-g*ed)/(1-g)))
        return exp(A + B*V0 + iu*log(S))
    end

    integrand = [begin
        ν = νs[j]
        ψ = char_fn(ν - (α+1)*im) / (α^2 + α - ν^2 + im*(2α+1)*ν)
        w = (j==1||j==N_fft) ? 1.0 : (iseven(j) ? 4.0 : 2.0)
        exp(-im*b*ν) * ψ * w * η / 3
    end for j in 1:N_fft]

    using FFTW: fft
    fft_val = fft(integrand)
    prices_ku = real.(exp.(-α .* ku) ./ π .* fft_val)

    log_K = log.(strikes ./ S)
    prices = zeros(n_strikes)
    for (i, lk) in enumerate(log_K)
        idx = searchsortedfirst(ku, lk)
        idx = clamp(idx, 2, N_fft)
        frac = (lk - ku[idx-1]) / (ku[idx] - ku[idx-1])
        prices[i] = prices_ku[idx-1]*(1-frac) + prices_ku[idx]*frac
    end
    return max.(prices, 0.0)
end

"""
    bench_calibration(; n_trials=5) → Vector{BenchmarkResult}
"""
function bench_calibration(; n_trials::Int=5)
    results = BenchmarkResult[]
    for nK in [5, 20, 50]
        br = benchmark_fn("SABR-smile K=$nK", n_trials,
                           () -> sabr_smile_bench(nK); warmup=1)
        push!(results, br)
    end
    for nK in [5, 20, 50]
        br = benchmark_fn("Heston-FFT K=$nK", n_trials,
                           () -> heston_fft_bench(nK, N_fft=1024); warmup=1)
        push!(results, br)
    end
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: ACCURACY BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    AccuracyBenchmark

Measures accuracy of a numerical method against an exact solution.
"""
struct AccuracyBenchmark
    method_name  :: String
    dt_vals      :: Vector{Float64}
    rmse_vals    :: Vector{Float64}
    time_vals    :: Vector{Float64}
    order        :: Float64    # estimated convergence order
end

"""
    gbm_exact(S0, r, q, σ, T, z) → Float64

Exact GBM value at time T given standard normal z.
"""
gbm_exact(S0, r, q, σ, T, z) = S0 * exp((r - q - 0.5*σ^2)*T + σ*sqrt(T)*z)

"""
    bench_accuracy_euler(; N=10000, seed=42) → AccuracyBenchmark

Convergence study for Euler-Maruyama vs exact GBM.
Strong convergence order should be ~0.5.
"""
function bench_accuracy_euler(; N::Int=10000, seed::Int=42)
    dt_vals = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    rmse_v  = Float64[]
    time_v  = Float64[]
    rng     = MersenneTwister(seed)
    S0, r, q, σ, T = 100.0, 0.05, 0.02, 0.20, 1.0

    # Fix Brownian paths
    n_max  = Int(round(T / minimum(dt_vals)))
    Z_full = randn(rng, n_max, N)   # (n_max × N)

    for dt in dt_vals
        n_steps = Int(round(T / dt))
        step_size = n_max ÷ n_steps

        t0 = time()
        S  = fill(S0, N)
        for i in 1:n_steps
            idx = (i-1) * step_size + 1
            z_i = Z_full[idx:min(idx+step_size-1, n_max), :]
            dW  = sum(z_i, dims=1)[:] .* sqrt(dt / step_size)
            S   = S .* exp.((r - q - 0.5*σ^2)*dt .+ σ .* dW)
        end
        elapsed = time() - t0

        # Exact solution using same total BM increment
        dW_total = sum(Z_full, dims=1)[:] .* sqrt(T / n_max)
        S_exact  = [gbm_exact(S0, r, q, σ, T, dW_total[i] / sqrt(T)) for i in 1:N]
        rmse     = sqrt(mean((S .- S_exact).^2))

        push!(rmse_v, rmse)
        push!(time_v, elapsed)
    end

    # Estimate convergence order: log(rmse1/rmse2) / log(dt1/dt2)
    orders = [log(rmse_v[i]/rmse_v[i+1]) / log(dt_vals[i]/dt_vals[i+1])
              for i in 1:length(dt_vals)-1]
    order  = mean(orders)

    return AccuracyBenchmark("Euler-Maruyama (GBM)", dt_vals, rmse_v, time_v, order)
end

"""
    bench_accuracy_milstein(; N=10000, seed=42) → AccuracyBenchmark

Convergence study for Milstein scheme.
Strong convergence order should be ~1.0.
"""
function bench_accuracy_milstein(; N::Int=10000, seed::Int=42)
    dt_vals = [0.1, 0.05, 0.02, 0.01, 0.005]
    rmse_v  = Float64[]
    time_v  = Float64[]
    rng     = MersenneTwister(seed)
    S0, r, q, σ, T = 100.0, 0.05, 0.02, 0.20, 1.0

    n_max  = Int(round(T / minimum(dt_vals)))
    Z_full = randn(rng, n_max, N)

    for dt in dt_vals
        n_steps = Int(round(T / dt))
        step_size = n_max ÷ n_steps

        t0 = time()
        S  = fill(S0, N)
        for i in 1:n_steps
            idx = (i-1) * step_size + 1
            z_i = Z_full[idx:min(idx+step_size-1, n_max), :]
            dW  = sum(z_i, dims=1)[:] .* sqrt(dt / step_size)
            S   = S .* (1 .+ (r-q)*dt .+ σ.*dW .+ 0.5*σ^2*(dW.^2 .- dt))
        end
        elapsed = time() - t0

        dW_total = sum(Z_full, dims=1)[:] .* sqrt(T / n_max)
        S_exact  = [gbm_exact(S0, r, q, σ, T, dW_total[i] / sqrt(T)) for i in 1:N]
        rmse     = sqrt(mean((S .- S_exact).^2))

        push!(rmse_v, rmse)
        push!(time_v, elapsed)
    end

    orders = [log(rmse_v[i]/rmse_v[i+1]) / log(dt_vals[i]/dt_vals[i+1])
              for i in 1:length(dt_vals)-1]
    order  = mean(orders)

    return AccuracyBenchmark("Milstein (GBM)", dt_vals, rmse_v, time_v, order)
end

"""
    print_accuracy_benchmark(ab::AccuracyBenchmark)
"""
function print_accuracy_benchmark(ab::AccuracyBenchmark)
    @printf "  %s  (est. order = %.2f)\n" ab.method_name ab.order
    @printf "  %-10s  %-12s  %-10s\n" "dt" "RMSE" "time(s)"
    for i in 1:length(ab.dt_vals)
        @printf "  %-10.4f  %-12.6f  %-10.4f\n" \
                ab.dt_vals[i] ab.rmse_vals[i] ab.time_vals[i]
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MEMORY PROFILING
# ─────────────────────────────────────────────────────────────────────────────

"""
    MemoryProfile

Memory usage profile for a computation.
"""
struct MemoryProfile
    name         :: String
    before_mb    :: Float64
    after_mb     :: Float64
    peak_mb      :: Float64
    allocated_mb :: Float64
end

"""
    profile_memory(name, fn) → MemoryProfile

Profile memory usage of a function call.
"""
function profile_memory(name::String, fn::Function)
    GC.gc()
    before = Base.gc_live_bytes() / (1024^2)
    alloc_before = Base.gc_num().malloc

    fn()

    GC.gc()
    after = Base.gc_live_bytes() / (1024^2)
    alloc_after = Base.gc_num().malloc
    allocated_mb = max(alloc_after - alloc_before, 0) / (1024^2)

    return MemoryProfile(name, before, after, max(before, after), allocated_mb)
end

"""
    bench_memory_scaling(; N_vals=[100,1000,10000], n_steps=252) → Vector{MemoryProfile}

Profile memory usage as a function of N paths.
"""
function bench_memory_scaling(; N_vals::Vector{Int}=[100,1000,10000],
                                n_steps::Int=252)
    profiles = MemoryProfile[]
    for N in N_vals
        p = profile_memory("Heston-EM N=$N",
                            () -> heston_euler_basic(N, n_steps))
        push!(profiles, p)
    end
    return profiles
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: COMPARISON TO BASELINES
# ─────────────────────────────────────────────────────────────────────────────

"""
    ComparisonResult

Comparison between our implementation and a reference.
"""
struct ComparisonResult
    our_method   :: String
    reference    :: String
    our_time     :: Float64
    ref_time     :: Float64
    speedup      :: Float64   # ref_time / our_time
    our_rmse     :: Float64
    ref_rmse     :: Float64
    notes        :: String
end

"""
    baseline_euler_reference(N, n_steps; seed=1) → Float64

Reference EM implementation (pure Julia, no optimisations).
Used as baseline for speedup comparison.
"""
function baseline_euler_reference(N::Int, n_steps::Int;
                                   seed::Int   = 1,
                                   σ::Float64  = 0.20,
                                   dt::Float64 = 1/252)
    rng = MersenneTwister(seed)
    S   = ones(N)
    for _ in 1:n_steps
        for i in 1:N
            S[i] *= exp((-0.5*σ^2)*dt + σ*sqrt(dt)*randn(rng))
        end
    end
    return mean(S)
end

"""
    bench_comparison(; n_trials=5, N=5000, n_steps=252) → Vector{ComparisonResult}

Compare vectorised vs scalar implementations.
"""
function bench_comparison(; n_trials::Int=5, N::Int=5000, n_steps::Int=252)
    results = ComparisonResult[]

    # Scalar baseline
    br_ref = benchmark_fn("Reference-Scalar N=$N", n_trials,
                           () -> baseline_euler_reference(N, n_steps); warmup=1)

    # Vectorised
    br_our = benchmark_fn("Vectorised-EM N=$N", n_trials,
                           () -> euler_maruyama_basic(N, n_steps, 1); warmup=1)

    speedup = br_ref.mean_time / max(br_our.mean_time, 1e-10)
    push!(results, ComparisonResult(
        "Vectorised EM", "Scalar EM",
        br_our.mean_time, br_ref.mean_time, speedup,
        NaN, NaN,
        "Vectorised uses broadcast; scalar uses explicit loop"
    ))

    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: SCALING BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

"""
    ScalingBenchmark

Result of a scaling study (throughput vs problem size).
"""
struct ScalingBenchmark
    parameter    :: String   # name of the scaling parameter
    values       :: Vector{Int}
    throughputs  :: Vector{Float64}  # paths/second or similar
    times        :: Vector{Float64}
    fits_linear  :: Bool     # does throughput scale linearly?
end

"""
    bench_path_scaling(; n_trials=3, n_steps=252) → ScalingBenchmark

Benchmark throughput (paths/second) as a function of N.
"""
function bench_path_scaling(; n_trials::Int=3, n_steps::Int=252)
    N_vals = [100, 500, 1000, 5000, 10000, 50000]
    times  = Float64[]
    thrp   = Float64[]

    for N in N_vals
        br = benchmark_fn("EM N=$N", n_trials,
                           () -> euler_maruyama_basic(N, n_steps, 1); warmup=1)
        push!(times, br.mean_time)
        push!(thrp, N / br.mean_time)
    end

    # Check linear scaling: if T ∝ N, then throughput is constant
    # Use log-log regression
    log_N = log.(Float64.(N_vals))
    log_T = log.(times)
    slope = cov(log_N, log_T) / var(log_N)   # should be ≈ 1

    return ScalingBenchmark("N_paths", N_vals, thrp, times, abs(slope - 1.0) < 0.2)
end

"""
    bench_step_scaling(; n_trials=3, N=1000) → ScalingBenchmark
"""
function bench_step_scaling(; n_trials::Int=3, N::Int=1000)
    step_vals = [10, 50, 100, 252, 500, 1000, 2000]
    times  = Float64[]
    thrp   = Float64[]

    for n_steps in step_vals
        br = benchmark_fn("EM T=$n_steps", n_trials,
                           () -> euler_maruyama_basic(N, n_steps, 1); warmup=1)
        push!(times, br.mean_time)
        push!(thrp, n_steps / br.mean_time)
    end

    log_T  = log.(Float64.(step_vals))
    log_TM = log.(times)
    slope  = cov(log_T, log_TM) / var(log_T)

    return ScalingBenchmark("n_steps", step_vals, thrp, times, abs(slope - 1.0) < 0.2)
end

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: BENCHMARK REPORT
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_all_benchmarks(; quick=true) → NamedTuple

Run all benchmarks and return results.
`quick=true` uses fewer trials and smaller problem sizes.
"""
function run_all_benchmarks(; quick::Bool=true)
    n_trials = quick ? 3 : 10
    N_vals   = quick ? [100, 1000] : [100, 1000, 10000]
    n_steps  = quick ? 50 : 252

    println("\n" * "═"^60)
    println("  NeuroSDE Benchmark Suite")
    println("═"^60)

    println("\n  SDE Solvers:")
    solver_bench = bench_sde_solvers(; n_trials=n_trials, N_vals=N_vals, n_steps=n_steps)
    for br in solver_bench.results
        print_benchmark(br)
    end

    println("\n  Particle Filters:")
    pf_bench = bench_particle_filters(; n_trials=n_trials,
                                        N_vals=quick ? [100, 500] : [100, 500, 2000],
                                        T=quick ? 100 : 500)
    for br in pf_bench
        print_benchmark(br)
    end

    println("\n  Calibration:")
    cal_bench = bench_calibration(; n_trials=n_trials)
    for br in cal_bench
        print_benchmark(br)
    end

    println("\n  Accuracy (Strong Convergence):")
    if !quick
        ab_em = bench_accuracy_euler(; N=5000)
        print_accuracy_benchmark(ab_em)
        ab_m  = bench_accuracy_milstein(; N=5000)
        print_accuracy_benchmark(ab_m)
    end

    println("\n  Path Scaling:")
    scale_br = bench_path_scaling(; n_trials=n_trials)
    @printf "  Linear scaling: %s  (slope≈1 ↔ linear)\n" string(scale_br.fits_linear)
    for (i, N) in enumerate(scale_br.values)
        @printf "  N=%-6d  throughput=%.0f paths/s\n" N scale_br.throughputs[i]
    end

    println("═"^60)
    return (
        solver_bench = solver_bench,
        pf_bench     = pf_bench,
        cal_bench    = cal_bench,
        scale_bench  = scale_br,
    )
end
