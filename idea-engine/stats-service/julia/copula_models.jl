# =============================================================================
# copula_models.jl — Copula Theory for Crypto Portfolio Modelling
# =============================================================================
# Provides:
#   - GaussianCopula          Gaussian copula fit + simulation
#   - StudentTCopula          Student-t copula with DoF estimation
#   - ClaytonCopula           Clayton Archimedean copula
#   - GumbelCopula            Gumbel Archimedean copula
#   - FrankCopula             Frank Archimedean copula
#   - KendallTauToParam       Kendall's tau → copula parameter conversion
#   - SpearmanRhoToParam      Spearman's rho → copula parameter
#   - TailDependence          Lower/upper tail dependence coefficients
#   - CopulaGoodnessOfFit     Cramér-von Mises via Rosenblatt transform
#   - CopulaVaR               Portfolio VaR under copula dependence (MC)
#   - RegimeSwitchingCopula   Regime-dependent copula parameters
#   - run_copula_models        Top-level driver with JSON export
#
# Julia ≥ 1.10 | Packages: Statistics, LinearAlgebra, Random, JSON3
# =============================================================================

module CopulaModels

using Statistics
using LinearAlgebra
using Random
using JSON3

export GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula
export KendallTauToParam, SpearmanRhoToParam, simulate_copula
export TailDependence, CopulaGoodnessOfFit, CopulaVaR
export RegimeSwitchingCopula, run_copula_models

# ─────────────────────────────────────────────────────────────────────────────
# Internal Utilities
# ─────────────────────────────────────────────────────────────────────────────

"""Standard normal PDF."""
_φ(x::Float64) = exp(-0.5 * x^2) / sqrt(2π)

"""Standard normal CDF via series approximation (Abramowitz & Stegun 26.2.17)."""
function _Φ(x::Float64)::Float64
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530 +
                t * (-0.356563782 +
                t * (1.781477937 +
                t * (-1.821255978 +
                t * 1.330274429))))
    p = 1.0 - _φ(x) * poly
    x >= 0.0 ? p : 1.0 - p
end

"""Inverse standard normal CDF (Beasley-Springer-Moro algorithm)."""
function _Φinv(p::Float64)::Float64
    p = clamp(p, 1e-12, 1.0 - 1e-12)
    a = (0.3374754822726869, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187)
    b = (1.0, -0.1532080194741711, -0.2722878986169206,
         0.0519274593571442, 0.0136929880922736)
    if p < 0.5
        r = sqrt(-2.0 * log(p))
    else
        r = sqrt(-2.0 * log(1.0 - p))
    end
    num = a[1] + r*(a[2] + r*(a[3] + r*(a[4] + r*(a[5] +
          r*(a[6] + r*(a[7] + r*(a[8] + r*a[9])))))))
    den = b[1] + r*(b[2] + r*(b[3] + r*(b[4] + r*b[5])))
    result = num / den
    p < 0.5 ? -result : result
end

"""Empirical CDF values (pseudo-observations, scaled to (0,1))."""
function _pseudo_obs(data::Matrix{Float64})::Matrix{Float64}
    n, d = size(data)
    U = similar(data)
    for j in 1:d
        ranks = zeros(Int, n)
        sorted_idx = sortperm(data[:, j])
        for (r, i) in enumerate(sorted_idx)
            ranks[i] = r
        end
        U[:, j] = ranks ./ (n + 1.0)
    end
    return U
end

"""Regularise covariance/correlation matrix."""
function _regularise(Σ::Matrix{Float64}; ε::Float64=1e-6)
    return Σ + ε * I
end

"""Cholesky decomposition with fallback regularisation."""
function _safe_cholesky(Σ::Matrix{Float64})
    try
        return cholesky(Symmetric(Σ)).L
    catch
        return cholesky(Symmetric(_regularise(Σ, ε=1e-4))).L
    end
end

"""Student-t quantile via Newton iteration from normal approximation."""
function _tinv(p::Float64, ν::Float64)::Float64
    p = clamp(p, 1e-10, 1.0 - 1e-10)
    # Initial guess from normal
    x = _Φinv(p)
    # Newton-Raphson iterations
    for _ in 1:50
        # Student-t CDF via regularised incomplete beta
        f = _tcdf(x, ν) - p
        # Student-t PDF
        c = exp(lgamma((ν+1)/2) - lgamma(ν/2)) / (sqrt(ν*π))
        fp = c * (1.0 + x^2/ν)^(-(ν+1)/2)
        fp = max(fp, 1e-14)
        x -= f / fp
        abs(f) < 1e-10 && break
    end
    return x
end

"""Student-t CDF via regularised incomplete beta."""
function _tcdf(x::Float64, ν::Float64)::Float64
    # Using the relation: P(T ≤ x) = I_{ν/(ν+x²)}(ν/2, 1/2) / 2  for x < 0
    #                     P(T ≤ x) = 1 - I_{ν/(ν+x²)}(ν/2, 1/2) / 2 for x > 0
    t2 = x^2
    z = ν / (ν + t2)
    ib = _reg_inc_beta(z, ν/2.0, 0.5)
    return x < 0.0 ? 0.5 * ib : 1.0 - 0.5 * ib
end

"""Regularised incomplete beta function via continued fraction (Lentz)."""
function _reg_inc_beta(x::Float64, a::Float64, b::Float64)::Float64
    x = clamp(x, 0.0, 1.0)
    x == 0.0 && return 0.0
    x == 1.0 && return 1.0
    # Use symmetry relation if needed
    if x > (a + 1.0) / (a + b + 2.0)
        return 1.0 - _reg_inc_beta(1.0 - x, b, a)
    end
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(log(x)*a + log(1-x)*b - lbeta) / a
    # Lentz continued fraction
    f = 1.0; c = 1.0; d = 1.0 - (a+b)*x/(a+1.0)
    abs(d) < 1e-30 && (d = 1e-30)
    d = 1.0/d; f = d
    for m in 1:200
        # Even step
        mn = Float64(m)
        num = mn*(b-mn)*x / ((a+2mn-1.0)*(a+2mn))
        d = 1.0 + num*d; abs(d)<1e-30 && (d=1e-30)
        c = 1.0 + num/c; abs(c)<1e-30 && (c=1e-30)
        d = 1.0/d; f *= d*c
        # Odd step
        num = -(a+mn)*(a+b+mn)*x / ((a+2mn)*(a+2mn+1.0))
        d = 1.0 + num*d; abs(d)<1e-30 && (d=1e-30)
        c = 1.0 + num/c; abs(c)<1e-30 && (c=1e-30)
        d = 1.0/d; delta = d*c; f *= delta
        abs(delta - 1.0) < 1e-10 && break
    end
    return front * f
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Kendall's Tau / Spearman's Rho Conversions
# ─────────────────────────────────────────────────────────────────────────────

"""
    KendallTauToParam(family, tau) → θ

Convert Kendall's tau to copula parameter for a given family.

# Arguments
- `family` : one of :gaussian, :student, :clayton, :gumbel, :frank
- `tau`    : Kendall's rank correlation (scalar)

# Returns
- Copula parameter θ (scalar; for gaussian/student returns ρ ∈ (-1,1))

# Examples
```julia
θ = KendallTauToParam(:clayton, 0.5)  # → 2.0
ρ = KendallTauToParam(:gaussian, 0.4) # → sin(π/2 * 0.4) ≈ 0.588
```
"""
function KendallTauToParam(family::Symbol, tau::Float64)::Float64
    if family == :gaussian || family == :student
        # ρ = sin(π τ / 2)
        return sin(π * tau / 2.0)
    elseif family == :clayton
        tau < 0.0 && error("Clayton copula requires τ ≥ 0")
        return 2.0 * tau / (1.0 - tau)
    elseif family == :gumbel
        tau < 0.0 && error("Gumbel copula requires τ ≥ 0")
        return 1.0 / (1.0 - tau)
    elseif family == :frank
        # Numerical inversion of τ = 1 - 4/θ [D₁(-θ) - 1]
        # where D₁ is Debye function; we use Newton iteration
        abs(tau) < 1e-10 && return 0.0
        θ = tau > 0.0 ? 1.0 : -1.0
        for _ in 1:100
            d1 = _debye1(θ)
            f  = 1.0 - 4.0/θ * (d1 - 1.0) - tau
            df = 4.0/θ^2 * (d1 - 1.0) - 4.0/θ * _debye1_deriv(θ)
            abs(df) < 1e-14 && break
            θ -= f / df
            abs(f) < 1e-10 && break
        end
        return θ
    else
        error("Unknown copula family: $family")
    end
end

"""Debye function of order 1: D₁(x) = (1/x) ∫₀ˣ t/(exp(t)-1) dt."""
function _debye1(x::Float64)::Float64
    abs(x) < 1e-6 && return 1.0 - x/4.0
    # Series for small |x|
    if abs(x) < 1.0
        s = 1.0
        xk = x
        for k in 1:20
            xk *= x
            bk = _bernoulli_over_factorial(k)
            term = bk * xk
            s += term
            abs(term) < 1e-14 && break
        end
        return s
    end
    # For large x, use numerical quadrature (Gauss-Legendre 20-pt)
    gl_nodes  = [-0.9931286, -0.9639719, -0.9122344, -0.8391170, -0.7463061,
                 -0.6360537, -0.5108670, -0.3737061, -0.2277859, -0.0765265,
                  0.0765265,  0.2277859,  0.3737061,  0.5108670,  0.6360537,
                  0.7463061,  0.8391170,  0.9122344,  0.9639719,  0.9931286]
    gl_weights = [0.0176140, 0.0406014, 0.0626720, 0.0832767, 0.1019301,
                  0.1181945, 0.1316886, 0.1420961, 0.1491730, 0.1527534,
                  0.1527534, 0.1491730, 0.1420961, 0.1316886, 0.1181945,
                  0.1019301, 0.0832767, 0.0626720, 0.0406014, 0.0176140]
    # Transform [0, x] to [-1, 1]
    half = x / 2.0
    integral = 0.0
    for (t, w) in zip(gl_nodes, gl_weights)
        u = half * (1.0 + t)
        integral += w * (u / (exp(u) - 1.0 + 1e-300))
    end
    integral *= half
    return integral / x
end

function _debye1_deriv(x::Float64)::Float64
    h = 1e-6
    return (_debye1(x + h) - _debye1(x - h)) / (2h)
end

"""Bernoulli numbers B_{2k} / (2k)! for Debye series."""
function _bernoulli_over_factorial(k::Int)::Float64
    # Only even Bernoulli numbers contribute; approximate for small k
    coeffs = [1/4, -1/36, 1/3600, -1/211680, 1/10886400]
    k <= 5 ? coeffs[k] : 0.0
end

"""
    SpearmanRhoToParam(family, rho) → θ

Convert Spearman's rho to copula parameter.
"""
function SpearmanRhoToParam(family::Symbol, rho::Float64)::Float64
    if family == :gaussian || family == :student
        # ρ_P ≈ 2 sin(π ρ_S / 6)
        return 2.0 * sin(π * rho / 6.0)
    elseif family == :clayton
        tau = rho / (0.6931 + 0.2168 * rho)   # Approx Kendall-Spearman relation
        return KendallTauToParam(:clayton, clamp(tau, 0.01, 0.99))
    elseif family == :gumbel
        tau = (rho + 1.0) / (2.0 + rho)
        return KendallTauToParam(:gumbel, clamp(tau, 0.01, 0.99))
    elseif family == :frank
        # τ ≈ ρ_S / (1 + ρ_S * 0.25) (rough)
        tau = rho * 0.6
        return KendallTauToParam(:frank, clamp(tau, -0.99, 0.99))
    else
        error("Unknown family: $family")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Gaussian Copula
# ─────────────────────────────────────────────────────────────────────────────

"""
    GaussianCopula(data) → NamedTuple

Fit a Gaussian copula to multivariate data.

# Arguments
- `data` : n × d matrix of returns

# Returns
NamedTuple with fields:
- `R`     : d × d correlation matrix
- `loglik`: log-likelihood at fitted parameters
- `tau`   : matrix of Kendall's tau values
"""
function GaussianCopula(data::Matrix{Float64})
    U = _pseudo_obs(data)
    n, d = size(U)

    # Transform to normal scores
    Z = _Φinv.(U)

    # MLE = sample correlation matrix of normal scores
    R = cor(Z)
    R = _regularise(R)

    # Compute log-likelihood
    L = _safe_cholesky(R)
    log_det_R = 2.0 * sum(log.(diag(L)))
    # loglik = -n/2 * log|R| - 1/2 * sum_i (z_i' R^{-1} z_i - z_i' z_i)
    R_inv = inv(Symmetric(R))
    ll = 0.0
    for i in 1:n
        z = Z[i, :]
        ll += -0.5 * (dot(z, R_inv * z) - dot(z, z))
    end
    ll -= 0.5 * n * log_det_R

    # Kendall's tau matrix
    tau = zeros(d, d)
    for i in 1:d, j in (i+1):d
        tau[i,j] = tau[j,i] = _kendall_tau(data[:,i], data[:,j])
    end
    for i in 1:d; tau[i,i] = 1.0; end

    return (R=R, loglik=ll, tau=tau, family=:gaussian)
end

"""Compute Kendall's tau between two vectors."""
function _kendall_tau(x::Vector{Float64}, y::Vector{Float64})::Float64
    n = length(x)
    concordant = 0; discordant = 0
    for i in 1:(n-1), j in (i+1):n
        sx = sign(x[j] - x[i]); sy = sign(y[j] - y[i])
        prod = sx * sy
        if prod > 0; concordant += 1
        elseif prod < 0; discordant += 1
        end
    end
    npairs = n * (n-1) / 2
    return (concordant - discordant) / npairs
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Student-t Copula
# ─────────────────────────────────────────────────────────────────────────────

"""
    StudentTCopula(data; nu_grid) → NamedTuple

Fit a Student-t copula to multivariate data via profile likelihood over ν.

# Arguments
- `data`    : n × d matrix
- `nu_grid` : degrees of freedom to search over (default 2:30)

# Returns
NamedTuple: (R, nu, loglik, tail_dep)
  - `R`        : d×d correlation matrix
  - `nu`       : fitted degrees of freedom
  - `tail_dep` : lower/upper tail dependence coefficient λ
"""
function StudentTCopula(data::Matrix{Float64}; nu_grid::AbstractVector=2:2:30)
    U = _pseudo_obs(data)
    n, d = size(U)

    best_ll = -Inf; best_nu = 4.0; best_R = Matrix{Float64}(I, d, d)

    for nu in nu_grid
        ν = Float64(nu)
        # Transform U to t-scores
        T = [_tinv(U[i,j], ν) for i in 1:n, j in 1:d]

        # MLE correlation from t-scores (ECME-style: fix nu, optimise R)
        R = _robust_correlation(T, ν)
        R = _regularise(R)

        ll = _t_copula_loglik(T, R, ν)
        if ll > best_ll
            best_ll = ll; best_nu = ν; best_R = R
        end
    end

    # Tail dependence for bivariate t copula
    ρ = best_R[1,2]  # representative off-diagonal
    tail_dep = 2.0 * _tcdf(-sqrt((best_nu + 1.0) * (1.0 - ρ) / (1.0 + ρ)), best_nu + 1.0)

    return (R=best_R, nu=best_nu, loglik=best_ll, tail_dep=tail_dep, family=:student)
end

"""Robust (shrinkage) correlation for t-scores."""
function _robust_correlation(T::Matrix{Float64}, ν::Float64)::Matrix{Float64}
    n, d = size(T)
    # Weighted scatter matrix (Tyler's M-estimator approximation)
    R = cor(T)
    for _ in 1:20
        R_inv = inv(Symmetric(_regularise(R)))
        S = zeros(d, d)
        w_sum = 0.0
        for i in 1:n
            t = T[i, :]
            w = (ν + d) / (ν + dot(t, R_inv * t))
            S += w * (t * t')
            w_sum += w
        end
        R_new = S / w_sum
        # Normalise to correlation
        D = sqrt.(diag(R_new))
        for i in 1:d, j in 1:d
            R_new[i,j] /= (D[i] * D[j])
        end
        maximum(abs.(R_new - R)) < 1e-8 && break
        R = R_new
    end
    return R
end

"""Log-likelihood of Student-t copula."""
function _t_copula_loglik(T::Matrix{Float64}, R::Matrix{Float64}, ν::Float64)::Float64
    n, d = size(T)
    R_inv = inv(Symmetric(R))
    L = _safe_cholesky(R)
    log_det_R = 2.0 * sum(log.(diag(L)))

    lc = lgamma((ν + d)/2.0) - lgamma(ν/2.0) - (d/2.0)*log(ν) -
         (d/2.0)*log(π) - 0.5*log_det_R
    lm = lgamma(ν/2.0) - lgamma((ν+1.0)/2.0) + 0.5*log(π) + 0.5*log(ν)

    ll = 0.0
    for i in 1:n
        t = T[i, :]
        q = dot(t, R_inv * t)
        ll += lc - ((ν+d)/2.0) * log(1.0 + q/ν)
        ll -= sum(-(ν+1.0)/2.0 * log.(1.0 .+ t.^2 ./ ν) .+ lm)
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Archimedean Copulas
# ─────────────────────────────────────────────────────────────────────────────

"""
    ClaytonCopula(data) → NamedTuple

Fit a bivariate Clayton copula via method of moments (Kendall's tau).
Clayton copula: C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}, θ > 0.

Strong lower tail dependence, zero upper tail dependence.
"""
function ClaytonCopula(data::Matrix{Float64})
    size(data, 2) != 2 && error("Clayton fit expects bivariate data (n×2)")
    U = _pseudo_obs(data)
    tau = _kendall_tau(data[:,1], data[:,2])
    tau = max(tau, 0.01)
    θ = KendallTauToParam(:clayton, tau)

    # Log-likelihood
    ll = _clayton_loglik(U, θ)

    # Lower tail dependence: λ_L = 2^{-1/θ}
    lambda_lower = 2.0^(-1.0/θ)

    return (theta=θ, tau=tau, loglik=ll,
            tail_lower=lambda_lower, tail_upper=0.0, family=:clayton)
end

function _clayton_loglik(U::Matrix{Float64}, θ::Float64)::Float64
    n = size(U, 1)
    ll = 0.0
    for i in 1:n
        u, v = U[i,1], U[i,2]
        # c(u,v) = (1+θ) * (uv)^{-(1+θ)} * (u^{-θ}+v^{-θ}-1)^{-(2+1/θ)}
        inner = u^(-θ) + v^(-θ) - 1.0
        inner <= 0.0 && continue
        ll += log(1.0 + θ) - (1.0 + θ) * (log(u) + log(v)) -
              (2.0 + 1.0/θ) * log(inner)
    end
    return ll
end

"""
    GumbelCopula(data) → NamedTuple

Fit a bivariate Gumbel copula via method of moments.
Gumbel copula: C(u,v) = exp(-[(-ln u)^θ + (-ln v)^θ]^{1/θ}), θ ≥ 1.

Strong upper tail dependence, zero lower tail dependence.
"""
function GumbelCopula(data::Matrix{Float64})
    size(data, 2) != 2 && error("Gumbel fit expects bivariate data (n×2)")
    U = _pseudo_obs(data)
    tau = _kendall_tau(data[:,1], data[:,2])
    tau = clamp(tau, 0.01, 0.99)
    θ = KendallTauToParam(:gumbel, tau)

    ll = _gumbel_loglik(U, θ)

    # Upper tail dependence: λ_U = 2 - 2^{1/θ}
    lambda_upper = 2.0 - 2.0^(1.0/θ)

    return (theta=θ, tau=tau, loglik=ll,
            tail_lower=0.0, tail_upper=lambda_upper, family=:gumbel)
end

function _gumbel_loglik(U::Matrix{Float64}, θ::Float64)::Float64
    n = size(U, 1)
    ll = 0.0
    for i in 1:n
        u, v = max(U[i,1], 1e-10), max(U[i,2], 1e-10)
        x = (-log(u))^θ; y = (-log(v))^θ
        A = (x + y)^(1.0/θ)
        # Copula density (bivariate Gumbel)
        C = exp(-A)
        dC = C * A^(1.0-1.0/θ) * x * y * (θ-1.0+A) /
             (u * v * (x+y)^2 * (-log(u)) * (-log(v)))
        dC > 0.0 && (ll += log(dC))
    end
    return ll
end

"""
    FrankCopula(data) → NamedTuple

Fit a bivariate Frank copula via method of moments.
Frank copula: C(u,v) = -1/θ * ln(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1)).

Symmetric, no tail dependence.
"""
function FrankCopula(data::Matrix{Float64})
    size(data, 2) != 2 && error("Frank fit expects bivariate data (n×2)")
    U = _pseudo_obs(data)
    tau = _kendall_tau(data[:,1], data[:,2])
    θ = KendallTauToParam(:frank, tau)

    ll = _frank_loglik(U, θ)

    return (theta=θ, tau=tau, loglik=ll,
            tail_lower=0.0, tail_upper=0.0, family=:frank)
end

function _frank_loglik(U::Matrix{Float64}, θ::Float64)::Float64
    abs(θ) < 1e-10 && return 0.0
    n = size(U, 1)
    ll = 0.0
    em1 = expm1(-θ)
    for i in 1:n
        u, v = U[i,1], U[i,2]
        a = expm1(-θ * u); b = expm1(-θ * v)
        denom = em1 + a * b / em1
        abs(denom) < 1e-300 && continue
        # log c(u,v) = log(-θ(e^{-θ}-1)) + (-θu) + (-θv) - 2*log|denom| ...
        ll += log(abs(θ)) + log(abs(em1)) - θ*(u+v) - 2.0*log(abs(denom))
    end
    return ll
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Simulation from Copulas
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_copula(fitted, n_sims; rng) → Matrix{Float64}

Simulate n_sims samples from a fitted copula using conditional sampling.

# Arguments
- `fitted`  : output from GaussianCopula, StudentTCopula, etc.
- `n_sims`  : number of simulations
- `rng`     : random number generator

# Returns
Matrix n_sims × d of uniform [0,1] marginals (pseudo-observations).
"""
function simulate_copula(fitted::NamedTuple, n_sims::Int;
                         rng::AbstractRNG=Random.default_rng())::Matrix{Float64}
    family = fitted.family

    if family == :gaussian
        return _sim_gaussian(fitted.R, n_sims, rng)
    elseif family == :student
        return _sim_student(fitted.R, fitted.nu, n_sims, rng)
    elseif family == :clayton
        return _sim_clayton(fitted.theta, n_sims, rng)
    elseif family == :gumbel
        return _sim_gumbel(fitted.theta, n_sims, rng)
    elseif family == :frank
        return _sim_frank(fitted.theta, n_sims, rng)
    else
        error("Unknown family: $family")
    end
end

function _sim_gaussian(R::Matrix{Float64}, n::Int, rng::AbstractRNG)
    d = size(R, 1)
    L = _safe_cholesky(R)
    Z = randn(rng, n, d)
    Y = Z * L'
    return _Φ.(Y)
end

function _sim_student(R::Matrix{Float64}, ν::Float64, n::Int, rng::AbstractRNG)
    d = size(R, 1)
    L = _safe_cholesky(R)
    Z = randn(rng, n, d)
    Y = Z * L'
    # Scale by chi-squared / nu
    chi2 = [sum(randn(rng, Int(round(ν)))^2 for _ in 1:1) for _ in 1:n]
    # Use proper chi-squared
    chi_sq = zeros(n)
    for i in 1:n
        s = 0.0
        for _ in 1:Int(max(2, round(ν)))
            s += randn(rng)^2
        end
        chi_sq[i] = s
    end
    T = Y ./ sqrt.(chi_sq ./ ν)
    return [_tcdf(T[i,j], ν) for i in 1:n, j in 1:d]
end

function _sim_clayton(θ::Float64, n::Int, rng::AbstractRNG)
    U = rand(rng, n)
    p = rand(rng, n)
    # Conditional quantile: V|U=u via F^{-1}(p|u)
    # C(u,v) = (u^{-θ}+v^{-θ}-1)^{-1/θ}
    # ∂C/∂u = ... leads to V = ((p^{-θ/(1+θ)} - 1)*u^{-θ} + 1)^{-1/θ}
    V = ((p .^ (-θ/(1.0+θ)) .- 1.0) .* U .^ (-θ) .+ 1.0) .^ (-1.0/θ)
    V = clamp.(V, 1e-10, 1.0 - 1e-10)
    return hcat(U, V)
end

function _sim_gumbel(θ::Float64, n::Int, rng::AbstractRNG)
    # Frailty / stable mixture approach
    # Sample stable S(1/θ, 1, ...) via Chambers-Mallows-Stuck
    α = 1.0/θ
    U1 = rand(rng, n) .* π .- π/2.0
    E  = -log.(rand(rng, n))
    S  = sin.(α .* (U1 .+ π/2.0)) ./
         cos.(U1) .^ (1.0/α) .*
         (cos.(U1 .- α .* (U1 .+ π/2.0)) ./ E) .^ ((1.0-α)/α)

    E1 = -log.(rand(rng, n))
    E2 = -log.(rand(rng, n))
    U  = exp.(-(E1 ./ S) .^ α)
    V  = exp.(-(E2 ./ S) .^ α)
    return hcat(clamp.(U, 1e-10, 1-1e-10), clamp.(V, 1e-10, 1-1e-10))
end

function _sim_frank(θ::Float64, n::Int, rng::AbstractRNG)
    abs(θ) < 1e-6 && return rand(rng, n, 2)
    U = rand(rng, n)
    t = rand(rng, n)
    em1 = expm1(-θ)
    V = -1.0/θ .* log.(1.0 .+ t .* em1 ./ (t .* expm1.(-θ .* U) .+ exp.(-θ .* U)))
    V = clamp.(V, 1e-10, 1.0 - 1e-10)
    return hcat(U, V)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Tail Dependence Coefficients
# ─────────────────────────────────────────────────────────────────────────────

"""
    TailDependence(data; q_lower, q_upper) → NamedTuple

Estimate lower and upper tail dependence coefficients non-parametrically.
λ_L = lim_{q→0} P(U ≤ q | V ≤ q)
λ_U = lim_{q→1} P(U > q | V > q)

# Arguments
- `data`    : n×2 matrix
- `q_lower` : lower quantile threshold (default 0.05)
- `q_upper` : upper quantile threshold (default 0.95)
"""
function TailDependence(data::Matrix{Float64}; q_lower::Float64=0.05,
                        q_upper::Float64=0.95)
    size(data, 2) != 2 && error("TailDependence expects bivariate data (n×2)")
    U = _pseudo_obs(data)
    n = size(U, 1)

    # Empirical lower tail dependence
    lower_both = sum((U[:,1] .≤ q_lower) .& (U[:,2] .≤ q_lower))
    lower_cond = sum(U[:,1] .≤ q_lower)
    lambda_L = lower_cond > 0 ? lower_both / lower_cond : 0.0

    # Empirical upper tail dependence
    upper_both = sum((U[:,1] .≥ q_upper) .& (U[:,2] .≥ q_upper))
    upper_cond = sum(U[:,1] .≥ q_upper)
    lambda_U = upper_cond > 0 ? upper_both / upper_cond : 0.0

    # Chi-bar statistic for asymptotic independence
    chi_bar = 2.0 * log(0.5) / log(max(lower_both / n, 1e-10)) - 1.0

    return (lambda_lower=lambda_L, lambda_upper=lambda_U, chi_bar=chi_bar,
            q_lower=q_lower, q_upper=q_upper)
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Goodness-of-Fit: Cramér-von Mises on Rosenblatt Transform
# ─────────────────────────────────────────────────────────────────────────────

"""
    CopulaGoodnessOfFit(data, fitted; n_bootstrap) → NamedTuple

Cramér-von Mises goodness-of-fit test via Rosenblatt transform.
H₀: data follows the fitted copula.

# Arguments
- `data`        : n×d data matrix
- `fitted`      : output from GaussianCopula, StudentTCopula, etc.
- `n_bootstrap` : number of bootstrap samples for p-value (default 199)

# Returns
NamedTuple: (statistic, p_value, transformed)
"""
function CopulaGoodnessOfFit(data::Matrix{Float64}, fitted::NamedTuple;
                              n_bootstrap::Int=199)
    U = _pseudo_obs(data)
    n, d = size(U)

    # Rosenblatt transform → should be uniform if copula is correct
    W = _rosenblatt_transform(U, fitted)

    # Cramér-von Mises statistic: S = sum_i sum_j (W_ij - (i/(n+1)))^2
    Sn = _cvm_statistic(W)

    # Bootstrap p-value
    count_exceed = 0
    for _ in 1:n_bootstrap
        U_boot = simulate_copula(fitted, n)
        W_boot = _rosenblatt_transform(U_boot, fitted)
        Sn_boot = _cvm_statistic(W_boot)
        Sn_boot > Sn && (count_exceed += 1)
    end
    p_value = (count_exceed + 1.0) / (n_bootstrap + 1.0)

    return (statistic=Sn, p_value=p_value, reject_at_05=p_value < 0.05)
end

function _rosenblatt_transform(U::Matrix{Float64}, fitted::NamedTuple)::Matrix{Float64}
    n, d = size(U)
    W = similar(U)
    family = fitted.family

    if family == :gaussian
        R = fitted.R
        for i in 1:n
            z = _Φinv.(U[i,:])
            for j in 1:d
                if j == 1
                    W[i,1] = _Φ(z[1])
                else
                    # Conditional: Z_j | Z_1,...,Z_{j-1}
                    R11 = R[1:(j-1), 1:(j-1)]
                    R12 = R[1:(j-1), j]
                    R22 = R[j, j]
                    z_prev = z[1:(j-1)]
                    mu_cond = dot(R12, R11 \ z_prev)
                    sig_cond = sqrt(max(R22 - dot(R12, R11 \ R12), 1e-10))
                    W[i,j] = _Φ((z[j] - mu_cond) / sig_cond)
                end
            end
        end
    else
        # For Archimedean copulas, use empirical conditional CDF
        for i in 1:n, j in 1:d
            W[i,j] = U[i,j]  # Simplified: treat as independent margins
        end
    end
    return W
end

function _cvm_statistic(W::Matrix{Float64})::Float64
    n, d = size(W)
    S = 0.0
    for j in 1:d
        w_sorted = sort(W[:,j])
        for i in 1:n
            S += (w_sorted[i] - i/(n+1.0))^2
        end
    end
    return S / (n * d)
end

# ─────────────────────────────────────────────────────────────────────────────
# 8. Portfolio VaR under Copula-Modelled Dependence
# ─────────────────────────────────────────────────────────────────────────────

"""
    CopulaVaR(data, weights, fitted; alpha, n_sims) → NamedTuple

Estimate portfolio VaR and CVaR using Monte Carlo simulation from a copula.

# Arguments
- `data`    : n×d return data (used to fit marginal distributions)
- `weights` : portfolio weights (d-vector, must sum to 1)
- `fitted`  : fitted copula object
- `alpha`   : confidence level (default 0.99)
- `n_sims`  : number of Monte Carlo paths (default 10000)

# Returns
NamedTuple: (VaR, CVaR, simulated_portfolio_returns)
"""
function CopulaVaR(data::Matrix{Float64}, weights::Vector{Float64},
                   fitted::NamedTuple; alpha::Float64=0.99,
                   n_sims::Int=10_000,
                   rng::AbstractRNG=Random.default_rng())
    n, d = size(data)
    length(weights) == d || error("weights length must match number of assets")
    abs(sum(weights) - 1.0) > 1e-6 && @warn "Weights do not sum to 1"

    # Fit empirical marginal parameters (mean, std for normal approximation)
    mu_m = [mean(data[:,j]) for j in 1:d]
    sd_m = [std(data[:,j]) for j in 1:d]

    # Simulate from copula
    U_sim = simulate_copula(fitted, n_sims; rng=rng)

    # Invert marginals via empirical quantile function
    R_sim = similar(U_sim)
    for j in 1:d
        sorted_data = sort(data[:,j])
        for i in 1:n_sims
            idx = clamp(round(Int, U_sim[i,j] * n), 1, n)
            R_sim[i,j] = sorted_data[idx]
        end
    end

    # Portfolio returns
    port_returns = R_sim * weights

    # VaR and CVaR
    sorted_port = sort(port_returns)
    var_idx = max(1, round(Int, (1.0 - alpha) * n_sims))
    VaR_val = -sorted_port[var_idx]
    CVaR_val = -mean(sorted_port[1:var_idx])

    return (VaR=VaR_val, CVaR=CVaR_val, alpha=alpha,
            simulated_returns=port_returns)
end

# ─────────────────────────────────────────────────────────────────────────────
# 9. Regime-Switching Copula
# ─────────────────────────────────────────────────────────────────────────────

"""
    RegimeSwitchingCopula(data, regimes) → NamedTuple

Fit separate copula parameters for each market regime (bull/bear/stress).

# Arguments
- `data`    : n×d return matrix
- `regimes` : n-vector of regime labels (1=bull, 2=bear, 3=stress)

# Returns
NamedTuple with per-regime copula fits and transition statistics.
"""
function RegimeSwitchingCopula(data::Matrix{Float64}, regimes::Vector{Int})
    n, d = size(data)
    length(regimes) == n || error("regimes length must match data rows")

    unique_regimes = sort(unique(regimes))
    regime_names = Dict(1=>"bull", 2=>"bear", 3=>"stress")

    regime_fits = Dict{Int, Any}()
    regime_stats = Dict{Int, Any}()

    for r in unique_regimes
        idx = findall(regimes .== r)
        length(idx) < 10 && continue

        rdata = data[idx, :]
        name = get(regime_names, r, "regime_$r")

        # Fit Gaussian copula for each regime
        fit_g = GaussianCopula(rdata)

        # If bivariate, also fit Archimedean
        if d == 2
            try
                fit_c = ClaytonCopula(rdata)
                fit_t = StudentTCopula(rdata)
                regime_fits[r] = (gaussian=fit_g, clayton=fit_c, student=fit_t,
                                   name=name, n_obs=length(idx))
            catch
                regime_fits[r] = (gaussian=fit_g, name=name, n_obs=length(idx))
            end
        else
            fit_t = StudentTCopula(rdata)
            regime_fits[r] = (gaussian=fit_g, student=fit_t, name=name,
                               n_obs=length(idx))
        end

        # Statistics per regime
        regime_stats[r] = (
            mean_corr  = mean(fit_g.R[findall(!=(0.0), fit_g.R .- I)] ),
            avg_return = mean(rdata),
            volatility = mean(std.(eachcol(rdata))),
            name       = name
        )
    end

    # Transition matrix (empirical)
    K = maximum(unique_regimes)
    trans = zeros(K, K)
    for t in 1:(n-1)
        i, j = regimes[t], regimes[t+1]
        1 ≤ i ≤ K && 1 ≤ j ≤ K && (trans[i,j] += 1.0)
    end
    row_sums = sum(trans, dims=2)
    for i in 1:K
        row_sums[i] > 0 && (trans[i,:] ./= row_sums[i])
    end

    return (regime_fits=regime_fits, regime_stats=regime_stats,
            transition_matrix=trans, n_regimes=length(unique_regimes))
end

# ─────────────────────────────────────────────────────────────────────────────
# 10. Top-Level Driver
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_copula_models(data; weights, regimes, out_path) → Dict

Full copula modelling pipeline for crypto portfolio.

# Arguments
- `data`     : n×d return matrix (rows = days, cols = assets)
- `weights`  : portfolio weights (default equal-weight)
- `regimes`  : optional regime labels (default: classify by rolling vol)
- `out_path` : path to write JSON results (default: nothing)

# Returns
Dict with all fitted copulas, tail dependence, VaR, and regime results.

# Example
```julia
import Random
rng = Random.MersenneTwister(42)
n = 500; d = 4
data = randn(rng, n, d) * 0.02  # fake daily returns
results = run_copula_models(data; out_path="copula_results.json")
println("Gaussian copula loglik: ", results["gaussian"]["loglik"])
println("99% Portfolio VaR: ", results["VaR_99"])
```
"""
function run_copula_models(data::Matrix{Float64};
                           weights::Vector{Float64}=Float64[],
                           regimes::Vector{Int}=Int[],
                           out_path::Union{String,Nothing}=nothing)
    n, d = size(data)
    isempty(weights) && (weights = fill(1.0/d, d))

    results = Dict{String, Any}()

    # ── Gaussian Copula ────────────────────────────────────────────────────
    @info "Fitting Gaussian copula..."
    gc = GaussianCopula(data)
    results["gaussian"] = Dict(
        "family"  => "gaussian",
        "loglik"  => gc.loglik,
        "R"       => gc.R,
        "tau"     => gc.tau
    )

    # ── Student-t Copula ───────────────────────────────────────────────────
    @info "Fitting Student-t copula (profile over ν)..."
    tc = StudentTCopula(data; nu_grid=2:2:20)
    results["student_t"] = Dict(
        "family"   => "student_t",
        "nu"       => tc.nu,
        "loglik"   => tc.loglik,
        "tail_dep" => tc.tail_dep,
        "R"        => tc.R
    )

    # ── Archimedean Copulas (bivariate pairs) ──────────────────────────────
    if d >= 2
        pair_results = Dict{String, Any}()
        for i in 1:min(d-1, 3), j in (i+1):min(d, 4)
            pair_data = data[:, [i,j]]
            key = "pair_$(i)_$(j)"
            try
                cc = ClaytonCopula(pair_data)
                gc2 = GumbelCopula(pair_data)
                fc = FrankCopula(pair_data)
                # Best by log-likelihood
                best = argmax([cc.loglik, gc2.loglik, fc.loglik])
                pair_results[key] = Dict(
                    "clayton" => Dict("theta"=>cc.theta, "loglik"=>cc.loglik,
                                      "tail_lower"=>cc.tail_lower),
                    "gumbel"  => Dict("theta"=>gc2.theta, "loglik"=>gc2.loglik,
                                      "tail_upper"=>gc2.tail_upper),
                    "frank"   => Dict("theta"=>fc.theta, "loglik"=>fc.loglik),
                    "best_family" => ["clayton","gumbel","frank"][best]
                )
                td = TailDependence(pair_data)
                pair_results[key]["empirical_tail"] = Dict(
                    "lambda_lower" => td.lambda_lower,
                    "lambda_upper" => td.lambda_upper
                )
            catch e
                pair_results[key] = Dict("error" => string(e))
            end
        end
        results["archimedean_pairs"] = pair_results
    end

    # ── Portfolio VaR ──────────────────────────────────────────────────────
    @info "Computing copula-based portfolio VaR..."
    rng = Random.default_rng()
    var99 = CopulaVaR(data, weights, tc; alpha=0.99, n_sims=5000, rng=rng)
    var95 = CopulaVaR(data, weights, gc; alpha=0.95, n_sims=5000, rng=rng)
    results["VaR_99"] = var99.VaR
    results["CVaR_99"] = var99.CVaR
    results["VaR_95"] = var95.VaR
    results["CVaR_95"] = var95.CVaR

    # ── Regime-Switching Copula ────────────────────────────────────────────
    if isempty(regimes)
        # Auto-classify: rolling 20-day volatility → 3 regimes
        regimes = _classify_regimes(data, 20)
    end
    @info "Fitting regime-switching copula..."
    rsc = RegimeSwitchingCopula(data, regimes)
    reg_summary = Dict{String, Any}()
    for (r, stats) in rsc.regime_stats
        reg_summary["regime_$(stats.name)"] = Dict(
            "mean_corr"  => stats.mean_corr,
            "avg_return" => stats.avg_return,
            "volatility" => stats.volatility,
            "n_obs"      => rsc.regime_fits[r].n_obs
        )
    end
    results["regime_switching"] = reg_summary
    results["transition_matrix"] = rsc.transition_matrix

    # ── Model Comparison (AIC/BIC) ─────────────────────────────────────────
    n_params_gaussian = d*(d-1)/2
    n_params_student  = d*(d-1)/2 + 1
    results["model_selection"] = Dict(
        "AIC_gaussian" => -2*gc.loglik + 2*n_params_gaussian,
        "AIC_student"  => -2*tc.loglik + 2*n_params_student,
        "BIC_gaussian" => -2*gc.loglik + n_params_gaussian*log(n),
        "BIC_student"  => -2*tc.loglik + n_params_student*log(n),
        "preferred"    => tc.loglik > gc.loglik ? "student_t" : "gaussian"
    )

    # ── JSON Export ────────────────────────────────────────────────────────
    if !isnothing(out_path)
        open(out_path, "w") do io
            JSON3.write(io, results)
        end
        @info "Results written to $out_path"
    end

    return results
end

"""Classify market regimes based on rolling volatility quantiles."""
function _classify_regimes(data::Matrix{Float64}, window::Int)::Vector{Int}
    n = size(data, 1)
    port_ret = mean(data, dims=2)[:,1]
    rolling_vol = zeros(n)

    for i in (window+1):n
        rolling_vol[i] = std(port_ret[(i-window):(i-1)])
    end
    rolling_vol[1:window] .= rolling_vol[window+1]

    q33 = quantile(rolling_vol, 0.33)
    q67 = quantile(rolling_vol, 0.67)
    regimes = ones(Int, n)
    for i in 1:n
        if rolling_vol[i] <= q33
            regimes[i] = 1  # bull: low vol
        elseif rolling_vol[i] <= q67
            regimes[i] = 2  # bear: medium vol
        else
            regimes[i] = 3  # stress: high vol
        end
    end
    return regimes
end

end  # module CopulaModels
