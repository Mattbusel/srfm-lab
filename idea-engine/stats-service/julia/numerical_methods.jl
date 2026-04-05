module NumericalMethods

# ============================================================
# numerical_methods.jl -- Numerical Methods for Quantitative Finance
# ============================================================
# Covers: Monte Carlo variance reduction (antithetic, control
# variates, quasi-MC Halton/Sobol), PDE solvers (Crank-Nicolson,
# explicit, implicit), numerical integration (Gauss-Legendre,
# adaptive Simpson, Romberg), root-finding (Newton-Raphson, Brent),
# finite difference Greeks, Richardson extrapolation, Cholesky
# decomposition, correlated random variable generation.
# Pure stdlib Julia -- no external packages.
# ============================================================

using Statistics, LinearAlgebra

struct MCResult
    price::Float64
    std_error::Float64
    ci_lo::Float64
    ci_hi::Float64
    n_paths::Int
    method::Symbol
end

struct PDEGrid
    S_min::Float64; S_max::Float64; n_S::Int
    T_expiry::Float64; n_t::Int
    dS::Float64; dt::Float64
    grid::Matrix{Float64}
end

struct GaussLegendreRule
    nodes::Vector{Float64}
    weights::Vector{Float64}
    n_pts::Int
end

struct FDGreeks
    delta::Float64; gamma::Float64; theta::Float64; vega::Float64; rho::Float64
end

# ---- 1. Low-Discrepancy Sequences ----

function halton(n::Int, base::Int)::Vector{Float64}
    seq = zeros(n)
    for i in 1:n
        f = 1.0; r = 0.0; k = i
        while k > 0
            f /= base; r += f * (k % base); k = k div base
        end
        seq[i] = r
    end
    return seq
end

function halton_2d(n::Int)::Matrix{Float64}
    return hcat(halton(n,2), halton(n,3))
end

function sobol_1d(n::Int)::Vector{Float64}
    seq = zeros(n); x = UInt32(0)
    v = [UInt32(1) << UInt32(31-k) for k in 0:30]
    for i in 1:n
        c = trailing_zeros(UInt32(i-1)) + 1
        x = xor(x, v[min(c, length(v))])
        seq[i] = Float64(x) / Float64(2^32)
    end
    return seq
end

function van_der_corput(n::Int, base::Int=2)::Vector{Float64}
    return halton(n, base)
end

# ---- 2. Normal Sampling ----

function lcg_normals(n::Int, seed::UInt64=UInt64(42))::Vector{Float64}
    res = zeros(n); state = seed; i = 1
    while i <= n
        state = 6364136223846793005*state + 1442695040888963407
        u1 = max(Float64(state>>33)/Float64(2^31), 1e-15)
        state = 6364136223846793005*state + 1442695040888963407
        u2 = max(Float64(state>>33)/Float64(2^31), 1e-15)
        z1 = sqrt(-2*log(u1))*cos(2pi*u2); z2 = sqrt(-2*log(u1))*sin(2pi*u2)
        res[i] = z1; i += 1
        if i <= n; res[i] = z2; i += 1; end
    end
    return res
end

function normal_quantile(p::Float64)::Float64
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_lo = 0.02425; p_hi = 1 - p_lo
    if p < p_lo
        q = sqrt(-2*log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= p_hi
        q = p-0.5; r = q^2
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2*log(1-p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

# ---- 3. Monte Carlo Pricing ----

function mc_european_call(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                           T::Float64, n_paths::Int; method::Symbol=:standard,
                           seed::Int=42)::MCResult
    payoffs = Float64[]
    if method == :antithetic
        z = lcg_normals(n_paths div 2, UInt64(seed))
        for zi in z
            for sg in (1.0, -1.0)
                ST = S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*sg*zi)
                push!(payoffs, max(ST-K,0.0))
            end
        end
    elseif method == :quasi
        u = halton(n_paths, 2)
        for ui in u
            z = normal_quantile(clamp(ui, 1e-10, 1-1e-10))
            ST = S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*z)
            push!(payoffs, max(ST-K,0.0))
        end
    else
        z = lcg_normals(n_paths, UInt64(seed))
        for zi in z
            ST = S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*zi)
            push!(payoffs, max(ST-K,0.0))
        end
    end
    if method == :control_variate
        z = lcg_normals(n_paths, UInt64(seed+1))
        payoffs = Float64[]; st_vals = Float64[]
        for zi in z
            ST = S0*exp((r-0.5*sigma^2)*T + sigma*sqrt(T)*zi)
            push!(payoffs, max(ST-K,0.0)); push!(st_vals, ST)
        end
        mu_ctrl = S0*exp(r*T)
        beta = cov(payoffs, st_vals) / (var(st_vals)+1e-12)
        payoffs = payoffs .- beta.*(st_vals.-mu_ctrl)
    end
    disc = payoffs .* exp(-r*T)
    pr = mean(disc); se = std(disc)/sqrt(length(disc))
    return MCResult(pr, se, pr-1.96*se, pr+1.96*se, length(disc), method)
end

function mc_asian_call(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                        T::Float64, n_steps::Int, n_paths::Int; seed::Int=42)::MCResult
    dt = T/n_steps; payoffs = Float64[]; rng = UInt64(seed)
    for _ in 1:n_paths
        path = zeros(n_steps+1); path[1] = S0
        for j in 2:n_steps+1
            rng = 6364136223846793005*rng + 1442695040888963407
            u1 = max(Float64(rng>>33)/Float64(2^31), 1e-15)
            rng = 6364136223846793005*rng + 1442695040888963407
            u2 = max(Float64(rng>>33)/Float64(2^31), 1e-15)
            z = sqrt(-2*log(u1))*cos(2pi*u2)
            path[j] = path[j-1]*exp((r-0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
        end
        push!(payoffs, exp(-r*T)*max(mean(path)-K, 0.0))
    end
    pr = mean(payoffs); se = std(payoffs)/sqrt(n_paths)
    return MCResult(pr, se, pr-1.96*se, pr+1.96*se, n_paths, :asian)
end

function mc_barrier_call(S0::Float64, K::Float64, B::Float64, r::Float64,
                          sigma::Float64, T::Float64, n_steps::Int,
                          n_paths::Int; seed::Int=42)::MCResult
    dt = T/n_steps; payoffs = Float64[]; rng = UInt64(seed)
    for _ in 1:n_paths
        S = S0; hit_barrier = false
        for j in 1:n_steps
            rng = 6364136223846793005*rng + 1442695040888963407
            u1 = max(Float64(rng>>33)/Float64(2^31), 1e-15)
            rng = 6364136223846793005*rng + 1442695040888963407
            u2 = max(Float64(rng>>33)/Float64(2^31), 1e-15)
            z = sqrt(-2*log(u1))*cos(2pi*u2)
            S = S*exp((r-0.5*sigma^2)*dt + sigma*sqrt(dt)*z)
            if S >= B; hit_barrier = true; break; end
        end
        pv = hit_barrier ? 0.0 : exp(-r*T)*max(S-K, 0.0)
        push!(payoffs, pv)
    end
    pr = mean(payoffs); se = std(payoffs)/sqrt(n_paths)
    return MCResult(pr, se, pr-1.96*se, pr+1.96*se, n_paths, :barrier)
end

# ---- 4. Crank-Nicolson PDE Solver ----

function thomas_algorithm(a::Vector{Float64}, b::Vector{Float64},
                           c::Vector{Float64}, d::Vector{Float64})::Vector{Float64}
    n = length(d); cp = zeros(n); dp = zeros(n); x = zeros(n)
    cp[1] = c[1]/b[1]; dp[1] = d[1]/b[1]
    for i in 2:n
        denom = b[i] - a[i]*cp[i-1]
        cp[i] = i < n ? c[i]/denom : 0.0
        dp[i] = (d[i] - a[i]*dp[i-1]) / denom
    end
    x[n] = dp[n]
    for i in (n-1):-1:1; x[i] = dp[i] - cp[i]*x[i+1]; end
    return x
end

function crank_nicolson_bs(S_max::Float64, K::Float64, r::Float64, sigma::Float64,
                             T::Float64, n_S::Int=100, n_t::Int=100)::PDEGrid
    dS = S_max/n_S; dt = T/n_t
    S = [i*dS for i in 0:n_S]
    grid = zeros(n_S+1, n_t+1)
    for i in 1:n_S+1; grid[i, n_t+1] = max(S[i]-K, 0.0); end
    aa = zeros(n_S+1); bb = zeros(n_S+1); cc = zeros(n_S+1)
    for i in 2:n_S
        s = S[i]
        aa[i] = 0.25*dt*(sigma^2*s^2/dS^2 - r*s/dS)
        bb[i] = -0.5*dt*(sigma^2*s^2/dS^2 + r)
        cc[i] = 0.25*dt*(sigma^2*s^2/dS^2 + r*s/dS)
    end
    for j in n_t:-1:1
        rhs = zeros(n_S+1)
        for i in 2:n_S
            rhs[i] = -aa[i]*grid[i-1,j+1] + (1-bb[i])*grid[i,j+1] - cc[i]*grid[i+1,j+1]
        end
        grid[1,j] = 0.0
        t_c = (n_t-j)*dt
        grid[n_S+1,j] = max(S_max - K*exp(-r*(T-t_c)), 0.0)
        rhs[2] -= aa[2]*grid[1,j]
        rhs[n_S] -= cc[n_S]*grid[n_S+1,j]
        sol = thomas_algorithm(aa[2:n_S], [1-bb[i] for i in 2:n_S], cc[2:n_S], rhs[2:n_S])
        for i in 2:n_S; grid[i,j] = sol[i-1]; end
    end
    return PDEGrid(0.0, S_max, n_S, T, n_t, dS, dt, grid)
end

function pde_price(g::PDEGrid, S::Float64)::Float64
    i = clamp(floor(Int, S/g.dS)+1, 1, g.n_S)
    i1 = min(i+1, g.n_S+1)
    frac = (S - (i-1)*g.dS) / g.dS
    return g.grid[i,1]*(1-frac) + g.grid[i1,1]*frac
end

function explicit_bs(S_max::Float64, K::Float64, r::Float64, sigma::Float64,
                      T::Float64, n_S::Int=100, n_t::Int=1000)::PDEGrid
    dS = S_max/n_S; dt = T/n_t; S = [i*dS for i in 0:n_S]
    grid = zeros(n_S+1, n_t+1)
    for i in 1:n_S+1; grid[i, n_t+1] = max(S[i]-K, 0.0); end
    for j in n_t:-1:1
        t_c = (n_t-j)*dt
        grid[1,j] = 0.0
        grid[n_S+1,j] = max(S_max - K*exp(-r*(T-t_c)), 0.0)
        for i in 2:n_S
            s = S[i]
            pu = 0.5*dt*(sigma^2*s^2/dS^2 + r*s/dS)
            pm = 1.0 - dt*(sigma^2*s^2/dS^2 + r)
            pd = 0.5*dt*(sigma^2*s^2/dS^2 - r*s/dS)
            grid[i,j] = pu*grid[i+1,j+1] + pm*grid[i,j+1] + pd*grid[i-1,j+1]
        end
    end
    return PDEGrid(0.0, S_max, n_S, T, n_t, dS, dt, grid)
end

# ---- 5. Gauss-Legendre Quadrature ----

function gauss_legendre_5()::GaussLegendreRule
    nodes   = [-0.9061798459386640, -0.5384693101056831, 0.0,
                0.5384693101056831,  0.9061798459386640]
    weights = [ 0.2369268850561891,  0.4786286704993665, 0.5688888888888889,
                0.4786286704993665,  0.2369268850561891]
    return GaussLegendreRule(nodes, weights, 5)
end

function gl_integrate(f::Function, a::Float64, b::Float64,
                       rule::GaussLegendreRule)::Float64
    mid = 0.5*(b+a); hl = 0.5*(b-a)
    return hl * sum(rule.weights[i]*f(mid + hl*rule.nodes[i]) for i in 1:rule.n_pts)
end

function adaptive_simpson(f::Function, a::Float64, b::Float64,
                           tol::Float64=1e-8, max_depth::Int=20)::Float64
    simp(l,r) = (r-l)/6*(f(l)+4*f((l+r)/2)+f(r))
    function recurse(l, r, tol_r, whole, depth)
        m = (l+r)/2
        ls = simp(l,m); rs = simp(m,r)
        if depth >= max_depth || abs(ls+rs-whole) <= 15*tol_r
            return ls+rs+(ls+rs-whole)/15
        end
        return recurse(l,m,tol_r/2,ls,depth+1) + recurse(m,r,tol_r/2,rs,depth+1)
    end
    return recurse(a, b, tol, simp(a,b), 0)
end

function romberg_integrate(f::Function, a::Float64, b::Float64, max_level::Int=8)::Float64
    R = zeros(max_level, max_level); h = b - a
    R[1,1] = (f(a)+f(b))*h/2
    for i in 2:max_level
        h /= 2; n_pts = 2^(i-2)
        R[i,1] = R[i-1,1]/2 + h*sum(f(a+(2k-1)*h) for k in 1:n_pts)
        for j in 2:i
            R[i,j] = R[i,j-1] + (R[i,j-1]-R[i-1,j-1])/(4^(j-1)-1)
        end
        if i > 2 && abs(R[i,i]-R[i-1,i-1]) < 1e-10; return R[i,i]; end
    end
    return R[max_level, max_level]
end

# ---- 6. Root Finding ----

function newton_raphson(f::Function, df::Function, x0::Float64,
                         tol::Float64=1e-10, max_iter::Int=100)::Float64
    x = x0
    for _ in 1:max_iter
        fx = f(x); dfx = df(x); if abs(dfx) < 1e-14; break; end
        xn = x - fx/dfx; if abs(xn-x) < tol; return xn; end; x = xn
    end
    return x
end

function brent_method(f::Function, a::Float64, b::Float64,
                       tol::Float64=1e-10, max_iter::Int=100)::Float64
    fa = f(a); fb = f(b)
    fa*fb > 0 && error("f(a) and f(b) must have opposite signs")
    c = a; fc = fa; s = b; d = b; mflag = true
    for _ in 1:max_iter
        if fb == 0.0; return b; end
        if abs(b-a) < tol; return (a+b)/2; end
        if fa != fc && fb != fc
            s = a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) + c*fa*fb/((fc-fa)*(fc-fb))
        else
            s = b - fb*(b-a)/(fb-fa+1e-300)
        end
        cond1 = !((3a+b)/4 < s < b || b < s < (3a+b)/4)
        cond2 = mflag && abs(s-b) >= abs(b-c)/2
        cond3 = !mflag && abs(s-b) >= abs(c-d)/2
        if cond1||cond2||cond3; s=(a+b)/2; mflag=true; else; mflag=false; end
        fs=f(s); d=c; c=b; fc=fb
        if fa*fs < 0; b=s; fb=fs; else; a=s; fa=fs; end
        if abs(fa) < abs(fb); a,b=b,a; fa,fb=fb,fa; end
    end
    return b
end

function implied_vol(mkt_price::Float64, S::Float64, K::Float64, r::Float64,
                      T::Float64, opt_type::Symbol=:call; seed_vol::Float64=0.25)::Float64
    function bs_price(sigma)
        d1 = (log(S/K)+(r+0.5*sigma^2)*T)/(sigma*sqrt(T)+1e-12)
        d2 = d1 - sigma*sqrt(T)
        nd1 = 0.5*(1+erf(d1/sqrt(2))); nd2 = 0.5*(1+erf(d2/sqrt(2)))
        return opt_type==:call ? S*nd1-K*exp(-r*T)*nd2 : K*exp(-r*T)*(1-nd2)-S*(1-nd1)
    end
    function bs_vega(sigma)
        d1 = (log(S/K)+(r+0.5*sigma^2)*T)/(sigma*sqrt(T)+1e-12)
        return S*sqrt(T)*exp(-0.5*d1^2)/sqrt(2pi)
    end
    clamp(newton_raphson(s->bs_price(s)-mkt_price, bs_vega, seed_vol, 1e-8, 50), 0.001, 5.0)
end

# ---- 7. Richardson Extrapolation ----

function richardson_extrap(f::Function, x::Float64, h::Float64, order::Int=2)::Float64
    d1 = (f(x+h)-f(x-h))/(2h); d2 = (f(x+h/2)-f(x-h/2))/h
    return (2^order*d2 - d1)/(2^order - 1)
end

# ---- 8. Finite Difference Greeks ----

function fd_greeks(price_fn::Function, S::Float64, K::Float64, r::Float64,
                   sigma::Float64, T::Float64)::FDGreeks
    dS = S*0.01; dsig = 0.001; dT = 1/365; dr = 0.001
    p0 = price_fn(S,K,r,sigma,T)
    pu = price_fn(S+dS,K,r,sigma,T); pd = price_fn(S-dS,K,r,sigma,T)
    psu = price_fn(S,K,r,sigma+dsig,T); psd = price_fn(S,K,r,sigma-dsig,T)
    ptu = price_fn(S,K,r,sigma,T+dT); ptd = price_fn(S,K,r,sigma,max(T-dT,1e-6))
    pru = price_fn(S,K,r+dr,sigma,T); prd = price_fn(S,K,r-dr,sigma,T)
    return FDGreeks(
        (pu-pd)/(2*dS),
        (pu-2p0+pd)/dS^2,
        (ptd-ptu)/(2*dT),
        (psu-psd)/(2*dsig),
        (pru-prd)/(2*dr)
    )
end

# ---- 9. Matrix Utilities ----

function cholesky_lower(A::Matrix{Float64})::Matrix{Float64}
    n = size(A,1); L = zeros(n,n)
    for i in 1:n, j in 1:i
        s = sum(L[i,k]*L[j,k] for k in 1:(j-1); init=0.0)
        L[i,j] = i==j ? sqrt(max(A[i,i]-s,0.0)) :
                         (L[j,j]>1e-12 ? (A[i,j]-s)/L[j,j] : 0.0)
    end
    return L
end

function correlated_normals(corr::Matrix{Float64}, n_samples::Int; seed::Int=42)::Matrix{Float64}
    n = size(corr,1); L = cholesky_lower(corr)
    indep = reshape(lcg_normals(n*n_samples, UInt64(seed)), n_samples, n)
    return indep * L'
end

function cholesky_solve(A::Matrix{Float64}, b::Vector{Float64})::Vector{Float64}
    L = cholesky_lower(A)
    n = length(b)
    y = zeros(n); x = zeros(n)
    for i in 1:n
        y[i] = (b[i] - sum(L[i,k]*y[k] for k in 1:(i-1); init=0.0)) / (L[i,i]+1e-12)
    end
    for i in n:-1:1
        x[i] = (y[i] - sum(L[j,i]*x[j] for j in (i+1):n; init=0.0)) / (L[i,i]+1e-12)
    end
    return x
end

# ---- Demo ----

function demo()
    println("=== NumericalMethods Demo ===")
    h2d = halton_2d(4)
    println("Halton 2D (4pts):")
    for i in 1:4; println("  ", round.(h2d[i,:], digits=4)); end

    S0=100.0; K=100.0; r=0.05; sig=0.2; T=1.0
    mcs = mc_european_call(S0,K,r,sig,T,20000; method=:standard)
    mca = mc_european_call(S0,K,r,sig,T,20000; method=:antithetic)
    mcc = mc_european_call(S0,K,r,sig,T,20000; method=:control_variate)
    mcq = mc_european_call(S0,K,r,sig,T,20000; method=:quasi)
    println("\nEuropean call (S=K=100, r=5%, vol=20%, T=1y):")
    println("  Standard:        ", round(mcs.price,digits=4), " +/- ", round(mcs.std_error,digits=4))
    println("  Antithetic:      ", round(mca.price,digits=4), " +/- ", round(mca.std_error,digits=4))
    println("  Control variate: ", round(mcc.price,digits=4), " +/- ", round(mcc.std_error,digits=4))
    println("  Quasi-MC:        ", round(mcq.price,digits=4), " +/- ", round(mcq.std_error,digits=4))

    pde = crank_nicolson_bs(200.0, 100.0, 0.05, 0.2, 1.0, 100, 100)
    println("\nCrank-Nicolson PDE at S=100: ", round(pde_price(pde, 100.0), digits=4))

    iv_val = implied_vol(10.45, 100.0, 100.0, 0.05, 1.0, :call)
    println("Implied vol for price=10.45: ", round(iv_val, digits=4))

    I = romberg_integrate(x -> exp(-x^2), 0.0, 3.0)
    println("Romberg integral exp(-x^2) [0,3]: ", round(I, digits=6))

    corr = [1.0 0.7 0.3; 0.7 1.0 0.5; 0.3 0.5 1.0]
    samps = correlated_normals(corr, 1000)
    emp_corr = cor(samps)
    println("\nCorr check [1,2]: target=0.7 got=", round(emp_corr[1,2], digits=3))

    function bs_call(S,K,r,s,T)
        d1=(log(S/K)+(r+0.5*s^2)*T)/(s*sqrt(T)+1e-12); d2=d1-s*sqrt(T)
        return S*0.5*(1+erf(d1/sqrt(2)))-K*exp(-r*T)*0.5*(1+erf(d2/sqrt(2)))
    end
    greeks = fd_greeks((S,K,r,s,T)->bs_call(S,K,r,s,T), 100.0, 100.0, 0.05, 0.2, 1.0)
    println("\nFD Greeks (ATM, 1y): delta=", round(greeks.delta,digits=4),
            " gamma=", round(greeks.gamma,digits=4), " vega=", round(greeks.vega,digits=4))
end

# ---- Additional Numerical Methods ----

function secant_method(f::Function, x0::Float64, x1::Float64,
                        tol::Float64=1e-10, max_iter::Int=100)::Float64
    for _ in 1:max_iter
        fx0 = f(x0); fx1 = f(x1)
        if abs(fx1 - fx0) < 1e-15; break; end
        x2 = x1 - fx1*(x1 - x0)/(fx1 - fx0)
        if abs(x2 - x1) < tol; return x2; end
        x0 = x1; x1 = x2
    end
    return x1
end

function fixed_point_iteration(g::Function, x0::Float64,
                                 tol::Float64=1e-10, max_iter::Int=200)::Float64
    x = x0
    for _ in 1:max_iter
        xn = g(x); if abs(xn - x) < tol; return xn; end; x = xn
    end
    return x
end

function bisection_method(f::Function, a::Float64, b::Float64,
                            tol::Float64=1e-10)::Float64
    fa = f(a); fb = f(b)
    fa*fb > 0 && error("Root not bracketed")
    for _ in 1:200
        m = (a + b)/2; fm = f(m)
        if abs(fm) < tol || (b-a)/2 < tol; return m; end
        if fa*fm < 0; b=m; fb=fm; else; a=m; fa=fm; end
    end
    return (a+b)/2
end

function euler_ode(f::Function, y0::Float64, t0::Float64, t1::Float64,
                    n_steps::Int)::Tuple{Vector{Float64}, Vector{Float64}}
    dt = (t1-t0)/n_steps; t = t0; y = y0
    ts = [t]; ys = [y]
    for _ in 1:n_steps
        y += dt*f(t, y); t += dt
        push!(ts, t); push!(ys, y)
    end
    return ts, ys
end

function runge_kutta_4(f::Function, y0::Float64, t0::Float64, t1::Float64,
                         n_steps::Int)::Tuple{Vector{Float64}, Vector{Float64}}
    dt = (t1-t0)/n_steps; t = t0; y = y0
    ts = [t]; ys = [y]
    for _ in 1:n_steps
        k1 = f(t, y); k2 = f(t+dt/2, y+dt*k1/2)
        k3 = f(t+dt/2, y+dt*k2/2); k4 = f(t+dt, y+dt*k3)
        y += dt*(k1 + 2k2 + 2k3 + k4)/6; t += dt
        push!(ts, t); push!(ys, y)
    end
    return ts, ys
end

function gauss_hermite_quadrature(f::Function, mu::Float64=0.0,
                                    sigma::Float64=1.0, n::Int=5)::Float64
    nodes_5 = [-2.020182, -0.958572, 0.0, 0.958572, 2.020182]
    weights_5 = [0.019953, 0.394424, 0.945309, 0.394424, 0.019953]
    nodes   = n==5 ? nodes_5 : nodes_5[1:min(n,5)]
    weights = n==5 ? weights_5 : weights_5[1:min(n,5)]
    x_transformed = mu .+ sigma .* sqrt(2) .* nodes
    return sum(weights .* f.(x_transformed)) / sqrt(pi)
end

function trapezoidal_rule(f::Function, a::Float64, b::Float64, n::Int=100)::Float64
    h = (b-a)/n; x = range(a, b, length=n+1)
    vals = f.(x); return h*(vals[1]/2 + sum(vals[2:end-1]) + vals[end]/2)
end

function finite_difference_2d(V::Matrix{Float64}, dx::Float64, dy::Float64)
    m, n = size(V)
    dVdx = zeros(m, n); dVdy = zeros(m, n)
    dVdx[2:end-1,:] = (V[3:end,:] - V[1:end-2,:]) ./ (2dx)
    dVdy[:,2:end-1] = (V[:,3:end] - V[:,1:end-2]) ./ (2dy)
    return dVdx, dVdy
end

function sparse_gauss_seidel(A::Matrix{Float64}, b::Vector{Float64},
                               tol::Float64=1e-8, max_iter::Int=1000)::Vector{Float64}
    n = length(b); x = zeros(n)
    for _ in 1:max_iter
        x_old = copy(x)
        for i in 1:n
            sigma = sum(A[i,j]*x[j] for j in 1:n if j != i; init=0.0)
            x[i] = (b[i] - sigma) / (A[i,i] + 1e-12)
        end
        if maximum(abs.(x - x_old)) < tol; break; end
    end
    return x
end

function antithetic_variate_correction(payoffs_pos::Vector{Float64},
                                         payoffs_neg::Vector{Float64})::Float64
    n = min(length(payoffs_pos), length(payoffs_neg))
    combined = (payoffs_pos[1:n] .+ payoffs_neg[1:n]) ./ 2
    return mean(combined)
end

function control_variate_estimator(target_payoffs::Vector{Float64},
                                    control_payoffs::Vector{Float64},
                                    control_mean::Float64)::Float64
    n = length(target_payoffs)
    beta = cov(target_payoffs, control_payoffs) / (var(control_payoffs) + 1e-12)
    adjusted = target_payoffs .- beta .* (control_payoffs .- control_mean)
    return mean(adjusted)
end

function quasi_mc_variance_reduction_factor(std_mc::Float64, std_qmc::Float64)::Float64
    return (std_mc / (std_qmc + 1e-12))^2
end

function chebyshev_approximation(f::Function, a::Float64, b::Float64,
                                   n::Int=10)::Vector{Float64}
    nodes = [0.5*(a+b) + 0.5*(b-a)*cos(pi*(2k-1)/(2n)) for k in 1:n]
    coeffs = zeros(n)
    fvals = f.(nodes)
    for k in 1:n
        coeffs[k] = 2/n * sum(fvals[j]*cos(pi*(k-1)*(2j-1)/(2n)) for j in 1:n)
    end
    coeffs[1] /= 2
    return coeffs
end


# ---- Numerical Methods Utilities (continued) ----

function neville_interpolation(xs::Vector{Float64}, ys::Vector{Float64},
                                 x_target::Float64)::Float64
    n = length(xs); P = copy(ys)
    for j in 1:(n-1)
        for i in n:-1:j+1
            P[i] = ((x_target - xs[i-j])*P[i] - (x_target - xs[i])*P[i-1]) /
                   (xs[i] - xs[i-j] + 1e-15)
        end
    end
    return P[end]
end

function cubic_spline_coefficients(x::Vector{Float64}, y::Vector{Float64})
    n = length(x) - 1
    h = diff(x); alpha = zeros(n+1)
    for i in 2:n
        alpha[i] = 3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1]
    end
    l = ones(n+1); mu = zeros(n+1); z = zeros(n+1)
    for i in 2:n
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / (l[i]+1e-15)
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / (l[i]+1e-15)
    end
    c = zeros(n+1); b = zeros(n); d = zeros(n)
    for j in n:-1:1
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    end
    return b, c[1:end-1], d
end


# ---- Numerical Methods Utilities (continued) ----

function neville_interpolation(xs::Vector{Float64}, ys::Vector{Float64},
                                 x_target::Float64)::Float64
    n = length(xs); P = copy(ys)
    for j in 1:(n-1)
        for i in n:-1:j+1
            P[i] = ((x_target - xs[i-j])*P[i] - (x_target - xs[i])*P[i-1]) /
                   (xs[i] - xs[i-j] + 1e-15)
        end
    end
    return P[end]
end

function cubic_spline_coefficients(x::Vector{Float64}, y::Vector{Float64})
    n = length(x) - 1
    h = diff(x); alpha = zeros(n+1)
    for i in 2:n
        alpha[i] = 3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1]
    end
    l = ones(n+1); mu = zeros(n+1); z = zeros(n+1)
    for i in 2:n
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / (l[i]+1e-15)
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / (l[i]+1e-15)
    end
    c = zeros(n+1); b = zeros(n); d = zeros(n)
    for j in n:-1:1
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    end
    return b, c[1:end-1], d
end


# ---- Numerical Methods Utilities (continued) ----

function neville_interpolation(xs::Vector{Float64}, ys::Vector{Float64},
                                 x_target::Float64)::Float64
    n = length(xs); P = copy(ys)
    for j in 1:(n-1)
        for i in n:-1:j+1
            P[i] = ((x_target - xs[i-j])*P[i] - (x_target - xs[i])*P[i-1]) /
                   (xs[i] - xs[i-j] + 1e-15)
        end
    end
    return P[end]
end

function cubic_spline_coefficients(x::Vector{Float64}, y::Vector{Float64})
    n = length(x) - 1
    h = diff(x); alpha = zeros(n+1)
    for i in 2:n
        alpha[i] = 3*(y[i+1]-y[i])/h[i] - 3*(y[i]-y[i-1])/h[i-1]
    end
    l = ones(n+1); mu = zeros(n+1); z = zeros(n+1)
    for i in 2:n
        l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i] / (l[i]+1e-15)
        z[i]  = (alpha[i] - h[i-1]*z[i-1]) / (l[i]+1e-15)
    end
    c = zeros(n+1); b = zeros(n); d = zeros(n)
    for j in n:-1:1
        c[j] = z[j] - mu[j]*c[j+1]
        b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
        d[j] = (c[j+1]-c[j])/(3*h[j])
    end
    return b, c[1:end-1], d
end

end # module NumericalMethods
