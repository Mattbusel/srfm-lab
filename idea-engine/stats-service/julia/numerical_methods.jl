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


# ============================================================
# SECTION 2: ROOT FINDING & OPTIMIZATION EXTENSIONS
# ============================================================

function bisection(f::Function, a::Float64, b::Float64; tol::Float64=1e-10, max_iter::Int=200)
    fa = f(a); fb = f(b)
    fa * fb > 0 && error("f(a) and f(b) must have opposite signs")
    for _ in 1:max_iter
        c = (a + b) / 2.0
        fc = f(c)
        abs(fc) < tol && return c
        if fa * fc < 0
            b = c; fb = fc
        else
            a = c; fa = fc
        end
    end
    return (a + b) / 2.0
end

function secant_method(f::Function, x0::Float64, x1::Float64;
                        tol::Float64=1e-10, max_iter::Int=100)
    for _ in 1:max_iter
        fx0 = f(x0); fx1 = f(x1)
        abs(fx1 - fx0) < 1e-15 && break
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        abs(x2 - x1) < tol && return x2
        x0 = x1; x1 = x2
    end
    return x1
end

function ridder_method(f::Function, a::Float64, b::Float64;
                        tol::Float64=1e-10, max_iter::Int=100)
    fa = f(a); fb = f(b)
    fa * fb > 0 && error("sign change required")
    for _ in 1:max_iter
        c = (a + b) / 2.0; fc = f(c)
        abs(fc) < tol && return c
        sign_diff = sign(fa - fb)
        d = c + (c - a) * sign_diff * fc / sqrt(fc^2 - fa*fb + 1e-20)
        fd = f(d)
        abs(fd) < tol && return d
        if fc * fd < 0; a = c; fa = fc; b = d; fb = fd
        elseif fa * fd < 0; b = d; fb = fd
        else; a = d; fa = fd; end
        abs(b - a) < tol && break
    end
    return (a + b) / 2.0
end

function fixed_point_iteration(g::Function, x0::Float64;
                                 tol::Float64=1e-10, max_iter::Int=500)
    x = x0
    for i in 1:max_iter
        x_new = g(x)
        abs(x_new - x) < tol && return (root=x_new, iterations=i, converged=true)
        x = x_new
    end
    return (root=x, iterations=max_iter, converged=false)
end

function steffensen_method(g::Function, x0::Float64;
                             tol::Float64=1e-10, max_iter::Int=100)
    # Aitken's delta-squared acceleration of fixed point iteration
    x = x0
    for i in 1:max_iter
        x1 = g(x); x2 = g(x1)
        denom = x2 - 2*x1 + x
        abs(denom) < 1e-15 && break
        x_new = x - (x1 - x)^2 / denom
        abs(x_new - x) < tol && return (root=x_new, iterations=i)
        x = x_new
    end
    return (root=x, iterations=max_iter)
end

# ============================================================
# SECTION 3: NUMERICAL INTEGRATION EXTENSIONS
# ============================================================

function trapezoid_rule(f::Function, a::Float64, b::Float64, n::Int=1000)
    h = (b - a) / n
    total = f(a) + f(b)
    for i in 1:n-1
        total += 2 * f(a + i*h)
    end
    return total * h / 2
end

function romberg_integration(f::Function, a::Float64, b::Float64;
                               max_order::Int=8, tol::Float64=1e-10)
    R = zeros(max_order, max_order)
    h = b - a
    R[1,1] = h * (f(a) + f(b)) / 2
    for j in 2:max_order
        h /= 2
        sum_new = sum(f(a + (2k-1)*h) for k in 1:2^(j-2))
        R[j,1] = R[j-1,1]/2 + h * sum_new
        for k in 2:j
            R[j,k] = R[j,k-1] + (R[j,k-1] - R[j-1,k-1]) / (4^(k-1) - 1)
        end
        j > 2 && abs(R[j,j] - R[j-1,j-1]) < tol && return R[j,j]
    end
    return R[max_order, max_order]
end

function monte_carlo_integrate(f::Function, a::Float64, b::Float64;
                                 n::Int=100000, seed::Int=42)
    rng_state = UInt64(seed)
    lcg_next() = (rng_state = rng_state * 6364136223846793005 + 1442695040888963407; rng_state)
    total = 0.0
    total_sq = 0.0
    scale = (b - a) / typemax(UInt64)
    for _ in 1:n
        x = a + lcg_next() * scale
        fx = f(x)
        total += fx; total_sq += fx^2
    end
    est = total * (b - a) / n
    variance = (total_sq/n - (total/n)^2) * (b-a)^2 / n
    return (estimate=est, std_error=sqrt(max(variance, 0.0)))
end

function double_integral(f::Function, ax::Float64, bx::Float64,
                           ay::Function, by::Function, nx::Int=50, ny::Int=50)
    # Adaptive double integral using Gaussian quadrature on outer, trapezoid inner
    hx = (bx - ax) / nx
    total = 0.0
    for i in 0:nx-1
        x = ax + (i + 0.5)*hx
        y0 = ay(x); y1 = by(x)
        hy = (y1 - y0) / ny
        inner = 0.0
        for j in 0:ny-1
            y = y0 + (j + 0.5)*hy
            inner += f(x, y)
        end
        total += inner * hy
    end
    return total * hx
end

# ============================================================
# SECTION 4: ODE SOLVERS
# ============================================================

function euler_method(f::Function, y0::Vector{Float64}, t_span::Tuple{Float64,Float64},
                       n::Int=1000)
    t0, t1 = t_span
    h = (t1 - t0) / n
    t = t0; y = copy(y0)
    ts = [t0]; ys = [copy(y0)]
    for _ in 1:n
        y = y .+ h .* f(t, y)
        t += h
        push!(ts, t); push!(ys, copy(y))
    end
    return (t=ts, y=ys)
end

function rk4(f::Function, y0::Vector{Float64}, t_span::Tuple{Float64,Float64}, n::Int=1000)
    t0, t1 = t_span
    h = (t1 - t0) / n
    t = t0; y = copy(y0)
    ts = [t0]; ys = [copy(y0)]
    for _ in 1:n
        k1 = f(t,       y)
        k2 = f(t + h/2, y .+ h/2 .* k1)
        k3 = f(t + h/2, y .+ h/2 .* k2)
        k4 = f(t + h,   y .+ h   .* k3)
        y = y .+ h/6 .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)
        t += h
        push!(ts, t); push!(ys, copy(y))
    end
    return (t=ts, y=ys)
end

function rk45_adaptive(f::Function, y0::Vector{Float64},
                        t_span::Tuple{Float64,Float64};
                        rtol::Float64=1e-6, atol::Float64=1e-8,
                        h_init::Float64=0.01, h_min::Float64=1e-10)
    # Dormand-Prince RK45 coefficients
    c2=1/5; c3=3/10; c4=4/5; c5=8/9
    a21=1/5
    a31=3/40; a32=9/40
    a41=44/45; a42=-56/15; a43=32/9
    a51=19372/6561; a52=-25360/2187; a53=64448/6561; a54=-212/729
    a61=9017/3168; a62=-355/33; a63=46732/5247; a64=49/176; a65=-5103/18656
    b1=35/384; b3=500/1113; b4=125/192; b5=-2187/6784; b6=11/84
    e1=71/57600; e3=-71/16695; e4=71/1920; e5=-17253/339200; e6=22/525; e7=-1/40

    t0, t1 = t_span; t = t0; y = copy(y0); h = h_init
    ts = [t0]; ys = [copy(y0)]
    while t < t1
        h = min(h, t1 - t)
        k1 = f(t, y)
        k2 = f(t+c2*h, y .+ h*(a21*k1))
        k3 = f(t+c3*h, y .+ h*(a31*k1 .+ a32*k2))
        k4 = f(t+c4*h, y .+ h*(a41*k1 .+ a42*k2 .+ a43*k3))
        k5 = f(t+c5*h, y .+ h*(a51*k1 .+ a52*k2 .+ a53*k3 .+ a54*k4))
        k6 = f(t+h,    y .+ h*(a61*k1 .+ a62*k2 .+ a63*k3 .+ a64*k4 .+ a65*k5))
        y5 = y .+ h*(b1*k1 .+ b3*k3 .+ b4*k4 .+ b5*k5 .+ b6*k6)
        k7 = f(t+h, y5)
        err_vec = h*(e1*k1 .+ e3*k3 .+ e4*k4 .+ e5*k5 .+ e6*k6 .+ e7*k7)
        err = sqrt(mean(((err_vec ./ (atol .+ rtol .* abs.(y5))).^2)))
        if err <= 1.0
            t += h; y = y5
            push!(ts, t); push!(ys, copy(y))
        end
        h *= min(5.0, max(0.1, 0.9 * err^(-0.2)))
        h < h_min && break
    end
    return (t=ts, y=ys)
end

function implicit_euler(f_jac::Function, y0::Vector{Float64},
                          t_span::Tuple{Float64,Float64}, n::Int=1000)
    # f_jac(t, y) = (f, J) where J is Jacobian
    t0, t1 = t_span; h = (t1 - t0) / n
    t = t0; y = copy(y0)
    ts = [t0]; ys = [copy(y0)]
    for _ in 1:n
        # Newton iteration: (I - h*J)*delta = h*f
        t_new = t + h
        y_k = copy(y)
        for _ in 1:10
            fval, J = f_jac(t_new, y_k)
            A = I(length(y)) .- h .* J
            g = y_k .- y .- h .* fval
            delta = A \ g
            y_k .-= delta
            norm(delta) < 1e-12 && break
        end
        t = t_new; y = y_k
        push!(ts, t); push!(ys, copy(y))
    end
    return (t=ts, y=ys)
end

# ============================================================
# SECTION 5: LINEAR ALGEBRA EXTENSIONS
# ============================================================

function lu_decomposition(A::Matrix{Float64})
    n = size(A, 1)
    L = zeros(n, n); U = copy(A); P = collect(1:n)
    for k in 1:n
        # Partial pivoting
        max_val, max_idx = findmax(abs.(U[k:n, k]))
        pivot = max_idx + k - 1
        if pivot != k
            U[[k, pivot], :] = U[[pivot, k], :]
            L[[k, pivot], 1:k-1] = L[[pivot, k], 1:k-1]
            P[k], P[pivot] = P[pivot], P[k]
        end
        L[k,k] = 1.0
        for i in k+1:n
            L[i,k] = U[i,k] / (U[k,k] + 1e-15)
            U[i,k:n] .-= L[i,k] .* U[k,k:n]
        end
    end
    return L, U, P
end

function solve_lu(L::Matrix{Float64}, U::Matrix{Float64},
                   P::Vector{Int}, b::Vector{Float64})
    n = length(b)
    pb = b[P]
    # Forward substitution
    y = zeros(n)
    for i in 1:n
        y[i] = pb[i] - dot(L[i,1:i-1], y[1:i-1])
    end
    # Backward substitution
    x = zeros(n)
    for i in n:-1:1
        x[i] = (y[i] - dot(U[i,i+1:n], x[i+1:n])) / (U[i,i] + 1e-15)
    end
    return x
end

function qr_decomposition(A::Matrix{Float64})
    m, n = size(A)
    Q = Matrix{Float64}(I, m, m)
    R = copy(A)
    for k in 1:min(m-1, n)
        x = R[k:m, k]
        norm_x = norm(x)
        v = copy(x)
        v[1] += sign(x[1]) * norm_x
        nv = norm(v)
        nv < 1e-14 && continue
        v ./= nv
        # Apply Householder
        R[k:m, k:n] .-= 2 .* v * (v' * R[k:m, k:n])
        Q[k:m, :] .-= 2 .* v * (v' * Q[k:m, :])
    end
    return Q', R
end

function cholesky_decomposition(A::Matrix{Float64})
    n = size(A, 1)
    L = zeros(n, n)
    for i in 1:n
        for j in 1:i-1
            L[i,j] = (A[i,j] - dot(L[i,1:j-1], L[j,1:j-1])) / (L[j,j] + 1e-15)
        end
        val = A[i,i] - sum(L[i,1:i-1].^2)
        L[i,i] = sqrt(max(val, 1e-15))
    end
    return L
end

function svd_power_iteration(A::Matrix{Float64}; n_components::Int=5, max_iter::Int=500)
    m, n = size(A)
    k = min(n_components, min(m, n))
    U_out = zeros(m, k)
    S_out = zeros(k)
    V_out = zeros(n, k)
    B = copy(A)
    for j in 1:k
        u = randn(m); u ./= norm(u)
        for _ in 1:max_iter
            v = B' * u; nv = norm(v); v ./= (nv + 1e-15)
            u_new = B * v; nu = norm(u_new); u_new ./= (nu + 1e-15)
            norm(u_new - u) < 1e-10 && (u = u_new; break)
            u = u_new
        end
        v = B' * u; sigma = norm(v); v ./= (sigma + 1e-15)
        U_out[:,j] = u; S_out[j] = sigma; V_out[:,j] = v
        B .-= sigma .* (u * v')
    end
    return (U=U_out, S=S_out, V=V_out)
end

function matrix_inverse_via_lu(A::Matrix{Float64})
    n = size(A, 1)
    L, U, P = lu_decomposition(A)
    A_inv = zeros(n, n)
    for i in 1:n
        e_i = zeros(n); e_i[i] = 1.0
        A_inv[:,i] = solve_lu(L, U, P, e_i)
    end
    return A_inv
end

function condition_number(A::Matrix{Float64})
    sv = svd_power_iteration(A; n_components=min(size(A)...))
    s_max = maximum(sv.S); s_min = minimum(sv.S[sv.S .> 1e-15])
    return s_max / (s_min + 1e-15)
end

# ============================================================
# SECTION 6: FFT & SPECTRAL ANALYSIS
# ============================================================

function cooley_tukey_fft(x::Vector{ComplexF64})
    n = length(x)
    n == 1 && return x
    if n % 2 != 0
        # DFT fallback for odd length
        W = [exp(-2π*im*j*k/n) for j in 0:n-1, k in 0:n-1]
        return W * x
    end
    even = cooley_tukey_fft(x[1:2:n])
    odd  = cooley_tukey_fft(x[2:2:n])
    T = [exp(-2π*im*(k-1)/n) * odd[k] for k in 1:n÷2]
    return vcat(even .+ T, even .- T)
end

function real_fft(x::Vector{Float64})
    n = length(x)
    # Pad to power of 2
    n2 = 1
    while n2 < n; n2 *= 2; end
    xc = complex(vcat(x, zeros(n2-n)))
    return cooley_tukey_fft(xc)
end

function power_spectral_density(x::Vector{Float64}; fs::Float64=1.0)
    n = length(x)
    X = real_fft(x)
    n2 = length(X)
    psd = abs.(X[1:n2÷2+1]).^2 ./ (n2 * fs)
    psd[2:end-1] .*= 2  # one-sided
    freqs = (0:n2÷2) .* (fs / n2)
    return (frequencies=collect(freqs), psd=psd)
end

function welch_psd(x::Vector{Float64}; fs::Float64=1.0, segment_length::Int=256,
                    overlap::Int=128)
    n = length(x)
    step = segment_length - overlap
    psds = Vector{Vector{Float64}}()
    i = 1
    while i + segment_length - 1 <= n
        seg = x[i:i+segment_length-1]
        # Hann window
        w = [0.5*(1 - cos(2π*(j-1)/(segment_length-1))) for j in 1:segment_length]
        seg_w = seg .* w
        result = power_spectral_density(seg_w; fs=fs)
        push!(psds, result.psd)
        freqs = result.frequencies
        i += step
    end
    isempty(psds) && return power_spectral_density(x; fs=fs)
    avg_psd = mean(psds)
    return (frequencies=freqs, psd=avg_psd)
end

function dominant_frequencies(x::Vector{Float64}; fs::Float64=1.0, top_k::Int=5)
    result = power_spectral_density(x; fs=fs)
    order = sortperm(result.psd; rev=true)
    freqs = result.frequencies[order[1:min(top_k,end)]]
    powers = result.psd[order[1:min(top_k,end)]]
    return (frequencies=freqs, powers=powers)
end

function bandpass_filter_fft(x::Vector{Float64}, f_low::Float64, f_high::Float64;
                               fs::Float64=1.0)
    n = length(x)
    X = real_fft(x)
    n2 = length(X)
    freqs = (0:n2-1) .* (fs / n2)
    mask = [(f >= f_low && f <= f_high) ? 1.0 : 0.0 for f in freqs]
    # Apply mask to full spectrum (conjugate symmetry)
    X_full = cooley_tukey_fft(complex(x))
    n_full = length(X_full)
    freqs_full = (0:n_full-1) .* (fs / n_full)
    for i in 1:n_full
        f = min(freqs_full[i], fs - freqs_full[i])
        (f < f_low || f > f_high) && (X_full[i] = 0.0)
    end
    # IFFT
    x_filt = real.(cooley_tukey_fft(conj.(X_full)) ./ n_full)
    return x_filt
end

# ============================================================
# SECTION 7: INTERPOLATION & APPROXIMATION
# ============================================================

function bilinear_interpolation(x::Float64, y::Float64,
                                  x_grid::Vector{Float64}, y_grid::Vector{Float64},
                                  z_grid::Matrix{Float64})
    # Find bracketing indices
    ix = searchsortedlast(x_grid, x)
    iy = searchsortedlast(y_grid, y)
    ix = clamp(ix, 1, length(x_grid)-1)
    iy = clamp(iy, 1, length(y_grid)-1)
    x1 = x_grid[ix]; x2 = x_grid[ix+1]
    y1 = y_grid[iy]; y2 = y_grid[iy+1]
    t = (x - x1) / (x2 - x1 + 1e-15)
    u = (y - y1) / (y2 - y1 + 1e-15)
    return (1-t)*(1-u)*z_grid[ix,iy] + t*(1-u)*z_grid[ix+1,iy] +
           (1-t)*u*z_grid[ix,iy+1]   + t*u*z_grid[ix+1,iy+1]
end

function polynomial_interpolation_lagrange(xs::Vector{Float64}, ys::Vector{Float64},
                                             x::Float64)
    n = length(xs)
    total = 0.0
    for i in 1:n
        basis = 1.0
        for j in 1:n
            j == i && continue
            basis *= (x - xs[j]) / (xs[i] - xs[j] + 1e-15)
        end
        total += ys[i] * basis
    end
    return total
end

function linear_interpolation(x::Float64, xs::Vector{Float64}, ys::Vector{Float64})
    i = searchsortedlast(xs, x)
    i = clamp(i, 1, length(xs)-1)
    t = (x - xs[i]) / (xs[i+1] - xs[i] + 1e-15)
    return ys[i] + t * (ys[i+1] - ys[i])
end

function pchip_monotone_interpolation(xs::Vector{Float64}, ys::Vector{Float64})
    # Piecewise cubic Hermite ensuring monotonicity
    n = length(xs)
    h = diff(xs); delta = diff(ys) ./ (h .+ 1e-15)
    d = zeros(n)
    # Interior slopes
    for i in 2:n-1
        if delta[i-1] * delta[i] > 0
            w1 = 2*h[i] + h[i-1]; w2 = h[i] + 2*h[i-1]
            d[i] = (w1 + w2) / (w1/delta[i-1] + w2/delta[i] + 1e-15)
        end
    end
    d[1] = delta[1]; d[n] = delta[end]
    # Monotonicity constraint
    for i in 1:n-1
        if abs(delta[i]) < 1e-15; d[i] = 0.0; d[i+1] = 0.0; continue; end
        alpha = d[i] / delta[i]; beta = d[i+1] / delta[i]
        r = alpha^2 + beta^2
        r > 9 && begin
            tau = 3 / sqrt(r)
            d[i] *= tau; d[i+1] *= tau
        end
    end
    return d  # slopes at knots; use with cubic Hermite basis
end

function rbf_interpolation(centers::Vector{Float64}, values::Vector{Float64},
                             x_query::Vector{Float64}; epsilon::Float64=1.0)
    # Radial basis function interpolation with Gaussian kernel
    n = length(centers)
    Phi = [exp(-epsilon * (centers[i] - centers[j])^2) for i in 1:n, j in 1:n]
    # Solve for weights
    w = (Phi + 1e-8*I(n)) \ values
    # Evaluate
    return [sum(w[j] * exp(-epsilon * (x - centers[j])^2) for j in 1:n)
            for x in x_query]
end

# ============================================================
# SECTION 8: MONTE CARLO & QUASI-MONTE CARLO
# ============================================================

function halton_sequence(n::Int, base::Int)
    result = zeros(n)
    for i in 1:n
        f = 1.0; r = 0.0; k = i
        while k > 0
            f /= base
            r += f * (k % base)
            k ÷= base
        end
        result[i] = r
    end
    return result
end

function sobol_sequence(n::Int, d::Int)
    # Van der Corput in base 2 for dimension 1, scrambled for higher dims
    seq = zeros(n, d)
    for k in 1:d
        for i in 1:n
            f = 1.0; r = 0.0; num = i * k
            while num > 0
                f /= 2; r += f * (num & 1); num >>= 1
            end
            seq[i,k] = r
        end
    end
    return seq
end

function antithetic_mc_option(S0::Float64, K::Float64, r::Float64, sigma::Float64,
                               T::Float64; n::Int=100000, seed::Int=0)
    rng = seed
    lcg() = (rng = rng * 1664525 + 1013904223; (rng & 0xFFFFFFFF) / 4294967295.0)
    box_muller() = begin
        u1 = max(lcg(), 1e-10); u2 = lcg()
        sqrt(-2*log(u1)) * cos(2π*u2)
    end
    payoffs = Float64[]
    for _ in 1:n÷2
        z = box_muller()
        for z_ in [z, -z]
            ST = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*z_)
            push!(payoffs, max(ST - K, 0.0))
        end
    end
    disc = exp(-r*T)
    est = disc * mean(payoffs)
    se  = disc * std(payoffs) / sqrt(length(payoffs))
    return (price=est, std_error=se)
end

function control_variate_mc(payoff_fn::Function, control_fn::Function,
                              E_control::Float64, n::Int=100000)
    # Generate samples — simplified with standard normal
    samples = [randn() for _ in 1:n]
    Y = [payoff_fn(s) for s in samples]
    X = [control_fn(s) for s in samples]
    # Optimal coefficient
    c_opt = cov(Y, X) / (var(X) + 1e-10)
    Y_cv = Y .- c_opt .* (X .- E_control)
    return (estimate=mean(Y_cv), std_error=std(Y_cv)/sqrt(n),
            variance_reduction=(var(Y)-var(Y_cv))/(var(Y)+1e-10))
end

function stratified_sampling_mc(f::Function, a::Float64, b::Float64,
                                  n_strata::Int=100, n_per_strata::Int=10)
    total = 0.0; total_var = 0.0
    stratum_width = (b - a) / n_strata
    for k in 1:n_strata
        lo = a + (k-1)*stratum_width
        hi = lo + stratum_width
        samples = [lo + rand()*(hi-lo) for _ in 1:n_per_strata]
        vals = f.(samples)
        total += mean(vals)
        total_var += var(vals) / n_per_strata
    end
    est = total * stratum_width
    se  = sqrt(total_var) * stratum_width / sqrt(n_strata)
    return (estimate=est, std_error=se)
end

function importance_sampling_mc(f::Function, w::Function, q_sampler::Function,
                                  n::Int=100000)
    # E_p[f] = E_q[f * p/q] ~ mean(f(x)*w(x)) where w=p/q
    vals = [f(x) * w(x) for x in [q_sampler() for _ in 1:n]]
    return (estimate=mean(vals), std_error=std(vals)/sqrt(n))
end

# ============================================================
# SECTION 9: FINITE DIFFERENCE & PDE METHODS
# ============================================================

function bs_pde_crank_nicolson(S_max::Float64, K::Float64, r::Float64,
                                sigma::Float64, T::Float64;
                                M::Int=200, N::Int=200)
    # Black-Scholes PDE via Crank-Nicolson
    dS = S_max / M; dt = T / N
    S = [j * dS for j in 0:M]
    V = max.(S .- K, 0.0)  # terminal condition (call)
    alpha = 0.25 * dt * (sigma^2 .* (0:M).^2 .- r .* (0:M))
    beta  = -0.5 * dt * (sigma^2 .* (0:M).^2 .+ r)
    gamma = 0.25 * dt * (sigma^2 .* (0:M).^2 .+ r .* (0:M))

    for _ in 1:N
        # Build tridiagonal system
        n_int = M - 1
        lower = Float64[]; main_d = Float64[]; upper = Float64[]
        rhs = zeros(n_int)
        for i in 1:n_int
            idx = i + 1  # 1-based: internal nodes 2..M
            push!(lower, -alpha[idx])
            push!(main_d, 1.0 - beta[idx])
            push!(upper, -gamma[idx])
            rhs[i] = alpha[idx]*V[idx-1] + (1 + beta[idx])*V[idx] + gamma[idx]*V[idx+1]
        end
        rhs[1]   += alpha[2]*V[1]
        rhs[end] += gamma[M]*V[M+1]
        # Thomas algorithm
        n_t = n_int
        c_star = zeros(n_t); d_star = zeros(n_t)
        c_star[1] = upper[1] / (main_d[1] + 1e-15)
        d_star[1] = rhs[1] / (main_d[1] + 1e-15)
        for i in 2:n_t
            denom = main_d[i] - lower[i] * c_star[i-1]
            c_star[i] = upper[min(i,n_t)] / (denom + 1e-15)
            d_star[i] = (rhs[i] - lower[i]*d_star[i-1]) / (denom + 1e-15)
        end
        x = zeros(n_t)
        x[n_t] = d_star[n_t]
        for i in n_t-1:-1:1; x[i] = d_star[i] - c_star[i]*x[i+1]; end
        V[2:M] = x
        # Boundary conditions
        V[1] = 0.0; V[M+1] = S_max - K*exp(-r*dt)
    end
    return (S=S, V=V)
end

function heat_equation_explicit(u0::Vector{Float64}, dx::Float64, dt::Float64,
                                  D::Float64, T::Float64)
    n = length(u0); steps = round(Int, T / dt)
    u = copy(u0); r = D * dt / dx^2
    r > 0.5 && @warn "Stability condition violated: r=$(round(r,digits=3)) > 0.5"
    for _ in 1:steps
        u_new = copy(u)
        for i in 2:n-1
            u_new[i] = u[i] + r*(u[i+1] - 2*u[i] + u[i-1])
        end
        u = u_new
    end
    return u
end

function heat_equation_implicit(u0::Vector{Float64}, dx::Float64, dt::Float64,
                                  D::Float64, T::Float64)
    n = length(u0); steps = round(Int, T / dt)
    r = D * dt / dx^2; u = copy(u0)
    # Build tridiagonal: (1+2r)*u_i - r*(u_{i-1}+u_{i+1}) = u_old_i
    n_int = n - 2
    lo = fill(-r, n_int-1); md = fill(1+2r, n_int); up = fill(-r, n_int-1)
    for _ in 1:steps
        rhs = u[2:n-1]
        # Thomas
        c_s = zeros(n_int); d_s = zeros(n_int)
        c_s[1] = up[1]/md[1]; d_s[1] = rhs[1]/md[1]
        for i in 2:n_int
            denom = md[i] - lo[i-1]*c_s[i-1]
            c_s[i] = i < n_int ? up[i]/denom : 0.0
            d_s[i] = (rhs[i] - lo[i-1]*d_s[i-1]) / denom
        end
        x = zeros(n_int); x[n_int] = d_s[n_int]
        for i in n_int-1:-1:1; x[i] = d_s[i] - c_s[i]*x[i+1]; end
        u[2:n-1] = x
    end
    return u
end

# ============================================================
# SECTION 10: OPTIMIZATION ALGORITHMS
# ============================================================

function gradient_descent(f_grad::Function, x0::Vector{Float64};
                            lr::Float64=0.01, tol::Float64=1e-6, max_iter::Int=10000)
    x = copy(x0)
    for i in 1:max_iter
        _, g = f_grad(x)
        x_new = x .- lr .* g
        norm(x_new - x) < tol && return (x=x_new, iterations=i, converged=true)
        x = x_new
    end
    return (x=x, iterations=max_iter, converged=false)
end

function adam_optimizer(f_grad::Function, x0::Vector{Float64};
                         lr::Float64=0.001, beta1::Float64=0.9, beta2::Float64=0.999,
                         eps::Float64=1e-8, max_iter::Int=10000, tol::Float64=1e-6)
    x = copy(x0); n = length(x)
    m = zeros(n); v = zeros(n)
    for t in 1:max_iter
        _, g = f_grad(x)
        m = beta1 .* m .+ (1-beta1) .* g
        v = beta2 .* v .+ (1-beta2) .* g.^2
        m_hat = m ./ (1 - beta1^t)
        v_hat = v ./ (1 - beta2^t)
        x_new = x .- lr .* m_hat ./ (sqrt.(v_hat) .+ eps)
        norm(x_new - x) < tol && return (x=x_new, iterations=t, converged=true)
        x = x_new
    end
    return (x=x, iterations=max_iter, converged=false)
end

function lbfgs(f_grad::Function, x0::Vector{Float64};
                m::Int=10, tol::Float64=1e-6, max_iter::Int=1000)
    x = copy(x0); n = length(x)
    f_val, g = f_grad(x)
    s_hist = Vector{Vector{Float64}}(); y_hist = Vector{Vector{Float64}}()
    rho_hist = Float64[]
    for iter in 1:max_iter
        # Two-loop recursion
        q = copy(g)
        alphas = zeros(length(s_hist))
        for i in length(s_hist):-1:1
            alphas[i] = rho_hist[i] * dot(s_hist[i], q)
            q .-= alphas[i] .* y_hist[i]
        end
        r = copy(q)
        if !isempty(y_hist)
            gamma = dot(s_hist[end], y_hist[end]) / (dot(y_hist[end], y_hist[end]) + 1e-15)
            r .*= gamma
        end
        for i in 1:length(s_hist)
            beta = rho_hist[i] * dot(y_hist[i], r)
            r .+= (alphas[i] - beta) .* s_hist[i]
        end
        p = -r
        # Line search (Armijo)
        alpha_ls = 1.0
        for _ in 1:20
            x_new = x .+ alpha_ls .* p
            f_new, _ = f_grad(x_new)
            f_new <= f_val + 1e-4 * alpha_ls * dot(g, p) && break
            alpha_ls *= 0.5
        end
        x_new = x .+ alpha_ls .* p
        f_new, g_new = f_grad(x_new)
        s_k = x_new - x; y_k = g_new - g
        sy = dot(s_k, y_k)
        if sy > 1e-15
            if length(s_hist) >= m
                popfirst!(s_hist); popfirst!(y_hist); popfirst!(rho_hist)
            end
            push!(s_hist, s_k); push!(y_hist, y_k); push!(rho_hist, 1/sy)
        end
        x = x_new; g = g_new; f_val = f_new
        norm(g) < tol && return (x=x, f=f_val, iterations=iter, converged=true)
    end
    return (x=x, f=f_val, iterations=max_iter, converged=false)
end

function nelder_mead(f::Function, x0::Vector{Float64};
                      tol::Float64=1e-8, max_iter::Int=10000)
    n = length(x0)
    # Initialize simplex
    simplex = [copy(x0) for _ in 1:n+1]
    for i in 2:n+1
        simplex[i][i-1] += 0.05 * max(1.0, abs(x0[i-1]))
    end
    vals = [f(s) for s in simplex]
    alpha=1.0; gamma=2.0; rho=0.5; sigma=0.5
    for iter in 1:max_iter
        order = sortperm(vals)
        simplex = simplex[order]; vals = vals[order]
        # Centroid (exclude worst)
        xo = mean(simplex[1:n])
        # Reflection
        xr = xo .+ alpha*(xo .- simplex[n+1]); fr = f(xr)
        if vals[1] <= fr < vals[n]
            simplex[n+1] = xr; vals[n+1] = fr
        elseif fr < vals[1]
            xe = xo .+ gamma*(xr .- xo); fe = f(xe)
            if fe < fr; simplex[n+1] = xe; vals[n+1] = fe
            else;       simplex[n+1] = xr; vals[n+1] = fr; end
        else
            xc = xo .+ rho*(simplex[n+1] .- xo); fc = f(xc)
            if fc < vals[n+1]; simplex[n+1] = xc; vals[n+1] = fc
            else
                for i in 2:n+1
                    simplex[i] = simplex[1] .+ sigma*(simplex[i] .- simplex[1])
                    vals[i] = f(simplex[i])
                end
            end
        end
        # Convergence
        std([f(s) for s in simplex]) < tol && return (x=simplex[1], f=vals[1],
                                                        iterations=iter, converged=true)
    end
    return (x=simplex[1], f=vals[1], iterations=max_iter, converged=false)
end

function simulated_annealing(f::Function, x0::Vector{Float64};
                               T_init::Float64=10.0, T_final::Float64=1e-4,
                               cooling::Float64=0.99, max_iter::Int=100000,
                               step_size::Float64=0.1)
    x = copy(x0); fx = f(x); best_x = copy(x); best_f = fx
    temp = T_init
    for iter in 1:max_iter
        dx = randn(length(x)) .* step_size
        x_new = x .+ dx; f_new = f(x_new)
        delta = f_new - fx
        if delta < 0 || rand() < exp(-delta/temp)
            x = x_new; fx = f_new
            if fx < best_f; best_f = fx; best_x = copy(x); end
        end
        temp = max(temp * cooling, T_final)
        temp <= T_final && break
    end
    return (x=best_x, f=best_f)
end

# ============================================================
# EXTENDED DEMO
# ============================================================

function demo_numerical_methods_extended()
    println("=== Numerical Methods Extended Demo ===")

    # Root finding
    f_poly(x) = x^3 - 2x - 5
    r1 = bisection(f_poly, 2.0, 3.0)
    r2 = secant_method(f_poly, 2.0, 3.0)
    println("Bisection root: ", round(r1,digits=8))
    println("Secant root: ",   round(r2,digits=8))

    # Integration
    f_int(x) = sin(x)^2 / (1 + x^2)
    rom = romberg_integration(f_int, 0.0, π)
    println("Romberg ∫sin²(x)/(1+x²) dx: ", round(rom,digits=8))

    # ODE: SIR model
    sir(t, y) = begin
        β=0.3; γ=0.1; N=1e6
        [-β*y[1]*y[2]/N, β*y[1]*y[2]/N - γ*y[2], γ*y[2]]
    end
    sol = rk4(sir, [999000.0, 1000.0, 0.0], (0.0, 160.0); n=160)
    println("SIR peak infected: ", round(maximum([y[2] for y in sol.y])/1e6, digits=3), "M")

    # FFT
    t_vec = collect(0:0.01:1-0.01)
    x_sig = sin.(2π*5*t_vec) .+ 0.5*sin.(2π*13*t_vec)
    df = dominant_frequencies(x_sig; fs=100.0)
    println("Dominant freqs: ", round.(df.frequencies[1:2], digits=1))

    # L-BFGS
    rosenbrock_grad(x) = begin
        a=1.0; b=100.0
        f = (a - x[1])^2 + b*(x[2] - x[1]^2)^2
        g = [-2*(a-x[1]) - 4*b*x[1]*(x[2]-x[1]^2),
              2*b*(x[2]-x[1]^2)]
        return f, g
    end
    result = lbfgs(rosenbrock_grad, [-1.0, 1.0])
    println("L-BFGS Rosenbrock min: ", round.(result.x, digits=6))

    # BS PDE
    pde = bs_pde_crank_nicolson(200.0, 100.0, 0.05, 0.2, 1.0; M=100, N=100)
    spot_idx = findfirst(s -> s >= 100.0, pde.S)
    println("BS PDE call price (ATM): ", round(pde.V[spot_idx], digits=4))

    # MC antithetic
    mc = antithetic_mc_option(100.0, 100.0, 0.05, 0.2, 1.0; n=200000)
    println("MC antithetic call: ", round(mc.price, digits=4),
            " ±", round(mc.std_error, digits=4))
end

end # module NumericalMethods
