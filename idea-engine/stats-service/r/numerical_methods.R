## numerical_methods.R
## Monte Carlo variance reduction, PDE solvers, numerical optimization, quadrature
## Pure base R -- no library() calls

# ============================================================
# 1. MONTE CARLO WITH VARIANCE REDUCTION
# ============================================================

mc_plain <- function(payoff_fn, n_sim = 1e5, seed = 42) {
  set.seed(seed)
  z    <- rnorm(n_sim)
  vals <- payoff_fn(z)
  se   <- sd(vals) / sqrt(n_sim)
  list(estimate = mean(vals), se = se, ci = mean(vals) + c(-2,2)*se)
}

mc_antithetic <- function(payoff_fn, n_sim = 5e4, seed = 42) {
  set.seed(seed)
  z    <- rnorm(n_sim)
  v1   <- payoff_fn(z); v2 <- payoff_fn(-z)
  vals <- (v1 + v2) / 2
  se   <- sd(vals) / sqrt(n_sim)
  list(estimate = mean(vals), se = se, ci = mean(vals) + c(-2,2)*se,
       variance_ratio = var(vals) / var(v1))
}

mc_control_variate <- function(payoff_fn, control_fn, control_mean,
                               n_sim = 5e4, seed = 42) {
  set.seed(seed)
  z   <- rnorm(n_sim)
  y   <- payoff_fn(z); cx <- control_fn(z)
  b   <- cov(y, cx) / (var(cx) + 1e-12)
  adj <- y - b * (cx - control_mean)
  se  <- sd(adj) / sqrt(n_sim)
  list(estimate = mean(adj), se = se, b = b,
       variance_ratio = var(adj) / var(y))
}

mc_importance_sampling <- function(payoff_fn, density_ratio_fn,
                                   proposal_rvs, n_sim = 5e4, seed = 42) {
  set.seed(seed)
  z      <- proposal_rvs(n_sim)
  w      <- density_ratio_fn(z)
  vals   <- payoff_fn(z) * w
  se     <- sd(vals) / sqrt(n_sim)
  list(estimate = mean(vals), se = se,
       ess = (sum(w))^2 / (sum(w^2) + 1e-12))
}

mc_stratified <- function(payoff_fn, n_strata = 10, n_per = 500, seed = 42) {
  set.seed(seed)
  n   <- n_strata * n_per
  u   <- (seq_len(n) - runif(n)) / n      # stratified uniform
  z   <- qnorm(u)
  val <- payoff_fn(z)
  strat_means <- tapply(val, rep(seq_len(n_strata), each=n_per), mean)
  est <- mean(strat_means)
  se  <- sd(strat_means) / sqrt(n_strata)
  list(estimate = est, se = se)
}

mc_quasi <- function(payoff_fn, n_sim = 1e4, seed = 42) {
  # Van der Corput low-discrepancy sequence
  vdc <- function(n, base = 2) {
    sapply(seq_len(n), function(k) {
      x <- 0; f <- 1; i <- k
      while (i > 0) { f <- f / base; x <- x + (i %% base) * f; i <- i %/% base }
      x
    })
  }
  u  <- vdc(n_sim)
  z  <- qnorm(pmin(pmax(u, 1e-8), 1 - 1e-8))
  val <- payoff_fn(z)
  list(estimate = mean(val), se = sd(val)/sqrt(n_sim))
}

# Option pricing via MC
bs_mc_price <- function(S, K, r, sigma, T, type = "call",
                        n_sim = 1e5, var_reduce = "antithetic", seed = 42) {
  set.seed(seed)
  if (var_reduce == "antithetic") {
    z  <- rnorm(n_sim / 2)
    zs <- c(z, -z)
  } else zs <- rnorm(n_sim)
  ST   <- S * exp((r - sigma^2/2)*T + sigma*sqrt(T)*zs)
  pf   <- if (type == "call") pmax(ST - K, 0) else pmax(K - ST, 0)
  disc <- exp(-r*T)
  list(price = mean(pf) * disc, se = sd(pf)*disc/sqrt(n_sim))
}

# ============================================================
# 2. PDE SOLVERS — BLACK-SCHOLES EXPLICIT FINITE DIFFERENCE
# ============================================================

bs_pde_explicit <- function(S_max, K, r, sigma, T,
                            M = 100, N = 1000, type = "call") {
  dS  <- S_max / M
  dt  <- T / N
  S   <- seq(0, S_max, by = dS)
  m   <- length(S)

  # Terminal condition
  V <- if (type == "call") pmax(S - K, 0) else pmax(K - S, 0)

  # Coefficients
  j  <- seq_len(m - 2) + 1  # interior nodes (1-indexed)
  a  <- 0.5 * dt * (sigma^2 * j^2 - r * j)
  b  <- 1  - dt * (sigma^2 * j^2 + r)
  cc <- 0.5 * dt * (sigma^2 * j^2 + r * j)

  for (n in seq_len(N)) {
    V_new <- V
    V_new[j] <- a * V[j-1] + b * V[j] + cc * V[j+1]
    # Boundary conditions
    if (type == "call") {
      V_new[1]   <- 0
      V_new[m]   <- S_max - K * exp(-r * (T - n*dt))
    } else {
      V_new[1]   <- K * exp(-r * (T - n*dt))
      V_new[m]   <- 0
    }
    V <- V_new
  }
  list(S = S, V = V, S0_price = approx(S, V, xout = K)$y)
}

bs_pde_implicit <- function(S_max, K, r, sigma, T,
                            M = 100, N = 500, type = "call") {
  dS <- S_max / M; dt <- T / N
  S  <- seq(0, S_max, by = dS); m <- length(S)
  V  <- if (type == "call") pmax(S - K, 0) else pmax(K - S, 0)

  j  <- seq_len(m - 2) + 1
  a  <- -0.5 * dt * (sigma^2 * j^2 - r * j)
  b  <-  1   + dt  * (sigma^2 * j^2 + r)
  cc <- -0.5 * dt  * (sigma^2 * j^2 + r * j)

  tri_solve <- function(lo, diag_, hi, rhs) {
    n <- length(diag_); x <- numeric(n)
    diag_[1] <- diag_[1]; hi[1] <- hi[1] / diag_[1]; rhs[1] <- rhs[1] / diag_[1]
    for (i in 2:n) {
      m_  <- lo[i] / diag_[i-1]
      diag_[i] <- diag_[i] - m_ * hi[i-1]
      rhs[i]   <- rhs[i]   - m_ * rhs[i-1]
    }
    x[n] <- rhs[n] / diag_[n]
    for (i in (n-1):1) x[i] <- (rhs[i] - hi[i]*x[i+1]) / diag_[i]
    x
  }

  for (n in seq_len(N)) {
    rhs    <- V[j]
    rhs[1] <- rhs[1] - a[1] * V[1]
    rhs[length(rhs)] <- rhs[length(rhs)] - cc[length(cc)] * V[m]
    V[j] <- tri_solve(a[-1], b, cc[-length(cc)], rhs)
  }
  list(S = S, V = V, S0_price = approx(S, V, xout = K)$y)
}

# ============================================================
# 3. NUMERICAL OPTIMIZATION
# ============================================================

gradient_descent <- function(f, grad_f, x0, lr = 0.01,
                             max_iter = 1000, tol = 1e-6) {
  x      <- x0; hist <- numeric(max_iter)
  for (i in seq_len(max_iter)) {
    g    <- grad_f(x)
    x    <- x - lr * g
    hist[i] <- f(x)
    if (i > 1 && abs(hist[i] - hist[i-1]) < tol) break
  }
  list(x = x, value = f(x), history = hist[1:i], iterations = i)
}

adam_optimizer <- function(f, grad_f, x0, lr = 0.001,
                           b1 = 0.9, b2 = 0.999, eps = 1e-8,
                           max_iter = 2000, tol = 1e-7) {
  x  <- x0; m <- rep(0, length(x0)); v <- rep(0, length(x0))
  hist <- numeric(max_iter)
  for (t in seq_len(max_iter)) {
    g   <- grad_f(x)
    m   <- b1*m + (1-b1)*g
    v   <- b2*v + (1-b2)*g^2
    m_h <- m / (1 - b1^t)
    v_h <- v / (1 - b2^t)
    x   <- x - lr * m_h / (sqrt(v_h) + eps)
    hist[t] <- f(x)
    if (t > 1 && abs(hist[t] - hist[t-1]) < tol) break
  }
  list(x = x, value = f(x), history = hist[1:t], iterations = t)
}

nelder_mead <- function(f, x0, alpha = 1, gamma = 2,
                        rho = 0.5, sigma = 0.5,
                        max_iter = 5000, tol = 1e-8) {
  n   <- length(x0)
  sim <- rbind(x0, x0 + diag(n) * 0.05)
  for (extra in seq_len(n - 1)) sim <- rbind(sim, x0 + rnorm(n)*0.05)
  fval <- apply(sim, 1, f)

  for (iter in seq_len(max_iter)) {
    ord  <- order(fval); sim <- sim[ord,]; fval <- fval[ord]
    if (diff(range(fval)) < tol) break
    xo   <- colMeans(sim[-nrow(sim),])
    xr   <- xo + alpha * (xo - sim[nrow(sim),])
    fr   <- f(xr)
    if (fr < fval[1]) {
      xe <- xo + gamma*(xr - xo); fe <- f(xe)
      if (fe < fr) { sim[nrow(sim),] <- xe; fval[nrow(sim)] <- fe }
      else         { sim[nrow(sim),] <- xr; fval[nrow(sim)] <- fr }
    } else if (fr < fval[nrow(sim)-1]) {
      sim[nrow(sim),] <- xr; fval[nrow(sim)] <- fr
    } else {
      xc <- xo + rho*(sim[nrow(sim),] - xo); fc <- f(xc)
      if (fc < fval[nrow(sim)]) { sim[nrow(sim),] <- xc; fval[nrow(sim)] <- fc }
      else {
        for (i in 2:nrow(sim)) {
          sim[i,]  <- sim[1,] + sigma*(sim[i,] - sim[1,])
          fval[i]  <- f(sim[i,])
        }
      }
    }
  }
  list(x = sim[1,], value = fval[1], iterations = iter)
}

bfgs_numeric <- function(f, x0, h = 1e-5, max_iter = 200, tol = 1e-7) {
  n    <- length(x0); x <- x0
  H    <- diag(n)  # inverse Hessian approx
  grad <- function(x) sapply(seq_len(n), function(j) {
    xp <- x; xp[j] <- xp[j]+h; xm <- x; xm[j] <- xm[j]-h
    (f(xp) - f(xm)) / (2*h)
  })

  for (iter in seq_len(max_iter)) {
    g  <- grad(x); p <- -H %*% g
    # Line search
    ls <- 1
    for (lsi in 1:20) {
      if (f(x + ls*p) < f(x) + 0.0001*ls*sum(g*p)) break
      ls <- ls * 0.5
    }
    s  <- ls * p; x_new <- x + s
    y  <- grad(x_new) - g; sy <- sum(s*y)
    if (abs(sy) > 1e-10) {
      rho_ <- 1/sy
      H    <- (diag(n) - rho_*s%o%y) %*% H %*% (diag(n) - rho_*y%o%s) + rho_*s%o%s
    }
    x <- x_new
    if (sqrt(sum(g^2)) < tol) break
  }
  list(x=x, value=f(x), iterations=iter)
}

# ============================================================
# 4. NUMERICAL QUADRATURE
# ============================================================

gauss_legendre <- function(f, a, b, n = 10) {
  if (n == 5) {
    nodes  <- c(-0.9061798459, -0.5384693101, 0, 0.5384693101, 0.9061798459)
    weights <- c(0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851)
  } else {
    nodes  <- seq(-1, 1, length.out = n)
    weights <- rep(2/n, n)
  }
  t  <- (b + a)/2 + (b - a)/2 * nodes
  (b - a)/2 * sum(weights * f(t))
}

adaptive_quadrature <- function(f, a, b, tol = 1e-8, max_depth = 20) {
  simpson <- function(f, a, b) {
    (b-a)/6 * (f(a) + 4*f((a+b)/2) + f(b))
  }
  recurse <- function(f, a, b, tol, depth) {
    m    <- (a+b)/2
    s1   <- simpson(f, a, b)
    s2   <- simpson(f, a, m) + simpson(f, m, b)
    if (depth >= max_depth || abs(s2 - s1) < 15*tol)
      return(s2 + (s2 - s1)/15)
    recurse(f, a, m, tol/2, depth+1) + recurse(f, m, b, tol/2, depth+1)
  }
  recurse(f, a, b, tol, 0)
}

romberg_integration <- function(f, a, b, max_order = 8) {
  R <- matrix(NA, max_order, max_order)
  h <- b - a; R[1,1] <- h/2 * (f(a) + f(b))
  for (i in 2:max_order) {
    h   <- h/2; n <- 2^(i-2)
    pts <- a + h*(2*seq_len(n)-1)
    R[i,1] <- R[i-1,1]/2 + h*sum(f(pts))
    for (j in 2:i)
      R[i,j] <- R[i,j-1] + (R[i,j-1]-R[i-1,j-1]) / (4^(j-1) - 1)
    if (i > 2 && abs(R[i,i] - R[i-1,i-1]) < 1e-12) break
  }
  R[i,i]
}

# ============================================================
# 5. ROOT FINDING
# ============================================================

bisection <- function(f, a, b, tol = 1e-10, max_iter = 200) {
  fa <- f(a); fb <- f(b)
  if (sign(fa) == sign(fb)) stop("f(a) and f(b) must have opposite signs")
  for (iter in seq_len(max_iter)) {
    m  <- (a+b)/2; fm <- f(m)
    if (abs(fm) < tol || (b-a)/2 < tol) break
    if (sign(fm) == sign(fa)) { a <- m; fa <- fm } else { b <- m; fb <- fm }
  }
  list(root=m, value=fm, iterations=iter)
}

brentq <- function(f, a, b, tol = 1e-10, max_iter = 500) {
  fa <- f(a); fb <- f(b)
  if (sign(fa) == sign(fb)) stop("bracket needed")
  c_ <- a; fc <- fa; s <- 0; mflag <- TRUE
  for (iter in seq_len(max_iter)) {
    if (abs(fb) < tol) return(list(root=b, value=fb, iterations=iter))
    if (fa != fc && fb != fc) {
      s <- a*fb*fc/((fa-fb)*(fa-fc)) + b*fa*fc/((fb-fa)*(fb-fc)) +
           c_*fa*fb/((fc-fa)*(fc-fb))
    } else s <- b - fb*(b-a)/(fb-fa)
    cond <- !((3*a+b)/4 < s && s < b ||
              (mflag && abs(s-b) >= abs(b-c_)/2) ||
              (!mflag && abs(s-b) >= abs(c_-d)/2))
    if (!cond) { s <- (a+b)/2; mflag <- TRUE } else mflag <- FALSE
    fs <- f(s); d <- c_; c_ <- b; fc <- fb
    if (sign(fa) == sign(fs)) { a <- s; fa <- fs } else { b <- s; fb <- fs }
    if (abs(fa) < abs(fb)) { tmp <- a; a <- b; b <- tmp; tmp <- fa; fa <- fb; fb <- tmp }
  }
  list(root=b, value=fb, iterations=iter)
}

newton_raphson <- function(f, df, x0, tol = 1e-10, max_iter = 100) {
  x <- x0
  for (iter in seq_len(max_iter)) {
    fx <- f(x); dfx <- df(x)
    if (abs(dfx) < 1e-14) break
    x_new <- x - fx/dfx
    if (abs(x_new - x) < tol) { x <- x_new; break }
    x <- x_new
  }
  list(root=x, value=f(x), iterations=iter)
}

# ============================================================
# 6. MATRIX UTILITIES & LINEAR ALGEBRA
# ============================================================

cholesky_solve <- function(A, b) {
  L  <- t(chol(A + diag(nrow(A)) * 1e-10))
  # Forward substitution
  n  <- nrow(L); y <- numeric(n)
  y[1] <- b[1] / L[1,1]
  for (i in 2:n)
    y[i] <- (b[i] - sum(L[i, 1:(i-1)] * y[1:(i-1)])) / L[i,i]
  # Back substitution
  x <- numeric(n); Lt <- t(L)
  x[n] <- y[n] / Lt[n,n]
  for (i in (n-1):1)
    x[i] <- (y[i] - sum(Lt[i, (i+1):n] * x[(i+1):n])) / Lt[i,i]
  x
}

power_iteration <- function(A, n_iter = 100, tol = 1e-10) {
  n <- nrow(A); v <- rnorm(n); v <- v/sqrt(sum(v^2))
  lambda <- 0
  for (i in seq_len(n_iter)) {
    w       <- A %*% v
    lambda_ <- sum(v * w)
    v       <- w / sqrt(sum(w^2))
    if (abs(lambda_ - lambda) < tol) { lambda <- lambda_; break }
    lambda  <- lambda_
  }
  list(eigenvalue=lambda, eigenvector=as.vector(v))
}

lanczos_svd <- function(A, k = 5, n_iter = 50) {
  m  <- nrow(A); n <- ncol(A)
  q  <- matrix(rnorm(n), n, 1); q <- q/sqrt(sum(q^2))
  Q  <- q; alpha_v <- numeric(n_iter); beta_v <- numeric(n_iter)
  for (j in seq_len(n_iter)) {
    z <- t(A) %*% (A %*% q)
    a <- sum(q * z); z <- z - a*q
    if (j > 1) z <- z - beta_v[j-1] * Q[, j-1]
    b <- sqrt(sum(z^2))
    if (b < 1e-12) break
    q <- z/b
    alpha_v[j] <- a; beta_v[j] <- b
    if (j < n_iter) Q <- cbind(Q, q)
  }
  list(approx_singular_values = sqrt(pmax(alpha_v[1:j], 0)))
}

# ============================================================
# 7. INTERPOLATION
# ============================================================

cubic_spline <- function(x, y) {
  n <- length(x); h <- diff(x)
  # Natural spline: set up tridiagonal system
  rhs <- 6 * (diff(y) / h[-1] - diff(y) / h[-length(h)])
  A   <- diag(2 * (h[-1] + h[-length(h)]))
  for (i in seq_len(nrow(A)-1)) {
    A[i, i+1] <- h[i+1]; A[i+1, i] <- h[i+1]
  }
  m   <- c(0, solve(A, rhs), 0)
  list(x=x, y=y, m=m,
       predict = function(xi) {
         j <- findInterval(xi, x, rightmost.closed=TRUE)
         j <- pmax(1, pmin(j, n-1))
         dx <- xi - x[j]; hj <- h[j]
         a <- (x[j+1]-xi)/hj; b <- dx/hj
         a*y[j] + b*y[j+1] + ((a^3-a)*m[j] + (b^3-b)*m[j+1])*hj^2/6
       })
}

bilinear_interp <- function(z_mat, x_grid, y_grid, xi, yi) {
  jx <- findInterval(xi, x_grid); jx <- pmax(1, pmin(jx, length(x_grid)-1))
  jy <- findInterval(yi, y_grid); jy <- pmax(1, pmin(jy, length(y_grid)-1))
  tx <- (xi - x_grid[jx]) / (x_grid[jx+1] - x_grid[jx] + 1e-12)
  ty <- (yi - y_grid[jy]) / (y_grid[jy+1] - y_grid[jy] + 1e-12)
  (1-tx)*(1-ty)*z_mat[jx,jy] + tx*(1-ty)*z_mat[jx+1,jy] +
  (1-tx)*ty*z_mat[jx,jy+1] + tx*ty*z_mat[jx+1,jy+1]
}


# ============================================================
# ADDITIONAL: STOCHASTIC DIFFERENTIAL EQUATIONS
# ============================================================

euler_maruyama <- function(mu_fn, sigma_fn, x0, T_, n_steps, n_paths = 1000,
                            seed = 42) {
  set.seed(seed)
  dt   <- T_ / n_steps
  sqdt <- sqrt(dt)
  paths <- matrix(NA, n_steps + 1, n_paths)
  paths[1, ] <- x0
  for (i in seq_len(n_steps)) {
    t_i <- (i - 1) * dt
    x   <- paths[i, ]
    dW  <- rnorm(n_paths) * sqdt
    paths[i + 1, ] <- x + mu_fn(t_i, x) * dt + sigma_fn(t_i, x) * dW
  }
  list(paths = paths, t = seq(0, T_, by = dt),
       mean_path = rowMeans(paths), sd_path = apply(paths, 1, sd))
}

milstein_scheme <- function(mu_fn, sigma_fn, dsigma_fn, x0, T_,
                              n_steps, n_paths = 1000, seed = 42) {
  set.seed(seed)
  dt <- T_ / n_steps; sqdt <- sqrt(dt)
  paths <- matrix(NA, n_steps + 1, n_paths); paths[1, ] <- x0
  for (i in seq_len(n_steps)) {
    t_i <- (i - 1) * dt; x <- paths[i, ]
    dW  <- rnorm(n_paths) * sqdt
    s   <- sigma_fn(t_i, x); ds <- dsigma_fn(t_i, x)
    paths[i + 1, ] <- x + mu_fn(t_i, x)*dt + s*dW + 0.5*s*ds*(dW^2 - dt)
  }
  list(paths = paths, t = seq(0, T_, by = dt),
       mean_path = rowMeans(paths))
}

geometric_brownian_motion <- function(S0, mu, sigma, T_, n_steps, n_paths = 1000,
                                       seed = 42) {
  mu_fn    <- function(t, x) mu * x
  sigma_fn <- function(t, x) sigma * x
  res <- euler_maruyama(mu_fn, sigma_fn, S0, T_, n_steps, n_paths, seed)
  list(paths = res$paths, t = res$t,
       final_mean = mean(res$paths[nrow(res$paths), ]),
       final_sd   = sd(res$paths[nrow(res$paths), ]),
       analytical_mean = S0 * exp(mu * T_))
}

heston_simulation <- function(S0, V0, kappa, theta, sigma_v, rho,
                                r, T_, n_steps = 252, n_paths = 1000,
                                seed = 42) {
  set.seed(seed)
  dt <- T_ / n_steps; sqdt <- sqrt(dt)
  S  <- matrix(NA, n_steps + 1, n_paths); S[1, ] <- S0
  V  <- matrix(NA, n_steps + 1, n_paths); V[1, ] <- V0
  for (i in seq_len(n_steps)) {
    Z1 <- rnorm(n_paths); Z2 <- rnorm(n_paths)
    Zv <- Z1; Zs <- rho * Z1 + sqrt(1 - rho^2) * Z2
    v_cur <- pmax(V[i, ], 0)
    V[i+1,] <- pmax(v_cur + kappa*(theta - v_cur)*dt +
                    sigma_v*sqrt(v_cur)*sqdt*Zv, 0)
    S[i+1,] <- S[i,] * exp((r - v_cur/2)*dt + sqrt(v_cur)*sqdt*Zs)
  }
  list(S = S, V = V, t = seq(0, T_, by = dt),
       final_prices = S[nrow(S), ])
}
