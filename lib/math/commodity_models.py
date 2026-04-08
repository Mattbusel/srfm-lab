"""
commodity_models.py
-------------------
Energy and commodity quantitative models.

Covers:
  - Gibson-Schwartz two-factor model (spot + convenience yield)
  - Schwartz-Smith two-factor (short-term deviation + long-term equilibrium)
  - Gabillon model for oil
  - Convenience yield curve fitting (Nelson-Siegel adapted)
  - Seasonal adjustment via Fourier decomposition
  - Storage cost arbitrage / no-arb condition
  - Calendar spread fair value (cost of carry)
  - Volatility term structure (contango signal)
  - Crack spread model (refinery margin)
  - Mean-reverting commodity price model (OU with seasonality)
  - VaR for commodity portfolios with fat tails (t-distribution)
  - Energy demand elasticity
  - Natural gas storage injection/withdrawal seasonal pattern

Dependencies: numpy, scipy only.
"""

import numpy as np
from scipy.optimize import minimize, brentq
from scipy.stats import t as t_dist, norm
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# 1. Gibson-Schwartz Two-Factor Model
# ---------------------------------------------------------------------------

class GibsonSchwartzModel:
    """
    Gibson-Schwartz (1990) two-factor commodity model.

    State variables:
        S  = spot price  (log-normal)
        delta = convenience yield (mean-reverting OU)

    SDEs (risk-neutral):
        dS/S     = (r - delta) dt + sigma_S  dW_1
        d(delta) = kappa*(alpha - delta) dt + sigma_d dW_2
        dW_1 dW_2 = rho dt

    Parameters
    ----------
    kappa   : float  -- speed of mean reversion for convenience yield
    alpha   : float  -- long-run mean of convenience yield (risk-neutral)
    sigma_S : float  -- spot price volatility
    sigma_d : float  -- convenience yield volatility
    rho     : float  -- correlation between the two Brownian motions
    r       : float  -- risk-free rate (continuous)
    """

    def __init__(self, kappa: float, alpha: float,
                 sigma_S: float, sigma_d: float,
                 rho: float, r: float):
        self.kappa   = kappa
        self.alpha   = alpha
        self.sigma_S = sigma_S
        self.sigma_d = sigma_d
        self.rho     = rho
        self.r       = r

    def futures_price(self, S0: float, delta0: float, T: float) -> float:
        """
        Analytical futures price for maturity T.

        F(0,T) = S0 * exp(A(T) + B(T)*delta0)

        where B(T) = -(1 - exp(-kappa*T)) / kappa
        and A(T) captures drift, variance, and covariance terms.
        """
        k = self.kappa
        a = self.alpha
        sS = self.sigma_S
        sd = self.sigma_d
        rho = self.rho
        r   = self.r

        B = -(1.0 - np.exp(-k * T)) / k

        # Variance contribution
        var_term = (
            sS**2 * T
            + sd**2 / k**2 * (T + 2.0 / k * np.exp(-k * T)
                               - 1.0 / (2.0 * k) * np.exp(-2.0 * k * T)
                               - 3.0 / (2.0 * k))
            + 2.0 * rho * sS * sd / k * (T - (1.0 - np.exp(-k * T)) / k)
        )

        A = ((r - a + 0.5 * sd**2 / k**2 - rho * sS * sd / k) * (T + B)
             + (a - sd**2 / (2.0 * k**2)) * T
             - sd**2 * B**2 / (4.0 * k)
             + 0.5 * var_term)

        return S0 * np.exp(A + B * delta0)

    def simulate(self, S0: float, delta0: float, T: float,
                 n_steps: int = 252, n_paths: int = 10_000,
                 seed: int = 42) -> tuple:
        """
        Monte Carlo simulation of (S, delta) paths under risk-neutral measure.

        Returns
        -------
        S_paths     : ndarray (n_steps+1, n_paths)
        delta_paths : ndarray (n_steps+1, n_paths)
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        k   = self.kappa
        a   = self.alpha
        sS  = self.sigma_S
        sd  = self.sigma_d
        rho = self.rho
        r   = self.r

        S     = np.zeros((n_steps + 1, n_paths))
        delta = np.zeros((n_steps + 1, n_paths))
        S[0]     = S0
        delta[0] = delta0

        cov = np.array([[1.0, rho], [rho, 1.0]])
        L   = np.linalg.cholesky(cov)

        for i in range(n_steps):
            z = L @ rng.standard_normal((2, n_paths))
            z1, z2 = z[0], z[1]

            delta[i + 1] = (delta[i]
                            + k * (a - delta[i]) * dt
                            + sd * np.sqrt(dt) * z2)
            log_S = (np.log(S[i])
                     + (r - delta[i] - 0.5 * sS**2) * dt
                     + sS * np.sqrt(dt) * z1)
            S[i + 1] = np.exp(log_S)

        return S, delta

    def convenience_yield_from_futures(self, S0: float, F: float,
                                       T: float, r: float) -> float:
        """Back out implied convenience yield from observed futures price."""
        # Approximate: ln(F/S) = (r - delta)*T  => delta = r - ln(F/S)/T
        if T <= 0:
            raise ValueError("T must be positive")
        return r - np.log(F / S0) / T


# ---------------------------------------------------------------------------
# 2. Schwartz-Smith Two-Factor Model
# ---------------------------------------------------------------------------

class SchwartzSmithModel:
    """
    Schwartz-Smith (2000) two-factor model.

    log(S_t) = chi_t + xi_t

    chi_t : short-term deviation (mean-reverting)
    xi_t  : long-term equilibrium price level (GBM)

    Under physical measure:
        d(chi) = -kappa*chi dt + sigma_chi dW_chi
        d(xi)  = mu_xi dt + sigma_xi dW_xi
        corr(dW_chi, dW_xi) = rho

    Under risk-neutral measure, lambda_chi and lambda_xi are market prices of risk.
    """

    def __init__(self, kappa: float, mu_xi: float,
                 sigma_chi: float, sigma_xi: float,
                 rho: float,
                 lambda_chi: float = 0.0, lambda_xi: float = 0.0):
        self.kappa      = kappa
        self.mu_xi      = mu_xi
        self.sigma_chi  = sigma_chi
        self.sigma_xi   = sigma_xi
        self.rho        = rho
        self.lambda_chi = lambda_chi
        self.lambda_xi  = lambda_xi

    def futures_log_price(self, chi0: float, xi0: float, T: float) -> float:
        """
        E_Q[log F(0,T)] = A(T) + e^{-kappa*T} * chi0 + xi0

        A(T) = (mu_xi* - lambda_xi) T
               - (1-e^{-kappa T})/kappa * lambda_chi
               + 0.5 * Var[log F(0,T)]
        """
        k    = self.kappa
        s_c  = self.sigma_chi
        s_x  = self.sigma_xi
        rho  = self.rho
        mu_x = self.mu_xi
        l_c  = self.lambda_chi
        l_x  = self.lambda_xi

        eT = np.exp(-k * T)

        var = (s_c**2 / (2.0 * k) * (1.0 - eT**2)
               + s_x**2 * T
               + 2.0 * rho * s_c * s_x / k * (1.0 - eT))

        A = ((mu_x - l_x) * T
             - (1.0 - eT) / k * l_c
             + 0.5 * var)

        return A + eT * chi0 + xi0

    def futures_price(self, chi0: float, xi0: float, T: float) -> float:
        return np.exp(self.futures_log_price(chi0, xi0, T))

    def kalman_filter(self, log_futures: np.ndarray,
                      maturities: np.ndarray,
                      dt: float = 1.0 / 52.0) -> dict:
        """
        Kalman filter to estimate (chi, xi) from a panel of futures prices.

        log_futures : (n_obs, n_contracts) array of log futures prices
        maturities  : (n_contracts,) array of maturities in years
        dt          : time step between observations

        Returns dict with filtered states and log-likelihood.
        """
        k   = self.kappa
        s_c = self.sigma_chi
        s_x = self.sigma_xi
        rho = self.rho

        n_obs, n_c = log_futures.shape

        # State transition matrix (discrete time)
        F = np.array([[np.exp(-k * dt), 0.0],
                      [0.0,             1.0]])

        # Process noise covariance
        q11 = s_c**2 / (2.0 * k) * (1.0 - np.exp(-2.0 * k * dt))
        q12 = rho * s_c * s_x / k * (1.0 - np.exp(-k * dt))
        q22 = s_x**2 * dt
        Q = np.array([[q11, q12],
                      [q12, q22]])

        # Measurement matrices: log F_i = A(T_i) + [e^{-kT_i}, 1] * [chi, xi]
        def A(T):
            eT  = np.exp(-k * T)
            var = (s_c**2 / (2.0 * k) * (1.0 - eT**2)
                   + s_x**2 * T
                   + 2.0 * rho * s_c * s_x / k * (1.0 - eT))
            return ((self.mu_xi - self.lambda_xi) * T
                    - (1.0 - eT) / k * self.lambda_chi
                    + 0.5 * var)

        H = np.column_stack([np.exp(-k * maturities),
                             np.ones(n_c)])  # (n_c, 2)
        a_vec = np.array([A(T) for T in maturities])  # (n_c,)

        # Measurement noise (assume small iid)
        R = np.eye(n_c) * 1e-4

        # Initial state
        x   = np.zeros(2)
        P   = np.eye(2) * 0.1
        log_lik = 0.0
        states  = []

        for t in range(n_obs):
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # Update
            y      = log_futures[t] - (H @ x_pred + a_vec)
            S_mat  = H @ P_pred @ H.T + R
            K      = P_pred @ H.T @ np.linalg.inv(S_mat)
            x      = x_pred + K @ y
            P      = (np.eye(2) - K @ H) @ P_pred

            # Log-likelihood
            sign, logdet = np.linalg.slogdet(S_mat)
            log_lik += -0.5 * (n_c * np.log(2 * np.pi)
                                + logdet
                                + y @ np.linalg.inv(S_mat) @ y)
            states.append(x.copy())

        return {"states": np.array(states), "log_likelihood": log_lik}


# ---------------------------------------------------------------------------
# 3. Gabillon Model for Oil
# ---------------------------------------------------------------------------

class GabillonModel:
    """
    Gabillon (1991) two-factor oil model.

    F(t, T) = G(T) * exp(spot_component(t, T))

    where G(T) is a deterministic long-run forward curve and the stochastic
    component captures deviations.

    Simplified form with two factors: short-term (S) and long-term (L).

    log F(t,T) = log L_t + (log S_t - log L_t) * e^{-kappa*(T-t)}
                 + correction terms

    Parameters
    ----------
    kappa   : float -- speed of reversion of short-term to long-term
    sigma_S : float -- short-term log price volatility
    sigma_L : float -- long-term log price volatility
    rho     : float -- correlation
    """

    def __init__(self, kappa: float, sigma_S: float,
                 sigma_L: float, rho: float):
        self.kappa   = kappa
        self.sigma_S = sigma_S
        self.sigma_L = sigma_L
        self.rho     = rho

    def futures_price(self, S0: float, L0: float, T: float) -> float:
        """
        Gabillon futures price (no drift adjustment for simplicity).

        log F(0,T) = log L0 + (log S0 - log L0) * exp(-kappa*T)
        """
        eT = np.exp(-self.kappa * T)
        log_F = np.log(L0) + (np.log(S0) - np.log(L0)) * eT
        return np.exp(log_F)

    def futures_volatility(self, T: float) -> float:
        """
        Implied futures return volatility at maturity T.

        sigma_F(T)^2 = sigma_S^2 * e^{-2 kappa T}
                     + sigma_L^2 * (1 - e^{-kappa T})^2
                     + 2*rho*sigma_S*sigma_L * e^{-kappa T} * (1-e^{-kappa T})
        """
        k  = self.kappa
        sS = self.sigma_S
        sL = self.sigma_L
        r  = self.rho
        eT = np.exp(-k * T)

        var = (sS**2 * eT**2
               + sL**2 * (1.0 - eT)**2
               + 2.0 * r * sS * sL * eT * (1.0 - eT))
        return np.sqrt(max(var, 0.0))

    def fit_to_curve(self, maturities: np.ndarray,
                     observed_prices: np.ndarray,
                     S0: float) -> float:
        """
        Fit L0 such that the model matches the observed long-end forward price.
        Uses the last maturity as anchor.
        """
        T_long = maturities[-1]
        F_long = observed_prices[-1]
        # log F = log L0 + (log S0 - log L0)*exp(-k*T)
        eT = np.exp(-self.kappa * T_long)
        log_L0 = (np.log(F_long) - eT * np.log(S0)) / (1.0 - eT)
        return np.exp(log_L0)


# ---------------------------------------------------------------------------
# 4. Convenience Yield Curve (Nelson-Siegel adapted)
# ---------------------------------------------------------------------------

def nelson_siegel_convenience_yield(maturities: np.ndarray,
                                    beta0: float, beta1: float,
                                    beta2: float, tau: float) -> np.ndarray:
    """
    Nelson-Siegel parametrisation adapted for commodity convenience yield curve.

    cy(T) = beta0
           + beta1 * (1 - exp(-T/tau)) / (T/tau)
           + beta2 * ((1 - exp(-T/tau)) / (T/tau) - exp(-T/tau))

    Parameters
    ----------
    maturities : array of maturities (years)
    beta0      : long-run level
    beta1      : short-end loading
    beta2      : hump loading
    tau        : decay parameter (> 0)
    """
    x  = maturities / tau
    f1 = (1.0 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    return beta0 + beta1 * f1 + beta2 * f2


def fit_convenience_yield_curve(maturities: np.ndarray,
                                observed_cy: np.ndarray) -> dict:
    """
    Fit Nelson-Siegel parameters to observed convenience yield term structure.

    Returns dict with parameters and fitted values.
    """
    def residuals(params):
        b0, b1, b2, tau = params
        if tau <= 0:
            return 1e10
        fitted = nelson_siegel_convenience_yield(maturities, b0, b1, b2, tau)
        return np.sum((fitted - observed_cy)**2)

    x0     = [0.05, -0.02, 0.01, 1.0]
    bounds = [(-1, 1), (-1, 1), (-1, 1), (0.01, 20.0)]
    result = minimize(residuals, x0, method="L-BFGS-B", bounds=bounds)
    b0, b1, b2, tau = result.x
    fitted = nelson_siegel_convenience_yield(maturities, b0, b1, b2, tau)

    return {
        "beta0": b0, "beta1": b1, "beta2": b2, "tau": tau,
        "fitted": fitted,
        "rmse": np.sqrt(np.mean((fitted - observed_cy)**2)),
        "success": result.success,
    }


# ---------------------------------------------------------------------------
# 5. Seasonal Adjustment: Fourier-Based Decomposition
# ---------------------------------------------------------------------------

def fourier_seasonal_decomposition(prices: np.ndarray,
                                   freq: int = 12,
                                   n_harmonics: int = 3) -> dict:
    """
    Decompose commodity price series into trend + seasonal + residual
    using Fourier harmonics.

    Parameters
    ----------
    prices     : 1-D array of commodity prices (length N)
    freq       : seasonal period (12 = monthly, 52 = weekly, 365 = daily)
    n_harmonics: number of Fourier harmonics to use for seasonality

    Returns
    -------
    dict with keys: trend, seasonal, residual, amplitudes, phases
    """
    N = len(prices)
    t = np.arange(N, dtype=float)

    # --- Trend: linear via OLS ---
    X_trend = np.column_stack([np.ones(N), t])
    beta, _, _, _ = np.linalg.lstsq(X_trend, prices, rcond=None)
    trend = X_trend @ beta

    detrended = prices - trend

    # --- Seasonal: Fourier harmonics ---
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * h * t / freq))
        cols.append(np.sin(2.0 * np.pi * h * t / freq))
    X_seas = np.column_stack(cols)

    gamma, _, _, _ = np.linalg.lstsq(X_seas, detrended, rcond=None)
    seasonal = X_seas @ gamma

    # Extract amplitudes and phases
    amplitudes = []
    phases     = []
    for h in range(n_harmonics):
        a = gamma[2 * h]
        b = gamma[2 * h + 1]
        amplitudes.append(np.sqrt(a**2 + b**2))
        phases.append(np.arctan2(b, a))

    residual = detrended - seasonal

    return {
        "trend":      trend,
        "seasonal":   seasonal,
        "residual":   residual,
        "amplitudes": np.array(amplitudes),
        "phases":     np.array(phases),
        "beta_trend": beta,
    }


def seasonal_forecast(t_future: np.ndarray, beta_trend: np.ndarray,
                      amplitudes: np.ndarray, phases: np.ndarray,
                      freq: int = 12) -> np.ndarray:
    """
    Forecast seasonal + trend component for future time indices.
    """
    trend = beta_trend[0] + beta_trend[1] * t_future
    seas  = np.zeros_like(t_future, dtype=float)
    for h, (amp, phi) in enumerate(zip(amplitudes, phases), start=1):
        seas += amp * np.cos(2.0 * np.pi * h * t_future / freq - phi)
    return trend + seas


# ---------------------------------------------------------------------------
# 6. Storage Cost Arbitrage / No-Arb Condition
# ---------------------------------------------------------------------------

def storage_no_arb_forward(S: float, r: float, u: float,
                            cy: float, T: float) -> float:
    """
    No-arbitrage forward / futures price for a storable commodity.

    F = S * exp((r + u - cy) * T)

    Parameters
    ----------
    S   : spot price
    r   : risk-free rate (continuous)
    u   : storage cost rate (continuous, per unit of value)
    cy  : convenience yield (continuous)
    T   : time to maturity (years)
    """
    return S * np.exp((r + u - cy) * T)


def arbitrage_profit(S: float, F_mkt: float, r: float,
                     u: float, cy: float, T: float) -> float:
    """
    Cash-and-carry arbitrage profit (positive = buy spot, sell forward).
    Negative implies reverse cash-and-carry opportunity.
    """
    F_fair = storage_no_arb_forward(S, r, u, cy, T)
    return F_mkt - F_fair


def implied_convenience_yield(S: float, F: float, r: float,
                               u: float, T: float) -> float:
    """Implied convenience yield from market spot and futures prices."""
    return r + u - np.log(F / S) / T


# ---------------------------------------------------------------------------
# 7. Calendar Spread Fair Value
# ---------------------------------------------------------------------------

def calendar_spread_fair_value(S: float, r: float, u: float,
                                cy: float, T1: float, T2: float) -> float:
    """
    Fair value of calendar spread = F(T2) - F(T1).

    Uses cost-of-carry: F(T) = S * exp((r + u - cy)*T)

    Parameters
    ----------
    T1 < T2 : near / far maturities
    """
    carry = r + u - cy
    F1    = S * np.exp(carry * T1)
    F2    = S * np.exp(carry * T2)
    return F2 - F1


def roll_yield(F_near: float, F_far: float,
               T_near: float, T_far: float) -> float:
    """
    Annualised roll yield for rolling from near to far contract.

    roll_yield = -[ln(F_far/F_near)] / (T_far - T_near)

    Positive in backwardation, negative in contango.
    """
    return -np.log(F_far / F_near) / (T_far - T_near)


# ---------------------------------------------------------------------------
# 8. Volatility Term Structure (Samuelson effect / contango signal)
# ---------------------------------------------------------------------------

def samuelson_vol_term_structure(T: np.ndarray, sigma_spot: float,
                                 kappa: float, sigma_long: float) -> np.ndarray:
    """
    Samuelson effect: futures volatility declines with maturity.

    sigma_F(T) = sqrt(sigma_spot^2 * exp(-2*kappa*T) + sigma_long^2 * (1-exp(-kappa*T))^2)

    This is the Gabillon/Gibson-Schwartz implied vol term structure.
    """
    eT  = np.exp(-kappa * T)
    var = (sigma_spot**2 * eT**2
           + sigma_long**2 * (1.0 - eT)**2
           + 0.0)  # cross term omitted for single-factor version
    return np.sqrt(np.maximum(var, 0.0))


def contango_signal(front_vol: float, deferred_vol: float,
                    front_price: float, deferred_price: float) -> dict:
    """
    Compute contango / backwardation signal from vol term structure and prices.

    Returns
    -------
    dict with:
      price_slope     : (deferred - front) / front  (positive = contango)
      vol_slope       : front_vol - deferred_vol     (positive = Samuelson effect)
      contango_score  : composite signal
    """
    price_slope = (deferred_price - front_price) / front_price
    vol_slope   = front_vol - deferred_vol
    # Normalised composite: high price_slope + low vol_slope => strong contango
    contango_score = price_slope - 0.5 * vol_slope
    return {
        "price_slope":    price_slope,
        "vol_slope":      vol_slope,
        "contango_score": contango_score,
        "regime":         "contango" if price_slope > 0 else "backwardation",
    }


# ---------------------------------------------------------------------------
# 9. Crack Spread Model (Refinery Margin)
# ---------------------------------------------------------------------------

def crack_spread(crude_price: float, gasoline_price: float,
                 distillate_price: float,
                 crude_bbl: float = 3.0,
                 gasoline_bbl: float = 2.0,
                 distillate_bbl: float = 1.0) -> float:
    """
    3-2-1 crack spread: refinery margin.

    Spread = (2 * gasoline + 1 * distillate - 3 * crude) / 3

    All prices in $/bbl.  Returns $/bbl refinery margin.
    """
    return (gasoline_bbl * gasoline_price
            + distillate_bbl * distillate_price
            - crude_bbl * crude_price) / crude_bbl


def crack_spread_hedge_ratio(sigma_crude: float, sigma_product: float,
                              rho: float,
                              n_product_bbls: float = 2.0,
                              n_crude_bbls: float = 3.0) -> float:
    """
    Minimum-variance hedge ratio for refiner hedging product output
    against crude input.

    h* = rho * (sigma_product / sigma_crude) * (n_product_bbls / n_crude_bbls)
    """
    return rho * (sigma_product / sigma_crude) * (n_product_bbls / n_crude_bbls)


def fair_crack_spread(crude_futures: np.ndarray, gasoline_futures: np.ndarray,
                       distillate_futures: np.ndarray,
                       maturities: np.ndarray) -> np.ndarray:
    """
    Term structure of fair crack spreads across delivery months.
    Returns array of crack spreads for each maturity.
    """
    return (2.0 * gasoline_futures + 1.0 * distillate_futures
            - 3.0 * crude_futures) / 3.0


# ---------------------------------------------------------------------------
# 10. Mean-Reverting Commodity Price Model (Ornstein-Uhlenbeck + Seasonality)
# ---------------------------------------------------------------------------

class OUSeasonalModel:
    """
    Ornstein-Uhlenbeck model with Fourier seasonality for commodity prices.

    d(log P) = [kappa*(mu(t) - log P) ] dt + sigma dW

    where mu(t) = mu0 + sum_k [A_k cos(2*pi*k*t/freq) + B_k sin(2*pi*k*t/freq)]
    """

    def __init__(self, kappa: float, mu0: float, sigma: float,
                 seasonal_coefs: np.ndarray, freq: float = 12.0):
        """
        Parameters
        ----------
        kappa           : mean-reversion speed
        mu0             : long-run mean log price
        sigma           : diffusion coefficient
        seasonal_coefs  : array of [A1, B1, A2, B2, ...] Fourier coefficients
        freq            : seasonal period (same units as simulation time step)
        """
        self.kappa          = kappa
        self.mu0            = mu0
        self.sigma          = sigma
        self.seasonal_coefs = seasonal_coefs
        self.freq           = freq

    def seasonal_mean(self, t: float) -> float:
        """Evaluate mu(t) at time t."""
        mu = self.mu0
        n  = len(self.seasonal_coefs) // 2
        for h in range(1, n + 1):
            A = self.seasonal_coefs[2 * (h - 1)]
            B = self.seasonal_coefs[2 * (h - 1) + 1]
            mu += A * np.cos(2.0 * np.pi * h * t / self.freq)
            mu += B * np.sin(2.0 * np.pi * h * t / self.freq)
        return mu

    def simulate(self, P0: float, t0: float, T: float,
                 n_steps: int = 252, n_paths: int = 5_000,
                 seed: int = 0) -> np.ndarray:
        """
        Euler-Maruyama simulation of log price paths.

        Returns price paths array (n_steps+1, n_paths).
        """
        rng = np.random.default_rng(seed)
        dt  = T / n_steps
        k   = self.kappa
        s   = self.sigma

        log_P = np.full(n_paths, np.log(P0))
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0] = P0

        for i in range(n_steps):
            t_i   = t0 + i * dt
            mu_t  = self.seasonal_mean(t_i)
            dW    = rng.standard_normal(n_paths) * np.sqrt(dt)
            log_P = log_P + k * (mu_t - log_P) * dt + s * dW
            paths[i + 1] = np.exp(log_P)

        return paths

    def fit(self, log_prices: np.ndarray, dt: float = 1.0 / 12.0) -> dict:
        """
        Fit OU + seasonal model to observed log prices via OLS on discrete
        Euler approximation:

        log P_{t+1} - log P_t = kappa*(mu(t) - log P_t)*dt + eps

        => Regress (log P_{t+1} - log P_t) on [log P_t, seasonal basis]
        """
        N  = len(log_prices)
        t  = np.arange(N - 1, dtype=float)

        # Dependent variable
        y = np.diff(log_prices)

        # Design matrix: [-kappa*dt * log_P_t, kappa*mu_basis_j * dt, ...]
        n_harm = len(self.seasonal_coefs) // 2
        cols   = [-log_prices[:-1] * dt]  # coefficient = kappa
        for h in range(1, n_harm + 1):
            cols.append(dt * np.cos(2.0 * np.pi * h * t / self.freq))
            cols.append(dt * np.sin(2.0 * np.pi * h * t / self.freq))

        X = np.column_stack(cols)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        kappa_hat = beta[0]
        # Remaining coefficients are kappa * seasonal terms
        seas_coef = beta[1:] / kappa_hat if kappa_hat != 0 else beta[1:]

        resid = y - X @ beta
        sigma_hat = np.std(resid) / np.sqrt(dt)

        return {
            "kappa": kappa_hat,
            "seasonal_coefs": seas_coef,
            "sigma": sigma_hat,
        }


# ---------------------------------------------------------------------------
# 11. VaR for Commodity Portfolios with Fat Tails (t-distribution)
# ---------------------------------------------------------------------------

def fit_t_distribution(returns: np.ndarray) -> tuple:
    """
    Fit univariate Student-t distribution to return series via MLE.

    Returns (df, loc, scale).
    """
    result = t_dist.fit(returns)
    df, loc, scale = result
    return df, loc, scale


def commodity_var_t(weights: np.ndarray, returns: np.ndarray,
                    confidence: float = 0.99,
                    horizon: int = 1) -> dict:
    """
    Parametric VaR for a commodity portfolio assuming multivariate t returns.

    Approach:
      1. Compute portfolio returns from asset returns and weights.
      2. Fit univariate t-distribution to portfolio returns.
      3. VaR = -quantile at (1 - confidence) level, scaled by sqrt(horizon).

    Parameters
    ----------
    weights    : (n_assets,) portfolio weights
    returns    : (n_obs, n_assets) return matrix
    confidence : VaR confidence level (e.g. 0.99)
    horizon    : holding period in days

    Returns
    -------
    dict with VaR, CVaR (Expected Shortfall), df, loc, scale
    """
    port_ret = returns @ weights

    df, loc, scale = fit_t_distribution(port_ret)

    alpha   = 1.0 - confidence
    var_1d  = -t_dist.ppf(alpha, df=df, loc=loc, scale=scale)

    # CVaR for t-distribution
    # ES = loc + scale * [t_pdf(t_alpha, df) / alpha] * (df + t_alpha**2) / (df - 1)
    t_alpha = t_dist.ppf(alpha, df=df)
    if df > 1:
        es_std = (t_dist.pdf(t_alpha, df=df) / alpha
                  * (df + t_alpha**2) / (df - 1.0))
        cvar_1d = -(loc + scale * (-es_std))
    else:
        cvar_1d = np.nan

    return {
        "VaR":   var_1d * np.sqrt(horizon),
        "CVaR":  cvar_1d * np.sqrt(horizon),
        "df":    df,
        "loc":   loc,
        "scale": scale,
        "port_returns": port_ret,
    }


def historical_var(port_returns: np.ndarray, confidence: float = 0.99,
                   horizon: int = 1) -> float:
    """Historical simulation VaR (non-parametric)."""
    alpha = 1.0 - confidence
    return -np.percentile(port_returns, alpha * 100) * np.sqrt(horizon)


def delta_normal_var(weights: np.ndarray, cov_matrix: np.ndarray,
                     confidence: float = 0.99, horizon: int = 1) -> float:
    """Delta-normal (variance-covariance) VaR for commodity portfolio."""
    port_var = weights @ cov_matrix @ weights
    port_std = np.sqrt(port_var)
    z = norm.ppf(confidence)
    return port_std * z * np.sqrt(horizon)


# ---------------------------------------------------------------------------
# 12. Energy Demand Elasticity
# ---------------------------------------------------------------------------

def price_elasticity_of_demand(pct_change_quantity: float,
                                pct_change_price: float) -> float:
    """
    Price elasticity of demand: epsilon = %dQ / %dP.

    Typically negative for normal goods (inelastic for energy).
    """
    if pct_change_price == 0:
        return np.inf
    return pct_change_quantity / pct_change_price


def log_log_demand_model(log_prices: np.ndarray,
                          log_quantities: np.ndarray,
                          income: np.ndarray = None) -> dict:
    """
    Estimate constant-elasticity demand model via OLS:

    log(Q) = alpha + epsilon * log(P) [+ beta * log(I)] + e

    Parameters
    ----------
    log_prices     : log prices time series
    log_quantities : log quantity (demand) time series
    income         : optional log income/GDP series

    Returns
    -------
    dict with elasticity, intercept, r_squared
    """
    N = len(log_prices)
    if income is not None:
        X = np.column_stack([np.ones(N), log_prices, income])
    else:
        X = np.column_stack([np.ones(N), log_prices])

    beta, _, _, _ = np.linalg.lstsq(X, log_quantities, rcond=None)
    y_hat   = X @ beta
    ss_res  = np.sum((log_quantities - y_hat)**2)
    ss_tot  = np.sum((log_quantities - np.mean(log_quantities))**2)
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    result = {"intercept": beta[0], "price_elasticity": beta[1], "r_squared": r2}
    if income is not None:
        result["income_elasticity"] = beta[2]
    return result


def demand_curve(price_grid: np.ndarray, intercept: float,
                 elasticity: float, base_price: float = 1.0) -> np.ndarray:
    """
    Evaluate constant-elasticity demand curve.

    Q(P) = exp(intercept) * (P / base_price)^elasticity
    """
    return np.exp(intercept) * (price_grid / base_price) ** elasticity


# ---------------------------------------------------------------------------
# 13. Natural Gas Storage: Injection / Withdrawal Seasonal Pattern
# ---------------------------------------------------------------------------

class NaturalGasStorageModel:
    """
    Seasonal injection/withdrawal model for natural gas storage.

    Storage levels follow a sinusoidal seasonal pattern driven by
    heating degree days (winter demand) and cooling degree days (summer demand).

    Storage_t = base_storage + amplitude * cos(2*pi*(t - peak_month)/12) + noise

    Injection season: April–October (months 4–10)
    Withdrawal season: November–March (months 11–3)
    """

    def __init__(self, base_storage: float = 2500.0,
                 amplitude: float = 1000.0,
                 peak_month: float = 10.5,
                 sigma_noise: float = 50.0):
        """
        Parameters (storage in Bcf)
        ----------
        base_storage : average storage level
        amplitude    : seasonal amplitude
        peak_month   : month when storage is at seasonal peak (~late Oct)
        sigma_noise  : weekly noise standard deviation
        """
        self.base_storage = base_storage
        self.amplitude    = amplitude
        self.peak_month   = peak_month
        self.sigma_noise  = sigma_noise

    def expected_storage(self, month: np.ndarray) -> np.ndarray:
        """
        Expected storage level for given month (1=Jan, ..., 12=Dec).
        """
        return (self.base_storage
                + self.amplitude * np.cos(
                    2.0 * np.pi * (month - self.peak_month) / 12.0))

    def injection_rate(self, month: int, current_storage: float,
                       max_storage: float = 4000.0) -> float:
        """
        Estimate weekly injection rate (Bcf/week) based on season.
        Positive = injection, negative = withdrawal.
        """
        expected = float(self.expected_storage(np.array([month]))[0])
        gap      = expected - current_storage
        # Sigmoid-like response: larger gap => faster injection/withdrawal
        rate     = gap / 4.0 * np.tanh(abs(gap) / (max_storage * 0.1))
        return np.clip(rate, -30.0, 30.0)

    def simulate(self, n_weeks: int = 520,
                 initial_storage: float = None,
                 seed: int = 42) -> dict:
        """
        Simulate weekly storage levels over n_weeks (10 years default).

        Returns
        -------
        dict with storage, months, net_flows, week_index
        """
        rng = np.random.default_rng(seed)
        if initial_storage is None:
            initial_storage = self.base_storage

        storage   = np.zeros(n_weeks + 1)
        net_flows = np.zeros(n_weeks)
        storage[0] = initial_storage

        for w in range(n_weeks):
            month   = (w % 52) / 52.0 * 12.0 + 1.0
            rate    = self.injection_rate(month, storage[w])
            noise   = rng.normal(0.0, self.sigma_noise / np.sqrt(4.33))
            storage[w + 1] = np.clip(storage[w] + rate + noise, 0.0, 4500.0)
            net_flows[w]   = rate

        weeks  = np.arange(n_weeks + 1)
        months = (weeks % 52) / 52.0 * 12.0 + 1.0

        return {
            "storage":   storage,
            "months":    months,
            "net_flows": net_flows,
            "weeks":     weeks,
        }

    def working_gas_percentile(self, current_storage: float,
                                historical_storage: np.ndarray) -> float:
        """
        Working gas storage percentile relative to historical range.
        Returns value between 0 (5-year low) and 1 (5-year high).
        """
        return float(np.mean(historical_storage <= current_storage))

    def storage_premium(self, current_storage: float,
                        month: int,
                        base_price: float,
                        price_sensitivity: float = 0.002) -> float:
        """
        Estimate price premium/discount based on storage deviation from norm.

        premium = -price_sensitivity * (current - expected) / base_storage

        Tight storage => positive premium (higher prices).
        """
        expected_level = float(self.expected_storage(np.array([month]))[0])
        deviation      = current_storage - expected_level
        return -price_sensitivity * deviation / self.base_storage * base_price


# ---------------------------------------------------------------------------
# Convenience: Composite strip pricing
# ---------------------------------------------------------------------------

def strip_price(futures_prices: np.ndarray) -> float:
    """
    Average of a futures strip (e.g. calendar year strip).
    """
    return float(np.mean(futures_prices))


def weighted_strip_price(futures_prices: np.ndarray,
                          weights: np.ndarray) -> float:
    """
    Volume-weighted strip price.
    weights should sum to 1.
    """
    w = weights / np.sum(weights)
    return float(np.dot(w, futures_prices))


# ---------------------------------------------------------------------------
# Utility: Forward curve from spot and basis
# ---------------------------------------------------------------------------

def build_forward_curve(spot: float, r: float, u: float,
                         convenience_yields: np.ndarray,
                         maturities: np.ndarray) -> np.ndarray:
    """
    Build full forward curve using cost-of-carry with a convenience yield curve.

    F(T_i) = S * exp((r + u - cy(T_i)) * T_i)
    """
    return spot * np.exp((r + u - convenience_yields) * maturities)


def implied_storage_cost(spot: float, forward: float, r: float,
                          cy: float, T: float) -> float:
    """Back out implied storage cost from observed spot and forward."""
    return np.log(forward / spot) / T - r + cy
