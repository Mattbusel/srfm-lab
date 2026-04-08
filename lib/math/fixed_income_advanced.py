"""
fixed_income_advanced.py
------------------------
Advanced fixed income mathematics.

Covers:
  - Full bond price from YTM (semi-annual and continuous conventions)
  - Modified / Macaulay / Effective duration and convexity
  - Key rate durations (2y, 5y, 10y, 30y vertex sensitivities)
  - OAS via binomial interest rate tree
  - Callable bond pricing
  - MBS: PSA prepayment model, duration
  - Z-spread computation
  - Yield curve bootstrap from Treasuries
  - DV01, PVBP, BPV
  - Interest rate swap pricing
  - Cross-currency basis swap
  - Bond futures CTD selection
  - Inflation-linked bond (TIPS)
  - Repo carry
  - Duration-neutral pairs / hedge ratio

Dependencies: numpy, scipy only.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# 1. Bond Price from Yield to Maturity
# ---------------------------------------------------------------------------

def bond_price_from_ytm(face: float, coupon_rate: float, ytm: float,
                         n_periods: int, freq: int = 2) -> float:
    """
    Full price of a coupon bond from YTM.

    Parameters
    ----------
    face        : face (par) value
    coupon_rate : annual coupon rate (decimal, e.g. 0.05 for 5%)
    ytm         : yield to maturity (annual, decimal)
    n_periods   : number of remaining coupon periods
    freq        : coupon frequency per year (2 = semi-annual, 1 = annual)

    Returns
    -------
    Full (dirty) price.
    """
    c  = face * coupon_rate / freq      # coupon per period
    y  = ytm / freq                     # yield per period
    t  = np.arange(1, n_periods + 1)

    pv_coupons   = c * np.sum(1.0 / (1.0 + y) ** t)
    pv_principal = face / (1.0 + y) ** n_periods

    return pv_coupons + pv_principal


def bond_ytm(price: float, face: float, coupon_rate: float,
             n_periods: int, freq: int = 2,
             tol: float = 1e-10) -> float:
    """
    Compute YTM from price via Brent's method.
    """
    def obj(ytm):
        return bond_price_from_ytm(face, coupon_rate, ytm, n_periods, freq) - price

    return brentq(obj, -0.9999, 20.0, xtol=tol)


def accrued_interest(face: float, coupon_rate: float,
                     days_since_last_coupon: int,
                     days_in_coupon_period: int,
                     freq: int = 2) -> float:
    """
    Accrued interest (30/360 or actual/actual depending on inputs).
    """
    coupon = face * coupon_rate / freq
    return coupon * days_since_last_coupon / days_in_coupon_period


def clean_price(dirty_price: float, accrued: float) -> float:
    return dirty_price - accrued


# ---------------------------------------------------------------------------
# 2. Duration and Convexity
# ---------------------------------------------------------------------------

def macaulay_duration(face: float, coupon_rate: float, ytm: float,
                       n_periods: int, freq: int = 2) -> float:
    """
    Macaulay duration in years: weighted average time to cash flows.
    """
    c = face * coupon_rate / freq
    y = ytm / freq
    t = np.arange(1, n_periods + 1)

    cf    = c * np.ones(n_periods)
    cf[-1] += face                          # add principal at final period

    pv_cf  = cf / (1.0 + y) ** t
    price  = np.sum(pv_cf)
    mac_d  = np.sum(t * pv_cf) / price / freq   # convert periods to years

    return mac_d


def modified_duration(face: float, coupon_rate: float, ytm: float,
                       n_periods: int, freq: int = 2) -> float:
    """
    Modified duration = Macaulay duration / (1 + ytm/freq).
    """
    mac = macaulay_duration(face, coupon_rate, ytm, n_periods, freq)
    return mac / (1.0 + ytm / freq)


def convexity(face: float, coupon_rate: float, ytm: float,
               n_periods: int, freq: int = 2) -> float:
    """
    Bond convexity (annualised).

    C = [1 / (P * (1+y/freq)^2)] * sum_t [t*(t+1)*CF_t / (1+y/freq)^t]
    """
    c = face * coupon_rate / freq
    y = ytm / freq
    t = np.arange(1, n_periods + 1)

    cf     = c * np.ones(n_periods)
    cf[-1] += face

    price = bond_price_from_ytm(face, coupon_rate, ytm, n_periods, freq)

    weight = t * (t + 1.0) * cf / (1.0 + y) ** (t + 2.0)
    return np.sum(weight) / price / freq**2


def effective_duration(price_up: float, price_down: float,
                        price_0: float, delta_y: float) -> float:
    """
    Effective (option-adjusted) duration via finite difference.

    D_eff = (P_down - P_up) / (2 * P_0 * delta_y)
    """
    return (price_down - price_up) / (2.0 * price_0 * delta_y)


def effective_convexity(price_up: float, price_down: float,
                         price_0: float, delta_y: float) -> float:
    """
    Effective convexity via finite difference.

    C_eff = (P_up + P_down - 2*P_0) / (P_0 * delta_y^2)
    """
    return (price_up + price_down - 2.0 * price_0) / (price_0 * delta_y**2)


def duration_price_approx(price: float, mod_dur: float,
                            conv: float, dy: float) -> float:
    """
    Duration-convexity price approximation for a yield change dy.

    dP/P ≈ -D_mod * dy + 0.5 * C * dy^2
    """
    return price * (-mod_dur * dy + 0.5 * conv * dy**2)


# ---------------------------------------------------------------------------
# 3. Key Rate Durations
# ---------------------------------------------------------------------------

def key_rate_duration(cash_flows: np.ndarray, times: np.ndarray,
                       zero_rates: np.ndarray,
                       key_tenors: np.ndarray = np.array([2., 5., 10., 30.]),
                       shift: float = 0.0001) -> dict:
    """
    Key rate durations at specified vertices.

    Procedure: shift each key rate by `shift` bps independently,
    reprice bond using piecewise-linear interpolated zero curve,
    compute price sensitivity.

    Parameters
    ----------
    cash_flows  : array of cash flows
    times       : array of payment times (years)
    zero_rates  : spot zero rates at `times` (matching tenor to key vertices
                  via interpolation)
    key_tenors  : tenor vertices for key rate shifts
    shift       : parallel shift applied at each vertex (in rate units)

    Returns
    -------
    dict mapping tenor -> KRD (in years)
    """
    def price_with_curve(rates):
        disc = np.exp(-rates * times)
        return float(np.sum(cash_flows * disc))

    # Interpolate base curve to cash flow times
    max_time = max(np.max(times), np.max(key_tenors))
    all_tenors = np.unique(np.concatenate([key_tenors, [0.0]]))
    # Build base rates at key tenors by interpolating zero_rates at times
    interp_fn  = interp1d(times, zero_rates, kind="linear",
                           fill_value="extrapolate")
    base_rates = interp_fn(times)
    P0 = price_with_curve(base_rates)

    krd = {}
    for k_tenor in key_tenors:
        # Shift key rate at k_tenor: use hat function (tent) centred at k_tenor
        # Width = neighbouring key tenors
        idx = np.searchsorted(key_tenors, k_tenor)
        lo  = key_tenors[idx - 1] if idx > 0 else 0.0
        hi  = key_tenors[idx + 1] if idx < len(key_tenors) - 1 else k_tenor * 2.0

        # Build tent function weights at cash flow times
        tent = np.zeros_like(times)
        for i, t in enumerate(times):
            if lo <= t <= k_tenor:
                tent[i] = (t - lo) / (k_tenor - lo) if k_tenor > lo else 1.0
            elif k_tenor < t <= hi:
                tent[i] = (hi - t) / (hi - k_tenor) if hi > k_tenor else 1.0

        rates_up   = base_rates + tent * shift
        rates_down = base_rates - tent * shift

        P_up   = price_with_curve(rates_up)
        P_down = price_with_curve(rates_down)

        krd[k_tenor] = (P_down - P_up) / (2.0 * P0 * shift)

    return krd


# ---------------------------------------------------------------------------
# 4. OAS via Binomial Interest Rate Tree
# ---------------------------------------------------------------------------

def build_binomial_rate_tree(short_rate: float, volatility: float,
                              n_steps: int, dt: float) -> np.ndarray:
    """
    Ho-Lee binomial rate tree (constant volatility, log-normal increments).

    r_{i,j} = r_0 * exp(2 * j * sigma * sqrt(dt) - sigma^2 * i * dt)

    where j = 0, 1, ..., i (number of up moves at step i).
    Returns 2-D array (n_steps+1, n_steps+1) where [i, j] is rate at step i,
    j up-moves.
    """
    tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            tree[i, j] = short_rate * np.exp(
                (2.0 * j - i) * volatility * np.sqrt(dt)
            )
    return tree


def oas_binomial(cash_flows: np.ndarray, maturities: np.ndarray,
                 market_price: float,
                 short_rate: float, vol: float,
                 dt: float = 0.5,
                 tol: float = 1e-6) -> float:
    """
    Compute Option-Adjusted Spread (OAS) via binomial tree.

    The OAS s is added to each node rate such that the theoretical price
    matches market_price.

    Parameters
    ----------
    cash_flows  : cash flows at each maturity (e.g. coupons + par)
    maturities  : payment times in years
    market_price: observed market price
    short_rate  : current short rate
    vol         : rate volatility for the tree
    dt          : time step (years)
    """
    n_steps = int(np.max(maturities) / dt)
    tree    = build_binomial_rate_tree(short_rate, vol, n_steps, dt)

    def price_with_oas(s):
        # Backward induction with spread s added to rates
        prices = np.zeros((n_steps + 1, n_steps + 1))

        # Add cash flows at final step
        for j in range(n_steps + 1):
            for k, T in enumerate(maturities):
                step = int(round(T / dt))
                if step == n_steps:
                    prices[n_steps, j] += cash_flows[k]

        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                r   = tree[i, j] + s
                pv  = 0.5 * (prices[i + 1, j] + prices[i + 1, j + 1])
                pv /= (1.0 + r * dt)
                # Add any cash flow due at this step
                for k, T in enumerate(maturities):
                    step = int(round(T / dt))
                    if step == i:
                        pv += cash_flows[k]
                prices[i, j] = pv

        return prices[0, 0]

    obj = lambda s: price_with_oas(s) - market_price
    try:
        oas = brentq(obj, -0.50, 2.00, xtol=tol)
    except ValueError:
        oas = np.nan

    return oas


# ---------------------------------------------------------------------------
# 5. Callable Bond Pricing
# ---------------------------------------------------------------------------

def callable_bond_price(straight_price: float, call_option_value: float) -> float:
    """
    Callable bond price = Straight bond price - Call option value.

    The call is held by the issuer; investors are short the call.
    """
    return straight_price - call_option_value


def call_option_value_binomial(face: float, coupon_rate: float,
                                call_price: float,
                                short_rate: float, vol: float,
                                call_dates: np.ndarray,
                                maturity: float,
                                freq: int = 2,
                                dt: float = 0.5) -> float:
    """
    Value issuer's call option on a bond via binomial tree.

    call_dates : array of call exercise dates (years)
    call_price : call strike price (e.g. 100.0 for par call)
    """
    n_steps = int(maturity / dt)
    tree    = build_binomial_rate_tree(short_rate, vol, n_steps, dt)
    c       = face * coupon_rate * dt   # coupon per period

    call_steps = set(int(round(d / dt)) for d in call_dates)

    # Node price array (straight bond with coupon)
    prices = np.zeros((n_steps + 1, n_steps + 1))
    prices[n_steps, :] = face  # par at maturity

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            r  = tree[i, j]
            pv = (0.5 * (prices[i + 1, j] + prices[i + 1, j + 1]) + c)
            pv /= (1.0 + r * dt)
            prices[i, j] = pv

    straight = prices[0, 0]

    # Now price the callable bond (issuer exercises when bond > call_price)
    prices_call = np.zeros((n_steps + 1, n_steps + 1))
    prices_call[n_steps, :] = face

    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            r  = tree[i, j]
            pv = (0.5 * (prices_call[i + 1, j] + prices_call[i + 1, j + 1]) + c)
            pv /= (1.0 + r * dt)
            if i in call_steps:
                pv = min(pv, call_price)     # issuer calls if above call price
            prices_call[i, j] = pv

    callable_price = prices_call[0, 0]
    return straight - callable_price


# ---------------------------------------------------------------------------
# 6. MBS: PSA Prepayment Model and Duration
# ---------------------------------------------------------------------------

def psa_cpr(month: int, psa_speed: float = 100.0) -> float:
    """
    PSA (Public Securities Association) standard prepayment model.

    CPR(m) = min(6% * m/30, 6%) * psa_speed/100

    Parameters
    ----------
    month     : loan age in months (1-based)
    psa_speed : PSA speed as percentage of baseline (100 = standard)
    """
    baseline_cpr = min(0.06 * month / 30.0, 0.06)
    return baseline_cpr * psa_speed / 100.0


def cpr_to_smm(cpr: float) -> float:
    """Convert annual CPR to monthly Single Monthly Mortality (SMM)."""
    return 1.0 - (1.0 - cpr) ** (1.0 / 12.0)


def mbs_cash_flows(balance: float, coupon_rate: float, wac: float,
                   wam: int, psa_speed: float = 100.0) -> dict:
    """
    Generate MBS cash flow schedule given PSA prepayment assumption.

    Parameters
    ----------
    balance     : outstanding pool balance
    coupon_rate : pass-through coupon rate (annual)
    wac         : weighted-average coupon of underlying mortgages (annual)
    wam         : weighted-average maturity in months
    psa_speed   : PSA prepayment speed

    Returns
    -------
    dict with arrays: scheduled_principal, prepayments, interest, total_cf, balance
    """
    monthly_rate_wac    = wac / 12.0
    monthly_rate_coupon = coupon_rate / 12.0

    outstanding = np.zeros(wam + 1)
    sched_prin  = np.zeros(wam)
    prepays     = np.zeros(wam)
    interest    = np.zeros(wam)
    total_cf    = np.zeros(wam)
    outstanding[0] = balance

    for m in range(1, wam + 1):
        bal = outstanding[m - 1]
        if bal <= 0:
            break

        # Scheduled payment based on WAC
        pmt = bal * monthly_rate_wac / (1.0 - (1.0 + monthly_rate_wac) ** (-(wam - m + 1)))

        sched_int  = bal * monthly_rate_wac
        sched_p    = pmt - sched_int

        # Prepayment
        cpr = psa_cpr(m, psa_speed)
        smm = cpr_to_smm(cpr)
        prepay = smm * (bal - sched_p)

        total_principal = sched_p + prepay
        interest[m - 1] = bal * monthly_rate_coupon    # pass-through coupon
        sched_prin[m - 1] = sched_p
        prepays[m - 1]    = prepay
        total_cf[m - 1]   = interest[m - 1] + total_principal
        outstanding[m]    = bal - total_principal

    return {
        "scheduled_principal": sched_prin,
        "prepayments":         prepays,
        "interest":            interest,
        "total_cash_flow":     total_cf,
        "balance":             outstanding,
        "months":              np.arange(1, wam + 1),
    }


def mbs_price(balance: float, coupon_rate: float, wac: float,
              wam: int, discount_rate: float,
              psa_speed: float = 100.0) -> float:
    """Price an MBS by discounting PSA-modelled cash flows."""
    cfs = mbs_cash_flows(balance, coupon_rate, wac, wam, psa_speed)
    t   = cfs["months"] / 12.0
    return float(np.sum(cfs["total_cash_flow"] * np.exp(-discount_rate * t)))


def mbs_duration(balance: float, coupon_rate: float, wac: float,
                 wam: int, discount_rate: float,
                 psa_speed: float = 100.0, shift: float = 0.0001) -> float:
    """
    Effective duration of MBS (including prepayment option) via finite difference.
    Note: PSA speed is assumed to remain constant (static model).
    """
    P0 = mbs_price(balance, coupon_rate, wac, wam, discount_rate, psa_speed)
    Pu = mbs_price(balance, coupon_rate, wac, wam, discount_rate + shift, psa_speed)
    Pd = mbs_price(balance, coupon_rate, wac, wam, discount_rate - shift, psa_speed)
    return (Pd - Pu) / (2.0 * P0 * shift)


# ---------------------------------------------------------------------------
# 7. Z-Spread Computation
# ---------------------------------------------------------------------------

def z_spread(cash_flows: np.ndarray, times: np.ndarray,
             spot_rates: np.ndarray, market_price: float,
             tol: float = 1e-8) -> float:
    """
    Z-spread: constant spread added to the spot rate curve such that
    the sum of discounted cash flows equals the market price.

    Parameters
    ----------
    cash_flows   : array of cash flows
    times        : cash flow payment times (years)
    spot_rates   : spot zero rates at `times` (interpolated or matched)
    market_price : observed clean price
    """
    def price_with_spread(z):
        disc = np.exp(-(spot_rates + z) * times)
        return float(np.sum(cash_flows * disc))

    obj = lambda z: price_with_spread(z) - market_price
    try:
        return brentq(obj, -1.0, 5.0, xtol=tol)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# 8. Yield Curve Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_zero_curve(instruments: list) -> dict:
    """
    Bootstrap spot zero curve from a list of Treasury instruments.

    Each instrument is a dict:
        {
          "maturity": float (years),
          "coupon_rate": float (annual, 0 for bills),
          "price": float (dirty price),
          "freq": int (coupon frequency, 1 or 2; 0 for discount instrument)
        }

    Returns dict with "maturities" and "zero_rates" arrays.

    Algorithm: exact bootstrap using previously solved zero rates to
    strip each successive coupon bond.
    """
    instruments = sorted(instruments, key=lambda x: x["maturity"])
    maturities  = []
    zero_rates  = []

    for inst in instruments:
        T    = inst["maturity"]
        cr   = inst["coupon_rate"]
        P    = inst["price"]
        freq = inst.get("freq", 2)

        if cr == 0 or freq == 0:
            # Zero-coupon / discount bill
            z = -np.log(P / 100.0) / T
            maturities.append(T)
            zero_rates.append(z)
        else:
            # Coupon bond: strip known coupons using existing zeros
            c     = 100.0 * cr / freq
            n     = int(round(T * freq))
            times = np.arange(1, n + 1) / freq

            # Interpolate existing zero curve
            if len(maturities) >= 2:
                interp = interp1d(maturities, zero_rates,
                                   kind="linear", fill_value="extrapolate")
            elif len(maturities) == 1:
                interp = lambda t: np.full_like(np.atleast_1d(t),
                                                 zero_rates[0], dtype=float)
            else:
                interp = lambda t: np.zeros_like(np.atleast_1d(t), dtype=float)

            # PV of intermediate coupons
            known_pv = 0.0
            for ti in times[:-1]:
                z_i = float(interp(np.array([ti]))[0])
                known_pv += c * np.exp(-z_i * ti)

            # Solve for final zero rate
            final_cf = c + 100.0         # coupon + par
            disc_T   = (P - known_pv) / final_cf
            if disc_T <= 0:
                raise ValueError(f"Bootstrap failed at T={T}: negative discount")

            z_T = -np.log(disc_T) / T
            maturities.append(T)
            zero_rates.append(z_T)

    return {
        "maturities": np.array(maturities),
        "zero_rates":  np.array(zero_rates),
    }


def spot_to_forward(maturities: np.ndarray,
                    zero_rates: np.ndarray) -> np.ndarray:
    """
    Convert spot zero rates to instantaneous forward rates.

    f(t) = d[t * z(t)] / dt  ≈  (z2*T2 - z1*T1) / (T2 - T1)
    """
    forwards = np.zeros(len(maturities) - 1)
    for i in range(len(maturities) - 1):
        T1, T2 = maturities[i], maturities[i + 1]
        z1, z2 = zero_rates[i], zero_rates[i + 1]
        forwards[i] = (z2 * T2 - z1 * T1) / (T2 - T1)
    return forwards


def discount_factor(zero_rate: float, T: float) -> float:
    """Continuous compounding discount factor."""
    return np.exp(-zero_rate * T)


# ---------------------------------------------------------------------------
# 9. DV01, PVBP, BPV
# ---------------------------------------------------------------------------

def dv01(face: float, coupon_rate: float, ytm: float,
          n_periods: int, freq: int = 2) -> float:
    """
    DV01 (Dollar Value of 01): price change for 1bp (0.0001) parallel shift.

    DV01 = -dP/dy * 0.0001  ≈  Modified Duration * Price * 0.0001
    """
    P0 = bond_price_from_ytm(face, coupon_rate, ytm, n_periods, freq)
    P1 = bond_price_from_ytm(face, coupon_rate, ytm + 0.0001, n_periods, freq)
    return P0 - P1


def pvbp(cash_flows: np.ndarray, times: np.ndarray,
          zero_rates: np.ndarray) -> float:
    """
    Price Value of a Basis Point for a general set of cash flows.
    Uses the full zero curve rather than a single YTM.

    PVBP = P(z) - P(z + 0.0001)
    """
    P0 = float(np.sum(cash_flows * np.exp(-zero_rates * times)))
    P1 = float(np.sum(cash_flows * np.exp(-(zero_rates + 0.0001) * times)))
    return P0 - P1


def bpv(face: float, mod_duration: float, ytm: float,
         n_periods: int, freq: int = 2) -> float:
    """
    Basis Point Value = Modified Duration * Full Price * 0.0001.
    """
    price = bond_price_from_ytm(face, 0.0, ytm, n_periods, freq)  # placeholder
    return mod_duration * price * 0.0001


# ---------------------------------------------------------------------------
# 10. Interest Rate Swap Pricing
# ---------------------------------------------------------------------------

def swap_fixed_leg_pv(notional: float, fixed_rate: float,
                       payment_times: np.ndarray,
                       discount_factors: np.ndarray,
                       day_count_fractions: np.ndarray) -> float:
    """
    PV of fixed leg of an interest rate swap.

    PV_fixed = notional * fixed_rate * sum_i(alpha_i * DF_i)

    Parameters
    ----------
    payment_times      : payment dates (years)
    discount_factors   : discount factors at payment times
    day_count_fractions: alpha_i (e.g. 0.5 for semi-annual)
    """
    return notional * fixed_rate * float(
        np.sum(day_count_fractions * discount_factors)
    )


def swap_float_leg_pv(notional: float, payment_times: np.ndarray,
                       discount_factors: np.ndarray,
                       day_count_fractions: np.ndarray,
                       forward_rates: np.ndarray) -> float:
    """
    PV of floating leg of an interest rate swap.

    PV_float = notional * sum_i(alpha_i * f_i * DF_i)
    """
    return notional * float(
        np.sum(day_count_fractions * forward_rates * discount_factors)
    )


def swap_par_rate(payment_times: np.ndarray,
                   discount_factors: np.ndarray,
                   day_count_fractions: np.ndarray,
                   forward_rates: np.ndarray) -> float:
    """
    Fair (par) fixed rate for a new swap: rate that sets PV_fixed = PV_float.

    R = sum_i(alpha_i * f_i * DF_i) / sum_i(alpha_i * DF_i)
    """
    annuity = float(np.sum(day_count_fractions * discount_factors))
    float_pv = float(np.sum(day_count_fractions * forward_rates * discount_factors))
    return float_pv / annuity


def swap_npv(notional: float, fixed_rate: float, pay_fixed: bool,
              payment_times: np.ndarray, discount_factors: np.ndarray,
              day_count_fractions: np.ndarray,
              forward_rates: np.ndarray) -> float:
    """
    Net present value of an interest rate swap.

    NPV (pay fixed) = PV_float - PV_fixed
    NPV (receive fixed) = PV_fixed - PV_float
    """
    pv_fixed = swap_fixed_leg_pv(notional, fixed_rate, payment_times,
                                   discount_factors, day_count_fractions)
    pv_float = swap_float_leg_pv(notional, payment_times, discount_factors,
                                   day_count_fractions, forward_rates)
    return (pv_float - pv_fixed) if pay_fixed else (pv_fixed - pv_float)


def swap_dv01(notional: float, payment_times: np.ndarray,
               discount_factors: np.ndarray,
               day_count_fractions: np.ndarray) -> float:
    """
    DV01 of an interest rate swap (receive fixed).

    Approximately = notional * duration_of_fixed_leg * 0.0001
    = notional * sum(alpha_i * t_i * DF_i) * 0.0001
    """
    weighted = float(np.sum(day_count_fractions * payment_times * discount_factors))
    return notional * weighted * 0.0001


# ---------------------------------------------------------------------------
# 11. Cross-Currency Basis Swap
# ---------------------------------------------------------------------------

def xccy_basis_swap_npv(notional_dom: float, notional_for: float,
                          fx_rate: float,
                          fixed_rate_dom: float, fixed_rate_for: float,
                          basis_spread: float,
                          payment_times: np.ndarray,
                          df_dom: np.ndarray, df_for: np.ndarray,
                          alpha: np.ndarray) -> dict:
    """
    Cross-currency basis swap NPV.

    Typical structure:
      - Domestic leg: pay fixed_rate_dom + basis_spread (quarterly)
      - Foreign leg:  receive fixed_rate_for
      - Initial and final exchange of notionals at fx_rate

    NPV_dom_ccy = PV_for_leg_in_dom - PV_dom_leg

    Parameters
    ----------
    notional_dom  : domestic notional
    notional_for  : foreign notional (= notional_dom / fx_rate)
    fx_rate       : spot exchange rate (dom per foreign)
    basis_spread  : spread added to domestic leg (negative = discount)
    df_dom        : domestic discount factors
    df_for        : foreign discount factors
    alpha         : day count fractions

    Returns
    -------
    dict with dom_leg_pv, for_leg_pv_in_dom, npv, basis_spread
    """
    # PV of domestic coupon payments (pay side)
    dom_coupons = notional_dom * (fixed_rate_dom + basis_spread) * np.sum(alpha * df_dom)
    dom_principal = notional_dom * (df_dom[-1] - 1.0)   # initial exchange at par minus final

    # PV of foreign coupon payments (receive side) converted to domestic
    for_coupons = notional_for * fixed_rate_for * np.sum(alpha * df_for) * fx_rate
    for_principal = notional_for * (df_for[-1] - 1.0) * fx_rate

    # Total
    pv_dom = dom_coupons + dom_principal
    pv_for = for_coupons + for_principal
    npv    = pv_for - pv_dom

    return {
        "dom_leg_pv":       pv_dom,
        "for_leg_pv_in_dom": pv_for,
        "npv":              npv,
        "basis_spread":     basis_spread,
    }


def implied_basis_spread(notional_dom: float, notional_for: float,
                          fx_rate: float,
                          fixed_rate_dom: float, fixed_rate_for: float,
                          payment_times: np.ndarray,
                          df_dom: np.ndarray, df_for: np.ndarray,
                          alpha: np.ndarray) -> float:
    """Find the basis spread that makes xccy swap NPV = 0."""
    def obj(s):
        res = xccy_basis_swap_npv(notional_dom, notional_for, fx_rate,
                                    fixed_rate_dom, fixed_rate_for, s,
                                    payment_times, df_dom, df_for, alpha)
        return res["npv"]
    return brentq(obj, -0.05, 0.05, xtol=1e-8)


# ---------------------------------------------------------------------------
# 12. Bond Futures CTD Selection
# ---------------------------------------------------------------------------

def bond_futures_invoice_price(futures_price: float,
                                conversion_factor: float,
                                accrued_interest: float) -> float:
    """
    Invoice (delivery) price for a bond delivered against futures.

    Invoice = Futures * Conversion Factor + Accrued Interest
    """
    return futures_price * conversion_factor + accrued_interest


def conversion_factor(coupon_rate: float, maturity_years: float,
                       notional_coupon: float = 0.06,
                       freq: int = 2) -> float:
    """
    Approximate conversion factor for US Treasury futures.

    CF = price of bond (per $1 face) discounted at notional coupon,
         with maturity rounded to nearest quarter.
    """
    n = int(round(maturity_years * freq))
    if n <= 0:
        return 1.0
    return bond_price_from_ytm(1.0, coupon_rate, notional_coupon, n, freq)


def ctd_selection(bonds: list, futures_price: float,
                   repo_rate: float, delivery_date: float) -> dict:
    """
    Select Cheapest to Deliver (CTD) bond for futures delivery.

    For each bond, compute the net basis:
        Net Basis = Bond Price - (Futures * CF + Carry)

    where Carry = (Coupon Income - Financing Cost) over delivery period.

    Parameters
    ----------
    bonds : list of dicts, each with:
        {name, price, coupon_rate, cf, accrued, maturity, freq}
    futures_price : quoted futures price
    repo_rate     : financing rate (annual)
    delivery_date : time to delivery in years

    Returns
    -------
    dict with ctd bond name, all bonds sorted by net basis
    """
    results = []
    for b in bonds:
        price  = b["price"]
        cf     = b.get("cf", conversion_factor(b["coupon_rate"],
                                                 b["maturity"], 0.06,
                                                 b.get("freq", 2)))
        acc    = b.get("accrued", 0.0)

        coupon_income    = price * b["coupon_rate"] * delivery_date
        financing_cost   = (price + acc) * repo_rate * delivery_date
        carry            = coupon_income - financing_cost

        theoretical_f    = (price - acc + carry) / cf
        delivery_invoice = bond_futures_invoice_price(futures_price, cf, acc)
        net_basis        = price - delivery_invoice
        gross_basis      = price - futures_price * cf

        results.append({
            "name":       b.get("name", "unknown"),
            "net_basis":  net_basis,
            "gross_basis": gross_basis,
            "cf":         cf,
            "carry":      carry,
            "theoretical_futures": theoretical_f,
        })

    results.sort(key=lambda x: x["net_basis"])
    return {"ctd": results[0]["name"], "bonds": results}


# ---------------------------------------------------------------------------
# 13. Inflation-Linked Bond (TIPS)
# ---------------------------------------------------------------------------

def tips_accrued_principal(face: float, base_cpi: float,
                            current_cpi: float) -> float:
    """
    TIPS inflation-adjusted (accreted) principal.

    Accreted Principal = Face * (Current CPI / Base CPI)
    """
    return face * current_cpi / base_cpi


def tips_real_yield(price: float, real_coupon: float,
                     n_periods: int, face: float = 100.0,
                     freq: int = 2) -> float:
    """
    Real yield to maturity of an inflation-linked bond (TIPS).
    Price is quoted in real terms (as % of accreted principal).
    """
    return bond_ytm(price, face, real_coupon, n_periods, freq)


def inflation_breakeven(nominal_ytm: float, real_ytm: float) -> float:
    """
    Breakeven inflation rate: the inflation rate that equates nominal and real bonds.

    Breakeven ≈ nominal_ytm - real_ytm   (Fisher approximation)

    Exact Fisher: (1 + nominal) = (1 + real) * (1 + inflation)
    => inflation = (1 + nominal)/(1 + real) - 1
    """
    return (1.0 + nominal_ytm) / (1.0 + real_ytm) - 1.0


def tips_carry(real_coupon: float, inflation_rate: float,
               repo_rate: float, price: float,
               face: float = 100.0, horizon: float = 1.0 / 12.0) -> float:
    """
    TIPS carry over a horizon period.

    Carry = Coupon Income + Inflation Accretion - Financing Cost

    (all expressed as % of price over the horizon)
    """
    coupon_income    = real_coupon * face * horizon
    accretion        = face * inflation_rate * horizon    # approx
    financing        = price * repo_rate * horizon
    return (coupon_income + accretion - financing) / price


def tips_duration(price: float, real_coupon: float,
                   n_periods: int, freq: int = 2,
                   face: float = 100.0) -> float:
    """Modified duration of TIPS in real terms."""
    real_ytm = tips_real_yield(price, real_coupon, n_periods, face, freq)
    return modified_duration(face, real_coupon, real_ytm, n_periods, freq)


# ---------------------------------------------------------------------------
# 14. Repo Financing Cost and Carry
# ---------------------------------------------------------------------------

def repo_carry(bond_price: float, coupon_rate: float,
               repo_rate: float, horizon: float,
               face: float = 100.0) -> float:
    """
    Net carry on a bond financed at repo rate over `horizon` years.

    Carry = Coupon Accrual - Repo Interest
          = face * coupon_rate * horizon - bond_price * repo_rate * horizon
    """
    coupon_accrual = face * coupon_rate * horizon
    repo_interest  = bond_price * repo_rate * horizon
    return coupon_accrual - repo_interest


def total_return_repo(bond_price: float, coupon_rate: float,
                       repo_rate: float, horizon: float,
                       price_end: float, face: float = 100.0) -> float:
    """
    Total return on a leveraged bond position financed via repo.

    TR = (Price_end - Price_0 + Coupon - Repo_cost) / Price_0
    """
    carry = repo_carry(bond_price, coupon_rate, repo_rate, horizon, face)
    capital_gain = price_end - bond_price
    return (capital_gain + carry) / bond_price


def repo_implied_forward_price(spot_price: float, accrued: float,
                                repo_rate: float,
                                coupon: float, coupon_time: float,
                                T: float) -> float:
    """
    Forward (futures) price of a bond implied by repo financing.

    F = (Spot + Accrued) * (1 + repo*T) - Coupon * (1 + repo*(T - coupon_time))
    """
    F = ((spot_price + accrued) * (1.0 + repo_rate * T)
         - coupon * (1.0 + repo_rate * (T - coupon_time)))
    return F


# ---------------------------------------------------------------------------
# 15. Duration-Neutral Pairs / Hedge Ratio
# ---------------------------------------------------------------------------

def duration_neutral_hedge_ratio(dv01_bond1: float,
                                   dv01_bond2: float) -> float:
    """
    Hedge ratio for a duration-neutral long/short bond pair.

    To be DV01-neutral: N2 = -N1 * DV01_1 / DV01_2

    Returns the number of units of bond 2 per unit of bond 1.
    """
    if dv01_bond2 == 0:
        raise ValueError("DV01 of bond 2 cannot be zero")
    return -dv01_bond1 / dv01_bond2


def spread_dv01(dv01_long: float, dv01_short: float,
                 n_long: float, n_short: float) -> float:
    """
    Net DV01 of a spread trade (long n_long of bond 1, short n_short of bond 2).
    """
    return n_long * dv01_long - n_short * dv01_short


def fly_hedge_ratios(dv01_belly: float, dv01_wing1: float,
                      dv01_wing2: float,
                      wing1_weight: float = 0.5) -> tuple:
    """
    Hedge ratios for a butterfly trade (long belly, short wings).

    Constraints:
      1. Net DV01 = 0: n1*dv01_w1 + n2*dv01_w2 = dv01_belly
      2. Wing allocation: n1/(n1+n2) = wing1_weight

    Returns (n_wing1, n_wing2) per unit of belly.
    """
    # From constraint 2: n1 = wing1_weight * (n1 + n2)
    # => n1 * (1 - wing1_weight) = wing1_weight * n2
    # => n1 = n2 * wing1_weight / (1 - wing1_weight)
    # Substitute into constraint 1:
    # n2 * wing1_weight / (1-w) * dv01_w1 + n2 * dv01_w2 = dv01_belly
    w  = wing1_weight
    n2 = dv01_belly / (w / (1.0 - w) * dv01_wing1 + dv01_wing2)
    n1 = n2 * w / (1.0 - w)
    return n1, n2


def pnl_attribution(price_changes: np.ndarray, dv01s: np.ndarray,
                     positions: np.ndarray) -> dict:
    """
    P&L attribution for a bond portfolio.

    Parameters
    ----------
    price_changes : array of bond price changes in dollars
    dv01s         : array of DV01s per bond
    positions     : array of positions (signed face amounts)

    Returns
    -------
    dict with total_pnl, duration_pnl, other
    """
    total_pnl    = float(np.sum(price_changes * positions))
    duration_pnl = float(np.sum(dv01s * positions))   # approximate rate contribution

    return {
        "total_pnl":    total_pnl,
        "duration_pnl": duration_pnl,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def par_yield(maturities: np.ndarray, zero_rates: np.ndarray,
              freq: int = 2) -> np.ndarray:
    """
    Compute par yields from spot zero curve.

    par_yield(T) = (1 - DF(T)) / annuity_factor(T)

    where DF(T) = exp(-z(T)*T) and annuity_factor = sum_i DF(t_i) * (1/freq).
    """
    interp = interp1d(maturities, zero_rates, kind="linear",
                       fill_value="extrapolate")
    par_yields = []

    for T in maturities:
        coupon_times = np.arange(1.0 / freq, T + 1e-9, 1.0 / freq)
        coupon_times = coupon_times[coupon_times <= T + 1e-9]
        z_coupon     = interp(coupon_times)
        df_coupon    = np.exp(-z_coupon * coupon_times)
        annuity      = np.sum(df_coupon) / freq
        df_T         = np.exp(-interp(T) * T)
        par_yields.append((1.0 - df_T) / annuity if annuity > 0 else np.nan)

    return np.array(par_yields)


def yield_spread(ytm1: float, ytm2: float, in_bps: bool = True) -> float:
    """
    Yield spread between two bonds.
    """
    spread = ytm1 - ytm2
    return spread * 10_000.0 if in_bps else spread


def bond_return_components(coupon_income: float, reinvestment_income: float,
                            capital_gain: float, initial_price: float) -> dict:
    """
    Decompose total bond return into components.
    """
    total = coupon_income + reinvestment_income + capital_gain
    return {
        "total_return":         total / initial_price,
        "coupon_return":        coupon_income / initial_price,
        "reinvestment_return":  reinvestment_income / initial_price,
        "capital_return":       capital_gain / initial_price,
    }
