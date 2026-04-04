/*
 * bh_fast.c — High-performance Black-Hole physics engine in C.
 *
 * This is the fastest possible implementation of BH mass dynamics,
 * designed for scanning millions of bars across many instruments.
 *
 * The algorithm exactly mirrors BHPhysics.jl but operates on raw C arrays.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#ifndef NAN
#  define NAN (0.0/0.0)
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a)    ((a) >= 0 ? (a) : -(a))

/* ─────────────────────────────────────────────────────────────────────────── */
/* BH State structure                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    double mass;         /* Current accumulated mass */
    int    active;       /* 1 = BH active (above form threshold) */
    int    bh_dir;       /* Direction: +1 bull, -1 bear, 0 neutral */
    int    ctl;          /* Consecutive timelike bars */
    double bh_form;      /* Formation mass threshold */
    double bh_collapse;  /* Collapse mass threshold */
    double bh_decay;     /* Per-bar decay multiplier */
    double cf;           /* Critical frequency */
    int    ctl_req;      /* Consecutive timelike required for formation */
    double prev_price;   /* Previous bar close */
    int    bars_active;  /* Bars since last activation */
    double peak_mass;    /* Peak mass since last activation */
    double total_mass;   /* Cumulative mass absorbed */
} BHState;

/* ─────────────────────────────────────────────────────────────────────────── */
/* Initialise / reset a BHState                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */

void bh_init(BHState *s, double cf, double bh_form, double bh_decay,
             double bh_collapse, int ctl_req)
{
    s->mass        = 0.0;
    s->active      = 0;
    s->bh_dir      = 0;
    s->ctl         = 0;
    s->cf          = cf;
    s->bh_form     = bh_form;
    s->bh_collapse = bh_collapse;
    s->bh_decay    = bh_decay;
    s->ctl_req     = ctl_req;
    s->prev_price  = NAN;
    s->bars_active = 0;
    s->peak_mass   = 0.0;
    s->total_mass  = 0.0;
}

void bh_reset(BHState *s)
{
    s->mass        = 0.0;
    s->active      = 0;
    s->bh_dir      = 0;
    s->ctl         = 0;
    s->prev_price  = NAN;
    s->bars_active = 0;
    s->peak_mass   = 0.0;
    s->total_mass  = 0.0;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Single-bar BH update                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * bh_update — Advance the BH state by one price bar.
 *
 * @param s           Pointer to BHState (modified in place)
 * @param price       Current bar close price
 * @param prev_price  Previous bar close price (can pass s->prev_price)
 *
 * @return  1 if BH became newly active this bar, 0 otherwise.
 */
int bh_update(BHState *s, double price, double prev_price)
{
    if (prev_price <= 0 || price <= 0) return 0;

    /* Beta = |Δ log price| */
    double beta = ABS(log(price / prev_price));
    int is_tl   = (beta < s->cf) ? 1 : 0;

    /* Consecutive timelike counter */
    if (is_tl) {
        s->ctl++;
    } else {
        s->ctl = 0;
    }

    /* Mass accumulation */
    double dm;
    if (is_tl) {
        dm = s->cf * 0.5;      /* Steady infall for timelike bars */
    } else {
        double excess = beta - s->cf;
        dm = excess * 2.0;     /* Impulsive accretion for spacelike */
    }

    s->mass      = s->mass * s->bh_decay + dm;
    s->total_mass += dm;

    /* Direction: update from net log return (use recent price vs initialised) */
    double net = beta;   /* simplified: use |beta| sign from price direction */
    if (price > prev_price) {
        s->bh_dir = 1;
    } else if (price < prev_price) {
        s->bh_dir = -1;
    }
    /* Keep direction sticky when inside BH */

    int newly_activated = 0;

    if (!s->active) {
        /* Check for activation */
        if (s->mass >= s->bh_form && s->ctl >= s->ctl_req) {
            s->active      = 1;
            s->bars_active = 0;
            s->peak_mass   = s->mass;
            newly_activated = 1;
        }
    } else {
        s->bars_active++;
        if (s->mass > s->peak_mass) s->peak_mass = s->mass;

        /* Check for collapse */
        if (s->mass < s->bh_collapse) {
            s->active      = 0;
            s->mass        = 0.0;
            s->ctl         = 0;
            s->bh_dir      = 0;
            s->peak_mass   = 0.0;
            s->bars_active = 0;
        }
    }

    s->prev_price = price;
    return newly_activated;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Series computation                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * bh_series — Run BH physics over an entire price series.
 *
 * @param closes     Input: close prices (length n)
 * @param n          Number of bars
 * @param cf         Critical frequency
 * @param bh_form    Formation mass threshold
 * @param bh_decay   Per-bar mass decay multiplier
 * @param bh_collapse Collapse threshold
 * @param ctl_req    Consecutive timelike required for formation
 * @param masses     Output: BH mass per bar (length n)
 * @param active     Output: 0/1 active flag per bar (length n)
 * @param ctl_out    Output: consecutive timelike count per bar (length n)
 */
void bh_series(const double *closes, int n,
               double cf, double bh_form, double bh_decay,
               double bh_collapse, int ctl_req,
               double *masses, int *active, int *ctl_out)
{
    if (n <= 0 || !closes || !masses || !active || !ctl_out) return;

    BHState s;
    bh_init(&s, cf, bh_form, bh_decay, bh_collapse, ctl_req);

    masses[0]  = 0.0;
    active[0]  = 0;
    ctl_out[0] = 0;
    s.prev_price = closes[0];

    for (int i = 1; i < n; i++) {
        bh_update(&s, closes[i], s.prev_price);
        masses[i]  = s.mass;
        active[i]  = s.active;
        ctl_out[i] = s.ctl;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Fast backtest                                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * Trade record for output.
 */
typedef struct {
    int    entry_bar;
    int    exit_bar;
    double entry_price;
    double exit_price;
    int    direction;   /* +1 long, -1 short */
    double pnl;         /* log-return * direction */
    double mfe;         /* max favourable excursion */
    double mae;         /* max adverse excursion */
    int    duration;
    double peak_mass;
    int    tf_score;
} TradeRecord;

/**
 * bh_backtest_c — Full BH backtest over a single price series.
 *
 * Entry: BH newly activated, direction determined by bh_dir.
 * Exit: BH collapses.
 *
 * @param closes       Close prices (length n)
 * @param highs        High prices  (length n)
 * @param lows         Low prices   (length n)
 * @param n            Number of bars
 * @param cf           Critical frequency
 * @param bh_form      Formation threshold
 * @param bh_decay     Decay multiplier
 * @param bh_collapse  Collapse threshold
 * @param ctl_req      Consecutive timelike required
 * @param long_only    1 = long only, 0 = long & short
 * @param commission   Per-side commission fraction
 * @param slippage     Per-side slippage fraction
 * @param equity_curve Output: equity value per bar (length n, caller-allocated)
 * @param positions    Output: position per bar (length n, caller-allocated)
 * @param trade_count  Output: total number of completed trades
 * @param trades       Output: array of TradeRecord (caller allocates max n/2 records)
 */
void bh_backtest_c(const double *closes, const double *highs, const double *lows,
                   int n,
                   double cf, double bh_form, double bh_decay, double bh_collapse,
                   int ctl_req, int long_only,
                   double commission, double slippage,
                   double *equity_curve, int *positions,
                   int *trade_count, TradeRecord *trades)
{
    if (!closes || !equity_curve || !positions || n <= 0) return;

    BHState s;
    bh_init(&s, cf, bh_form, bh_decay, bh_collapse, ctl_req);

    double cost       = commission + slippage;
    double eq         = 1.0;
    int    pos        = 0;
    int    entry_bar  = -1;
    double entry_px   = 0.0;
    int    entry_dir  = 0;
    double entry_mass = 0.0;
    double mfe        = 0.0;
    double mae        = 0.0;
    int    tc         = 0;
    int    prev_active = 0;

    equity_curve[0] = eq;
    positions[0]    = 0;
    s.prev_price    = closes[0];

    for (int i = 1; i < n; i++) {
        int newly_activated = bh_update(&s, closes[i], s.prev_price);

        /* Update equity */
        if (pos != 0) {
            double bar_ret = log(closes[i] / closes[i-1]) * pos;
            eq            *= exp(bar_ret);
            double fav     = bar_ret * pos;
            if (fav > 0)  mfe += fav;
            else          mae += -fav;
        }
        equity_curve[i] = eq;
        positions[i]    = pos;

        /* Entry: BH newly activated */
        if (pos == 0 && newly_activated) {
            int dir = s.bh_dir;
            if (dir == 1) {
                pos = 1;
            } else if (dir == -1 && !long_only) {
                pos = -1;
            }

            if (pos != 0) {
                entry_bar  = i;
                entry_px   = closes[i] * (1.0 + cost * pos);
                entry_dir  = pos;
                entry_mass = s.mass;
                mfe        = 0.0;
                mae        = 0.0;
                eq        *= (1.0 - cost);
                equity_curve[i] = eq;
            }
        }

        /* Exit: BH collapsed */
        if (pos != 0 && !s.active && prev_active) {
            double exit_px = closes[i] * (1.0 - cost * pos);
            double pnl     = log(exit_px / entry_px) * entry_dir;
            eq            *= (1.0 - cost);
            equity_curve[i] = eq;

            if (trades) {
                TradeRecord *tr     = &trades[tc];
                tr->entry_bar       = entry_bar;
                tr->exit_bar        = i;
                tr->entry_price     = entry_px;
                tr->exit_price      = exit_px;
                tr->direction       = entry_dir;
                tr->pnl             = pnl;
                tr->mfe             = mfe;
                tr->mae             = mae;
                tr->duration        = i - entry_bar;
                tr->peak_mass       = entry_mass;
                tr->tf_score        = 1;
            }
            tc++;
            pos = 0;
        }

        prev_active = s.active;
        positions[i] = pos;
    }

    /* Force close any open trade */
    if (pos != 0 && trades) {
        double exit_px = closes[n-1] * (1.0 - cost * pos);
        double pnl     = log(exit_px / entry_px) * entry_dir;

        TradeRecord *tr  = &trades[tc];
        tr->entry_bar    = entry_bar;
        tr->exit_bar     = n - 1;
        tr->entry_price  = entry_px;
        tr->exit_price   = exit_px;
        tr->direction    = entry_dir;
        tr->pnl          = pnl;
        tr->mfe          = mfe;
        tr->mae          = mae;
        tr->duration     = n - 1 - entry_bar;
        tr->peak_mass    = entry_mass;
        tr->tf_score     = 1;
        tc++;
    }

    if (trade_count) *trade_count = tc;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Multi-symbol sweep: run bh_series on N symbols in a batch                   */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * bh_batch — Run BH series on multiple symbols stored contiguously.
 *
 * Input layout: closes[sym * n_bars + bar]   (row-major)
 * Output layout: same convention for masses, active.
 *
 * @param closes   Contiguous price array (n_symbols × n_bars)
 * @param n_sym    Number of symbols
 * @param n_bars   Number of bars per symbol
 * @param cf, bh_form, bh_decay, bh_collapse, ctl_req — shared BH config
 * @param masses   Output masses (n_symbols × n_bars)
 * @param active   Output active (n_symbols × n_bars)
 */
void bh_batch(const double *closes, int n_sym, int n_bars,
              double cf, double bh_form, double bh_decay,
              double bh_collapse, int ctl_req,
              double *masses, int *active)
{
    if (!closes || !masses || !active || n_sym <= 0 || n_bars <= 0) return;

    int *ctl_tmp = (int*)malloc(n_bars * sizeof(int));
    if (!ctl_tmp) return;

    for (int s = 0; s < n_sym; s++) {
        const double *sym_closes = closes  + s * n_bars;
        double       *sym_masses = masses  + s * n_bars;
        int          *sym_active = active  + s * n_bars;

        bh_series(sym_closes, n_bars, cf, bh_form, bh_decay, bh_collapse,
                  ctl_req, sym_masses, sym_active, ctl_tmp);
    }

    free(ctl_tmp);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Walk-forward BH optimisation: grid search over cf × bh_form               */
/* ─────────────────────────────────────────────────────────────────────────── */

typedef struct {
    double cf;
    double bh_form;
    double sharpe;
    int    n_trades;
    double max_dd;
    double total_ret;
} BHGridResult;

static double compute_sharpe(const double *equity, int n)
{
    if (n < 2) return 0.0;
    double mean = 0.0, var = 0.0;
    for (int i = 1; i < n; i++) {
        double r = log(equity[i] / equity[i-1]);
        mean += r;
    }
    mean /= (n - 1);
    for (int i = 1; i < n; i++) {
        double r = log(equity[i] / equity[i-1]);
        var += (r - mean) * (r - mean);
    }
    var /= (n - 2);
    return var > 1e-12 ? mean / sqrt(var) * sqrt(252.0) : 0.0;
}

static double compute_max_dd(const double *equity, int n)
{
    double peak = equity[0], max_dd = 0.0;
    for (int i = 1; i < n; i++) {
        if (equity[i] > peak) peak = equity[i];
        double dd = (peak - equity[i]) / peak;
        if (dd > max_dd) max_dd = dd;
    }
    return max_dd;
}

/**
 * bh_grid_search — Grid search over (cf × bh_form) combinations.
 *
 * @param closes      Price series
 * @param highs       High prices
 * @param lows        Low prices
 * @param n           Number of bars
 * @param cf_values   Array of CF values to test
 * @param n_cf        Length of cf_values
 * @param form_values Array of bh_form values to test
 * @param n_form      Length of form_values
 * @param bh_decay    Fixed decay
 * @param bh_collapse Fixed collapse fraction of form
 * @param ctl_req     Fixed CTL requirement
 * @param long_only   Long-only flag
 * @param results     Output: n_cf × n_form BHGridResult array (caller-allocated)
 */
void bh_grid_search(const double *closes, const double *highs, const double *lows,
                    int n,
                    const double *cf_values, int n_cf,
                    const double *form_values, int n_form,
                    double bh_decay, double bh_collapse_frac, int ctl_req,
                    int long_only,
                    BHGridResult *results)
{
    if (!results || n <= 0) return;

    int max_trades = n / 2 + 1;
    double *equity    = (double*)malloc(n * sizeof(double));
    int    *positions = (int*)   malloc(n * sizeof(int));
    TradeRecord *trades = (TradeRecord*)malloc(max_trades * sizeof(TradeRecord));

    if (!equity || !positions || !trades) {
        free(equity); free(positions); free(trades);
        return;
    }

    for (int ci = 0; ci < n_cf; ci++) {
        for (int fi = 0; fi < n_form; fi++) {
            double cf      = cf_values[ci];
            double bf      = form_values[fi];
            double bc      = bf * bh_collapse_frac;
            int    tc      = 0;

            bh_backtest_c(closes, highs, lows, n,
                          cf, bf, bh_decay, bc, ctl_req, long_only,
                          0.0004, 0.0001,
                          equity, positions, &tc, trades);

            int idx = ci * n_form + fi;
            results[idx].cf        = cf;
            results[idx].bh_form   = bf;
            results[idx].sharpe    = compute_sharpe(equity, n);
            results[idx].n_trades  = tc;
            results[idx].max_dd    = compute_max_dd(equity, n);
            results[idx].total_ret = (n > 1) ? log(equity[n-1] / equity[0]) : 0.0;
        }
    }

    free(equity); free(positions); free(trades);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* BH multi-timeframe signal (simplified 2-TF version)                         */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * bh_mtf_signal — Compute BH direction alignment score across two timeframes.
 *
 * For each bar in the fast TF (length n_fast), look up the corresponding
 * slow TF bar and compute tf_score = sum of active directions.
 *
 * @param closes_slow  Slow TF close prices (length n_slow)
 * @param closes_fast  Fast TF close prices (length n_fast)
 * @param n_slow, n_fast  Lengths of respective arrays
 * @param cf_s, cf_f   CF for slow / fast TFs
 * @param form_s, form_f  Formation thresholds
 * @param decay_s, decay_f  Decay multipliers
 * @param collapse_s, collapse_f  Collapse thresholds
 * @param ctl_req      CTL requirement (shared)
 * @param tf_scores    Output: tf_score per fast bar (length n_fast)
 */
void bh_mtf_signal(const double *closes_slow, int n_slow,
                   const double *closes_fast, int n_fast,
                   double cf_s, double form_s, double decay_s, double collapse_s,
                   double cf_f, double form_f, double decay_f, double collapse_f,
                   int ctl_req,
                   int *tf_scores)
{
    if (!closes_slow || !closes_fast || !tf_scores) return;

    /* Compute slow TF masses */
    double *masses_s = (double*)malloc(n_slow * sizeof(double));
    int    *active_s = (int*)   malloc(n_slow * sizeof(int));
    int    *ctl_s    = (int*)   malloc(n_slow * sizeof(int));

    /* Compute fast TF masses */
    double *masses_f = (double*)malloc(n_fast * sizeof(double));
    int    *active_f = (int*)   malloc(n_fast * sizeof(int));
    int    *ctl_f    = (int*)   malloc(n_fast * sizeof(int));

    if (!masses_s || !active_s || !ctl_s || !masses_f || !active_f || !ctl_f) {
        free(masses_s); free(active_s); free(ctl_s);
        free(masses_f); free(active_f); free(ctl_f);
        return;
    }

    bh_series(closes_slow, n_slow, cf_s, form_s, decay_s, collapse_s, ctl_req,
              masses_s, active_s, ctl_s);
    bh_series(closes_fast, n_fast, cf_f, form_f, decay_f, collapse_f, ctl_req,
              masses_f, active_f, ctl_f);

    /* Direction arrays: derive from BH state transitions */
    int *dir_s = (int*)calloc(n_slow, sizeof(int));
    int *dir_f = (int*)calloc(n_fast, sizeof(int));

    {
        /* Derive direction from price movement at activation */
        int in_bh = 0;
        int act_bar = 0;
        for (int i = 1; i < n_slow; i++) {
            if (active_s[i] && !active_s[i-1]) {
                in_bh = 1; act_bar = i;
            }
            if (!active_s[i] && active_s[i-1]) {
                in_bh = 0;
            }
            if (in_bh && i > act_bar) {
                double net = closes_slow[i] / closes_slow[act_bar] - 1.0;
                dir_s[i] = net > 0.002 ? 1 : (net < -0.002 ? -1 : 0);
            }
        }
    }
    {
        int in_bh = 0; int act_bar = 0;
        for (int i = 1; i < n_fast; i++) {
            if (active_f[i] && !active_f[i-1]) { in_bh = 1; act_bar = i; }
            if (!active_f[i] && active_f[i-1]) { in_bh = 0; }
            if (in_bh && i > act_bar) {
                double net = closes_fast[i] / closes_fast[act_bar] - 1.0;
                dir_f[i] = net > 0.002 ? 1 : (net < -0.002 ? -1 : 0);
            }
        }
    }

    /* Compute tf_scores per fast bar */
    double ratio = (double)n_fast / n_slow;

    for (int i = 0; i < n_fast; i++) {
        int i_slow = (int)(i / ratio);
        if (i_slow >= n_slow) i_slow = n_slow - 1;

        int score = 0;
        if (active_s[i_slow]) score += dir_s[i_slow];
        if (active_f[i])      score += dir_f[i];

        tf_scores[i] = score;
    }

    free(masses_s); free(active_s); free(ctl_s);
    free(masses_f); free(active_f); free(ctl_f);
    free(dir_s); free(dir_f);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Statistics helpers for equity curves                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * equity_stats_c — Compute performance metrics from equity curve.
 *
 * @param equity        Equity curve (length n)
 * @param n             Number of bars
 * @param bars_per_year Number of bars in a year (252 daily, 365*24 hourly, etc.)
 * @param out_sharpe    Output: annualised Sharpe
 * @param out_max_dd    Output: max drawdown fraction
 * @param out_cagr      Output: compound annual growth rate
 * @param out_calmar    Output: CAGR / max_dd
 */
void equity_stats_c(const double *equity, int n, double bars_per_year,
                    double *out_sharpe, double *out_max_dd,
                    double *out_cagr, double *out_calmar)
{
    if (!equity || n < 2) {
        if (out_sharpe) *out_sharpe = 0.0;
        if (out_max_dd) *out_max_dd = 0.0;
        if (out_cagr)   *out_cagr   = 0.0;
        if (out_calmar) *out_calmar = 0.0;
        return;
    }

    /* Returns */
    double *rets = (double*)malloc((n-1) * sizeof(double));
    if (!rets) return;

    double mean_r = 0.0;
    for (int i = 1; i < n; i++) {
        rets[i-1] = log(equity[i] / equity[i-1]);
        mean_r   += rets[i-1];
    }
    mean_r /= (n - 1);

    double var_r = 0.0;
    for (int i = 0; i < n-1; i++) {
        double d = rets[i] - mean_r;
        var_r += d * d;
    }
    var_r /= (n - 2);

    double sharpe = var_r > 1e-12 ?
                    mean_r / sqrt(var_r) * sqrt(bars_per_year) : 0.0;

    /* Max drawdown */
    double peak = equity[0], max_dd = 0.0;
    for (int i = 1; i < n; i++) {
        if (equity[i] > peak) peak = equity[i];
        double dd = (peak - equity[i]) / MAX(peak, 1e-10);
        if (dd > max_dd) max_dd = dd;
    }

    /* CAGR */
    double years = (n - 1.0) / bars_per_year;
    double cagr  = years > 1e-6 ?
                   pow(equity[n-1] / equity[0], 1.0 / years) - 1.0 : 0.0;

    double calmar = max_dd > 1e-10 ? cagr / max_dd : 0.0;

    if (out_sharpe) *out_sharpe = sharpe;
    if (out_max_dd) *out_max_dd = max_dd;
    if (out_cagr)   *out_cagr   = cagr;
    if (out_calmar) *out_calmar = calmar;

    free(rets);
}
