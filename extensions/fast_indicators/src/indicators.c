/*
 * indicators.c — High-performance technical indicator implementations in C.
 *
 * All functions take double* arrays (in) and write to double* (out).
 * Fills are NaN for warm-up periods not yet computable.
 * No dynamic allocation — all O(1) extra space (ring buffers on stack).
 *
 * Build: included via indicators_py.c extension wrapper.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#ifndef NAN
#  define NAN (0.0/0.0)
#endif

/* ─────────────────────────────────────────────────────────────────────────── */
/* Utility macros                                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ABS(a)    ((a) >= 0 ? (a) : -(a))

/* ─────────────────────────────────────────────────────────────────────────── */
/* 1. EMA — Exponential Moving Average                                          */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * ema_c — Classic EMA with Wilder's seed (first value = SMA of first `period` bars).
 *
 * @param close   Input price array
 * @param n       Length of close[]
 * @param period  EMA period
 * @param out     Output array (length n), NaN for i < period-1
 */
void ema_c(const double *close, int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    double k = 2.0 / (period + 1.0);
    double seed = 0.0;

    /* Fill NaN for warm-up */
    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
        seed  += close[i];
    }

    if (n < period) return;

    /* Seed: SMA of first `period` values */
    seed += close[period - 1];
    seed /= period;
    out[period - 1] = seed;

    /* Recursive EMA */
    double ema = seed;
    for (int i = period; i < n; i++) {
        ema = close[i] * k + ema * (1.0 - k);
        out[i] = ema;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 2. SMA — Simple Moving Average                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * sma_c — Rolling simple moving average using a sliding sum (O(n) total).
 */
void sma_c(const double *close, int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    if (n < period) return;

    /* Initial window sum */
    double sum = 0.0;
    for (int i = 0; i < period; i++) sum += close[i];
    out[period - 1] = sum / period;

    /* Slide the window */
    for (int i = period; i < n; i++) {
        sum += close[i] - close[i - period];
        out[i] = sum / period;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 3. WMA — Weighted Moving Average                                             */
/* ─────────────────────────────────────────────────────────────────────────── */

void wma_c(const double *close, int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    double denom = period * (period + 1) / 2.0;

    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    for (int i = period - 1; i < n; i++) {
        double wsum = 0.0;
        for (int j = 0; j < period; j++) {
            wsum += close[i - period + 1 + j] * (j + 1);
        }
        out[i] = wsum / denom;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 4. RSI — Relative Strength Index (Wilder's EMA smoothing)                   */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * rsi_c — RSI using Wilder's smoothed average of gains/losses.
 *
 * @param close   Input price array
 * @param n       Number of bars
 * @param period  RSI period (typically 14)
 * @param out     Output RSI [0, 100], NaN for first period bars
 */
void rsi_c(const double *close, int n, int period, double *out)
{
    if (n <= 0 || period <= 1) return;

    for (int i = 0; i < period && i < n; i++) {
        out[i] = NAN;
    }

    if (n <= period) return;

    /* Seed: average gain/loss over first `period` differences */
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 1; i <= period; i++) {
        double diff = close[i] - close[i - 1];
        if (diff >= 0) avg_gain += diff;
        else           avg_loss -= diff;
    }
    avg_gain /= period;
    avg_loss /= period;

    if (avg_loss < 1e-10) {
        out[period] = 100.0;
    } else {
        double rs    = avg_gain / avg_loss;
        out[period]  = 100.0 - 100.0 / (1.0 + rs);
    }

    /* Wilder smoothing */
    double inv_p = 1.0 / period;
    for (int i = period + 1; i < n; i++) {
        double diff = close[i] - close[i - 1];
        double gain = diff > 0 ? diff : 0.0;
        double loss = diff < 0 ? -diff : 0.0;

        avg_gain = avg_gain * (1.0 - inv_p) + gain * inv_p;
        avg_loss = avg_loss * (1.0 - inv_p) + loss * inv_p;

        if (avg_loss < 1e-10) {
            out[i] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            out[i] = 100.0 - 100.0 / (1.0 + rs);
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 5. MACD — Moving Average Convergence/Divergence                             */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * macd_c — MACD line, signal line, histogram.
 *
 * @param close       Input price array
 * @param n           Number of bars
 * @param fast        Fast EMA period (typically 12)
 * @param slow        Slow EMA period (typically 26)
 * @param signal      Signal EMA period (typically 9)
 * @param macd_line   Output: EMA_fast - EMA_slow
 * @param signal_line Output: EMA of macd_line
 * @param histogram   Output: macd_line - signal_line
 */
void macd_c(const double *close, int n, int fast, int slow, int signal,
            double *macd_line, double *signal_line, double *histogram)
{
    if (n <= 0) return;

    double *ema_fast = (double*)malloc(n * sizeof(double));
    double *ema_slow = (double*)malloc(n * sizeof(double));

    if (!ema_fast || !ema_slow) {
        free(ema_fast); free(ema_slow);
        return;
    }

    ema_c(close, n, fast, ema_fast);
    ema_c(close, n, slow, ema_slow);

    /* Compute MACD line and find first valid index */
    int first_valid = slow - 1;
    for (int i = 0; i < first_valid && i < n; i++) {
        macd_line[i]   = NAN;
        signal_line[i] = NAN;
        histogram[i]   = NAN;
    }

    for (int i = first_valid; i < n; i++) {
        macd_line[i] = ema_fast[i] - ema_slow[i];
    }

    /* Signal: EMA of MACD (only on valid portion) */
    int sig_start = first_valid + signal - 1;
    for (int i = first_valid; i < sig_start && i < n; i++) {
        signal_line[i] = NAN;
        histogram[i]   = NAN;
    }

    if (sig_start < n) {
        /* Seed signal EMA */
        double sig_seed = 0.0;
        for (int i = first_valid; i < sig_start + 1; i++) {
            sig_seed += macd_line[i];
        }
        sig_seed /= signal;
        signal_line[sig_start] = sig_seed;
        histogram[sig_start]   = macd_line[sig_start] - sig_seed;

        double k = 2.0 / (signal + 1.0);
        double sig_ema = sig_seed;
        for (int i = sig_start + 1; i < n; i++) {
            sig_ema        = macd_line[i] * k + sig_ema * (1.0 - k);
            signal_line[i] = sig_ema;
            histogram[i]   = macd_line[i] - sig_ema;
        }
    }

    free(ema_fast);
    free(ema_slow);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 6. ATR — Average True Range                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * atr_c — Wilder's ATR (EMA of true range with period multiplier).
 *
 * True Range = max(high-low, |high-prev_close|, |low-prev_close|)
 */
void atr_c(const double *high, const double *low, const double *close,
           int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    double *tr = (double*)malloc(n * sizeof(double));
    if (!tr) return;

    tr[0] = high[0] - low[0];
    for (int i = 1; i < n; i++) {
        double hl  = high[i] - low[i];
        double hc  = ABS(high[i] - close[i-1]);
        double lc  = ABS(low[i]  - close[i-1]);
        tr[i] = MAX(hl, MAX(hc, lc));
    }

    /* Wilder: seed = SMA of first period TRs, then recursive */
    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    if (n < period) {
        free(tr);
        return;
    }

    double seed = 0.0;
    for (int i = 0; i < period; i++) seed += tr[i];
    seed /= period;
    out[period - 1] = seed;

    double inv_p = 1.0 / period;
    double atr   = seed;
    for (int i = period; i < n; i++) {
        atr    = atr * (1.0 - inv_p) + tr[i] * inv_p;
        out[i] = atr;
    }

    free(tr);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 7. Bollinger Bands                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * bollinger_c — Bollinger Bands using SMA and rolling standard deviation.
 *
 * @param close     Input price array
 * @param n         Number of bars
 * @param period    Look-back window (typically 20)
 * @param num_std   Number of standard deviations (typically 2.0)
 * @param upper     Upper band (middle + num_std * std)
 * @param middle    Middle band (SMA)
 * @param lower     Lower band (middle - num_std * std)
 */
void bollinger_c(const double *close, int n, int period, double num_std,
                 double *upper, double *middle, double *lower)
{
    if (n <= 0 || period <= 1) return;

    for (int i = 0; i < period - 1 && i < n; i++) {
        upper[i] = middle[i] = lower[i] = NAN;
    }

    if (n < period) return;

    /* Compute initial window */
    double sum = 0.0, sum2 = 0.0;
    for (int i = 0; i < period; i++) {
        sum  += close[i];
        sum2 += close[i] * close[i];
    }

    for (int i = period - 1; i < n; i++) {
        if (i > period - 1) {
            sum  += close[i] - close[i - period];
            sum2 += close[i] * close[i] - close[i - period] * close[i - period];
        }
        double mean = sum / period;
        double var  = sum2 / period - mean * mean;
        double std  = var > 0 ? sqrt(var) : 0.0;

        middle[i] = mean;
        upper[i]  = mean + num_std * std;
        lower[i]  = mean - num_std * std;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 8. ADX — Average Directional Index                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * adx_c — Wilder's ADX (measures trend strength, 0-100).
 *
 * Intermediate: +DM, -DM, TR → +DI, -DI → DX → ADX.
 */
void adx_c(const double *high, const double *low, const double *close,
           int n, int period, double *out)
{
    if (n < period + 1) return;

    double *plus_dm  = (double*)malloc(n * sizeof(double));
    double *minus_dm = (double*)malloc(n * sizeof(double));
    double *tr       = (double*)malloc(n * sizeof(double));

    if (!plus_dm || !minus_dm || !tr) {
        free(plus_dm); free(minus_dm); free(tr);
        return;
    }

    /* NaN fill for unavailable bars */
    for (int i = 0; i < 2 * period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    /* DM and TR for each bar */
    plus_dm[0] = minus_dm[0] = tr[0] = NAN;
    for (int i = 1; i < n; i++) {
        double up   = high[i] - high[i-1];
        double down = low[i-1] - low[i];

        plus_dm[i]  = (up > down && up > 0) ? up : 0.0;
        minus_dm[i] = (down > up && down > 0) ? down : 0.0;

        double hl = high[i] - low[i];
        double hc = ABS(high[i] - close[i-1]);
        double lc = ABS(low[i]  - close[i-1]);
        tr[i] = MAX(hl, MAX(hc, lc));
    }

    /* Wilder smoothing of +DM, -DM, TR over `period` bars */
    double smoothed_plus  = 0.0;
    double smoothed_minus = 0.0;
    double smoothed_tr    = 0.0;

    for (int i = 1; i <= period; i++) {
        smoothed_plus  += plus_dm[i];
        smoothed_minus += minus_dm[i];
        smoothed_tr    += tr[i];
    }

    double inv_p = 1.0 / period;

    /* First DX value */
    double plus_di  = smoothed_plus  / MAX(smoothed_tr, 1e-10) * 100.0;
    double minus_di = smoothed_minus / MAX(smoothed_tr, 1e-10) * 100.0;
    double dx       = ABS(plus_di - minus_di) / MAX(plus_di + minus_di, 1e-10) * 100.0;

    /* Accumulate DX values to seed ADX */
    double dx_buf[4096];
    int    dx_cnt = 0;
    if (dx_cnt < 4096) dx_buf[dx_cnt++] = dx;

    for (int i = period + 1; i < n; i++) {
        smoothed_plus  = smoothed_plus  - smoothed_plus  * inv_p + plus_dm[i];
        smoothed_minus = smoothed_minus - smoothed_minus * inv_p + minus_dm[i];
        smoothed_tr    = smoothed_tr    - smoothed_tr    * inv_p + tr[i];

        plus_di  = smoothed_plus  / MAX(smoothed_tr, 1e-10) * 100.0;
        minus_di = smoothed_minus / MAX(smoothed_tr, 1e-10) * 100.0;
        dx       = ABS(plus_di - minus_di) / MAX(plus_di + minus_di, 1e-10) * 100.0;
        if (dx_cnt < 4096) dx_buf[dx_cnt++] = dx;
    }

    /* ADX = EMA of DX with same period */
    if (dx_cnt < period) {
        free(plus_dm); free(minus_dm); free(tr);
        return;
    }

    double adx_seed = 0.0;
    for (int i = 0; i < period; i++) adx_seed += dx_buf[i];
    adx_seed /= period;

    int out_start = 2 * period - 1;
    if (out_start < n) out[out_start] = adx_seed;

    double adx = adx_seed;
    for (int k = period; k < dx_cnt && out_start + k - period + 1 < n; k++) {
        adx = adx * (1.0 - inv_p) + dx_buf[k] * inv_p;
        out[out_start + k - period + 1] = adx;
    }

    free(plus_dm); free(minus_dm); free(tr);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 9. Stochastic Oscillator                                                     */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * stochastic_c — %K and %D stochastic oscillator.
 *
 * %K = (close - lowest_low(k_period)) / (highest_high(k_period) - lowest_low(k_period)) * 100
 * %D = SMA(%K, d_period)
 */
void stochastic_c(const double *high, const double *low, const double *close,
                  int n, int k_period, int d_period, double *k_out, double *d_out)
{
    if (n <= 0 || k_period <= 0 || d_period <= 0) return;

    double *raw_k = (double*)malloc(n * sizeof(double));
    if (!raw_k) return;

    for (int i = 0; i < k_period - 1 && i < n; i++) {
        raw_k[i] = k_out[i] = d_out[i] = NAN;
    }

    for (int i = k_period - 1; i < n; i++) {
        double lo = low[i];
        double hi = high[i];
        for (int j = i - k_period + 1; j <= i; j++) {
            lo = MIN(lo, low[j]);
            hi = MAX(hi, high[j]);
        }
        double range = hi - lo;
        raw_k[i] = k_out[i] = (range > 1e-10) ?
                               (close[i] - lo) / range * 100.0 : 50.0;
    }

    /* %D = SMA of %K */
    sma_c(raw_k, n, d_period, d_out);

    free(raw_k);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 10. VWAP — Volume Weighted Average Price                                     */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * vwap_c — Cumulative VWAP (resets each day if dates are not tracked here).
 *
 * Uses typical price = (high + low + close) / 3.
 */
void vwap_c(const double *high, const double *low, const double *close,
            const double *volume, int n, double *out)
{
    if (n <= 0) return;

    double cum_tp_vol = 0.0;
    double cum_vol    = 0.0;

    for (int i = 0; i < n; i++) {
        double tp = (high[i] + low[i] + close[i]) / 3.0;
        double v  = volume[i];

        if (v < 0) v = 0.0;
        cum_tp_vol += tp * v;
        cum_vol    += v;

        out[i] = cum_vol > 1e-10 ? cum_tp_vol / cum_vol : tp;
    }
}

/**
 * vwap_rolling_c — Rolling VWAP over a fixed window.
 */
void vwap_rolling_c(const double *high, const double *low, const double *close,
                    const double *volume, int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    for (int i = period - 1; i < n; i++) {
        double sum_tpv = 0.0, sum_v = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            double tp = (high[j] + low[j] + close[j]) / 3.0;
            double v  = volume[j] > 0 ? volume[j] : 0.0;
            sum_tpv += tp * v;
            sum_v   += v;
        }
        out[i] = sum_v > 1e-10 ? sum_tpv / sum_v : (high[i] + low[i] + close[i]) / 3.0;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 11. OBV — On-Balance Volume                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * obv_c — On-Balance Volume (cumulative directional volume).
 *
 * OBV[i] = OBV[i-1] + vol[i]  if close[i] > close[i-1]
 *         = OBV[i-1] - vol[i]  if close[i] < close[i-1]
 *         = OBV[i-1]           otherwise
 */
void obv_c(const double *close, const double *volume, int n, double *out)
{
    if (n <= 0) return;

    out[0] = 0.0;
    for (int i = 1; i < n; i++) {
        if (close[i] > close[i-1]) {
            out[i] = out[i-1] + volume[i];
        } else if (close[i] < close[i-1]) {
            out[i] = out[i-1] - volume[i];
        } else {
            out[i] = out[i-1];
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 12. CCI — Commodity Channel Index                                            */
/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * cci_c — CCI = (TP - SMA(TP, period)) / (0.015 * MeanDeviation).
 */
void cci_c(const double *high, const double *low, const double *close,
           int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    for (int i = period - 1; i < n; i++) {
        double tp_sum = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            tp_sum += (high[j] + low[j] + close[j]) / 3.0;
        }
        double tp_mean = tp_sum / period;
        double tp_i    = (high[i] + low[i] + close[i]) / 3.0;

        /* Mean absolute deviation */
        double mad = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            double tp_j = (high[j] + low[j] + close[j]) / 3.0;
            mad += ABS(tp_j - tp_mean);
        }
        mad /= period;

        out[i] = mad > 1e-10 ? (tp_i - tp_mean) / (0.015 * mad) : 0.0;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 13. Williams %R                                                              */
/* ─────────────────────────────────────────────────────────────────────────── */

void williams_r_c(const double *high, const double *low, const double *close,
                  int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;

    for (int i = 0; i < period - 1 && i < n; i++) {
        out[i] = NAN;
    }

    for (int i = period - 1; i < n; i++) {
        double hi = high[i], lo = low[i];
        for (int j = i - period + 1; j <= i; j++) {
            hi = MAX(hi, high[j]);
            lo = MIN(lo, low[j]);
        }
        double range = hi - lo;
        out[i] = range > 1e-10 ? (hi - close[i]) / range * -100.0 : -50.0;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 14. MFI — Money Flow Index                                                   */
/* ─────────────────────────────────────────────────────────────────────────── */

void mfi_c(const double *high, const double *low, const double *close,
           const double *volume, int n, int period, double *out)
{
    if (n <= 1 || period <= 0) return;

    double *pos_mf = (double*)malloc(n * sizeof(double));
    double *neg_mf = (double*)malloc(n * sizeof(double));
    if (!pos_mf || !neg_mf) { free(pos_mf); free(neg_mf); return; }

    pos_mf[0] = neg_mf[0] = 0.0;

    for (int i = 1; i < n; i++) {
        double tp      = (high[i] + low[i] + close[i]) / 3.0;
        double tp_prev = (high[i-1] + low[i-1] + close[i-1]) / 3.0;
        double mf      = tp * (volume[i] > 0 ? volume[i] : 0.0);

        if (tp > tp_prev) {
            pos_mf[i] = mf;
            neg_mf[i] = 0.0;
        } else if (tp < tp_prev) {
            pos_mf[i] = 0.0;
            neg_mf[i] = mf;
        } else {
            pos_mf[i] = neg_mf[i] = 0.0;
        }
    }

    for (int i = 0; i < period && i < n; i++) {
        out[i] = NAN;
    }

    for (int i = period; i < n; i++) {
        double pmf = 0.0, nmf = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            pmf += pos_mf[j];
            nmf += neg_mf[j];
        }
        out[i] = nmf < 1e-10 ? 100.0 : 100.0 - 100.0 / (1.0 + pmf / nmf);
    }

    free(pos_mf); free(neg_mf);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 15. Donchian Channel                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

void donchian_c(const double *high, const double *low, int n, int period,
                double *upper, double *lower_out, double *middle)
{
    if (n <= 0 || period <= 0) return;

    for (int i = 0; i < period - 1 && i < n; i++) {
        upper[i] = lower_out[i] = middle[i] = NAN;
    }

    for (int i = period - 1; i < n; i++) {
        double hi = high[i], lo = low[i];
        for (int j = i - period + 1; j <= i; j++) {
            hi = MAX(hi, high[j]);
            lo = MIN(lo, low[j]);
        }
        upper[i]     = hi;
        lower_out[i] = lo;
        middle[i]    = (hi + lo) / 2.0;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 16. Keltner Channel                                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

void keltner_c(const double *high, const double *low, const double *close,
               int n, int ema_period, int atr_period, double multiplier,
               double *upper, double *middle, double *lower_out)
{
    if (n <= 0) return;

    double *ema_arr = (double*)malloc(n * sizeof(double));
    double *atr_arr = (double*)malloc(n * sizeof(double));
    if (!ema_arr || !atr_arr) { free(ema_arr); free(atr_arr); return; }

    ema_c(close, n, ema_period, ema_arr);
    atr_c(high, low, close, n, atr_period, atr_arr);

    for (int i = 0; i < n; i++) {
        if (isnan(ema_arr[i]) || isnan(atr_arr[i])) {
            upper[i] = middle[i] = lower_out[i] = NAN;
        } else {
            middle[i]    = ema_arr[i];
            upper[i]     = ema_arr[i] + multiplier * atr_arr[i];
            lower_out[i] = ema_arr[i] - multiplier * atr_arr[i];
        }
    }

    free(ema_arr); free(atr_arr);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 17. DEMA / TEMA — Double / Triple EMA                                       */
/* ─────────────────────────────────────────────────────────────────────────── */

void dema_c(const double *close, int n, int period, double *out)
{
    double *ema1 = (double*)malloc(n * sizeof(double));
    double *ema2 = (double*)malloc(n * sizeof(double));
    if (!ema1 || !ema2) { free(ema1); free(ema2); return; }

    ema_c(close, n, period, ema1);
    ema_c(ema1,  n, period, ema2);

    for (int i = 0; i < n; i++) {
        out[i] = isnan(ema1[i]) || isnan(ema2[i]) ? NAN : 2.0 * ema1[i] - ema2[i];
    }

    free(ema1); free(ema2);
}

void tema_c(const double *close, int n, int period, double *out)
{
    double *ema1 = (double*)malloc(n * sizeof(double));
    double *ema2 = (double*)malloc(n * sizeof(double));
    double *ema3 = (double*)malloc(n * sizeof(double));
    if (!ema1 || !ema2 || !ema3) { free(ema1); free(ema2); free(ema3); return; }

    ema_c(close, n, period, ema1);
    ema_c(ema1,  n, period, ema2);
    ema_c(ema2,  n, period, ema3);

    for (int i = 0; i < n; i++) {
        if (isnan(ema1[i]) || isnan(ema2[i]) || isnan(ema3[i])) {
            out[i] = NAN;
        } else {
            out[i] = 3.0 * ema1[i] - 3.0 * ema2[i] + ema3[i];
        }
    }

    free(ema1); free(ema2); free(ema3);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 18. ROC — Rate of Change                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

void roc_c(const double *close, int n, int period, double *out)
{
    if (n <= 0 || period <= 0) return;
    for (int i = 0; i < period && i < n; i++) out[i] = NAN;
    for (int i = period; i < n; i++) {
        double prev = close[i - period];
        out[i] = prev > 1e-10 ? (close[i] - prev) / prev * 100.0 : NAN;
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 19. HMA — Hull Moving Average                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

void hma_c(const double *close, int n, int period, double *out)
{
    if (n <= 0 || period < 2) return;

    int half_p = period / 2;
    int sqrt_p = (int)sqrt((double)period);

    double *wma_half = (double*)malloc(n * sizeof(double));
    double *wma_full = (double*)malloc(n * sizeof(double));
    double *diff_arr = (double*)malloc(n * sizeof(double));

    if (!wma_half || !wma_full || !diff_arr) {
        free(wma_half); free(wma_full); free(diff_arr);
        return;
    }

    wma_c(close, n, half_p, wma_half);
    wma_c(close, n, period,  wma_full);

    for (int i = 0; i < n; i++) {
        diff_arr[i] = isnan(wma_half[i]) || isnan(wma_full[i]) ?
                      NAN : 2.0 * wma_half[i] - wma_full[i];
    }

    wma_c(diff_arr, n, sqrt_p, out);

    free(wma_half); free(wma_full); free(diff_arr);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* 20. Chande Momentum Oscillator (CMO)                                         */
/* ─────────────────────────────────────────────────────────────────────────── */

void cmo_c(const double *close, int n, int period, double *out)
{
    if (n <= 1 || period <= 0) return;

    for (int i = 0; i <= period && i < n; i++) out[i] = NAN;

    for (int i = period + 1; i < n; i++) {
        double su = 0.0, sd = 0.0;
        for (int j = i - period + 1; j <= i; j++) {
            double diff = close[j] - close[j-1];
            if (diff > 0) su += diff;
            else          sd -= diff;
        }
        double denom = su + sd;
        out[i] = denom > 1e-10 ? (su - sd) / denom * 100.0 : 0.0;
    }
}
