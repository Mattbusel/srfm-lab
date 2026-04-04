/*
 * indicators_py.c — Python C extension wrapper for fast_indicators.
 *
 * Wraps indicators.c and bh_fast.c with CPython API.
 * All functions accept numpy arrays (PyObject*) via buffer protocol.
 *
 * Build: python setup.py build_ext --inplace
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Include the actual implementations */
#include "indicators.c"
#include "bh_fast.c"

/* ─────────────────────────────────────────────────────────────────────────── */
/* Numpy minimal import (buffer protocol only, no numpy.h needed)              */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Helper: extract double* from a Python buffer object */
static int get_buffer(PyObject *obj, Py_buffer *view, double **data, Py_ssize_t *len)
{
    if (PyObject_GetBuffer(obj, view, PyBUF_SIMPLE | PyBUF_FORMAT) < 0)
        return -1;

    if (view->format && view->format[0] != 'd') {
        PyErr_SetString(PyExc_TypeError,
            "fast_indicators: expected float64 (double) array");
        PyBuffer_Release(view);
        return -1;
    }

    *data = (double*)view->buf;
    *len  = view->len / sizeof(double);
    return 0;
}

/* Helper: create a new Python bytes buffer of n doubles, initialised to NaN */
static PyObject *make_output(Py_ssize_t n, double **data_out)
{
    PyObject *bytes = PyBytes_FromStringAndSize(NULL, n * sizeof(double));
    if (!bytes) return NULL;
    double *d = (double*)PyBytes_AS_STRING(bytes);
    for (Py_ssize_t i = 0; i < n; i++) d[i] = NAN;
    *data_out = d;
    return bytes;
}

/*
 * All returned arrays are PyBytes objects containing raw double data.
 * The Python wrapper in __init__.py converts these to numpy arrays via
 * numpy.frombuffer().
 */

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_ema(close, period) → bytes                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_ema(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;

    if (!PyArg_ParseTuple(args, "Oi", &close_obj, &period))
        return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (!result) { PyBuffer_Release(&view); return NULL; }

    ema_c(close, (int)n, period, out_data);

    PyBuffer_Release(&view);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_sma(close, period) → bytes                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_sma(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "Oi", &close_obj, &period)) return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (!result) { PyBuffer_Release(&view); return NULL; }

    sma_c(close, (int)n, period, out_data);

    PyBuffer_Release(&view);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_rsi(close, period) → bytes                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_rsi(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "Oi", &close_obj, &period)) return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (!result) { PyBuffer_Release(&view); return NULL; }

    rsi_c(close, (int)n, period, out_data);

    PyBuffer_Release(&view);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_macd(close, fast, slow, signal) → (macd_bytes, signal_bytes, hist_bytes) */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_macd(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int fast, slow, signal;
    if (!PyArg_ParseTuple(args, "Oiii", &close_obj, &fast, &slow, &signal))
        return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *macd_d, *sig_d, *hist_d;
    PyObject *macd_b = make_output(n, &macd_d);
    PyObject *sig_b  = make_output(n, &sig_d);
    PyObject *hist_b = make_output(n, &hist_d);

    if (!macd_b || !sig_b || !hist_b) {
        Py_XDECREF(macd_b); Py_XDECREF(sig_b); Py_XDECREF(hist_b);
        PyBuffer_Release(&view);
        return NULL;
    }

    macd_c(close, (int)n, fast, slow, signal, macd_d, sig_d, hist_d);

    PyBuffer_Release(&view);
    return PyTuple_Pack(3, macd_b, sig_b, hist_b);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_atr(high, low, close, period) → bytes                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_atr(PyObject *self, PyObject *args)
{
    PyObject *high_obj, *low_obj, *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "OOOi", &high_obj, &low_obj, &close_obj, &period))
        return NULL;

    Py_buffer vh, vl, vc;
    double *high, *low, *close;
    Py_ssize_t nh, nl, nc;

    if (get_buffer(high_obj,  &vh, &high,  &nh) < 0) return NULL;
    if (get_buffer(low_obj,   &vl, &low,   &nl) < 0) { PyBuffer_Release(&vh); return NULL; }
    if (get_buffer(close_obj, &vc, &close, &nc) < 0) {
        PyBuffer_Release(&vh); PyBuffer_Release(&vl); return NULL;
    }

    int n = (int)MIN(MIN(nh, nl), nc);
    double *out_data;
    PyObject *result = make_output(n, &out_data);

    if (result) {
        atr_c(high, low, close, n, period, out_data);
    }

    PyBuffer_Release(&vh); PyBuffer_Release(&vl); PyBuffer_Release(&vc);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_bollinger(close, period, num_std) → (upper, middle, lower)               */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_bollinger(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;
    double num_std;
    if (!PyArg_ParseTuple(args, "Oid", &close_obj, &period, &num_std)) return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *upper_d, *mid_d, *lower_d;
    PyObject *upper_b  = make_output(n, &upper_d);
    PyObject *mid_b    = make_output(n, &mid_d);
    PyObject *lower_b  = make_output(n, &lower_d);

    if (!upper_b || !mid_b || !lower_b) {
        Py_XDECREF(upper_b); Py_XDECREF(mid_b); Py_XDECREF(lower_b);
        PyBuffer_Release(&view);
        return NULL;
    }

    bollinger_c(close, (int)n, period, num_std, upper_d, mid_d, lower_d);

    PyBuffer_Release(&view);
    return PyTuple_Pack(3, upper_b, mid_b, lower_b);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_adx(high, low, close, period) → bytes                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_adx(PyObject *self, PyObject *args)
{
    PyObject *high_obj, *low_obj, *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "OOOi", &high_obj, &low_obj, &close_obj, &period))
        return NULL;

    Py_buffer vh, vl, vc;
    double *high, *low, *close;
    Py_ssize_t nh, nl, nc;

    if (get_buffer(high_obj,  &vh, &high,  &nh) < 0) return NULL;
    if (get_buffer(low_obj,   &vl, &low,   &nl) < 0) { PyBuffer_Release(&vh); return NULL; }
    if (get_buffer(close_obj, &vc, &close, &nc) < 0) {
        PyBuffer_Release(&vh); PyBuffer_Release(&vl); return NULL;
    }

    int n = (int)MIN(MIN(nh, nl), nc);
    double *out_data;
    PyObject *result = make_output(n, &out_data);

    if (result) {
        adx_c(high, low, close, n, period, out_data);
    }

    PyBuffer_Release(&vh); PyBuffer_Release(&vl); PyBuffer_Release(&vc);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_stochastic(high, low, close, k_period, d_period) → (k_bytes, d_bytes)   */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_stochastic(PyObject *self, PyObject *args)
{
    PyObject *high_obj, *low_obj, *close_obj;
    int k_period, d_period;
    if (!PyArg_ParseTuple(args, "OOOii", &high_obj, &low_obj, &close_obj,
                           &k_period, &d_period)) return NULL;

    Py_buffer vh, vl, vc;
    double *high, *low, *close;
    Py_ssize_t nh, nl, nc;

    if (get_buffer(high_obj,  &vh, &high,  &nh) < 0) return NULL;
    if (get_buffer(low_obj,   &vl, &low,   &nl) < 0) { PyBuffer_Release(&vh); return NULL; }
    if (get_buffer(close_obj, &vc, &close, &nc) < 0) {
        PyBuffer_Release(&vh); PyBuffer_Release(&vl); return NULL;
    }

    int n = (int)MIN(MIN(nh, nl), nc);
    double *k_d, *d_d;
    PyObject *k_b = make_output(n, &k_d);
    PyObject *d_b = make_output(n, &d_d);

    if (k_b && d_b) {
        stochastic_c(high, low, close, n, k_period, d_period, k_d, d_d);
    }

    PyBuffer_Release(&vh); PyBuffer_Release(&vl); PyBuffer_Release(&vc);
    if (!k_b || !d_b) { Py_XDECREF(k_b); Py_XDECREF(d_b); return NULL; }
    return PyTuple_Pack(2, k_b, d_b);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_vwap(high, low, close, volume) → bytes                                   */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_vwap(PyObject *self, PyObject *args)
{
    PyObject *high_obj, *low_obj, *close_obj, *vol_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &high_obj, &low_obj, &close_obj, &vol_obj))
        return NULL;

    Py_buffer vh, vl, vc, vv;
    double *high, *low, *close, *vol;
    Py_ssize_t nh, nl, nc, nv;

    if (get_buffer(high_obj,  &vh, &high,  &nh) < 0) return NULL;
    if (get_buffer(low_obj,   &vl, &low,   &nl) < 0) { PyBuffer_Release(&vh); return NULL; }
    if (get_buffer(close_obj, &vc, &close, &nc) < 0) {
        PyBuffer_Release(&vh); PyBuffer_Release(&vl); return NULL;
    }
    if (get_buffer(vol_obj,   &vv, &vol,   &nv) < 0) {
        PyBuffer_Release(&vh); PyBuffer_Release(&vl); PyBuffer_Release(&vc);
        return NULL;
    }

    int n = (int)MIN(MIN(nh, nl), MIN(nc, nv));
    double *out_data;
    PyObject *result = make_output(n, &out_data);

    if (result) {
        vwap_c(high, low, close, vol, n, out_data);
    }

    PyBuffer_Release(&vh); PyBuffer_Release(&vl);
    PyBuffer_Release(&vc); PyBuffer_Release(&vv);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_obv(close, volume) → bytes                                               */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_obv(PyObject *self, PyObject *args)
{
    PyObject *close_obj, *vol_obj;
    if (!PyArg_ParseTuple(args, "OO", &close_obj, &vol_obj)) return NULL;

    Py_buffer vc, vv;
    double *close, *vol;
    Py_ssize_t nc, nv;

    if (get_buffer(close_obj, &vc, &close, &nc) < 0) return NULL;
    if (get_buffer(vol_obj,   &vv, &vol,   &nv) < 0) { PyBuffer_Release(&vc); return NULL; }

    int n = (int)MIN(nc, nv);
    double *out_data;
    PyObject *result = make_output(n, &out_data);

    if (result) {
        obv_c(close, vol, n, out_data);
    }

    PyBuffer_Release(&vc); PyBuffer_Release(&vv);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_bh_series(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req)           */
/*   → (masses_bytes, active_bytes, ctl_bytes)                                 */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_bh_series(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    double cf, bh_form, bh_decay, bh_collapse;
    int ctl_req;

    if (!PyArg_ParseTuple(args, "Oddddi",
                           &close_obj, &cf, &bh_form, &bh_decay, &bh_collapse, &ctl_req))
        return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    /* Allocate output arrays */
    double *masses_d = (double*)malloc(n * sizeof(double));
    int    *active_i = (int*)   malloc(n * sizeof(int));
    int    *ctl_i    = (int*)   malloc(n * sizeof(int));

    if (!masses_d || !active_i || !ctl_i) {
        free(masses_d); free(active_i); free(ctl_i);
        PyBuffer_Release(&view);
        PyErr_NoMemory();
        return NULL;
    }

    bh_series(close, (int)n, cf, bh_form, bh_decay, bh_collapse, ctl_req,
              masses_d, active_i, ctl_i);

    PyBuffer_Release(&view);

    /* Convert to Python bytes */
    PyObject *masses_b = PyBytes_FromStringAndSize((char*)masses_d, n * sizeof(double));
    PyObject *active_b = PyBytes_FromStringAndSize((char*)active_i, n * sizeof(int));
    PyObject *ctl_b    = PyBytes_FromStringAndSize((char*)ctl_i,    n * sizeof(int));

    free(masses_d); free(active_i); free(ctl_i);

    if (!masses_b || !active_b || !ctl_b) {
        Py_XDECREF(masses_b); Py_XDECREF(active_b); Py_XDECREF(ctl_b);
        return NULL;
    }

    return PyTuple_Pack(3, masses_b, active_b, ctl_b);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_bh_backtest(closes, highs, lows, cf, bh_form, bh_decay, bh_collapse,    */
/*                ctl_req, long_only, commission, slippage)                    */
/*   → (equity_bytes, positions_bytes, n_trades)                               */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_bh_backtest(PyObject *self, PyObject *args)
{
    PyObject *close_obj, *high_obj, *low_obj;
    double cf, bh_form, bh_decay, bh_collapse, commission, slippage;
    int ctl_req, long_only;

    if (!PyArg_ParseTuple(args, "OOOddddiidd",
                           &close_obj, &high_obj, &low_obj,
                           &cf, &bh_form, &bh_decay, &bh_collapse,
                           &ctl_req, &long_only, &commission, &slippage))
        return NULL;

    Py_buffer vc, vh, vl;
    double *close, *high, *low;
    Py_ssize_t nc, nh, nl;

    if (get_buffer(close_obj, &vc, &close, &nc) < 0) return NULL;
    if (get_buffer(high_obj,  &vh, &high,  &nh) < 0) { PyBuffer_Release(&vc); return NULL; }
    if (get_buffer(low_obj,   &vl, &low,   &nl) < 0) {
        PyBuffer_Release(&vc); PyBuffer_Release(&vh); return NULL;
    }

    int n = (int)MIN(MIN(nc, nh), nl);

    double *equity    = (double*)malloc(n * sizeof(double));
    int    *positions = (int*)   malloc(n * sizeof(int));
    int     max_trades = n / 2 + 1;
    TradeRecord *trades = (TradeRecord*)malloc(max_trades * sizeof(TradeRecord));
    int     trade_count = 0;

    if (!equity || !positions || !trades) {
        free(equity); free(positions); free(trades);
        PyBuffer_Release(&vc); PyBuffer_Release(&vh); PyBuffer_Release(&vl);
        PyErr_NoMemory();
        return NULL;
    }

    bh_backtest_c(close, high, low, n,
                  cf, bh_form, bh_decay, bh_collapse, ctl_req, long_only,
                  commission, slippage,
                  equity, positions, &trade_count, trades);

    PyBuffer_Release(&vc); PyBuffer_Release(&vh); PyBuffer_Release(&vl);

    PyObject *eq_b  = PyBytes_FromStringAndSize((char*)equity,    n * sizeof(double));
    PyObject *pos_b = PyBytes_FromStringAndSize((char*)positions, n * sizeof(int));
    PyObject *tc_py = PyLong_FromLong(trade_count);

    free(equity); free(positions); free(trades);

    if (!eq_b || !pos_b || !tc_py) {
        Py_XDECREF(eq_b); Py_XDECREF(pos_b); Py_XDECREF(tc_py);
        return NULL;
    }

    return PyTuple_Pack(3, eq_b, pos_b, tc_py);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* py_cci, py_roc, py_wma — additional indicators                              */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyObject *py_cci(PyObject *self, PyObject *args)
{
    PyObject *high_obj, *low_obj, *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "OOOi", &high_obj, &low_obj, &close_obj, &period))
        return NULL;

    Py_buffer vh, vl, vc;
    double *high, *low, *close;
    Py_ssize_t nh, nl, nc;

    if (get_buffer(high_obj,  &vh, &high,  &nh) < 0) return NULL;
    if (get_buffer(low_obj,   &vl, &low,   &nl) < 0) { PyBuffer_Release(&vh); return NULL; }
    if (get_buffer(close_obj, &vc, &close, &nc) < 0) {
        PyBuffer_Release(&vh); PyBuffer_Release(&vl); return NULL;
    }

    int n = (int)MIN(MIN(nh, nl), nc);
    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (result) cci_c(high, low, close, n, period, out_data);

    PyBuffer_Release(&vh); PyBuffer_Release(&vl); PyBuffer_Release(&vc);
    return result;
}

static PyObject *py_roc(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "Oi", &close_obj, &period)) return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (result) roc_c(close, (int)n, period, out_data);

    PyBuffer_Release(&view);
    return result;
}

static PyObject *py_wma(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "Oi", &close_obj, &period)) return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (result) wma_c(close, (int)n, period, out_data);

    PyBuffer_Release(&view);
    return result;
}

static PyObject *py_hma(PyObject *self, PyObject *args)
{
    PyObject *close_obj;
    int period;
    if (!PyArg_ParseTuple(args, "Oi", &close_obj, &period)) return NULL;

    Py_buffer view;
    double *close; Py_ssize_t n;
    if (get_buffer(close_obj, &view, &close, &n) < 0) return NULL;

    double *out_data;
    PyObject *result = make_output(n, &out_data);
    if (result) hma_c(close, (int)n, period, out_data);

    PyBuffer_Release(&view);
    return result;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Module method table                                                          */
/* ─────────────────────────────────────────────────────────────────────────── */

static PyMethodDef FastIndicatorsMethods[] = {
    /* Core indicators */
    {"_ema",        py_ema,         METH_VARARGS, "EMA(close, period) -> bytes"},
    {"_sma",        py_sma,         METH_VARARGS, "SMA(close, period) -> bytes"},
    {"_wma",        py_wma,         METH_VARARGS, "WMA(close, period) -> bytes"},
    {"_hma",        py_hma,         METH_VARARGS, "HMA(close, period) -> bytes"},
    {"_rsi",        py_rsi,         METH_VARARGS, "RSI(close, period) -> bytes"},
    {"_macd",       py_macd,        METH_VARARGS, "MACD(close, fast, slow, signal) -> (macd, signal, hist)"},
    {"_atr",        py_atr,         METH_VARARGS, "ATR(high, low, close, period) -> bytes"},
    {"_bollinger",  py_bollinger,   METH_VARARGS, "Bollinger(close, period, num_std) -> (upper, mid, lower)"},
    {"_adx",        py_adx,         METH_VARARGS, "ADX(high, low, close, period) -> bytes"},
    {"_stochastic", py_stochastic,  METH_VARARGS, "Stochastic(high, low, close, k, d) -> (k, d)"},
    {"_vwap",       py_vwap,        METH_VARARGS, "VWAP(high, low, close, volume) -> bytes"},
    {"_obv",        py_obv,         METH_VARARGS, "OBV(close, volume) -> bytes"},
    {"_cci",        py_cci,         METH_VARARGS, "CCI(high, low, close, period) -> bytes"},
    {"_roc",        py_roc,         METH_VARARGS, "ROC(close, period) -> bytes"},
    /* BH physics */
    {"_bh_series",  py_bh_series,   METH_VARARGS,
     "bh_series(closes, cf, bh_form, bh_decay, bh_collapse, ctl_req) -> (masses, active, ctl)"},
    {"_bh_backtest",py_bh_backtest, METH_VARARGS,
     "bh_backtest(closes, highs, lows, cf, bh_form, bh_decay, bh_collapse, ctl_req, long_only, commission, slippage) -> (equity, positions, n_trades)"},
    {NULL, NULL, 0, NULL}
};

/* ─────────────────────────────────────────────────────────────────────────── */
/* Module init                                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

static struct PyModuleDef fast_indicators_module = {
    PyModuleDef_HEAD_INIT,
    "fast_indicators",        /* module name */
    "Fast C indicator library for SRFM quant lab.\n"
    "All arrays use float64 (double). Returns raw bytes to be converted "
    "via numpy.frombuffer().",
    -1,
    FastIndicatorsMethods
};

PyMODINIT_FUNC PyInit_fast_indicators(void)
{
    return PyModule_Create(&fast_indicators_module);
}
