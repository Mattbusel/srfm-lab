"""
Advanced wavelet methods for financial time series analysis.

Implements DWT, MODWT, wavelet packets, denoising, cross-wavelet transforms,
multifractal analysis via wavelet leaders, and scale-dependent correlations.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy import signal
from scipy.special import comb


# ---------------------------------------------------------------------------
# Wavelet filter banks
# ---------------------------------------------------------------------------

def haar_filters():
    """Return Haar wavelet scaling and wavelet filters."""
    h = np.array([1.0, 1.0]) / np.sqrt(2)  # scaling (lowpass)
    g = np.array([1.0, -1.0]) / np.sqrt(2)  # wavelet (highpass)
    return h, g


def db4_filters():
    """Return Daubechies-4 wavelet filters (8 taps)."""
    h = np.array([
        -0.010597401784997278,
         0.032883011666982945,
         0.030841381835986965,
        -0.18703481171888114,
        -0.02798376941698385,
         0.6308807679295904,
         0.7148465705525415,
         0.23037781330885523,
    ])
    g = np.array([
        -h[7], h[6], -h[5], h[4], -h[3], h[2], -h[1], h[0]
    ])
    return h, g


def symlet4_filters():
    """Return Symlet-4 wavelet filters (8 taps)."""
    h = np.array([
        -0.07576571478927333,
        -0.02963552764599851,
         0.49761866763201545,
         0.8037387518059161,
         0.29785779560527736,
        -0.09921954357684722,
        -0.01260396726203783,
         0.03222310060404270,
    ])
    g = np.array([(-1) ** k * h[7 - k] for k in range(8)])
    return h, g


def get_filters(wavelet: str = 'haar'):
    """Get wavelet filters by name."""
    filters = {
        'haar': haar_filters,
        'db4': db4_filters,
        'sym4': symlet4_filters,
    }
    if wavelet not in filters:
        raise ValueError(f"Unknown wavelet: {wavelet}. Choose from {list(filters.keys())}")
    return filters[wavelet]()


# ---------------------------------------------------------------------------
# Discrete Wavelet Transform (DWT)
# ---------------------------------------------------------------------------

def _periodic_extend(x: np.ndarray, L: int) -> np.ndarray:
    """Periodically extend signal for circular convolution."""
    n = len(x)
    extended = np.tile(x, (L // n) + 2)
    return extended[:n + L - 1]


def dwt_1level(x: np.ndarray, h: np.ndarray, g: np.ndarray):
    """
    Single level DWT decomposition with periodic boundary.

    Returns
    -------
    approx : approximation coefficients (downsampled lowpass)
    detail : detail coefficients (downsampled highpass)
    """
    n = len(x)
    # convolve and downsample by 2
    approx_full = np.convolve(x, h[::-1], mode='full')[:n]
    detail_full = np.convolve(x, g[::-1], mode='full')[:n]
    approx = approx_full[::2]
    detail = detail_full[::2]
    return approx, detail


def idwt_1level(approx: np.ndarray, detail: np.ndarray,
                h: np.ndarray, g: np.ndarray, n_out: int) -> np.ndarray:
    """Single level inverse DWT reconstruction."""
    # upsample by 2
    n_half = len(approx)
    up_a = np.zeros(2 * n_half)
    up_d = np.zeros(2 * n_half)
    up_a[::2] = approx
    up_d[::2] = detail

    rec = np.convolve(up_a, h, mode='full')[:n_out] + \
          np.convolve(up_d, g, mode='full')[:n_out]
    return rec


def dwt(x: np.ndarray, wavelet: str = 'haar', levels: int = None) -> dict:
    """
    Multi-level discrete wavelet transform.

    Parameters
    ----------
    x : 1-D signal
    wavelet : 'haar', 'db4', or 'sym4'
    levels : number of decomposition levels (default: max possible)

    Returns
    -------
    dict with 'approx' (final approximation) and 'details' (list per level)
    """
    h, g = get_filters(wavelet)
    n = len(x)

    if levels is None:
        levels = int(np.floor(np.log2(n / (len(h) - 1))))
        levels = max(1, levels)

    current = x.copy()
    details = []

    for j in range(levels):
        approx, detail = dwt_1level(current, h, g)
        details.append(detail)
        current = approx

    return {'approx': current, 'details': details, 'levels': levels, 'wavelet': wavelet}


def idwt(decomp: dict) -> np.ndarray:
    """Inverse DWT: reconstruct signal from decomposition."""
    h, g = get_filters(decomp['wavelet'])
    current = decomp['approx']
    details = decomp['details']

    for j in range(len(details) - 1, -1, -1):
        n_out = 2 * len(current)
        if n_out > 2 * len(details[j]):
            n_out = 2 * len(details[j])
        current = idwt_1level(current, details[j], h, g, n_out)

    return current


# ---------------------------------------------------------------------------
# Maximal Overlap DWT (MODWT)
# ---------------------------------------------------------------------------

def modwt_filters(h: np.ndarray, g: np.ndarray, level: int):
    """
    Compute MODWT filters at given level via the 'a trous' algorithm.
    At level j, insert 2^(j-1) - 1 zeros between filter taps.
    """
    if level == 1:
        return h / np.sqrt(2), g / np.sqrt(2)

    # upsample by inserting zeros
    L = len(h)
    step = 2 ** (level - 1)
    new_len = (L - 1) * step + 1
    h_j = np.zeros(new_len)
    g_j = np.zeros(new_len)
    for i in range(L):
        h_j[i * step] = h[i] / np.sqrt(2)
        g_j[i * step] = g[i] / np.sqrt(2)
    return h_j, g_j


def modwt(x: np.ndarray, wavelet: str = 'haar', levels: int = None) -> dict:
    """
    Maximal Overlap Discrete Wavelet Transform (non-decimated).

    Unlike the DWT, MODWT does not downsample, producing coefficients of
    the same length as the input at each level. This makes it shift-invariant.

    Returns
    -------
    dict with 'details' (list of arrays, one per level) and 'approx'
    """
    h, g = get_filters(wavelet)
    n = len(x)

    if levels is None:
        levels = int(np.floor(np.log2(n)))
        levels = max(1, min(levels, 10))

    details = []
    current = x.copy()

    for j in range(1, levels + 1):
        h_j, g_j = modwt_filters(h, g, j)

        # circular convolution
        detail = np.zeros(n)
        approx_new = np.zeros(n)
        L_j = len(h_j)

        for t in range(n):
            d_val = 0.0
            a_val = 0.0
            for l in range(L_j):
                idx = (t - l) % n
                d_val += g_j[l] * current[idx]
                a_val += h_j[l] * current[idx]
            detail[t] = d_val
            approx_new[t] = a_val

        details.append(detail)
        current = approx_new

    return {'approx': current, 'details': details, 'levels': levels, 'wavelet': wavelet}


def modwt_variance(decomp: dict) -> np.ndarray:
    """
    Wavelet variance at each scale: Var(W_j) = (1/N) sum W_j^2.
    Returns array of variances for each level.
    """
    variances = []
    for detail in decomp['details']:
        variances.append(np.mean(detail ** 2))
    return np.array(variances)


# ---------------------------------------------------------------------------
# Wavelet packet decomposition
# ---------------------------------------------------------------------------

class WaveletPacketNode:
    """Node in a wavelet packet tree."""

    def __init__(self, coeffs: np.ndarray, level: int, index: int):
        self.coeffs = coeffs
        self.level = level
        self.index = index  # node index at this level (0 to 2^level - 1)
        self.children = []


def wavelet_packet_decompose(x: np.ndarray, wavelet: str = 'haar',
                             max_level: int = 3) -> dict:
    """
    Full wavelet packet decomposition.

    Unlike the DWT which only decomposes the approximation branch,
    wavelet packets decompose both approximation and detail at each level.

    Returns
    -------
    dict with 'tree' (nested structure) and 'leaf_coeffs' (list of arrays)
    """
    h, g = get_filters(wavelet)

    root = WaveletPacketNode(x.copy(), 0, 0)
    nodes_by_level = {0: [root]}
    all_nodes = [root]

    for level in range(1, max_level + 1):
        nodes_by_level[level] = []
        for parent in nodes_by_level[level - 1]:
            approx, detail = dwt_1level(parent.coeffs, h, g)
            node_a = WaveletPacketNode(approx, level, 2 * parent.index)
            node_d = WaveletPacketNode(detail, level, 2 * parent.index + 1)
            parent.children = [node_a, node_d]
            nodes_by_level[level].extend([node_a, node_d])
            all_nodes.extend([node_a, node_d])

    leaf_coeffs = [node.coeffs for node in nodes_by_level[max_level]]
    leaf_energies = [np.sum(c ** 2) for c in leaf_coeffs]

    return {
        'root': root,
        'nodes_by_level': nodes_by_level,
        'leaf_coeffs': leaf_coeffs,
        'leaf_energies': leaf_energies,
        'max_level': max_level,
    }


def best_basis_selection(x: np.ndarray, wavelet: str = 'haar',
                         max_level: int = 3,
                         cost_fn=None) -> dict:
    """
    Best basis selection from wavelet packet tree using an additive cost
    function (default: Shannon entropy of squared coefficients).
    """
    if cost_fn is None:
        def cost_fn(c):
            e = c ** 2
            e = e[e > 0]
            e = e / np.sum(e) if np.sum(e) > 0 else e
            return -np.sum(e * np.log(e + 1e-15))

    decomp = wavelet_packet_decompose(x, wavelet, max_level)
    nodes = decomp['nodes_by_level']

    # bottom-up cost evaluation
    costs = {}
    best_nodes = {}

    # leaf costs
    for node in nodes[max_level]:
        key = (node.level, node.index)
        costs[key] = cost_fn(node.coeffs)
        best_nodes[key] = [node]

    # propagate upward
    for level in range(max_level - 1, -1, -1):
        for node in nodes[level]:
            key = (node.level, node.index)
            parent_cost = cost_fn(node.coeffs)

            if len(node.children) == 2:
                child_key_a = (node.children[0].level, node.children[0].index)
                child_key_d = (node.children[1].level, node.children[1].index)
                children_cost = costs[child_key_a] + costs[child_key_d]

                if parent_cost <= children_cost:
                    costs[key] = parent_cost
                    best_nodes[key] = [node]
                else:
                    costs[key] = children_cost
                    best_nodes[key] = best_nodes[child_key_a] + best_nodes[child_key_d]
            else:
                costs[key] = parent_cost
                best_nodes[key] = [node]

    selected = best_nodes[(0, 0)]
    return {
        'selected_nodes': selected,
        'total_cost': costs[(0, 0)],
        'n_nodes': len(selected),
    }


# ---------------------------------------------------------------------------
# Wavelet denoising: VisuShrink, SureShrink
# ---------------------------------------------------------------------------

def visu_shrink(x: np.ndarray, wavelet: str = 'haar', levels: int = None) -> np.ndarray:
    """
    VisuShrink (universal threshold) denoising.
    Threshold = sigma * sqrt(2 * log(n)), where sigma estimated from
    finest-scale detail coefficients via MAD.
    """
    decomp = dwt(x, wavelet, levels)
    n = len(x)

    # estimate noise from finest detail
    finest = decomp['details'][0]
    sigma = np.median(np.abs(finest)) / 0.6745

    threshold = sigma * np.sqrt(2.0 * np.log(n))

    # apply soft thresholding to all detail levels
    for j in range(len(decomp['details'])):
        decomp['details'][j] = prox_l1_array(decomp['details'][j], threshold)

    return idwt(decomp)


def sure_shrink(x: np.ndarray, wavelet: str = 'haar', levels: int = None) -> np.ndarray:
    """
    SureShrink: level-dependent threshold chosen by minimizing Stein's
    Unbiased Risk Estimate (SURE) at each decomposition level.
    """
    decomp = dwt(x, wavelet, levels)
    n = len(x)

    finest = decomp['details'][0]
    sigma = np.median(np.abs(finest)) / 0.6745

    for j in range(len(decomp['details'])):
        d = decomp['details'][j]
        m = len(d)

        if sigma < 1e-12:
            continue

        d_normalized = d / sigma

        # SURE for soft threshold
        def sure_risk(lam):
            soft = np.sign(d_normalized) * np.maximum(np.abs(d_normalized) - lam, 0)
            return m - 2.0 * np.sum(np.abs(d_normalized) <= lam) + np.sum(soft ** 2)

        # search over candidate thresholds
        abs_d = np.sort(np.abs(d_normalized))
        candidates = np.concatenate([[0.0], abs_d, [np.sqrt(2.0 * np.log(m))]])
        risks = np.array([sure_risk(lam) for lam in candidates])
        best_lam = candidates[np.argmin(risks)]

        threshold = best_lam * sigma
        decomp['details'][j] = prox_l1_array(d, threshold)

    return idwt(decomp)


def prox_l1_array(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


# ---------------------------------------------------------------------------
# Cross-wavelet transform and coherence
# ---------------------------------------------------------------------------

def cross_wavelet_transform(x: np.ndarray, y: np.ndarray,
                            wavelet: str = 'haar',
                            levels: int = None) -> dict:
    """
    Cross-wavelet analysis between two signals.

    Computes wavelet cross-spectrum at each scale and the wavelet coherence.
    Uses MODWT for shift-invariant decomposition.
    """
    decomp_x = modwt(x, wavelet, levels)
    decomp_y = modwt(y, wavelet, levels)

    n_levels = decomp_x['levels']

    cross_spectra = []
    coherences = []
    phase_differences = []

    for j in range(n_levels):
        wx = decomp_x['details'][j]
        wy = decomp_y['details'][j]

        # cross-wavelet power
        cross = wx * wy
        cross_power = np.mean(cross)
        cross_spectra.append(cross)

        # wavelet coherence: smoothed cross-spectrum / sqrt(smoothed auto-spectra)
        # Use a simple running mean as smoother
        window = max(3, 2 ** (j + 1) + 1)
        kernel = np.ones(window) / window

        smooth_cross = np.convolve(cross, kernel, mode='same')
        smooth_xx = np.convolve(wx ** 2, kernel, mode='same')
        smooth_yy = np.convolve(wy ** 2, kernel, mode='same')

        denom = np.sqrt(smooth_xx * smooth_yy)
        denom = np.where(denom < 1e-15, 1e-15, denom)
        coherence = smooth_cross / denom
        coherences.append(coherence)

        # phase difference (using analytic signal approach)
        # Approximate: sign of cross-spectrum indicates phase relationship
        phase = np.arctan2(
            np.convolve(wx * np.roll(wy, 1), kernel, mode='same'),
            smooth_cross
        )
        phase_differences.append(phase)

    return {
        'cross_spectra': cross_spectra,
        'coherences': coherences,
        'phase_differences': phase_differences,
        'levels': n_levels,
    }


def wavelet_correlation(x: np.ndarray, y: np.ndarray,
                        wavelet: str = 'haar',
                        levels: int = None) -> dict:
    """
    Scale-dependent wavelet correlation.

    At each wavelet scale j, compute correlation between wavelet coefficients
    of x and y. This reveals how correlation varies across time horizons.
    """
    decomp_x = modwt(x, wavelet, levels)
    decomp_y = modwt(y, wavelet, levels)

    n_levels = decomp_x['levels']
    correlations = []
    scales = []

    for j in range(n_levels):
        wx = decomp_x['details'][j]
        wy = decomp_y['details'][j]

        # trim boundary-affected coefficients
        L_j = (2 ** (j + 1) - 1) * (len(get_filters(wavelet)[0]) - 1) + 1
        n = len(wx)
        if L_j < n:
            wx_trim = wx[L_j:]
            wy_trim = wy[L_j:]
        else:
            wx_trim = wx
            wy_trim = wy

        if len(wx_trim) > 1:
            corr = np.corrcoef(wx_trim, wy_trim)[0, 1]
        else:
            corr = np.nan
        correlations.append(corr)
        scales.append(2 ** j)

    return {
        'correlations': np.array(correlations),
        'scales': np.array(scales),
        'levels': n_levels,
    }


# ---------------------------------------------------------------------------
# Wavelet leaders for multifractal analysis
# ---------------------------------------------------------------------------

def wavelet_leaders(x: np.ndarray, wavelet: str = 'haar',
                    levels: int = None) -> dict:
    """
    Wavelet leader multifractal analysis.

    Wavelet leaders are defined as the supremum of wavelet coefficients
    over all finer scales within a dyadic interval. They provide robust
    estimates of local regularity (Holder exponents).

    Returns
    -------
    dict with 'leaders' per level, 'structure_function', 'scaling_exponents'
    """
    decomp = dwt(x, wavelet, levels)
    n_levels = decomp['levels']
    details = decomp['details']

    # compute leaders: at each scale j, the leader is the max |d| over
    # all finer scales in the corresponding dyadic interval
    leaders = []

    for j in range(n_levels):
        d_j = np.abs(details[j])
        leader_j = d_j.copy()

        # incorporate finer scales
        for jj in range(j):
            d_fine = np.abs(details[jj])
            ratio = len(d_fine) // len(d_j)
            if ratio >= 1:
                for k in range(len(d_j)):
                    start = k * ratio
                    end = min((k + 1) * ratio, len(d_fine))
                    if start < len(d_fine):
                        leader_j[k] = max(leader_j[k], np.max(d_fine[start:end]))

        leaders.append(leader_j)

    # structure function: S(q, j) = (1/n_j) sum |leader_j|^q
    q_values = np.linspace(-3, 3, 25)
    structure_fn = np.zeros((len(q_values), n_levels))

    for qi, q in enumerate(q_values):
        for j in range(n_levels):
            lj = leaders[j]
            lj = lj[lj > 0]
            if len(lj) > 0:
                structure_fn[qi, j] = np.mean(lj ** q)

    # scaling exponents: log S(q,j) ~ zeta(q) * j
    scales = np.arange(1, n_levels + 1)
    zeta = np.zeros(len(q_values))

    for qi in range(len(q_values)):
        s_vals = structure_fn[qi, :]
        valid = s_vals > 0
        if np.sum(valid) >= 2:
            log_s = np.log2(s_vals[valid])
            log_j = scales[valid]
            # linear regression
            A = np.vstack([log_j, np.ones(np.sum(valid))]).T
            result = np.linalg.lstsq(A, log_s, rcond=None)
            zeta[qi] = result[0][0]

    # multifractal spectrum via Legendre transform
    # h(q) = d zeta / dq,  D(h) = q*h - zeta(q) + 1
    dq = q_values[1] - q_values[0]
    h_q = np.gradient(zeta, dq)
    D_h = q_values * h_q - zeta + 1.0

    return {
        'leaders': leaders,
        'q_values': q_values,
        'structure_function': structure_fn,
        'scaling_exponents': zeta,
        'singularity_spectrum_h': h_q,
        'singularity_spectrum_D': D_h,
    }


# ---------------------------------------------------------------------------
# Application: multi-scale trend detection
# ---------------------------------------------------------------------------

def multiscale_trend(x: np.ndarray, wavelet: str = 'db4',
                     levels: int = None,
                     trend_levels: list = None) -> dict:
    """
    Multi-scale trend decomposition.

    Decomposes signal into components at different time scales, identifies
    trend as the smooth (approximation) component, and separates
    short-term noise from medium-term cycles.

    Parameters
    ----------
    x : 1-D price/return series
    wavelet : wavelet name
    levels : number of decomposition levels
    trend_levels : which levels to include in 'trend' (default: coarsest 2)

    Returns
    -------
    dict with 'trend', 'cycles', 'noise', 'components'
    """
    decomp = modwt(x, wavelet, levels)
    n = len(x)
    n_levels = decomp['levels']

    if trend_levels is None:
        trend_levels = list(range(max(0, n_levels - 2), n_levels))

    # reconstruct components
    trend = decomp['approx'].copy()
    for j in trend_levels:
        if j < n_levels:
            trend = trend + decomp['details'][j]

    # noise: finest scale details
    noise_levels = [0] if n_levels > 1 else []
    noise = np.zeros(n)
    for j in noise_levels:
        noise += decomp['details'][j]

    # cycles: remaining levels
    cycle_levels = [j for j in range(n_levels)
                    if j not in trend_levels and j not in noise_levels]
    cycles = np.zeros(n)
    for j in cycle_levels:
        cycles += decomp['details'][j]

    # variance contribution
    total_var = np.var(x)
    components = []
    for j in range(n_levels):
        var_j = np.var(decomp['details'][j])
        components.append({
            'level': j + 1,
            'scale': 2 ** (j + 1),
            'variance': var_j,
            'variance_share': var_j / total_var if total_var > 0 else 0,
            'detail': decomp['details'][j],
        })

    return {
        'trend': trend,
        'cycles': cycles,
        'noise': noise,
        'approx': decomp['approx'],
        'components': components,
        'trend_levels': trend_levels,
        'noise_levels': noise_levels,
        'cycle_levels': cycle_levels,
    }


# ---------------------------------------------------------------------------
# Wavelet-based VaR
# ---------------------------------------------------------------------------

def wavelet_var_decomposition(returns: np.ndarray, wavelet: str = 'haar',
                              levels: int = None,
                              confidence: float = 0.95) -> dict:
    """
    Wavelet-based Value at Risk decomposition.

    Decomposes return variance by frequency band (wavelet scale),
    then computes VaR contributions from each band assuming Gaussian
    returns within each band.

    Parameters
    ----------
    returns : 1-D return series
    wavelet : wavelet name
    levels : decomposition levels
    confidence : VaR confidence level

    Returns
    -------
    dict with per-scale variance, VaR contributions, total VaR
    """
    decomp = modwt(returns, wavelet, levels)
    n_levels = decomp['levels']
    n = len(returns)

    from scipy.stats import norm
    z = norm.ppf(confidence)

    scale_variances = []
    scale_var_contributions = []
    total_variance = np.var(returns)

    for j in range(n_levels):
        d = decomp['details'][j]
        var_j = np.mean(d ** 2)  # wavelet variance at scale j
        scale_variances.append(var_j)
        # VaR contribution assuming independence across scales
        var_contribution = z * np.sqrt(var_j)
        scale_var_contributions.append(var_contribution)

    # approximation variance
    var_approx = np.mean(decomp['approx'] ** 2)

    # total VaR
    total_wavelet_var = sum(scale_variances) + var_approx
    total_var = z * np.sqrt(total_wavelet_var)

    # parametric VaR (for comparison)
    parametric_var = z * np.std(returns)

    return {
        'scale_variances': np.array(scale_variances),
        'approx_variance': var_approx,
        'total_wavelet_variance': total_wavelet_var,
        'scale_var_contributions': np.array(scale_var_contributions),
        'total_VaR': total_var,
        'parametric_VaR': parametric_var,
        'variance_shares': np.array(scale_variances) / total_wavelet_var,
        'scales': 2 ** np.arange(1, n_levels + 1),
        'confidence': confidence,
    }


# ---------------------------------------------------------------------------
# Wavelet variance decomposition (detailed)
# ---------------------------------------------------------------------------

def wavelet_variance_analysis(x: np.ndarray, wavelet: str = 'haar',
                              levels: int = None,
                              ci_method: str = 'chi2',
                              confidence: float = 0.95) -> dict:
    """
    Detailed wavelet variance analysis with confidence intervals.

    Parameters
    ----------
    x : 1-D time series
    ci_method : 'chi2' for chi-squared based CI, 'bootstrap' for bootstrap
    confidence : CI confidence level
    """
    decomp = modwt(x, wavelet, levels)
    n = len(x)
    n_levels = decomp['levels']
    h, g = get_filters(wavelet)
    L = len(h)

    results = []
    for j in range(n_levels):
        d = decomp['details'][j]
        L_j = (2 ** (j + 1) - 1) * (L - 1) + 1
        n_eff = n - L_j + 1

        if n_eff < 1:
            n_eff = len(d)

        var_j = np.mean(d ** 2)

        # confidence intervals
        if ci_method == 'chi2':
            from scipy.stats import chi2
            alpha = 1.0 - confidence
            # effective degrees of freedom (approximate)
            eta_j = max(2, n_eff // (2 ** (j + 1)))
            ci_lower = eta_j * var_j / chi2.ppf(1 - alpha / 2, eta_j)
            ci_upper = eta_j * var_j / chi2.ppf(alpha / 2, eta_j)
        else:
            # bootstrap
            n_boot = 1000
            boot_vars = np.zeros(n_boot)
            for b in range(n_boot):
                idx = np.random.choice(len(d), size=len(d), replace=True)
                boot_vars[b] = np.mean(d[idx] ** 2)
            alpha = 1.0 - confidence
            ci_lower = np.percentile(boot_vars, 100 * alpha / 2)
            ci_upper = np.percentile(boot_vars, 100 * (1 - alpha / 2))

        results.append({
            'level': j + 1,
            'scale': 2 ** (j + 1),
            'variance': var_j,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'effective_n': n_eff,
        })

    return {
        'variance_by_scale': results,
        'total_variance': np.var(x),
        'wavelet': wavelet,
        'n_levels': n_levels,
    }
