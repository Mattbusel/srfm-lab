"""
Mixture Density Network (MDN) for multi-modal return distributions.

Predicts Gaussian mixture parameters (pi, mu, sigma) for conditional
density estimation of financial returns.  NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Activations & utilities
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -20, 20)))


def _log_softplus(x: np.ndarray) -> np.ndarray:
    """log(softplus(x)) = log(log(1 + exp(x)))."""
    sp = _softplus(x)
    return np.log(sp + 1e-12)


def _logsumexp(x: np.ndarray, axis: int = -1,
               keepdims: bool = False) -> np.ndarray:
    mx = x.max(axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=keepdims))
    if keepdims:
        return s + mx
    return s + mx.squeeze(axis=axis)


def _gaussian_log_prob(x: np.ndarray, mu: np.ndarray,
                       sigma: np.ndarray) -> np.ndarray:
    """Log probability of x under N(mu, sigma^2)."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma + 1e-12) \
           - 0.5 * ((x - mu) / (sigma + 1e-12)) ** 2


def _gaussian_cdf(x: np.ndarray, mu: np.ndarray,
                  sigma: np.ndarray) -> np.ndarray:
    """CDF of Gaussian."""
    z = (x - mu) / (sigma + 1e-12)
    return 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation to erf."""
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = (0.254829592, -0.284496736, 1.421413741,
                           -1.453152027, 1.061405429)
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


# ---------------------------------------------------------------------------
# Dense layer with Adam
# ---------------------------------------------------------------------------

class _Dense:
    def __init__(self, in_d: int, out_d: int, act: str = "relu",
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        s = np.sqrt(2.0 / in_d)
        self.W = rng.normal(0, s, (in_d, out_d))
        self.b = np.zeros(out_d)
        self.act = act
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)
        self._x: Optional[np.ndarray] = None
        self._z: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        z = x @ self.W + self.b
        self._z = z
        if self.act == "relu":
            return _relu(z)
        if self.act == "tanh":
            return np.tanh(z)
        return z

    def backward(self, g: np.ndarray) -> np.ndarray:
        z = self._z
        if self.act == "relu":
            g = g * (z > 0).astype(np.float64)
        elif self.act == "tanh":
            g = g * (1 - np.tanh(z) ** 2)
        self.dW = self._x.T @ g / g.shape[0]
        self.db = g.mean(axis=0)
        return g @ self.W.T

    def update(self, lr: float, t: int = 1,
               b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.mW = b1 * self.mW + (1 - b1) * self.dW
        self.vW = b2 * self.vW + (1 - b2) * self.dW ** 2
        self.W -= lr * (self.mW / (1 - b1**t)) / (np.sqrt(self.vW / (1 - b2**t)) + eps)
        self.mb = b1 * self.mb + (1 - b1) * self.db
        self.vb = b2 * self.vb + (1 - b2) * self.db ** 2
        self.b -= lr * (self.mb / (1 - b1**t)) / (np.sqrt(self.vb / (1 - b2**t)) + eps)


# ---------------------------------------------------------------------------
# Mixture Density Network
# ---------------------------------------------------------------------------

class MixtureDensityNetwork:
    """Predicts Gaussian mixture parameters for conditional density p(y|x).

    Parameters
    ----------
    input_dim   : number of input features
    hidden_dims : list of hidden widths
    n_components: K  (number of Gaussian mixture components)
    seed        : random seed
    """

    def __init__(self, input_dim: int, hidden_dims: Optional[List[int]] = None,
                 n_components: int = 5, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.K = n_components
        self.input_dim = input_dim

        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Backbone
        self.layers: List[_Dense] = []
        d = input_dim
        for h in hidden_dims:
            self.layers.append(_Dense(d, h, "relu", self.rng))
            d = h

        # Output heads: pi (K), mu (K), log_sigma (K)
        self.head_pi = _Dense(d, n_components, "none", self.rng)
        self.head_mu = _Dense(d, n_components, "none", self.rng)
        self.head_log_sigma = _Dense(d, n_components, "none", self.rng)

        self.step_count = 0
        self._h: Optional[np.ndarray] = None

    # ---- forward ---------------------------------------------------------

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        pi    : (B, K) mixing coefficients (sum to 1)
        mu    : (B, K) component means
        sigma : (B, K) component std devs (> 0)
        """
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        self._h = h

        logits = self.head_pi.forward(h)
        pi = _softmax(logits, axis=-1)

        mu = self.head_mu.forward(h)
        log_sigma_raw = self.head_log_sigma.forward(h)
        sigma = _softplus(log_sigma_raw) + 1e-6

        return pi, mu, sigma

    # ---- loss ------------------------------------------------------------

    def nll_loss(self, pi: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Negative log-likelihood of Gaussian mixture.

        Parameters
        ----------
        y : (B, 1) or (B,)

        Returns
        -------
        loss     : scalar
        d_logits : gradient w.r.t. pre-softmax logits (for backprop)
        """
        if y.ndim == 1:
            y = y[:, None]

        # log p(y|x) = logsumexp_k [log pi_k + log N(y; mu_k, sigma_k)]
        log_pi = np.log(pi + 1e-12)                      # (B, K)
        log_comp = _gaussian_log_prob(y, mu, sigma)       # (B, K)
        log_joint = log_pi + log_comp                     # (B, K)
        log_p = _logsumexp(log_joint, axis=-1)            # (B,)
        loss = -log_p.mean()

        # Responsibilities (posterior)
        gamma = np.exp(log_joint - log_p[:, None])        # (B, K)

        B = y.shape[0]

        # Gradient w.r.t. pi (via logits)
        d_logits = (pi - gamma) / B                       # (B, K)

        # Gradient w.r.t. mu
        diff = (y - mu) / (sigma ** 2 + 1e-12)
        d_mu = -gamma * diff / B                          # (B, K)

        # Gradient w.r.t. log_sigma_raw  (chain through softplus)
        sp = sigma - 1e-6
        d_sigma = gamma * (((y - mu) ** 2) / (sigma ** 3 + 1e-12) - 1.0 / (sigma + 1e-12)) / B
        # softplus derivative: sigmoid(raw)
        raw = self.head_log_sigma._z
        sp_deriv = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
        d_log_sigma_raw = -d_sigma * sp_deriv

        # Cache for backward
        self._d_logits = d_logits
        self._d_mu = d_mu
        self._d_log_sigma_raw = d_log_sigma_raw

        return float(loss), gamma

    # ---- backward --------------------------------------------------------

    def backward(self) -> None:
        # Head gradients
        g_pi = self.head_pi.backward(self._d_logits)
        g_mu = self.head_mu.backward(self._d_mu)
        g_ls = self.head_log_sigma.backward(self._d_log_sigma_raw)

        g = g_pi + g_mu + g_ls
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def update(self, lr: float) -> None:
        self.step_count += 1
        t = self.step_count
        for layer in self.layers:
            layer.update(lr, t)
        self.head_pi.update(lr, t)
        self.head_mu.update(lr, t)
        self.head_log_sigma.update(lr, t)

    # ---- training step ---------------------------------------------------

    def train_step(self, x: np.ndarray, y: np.ndarray,
                   lr: float = 1e-3) -> float:
        pi, mu, sigma = self.forward(x)
        loss, _ = self.nll_loss(pi, mu, sigma, y)
        self.backward()
        self.update(lr)
        return loss

    # ---- full training ---------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200,
            batch_size: int = 64, lr: float = 1e-3,
            val_X: Optional[np.ndarray] = None,
            val_y: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict[str, List[float]]:
        N = X.shape[0]
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for ep in range(epochs):
            perm = self.rng.permutation(N)
            running = 0.0
            nb = 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                loss = self.train_step(X[idx], y[idx], lr)
                running += loss
                nb += 1
            avg = running / max(nb, 1)
            history["train_loss"].append(avg)

            if val_X is not None:
                pi, mu, sig = self.forward(val_X)
                vl, _ = self.nll_loss(pi, mu, sig, val_y)
                history["val_loss"].append(vl)

            if verbose and (ep + 1) % max(1, epochs // 10) == 0:
                msg = f"  epoch {ep+1:4d}/{epochs}  train_nll={avg:.4f}"
                if val_X is not None:
                    msg += f"  val_nll={history['val_loss'][-1]:.4f}"
                print(msg)

        return history

    # ---- inference helpers -----------------------------------------------

    def predict_params(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mixture parameters (pi, mu, sigma) for new inputs."""
        return self.forward(x)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        """Expected value E[y|x] = sum_k pi_k * mu_k."""
        pi, mu, _ = self.forward(x)
        return np.sum(pi * mu, axis=-1)

    def predict_variance(self, x: np.ndarray) -> np.ndarray:
        """Total variance Var[y|x] from law of total variance."""
        pi, mu, sigma = self.forward(x)
        mean = np.sum(pi * mu, axis=-1, keepdims=True)
        var_within = np.sum(pi * sigma ** 2, axis=-1)
        var_between = np.sum(pi * (mu - mean) ** 2, axis=-1)
        return var_within + var_between

    def mode(self, x: np.ndarray) -> np.ndarray:
        """Most likely return: mu of the component with highest pi."""
        pi, mu, _ = self.forward(x)
        idx = np.argmax(pi, axis=-1)
        return mu[np.arange(len(mu)), idx]

    def sample(self, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Draw samples from mixture distribution.

        Returns (B, n_samples).
        """
        pi, mu, sigma = self.forward(x)
        B, K = pi.shape
        out = np.empty((B, n_samples))
        for i in range(B):
            comps = self.rng.choice(K, size=n_samples, p=pi[i])
            out[i] = mu[i, comps] + sigma[i, comps] * self.rng.standard_normal(n_samples)
        return out

    # ---- tail probability ------------------------------------------------

    def tail_probability(self, x: np.ndarray,
                         threshold: float) -> np.ndarray:
        """P(y < threshold | x) from mixture CDF.

        Returns (B,).
        """
        pi, mu, sigma = self.forward(x)
        cdf_vals = _gaussian_cdf(threshold, mu, sigma)   # (B, K)
        return np.sum(pi * cdf_vals, axis=-1)

    def var_estimate(self, x: np.ndarray,
                     alpha: float = 0.05,
                     n_samples: int = 10000) -> np.ndarray:
        """Value-at-Risk at level alpha via Monte Carlo.

        Returns (B,) — the alpha-quantile of the predictive distribution.
        """
        samples = self.sample(x, n_samples)  # (B, n_samples)
        return np.quantile(samples, alpha, axis=1)

    def cvar_estimate(self, x: np.ndarray,
                      alpha: float = 0.05,
                      n_samples: int = 10000) -> np.ndarray:
        """Conditional VaR (Expected Shortfall)."""
        samples = self.sample(x, n_samples)
        var = np.quantile(samples, alpha, axis=1, keepdims=True)
        masked = np.where(samples <= var, samples, np.nan)
        return np.nanmean(masked, axis=1)

    # ---- aleatoric uncertainty -------------------------------------------

    def aleatoric_uncertainty(self, x: np.ndarray) -> np.ndarray:
        """Mixture variance as measure of aleatoric uncertainty."""
        return self.predict_variance(x)

    # ---- multi-step density forecasting ----------------------------------

    def multi_step_forecast(self, x0: np.ndarray, steps: int = 5,
                            n_samples: int = 500) -> Dict[str, np.ndarray]:
        """Iterative one-step-ahead density forecast.

        Assumes last (steps) features are lagged returns that shift
        each step.

        Returns dict with 'mean', 'std', 'samples' arrays.
        """
        B = x0.shape[0]
        D = x0.shape[1]
        all_samples = np.empty((B, steps, n_samples))
        means = np.empty((B, steps))
        stds = np.empty((B, steps))

        x_cur = x0.copy()
        for s in range(steps):
            samp = self.sample(x_cur, n_samples)          # (B, n_samples)
            all_samples[:, s, :] = samp
            means[:, s] = samp.mean(axis=1)
            stds[:, s] = samp.std(axis=1)

            # Shift lagged features: drop oldest, append new mean
            if D > 1:
                x_cur[:, :-1] = x_cur[:, 1:]
                x_cur[:, -1] = means[:, s]

        return {"mean": means, "std": stds, "samples": all_samples}


# ---------------------------------------------------------------------------
# Ensemble of MDNs for epistemic uncertainty
# ---------------------------------------------------------------------------

class MDNEnsemble:
    """Ensemble of Mixture Density Networks.

    Epistemic uncertainty estimated from disagreement across members.
    """

    def __init__(self, input_dim: int, n_components: int = 5,
                 n_members: int = 5, hidden_dims: Optional[List[int]] = None,
                 seed: int = 42):
        self.members: List[MixtureDensityNetwork] = []
        for i in range(n_members):
            self.members.append(
                MixtureDensityNetwork(input_dim, hidden_dims,
                                      n_components, seed=seed + i)
            )
        self.n_members = n_members

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train each member on a bootstrap sample."""
        N = X.shape[0]
        for i, mdn in enumerate(self.members):
            rng = np.random.default_rng(42 + i)
            idx = rng.choice(N, size=N, replace=True)
            mdn.fit(X[idx], y[idx], **kwargs)

    def predict_mean_ensemble(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ensemble mean and std (epistemic uncertainty).

        Returns (mean, epistemic_std) each of shape (B,).
        """
        preds = np.array([m.predict_mean(x) for m in self.members])  # (M, B)
        return preds.mean(axis=0), preds.std(axis=0)

    def predict_full_uncertainty(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Decompose total uncertainty into aleatoric + epistemic."""
        means = np.array([m.predict_mean(x) for m in self.members])
        ale = np.array([m.aleatoric_uncertainty(x) for m in self.members])

        ens_mean = means.mean(axis=0)
        aleatoric = ale.mean(axis=0)
        epistemic = means.var(axis=0)
        total = aleatoric + epistemic

        return {
            "mean": ens_mean,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total,
        }

    def var_ensemble(self, x: np.ndarray, alpha: float = 0.05,
                     n_samples: int = 5000) -> Dict[str, np.ndarray]:
        """VaR from pooled ensemble samples."""
        all_samp = []
        for m in self.members:
            all_samp.append(m.sample(x, n_samples))
        pooled = np.concatenate(all_samp, axis=1)  # (B, M*n_samples)
        var = np.quantile(pooled, alpha, axis=1)
        cvar_masked = np.where(pooled <= var[:, None], pooled, np.nan)
        cvar = np.nanmean(cvar_masked, axis=1)
        return {"var": var, "cvar": cvar}


# ---------------------------------------------------------------------------
# Density evaluation utilities
# ---------------------------------------------------------------------------

def mixture_pdf(y_grid: np.ndarray, pi: np.ndarray, mu: np.ndarray,
                sigma: np.ndarray) -> np.ndarray:
    """Evaluate mixture PDF on a grid of y values.

    Parameters
    ----------
    y_grid : (G,)     grid points
    pi     : (K,)     mixing weights  (single sample)
    mu     : (K,)     means
    sigma  : (K,)     std devs

    Returns
    -------
    pdf : (G,)
    """
    G = len(y_grid)
    K = len(pi)
    pdf = np.zeros(G)
    for k in range(K):
        pdf += pi[k] * np.exp(_gaussian_log_prob(y_grid, mu[k], sigma[k]))
    return pdf


def mixture_cdf(y_grid: np.ndarray, pi: np.ndarray, mu: np.ndarray,
                sigma: np.ndarray) -> np.ndarray:
    """Evaluate mixture CDF on a grid."""
    G = len(y_grid)
    K = len(pi)
    cdf = np.zeros(G)
    for k in range(K):
        cdf += pi[k] * _gaussian_cdf(y_grid, mu[k], sigma[k])
    return cdf


def pit_values(y_true: np.ndarray, pi: np.ndarray, mu: np.ndarray,
               sigma: np.ndarray) -> np.ndarray:
    """Probability Integral Transform values for calibration check.

    Parameters
    ----------
    y_true : (B,) observed values
    pi     : (B, K)
    mu     : (B, K)
    sigma  : (B, K)

    Returns
    -------
    pit : (B,) values in [0,1] — should be uniform if well-calibrated
    """
    B, K = pi.shape
    pit = np.zeros(B)
    for k in range(K):
        pit += pi[:, k] * _gaussian_cdf(y_true, mu[:, k], sigma[:, k])
    return pit


def calibration_histogram(pit: np.ndarray,
                           n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Histogram of PIT values for reliability assessment."""
    counts, edges = np.histogram(pit, bins=n_bins, range=(0, 1))
    freq = counts / counts.sum()
    return {"bin_edges": edges, "frequencies": freq,
            "expected": np.full(n_bins, 1.0 / n_bins)}


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def fit_mdn_for_returns(features: np.ndarray, returns: np.ndarray,
                        n_components: int = 5,
                        hidden_dims: Optional[List[int]] = None,
                        epochs: int = 200, lr: float = 1e-3,
                        val_frac: float = 0.2,
                        seed: int = 42) -> Dict:
    """Train MDN and return model + diagnostics.

    Parameters
    ----------
    features : (N, D) predictor matrix
    returns  : (N,) target returns

    Returns
    -------
    dict with model, history, VaR estimates, calibration info
    """
    N = features.shape[0]
    split = int(N * (1 - val_frac))
    X_tr, X_val = features[:split], features[split:]
    y_tr, y_val = returns[:split], returns[split:]

    mdn = MixtureDensityNetwork(features.shape[1], hidden_dims,
                                n_components, seed)
    history = mdn.fit(X_tr, y_tr, epochs=epochs, lr=lr,
                      val_X=X_val, val_y=y_val)

    # VaR estimates on validation set
    var_05 = mdn.var_estimate(X_val, alpha=0.05)
    var_01 = mdn.var_estimate(X_val, alpha=0.01)

    # Calibration
    pi, mu, sigma = mdn.predict_params(X_val)
    pit = pit_values(y_val, pi, mu, sigma)
    cal = calibration_histogram(pit)

    # VaR exceedance rate
    exc_05 = float(np.mean(y_val < var_05))
    exc_01 = float(np.mean(y_val < var_01))

    return {
        "model": mdn,
        "history": history,
        "var_05": var_05,
        "var_01": var_01,
        "exceedance_05": exc_05,
        "exceedance_01": exc_01,
        "pit": pit,
        "calibration": cal,
    }
