"""
Contrastive Learning for Market Regime Representation.

SimCLR-style framework with time-series augmentations, NT-Xent loss,
projection head, nearest-neighbor classifier, and similarity search.
NumPy only.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Activations / helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def _l2_normalize(x: np.ndarray, axis: int = -1,
                  eps: float = 1e-8) -> np.ndarray:
    norm = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True) + eps)
    return x / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix. (N, D) x (M, D) -> (N, M)."""
    a_n = _l2_normalize(a)
    b_n = _l2_normalize(b)
    return a_n @ b_n.T


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Time-series augmentations
# ---------------------------------------------------------------------------

class TimeSeriesAugmentor:
    """Suite of augmentations for financial time-series windows."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def jitter(self, x: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        """Add Gaussian noise."""
        return x + self.rng.normal(0, sigma, x.shape)

    def magnitude_scale(self, x: np.ndarray,
                        sigma: float = 0.1) -> np.ndarray:
        """Multiply by random scalar per sample."""
        if x.ndim == 1:
            scale = 1.0 + self.rng.normal(0, sigma)
            return x * scale
        scales = 1.0 + self.rng.normal(0, sigma, (x.shape[0], 1))
        return x * scales

    def time_warp(self, x: np.ndarray, sigma: float = 0.2,
                  n_knots: int = 4) -> np.ndarray:
        """Smooth time warping via cubic spline-like distortion."""
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        else:
            squeeze = False

        B, T = x.shape
        orig = np.linspace(0, 1, T)
        out = np.empty_like(x)

        for i in range(B):
            knot_pos = np.linspace(0, 1, n_knots + 2)
            knot_vals = knot_pos.copy()
            knot_vals[1:-1] += self.rng.normal(0, sigma, n_knots)
            knot_vals = np.sort(np.clip(knot_vals, 0, 1))
            warped = np.interp(orig, knot_pos, knot_vals)
            warped = np.clip(warped, 0, 1) * (T - 1)
            idx_low = np.floor(warped).astype(int)
            idx_high = np.minimum(idx_low + 1, T - 1)
            frac = warped - idx_low
            out[i] = x[i, idx_low] * (1 - frac) + x[i, idx_high] * frac

        return out.squeeze(0) if squeeze else out

    def permutation(self, x: np.ndarray,
                    n_segments: int = 4) -> np.ndarray:
        """Randomly permute temporal segments."""
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        else:
            squeeze = False

        B, T = x.shape
        seg_len = T // n_segments
        out = np.empty_like(x)

        for i in range(B):
            segs = []
            for s in range(n_segments):
                start = s * seg_len
                end = start + seg_len if s < n_segments - 1 else T
                segs.append(x[i, start:end])
            order = self.rng.permutation(len(segs))
            out[i] = np.concatenate([segs[o] for o in order])[:T]

        return out.squeeze(0) if squeeze else out

    def window_crop(self, x: np.ndarray,
                    crop_ratio: float = 0.8) -> np.ndarray:
        """Random crop and resize back to original length."""
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        else:
            squeeze = False

        B, T = x.shape
        crop_len = max(1, int(T * crop_ratio))
        out = np.empty_like(x)

        for i in range(B):
            start = self.rng.integers(0, T - crop_len + 1)
            cropped = x[i, start:start + crop_len]
            out[i] = np.interp(np.linspace(0, 1, T),
                               np.linspace(0, 1, crop_len), cropped)

        return out.squeeze(0) if squeeze else out

    def augment(self, x: np.ndarray,
                methods: Optional[List[str]] = None) -> np.ndarray:
        """Apply a random subset of augmentations."""
        if methods is None:
            methods = self.rng.choice(
                ["jitter", "magnitude_scale", "time_warp", "permutation",
                 "window_crop"],
                size=2, replace=False,
            ).tolist()
        out = x.copy()
        for m in methods:
            out = getattr(self, m)(out)
        return out


# ---------------------------------------------------------------------------
# Dense layer
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
        return z

    def backward(self, g: np.ndarray) -> np.ndarray:
        z = self._z
        if self.act == "relu":
            g = g * (z > 0).astype(np.float64)
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
# Encoder + Projection Head
# ---------------------------------------------------------------------------

class Encoder:
    """Dense encoder mapping time-series windows to embeddings."""

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 embed_dim: int = 64,
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        self.layers: List[_Dense] = []
        d = input_dim
        for h in hidden_dims:
            self.layers.append(_Dense(d, h, "relu", rng))
            d = h
        self.layers.append(_Dense(d, embed_dim, "none", rng))
        self.embed_dim = embed_dim

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

    def backward(self, g: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def update(self, lr: float, t: int = 1):
        for layer in self.layers:
            layer.update(lr, t)


class ProjectionHead:
    """MLP projection head (encoder output -> contrastive space)."""

    def __init__(self, embed_dim: int, proj_dim: int = 32,
                 hidden_dim: int = 64,
                 rng: Optional[np.random.Generator] = None):
        rng = rng or np.random.default_rng()
        self.fc1 = _Dense(embed_dim, hidden_dim, "relu", rng)
        self.fc2 = _Dense(hidden_dim, proj_dim, "none", rng)

    def forward(self, h: np.ndarray) -> np.ndarray:
        return self.fc2.forward(self.fc1.forward(h))

    def backward(self, g: np.ndarray) -> np.ndarray:
        g = self.fc2.backward(g)
        return self.fc1.backward(g)

    def update(self, lr: float, t: int = 1):
        self.fc1.update(lr, t)
        self.fc2.update(lr, t)


# ---------------------------------------------------------------------------
# NT-Xent Loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z_i: np.ndarray, z_j: np.ndarray,
                 temperature: float = 0.5
                 ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Normalized Temperature-scaled Cross-Entropy loss.

    Parameters
    ----------
    z_i, z_j : (B, D) L2-normalized projection vectors for two views

    Returns
    -------
    loss       : scalar
    grad_z_i   : (B, D)
    grad_z_j   : (B, D)
    """
    B = z_i.shape[0]
    z_i = _l2_normalize(z_i)
    z_j = _l2_normalize(z_j)

    # Concatenate: [z_i; z_j] => (2B, D)
    z = np.concatenate([z_i, z_j], axis=0)  # (2B, D)
    N = 2 * B

    # Similarity matrix
    sim = (z @ z.T) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = np.eye(N, dtype=bool)
    sim[mask] = -1e9

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = np.concatenate([np.arange(B, N), np.arange(B)])

    # Softmax rows
    probs = _softmax(sim, axis=-1)  # (2B, 2B)

    # Loss = -log(prob of positive pair)
    log_probs = np.log(probs + 1e-12)
    loss = 0.0
    for n in range(N):
        loss -= log_probs[n, labels[n]]
    loss /= N

    # Gradient w.r.t. z (through softmax and cosine sim)
    grad_sim = probs.copy()  # (2B, 2B)
    for n in range(N):
        grad_sim[n, labels[n]] -= 1.0
    grad_sim /= N

    grad_z = (grad_sim + grad_sim.T) @ z / temperature  # (2B, D)

    # Gradient of L2 normalization
    grad_z_i_raw = grad_z[:B]
    grad_z_j_raw = grad_z[B:]

    grad_z_i = _grad_l2_norm(z_i, grad_z_i_raw)
    grad_z_j = _grad_l2_norm(z_j, grad_z_j_raw)

    return float(loss), grad_z_i, grad_z_j


def _grad_l2_norm(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Gradient through L2 normalization."""
    norm = np.sqrt(np.sum(x ** 2, axis=-1, keepdims=True) + 1e-8)
    x_hat = x / norm
    return (grad - x_hat * np.sum(x_hat * grad, axis=-1, keepdims=True)) / norm


# ---------------------------------------------------------------------------
# Contrastive Learning Framework
# ---------------------------------------------------------------------------

class ContrastiveLearner:
    """SimCLR-style contrastive learner for market regime embeddings.

    Parameters
    ----------
    input_dim   : dimensionality of input windows
    embed_dim   : encoder output dimension
    proj_dim    : projection head output dimension
    hidden_dims : encoder hidden layers
    temperature : NT-Xent temperature
    seed        : random seed
    """

    def __init__(self, input_dim: int, embed_dim: int = 64,
                 proj_dim: int = 32,
                 hidden_dims: Optional[List[int]] = None,
                 temperature: float = 0.5,
                 seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.embed_dim = embed_dim

        if hidden_dims is None:
            hidden_dims = [128, 128]

        self.encoder = Encoder(input_dim, hidden_dims, embed_dim, self.rng)
        self.projector = ProjectionHead(embed_dim, proj_dim, 64, self.rng)
        self.augmentor = TimeSeriesAugmentor(seed)
        self.step_count = 0

    # ---- forward ---------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Get embeddings (without projection head)."""
        return self.encoder.forward(x)

    def project(self, h: np.ndarray) -> np.ndarray:
        """Apply projection head."""
        return self.projector.forward(h)

    # ---- training step ---------------------------------------------------

    def train_step(self, x: np.ndarray, lr: float = 1e-3) -> float:
        """One contrastive learning step.

        Creates two augmented views, computes NT-Xent, backprops.
        """
        # Create two views
        v1 = self.augmentor.augment(x)
        v2 = self.augmentor.augment(x)

        # Forward
        h1 = self.encoder.forward(v1)
        z1 = self.projector.forward(h1)

        # Need separate forward for second view (cache issue)
        # Store encoder state, run second view
        h2 = self._encoder_forward_copy(v2)
        z2 = self._projector_forward_copy(h2)

        # Loss
        loss, gz1, gz2 = nt_xent_loss(z1, z2, self.temperature)

        # Backward through projector + encoder for view 1
        gh1 = self.projector.backward(gz1)
        self.encoder.backward(gh1)

        # Backward through projector + encoder for view 2
        gh2 = self._projector_backward_copy(gz2)
        self._encoder_backward_copy(gh2)

        # Average gradients and update
        self.step_count += 1
        self._average_and_update(lr)

        return loss

    def _encoder_forward_copy(self, x: np.ndarray) -> np.ndarray:
        """Forward pass saving to separate cache."""
        h = x
        self._enc2_cache: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer in self.encoder.layers:
            inp = h.copy()
            z = h @ layer.W + layer.b
            pre = z.copy()
            self._enc2_cache.append((inp, pre))
            h = _relu(z) if layer.act == "relu" else z
        return h

    def _projector_forward_copy(self, h: np.ndarray) -> np.ndarray:
        inp1 = h.copy()
        z1 = h @ self.projector.fc1.W + self.projector.fc1.b
        pre1 = z1.copy()
        a1 = _relu(z1)
        inp2 = a1.copy()
        z2 = a1 @ self.projector.fc2.W + self.projector.fc2.b
        pre2 = z2.copy()
        self._proj2_cache = [(inp1, pre1), (inp2, pre2)]
        return z2

    def _projector_backward_copy(self, g: np.ndarray) -> np.ndarray:
        inp2, pre2 = self._proj2_cache[1]
        self._proj2_dW2 = inp2.T @ g / g.shape[0]
        self._proj2_db2 = g.mean(axis=0)
        g = g @ self.projector.fc2.W.T

        inp1, pre1 = self._proj2_cache[0]
        g = g * (pre1 > 0).astype(np.float64)
        self._proj2_dW1 = inp1.T @ g / g.shape[0]
        self._proj2_db1 = g.mean(axis=0)
        return g @ self.projector.fc1.W.T

    def _encoder_backward_copy(self, g: np.ndarray) -> np.ndarray:
        self._enc2_grads: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, layer in enumerate(reversed(self.encoder.layers)):
            idx = len(self.encoder.layers) - 1 - i
            inp, pre = self._enc2_cache[idx]
            if layer.act == "relu":
                g = g * (pre > 0).astype(np.float64)
            dW = inp.T @ g / g.shape[0]
            db = g.mean(axis=0)
            self._enc2_grads.append((dW, db))
            g = g @ layer.W.T
        self._enc2_grads.reverse()
        return g

    def _average_and_update(self, lr: float) -> None:
        t = self.step_count
        # Average encoder gradients from both views
        for i, layer in enumerate(self.encoder.layers):
            dW2, db2 = self._enc2_grads[i]
            layer.dW = (layer.dW + dW2) / 2.0
            layer.db = (layer.db + db2) / 2.0
            layer.update(lr, t)

        # Average projector gradients
        self.projector.fc1.dW = (self.projector.fc1.dW + self._proj2_dW1) / 2.0
        self.projector.fc1.db = (self.projector.fc1.db + self._proj2_db1) / 2.0
        self.projector.fc2.dW = (self.projector.fc2.dW + self._proj2_dW2) / 2.0
        self.projector.fc2.db = (self.projector.fc2.db + self._proj2_db2) / 2.0
        self.projector.update(lr, t)

    # ---- full training ---------------------------------------------------

    def fit(self, data: np.ndarray, epochs: int = 100,
            batch_size: int = 64, lr: float = 1e-3,
            verbose: bool = True) -> List[float]:
        N = data.shape[0]
        losses: List[float] = []

        for ep in range(epochs):
            perm = self.rng.permutation(N)
            running = 0.0
            nb = 0
            for s in range(0, N, batch_size):
                idx = perm[s:s + batch_size]
                if len(idx) < 2:
                    continue
                loss = self.train_step(data[idx], lr)
                running += loss
                nb += 1
            avg = running / max(nb, 1)
            losses.append(avg)
            if verbose and (ep + 1) % max(1, epochs // 10) == 0:
                print(f"  epoch {ep+1:4d}/{epochs}  nt_xent={avg:.4f}")

        return losses

    # ---- embedding extraction --------------------------------------------

    def get_embeddings(self, data: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
        """Get encoder embeddings (no projection head)."""
        parts = []
        for s in range(0, data.shape[0], batch_size):
            parts.append(self.encode(data[s:s + batch_size]))
        return np.concatenate(parts, axis=0)

    # ---- nearest-neighbor classifier -------------------------------------

    def fit_knn_classifier(self, data: np.ndarray,
                           labels: np.ndarray) -> None:
        """Store embeddings + labels for kNN classification."""
        self._knn_embeddings = self.get_embeddings(data)
        self._knn_labels = labels.copy()

    def predict_knn(self, x: np.ndarray, k: int = 5) -> np.ndarray:
        """Classify using k-nearest neighbors in embedding space."""
        emb = self.get_embeddings(x)
        sim = _cosine_similarity(emb, self._knn_embeddings)  # (N, M)
        preds = np.empty(x.shape[0], dtype=self._knn_labels.dtype)

        for i in range(x.shape[0]):
            top_k = np.argsort(-sim[i])[:k]
            neighbor_labels = self._knn_labels[top_k]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            preds[i] = unique[np.argmax(counts)]

        return preds

    def knn_accuracy(self, x: np.ndarray, labels: np.ndarray,
                     k: int = 5) -> float:
        preds = self.predict_knn(x, k)
        return float(np.mean(preds == labels))

    # ---- similarity search -----------------------------------------------

    def find_similar_periods(self, query: np.ndarray,
                             database: np.ndarray,
                             top_k: int = 10) -> Dict[str, np.ndarray]:
        """Find historical periods most similar to query.

        Parameters
        ----------
        query    : (1, D) or (D,) single window
        database : (M, D) historical windows

        Returns
        -------
        dict with 'indices', 'similarities', 'windows'
        """
        if query.ndim == 1:
            query = query[None, :]
        q_emb = self.get_embeddings(query)
        db_emb = self.get_embeddings(database)
        sim = _cosine_similarity(q_emb, db_emb).ravel()
        top_idx = np.argsort(-sim)[:top_k]
        return {
            "indices": top_idx,
            "similarities": sim[top_idx],
            "windows": database[top_idx],
        }

    # ---- 2D visualization ------------------------------------------------

    @staticmethod
    def project_2d(embeddings: np.ndarray,
                   method: str = "pca") -> np.ndarray:
        """Project embeddings to 2D for visualization.

        Parameters
        ----------
        method : 'pca' or 'random'
        """
        if method == "pca":
            centered = embeddings - embeddings.mean(axis=0)
            cov = centered.T @ centered / (centered.shape[0] - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            top2 = eigvecs[:, -2:][:, ::-1]
            return centered @ top2
        elif method == "random":
            rng = np.random.default_rng(0)
            proj = rng.standard_normal((embeddings.shape[1], 2))
            proj /= np.linalg.norm(proj, axis=0, keepdims=True)
            return embeddings @ proj
        raise ValueError(f"Unknown method: {method}")

    # ---- online update ---------------------------------------------------

    def online_update(self, new_data: np.ndarray,
                      n_steps: int = 10, lr: float = 1e-4) -> List[float]:
        """Fine-tune on new data (streaming adaptation)."""
        losses = []
        for _ in range(n_steps):
            loss = self.train_step(new_data, lr)
            losses.append(loss)
        return losses

    # ---- transfer: extract features for downstream -----------------------

    def extract_features(self, data: np.ndarray,
                         normalize: bool = True) -> np.ndarray:
        """Get embeddings suitable for downstream ML models."""
        emb = self.get_embeddings(data)
        if normalize:
            emb = _l2_normalize(emb)
        return emb


# ---------------------------------------------------------------------------
# Regime clustering from embeddings
# ---------------------------------------------------------------------------

class RegimeClusterer:
    """K-means clustering in contrastive embedding space."""

    def __init__(self, n_clusters: int = 4, seed: int = 42):
        self.n_clusters = n_clusters
        self.rng = np.random.default_rng(seed)
        self.centroids: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Fit k-means and return cluster labels."""
        N, D = embeddings.shape
        idx = self.rng.choice(N, self.n_clusters, replace=False)
        self.centroids = embeddings[idx].copy()

        for _ in range(max_iter):
            dists = self._distances(embeddings)
            labels = np.argmin(dists, axis=1)
            new_centroids = np.empty_like(self.centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = embeddings[mask].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            if np.allclose(new_centroids, self.centroids, atol=1e-6):
                break
            self.centroids = new_centroids

        return labels

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        dists = self._distances(embeddings)
        return np.argmin(dists, axis=1)

    def _distances(self, embeddings: np.ndarray) -> np.ndarray:
        return np.sum(
            (embeddings[:, None] - self.centroids[None, :]) ** 2, axis=-1
        )

    def silhouette_score(self, embeddings: np.ndarray,
                         labels: np.ndarray) -> float:
        """Compute mean silhouette coefficient."""
        N = len(labels)
        if N < 2:
            return 0.0
        scores = np.zeros(N)
        for i in range(N):
            own = labels[i]
            own_mask = labels == own
            if own_mask.sum() <= 1:
                scores[i] = 0.0
                continue
            dists = np.sqrt(np.sum((embeddings[i] - embeddings) ** 2, axis=-1))
            a = dists[own_mask].sum() / (own_mask.sum() - 1)
            b = np.inf
            for k in range(self.n_clusters):
                if k == own:
                    continue
                k_mask = labels == k
                if k_mask.any():
                    b = min(b, dists[k_mask].mean())
            scores[i] = (b - a) / max(a, b, 1e-12)
        return float(scores.mean())


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def train_regime_embeddings(windows: np.ndarray,
                            labels: Optional[np.ndarray] = None,
                            embed_dim: int = 64,
                            epochs: int = 100,
                            n_clusters: int = 4,
                            seed: int = 42) -> Dict:
    """High-level: train contrastive model and cluster regimes.

    Parameters
    ----------
    windows  : (N, T) windowed time-series data
    labels   : (N,) optional ground-truth regime labels
    """
    input_dim = windows.shape[1]
    learner = ContrastiveLearner(input_dim, embed_dim, seed=seed)
    losses = learner.fit(windows, epochs=epochs)

    embeddings = learner.get_embeddings(windows)
    proj_2d = ContrastiveLearner.project_2d(embeddings, "pca")

    clusterer = RegimeClusterer(n_clusters, seed)
    cluster_labels = clusterer.fit(embeddings)
    silhouette = clusterer.silhouette_score(embeddings, cluster_labels)

    result: Dict = {
        "learner": learner,
        "embeddings": embeddings,
        "proj_2d": proj_2d,
        "cluster_labels": cluster_labels,
        "silhouette": silhouette,
        "losses": losses,
        "clusterer": clusterer,
    }

    if labels is not None:
        learner.fit_knn_classifier(windows, labels)
        acc = learner.knn_accuracy(windows, labels, k=5)
        result["knn_accuracy"] = acc

    return result
