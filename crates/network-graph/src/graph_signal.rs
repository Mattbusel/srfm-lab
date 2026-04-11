/// Graph signal processing: Graph Fourier Transform, wavelets, smoothing,
/// spectral clustering, node embeddings, and link prediction.

use std::collections::HashMap;
use rayon::prelude::*;

use crate::ricci_curvature::WeightedGraph;

// ── Laplacian eigensystem ─────────────────────────────────────────────────────

/// Compute the symmetric normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}.
/// Returns (n x n) matrix as flat Vec<f64> (row-major).
pub fn normalized_laplacian(graph: &WeightedGraph) -> Vec<f64> {
    let n = graph.n;
    let mut l = vec![0.0f64; n * n];

    // Degree vector (sum of weights)
    let mut deg = vec![0.0f64; n];
    for u in 0..n {
        for &(v, w) in &graph.adj[u] {
            deg[u] += w;
            deg[v] += w; // undirected symmetrization
        }
    }
    // Deduplicate (each edge counted twice)
    for d in &mut deg { *d /= 2.0; }

    // Build L = D - A (symmetric)
    for u in 0..n {
        l[u * n + u] = 1.0; // D^{-1/2} D D^{-1/2} = I on diagonal
        for &(v, w) in &graph.adj[u] {
            let sym_w = (w + graph.adj[v].iter().find(|&&(x, _)| x == u).map(|(_, ww)| ww).unwrap_or(&0.0)) / 2.0;
            let di = deg[u];
            let dj = deg[v];
            if di > 0.0 && dj > 0.0 {
                let val = -sym_w / (di * dj).sqrt();
                l[u * n + v] = val;
                l[v * n + u] = val;
            }
        }
    }
    l
}

/// Eigendecomposition via symmetric QR iteration (Jacobi method).
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns.
/// For small n (< 100). For large n, use approximate/truncated methods.
pub fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let max_iter = 500;
    let tol = 1e-10;

    // Copy to working matrix
    let mut a = matrix.to_vec();
    // Initialize eigenvector matrix as identity
    let mut v = vec![0.0f64; n * n];
    for i in 0..n { v[i * n + i] = 1.0; }

    // Jacobi sweeps
    for _iter in 0..max_iter {
        // Find max off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i+1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol { break; }

        // Compute rotation angle
        let theta = if (a[q * n + q] - a[p * n + p]).abs() < 1e-14 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * ((2.0 * a[p * n + q]) / (a[q * n + q] - a[p * n + p])).atan()
        };
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply rotation to A (both rows and columns)
        let new_app = cos_t.powi(2) * a[p * n + p]
            - 2.0 * sin_t * cos_t * a[p * n + q]
            + sin_t.powi(2) * a[q * n + q];
        let new_aqq = sin_t.powi(2) * a[p * n + p]
            + 2.0 * sin_t * cos_t * a[p * n + q]
            + cos_t.powi(2) * a[q * n + q];

        for k in 0..n {
            if k == p || k == q { continue; }
            let apk = a[p * n + k];
            let aqk = a[q * n + k];
            a[p * n + k] = cos_t * apk - sin_t * aqk;
            a[k * n + p] = a[p * n + k];
            a[q * n + k] = sin_t * apk + cos_t * aqk;
            a[k * n + q] = a[q * n + k];
        }
        a[p * n + p] = new_app;
        a[q * n + q] = new_aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update eigenvectors
        for k in 0..n {
            let vpk = v[k * n + p];
            let vqk = v[k * n + q];
            v[k * n + p] = cos_t * vpk - sin_t * vqk;
            v[k * n + q] = sin_t * vpk + cos_t * vqk;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

// ── Graph Fourier Transform ───────────────────────────────────────────────────

/// Result of Graph Fourier Transform.
#[derive(Debug, Clone)]
pub struct GFTResult {
    /// Eigenvalues (frequencies) of the graph Laplacian.
    pub frequencies: Vec<f64>,
    /// Eigenvectors (Fourier basis).
    pub basis: Vec<f64>,
    /// GFT coefficients for each input signal.
    pub coefficients: Vec<Vec<f64>>,
    /// Signal energy per frequency.
    pub spectral_energy: Vec<Vec<f64>>,
}

/// Compute the Graph Fourier Transform of signals on the graph.
/// `signals`: n_signals x n_nodes matrix (row = one signal).
pub fn graph_fourier_transform(graph: &WeightedGraph, signals: &[Vec<f64>]) -> GFTResult {
    let n = graph.n;
    let l = normalized_laplacian(graph);
    let (mut eigenvalues, eigenvectors) = symmetric_eigen(&l, n);

    // Sort by eigenvalue (frequency)
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap_or(std::cmp::Ordering::Equal));

    eigenvalues = idx.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_basis: Vec<f64> = {
        let mut sb = vec![0.0f64; n * n];
        for (col_new, &col_old) in idx.iter().enumerate() {
            for row in 0..n {
                sb[row * n + col_new] = eigenvectors[row * n + col_old];
            }
        }
        sb
    };

    // Project each signal onto the basis: X_hat = U^T x
    let coefficients: Vec<Vec<f64>> = signals.iter().map(|sig| {
        (0..n).map(|k| {
            (0..n).map(|i| sorted_basis[i * n + k] * sig.get(i).copied().unwrap_or(0.0)).sum()
        }).collect()
    }).collect();

    // Spectral energy: |X_hat_k|^2
    let spectral_energy: Vec<Vec<f64>> = coefficients.iter().map(|coeff| {
        coeff.iter().map(|c| c * c).collect()
    }).collect();

    GFTResult {
        frequencies: eigenvalues,
        basis: sorted_basis,
        coefficients,
        spectral_energy,
    }
}

/// Inverse Graph Fourier Transform: reconstruct signal from GFT coefficients.
/// x = U * x_hat
pub fn inverse_gft(gft: &GFTResult, coefficients: &[f64]) -> Vec<f64> {
    let n = gft.frequencies.len();
    (0..n).map(|i| {
        (0..n).map(|k| gft.basis[i * n + k] * coefficients.get(k).copied().unwrap_or(0.0)).sum()
    }).collect()
}

// ── Graph Wavelet Transform ───────────────────────────────────────────────────

/// Graph wavelet transform for multi-scale signal analysis.
/// Uses the spectral graph wavelets (Hammond et al. 2011).
#[derive(Debug, Clone)]
pub struct GraphWavelet {
    /// Scale parameter (controls frequency localization).
    pub scale: f64,
    /// Wavelet coefficients per node.
    pub coefficients: Vec<Vec<f64>>,
    /// Scale values used.
    pub scales: Vec<f64>,
}

/// Kernel function for the wavelet (Mexican hat / Marr wavelet on eigenvalues).
fn wavelet_kernel(lambda: f64, scale: f64) -> f64 {
    let x = scale * lambda;
    // Mexican hat: x * exp(-x)
    x * (-x).exp()
}

/// Compute multi-scale graph wavelet transform.
pub fn graph_wavelet_transform(
    graph: &WeightedGraph,
    signal: &[f64],
    scales: &[f64],
) -> GraphWavelet {
    let n = graph.n;
    let l = normalized_laplacian(graph);
    let (eigenvalues, eigenvectors) = symmetric_eigen(&l, n);

    // GFT of signal
    let signal_gft: Vec<f64> = (0..n).map(|k| {
        (0..n).map(|i| eigenvectors[i * n + k] * signal.get(i).copied().unwrap_or(0.0)).sum()
    }).collect();

    let coefficients: Vec<Vec<f64>> = scales.iter().map(|&s| {
        // Apply wavelet kernel in spectral domain
        let filtered_gft: Vec<f64> = eigenvalues.iter().zip(signal_gft.iter())
            .map(|(&lam, &x_hat)| wavelet_kernel(lam, s) * x_hat)
            .collect();

        // Inverse GFT
        (0..n).map(|i| {
            (0..n).map(|k| eigenvectors[i * n + k] * filtered_gft.get(k).copied().unwrap_or(0.0)).sum()
        }).collect()
    }).collect();

    GraphWavelet {
        scale: scales.iter().copied().fold(f64::NAN, f64::max),
        coefficients,
        scales: scales.to_vec(),
    }
}

// ── Graph signal smoothing ────────────────────────────────────────────────────

/// Tikhonov regularization on graph: smooth signal x while preserving fit to observations.
/// Solves: min ||x - y||^2 + lambda * x^T L x
/// Solution: x = (I + lambda * L)^{-1} y  via conjugate gradient.
pub fn tikhonov_smooth(
    graph: &WeightedGraph,
    signal: &[f64],
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = graph.n;
    let l = normalized_laplacian(graph);

    // Build A = I + lambda * L
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = lambda * l[i * n + j];
            if i == j { a[i * n + j] += 1.0; }
        }
    }

    // Conjugate gradient solver for A x = y
    conjugate_gradient(&a, signal, n, max_iter, tol)
}

/// Conjugate gradient solver for Ax = b.
fn conjugate_gradient(a: &[f64], b: &[f64], n: usize, max_iter: usize, tol: f64) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    let ax = mat_vec_flat(a, &x, n);
    let mut r: Vec<f64> = (0..n).map(|i| b.get(i).copied().unwrap_or(0.0) - ax[i]).collect();
    let mut p = r.clone();
    let mut rr: f64 = r.iter().map(|x| x * x).sum();

    for _ in 0..max_iter {
        if rr.sqrt() < tol { break; }
        let ap = mat_vec_flat(a, &p, n);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(pi, api)| pi * api).sum();
        if pap.abs() < 1e-15 { break; }
        let alpha = rr / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rr_new: f64 = r.iter().map(|x| x * x).sum();
        let beta = rr_new / rr;
        rr = rr_new;
        for i in 0..n { p[i] = r[i] + beta * p[i]; }
    }
    x
}

fn mat_vec_flat(a: &[f64], v: &[f64], n: usize) -> Vec<f64> {
    (0..n).map(|i| (0..n).map(|j| a[i * n + j] * v.get(j).copied().unwrap_or(0.0)).sum()).collect()
}

// ── Spectral Clustering ───────────────────────────────────────────────────────

/// Result of spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralClusterResult {
    pub assignments: Vec<usize>,
    pub k: usize,
    pub inertia: f64,
}

/// Spectral clustering using the k smallest eigenvectors of the normalized Laplacian.
pub fn spectral_clustering(graph: &WeightedGraph, k: usize, max_kmeans_iter: usize) -> SpectralClusterResult {
    let n = graph.n;
    if n == 0 || k == 0 { return SpectralClusterResult { assignments: Vec::new(), k, inertia: 0.0 }; }

    let l = normalized_laplacian(graph);
    let (eigenvalues, eigenvectors) = symmetric_eigen(&l, n);

    // Sort by eigenvalue and take k smallest
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap_or(std::cmp::Ordering::Equal));
    let k_use = k.min(n);

    // Embedding matrix: n x k (each row is the embedding of a node)
    let embedding: Vec<Vec<f64>> = (0..n).map(|node| {
        (0..k_use).map(|dim| eigenvectors[node * n + idx[dim]]).collect()
    }).collect();

    // Normalize rows of embedding
    let embedding_norm: Vec<Vec<f64>> = embedding.iter().map(|row| {
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 { row.clone() } else { row.iter().map(|x| x / norm).collect() }
    }).collect();

    // K-means on the embedding
    kmeans(&embedding_norm, k_use, max_kmeans_iter)
}

/// Simple k-means clustering on embeddings.
fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize) -> SpectralClusterResult {
    let n = data.len();
    if n == 0 { return SpectralClusterResult { assignments: Vec::new(), k, inertia: 0.0 }; }
    let d = data[0].len();
    let k_use = k.min(n);

    // Initialize centroids as first k data points
    let mut centroids: Vec<Vec<f64>> = (0..k_use).map(|i| data[i % n].clone()).collect();
    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iter {
        let mut changed = false;

        // Assign each point to nearest centroid
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for (ci, centroid) in centroids.iter().enumerate() {
                let dist: f64 = data[i].iter().zip(centroid.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                if dist < best_dist {
                    best_dist = dist;
                    best = ci;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }

        if !changed { break; }

        // Update centroids
        let mut new_centroids = vec![vec![0.0f64; d]; k_use];
        let mut counts = vec![0usize; k_use];
        for (i, &a) in assignments.iter().enumerate() {
            counts[a] += 1;
            for j in 0..d { new_centroids[a][j] += data[i][j]; }
        }
        for c in 0..k_use {
            if counts[c] > 0 {
                for j in 0..d { new_centroids[c][j] /= counts[c] as f64; }
                centroids[c] = new_centroids[c].clone();
            }
        }
    }

    // Compute inertia
    let inertia: f64 = (0..n).map(|i| {
        let c = &centroids[assignments[i]];
        data[i].iter().zip(c.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>()
    }).sum();

    SpectralClusterResult { assignments, k: k_use, inertia }
}

// ── Node Embedding via Random Walks (DeepWalk-style) ─────────────────────────

/// Node embedding configuration.
#[derive(Debug, Clone)]
pub struct RandomWalkConfig {
    pub embedding_dim: usize,
    pub walk_length: usize,
    pub walks_per_node: usize,
    pub window_size: usize,
    pub learning_rate: f64,
    pub epochs: usize,
    pub negative_samples: usize,
}

impl Default for RandomWalkConfig {
    fn default() -> Self {
        RandomWalkConfig {
            embedding_dim: 32,
            walk_length: 40,
            walks_per_node: 10,
            window_size: 5,
            learning_rate: 0.025,
            epochs: 5,
            negative_samples: 5,
        }
    }
}

/// Generate random walks from a weighted graph (biased by edge weights).
pub fn generate_random_walks(
    graph: &WeightedGraph,
    config: &RandomWalkConfig,
    seed: u64,
) -> Vec<Vec<usize>> {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let n = graph.n;
    let mut rng_state = seed;

    let lcg_next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (u64::MAX >> 11) as f64
    };

    let mut walks = Vec::new();

    for _walk_num in 0..config.walks_per_node {
        for start in 0..n {
            let mut walk = vec![start];
            let mut current = start;

            for _ in 0..config.walk_length {
                let neighbors = &graph.adj[current];
                if neighbors.is_empty() { break; }

                // Weighted sampling
                let total: f64 = neighbors.iter().map(|(_, w)| w).sum();
                let r = lcg_next(&mut rng_state) * total;
                let mut cum = 0.0;
                let mut chosen = neighbors[0].0;
                for &(v, w) in neighbors {
                    cum += w;
                    if r <= cum {
                        chosen = v;
                        break;
                    }
                }
                walk.push(chosen);
                current = chosen;
            }
            walks.push(walk);
        }
    }
    walks
}

/// Train node embeddings using skip-gram with negative sampling.
/// Returns embedding matrix: n_nodes x embedding_dim.
pub fn train_node_embeddings(
    graph: &WeightedGraph,
    config: &RandomWalkConfig,
) -> Vec<Vec<f64>> {
    let n = graph.n;
    let d = config.embedding_dim;

    // Initialize embeddings randomly
    let mut embeddings: Vec<Vec<f64>> = (0..n).map(|i| {
        (0..d).map(|j| {
            let seed = i as f64 * 1000.0 + j as f64;
            (seed.sin() * 43758.5453).fract() * 0.1
        }).collect()
    }).collect();

    let mut context_embeddings: Vec<Vec<f64>> = embeddings.clone();

    let walks = generate_random_walks(graph, config, 42);
    let mut rng = 12345u64;
    let lcg_next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (u64::MAX >> 11) as f64
    };

    for _epoch in 0..config.epochs {
        for walk in &walks {
            for (pos, &center) in walk.iter().enumerate() {
                let start = pos.saturating_sub(config.window_size);
                let end = (pos + config.window_size + 1).min(walk.len());

                for ctx_pos in start..end {
                    if ctx_pos == pos { continue; }
                    let ctx = walk[ctx_pos];

                    // Positive sample gradient
                    let score_pos: f64 = embeddings[center].iter().zip(context_embeddings[ctx].iter())
                        .map(|(a, b)| a * b).sum();
                    let sig_pos = sigmoid(score_pos);
                    let grad_pos = (1.0 - sig_pos) * config.learning_rate;

                    let e_center = embeddings[center].clone();
                    let e_ctx = context_embeddings[ctx].clone();
                    for k in 0..d {
                        embeddings[center][k] += grad_pos * e_ctx[k];
                        context_embeddings[ctx][k] += grad_pos * e_center[k];
                    }

                    // Negative samples
                    for _ in 0..config.negative_samples {
                        let neg = (lcg_next(&mut rng) * n as f64) as usize % n;
                        if neg == ctx { continue; }
                        let score_neg: f64 = embeddings[center].iter().zip(context_embeddings[neg].iter())
                            .map(|(a, b)| a * b).sum();
                        let sig_neg = sigmoid(score_neg);
                        let grad_neg = -sig_neg * config.learning_rate;
                        let e_neg = context_embeddings[neg].clone();
                        let e_ctr = embeddings[center].clone();
                        for k in 0..d {
                            embeddings[center][k] += grad_neg * e_neg[k];
                            context_embeddings[neg][k] += grad_neg * e_ctr[k];
                        }
                    }
                }
            }
        }
    }
    embeddings
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

// ── Link Prediction ───────────────────────────────────────────────────────────

/// Link prediction scores for a candidate edge (u, v).
#[derive(Debug, Clone)]
pub struct LinkPredictionScore {
    pub src: usize,
    pub dst: usize,
    pub common_neighbors: usize,
    pub adamic_adar: f64,
    pub jaccard: f64,
    pub preferential_attachment: f64,
    pub resource_allocation: f64,
    pub katz_score: f64,
}

/// Compute link prediction scores for all non-existing edges.
pub fn link_prediction_scores(graph: &WeightedGraph) -> Vec<LinkPredictionScore> {
    let n = graph.n;

    // Build adjacency sets
    let adj_sets: Vec<std::collections::HashSet<usize>> = (0..n).map(|u| {
        let mut s = std::collections::HashSet::new();
        for &(v, _) in &graph.adj[u] { s.insert(v); }
        for &(v, _) in &graph.radj[u] { s.insert(v); }
        s
    }).collect();

    let degrees: Vec<usize> = (0..n).map(|u| adj_sets[u].len()).collect();

    // Katz matrix (approximate): K ≈ (I - beta * A)^{-1} - I, use power series
    let katz_scores = compute_katz_scores(graph, 0.01, 5);

    let existing_edges: std::collections::HashSet<(usize, usize)> = graph.edges().iter()
        .map(|e| (e.src, e.dst)).collect();

    let candidates: Vec<(usize, usize)> = (0..n)
        .flat_map(|u| (u+1..n).map(move |v| (u, v)))
        .filter(|&(u, v)| !existing_edges.contains(&(u, v)) && !existing_edges.contains(&(v, u)))
        .collect();

    candidates.par_iter().map(|&(u, v)| {
        let common: Vec<usize> = adj_sets[u].intersection(&adj_sets[v]).copied().collect();
        let n_common = common.len();

        let adamic_adar: f64 = common.iter().map(|&w| {
            let d = degrees[w];
            if d > 1 { 1.0 / (d as f64).ln() } else { 0.0 }
        }).sum();

        let union_size = adj_sets[u].union(&adj_sets[v]).count();
        let jaccard = if union_size == 0 { 0.0 } else { n_common as f64 / union_size as f64 };

        let pref_attach = (degrees[u] * degrees[v]) as f64;

        let resource_allocation: f64 = common.iter().map(|&w| {
            let d = degrees[w];
            if d > 0 { 1.0 / d as f64 } else { 0.0 }
        }).sum();

        let katz = katz_scores.get(&(u, v)).copied().unwrap_or(0.0);

        LinkPredictionScore {
            src: u,
            dst: v,
            common_neighbors: n_common,
            adamic_adar,
            jaccard,
            preferential_attachment: pref_attach,
            resource_allocation,
            katz_score: katz,
        }
    }).collect()
}

/// Approximate Katz centrality scores for pairs (power series expansion).
fn compute_katz_scores(graph: &WeightedGraph, beta: f64, depth: usize) -> HashMap<(usize, usize), f64> {
    let n = graph.n;
    // A^k contribution summed
    let mut scores: HashMap<(usize, usize), f64> = HashMap::new();

    // Sparse matrix powers via repeated multiplication
    // Start with adjacency matrix as sparse
    let mut current: HashMap<(usize, usize), f64> = graph.edges().iter()
        .map(|e| ((e.src, e.dst), e.weight))
        .collect();

    let mut beta_power = beta;
    for _k in 1..=depth {
        for (&(u, v), &val) in &current {
            *scores.entry((u, v)).or_insert(0.0) += beta_power * val;
        }
        // Multiply by adjacency matrix
        let mut next: HashMap<(usize, usize), f64> = HashMap::new();
        for (&(u, w), &val) in &current {
            for &(v, ew) in &graph.adj[w] {
                *next.entry((u, v)).or_insert(0.0) += val * ew;
            }
        }
        current = next;
        beta_power *= beta;
    }
    scores
}

/// Predict top-k most likely new edges.
pub fn top_k_link_predictions(
    graph: &WeightedGraph,
    k: usize,
    score_type: &str,
) -> Vec<LinkPredictionScore> {
    let mut scores = link_prediction_scores(graph);
    let cmp = |a: &LinkPredictionScore, b: &LinkPredictionScore| {
        let sa = match score_type {
            "adamic_adar" => a.adamic_adar,
            "jaccard" => a.jaccard,
            "preferential_attachment" => a.preferential_attachment,
            "resource_allocation" => a.resource_allocation,
            "katz" => a.katz_score,
            _ => a.common_neighbors as f64,
        };
        let sb = match score_type {
            "adamic_adar" => b.adamic_adar,
            "jaccard" => b.jaccard,
            "preferential_attachment" => b.preferential_attachment,
            "resource_allocation" => b.resource_allocation,
            "katz" => b.katz_score,
            _ => b.common_neighbors as f64,
        };
        sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
    };
    scores.sort_by(cmp);
    scores.truncate(k);
    scores
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> WeightedGraph {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        g
    }

    #[test]
    fn test_normalized_laplacian_size() {
        let g = triangle_graph();
        let l = normalized_laplacian(&g);
        assert_eq!(l.len(), 9); // 3x3
    }

    #[test]
    fn test_gft_preserves_energy() {
        let g = triangle_graph();
        let signal = vec![vec![1.0, 0.0, 0.0]];
        let gft = graph_fourier_transform(&g, &signal);
        // Parseval: sum of |X_hat|^2 should ≈ sum of |x|^2 = 1
        let energy: f64 = gft.coefficients[0].iter().map(|c| c * c).sum();
        assert!((energy - 1.0).abs() < 1e-6, "Parseval theorem violated: {}", energy);
    }

    #[test]
    fn test_tikhonov_smooth() {
        let g = triangle_graph();
        let noisy = vec![1.0, 0.0, 0.5];
        let smoothed = tikhonov_smooth(&g, &noisy, 1.0, 100, 1e-8);
        assert_eq!(smoothed.len(), 3);
        // Smoothed values should be between min and max of noisy
        for &v in &smoothed {
            assert!(v >= -0.1 && v <= 1.1, "Value out of range: {}", v);
        }
    }

    #[test]
    fn test_spectral_clustering() {
        // Two disconnected triangles
        let mut g = WeightedGraph::new(6);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0); g.add_edge(0, 2, 1.0);
        g.add_edge(3, 4, 1.0); g.add_edge(4, 5, 1.0); g.add_edge(3, 5, 1.0);
        let result = spectral_clustering(&g, 2, 100);
        assert_eq!(result.assignments.len(), 6);
        assert_eq!(result.k, 2);
    }

    #[test]
    fn test_link_prediction() {
        let g = triangle_graph();
        // Remove one edge to create a candidate
        let mut g2 = WeightedGraph::new(4);
        g2.add_edge(0, 1, 1.0);
        g2.add_edge(1, 2, 1.0);
        g2.add_edge(0, 3, 1.0);
        let scores = link_prediction_scores(&g2);
        assert!(!scores.is_empty());
    }

    #[test]
    fn test_random_walk_generation() {
        let g = triangle_graph();
        let config = RandomWalkConfig { walks_per_node: 2, walk_length: 5, ..Default::default() };
        let walks = generate_random_walks(&g, &config, 42);
        assert!(!walks.is_empty());
        for walk in &walks {
            assert!(!walk.is_empty());
        }
    }

    #[test]
    fn test_wavelet_transform() {
        let g = triangle_graph();
        let signal = vec![1.0, 0.5, 0.0];
        let scales = vec![0.5, 1.0, 2.0];
        let wt = graph_wavelet_transform(&g, &signal, &scales);
        assert_eq!(wt.coefficients.len(), 3);
        assert_eq!(wt.scales.len(), 3);
    }
}
