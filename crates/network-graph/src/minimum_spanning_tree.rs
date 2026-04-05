use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An edge in the minimum spanning tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSTEdge {
    pub sym_a: String,
    pub sym_b: String,
    /// Mantegna distance: sqrt(2 * (1 - rho)).
    pub distance: f64,
    /// Original correlation coefficient.
    pub correlation: f64,
}

/// Minimum spanning tree result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSTResult {
    pub edges: Vec<MSTEdge>,
    pub symbols: Vec<String>,
    /// Degree of each node in the MST (number of connected edges).
    pub degree: HashMap<String, u32>,
    /// Nodes sorted by degree descending: most-connected are systemic hubs.
    pub hubs: Vec<(String, u32)>,
    /// Total MST weight (sum of Mantegna distances).
    pub total_weight: f64,
}

/// Compute the minimum spanning tree using Prim's algorithm.
///
/// Distance metric: Mantegna (1999) — d(i,j) = sqrt(2 * (1 - rho_ij)).
/// This metric is a proper metric on the correlation space.
pub fn minimum_spanning_tree(
    symbols: &[String],
    corr_matrix: &[Vec<f64>],
) -> MSTResult {
    let n = symbols.len();

    if n == 0 {
        return MSTResult {
            edges: vec![],
            symbols: vec![],
            degree: HashMap::new(),
            hubs: vec![],
            total_weight: 0.0,
        };
    }

    if n == 1 {
        return MSTResult {
            edges: vec![],
            symbols: symbols.to_vec(),
            degree: HashMap::from([(symbols[0].clone(), 0)]),
            hubs: vec![(symbols[0].clone(), 0)],
            total_weight: 0.0,
        };
    }

    // Convert to Mantegna distance matrix.
    let dist = |i: usize, j: usize| -> f64 {
        let rho = corr_matrix[i][j].clamp(-1.0, 1.0);
        (2.0 * (1.0 - rho)).sqrt()
    };

    // Prim's algorithm.
    let mut in_tree = vec![false; n];
    let mut min_dist = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];

    // Start from node 0.
    min_dist[0] = 0.0;

    let mut mst_edges: Vec<MSTEdge> = Vec::with_capacity(n - 1);

    for _ in 0..n {
        // Pick the non-tree node with minimum key.
        let u = (0..n)
            .filter(|&i| !in_tree[i])
            .min_by(|&a, &b| {
                min_dist[a]
                    .partial_cmp(&min_dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        in_tree[u] = true;

        if parent[u] != usize::MAX {
            let p = parent[u];
            mst_edges.push(MSTEdge {
                sym_a: symbols[p].clone(),
                sym_b: symbols[u].clone(),
                distance: dist(p, u),
                correlation: corr_matrix[p][u],
            });
        }

        // Update keys for neighbours.
        for v in 0..n {
            if !in_tree[v] {
                let d = dist(u, v);
                if d < min_dist[v] {
                    min_dist[v] = d;
                    parent[v] = u;
                }
            }
        }
    }

    // Compute degrees.
    let mut degree: HashMap<String, u32> = symbols.iter().map(|s| (s.clone(), 0)).collect();
    let mut total_weight = 0.0_f64;
    for e in &mst_edges {
        *degree.entry(e.sym_a.clone()).or_insert(0) += 1;
        *degree.entry(e.sym_b.clone()).or_insert(0) += 1;
        total_weight += e.distance;
    }

    // Sort hubs by degree descending.
    let mut hubs: Vec<(String, u32)> = degree.iter().map(|(s, &d)| (s.clone(), d)).collect();
    hubs.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    MSTResult {
        edges: mst_edges,
        symbols: symbols.to_vec(),
        degree,
        hubs,
        total_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn star_corr(n: usize) -> Vec<Vec<f64>> {
        // Hub (node 0) highly correlated with all leaves; leaves uncorrelated with each other.
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        for i in 1..n {
            m[0][i] = 0.9;
            m[i][0] = 0.9;
        }
        m
    }

    #[test]
    fn test_mst_edge_count() {
        let syms: Vec<String> = (0..5).map(|i| format!("S{}", i)).collect();
        let m = star_corr(5);
        let result = minimum_spanning_tree(&syms, &m);
        // MST of N nodes has N-1 edges.
        assert_eq!(result.edges.len(), 4);
    }

    #[test]
    fn test_mst_hub_is_most_connected() {
        let syms: Vec<String> = (0..5).map(|i| format!("S{}", i)).collect();
        let m = star_corr(5);
        let result = minimum_spanning_tree(&syms, &m);
        // S0 is the hub; should have degree 4.
        let hub_degree = result.degree.get("S0").copied().unwrap_or(0);
        assert_eq!(hub_degree, 4, "star hub should have degree 4");
    }

    #[test]
    fn test_mantegna_distance_range() {
        // For rho in [-1,1]: d = sqrt(2*(1-rho)) in [0, 2].
        let rho_vals = [-1.0_f64, -0.5, 0.0, 0.5, 1.0];
        for rho in rho_vals {
            let d = (2.0 * (1.0 - rho)).sqrt();
            assert!(d >= 0.0 && d <= 2.0 + 1e-9, "d={} for rho={}", d, rho);
        }
    }

    #[test]
    fn test_mantegna_distance_zero_for_perfect_corr() {
        let d = (2.0 * (1.0 - 1.0_f64)).sqrt();
        assert_abs_diff_eq!(d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mst_single_node() {
        let syms = vec!["BTC".to_string()];
        let m = vec![vec![1.0]];
        let result = minimum_spanning_tree(&syms, &m);
        assert!(result.edges.is_empty());
    }

    #[test]
    fn test_hubs_sorted_descending() {
        let syms: Vec<String> = (0..4).map(|i| format!("S{}", i)).collect();
        let m = star_corr(4);
        let result = minimum_spanning_tree(&syms, &m);
        // Verify hubs are sorted high to low.
        let degrees: Vec<u32> = result.hubs.iter().map(|(_, d)| *d).collect();
        for w in degrees.windows(2) {
            assert!(w[0] >= w[1]);
        }
    }
}
