use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of community detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    /// Community assignment per symbol index (0-indexed community IDs).
    pub assignments: Vec<usize>,
    /// Symbol names in the same order as `assignments`.
    pub symbols: Vec<String>,
    /// Modularity score Q (higher = better community structure).
    pub modularity: f64,
    /// Number of communities detected.
    pub n_communities: usize,
    /// Map from community ID to list of symbols in that community.
    pub communities: HashMap<usize, Vec<String>>,
}

/// Detect communities using Louvain-style greedy modularity maximization.
///
/// # Arguments
/// * `symbols` — ordered list of asset names.
/// * `corr_matrix` — N×N correlation matrix (symmetric, diagonal = 1.0).
/// * `threshold` — minimum |correlation| to include an edge (default 0.5).
pub fn detect_communities(
    symbols: &[String],
    corr_matrix: &[Vec<f64>],
    threshold: f64,
) -> CommunityResult {
    let n = symbols.len();
    if n == 0 {
        return CommunityResult {
            assignments: vec![],
            symbols: vec![],
            modularity: 0.0,
            n_communities: 0,
            communities: HashMap::new(),
        };
    }

    // Build adjacency matrix from |correlation| > threshold.
    // Edge weight = |correlation| - threshold (so weak edges have small weight).
    let mut adj = vec![vec![0.0_f64; n]; n];
    let mut two_m = 0.0_f64; // 2 * sum of all edge weights

    for i in 0..n {
        for j in (i + 1)..n {
            let c = corr_matrix[i][j].abs();
            if c > threshold {
                let w = c - threshold;
                adj[i][j] = w;
                adj[j][i] = w;
                two_m += 2.0 * w;
            }
        }
    }

    if two_m < 1e-12 {
        // No edges: each node is its own community.
        let assignments: Vec<usize> = (0..n).collect();
        let communities = build_community_map(symbols, &assignments);
        let q = 0.0;
        return CommunityResult {
            assignments,
            symbols: symbols.to_vec(),
            modularity: q,
            n_communities: n,
            communities,
        };
    }

    // Node strengths (weighted degree).
    let strength: Vec<f64> = (0..n).map(|i| adj[i].iter().sum::<f64>()).collect();

    // Initialise: each node in its own community.
    let mut community: Vec<usize> = (0..n).collect();

    // Phase 1: greedy node moves.
    // For each node, compute the modularity gain of moving it to each neighbour's community.
    // Repeat until no improvement.
    let mut improved = true;
    let mut iterations = 0;
    let max_iter = n * 10;

    while improved && iterations < max_iter {
        improved = false;
        iterations += 1;

        for i in 0..n {
            let current_c = community[i];
            let mut best_c = current_c;
            let mut best_gain = 0.0_f64;

            // Find unique neighbour communities.
            let mut neighbor_communities: Vec<usize> = (0..n)
                .filter(|&j| j != i && adj[i][j] > 0.0)
                .map(|j| community[j])
                .collect();
            neighbor_communities.sort_unstable();
            neighbor_communities.dedup();

            for &target_c in &neighbor_communities {
                if target_c == current_c {
                    continue;
                }

                // Modularity gain ΔQ for moving i from current_c to target_c.
                // ΔQ = [k_{i,target} - k_{i,current}] / m
                //     - (k_target - k_current) * k_i / (2m^2)
                // where k_{i,c} = sum of weights from i to community c.
                let k_i = strength[i];
                let k_target: f64 = (0..n)
                    .filter(|&j| community[j] == target_c)
                    .map(|j| adj[i][j])
                    .sum();
                let k_current: f64 = (0..n)
                    .filter(|&j| community[j] == current_c && j != i)
                    .map(|j| adj[i][j])
                    .sum();

                let strength_target: f64 =
                    (0..n).filter(|&j| community[j] == target_c).map(|j| strength[j]).sum();
                let strength_current: f64 = (0..n)
                    .filter(|&j| community[j] == current_c && j != i)
                    .map(|j| strength[j])
                    .sum();

                let gain = (k_target - k_current) / (two_m / 2.0)
                    - (strength_target - strength_current) * k_i / (two_m * two_m / 4.0);

                if gain > best_gain {
                    best_gain = gain;
                    best_c = target_c;
                }
            }

            if best_c != current_c {
                community[i] = best_c;
                improved = true;
            }
        }
    }

    // Normalise community IDs to 0..k.
    let mut id_map: HashMap<usize, usize> = HashMap::new();
    let mut next_id = 0usize;
    for &c in &community {
        if !id_map.contains_key(&c) {
            id_map.insert(c, next_id);
            next_id += 1;
        }
    }
    let assignments: Vec<usize> = community.iter().map(|c| id_map[c]).collect();
    let n_communities = next_id;

    // Compute final modularity.
    let modularity = compute_modularity(&adj, &assignments, &strength, two_m);
    let communities = build_community_map(symbols, &assignments);

    CommunityResult {
        assignments,
        symbols: symbols.to_vec(),
        modularity,
        n_communities,
        communities,
    }
}

fn compute_modularity(
    adj: &[Vec<f64>],
    community: &[usize],
    strength: &[f64],
    two_m: f64,
) -> f64 {
    let n = adj.len();
    let mut q = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if community[i] == community[j] {
                q += adj[i][j] - strength[i] * strength[j] / two_m;
            }
        }
    }
    q / two_m
}

fn build_community_map(
    symbols: &[String],
    assignments: &[usize],
) -> HashMap<usize, Vec<String>> {
    let mut map: HashMap<usize, Vec<String>> = HashMap::new();
    for (i, &c) in assignments.iter().enumerate() {
        map.entry(c).or_default().push(symbols[i].clone());
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_matrix() -> (Vec<String>, Vec<Vec<f64>>) {
        // Two obvious clusters: {A,B,C} and {D,E,F}.
        // High correlation within cluster, zero between.
        let syms: Vec<String> = ["A", "B", "C", "D", "E", "F"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let n = 6;
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        // Cluster 1: 0,1,2
        for &i in &[0, 1, 2] {
            for &j in &[0, 1, 2] {
                if i != j {
                    m[i][j] = 0.9;
                }
            }
        }
        // Cluster 2: 3,4,5
        for &i in &[3, 4, 5] {
            for &j in &[3, 4, 5] {
                if i != j {
                    m[i][j] = 0.85;
                }
            }
        }
        (syms, m)
    }

    #[test]
    fn test_detect_two_clusters() {
        let (syms, m) = two_cluster_matrix();
        let result = detect_communities(&syms, &m, 0.5);
        assert_eq!(result.n_communities, 2, "expected 2 communities, got {}", result.n_communities);
    }

    #[test]
    fn test_modularity_positive_for_clustered() {
        let (syms, m) = two_cluster_matrix();
        let result = detect_communities(&syms, &m, 0.5);
        assert!(result.modularity > 0.0, "modularity should be positive for clear clusters");
    }

    #[test]
    fn test_community_map_covers_all_symbols() {
        let (syms, m) = two_cluster_matrix();
        let result = detect_communities(&syms, &m, 0.5);
        let all_in_communities: Vec<String> = result
            .communities
            .values()
            .flat_map(|v| v.iter().cloned())
            .collect();
        assert_eq!(all_in_communities.len(), syms.len());
    }

    #[test]
    fn test_no_edges_each_own_community() {
        let syms: Vec<String> = vec!["A".to_string(), "B".to_string()];
        let m = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = detect_communities(&syms, &m, 0.5);
        assert_eq!(result.n_communities, 2);
    }

    #[test]
    fn test_assignments_length() {
        let (syms, m) = two_cluster_matrix();
        let result = detect_communities(&syms, &m, 0.5);
        assert_eq!(result.assignments.len(), syms.len());
    }
}
