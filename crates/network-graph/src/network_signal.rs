use crate::community_detection::CommunityResult;
use crate::lead_lag::LeadLagNetwork;
use crate::minimum_spanning_tree::MSTResult;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Trading signals derived from network analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSignals {
    /// Portfolio diversification score (0 = all same community, 1 = perfectly diversified).
    pub diversification_score: f64,
    /// True if portfolio holds too many high-MST-degree (hub) coins.
    pub hub_concentration_risk: bool,
    /// Symbols flagged as systemic hubs (top quartile MST degree).
    pub hub_symbols: Vec<String>,
    /// Lead-based entry signals: (follower_symbol, leader_symbol, lag_bars, signal_strength).
    pub lead_signals: Vec<LeadSignal>,
    /// Warning if portfolio holds too many coins from same community.
    pub community_overlap_warning: Option<String>,
    /// Number of distinct communities represented in portfolio.
    pub portfolio_community_count: usize,
    /// Raw MST-based diversity score (average Mantegna distance within portfolio).
    pub mst_diversity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadSignal {
    pub follower: String,
    pub leader: String,
    pub lag_bars: u32,
    pub signal_strength: f64,
    /// Direction: +1 if leader moved up, -1 if down.
    pub direction: i8,
}

/// Generate network-based trading signals.
///
/// # Arguments
/// * `portfolio_symbols` — symbols currently held or under consideration.
/// * `mst` — minimum spanning tree result.
/// * `communities` — community detection result.
/// * `lead_lag_net` — lead-lag network.
/// * `leader_recent_returns` — most recent return for each leader symbol (for direction).
/// * `hub_degree_threshold` — MST degree above which a node is considered a hub (default 3).
pub fn generate_network_signals(
    portfolio_symbols: &[String],
    mst: &MSTResult,
    communities: &CommunityResult,
    lead_lag_net: &LeadLagNetwork,
    leader_recent_returns: &HashMap<String, f64>,
    hub_degree_threshold: u32,
) -> NetworkSignals {
    // ── Hub concentration risk ────────────────────────────────────────────
    let hub_symbols: Vec<String> = mst
        .hubs
        .iter()
        .filter(|(_, d)| *d >= hub_degree_threshold)
        .map(|(s, _)| s.clone())
        .collect();

    let portfolio_set: HashSet<&str> = portfolio_symbols.iter().map(|s| s.as_str()).collect();
    let hub_count = hub_symbols
        .iter()
        .filter(|s| portfolio_set.contains(s.as_str()))
        .count();
    let hub_concentration_risk = hub_count > portfolio_symbols.len() / 2 && hub_count >= 2;

    // ── Community overlap ─────────────────────────────────────────────────
    let mut community_ids: Vec<usize> = portfolio_symbols
        .iter()
        .filter_map(|s| {
            communities
                .symbols
                .iter()
                .position(|cs| cs == s)
                .map(|idx| communities.assignments[idx])
        })
        .collect();
    community_ids.sort_unstable();
    community_ids.dedup();
    let portfolio_community_count = community_ids.len();

    let community_overlap_warning = if portfolio_symbols.len() >= 3
        && portfolio_community_count == 1
    {
        Some(format!(
            "All {} portfolio symbols are in the same community — high correlation risk",
            portfolio_symbols.len()
        ))
    } else if portfolio_symbols.len() >= 4
        && portfolio_community_count <= portfolio_symbols.len() / 3
    {
        Some(format!(
            "Only {} communities across {} symbols — limited diversification",
            portfolio_community_count,
            portfolio_symbols.len()
        ))
    } else {
        None
    };

    // ── Diversification score ─────────────────────────────────────────────
    // Fraction of distinct communities relative to maximum possible.
    let max_possible = portfolio_symbols.len().min(communities.n_communities).max(1);
    let diversification_score = (portfolio_community_count as f64 / max_possible as f64)
        .clamp(0.0, 1.0);

    // ── MST diversity: average Mantegna distance among portfolio pairs ────
    let mst_diversity_score = mst_portfolio_diversity(portfolio_symbols, mst);

    // ── Lead-lag signals ──────────────────────────────────────────────────
    let mut lead_signals: Vec<LeadSignal> = Vec::new();

    for edge in &lead_lag_net.edges {
        // Generate a signal if the leader is in the portfolio or is BTC/ETH,
        // and the follower is in the portfolio.
        let follower_in_portfolio = portfolio_set.contains(edge.follower.as_str());
        let leader_active = portfolio_set.contains(edge.leader.as_str())
            || edge.leader == "BTC"
            || edge.leader == "ETH";

        if follower_in_portfolio && leader_active {
            let recent_ret = leader_recent_returns
                .get(&edge.leader)
                .copied()
                .unwrap_or(0.0);
            let direction = if recent_ret > 0.0 { 1i8 } else if recent_ret < 0.0 { -1 } else { 0 };
            let signal_strength = edge.correlation.abs() * (1.0 / (1.0 + edge.lag_bars as f64));

            lead_signals.push(LeadSignal {
                follower: edge.follower.clone(),
                leader: edge.leader.clone(),
                lag_bars: edge.lag_bars,
                signal_strength,
                direction,
            });
        }
    }

    // Sort by signal strength descending.
    lead_signals.sort_unstable_by(|a, b| {
        b.signal_strength
            .partial_cmp(&a.signal_strength)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    NetworkSignals {
        diversification_score,
        hub_concentration_risk,
        hub_symbols,
        lead_signals,
        community_overlap_warning,
        portfolio_community_count,
        mst_diversity_score,
    }
}

/// Compute the average Mantegna distance across all pairs in the portfolio,
/// using the MST edges as a proxy for distance.
fn mst_portfolio_diversity(portfolio_symbols: &[String], mst: &MSTResult) -> f64 {
    let port_set: HashSet<&str> = portfolio_symbols.iter().map(|s| s.as_str()).collect();
    let relevant_edges: Vec<f64> = mst
        .edges
        .iter()
        .filter(|e| port_set.contains(e.sym_a.as_str()) && port_set.contains(e.sym_b.as_str()))
        .map(|e| e.distance)
        .collect();

    if relevant_edges.is_empty() {
        // No MST edges among portfolio — assume max diversity.
        return 2.0_f64.sqrt(); // sqrt(2*(1-0)) for rho=0
    }

    relevant_edges.iter().sum::<f64>() / relevant_edges.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::community_detection::CommunityResult;
    use crate::lead_lag::{LeadLagEdge, LeadLagNetwork};
    use crate::minimum_spanning_tree::{MSTEdge, MSTResult};

    fn make_mst() -> MSTResult {
        MSTResult {
            edges: vec![
                MSTEdge { sym_a: "BTC".into(), sym_b: "ETH".into(), distance: 0.3, correlation: 0.96 },
                MSTEdge { sym_a: "BTC".into(), sym_b: "SOL".into(), distance: 0.5, correlation: 0.88 },
            ],
            symbols: vec!["BTC".into(), "ETH".into(), "SOL".into()],
            degree: [("BTC".into(), 2), ("ETH".into(), 1), ("SOL".into(), 1)].into(),
            hubs: vec![("BTC".into(), 2), ("ETH".into(), 1), ("SOL".into(), 1)],
            total_weight: 0.8,
        }
    }

    fn make_communities() -> CommunityResult {
        CommunityResult {
            assignments: vec![0, 0, 1],
            symbols: vec!["BTC".into(), "ETH".into(), "SOL".into()],
            modularity: 0.3,
            n_communities: 2,
            communities: [(0, vec!["BTC".into(), "ETH".into()]), (1, vec!["SOL".into()])].into(),
        }
    }

    fn make_lead_lag_net() -> LeadLagNetwork {
        LeadLagNetwork {
            edges: vec![LeadLagEdge {
                leader: "BTC".into(),
                follower: "ETH".into(),
                lag_bars: 2,
                correlation: 0.8,
            }],
            in_degree: [("ETH".into(), 1)].into(),
            out_degree: [("BTC".into(), 1)].into(),
        }
    }

    #[test]
    fn test_hub_concentration_detected() {
        let mst = make_mst();
        let communities = make_communities();
        let net = make_lead_lag_net();
        // Portfolio with BTC (degree=2) as a hub.
        let portfolio = vec!["BTC".to_string(), "ETH".to_string()];
        let signals = generate_network_signals(&portfolio, &mst, &communities, &net, &Default::default(), 2);
        assert!(signals.hub_symbols.contains(&"BTC".to_string()));
    }

    #[test]
    fn test_diversification_score_range() {
        let mst = make_mst();
        let communities = make_communities();
        let net = make_lead_lag_net();
        let portfolio = vec!["BTC".to_string(), "SOL".to_string()];
        let signals = generate_network_signals(&portfolio, &mst, &communities, &net, &Default::default(), 3);
        assert!((0.0..=1.0).contains(&signals.diversification_score));
    }

    #[test]
    fn test_lead_signal_generated() {
        let mst = make_mst();
        let communities = make_communities();
        let net = make_lead_lag_net();
        // ETH is a follower of BTC.
        let portfolio = vec!["ETH".to_string()];
        let recent: HashMap<String, f64> = [("BTC".to_string(), 0.02)].into();
        let signals = generate_network_signals(&portfolio, &mst, &communities, &net, &recent, 3);
        assert!(!signals.lead_signals.is_empty(), "expected a BTC->ETH lead signal");
        assert_eq!(signals.lead_signals[0].leader, "BTC");
        assert_eq!(signals.lead_signals[0].direction, 1);
    }

    #[test]
    fn test_community_overlap_warning() {
        let mst = make_mst();
        // All three in same community.
        let mut communities = make_communities();
        communities.assignments = vec![0, 0, 0];
        communities.n_communities = 1;
        communities.communities = [(0, vec!["BTC".into(), "ETH".into(), "SOL".into()])].into();
        let net = make_lead_lag_net();
        let portfolio = vec!["BTC".into(), "ETH".into(), "SOL".into()];
        let signals = generate_network_signals(&portfolio, &mst, &communities, &net, &Default::default(), 3);
        assert!(signals.community_overlap_warning.is_some());
    }
}
