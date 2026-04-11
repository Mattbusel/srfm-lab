pub mod community_detection;
pub mod correlation_matrix;
pub mod granger_causality;
pub mod lead_lag;
pub mod minimum_spanning_tree;
pub mod network_signal;
pub mod regime_correlation;

// ── New extensions ────────────────────────────────────────────────────────────
pub mod correlation_graph;
pub mod lead_lag_analyzer;
pub mod contagion_model;
pub mod network_risk;

// ── Module 4: Omni-Graph additions ───────────────────────────────────────────
pub mod ricci_curvature;
pub mod temporal_graph;
pub mod graph_signal;
pub mod centrality;
pub mod hypergraph;

pub use community_detection::{detect_communities, CommunityResult};
pub use correlation_matrix::CorrelationMatrix;
pub use granger_causality::{granger_causality_dag, GrangerResult};
pub use lead_lag::{build_lead_lag_network, LeadLagNetwork, LeadLagResult};
pub use minimum_spanning_tree::{minimum_spanning_tree, MSTResult};
pub use network_signal::{NetworkSignals, generate_network_signals};
pub use regime_correlation::{regime_correlation_analysis, RegimeCorrelationReport};

pub use correlation_graph::{CorrelationGraph, CorrelationEdge, CorrelationRegime};
pub use lead_lag_analyzer::{LeadLagAnalyzer, LeadLagResult as LeadLagAnalysisResult};
pub use contagion_model::{ContagionDetector, ContagionEvent};
pub use network_risk::{NetworkRiskAnalyzer, RiskProfile};

// Omni-Graph re-exports
pub use ricci_curvature::{
    WeightedGraph, Edge, RicciReport, RicciEdge, RollingRicci,
    RicciFlowConfig, RicciFlowResult, CurvatureTimeSeries,
    compute_ricci_curvature, ricci_flow, fiedler_value,
};
pub use temporal_graph::{
    TemporalGraph, GraphSnapshot, StructuralChange, WormholeEvent,
    PersistentEdge, EdgeFluxMetrics, ContagionPath, TemporalMotif, MotifType,
    structural_change, graph_edit_distance, shortest_contagion_paths,
    contagion_reachability, detect_temporal_motifs, structural_velocity,
    detect_velocity_spikes,
};
pub use graph_signal::{
    GFTResult, GraphWavelet, SpectralClusterResult, RandomWalkConfig,
    LinkPredictionScore,
    graph_fourier_transform, inverse_gft, graph_wavelet_transform,
    tikhonov_smooth, spectral_clustering, generate_random_walks,
    train_node_embeddings, link_prediction_scores, top_k_link_predictions,
};
pub use centrality::{
    CentralityReport, PageRankConfig, HITSResult, CentralityTracker,
    pagerank, betweenness_centrality, eigenvector_centrality, katz_centrality,
    closeness_centrality, hits, compute_all_centralities,
};
pub use hypergraph::{
    Hyperedge, Hypergraph, HypergraphLaplacian, CliqueConfig, Clique,
    MotifCounts, PersistentClique, HypergraphRiskMetrics, HypergraphTimeSeries,
    find_cliques, count_motifs, find_persistent_cliques,
    compute_hypergraph_risk,
};
