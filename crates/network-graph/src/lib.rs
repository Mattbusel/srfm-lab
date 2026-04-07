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
