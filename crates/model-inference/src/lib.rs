pub mod tensor;
pub mod dense_layer;
pub mod lstm;
pub mod attention;
pub mod tree;
pub mod ensemble;
pub mod model_zoo;

pub use tensor::Tensor;
pub use dense_layer::{
    Activation, DenseLayer, MLP, BatchNorm1d, LayerNorm, RMSNorm, Dropout,
    ResidualBlock, ResNet, GLU, MixtureOfExperts, HighwayLayer, SEBlock,
    FiLM, Embedding, InputStandardizer, FeatureCross, BottleneckLayer, DenseBlock,
};
pub use lstm::{
    LSTMWeights, GRUWeights, StackedLSTM, StackedGRU,
    BidirectionalLSTM, BidirectionalGRU, MergeMode,
    SeqEncoder, SeqDecoder, Seq2Seq, PoolingMode,
    LSTMWithAttention, SkipLSTM, MultiScaleLSTM,
    PeepholeLSTMWeights,
};
pub use attention::{
    MultiHeadAttention, MultiQueryAttention, GroupedQueryAttention,
    TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer,
    FeedForward, SwiGLUFeedForward, CrossAttention, KVCache,
    RelativePositionBias, LearnedPositionalEncoding,
    scaled_dot_product_attention, batched_attention,
    causal_mask, padding_mask, combined_mask, sliding_window_mask,
    sinusoidal_positional_encoding, apply_rope, alibi_slopes, alibi_bias,
    linear_attention, random_feature_attention,
};
pub use tree::{
    DecisionTree, TreeNode, SplitCondition,
    RandomForest, GradientBoostedTrees, GBTLoss, MultiClassGBT,
    IsolationForest, QuantileForest, MultiOutputTree,
    XGBoostTree, LightGBMTree, ExtraTreesEnsemble,
    serialize_tree, deserialize_tree, permutation_importance,
};
pub use ensemble::{
    Ensemble, EnsembleMethod, Predictor,
    OnlineWeightUpdater, OnlineMethod,
    PlattScaling, IsotonicCalibration, TemperatureScaling, ConformalPredictor,
    SelectionCriterion, ModelSelectionResult, select_model,
    expected_calibration_error, brier_score, log_loss, reliability_diagram,
    disagreement_measure, correlation_diversity,
    aic, bic, aicc, hqic,
};
pub use model_zoo::{
    MLPSignalPredictor, ResNetSignalPredictor,
    LSTMRegimeClassifier, BiLSTMClassifier,
    AttentionAlphaModel, LSTMAttentionModel, TreeNNHybrid,
    ModelSerializer, ModelRegistry, ModelTracker,
    save_model_bytes, load_model_bytes,
};
