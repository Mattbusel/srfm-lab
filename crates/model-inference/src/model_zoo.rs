// model_zoo.rs — Pre-defined architectures, model serialization
use crate::tensor::Tensor;
use crate::dense_layer::{MLP, DenseLayer, Activation, LayerNorm, ResNet, Dropout};
use crate::lstm::{StackedLSTM, StackedGRU, BidirectionalLSTM, MergeMode, SeqEncoder, PoolingMode};
use crate::attention::{TransformerEncoder, MultiHeadAttention, FeedForward};
use crate::tree::{RandomForest, GradientBoostedTrees, DecisionTree};
use crate::ensemble::{Ensemble, EnsembleMethod};

/// MLP-based signal predictor for tick data
#[derive(Clone, Debug)]
pub struct MLPSignalPredictor {
    pub mlp: MLP,
    pub input_size: usize,
    pub output_size: usize,
}

impl MLPSignalPredictor {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut sizes = vec![input_size];
        sizes.extend_from_slice(hidden_sizes);
        sizes.push(output_size);
        let mlp = MLP::new(&sizes, Activation::GELU, false, 0.0);
        Self { mlp, input_size, output_size }
    }

    pub fn small(input_size: usize) -> Self {
        Self::new(input_size, &[64, 32], 1)
    }

    pub fn medium(input_size: usize) -> Self {
        Self::new(input_size, &[128, 64, 32], 1)
    }

    pub fn large(input_size: usize) -> Self {
        Self::new(input_size, &[256, 128, 64, 32], 1)
    }

    pub fn predict(&self, features: &Tensor) -> Tensor {
        self.mlp.forward(features)
    }

    pub fn predict_single(&self, features: &Tensor) -> Tensor {
        self.mlp.forward_single(features)
    }
}

/// ResNet signal predictor
#[derive(Clone, Debug)]
pub struct ResNetSignalPredictor {
    pub net: ResNet,
}

impl ResNetSignalPredictor {
    pub fn new(input_size: usize, hidden: usize, output_size: usize, num_blocks: usize) -> Self {
        Self { net: ResNet::new(input_size, hidden, output_size, num_blocks, Activation::GELU) }
    }

    pub fn predict(&self, features: &Tensor) -> Tensor { self.net.forward(features) }
    pub fn predict_single(&self, features: &Tensor) -> Tensor { self.net.forward_single(features) }
}

/// LSTM regime classifier
#[derive(Clone, Debug)]
pub struct LSTMRegimeClassifier {
    pub encoder: SeqEncoder,
    pub classifier: MLP,
    pub num_regimes: usize,
}

impl LSTMRegimeClassifier {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, num_regimes: usize) -> Self {
        let encoder = SeqEncoder::new(input_size, hidden_size, num_layers, PoolingMode::Last);
        let classifier = MLP::new(&[hidden_size, hidden_size / 2, num_regimes], Activation::ReLU, false, 0.0);
        Self { encoder, classifier, num_regimes }
    }

    pub fn classify(&self, sequence: &Tensor) -> (usize, Vec<f64>) {
        let encoding = self.encoder.encode(sequence);
        let logits = self.classifier.forward_single(&encoding);
        let probs = logits.softmax(0);
        let class = probs.data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        (class, probs.data.clone())
    }

    pub fn regime_probabilities(&self, sequence: &Tensor) -> Vec<f64> {
        let (_, probs) = self.classify(sequence);
        probs
    }
}

/// Bidirectional LSTM classifier
#[derive(Clone, Debug)]
pub struct BiLSTMClassifier {
    pub bilstm: BidirectionalLSTM,
    pub classifier: MLP,
}

impl BiLSTMClassifier {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, num_classes: usize) -> Self {
        let bilstm = BidirectionalLSTM::new(input_size, hidden_size, num_layers, MergeMode::Concat);
        let feat = hidden_size * 2;
        let classifier = MLP::new(&[feat, feat / 2, num_classes], Activation::ReLU, false, 0.0);
        Self { bilstm, classifier }
    }

    pub fn classify(&self, sequence: &Tensor) -> usize {
        let out = self.bilstm.forward(sequence);
        let seq_len = out.shape[0];
        let feat = out.shape[1];
        let last = Tensor::from_slice(&out.data[(seq_len - 1) * feat..seq_len * feat], &[feat]);
        let logits = self.classifier.forward_single(&last);
        logits.data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
}

/// Attention-based alpha model
#[derive(Clone, Debug)]
pub struct AttentionAlphaModel {
    pub encoder: TransformerEncoder,
    pub head: MLP,
    pub d_model: usize,
}

impl AttentionAlphaModel {
    pub fn new(input_size: usize, d_model: usize, num_heads: usize, num_layers: usize, max_seq_len: usize) -> Self {
        let encoder = TransformerEncoder::new(d_model, num_heads, d_model * 4, num_layers, max_seq_len, true);
        let head = MLP::new(&[d_model, d_model / 2, 1], Activation::GELU, false, 0.0);
        Self { encoder, head, d_model }
    }

    pub fn small(input_size: usize) -> Self {
        Self::new(input_size, 32, 4, 2, 256)
    }

    pub fn medium(input_size: usize) -> Self {
        Self::new(input_size, 64, 8, 4, 512)
    }

    pub fn predict(&self, sequence: &Tensor) -> Tensor {
        let encoded = self.encoder.forward(sequence, None);
        // use last position
        let seq_len = encoded.shape[0];
        let d = encoded.shape[1];
        let last = Tensor::from_slice(&encoded.data[(seq_len - 1) * d..seq_len * d], &[d]);
        self.head.forward_single(&last)
    }

    pub fn predict_all_positions(&self, sequence: &Tensor) -> Tensor {
        let encoded = self.encoder.forward(sequence, None);
        let seq_len = encoded.shape[0];
        let d = encoded.shape[1];
        let mut results = Vec::new();
        for t in 0..seq_len {
            let pos = Tensor::from_slice(&encoded.data[t * d..(t + 1) * d], &[d]);
            let pred = self.head.forward_single(&pos);
            results.push(pred.data[0]);
        }
        Tensor::from_vec(results, &[seq_len])
    }
}

/// LSTM + Attention hybrid model
#[derive(Clone, Debug)]
pub struct LSTMAttentionModel {
    pub lstm: StackedLSTM,
    pub attention: MultiHeadAttention,
    pub head: MLP,
}

impl LSTMAttentionModel {
    pub fn new(input_size: usize, hidden_size: usize, num_heads: usize, output_size: usize) -> Self {
        let d = hidden_size;
        assert!(d % num_heads == 0);
        Self {
            lstm: StackedLSTM::new(input_size, hidden_size, 2),
            attention: MultiHeadAttention::new(hidden_size, num_heads),
            head: MLP::new(&[hidden_size, hidden_size / 2, output_size], Activation::GELU, false, 0.0),
        }
    }

    pub fn predict(&self, sequence: &Tensor) -> Tensor {
        let (lstm_out, _, _) = self.lstm.forward(sequence);
        let attn_out = self.attention.forward(&lstm_out, None);
        let seq_len = attn_out.shape[0];
        let d = attn_out.shape[1];
        let last = Tensor::from_slice(&attn_out.data[(seq_len - 1) * d..seq_len * d], &[d]);
        self.head.forward_single(&last)
    }
}

/// Hybrid tree + neural network model
#[derive(Clone, Debug)]
pub struct TreeNNHybrid {
    pub forest: RandomForest,
    pub nn: MLP,
    pub combiner: MLP,
}

impl TreeNNHybrid {
    pub fn new(num_features: usize, num_trees: usize, tree_depth: usize, hidden: usize) -> Self {
        let forest = RandomForest::test_forest(num_trees, num_features, tree_depth);
        let nn = MLP::new(&[num_features, hidden, hidden / 2], Activation::ReLU, false, 0.0);
        let combiner = MLP::new(&[hidden / 2 + 1, 16, 1], Activation::ReLU, false, 0.0);
        Self { forest, nn, combiner }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        let tree_pred = self.forest.predict_single(features);
        let nn_input = Tensor::from_slice(features, &[features.len()]);
        let nn_out = self.nn.forward_single(&nn_input);
        let mut combined = nn_out.data.clone();
        combined.push(tree_pred);
        let comb_input = Tensor::from_vec(combined, &[nn_out.shape[0] + 1]);
        self.combiner.forward_single(&comb_input).data[0]
    }
}

// ============ MODEL SERIALIZATION ============

const MAGIC: u32 = 0x4D4F444C; // "MODL"
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum ModelType {
    MLP = 0,
    DenseLayer = 1,
    LSTM = 2,
    GRU = 3,
    Tensor = 4,
}

pub struct ModelSerializer;

impl ModelSerializer {
    pub fn serialize_tensor(t: &Tensor) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(t.ndim() as u32).to_le_bytes());
        for &s in &t.shape {
            bytes.extend_from_slice(&(s as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&(t.data.len() as u64).to_le_bytes());
        for &v in &t.data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    pub fn deserialize_tensor(bytes: &[u8], pos: &mut usize) -> Option<Tensor> {
        if *pos + 4 > bytes.len() { return None; }
        let ndim = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().ok()?) as usize;
        *pos += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            if *pos + 8 > bytes.len() { return None; }
            let s = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().ok()?) as usize;
            *pos += 8;
            shape.push(s);
        }
        if *pos + 8 > bytes.len() { return None; }
        let n = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().ok()?) as usize;
        *pos += 8;
        if *pos + n * 8 > bytes.len() { return None; }
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let v = f64::from_le_bytes(bytes[*pos..*pos + 8].try_into().ok()?);
            *pos += 8;
            data.push(v);
        }
        Some(Tensor::from_vec(data, &shape))
    }

    pub fn serialize_dense_layer(layer: &DenseLayer) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(ModelType::DenseLayer as u8);
        bytes.extend(Self::serialize_tensor(&layer.weights));
        bytes.push(if layer.bias.is_some() { 1 } else { 0 });
        if let Some(ref b) = layer.bias {
            bytes.extend(Self::serialize_tensor(b));
        }
        bytes.push(match layer.activation {
            Activation::None => 0,
            Activation::ReLU => 1,
            Activation::GELU => 2,
            Activation::Swish => 3,
            Activation::Tanh => 4,
            Activation::Sigmoid => 5,
            Activation::LeakyReLU(_) => 6,
            Activation::ELU(_) => 7,
            Activation::SELU => 8,
            Activation::Softplus => 9,
            Activation::Mish => 10,
            Activation::HardSwish => 11,
            Activation::HardSigmoid => 12,
        });
        bytes
    }

    pub fn deserialize_dense_layer(bytes: &[u8], pos: &mut usize) -> Option<DenseLayer> {
        if *pos >= bytes.len() { return None; }
        let _type_tag = bytes[*pos]; *pos += 1;
        let weights = Self::deserialize_tensor(bytes, pos)?;
        let has_bias = bytes[*pos]; *pos += 1;
        let bias = if has_bias == 1 {
            Some(Self::deserialize_tensor(bytes, pos)?)
        } else {
            None
        };
        let act_byte = bytes[*pos]; *pos += 1;
        let activation = match act_byte {
            0 => Activation::None,
            1 => Activation::ReLU,
            2 => Activation::GELU,
            3 => Activation::Swish,
            4 => Activation::Tanh,
            5 => Activation::Sigmoid,
            6 => Activation::LeakyReLU(0.01),
            7 => Activation::ELU(1.0),
            8 => Activation::SELU,
            9 => Activation::Softplus,
            10 => Activation::Mish,
            11 => Activation::HardSwish,
            12 => Activation::HardSigmoid,
            _ => Activation::None,
        };
        Some(DenseLayer::from_weights(weights, bias, activation))
    }

    pub fn serialize_mlp(mlp: &MLP) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(ModelType::MLP as u8);
        bytes.extend_from_slice(&(mlp.layers.len() as u32).to_le_bytes());
        for layer in &mlp.layers {
            bytes.extend(Self::serialize_dense_layer(layer));
        }
        bytes
    }

    pub fn deserialize_mlp(bytes: &[u8]) -> Option<MLP> {
        let mut pos = 0;
        if pos + 9 > bytes.len() { return None; }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        if magic != MAGIC { return None; }
        pos += 4;
        let _version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
        pos += 4;
        let _model_type = bytes[pos]; pos += 1;
        let num_layers = u32::from_le_bytes(bytes[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut layers = Vec::new();
        for _ in 0..num_layers {
            layers.push(Self::deserialize_dense_layer(bytes, &mut pos)?);
        }
        Some(MLP {
            layers,
            norms: vec![None; num_layers],
            dropout: Dropout::new(0.0),
        })
    }

    pub fn serialize_model_header(model_type: ModelType, extra: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.push(model_type as u8);
        bytes.extend_from_slice(&(extra.len() as u64).to_le_bytes());
        bytes.extend_from_slice(extra);
        bytes
    }
}

/// Save/load model to/from file
pub fn save_model_bytes(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    std::fs::write(path, bytes)
}

pub fn load_model_bytes(path: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Model registry for named models
#[derive(Default)]
pub struct ModelRegistry {
    pub models: Vec<(String, Vec<u8>)>,
}

impl ModelRegistry {
    pub fn new() -> Self { Self { models: Vec::new() } }

    pub fn register(&mut self, name: &str, bytes: Vec<u8>) {
        self.models.push((name.to_string(), bytes));
    }

    pub fn get(&self, name: &str) -> Option<&[u8]> {
        self.models.iter().find(|(n, _)| n == name).map(|(_, b)| b.as_slice())
    }

    pub fn list(&self) -> Vec<&str> {
        self.models.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn remove(&mut self, name: &str) -> bool {
        let len = self.models.len();
        self.models.retain(|(n, _)| n != name);
        self.models.len() < len
    }
}

/// Model performance tracker
#[derive(Clone, Debug)]
pub struct ModelTracker {
    pub predictions: Vec<f64>,
    pub actuals: Vec<f64>,
    pub timestamps: Vec<u64>,
    pub model_name: String,
}

impl ModelTracker {
    pub fn new(name: &str) -> Self {
        Self {
            predictions: Vec::new(),
            actuals: Vec::new(),
            timestamps: Vec::new(),
            model_name: name.to_string(),
        }
    }

    pub fn record(&mut self, prediction: f64, actual: f64, timestamp: u64) {
        self.predictions.push(prediction);
        self.actuals.push(actual);
        self.timestamps.push(timestamp);
    }

    pub fn mse(&self) -> f64 {
        if self.predictions.is_empty() { return 0.0; }
        self.predictions.iter().zip(self.actuals.iter())
            .map(|(&p, &a)| (p - a) * (p - a))
            .sum::<f64>() / self.predictions.len() as f64
    }

    pub fn mae(&self) -> f64 {
        if self.predictions.is_empty() { return 0.0; }
        self.predictions.iter().zip(self.actuals.iter())
            .map(|(&p, &a)| (p - a).abs())
            .sum::<f64>() / self.predictions.len() as f64
    }

    pub fn r_squared(&self) -> f64 {
        let n = self.actuals.len() as f64;
        if n < 2.0 { return 0.0; }
        let mean_a = self.actuals.iter().sum::<f64>() / n;
        let ss_res: f64 = self.predictions.iter().zip(self.actuals.iter())
            .map(|(&p, &a)| (a - p) * (a - p)).sum();
        let ss_tot: f64 = self.actuals.iter().map(|&a| (a - mean_a) * (a - mean_a)).sum();
        if ss_tot < 1e-15 { return 0.0; }
        1.0 - ss_res / ss_tot
    }

    pub fn directional_accuracy(&self) -> f64 {
        if self.predictions.len() < 2 { return 0.0; }
        let n = self.predictions.len() - 1;
        let correct: usize = (1..=n).filter(|&i| {
            let pred_dir = self.predictions[i] - self.predictions[i - 1];
            let actual_dir = self.actuals[i] - self.actuals[i - 1];
            pred_dir * actual_dir > 0.0
        }).count();
        correct as f64 / n as f64
    }

    pub fn information_coefficient(&self) -> f64 {
        // rank correlation (Spearman)
        let n = self.predictions.len();
        if n < 2 { return 0.0; }
        let rank = |vals: &[f64]| -> Vec<f64> {
            let mut indexed: Vec<(usize, f64)> = vals.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
            let mut ranks = vec![0.0; n];
            for (rank, &(idx, _)) in indexed.iter().enumerate() {
                ranks[idx] = rank as f64;
            }
            ranks
        };
        let pr = rank(&self.predictions);
        let ar = rank(&self.actuals);
        let mean_p = pr.iter().sum::<f64>() / n as f64;
        let mean_a = ar.iter().sum::<f64>() / n as f64;
        let cov: f64 = pr.iter().zip(ar.iter()).map(|(&p, &a)| (p - mean_p) * (a - mean_a)).sum();
        let std_p = (pr.iter().map(|&p| (p - mean_p).powi(2)).sum::<f64>()).sqrt();
        let std_a = (ar.iter().map(|&a| (a - mean_a).powi(2)).sum::<f64>()).sqrt();
        if std_p < 1e-15 || std_a < 1e-15 { return 0.0; }
        cov / (std_p * std_a)
    }

    pub fn rolling_mse(&self, window: usize) -> Vec<f64> {
        if self.predictions.len() < window { return vec![]; }
        (0..=self.predictions.len() - window).map(|i| {
            let mse: f64 = self.predictions[i..i + window].iter()
                .zip(self.actuals[i..i + window].iter())
                .map(|(&p, &a)| (p - a) * (p - a))
                .sum::<f64>() / window as f64;
            mse
        }).collect()
    }

    pub fn count(&self) -> usize { self.predictions.len() }

    pub fn last_n_mse(&self, n: usize) -> f64 {
        let start = if self.predictions.len() > n { self.predictions.len() - n } else { 0 };
        let slice_p = &self.predictions[start..];
        let slice_a = &self.actuals[start..];
        if slice_p.is_empty() { return 0.0; }
        slice_p.iter().zip(slice_a.iter())
            .map(|(&p, &a)| (p - a) * (p - a))
            .sum::<f64>() / slice_p.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_predictor() {
        let pred = MLPSignalPredictor::small(10);
        let x = Tensor::from_vec(vec![0.1; 10], &[1, 10]);
        let out = pred.predict(&x);
        assert_eq!(out.shape, vec![1, 1]);
    }

    #[test]
    fn test_regime_classifier() {
        let cls = LSTMRegimeClassifier::new(4, 8, 1, 3);
        let seq = Tensor::from_vec(vec![0.1; 12], &[3, 4]);
        let (regime, probs) = cls.classify(&seq);
        assert!(regime < 3);
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mlp = MLP::new(&[4, 8, 2], Activation::ReLU, false, 0.0);
        let bytes = ModelSerializer::serialize_mlp(&mlp);
        let mlp2 = ModelSerializer::deserialize_mlp(&bytes).unwrap();
        assert_eq!(mlp2.layers.len(), 2);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let out1 = mlp.forward(&x);
        let out2 = mlp2.forward(&x);
        assert!(out1.allclose(&out2, 1e-10));
    }

    #[test]
    fn test_model_tracker() {
        let mut tracker = ModelTracker::new("test");
        tracker.record(1.1, 1.0, 0);
        tracker.record(2.1, 2.0, 1);
        tracker.record(3.1, 3.0, 2);
        assert!(tracker.mse() < 0.02);
        assert!(tracker.r_squared() > 0.95);
    }

    #[test]
    fn test_model_registry() {
        let mut reg = ModelRegistry::new();
        reg.register("model_a", vec![1, 2, 3]);
        reg.register("model_b", vec![4, 5, 6]);
        assert_eq!(reg.list().len(), 2);
        assert_eq!(reg.get("model_a"), Some(&[1u8, 2, 3][..]));
        assert!(reg.remove("model_a"));
        assert_eq!(reg.list().len(), 1);
    }

    #[test]
    fn test_attention_alpha() {
        let model = AttentionAlphaModel::small(32);
        let seq = Tensor::from_vec((0..96).map(|i| i as f64 * 0.01).collect(), &[3, 32]);
        let out = model.predict(&seq);
        assert_eq!(out.shape, vec![1]);
    }
}
