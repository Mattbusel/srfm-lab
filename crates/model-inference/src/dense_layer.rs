// dense_layer.rs — Dense layer forward pass, batch norm, dropout, activations, residual connections
use crate::tensor::Tensor;

/// Activation function types
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    None,
    ReLU,
    LeakyReLU(f64),
    GELU,
    Swish,
    Tanh,
    Sigmoid,
    ELU(f64),
    SELU,
    Softplus,
    Mish,
    HardSwish,
    HardSigmoid,
}

pub fn apply_activation(x: &Tensor, act: Activation) -> Tensor {
    match act {
        Activation::None => x.clone(),
        Activation::ReLU => x.relu(),
        Activation::LeakyReLU(alpha) => {
            Tensor::from_vec(x.data.iter().map(|&v| if v > 0.0 { v } else { alpha * v }).collect(), &x.shape)
        }
        Activation::GELU => x.gelu(),
        Activation::Swish => x.swish(),
        Activation::Tanh => x.tanh_elem(),
        Activation::Sigmoid => x.sigmoid(),
        Activation::ELU(alpha) => {
            Tensor::from_vec(x.data.iter().map(|&v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }).collect(), &x.shape)
        }
        Activation::SELU => {
            let alpha = 1.6732632423543772;
            let lambda = 1.0507009873554805;
            Tensor::from_vec(x.data.iter().map(|&v| {
                lambda * if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }
            }).collect(), &x.shape)
        }
        Activation::Softplus => {
            Tensor::from_vec(x.data.iter().map(|&v| (1.0 + v.exp()).ln()).collect(), &x.shape)
        }
        Activation::Mish => {
            Tensor::from_vec(x.data.iter().map(|&v| v * ((1.0 + v.exp()).ln()).tanh()).collect(), &x.shape)
        }
        Activation::HardSwish => {
            Tensor::from_vec(x.data.iter().map(|&v| {
                if v <= -3.0 { 0.0 } else if v >= 3.0 { v } else { v * (v + 3.0) / 6.0 }
            }).collect(), &x.shape)
        }
        Activation::HardSigmoid => {
            Tensor::from_vec(x.data.iter().map(|&v| {
                if v <= -3.0 { 0.0 } else if v >= 3.0 { 1.0 } else { (v + 3.0) / 6.0 }
            }).collect(), &x.shape)
        }
    }
}

/// Batch normalization parameters for inference
#[derive(Clone, Debug)]
pub struct BatchNorm1d {
    pub num_features: usize,
    pub running_mean: Vec<f64>,
    pub running_var: Vec<f64>,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub eps: f64,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            gamma: vec![1.0; num_features],
            beta: vec![0.0; num_features],
            eps: 1e-5,
        }
    }

    pub fn from_params(mean: Vec<f64>, var: Vec<f64>, gamma: Vec<f64>, beta: Vec<f64>, eps: f64) -> Self {
        let n = mean.len();
        Self { num_features: n, running_mean: mean, running_var: var, gamma, beta, eps }
    }

    /// Forward pass for inference: input [batch, features]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 2);
        let batch = x.shape[0];
        let feat = x.shape[1];
        assert_eq!(feat, self.num_features);
        let mut out = Tensor::zeros(&x.shape);
        for b in 0..batch {
            for f in 0..feat {
                let val = x.get(&[b, f]);
                let normed = (val - self.running_mean[f]) / (self.running_var[f] + self.eps).sqrt();
                out.set(&[b, f], self.gamma[f] * normed + self.beta[f]);
            }
        }
        out
    }

    /// Forward for single sample (no batch dim)
    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        assert_eq!(x.shape[0], self.num_features);
        let mut out = Tensor::zeros(&x.shape);
        for f in 0..self.num_features {
            let val = x.data[f];
            let normed = (val - self.running_mean[f]) / (self.running_var[f] + self.eps).sqrt();
            out.data[f] = self.gamma[f] * normed + self.beta[f];
        }
        out
    }
}

/// Layer normalization
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub normalized_shape: usize,
    pub gamma: Vec<f64>,
    pub beta: Vec<f64>,
    pub eps: f64,
}

impl LayerNorm {
    pub fn new(size: usize) -> Self {
        Self { normalized_shape: size, gamma: vec![1.0; size], beta: vec![0.0; size], eps: 1e-5 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let last = *x.shape.last().unwrap();
        assert_eq!(last, self.normalized_shape);
        let batch: usize = x.numel() / last;
        let mut out = x.clone();
        for b in 0..batch {
            let off = b * last;
            let sl = &x.data[off..off + last];
            let mean: f64 = sl.iter().sum::<f64>() / last as f64;
            let var: f64 = sl.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / last as f64;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for i in 0..last {
                out.data[off + i] = self.gamma[i] * (sl[i] - mean) * inv_std + self.beta[i];
            }
        }
        out
    }
}

/// RMS normalization
#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub size: usize,
    pub gamma: Vec<f64>,
    pub eps: f64,
}

impl RMSNorm {
    pub fn new(size: usize) -> Self {
        Self { size, gamma: vec![1.0; size], eps: 1e-6 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let last = *x.shape.last().unwrap();
        assert_eq!(last, self.size);
        let batch: usize = x.numel() / last;
        let mut out = x.clone();
        for b in 0..batch {
            let off = b * last;
            let sl = &x.data[off..off + last];
            let rms: f64 = (sl.iter().map(|v| v * v).sum::<f64>() / last as f64 + self.eps).sqrt();
            for i in 0..last {
                out.data[off + i] = self.gamma[i] * sl[i] / rms;
            }
        }
        out
    }
}

/// Deterministic dropout (inference mode = identity, or scale mask)
#[derive(Clone, Debug)]
pub struct Dropout {
    pub rate: f64,
    pub training: bool,
    seed: u64,
}

impl Dropout {
    pub fn new(rate: f64) -> Self {
        Self { rate, training: false, seed: 42 }
    }

    pub fn inference() -> Self {
        Self { rate: 0.0, training: false, seed: 0 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.rate == 0.0 {
            return x.clone();
        }
        // deterministic dropout using simple LCG hash
        let scale = 1.0 / (1.0 - self.rate);
        let threshold = (self.rate * u32::MAX as f64) as u32;
        let mut state = self.seed;
        let mut out = x.clone();
        for i in 0..out.data.len() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state >> 33) as u32;
            if r < threshold {
                out.data[i] = 0.0;
            } else {
                out.data[i] *= scale;
            }
        }
        out
    }

    pub fn with_seed(rate: f64, seed: u64, training: bool) -> Self {
        Self { rate, training, seed }
    }
}

/// Dense (fully connected) layer
#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub weights: Tensor,  // [in_features, out_features]
    pub bias: Option<Tensor>,  // [out_features]
    pub activation: Activation,
}

impl DenseLayer {
    pub fn new(in_features: usize, out_features: usize, activation: Activation, use_bias: bool) -> Self {
        // Xavier initialization
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut w_data = vec![0.0; in_features * out_features];
        let mut state = 123456789u64;
        for v in w_data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = (state >> 11) as f64 / (1u64 << 53) as f64;
            *v = (u * 2.0 - 1.0) * limit;
        }
        let weights = Tensor::from_vec(w_data, &[in_features, out_features]);
        let bias = if use_bias { Some(Tensor::zeros(&[out_features])) } else { None };
        Self { weights, bias, activation }
    }

    pub fn from_weights(weights: Tensor, bias: Option<Tensor>, activation: Activation) -> Self {
        Self { weights, bias, activation }
    }

    /// Forward: input [batch, in_features] -> [batch, out_features]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out = x.matmul(&self.weights);
        let out = if let Some(ref b) = self.bias {
            out.add(&b.unsqueeze(0).broadcast_to(&out.shape))
        } else {
            out
        };
        apply_activation(&out, self.activation)
    }

    /// Forward for single sample: [in_features] -> [out_features]
    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let out = x.vecmat(&self.weights);
        let out = if let Some(ref b) = self.bias {
            out.add(b)
        } else {
            out
        };
        apply_activation(&out, self.activation)
    }

    pub fn in_features(&self) -> usize { self.weights.shape[0] }
    pub fn out_features(&self) -> usize { self.weights.shape[1] }
    pub fn num_params(&self) -> usize {
        let w = self.weights.numel();
        let b = self.bias.as_ref().map_or(0, |b| b.numel());
        w + b
    }
}

/// Residual connection wrapping a sub-network
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    pub layers: Vec<DenseLayer>,
    pub norm: Option<LayerNorm>,
    pub dropout: Dropout,
    pub pre_norm: bool,
}

impl ResidualBlock {
    pub fn new(layers: Vec<DenseLayer>, norm: Option<LayerNorm>, dropout_rate: f64, pre_norm: bool) -> Self {
        Self { layers, norm, dropout: Dropout::new(dropout_rate), pre_norm }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x.clone();
        let mut h = x.clone();
        if self.pre_norm {
            if let Some(ref norm) = self.norm {
                h = norm.forward(&h);
            }
        }
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h = self.dropout.forward(&h);
        let out = h.add(&residual);
        if !self.pre_norm {
            if let Some(ref norm) = self.norm {
                return norm.forward(&out);
            }
        }
        out
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let residual = x.clone();
        let mut h = x.clone();
        if self.pre_norm {
            if let Some(ref norm) = self.norm {
                h = norm.forward(&h);
            }
        }
        for layer in &self.layers {
            h = layer.forward_single(&h);
        }
        h = self.dropout.forward(&h);
        let out = h.add(&residual);
        if !self.pre_norm {
            if let Some(ref norm) = self.norm {
                return norm.forward(&out);
            }
        }
        out
    }
}

/// Multi-layer perceptron
#[derive(Clone, Debug)]
pub struct MLP {
    pub layers: Vec<DenseLayer>,
    pub norms: Vec<Option<BatchNorm1d>>,
    pub dropout: Dropout,
}

impl MLP {
    pub fn new(layer_sizes: &[usize], activation: Activation, use_bn: bool, dropout_rate: f64) -> Self {
        let mut layers = Vec::new();
        let mut norms = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let act = if i == layer_sizes.len() - 2 { Activation::None } else { activation };
            layers.push(DenseLayer::new(layer_sizes[i], layer_sizes[i + 1], act, true));
            if use_bn && i < layer_sizes.len() - 2 {
                norms.push(Some(BatchNorm1d::new(layer_sizes[i + 1])));
            } else {
                norms.push(None);
            }
        }
        MLP { layers, norms, dropout: Dropout::new(dropout_rate) }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if let Some(Some(ref bn)) = self.norms.get(i) {
                h = bn.forward(&h);
            }
            if i < self.layers.len() - 1 {
                h = self.dropout.forward(&h);
            }
        }
        h
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward_single(&h);
            if let Some(Some(ref bn)) = self.norms.get(i) {
                h = bn.forward_single(&h);
            }
            if i < self.layers.len() - 1 {
                h = self.dropout.forward(&h);
            }
        }
        h
    }

    pub fn num_params(&self) -> usize { self.layers.iter().map(|l| l.num_params()).sum() }
    pub fn depth(&self) -> usize { self.layers.len() }
}

/// Highway network layer: T * H(x) + (1-T) * x
#[derive(Clone, Debug)]
pub struct HighwayLayer {
    pub transform_gate: DenseLayer,
    pub hidden: DenseLayer,
}

impl HighwayLayer {
    pub fn new(size: usize, activation: Activation) -> Self {
        Self {
            transform_gate: DenseLayer::new(size, size, Activation::Sigmoid, true),
            hidden: DenseLayer::new(size, size, activation, true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let t = self.transform_gate.forward(x);
        let h = self.hidden.forward(x);
        let carry = Tensor::ones(&t.shape).sub(&t);
        t.mul_elem(&h).add(&carry.mul_elem(x))
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let t = self.transform_gate.forward_single(x);
        let h = self.hidden.forward_single(x);
        let carry = Tensor::ones(&t.shape).sub(&t);
        t.mul_elem(&h).add(&carry.mul_elem(x))
    }
}

/// Squeeze-and-Excitation block
#[derive(Clone, Debug)]
pub struct SEBlock {
    pub fc1: DenseLayer,
    pub fc2: DenseLayer,
}

impl SEBlock {
    pub fn new(channels: usize, reduction: usize) -> Self {
        let mid = channels / reduction;
        Self {
            fc1: DenseLayer::new(channels, mid, Activation::ReLU, true),
            fc2: DenseLayer::new(mid, channels, Activation::Sigmoid, true),
        }
    }

    /// x: [batch, channels] — squeeze is just identity for 1D features
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let scale = self.fc1.forward(x);
        let scale = self.fc2.forward(&scale);
        x.mul_elem(&scale)
    }
}

/// Gated Linear Unit
#[derive(Clone, Debug)]
pub struct GLU {
    pub linear: DenseLayer,
    pub gate: DenseLayer,
}

impl GLU {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            linear: DenseLayer::new(in_features, out_features, Activation::None, true),
            gate: DenseLayer::new(in_features, out_features, Activation::Sigmoid, true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let lin = self.linear.forward(x);
        let g = self.gate.forward(x);
        lin.mul_elem(&g)
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let lin = self.linear.forward_single(x);
        let g = self.gate.forward_single(x);
        lin.mul_elem(&g)
    }
}

/// Mixture of Experts layer
#[derive(Clone, Debug)]
pub struct MixtureOfExperts {
    pub experts: Vec<MLP>,
    pub gating: DenseLayer,
    pub top_k: usize,
}

impl MixtureOfExperts {
    pub fn new(in_features: usize, hidden: usize, out_features: usize, num_experts: usize, top_k: usize) -> Self {
        let experts: Vec<MLP> = (0..num_experts)
            .map(|_| MLP::new(&[in_features, hidden, out_features], Activation::ReLU, false, 0.0))
            .collect();
        let gating = DenseLayer::new(in_features, num_experts, Activation::None, true);
        Self { experts, gating, top_k }
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let logits = self.gating.forward_single(x);
        let probs = logits.softmax(0);
        // top-k selection
        let n = self.experts.len();
        let mut indices: Vec<(usize, f64)> = (0..n).map(|i| (i, probs.data[i])).collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indices.truncate(self.top_k);
        let total: f64 = indices.iter().map(|(_, w)| w).sum();
        let out_size = self.experts[0].layers.last().unwrap().out_features();
        let mut result = Tensor::zeros(&[out_size]);
        for (idx, weight) in &indices {
            let expert_out = self.experts[*idx].forward_single(x);
            let scaled = expert_out.mul_scalar(*weight / total);
            result = result.add(&scaled);
        }
        result
    }
}

/// Feature-wise Linear Modulation
#[derive(Clone, Debug)]
pub struct FiLM {
    pub gamma_net: DenseLayer,
    pub beta_net: DenseLayer,
}

impl FiLM {
    pub fn new(cond_features: usize, target_features: usize) -> Self {
        Self {
            gamma_net: DenseLayer::new(cond_features, target_features, Activation::None, true),
            beta_net: DenseLayer::new(cond_features, target_features, Activation::None, true),
        }
    }

    pub fn modulate(&self, x: &Tensor, cond: &Tensor) -> Tensor {
        let gamma = self.gamma_net.forward(cond).add_scalar(1.0); // center around 1
        let beta = self.beta_net.forward(cond);
        x.mul_elem(&gamma).add(&beta)
    }

    pub fn modulate_single(&self, x: &Tensor, cond: &Tensor) -> Tensor {
        let gamma = self.gamma_net.forward_single(cond).add_scalar(1.0);
        let beta = self.beta_net.forward_single(cond);
        x.mul_elem(&gamma).add(&beta)
    }
}

/// Weight normalization helper
pub fn weight_normalize(w: &Tensor, g: f64) -> Tensor {
    let norm = w.frobenius_norm();
    if norm < 1e-12 { return w.clone(); }
    w.mul_scalar(g / norm)
}

/// Spectral normalization (power iteration, inference only stores u,v)
#[derive(Clone, Debug)]
pub struct SpectralNorm {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

impl SpectralNorm {
    pub fn new(out_features: usize, in_features: usize) -> Self {
        let u_len = out_features;
        let v_len = in_features;
        let norm_u = 1.0 / (u_len as f64).sqrt();
        let norm_v = 1.0 / (v_len as f64).sqrt();
        Self {
            u: vec![norm_u; u_len],
            v: vec![norm_v; v_len],
        }
    }

    pub fn normalize(&self, w: &Tensor) -> Tensor {
        assert_eq!(w.ndim(), 2);
        let (m, n) = (w.shape[0], w.shape[1]);
        // sigma = u^T W v
        let wv: Vec<f64> = (0..m).map(|i| {
            (0..n).map(|j| w.data[i * n + j] * self.v[j]).sum::<f64>()
        }).collect();
        let sigma: f64 = self.u.iter().zip(wv.iter()).map(|(u, wv)| u * wv).sum();
        if sigma.abs() < 1e-12 { return w.clone(); }
        w.mul_scalar(1.0 / sigma)
    }
}

/// Deep residual network
#[derive(Clone, Debug)]
pub struct ResNet {
    pub input_proj: DenseLayer,
    pub blocks: Vec<ResidualBlock>,
    pub output_proj: DenseLayer,
}

impl ResNet {
    pub fn new(in_features: usize, hidden: usize, out_features: usize, num_blocks: usize, activation: Activation) -> Self {
        let input_proj = DenseLayer::new(in_features, hidden, activation, true);
        let mut blocks = Vec::new();
        for _ in 0..num_blocks {
            let layers = vec![
                DenseLayer::new(hidden, hidden, activation, true),
                DenseLayer::new(hidden, hidden, Activation::None, true),
            ];
            blocks.push(ResidualBlock::new(layers, Some(LayerNorm::new(hidden)), 0.0, true));
        }
        let output_proj = DenseLayer::new(hidden, out_features, Activation::None, true);
        Self { input_proj, blocks, output_proj }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = self.input_proj.forward(x);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.output_proj.forward(&h)
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let mut h = self.input_proj.forward_single(x);
        for block in &self.blocks {
            h = block.forward_single(&h);
        }
        self.output_proj.forward_single(&h)
    }

    pub fn num_params(&self) -> usize {
        let mut total = self.input_proj.num_params() + self.output_proj.num_params();
        for block in &self.blocks {
            for layer in &block.layers { total += layer.num_params(); }
        }
        total
    }
}

/// Embedding layer: lookup table [vocab_size, embed_dim]
#[derive(Clone, Debug)]
pub struct Embedding {
    pub weight: Tensor, // [vocab_size, embed_dim]
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let mut data = vec![0.0; vocab_size * embed_dim];
        let mut state = 987654321u64;
        for v in data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.1;
        }
        Self { weight: Tensor::from_vec(data, &[vocab_size, embed_dim]) }
    }

    pub fn lookup(&self, idx: usize) -> Tensor {
        let dim = self.weight.shape[1];
        let off = idx * dim;
        Tensor::from_slice(&self.weight.data[off..off + dim], &[dim])
    }

    pub fn lookup_batch(&self, indices: &[usize]) -> Tensor {
        let dim = self.weight.shape[1];
        let n = indices.len();
        let mut data = vec![0.0; n * dim];
        for (i, &idx) in indices.iter().enumerate() {
            let off = idx * dim;
            data[i * dim..(i + 1) * dim].copy_from_slice(&self.weight.data[off..off + dim]);
        }
        Tensor::from_vec(data, &[n, dim])
    }
}

/// Adaptive input: different embedding dimensions for different frequency buckets
#[derive(Clone, Debug)]
pub struct AdaptiveInput {
    pub embeddings: Vec<Embedding>,
    pub projections: Vec<DenseLayer>,
    pub cutoffs: Vec<usize>,
}

impl AdaptiveInput {
    pub fn new(cutoffs: &[usize], dims: &[usize], target_dim: usize) -> Self {
        assert_eq!(cutoffs.len(), dims.len());
        let mut embeddings = Vec::new();
        let mut projections = Vec::new();
        let mut prev = 0usize;
        for (i, (&cutoff, &dim)) in cutoffs.iter().zip(dims.iter()).enumerate() {
            let vocab = cutoff - prev;
            embeddings.push(Embedding::new(vocab, dim));
            projections.push(DenseLayer::new(dim, target_dim, Activation::None, false));
            prev = cutoff;
        }
        Self { embeddings, projections, cutoffs: cutoffs.to_vec() }
    }

    pub fn lookup(&self, idx: usize) -> Tensor {
        let mut prev = 0;
        for (i, &cutoff) in self.cutoffs.iter().enumerate() {
            if idx < cutoff {
                let local_idx = idx - prev;
                let emb = self.embeddings[i].lookup(local_idx);
                return self.projections[i].forward_single(&emb);
            }
            prev = cutoff;
        }
        panic!("index {} out of range", idx);
    }
}

/// Gradient-free feature crossing: x_i * x_j for selected pairs
#[derive(Clone, Debug)]
pub struct FeatureCross {
    pub pairs: Vec<(usize, usize)>,
}

impl FeatureCross {
    pub fn all_pairs(n: usize) -> Self {
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                pairs.push((i, j));
            }
        }
        Self { pairs }
    }

    pub fn selected(pairs: Vec<(usize, usize)>) -> Self {
        Self { pairs }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let n = self.pairs.len();
        let total = x.shape[0] + n;
        let mut data = vec![0.0; total];
        data[..x.shape[0]].copy_from_slice(&x.data);
        for (i, &(a, b)) in self.pairs.iter().enumerate() {
            data[x.shape[0] + i] = x.data[a] * x.data[b];
        }
        Tensor::from_vec(data, &[total])
    }

    pub fn forward_batch(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 2);
        let batch = x.shape[0];
        let feat = x.shape[1];
        let n = self.pairs.len();
        let new_feat = feat + n;
        let mut data = vec![0.0; batch * new_feat];
        for b in 0..batch {
            let off_in = b * feat;
            let off_out = b * new_feat;
            data[off_out..off_out + feat].copy_from_slice(&x.data[off_in..off_in + feat]);
            for (i, &(a, bb)) in self.pairs.iter().enumerate() {
                data[off_out + feat + i] = x.data[off_in + a] * x.data[off_in + bb];
            }
        }
        Tensor::from_vec(data, &[batch, new_feat])
    }

    pub fn output_size(&self, input_size: usize) -> usize { input_size + self.pairs.len() }
}

/// Input standardizer (online statistics or pre-computed)
#[derive(Clone, Debug)]
pub struct InputStandardizer {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl InputStandardizer {
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Self {
        assert_eq!(mean.len(), std.len());
        Self { mean, std }
    }

    pub fn identity(n: usize) -> Self {
        Self { mean: vec![0.0; n], std: vec![1.0; n] }
    }

    pub fn from_data(data: &[Vec<f64>]) -> Self {
        if data.is_empty() { return Self { mean: vec![], std: vec![] }; }
        let n = data[0].len();
        let m = data.len() as f64;
        let mut mean = vec![0.0; n];
        for row in data {
            for (i, &v) in row.iter().enumerate() { mean[i] += v; }
        }
        for v in mean.iter_mut() { *v /= m; }
        let mut var = vec![0.0; n];
        for row in data {
            for (i, &v) in row.iter().enumerate() { var[i] += (v - mean[i]) * (v - mean[i]); }
        }
        let std: Vec<f64> = var.iter().map(|&v| (v / m).sqrt().max(1e-8)).collect();
        Self { mean, std }
    }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let n = x.shape[0];
        let mut data = vec![0.0; n];
        for i in 0..n {
            data[i] = (x.data[i] - self.mean[i]) / self.std[i];
        }
        Tensor::from_vec(data, &[n])
    }

    pub fn transform_batch(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 2);
        let (batch, feat) = (x.shape[0], x.shape[1]);
        let mut data = vec![0.0; batch * feat];
        for b in 0..batch {
            for f in 0..feat {
                data[b * feat + f] = (x.data[b * feat + f] - self.mean[f]) / self.std[f];
            }
        }
        Tensor::from_vec(data, &[batch, feat])
    }

    pub fn inverse_transform(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let n = x.shape[0];
        let mut data = vec![0.0; n];
        for i in 0..n {
            data[i] = x.data[i] * self.std[i] + self.mean[i];
        }
        Tensor::from_vec(data, &[n])
    }
}

/// Multi-head bottleneck: project to smaller dim, apply function, project back
#[derive(Clone, Debug)]
pub struct BottleneckLayer {
    pub down_proj: DenseLayer,
    pub up_proj: DenseLayer,
    pub activation: Activation,
}

impl BottleneckLayer {
    pub fn new(dim: usize, bottleneck: usize, activation: Activation) -> Self {
        Self {
            down_proj: DenseLayer::new(dim, bottleneck, activation, true),
            up_proj: DenseLayer::new(bottleneck, dim, Activation::None, true),
            activation,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.down_proj.forward(x);
        self.up_proj.forward(&h)
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let h = self.down_proj.forward_single(x);
        self.up_proj.forward_single(&h)
    }
}

/// Dense block (DenseNet-style): each layer receives all previous outputs
#[derive(Clone, Debug)]
pub struct DenseBlock {
    pub layers: Vec<DenseLayer>,
    pub growth_rate: usize,
}

impl DenseBlock {
    pub fn new(in_features: usize, growth_rate: usize, num_layers: usize, activation: Activation) -> Self {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let in_f = in_features + i * growth_rate;
            layers.push(DenseLayer::new(in_f, growth_rate, activation, true));
        }
        Self { layers, growth_rate }
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let mut features = x.data.clone();
        let mut current_size = x.shape[0];
        for layer in &self.layers {
            let input = Tensor::from_vec(features.clone(), &[current_size]);
            let out = layer.forward_single(&input);
            features.extend_from_slice(&out.data);
            current_size += self.growth_rate;
        }
        Tensor::from_vec(features, &[current_size])
    }
}

/// Gradient checkpointing helper (inference: just pass through)
pub struct CheckpointSequential {
    pub layers: Vec<DenseLayer>,
}

impl CheckpointSequential {
    pub fn new(layers: Vec<DenseLayer>) -> Self { Self { layers } }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for layer in &self.layers { h = layer.forward(&h); }
        h
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for layer in &self.layers { h = layer.forward_single(&h); }
        h
    }
}

/// Grouped linear layer: split input into groups, apply separate linear to each
#[derive(Clone, Debug)]
pub struct GroupedLinear {
    pub num_groups: usize,
    pub group_layers: Vec<DenseLayer>,
}

impl GroupedLinear {
    pub fn new(in_features: usize, out_features: usize, num_groups: usize, activation: Activation) -> Self {
        assert!(in_features % num_groups == 0);
        assert!(out_features % num_groups == 0);
        let in_per = in_features / num_groups;
        let out_per = out_features / num_groups;
        let layers: Vec<DenseLayer> = (0..num_groups)
            .map(|_| DenseLayer::new(in_per, out_per, activation, true))
            .collect();
        Self { num_groups, group_layers: layers }
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let in_per = x.shape[0] / self.num_groups;
        let out_per = self.group_layers[0].out_features();
        let mut result = Vec::with_capacity(self.num_groups * out_per);
        for (g, layer) in self.group_layers.iter().enumerate() {
            let start = g * in_per;
            let group_input = Tensor::from_slice(&x.data[start..start + in_per], &[in_per]);
            let group_out = layer.forward_single(&group_input);
            result.extend_from_slice(&group_out.data);
        }
        Tensor::from_vec(result, &[self.num_groups * out_per])
    }
}

/// Multi-head linear projection (used in attention-like patterns)
#[derive(Clone, Debug)]
pub struct MultiHeadLinear {
    pub num_heads: usize,
    pub head_layers: Vec<DenseLayer>,
    pub output_proj: DenseLayer,
}

impl MultiHeadLinear {
    pub fn new(in_features: usize, head_dim: usize, num_heads: usize, out_features: usize) -> Self {
        let head_layers: Vec<DenseLayer> = (0..num_heads)
            .map(|_| DenseLayer::new(in_features, head_dim, Activation::None, true))
            .collect();
        let output_proj = DenseLayer::new(num_heads * head_dim, out_features, Activation::None, true);
        Self { num_heads, head_layers, output_proj }
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let mut heads = Vec::new();
        for layer in &self.head_layers {
            let h = layer.forward_single(x);
            heads.extend_from_slice(&h.data);
        }
        let concat = Tensor::from_vec(heads, &[self.num_heads * self.head_layers[0].out_features()]);
        self.output_proj.forward_single(&concat)
    }
}

/// Depthwise separable linear: depthwise (per-feature) then pointwise
#[derive(Clone, Debug)]
pub struct DepthwiseSeparableLinear {
    pub depthwise_scales: Vec<f64>,
    pub depthwise_biases: Vec<f64>,
    pub pointwise: DenseLayer,
}

impl DepthwiseSeparableLinear {
    pub fn new(features: usize, out_features: usize, activation: Activation) -> Self {
        let mut rng_state = 54321u64;
        let scales: Vec<f64> = (0..features).map(|_| {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state >> 32) as f64 / u32::MAX as f64 * 0.2 + 0.9
        }).collect();
        Self {
            depthwise_scales: scales,
            depthwise_biases: vec![0.0; features],
            pointwise: DenseLayer::new(features, out_features, activation, true),
        }
    }

    pub fn forward_single(&self, x: &Tensor) -> Tensor {
        let n = x.shape[0];
        let mut dw = vec![0.0; n];
        for i in 0..n {
            dw[i] = x.data[i] * self.depthwise_scales[i] + self.depthwise_biases[i];
        }
        let dw_t = Tensor::from_vec(dw, &[n]);
        self.pointwise.forward_single(&dw_t)
    }
}

/// Bilinear layer: f(x, y) = x^T W y + b
#[derive(Clone, Debug)]
pub struct BilinearLayer {
    pub weight: Tensor, // [in1, in2, out]
    pub bias: Tensor,   // [out]
    pub in1_features: usize,
    pub in2_features: usize,
    pub out_features: usize,
}

impl BilinearLayer {
    pub fn new(in1: usize, in2: usize, out: usize) -> Self {
        let scale = 1.0 / ((in1 * in2) as f64).sqrt();
        let mut state = 777u64;
        let data: Vec<f64> = (0..in1 * in2 * out).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 32) as f64 / u32::MAX as f64 * 2.0 - 1.0) * scale
        }).collect();
        Self {
            weight: Tensor::from_vec(data, &[in1, in2 * out]),
            bias: Tensor::zeros(&[out]),
            in1_features: in1, in2_features: in2, out_features: out,
        }
    }

    pub fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor {
        assert_eq!(x.shape[0], self.in1_features);
        assert_eq!(y.shape[0], self.in2_features);
        let mut result = vec![0.0; self.out_features];
        for o in 0..self.out_features {
            let mut val = 0.0;
            for i in 0..self.in1_features {
                for j in 0..self.in2_features {
                    val += x.data[i] * self.weight.data[i * (self.in2_features * self.out_features) + j * self.out_features + o] * y.data[j];
                }
            }
            result[o] = val + self.bias.data[o];
        }
        Tensor::from_vec(result, &[self.out_features])
    }
}

/// Polynomial feature expansion
#[derive(Clone, Debug)]
pub struct PolynomialFeatures {
    pub degree: usize,
    pub include_bias: bool,
}

impl PolynomialFeatures {
    pub fn new(degree: usize) -> Self { Self { degree, include_bias: true } }

    pub fn transform(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let n = x.shape[0];
        let mut features = Vec::new();
        if self.include_bias { features.push(1.0); }
        // degree 1
        features.extend_from_slice(&x.data);
        // degree 2+
        for d in 2..=self.degree {
            for i in 0..n {
                features.push(x.data[i].powi(d as i32));
            }
            // cross terms for degree 2
            if d == 2 {
                for i in 0..n {
                    for j in i + 1..n {
                        features.push(x.data[i] * x.data[j]);
                    }
                }
            }
        }
        let len = features.len();
        Tensor::from_vec(features, &[len])
    }

    pub fn output_size(&self, input_size: usize) -> usize {
        let mut size = if self.include_bias { 1 } else { 0 };
        size += input_size; // degree 1
        for d in 2..=self.degree {
            size += input_size; // pure powers
            if d == 2 { size += input_size * (input_size - 1) / 2; } // cross terms
        }
        size
    }
}

/// Learnable positional bias for 1D sequences
#[derive(Clone, Debug)]
pub struct PositionalBias {
    pub max_len: usize,
    pub bias: Vec<f64>,
}

impl PositionalBias {
    pub fn new(max_len: usize) -> Self {
        Self { max_len, bias: vec![0.0; max_len] }
    }

    pub fn apply(&self, x: &Tensor) -> Tensor {
        assert_eq!(x.ndim(), 1);
        let n = x.shape[0].min(self.max_len);
        let mut out = x.clone();
        for i in 0..n {
            out.data[i] += self.bias[i];
        }
        out
    }
}

/// Hinge loss computation
pub fn hinge_loss(predictions: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(predictions.len(), labels.len());
    predictions.iter().zip(labels.iter())
        .map(|(&p, &l)| (1.0 - l * p).max(0.0))
        .sum::<f64>() / predictions.len() as f64
}

/// Huber loss
pub fn huber_loss(predictions: &[f64], targets: &[f64], delta: f64) -> f64 {
    predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| {
            let diff = (p - t).abs();
            if diff <= delta { 0.5 * diff * diff } else { delta * (diff - 0.5 * delta) }
        })
        .sum::<f64>() / predictions.len() as f64
}

/// Quantile loss
pub fn quantile_loss(predictions: &[f64], targets: &[f64], quantile: f64) -> f64 {
    predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| {
            let diff = t - p;
            if diff >= 0.0 { quantile * diff } else { (quantile - 1.0) * diff }
        })
        .sum::<f64>() / predictions.len() as f64
}

/// Cross-entropy loss for binary classification
pub fn binary_cross_entropy(predictions: &[f64], targets: &[f64]) -> f64 {
    let eps = 1e-15;
    -predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| {
            let p = p.max(eps).min(1.0 - eps);
            t * p.ln() + (1.0 - t) * (1.0 - p).ln()
        })
        .sum::<f64>() / predictions.len() as f64
}

/// MSE loss
pub fn mse_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum::<f64>() / predictions.len() as f64
}

/// MAE loss
pub fn mae_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| (p - t).abs())
        .sum::<f64>() / predictions.len() as f64
}

/// R-squared metric
pub fn r_squared(predictions: &[f64], targets: &[f64]) -> f64 {
    let mean_t = targets.iter().sum::<f64>() / targets.len() as f64;
    let ss_res: f64 = predictions.iter().zip(targets.iter()).map(|(&p, &t)| (t - p).powi(2)).sum();
    let ss_tot: f64 = targets.iter().map(|&t| (t - mean_t).powi(2)).sum();
    if ss_tot < 1e-15 { 0.0 } else { 1.0 - ss_res / ss_tot }
}

/// Explained variance score
pub fn explained_variance(predictions: &[f64], targets: &[f64]) -> f64 {
    let n = predictions.len() as f64;
    let errors: Vec<f64> = predictions.iter().zip(targets.iter()).map(|(&p, &t)| t - p).collect();
    let mean_err = errors.iter().sum::<f64>() / n;
    let var_err = errors.iter().map(|&e| (e - mean_err).powi(2)).sum::<f64>() / n;
    let mean_t = targets.iter().sum::<f64>() / n;
    let var_t = targets.iter().map(|&t| (t - mean_t).powi(2)).sum::<f64>() / n;
    if var_t < 1e-15 { 0.0 } else { 1.0 - var_err / var_t }
}

/// Max error
pub fn max_error(predictions: &[f64], targets: &[f64]) -> f64 {
    predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| (p - t).abs())
        .fold(0.0f64, f64::max)
}

/// Median absolute error
pub fn median_absolute_error(predictions: &[f64], targets: &[f64]) -> f64 {
    let mut errors: Vec<f64> = predictions.iter().zip(targets.iter())
        .map(|(&p, &t)| (p - t).abs()).collect();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if errors.is_empty() { 0.0 }
    else if errors.len() % 2 == 0 { (errors[errors.len() / 2 - 1] + errors[errors.len() / 2]) / 2.0 }
    else { errors[errors.len() / 2] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_layer() {
        let layer = DenseLayer::new(4, 3, Activation::ReLU, true);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);
        let out = layer.forward(&x);
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    fn test_mlp() {
        let mlp = MLP::new(&[4, 8, 4, 2], Activation::ReLU, false, 0.0);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let out = mlp.forward(&x);
        assert_eq!(out.shape, vec![1, 2]);
    }

    #[test]
    fn test_batch_norm() {
        let bn = BatchNorm1d::new(3);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = bn.forward(&x);
        assert_eq!(out.shape, vec![2, 3]);
    }

    #[test]
    fn test_residual() {
        let layers = vec![
            DenseLayer::new(4, 4, Activation::ReLU, true),
            DenseLayer::new(4, 4, Activation::None, true),
        ];
        let block = ResidualBlock::new(layers, Some(LayerNorm::new(4)), 0.0, false);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let out = block.forward(&x);
        assert_eq!(out.shape, vec![1, 4]);
    }

    #[test]
    fn test_resnet() {
        let net = ResNet::new(4, 8, 2, 3, Activation::ReLU);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let out = net.forward(&x);
        assert_eq!(out.shape, vec![1, 2]);
    }

    #[test]
    fn test_glu() {
        let glu = GLU::new(4, 3);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let out = glu.forward(&x);
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    fn test_embedding() {
        let emb = Embedding::new(100, 16);
        let v = emb.lookup(5);
        assert_eq!(v.shape, vec![16]);
        let batch = emb.lookup_batch(&[0, 5, 10]);
        assert_eq!(batch.shape, vec![3, 16]);
    }

    #[test]
    fn test_standardizer() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let std = InputStandardizer::from_data(&data);
        let x = Tensor::from_vec(vec![3.0, 4.0], &[2]);
        let t = std.transform(&x);
        assert!((t.data[0]).abs() < 1e-10); // mean should be ~0
    }

    #[test]
    fn test_activations() {
        let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]);
        let relu = apply_activation(&x, Activation::ReLU);
        assert_eq!(relu.data[0], 0.0);
        assert_eq!(relu.data[2], 1.0);
        let gelu = apply_activation(&x, Activation::GELU);
        assert!(gelu.data[0] < 0.0); // GELU(-1) is slightly negative
        let swish = apply_activation(&x, Activation::Swish);
        assert!(swish.data[2] > 0.7);
    }

    #[test]
    fn test_dense_block() {
        let block = DenseBlock::new(4, 2, 3, Activation::ReLU);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let out = block.forward_single(&x);
        assert_eq!(out.shape, vec![10]); // 4 + 3*2
    }

    #[test]
    fn test_feature_cross() {
        let fc = FeatureCross::all_pairs(3);
        let x = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]);
        let out = fc.forward(&x);
        assert_eq!(out.shape[0], 6); // 3 original + 3 pairs
        assert!((out.data[3] - 6.0).abs() < 1e-10); // 2*3
        assert!((out.data[4] - 8.0).abs() < 1e-10); // 2*4
        assert!((out.data[5] - 12.0).abs() < 1e-10); // 3*4
    }
}
