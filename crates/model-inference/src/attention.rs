// attention.rs — Scaled dot-product attention, multi-head, causal mask, positional encoding, cross-attention
use crate::tensor::Tensor;
use crate::dense_layer::{DenseLayer, Activation, LayerNorm, Dropout};

/// Scaled dot-product attention
/// Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v]
/// Returns: [seq_q, d_v]
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> (Tensor, Tensor) {
    let d_k = q.shape[1] as f64;
    let scale = 1.0 / d_k.sqrt();

    // scores = Q K^T / sqrt(d_k) : [seq_q, seq_k]
    let kt = k.transpose();
    let mut scores = q.matmul(&kt).mul_scalar(scale);

    // apply mask (additive, -inf for masked positions)
    if let Some(m) = mask {
        assert_eq!(scores.shape, m.shape);
        let data: Vec<f64> = scores.data.iter().zip(m.data.iter())
            .map(|(&s, &mv)| if mv < 0.5 { s + (-1e9) } else { s })
            .collect();
        scores = Tensor::from_vec(data, &scores.shape);
    }

    // softmax along last dim
    let weights = scores.softmax(1);

    // output = weights V : [seq_q, d_v]
    let output = weights.matmul(v);

    (output, weights)
}

/// Batched scaled dot-product attention
/// Q: [batch, seq_q, d_k], K: [batch, seq_k, d_k], V: [batch, seq_k, d_v]
pub fn batched_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Tensor {
    assert_eq!(q.ndim(), 3);
    let batch = q.shape[0];
    let seq_q = q.shape[1];
    let d_k = q.shape[2];
    let seq_k = k.shape[1];
    let d_v = v.shape[2];

    let scale = 1.0 / (d_k as f64).sqrt();
    let mut out_data = vec![0.0; batch * seq_q * d_v];

    for b in 0..batch {
        // extract Q[b], K[b], V[b]
        let q_off = b * seq_q * d_k;
        let k_off = b * seq_k * d_k;
        let v_off = b * seq_k * d_v;

        // scores[i][j] = sum_d Q[b,i,d] * K[b,j,d] / sqrt(d_k)
        for i in 0..seq_q {
            // compute scores for row i
            let mut scores = vec![0.0; seq_k];
            for j in 0..seq_k {
                let mut s = 0.0;
                for d in 0..d_k {
                    s += q.data[q_off + i * d_k + d] * k.data[k_off + j * d_k + d];
                }
                scores[j] = s * scale;
            }

            // apply mask
            if let Some(m) = mask {
                if m.ndim() == 2 {
                    for j in 0..seq_k {
                        if m.data[i * seq_k + j] < 0.5 { scores[j] = -1e9; }
                    }
                } else if m.ndim() == 3 {
                    let m_off = b * seq_q * seq_k;
                    for j in 0..seq_k {
                        if m.data[m_off + i * seq_k + j] < 0.5 { scores[j] = -1e9; }
                    }
                }
            }

            // softmax
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f64 = exp_s.iter().sum();
            for v_s in exp_s.iter_mut() { *v_s /= sum; }

            // weighted sum of V
            let o_off = b * seq_q * d_v + i * d_v;
            for d in 0..d_v {
                let mut val = 0.0;
                for j in 0..seq_k {
                    val += exp_s[j] * v.data[v_off + j * d_v + d];
                }
                out_data[o_off + d] = val;
            }
        }
    }

    Tensor::from_vec(out_data, &[batch, seq_q, d_v])
}

/// Create causal mask: [seq_len, seq_len] where mask[i][j] = 1 if j <= i
pub fn causal_mask(seq_len: usize) -> Tensor {
    let mut data = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..=i {
            data[i * seq_len + j] = 1.0;
        }
    }
    Tensor::from_vec(data, &[seq_len, seq_len])
}

/// Create padding mask from lengths: [batch, max_len]
pub fn padding_mask(lengths: &[usize], max_len: usize) -> Tensor {
    let batch = lengths.len();
    let mut data = vec![0.0; batch * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for j in 0..len.min(max_len) {
            data[b * max_len + j] = 1.0;
        }
    }
    Tensor::from_vec(data, &[batch, max_len])
}

/// Combine causal + padding mask
pub fn combined_mask(lengths: &[usize], max_len: usize) -> Tensor {
    let batch = lengths.len();
    let cm = causal_mask(max_len);
    let pm = padding_mask(lengths, max_len);
    let mut data = vec![0.0; batch * max_len * max_len];
    for b in 0..batch {
        for i in 0..max_len {
            for j in 0..max_len {
                data[b * max_len * max_len + i * max_len + j] =
                    cm.data[i * max_len + j] * pm.data[b * max_len + j];
            }
        }
    }
    Tensor::from_vec(data, &[batch, max_len, max_len])
}

/// Sinusoidal positional encoding
pub fn sinusoidal_positional_encoding(max_len: usize, d_model: usize) -> Tensor {
    let mut data = vec![0.0; max_len * d_model];
    for pos in 0..max_len {
        for i in 0..d_model {
            let angle = pos as f64 / (10000.0f64).powf((2 * (i / 2)) as f64 / d_model as f64);
            data[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    Tensor::from_vec(data, &[max_len, d_model])
}

/// Rotary positional encoding (RoPE) — apply to Q,K
pub fn apply_rope(x: &Tensor, pos_start: usize) -> Tensor {
    assert_eq!(x.ndim(), 2); // [seq_len, d]
    let seq_len = x.shape[0];
    let d = x.shape[1];
    assert!(d % 2 == 0);
    let mut out = x.clone();
    for t in 0..seq_len {
        let pos = (pos_start + t) as f64;
        for i in 0..d / 2 {
            let theta = pos / (10000.0f64).powf(2.0 * i as f64 / d as f64);
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let x0 = x.data[t * d + 2 * i];
            let x1 = x.data[t * d + 2 * i + 1];
            out.data[t * d + 2 * i] = x0 * cos_t - x1 * sin_t;
            out.data[t * d + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
    out
}

/// Learned positional embedding
#[derive(Clone, Debug)]
pub struct LearnedPositionalEncoding {
    pub embeddings: Tensor, // [max_len, d_model]
}

impl LearnedPositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        // init with small random values
        let mut data = vec![0.0; max_len * d_model];
        let mut state = 42u64;
        for v in data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02;
        }
        Self { embeddings: Tensor::from_vec(data, &[max_len, d_model]) }
    }

    pub fn forward(&self, seq_len: usize) -> Tensor {
        let d = self.embeddings.shape[1];
        Tensor::from_slice(&self.embeddings.data[..seq_len * d], &[seq_len, d])
    }
}

/// ALiBi (Attention with Linear Biases) slopes
pub fn alibi_slopes(num_heads: usize) -> Vec<f64> {
    let base = 2.0f64.powf(-(8.0 / num_heads as f64));
    (0..num_heads).map(|h| base.powi((h + 1) as i32)).collect()
}

/// Generate ALiBi bias matrix for given slope: [seq_q, seq_k]
pub fn alibi_bias(slope: f64, seq_q: usize, seq_k: usize) -> Tensor {
    let mut data = vec![0.0; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            data[i * seq_k + j] = slope * (j as f64 - i as f64);
        }
    }
    Tensor::from_vec(data, &[seq_q, seq_k])
}

/// Multi-head attention
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub w_q: Tensor, // [d_model, num_heads * d_k]
    pub w_k: Tensor,
    pub w_v: Tensor, // [d_model, num_heads * d_v]
    pub w_o: Tensor, // [num_heads * d_v, d_model]
    pub b_q: Tensor,
    pub b_k: Tensor,
    pub b_v: Tensor,
    pub b_o: Tensor,
    pub dropout: Dropout,
}

struct Rng64(u64);
impl Rng64 {
    fn new(s: u64) -> Self { Self(s) }
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
    fn fill(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(d_model % num_heads == 0);
        let d_k = d_model / num_heads;
        let d_v = d_k;
        let mut rng = Rng64::new(42);
        let sc = 2.0 / (d_model as f64).sqrt();
        Self {
            num_heads, d_model, d_k, d_v,
            w_q: Tensor::from_vec(rng.fill(d_model * d_model, sc), &[d_model, d_model]),
            w_k: Tensor::from_vec(rng.fill(d_model * d_model, sc), &[d_model, d_model]),
            w_v: Tensor::from_vec(rng.fill(d_model * d_model, sc), &[d_model, d_model]),
            w_o: Tensor::from_vec(rng.fill(d_model * d_model, sc), &[d_model, d_model]),
            b_q: Tensor::zeros(&[d_model]),
            b_k: Tensor::zeros(&[d_model]),
            b_v: Tensor::zeros(&[d_model]),
            b_o: Tensor::zeros(&[d_model]),
            dropout: Dropout::new(0.0),
        }
    }

    pub fn with_dims(d_model: usize, num_heads: usize, d_k: usize, d_v: usize) -> Self {
        let mut rng = Rng64::new(42);
        let sc = 2.0 / (d_model as f64).sqrt();
        let hdk = num_heads * d_k;
        let hdv = num_heads * d_v;
        Self {
            num_heads, d_model, d_k, d_v,
            w_q: Tensor::from_vec(rng.fill(d_model * hdk, sc), &[d_model, hdk]),
            w_k: Tensor::from_vec(rng.fill(d_model * hdk, sc), &[d_model, hdk]),
            w_v: Tensor::from_vec(rng.fill(d_model * hdv, sc), &[d_model, hdv]),
            w_o: Tensor::from_vec(rng.fill(hdv * d_model, sc), &[hdv, d_model]),
            b_q: Tensor::zeros(&[hdk]),
            b_k: Tensor::zeros(&[hdk]),
            b_v: Tensor::zeros(&[hdv]),
            b_o: Tensor::zeros(&[d_model]),
            dropout: Dropout::new(0.0),
        }
    }

    /// Forward: input [seq_len, d_model] (self-attention)
    /// Returns [seq_len, d_model]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        self.forward_qkv(x, x, x, mask)
    }

    /// Forward with separate Q, K, V sources (cross-attention)
    pub fn forward_qkv(&self, q_src: &Tensor, k_src: &Tensor, v_src: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let seq_q = q_src.shape[0];
        let seq_k = k_src.shape[0];

        // project: Q = q_src @ W_q + b_q
        let q_proj = self.linear_proj(q_src, &self.w_q, &self.b_q);
        let k_proj = self.linear_proj(k_src, &self.w_k, &self.b_k);
        let v_proj = self.linear_proj(v_src, &self.w_v, &self.b_v);

        let hdk = self.num_heads * self.d_k;
        let hdv = self.num_heads * self.d_v;

        // Split heads: [seq, num_heads * d_k] -> [num_heads, seq, d_k]
        let q_heads = self.split_heads(&q_proj, seq_q, self.d_k);
        let k_heads = self.split_heads(&k_proj, seq_k, self.d_k);
        let v_heads = self.split_heads(&v_proj, seq_k, self.d_v);

        // Attention per head
        let attn_out = batched_attention(&q_heads, &k_heads, &v_heads, mask);

        // Merge heads: [num_heads, seq_q, d_v] -> [seq_q, num_heads * d_v]
        let merged = self.merge_heads(&attn_out, seq_q, self.d_v);

        // Output projection
        self.linear_proj(&merged, &self.w_o, &self.b_o)
    }

    fn linear_proj(&self, x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
        let out = x.matmul(w);
        let b_broad = b.unsqueeze(0).broadcast_to(&out.shape);
        out.add(&b_broad)
    }

    fn split_heads(&self, x: &Tensor, seq_len: usize, d: usize) -> Tensor {
        // x: [seq_len, num_heads * d] -> [num_heads, seq_len, d]
        let nh = self.num_heads;
        let mut data = vec![0.0; nh * seq_len * d];
        for t in 0..seq_len {
            for h in 0..nh {
                for dd in 0..d {
                    data[h * seq_len * d + t * d + dd] = x.data[t * nh * d + h * d + dd];
                }
            }
        }
        Tensor::from_vec(data, &[nh, seq_len, d])
    }

    fn merge_heads(&self, x: &Tensor, seq_len: usize, d: usize) -> Tensor {
        // x: [num_heads, seq_len, d] -> [seq_len, num_heads * d]
        let nh = self.num_heads;
        let mut data = vec![0.0; seq_len * nh * d];
        for t in 0..seq_len {
            for h in 0..nh {
                for dd in 0..d {
                    data[t * nh * d + h * d + dd] = x.data[h * seq_len * d + t * d + dd];
                }
            }
        }
        Tensor::from_vec(data, &[seq_len, nh * d])
    }

    pub fn num_params(&self) -> usize {
        self.w_q.numel() + self.w_k.numel() + self.w_v.numel() + self.w_o.numel()
            + self.b_q.numel() + self.b_k.numel() + self.b_v.numel() + self.b_o.numel()
    }
}

/// Multi-query attention (shared K,V across heads)
#[derive(Clone, Debug)]
pub struct MultiQueryAttention {
    pub num_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Tensor,  // [d_model, num_heads * d_k]
    pub w_k: Tensor,  // [d_model, d_k] — single head
    pub w_v: Tensor,  // [d_model, d_k]
    pub w_o: Tensor,
    pub b_q: Tensor,
    pub b_k: Tensor,
    pub b_v: Tensor,
    pub b_o: Tensor,
}

impl MultiQueryAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        let d_k = d_model / num_heads;
        let mut rng = Rng64::new(99);
        let sc = 2.0 / (d_model as f64).sqrt();
        let hdk = num_heads * d_k;
        Self {
            num_heads, d_model, d_k,
            w_q: Tensor::from_vec(rng.fill(d_model * hdk, sc), &[d_model, hdk]),
            w_k: Tensor::from_vec(rng.fill(d_model * d_k, sc), &[d_model, d_k]),
            w_v: Tensor::from_vec(rng.fill(d_model * d_k, sc), &[d_model, d_k]),
            w_o: Tensor::from_vec(rng.fill(hdk * d_model, sc), &[hdk, d_model]),
            b_q: Tensor::zeros(&[hdk]),
            b_k: Tensor::zeros(&[d_k]),
            b_v: Tensor::zeros(&[d_k]),
            b_o: Tensor::zeros(&[d_model]),
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let seq_len = x.shape[0];
        let d_k = self.d_k;
        let nh = self.num_heads;

        let q_all = x.matmul(&self.w_q).add(&self.b_q.unsqueeze(0).broadcast_to(&[seq_len, nh * d_k]));
        let k_single = x.matmul(&self.w_k).add(&self.b_k.unsqueeze(0).broadcast_to(&[seq_len, d_k]));
        let v_single = x.matmul(&self.w_v).add(&self.b_v.unsqueeze(0).broadcast_to(&[seq_len, d_k]));

        let scale = 1.0 / (d_k as f64).sqrt();
        let mut out_data = vec![0.0; seq_len * nh * d_k];

        for h in 0..nh {
            for i in 0..seq_len {
                let mut scores = vec![0.0; seq_len];
                for j in 0..seq_len {
                    let mut s = 0.0;
                    for d in 0..d_k {
                        s += q_all.data[i * nh * d_k + h * d_k + d] * k_single.data[j * d_k + d];
                    }
                    scores[j] = s * scale;
                }
                if let Some(m) = mask {
                    for j in 0..seq_len {
                        if m.data[i * seq_len + j] < 0.5 { scores[j] = -1e9; }
                    }
                }
                // softmax
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum: f64 = exp_s.iter().sum();
                for v in exp_s.iter_mut() { *v /= sum; }

                for d in 0..d_k {
                    let mut val = 0.0;
                    for j in 0..seq_len {
                        val += exp_s[j] * v_single.data[j * d_k + d];
                    }
                    out_data[i * nh * d_k + h * d_k + d] = val;
                }
            }
        }

        let merged = Tensor::from_vec(out_data, &[seq_len, nh * d_k]);
        let out = merged.matmul(&self.w_o);
        out.add(&self.b_o.unsqueeze(0).broadcast_to(&out.shape))
    }
}

/// Grouped-query attention
#[derive(Clone, Debug)]
pub struct GroupedQueryAttention {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub d_model: usize,
    pub d_k: usize,
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
    pub w_o: Tensor,
}

impl GroupedQueryAttention {
    pub fn new(d_model: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        assert!(num_heads % num_kv_heads == 0);
        let d_k = d_model / num_heads;
        let mut rng = Rng64::new(77);
        let sc = 2.0 / (d_model as f64).sqrt();
        Self {
            num_heads, num_kv_heads, d_model, d_k,
            w_q: Tensor::from_vec(rng.fill(d_model * num_heads * d_k, sc), &[d_model, num_heads * d_k]),
            w_k: Tensor::from_vec(rng.fill(d_model * num_kv_heads * d_k, sc), &[d_model, num_kv_heads * d_k]),
            w_v: Tensor::from_vec(rng.fill(d_model * num_kv_heads * d_k, sc), &[d_model, num_kv_heads * d_k]),
            w_o: Tensor::from_vec(rng.fill(num_heads * d_k * d_model, sc), &[num_heads * d_k, d_model]),
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let seq_len = x.shape[0];
        let d_k = self.d_k;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let heads_per_group = nh / nkv;

        let q_all = x.matmul(&self.w_q); // [seq, nh*dk]
        let k_all = x.matmul(&self.w_k); // [seq, nkv*dk]
        let v_all = x.matmul(&self.w_v);

        let scale = 1.0 / (d_k as f64).sqrt();
        let mut out_data = vec![0.0; seq_len * nh * d_k];

        for h in 0..nh {
            let kv_h = h / heads_per_group;
            for i in 0..seq_len {
                let mut scores = vec![0.0; seq_len];
                for j in 0..seq_len {
                    let mut s = 0.0;
                    for d in 0..d_k {
                        s += q_all.data[i * nh * d_k + h * d_k + d]
                           * k_all.data[j * nkv * d_k + kv_h * d_k + d];
                    }
                    scores[j] = s * scale;
                }
                if let Some(m) = mask {
                    for j in 0..seq_len {
                        if m.data[i * seq_len + j] < 0.5 { scores[j] = -1e9; }
                    }
                }
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum: f64 = exp_s.iter().sum();
                for v in exp_s.iter_mut() { *v /= sum; }

                for d in 0..d_k {
                    let mut val = 0.0;
                    for j in 0..seq_len {
                        val += exp_s[j] * v_all.data[j * nkv * d_k + kv_h * d_k + d];
                    }
                    out_data[i * nh * d_k + h * d_k + d] = val;
                }
            }
        }

        let merged = Tensor::from_vec(out_data, &[seq_len, nh * d_k]);
        merged.matmul(&self.w_o)
    }
}

/// Feed-forward network (for transformer blocks)
#[derive(Clone, Debug)]
pub struct FeedForward {
    pub w1: Tensor,
    pub b1: Tensor,
    pub w2: Tensor,
    pub b2: Tensor,
    pub activation: Activation,
    pub dropout: Dropout,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize, activation: Activation) -> Self {
        let mut rng = Rng64::new(55);
        let sc1 = 2.0 / (d_model as f64).sqrt();
        let sc2 = 2.0 / (d_ff as f64).sqrt();
        Self {
            w1: Tensor::from_vec(rng.fill(d_model * d_ff, sc1), &[d_model, d_ff]),
            b1: Tensor::zeros(&[d_ff]),
            w2: Tensor::from_vec(rng.fill(d_ff * d_model, sc2), &[d_ff, d_model]),
            b2: Tensor::zeros(&[d_model]),
            activation,
            dropout: Dropout::new(0.0),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = x.matmul(&self.w1).add(&self.b1.unsqueeze(0).broadcast_to(&[x.shape[0], self.w1.shape[1]]));
        let h = crate::dense_layer::apply_activation(&h, self.activation);
        let h = self.dropout.forward(&h);
        let out = h.matmul(&self.w2);
        out.add(&self.b2.unsqueeze(0).broadcast_to(&out.shape))
    }
}

/// SwiGLU feed-forward (used in LLaMA-style)
#[derive(Clone, Debug)]
pub struct SwiGLUFeedForward {
    pub w1: Tensor,
    pub w3: Tensor, // gate
    pub w2: Tensor,
}

impl SwiGLUFeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        let mut rng = Rng64::new(66);
        let sc = 2.0 / (d_model as f64).sqrt();
        Self {
            w1: Tensor::from_vec(rng.fill(d_model * d_ff, sc), &[d_model, d_ff]),
            w3: Tensor::from_vec(rng.fill(d_model * d_ff, sc), &[d_model, d_ff]),
            w2: Tensor::from_vec(rng.fill(d_ff * d_model, sc), &[d_ff, d_model]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = x.matmul(&self.w1).swish();
        let h3 = x.matmul(&self.w3);
        let h = h1.mul_elem(&h3);
        h.matmul(&self.w2)
    }
}

/// Transformer encoder layer
#[derive(Clone, Debug)]
pub struct TransformerEncoderLayer {
    pub self_attn: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub dropout: Dropout,
    pub pre_norm: bool,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, pre_norm: bool) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            ff: FeedForward::new(d_model, d_ff, Activation::GELU),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            dropout: Dropout::new(0.0),
            pre_norm,
        }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        if self.pre_norm {
            let normed = self.norm1.forward(x);
            let attn_out = self.self_attn.forward(&normed, mask);
            let h = x.add(&self.dropout.forward(&attn_out));
            let normed2 = self.norm2.forward(&h);
            let ff_out = self.ff.forward(&normed2);
            h.add(&self.dropout.forward(&ff_out))
        } else {
            let attn_out = self.self_attn.forward(x, mask);
            let h = self.norm1.forward(&x.add(&self.dropout.forward(&attn_out)));
            let ff_out = self.ff.forward(&h);
            self.norm2.forward(&h.add(&self.dropout.forward(&ff_out)))
        }
    }
}

/// Transformer decoder layer (with cross-attention)
#[derive(Clone, Debug)]
pub struct TransformerDecoderLayer {
    pub self_attn: MultiHeadAttention,
    pub cross_attn: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub dropout: Dropout,
    pub pre_norm: bool,
}

impl TransformerDecoderLayer {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, pre_norm: bool) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, num_heads),
            cross_attn: MultiHeadAttention::new(d_model, num_heads),
            ff: FeedForward::new(d_model, d_ff, Activation::GELU),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            norm3: LayerNorm::new(d_model),
            dropout: Dropout::new(0.0),
            pre_norm,
        }
    }

    pub fn forward(&self, x: &Tensor, memory: &Tensor, self_mask: Option<&Tensor>, cross_mask: Option<&Tensor>) -> Tensor {
        if self.pre_norm {
            let n1 = self.norm1.forward(x);
            let sa = self.self_attn.forward(&n1, self_mask);
            let h = x.add(&self.dropout.forward(&sa));
            let n2 = self.norm2.forward(&h);
            let ca = self.cross_attn.forward_qkv(&n2, memory, memory, cross_mask);
            let h2 = h.add(&self.dropout.forward(&ca));
            let n3 = self.norm3.forward(&h2);
            let ff = self.ff.forward(&n3);
            h2.add(&self.dropout.forward(&ff))
        } else {
            let sa = self.self_attn.forward(x, self_mask);
            let h = self.norm1.forward(&x.add(&self.dropout.forward(&sa)));
            let ca = self.cross_attn.forward_qkv(&h, memory, memory, cross_mask);
            let h2 = self.norm2.forward(&h.add(&self.dropout.forward(&ca)));
            let ff = self.ff.forward(&h2);
            self.norm3.forward(&h2.add(&self.dropout.forward(&ff)))
        }
    }
}

/// Full transformer encoder
#[derive(Clone, Debug)]
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    pub pos_enc: Tensor,
    pub final_norm: Option<LayerNorm>,
}

impl TransformerEncoder {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize, num_layers: usize, max_len: usize, pre_norm: bool) -> Self {
        let layers: Vec<_> = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new(d_model, num_heads, d_ff, pre_norm))
            .collect();
        let pos_enc = sinusoidal_positional_encoding(max_len, d_model);
        let final_norm = if pre_norm { Some(LayerNorm::new(d_model)) } else { None };
        Self { layers, pos_enc, final_norm }
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let seq_len = x.shape[0];
        let d = x.shape[1];
        let pe = Tensor::from_slice(&self.pos_enc.data[..seq_len * d], &[seq_len, d]);
        let mut h = x.add(&pe);
        for layer in &self.layers {
            h = layer.forward(&h, mask);
        }
        if let Some(ref norm) = self.final_norm {
            h = norm.forward(&h);
        }
        h
    }
}

/// KV-cache for autoregressive inference
#[derive(Clone, Debug)]
pub struct KVCache {
    pub k_cache: Vec<Vec<f64>>, // [layer][accumulated seq * d_k]
    pub v_cache: Vec<Vec<f64>>,
    pub num_layers: usize,
    pub d_k: usize,
    pub seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, d_k: usize) -> Self {
        Self {
            k_cache: vec![vec![]; num_layers],
            v_cache: vec![vec![]; num_layers],
            num_layers, d_k, seq_len: 0,
        }
    }

    pub fn append(&mut self, layer: usize, k: &[f64], v: &[f64]) {
        self.k_cache[layer].extend_from_slice(k);
        self.v_cache[layer].extend_from_slice(v);
        if layer == 0 {
            self.seq_len += k.len() / self.d_k;
        }
    }

    pub fn get_k(&self, layer: usize) -> Tensor {
        let len = self.k_cache[layer].len() / self.d_k;
        Tensor::from_vec(self.k_cache[layer].clone(), &[len, self.d_k])
    }

    pub fn get_v(&self, layer: usize) -> Tensor {
        let len = self.v_cache[layer].len() / self.d_k;
        Tensor::from_vec(self.v_cache[layer].clone(), &[len, self.d_k])
    }

    pub fn clear(&mut self) {
        for l in 0..self.num_layers {
            self.k_cache[l].clear();
            self.v_cache[l].clear();
        }
        self.seq_len = 0;
    }
}

/// Sliding window attention mask
pub fn sliding_window_mask(seq_len: usize, window_size: usize) -> Tensor {
    let mut data = vec![0.0; seq_len * seq_len];
    for i in 0..seq_len {
        let start = if i >= window_size { i - window_size } else { 0 };
        for j in start..=i {
            data[i * seq_len + j] = 1.0;
        }
    }
    Tensor::from_vec(data, &[seq_len, seq_len])
}

/// Relative position bias (T5-style)
#[derive(Clone, Debug)]
pub struct RelativePositionBias {
    pub num_buckets: usize,
    pub max_distance: usize,
    pub num_heads: usize,
    pub bias_table: Tensor, // [num_buckets, num_heads]
}

impl RelativePositionBias {
    pub fn new(num_buckets: usize, max_distance: usize, num_heads: usize) -> Self {
        let mut rng = Rng64::new(321);
        let data: Vec<f64> = (0..num_buckets * num_heads).map(|_| rng.next() * 0.02).collect();
        Self {
            num_buckets, max_distance, num_heads,
            bias_table: Tensor::from_vec(data, &[num_buckets, num_heads]),
        }
    }

    fn relative_position_bucket(relative_pos: i64, num_buckets: usize, max_distance: usize) -> usize {
        let mut nb = num_buckets;
        let mut ret = 0usize;
        nb /= 2;
        let is_neg = relative_pos < 0;
        let rp = relative_pos.unsigned_abs() as usize;
        if is_neg { ret += nb; }
        let max_exact = nb / 2;
        if rp < max_exact {
            ret += rp;
        } else {
            let val = max_exact as f64
                + ((rp as f64 / max_exact as f64).ln() / (max_distance as f64 / max_exact as f64).ln()
                   * (nb - max_exact) as f64);
            ret += (val as usize).min(nb - 1);
        }
        ret
    }

    pub fn compute_bias(&self, seq_q: usize, seq_k: usize) -> Tensor {
        // returns [num_heads, seq_q, seq_k]
        let nh = self.num_heads;
        let mut data = vec![0.0; nh * seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                let rp = j as i64 - i as i64;
                let bucket = Self::relative_position_bucket(rp, self.num_buckets, self.max_distance);
                for h in 0..nh {
                    data[h * seq_q * seq_k + i * seq_k + j] = self.bias_table.data[bucket * nh + h];
                }
            }
        }
        Tensor::from_vec(data, &[nh, seq_q, seq_k])
    }
}

/// Cross-attention layer (separate encoder/decoder)
#[derive(Clone, Debug)]
pub struct CrossAttention {
    pub mha: MultiHeadAttention,
    pub norm_q: LayerNorm,
    pub norm_kv: LayerNorm,
}

impl CrossAttention {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            mha: MultiHeadAttention::new(d_model, num_heads),
            norm_q: LayerNorm::new(d_model),
            norm_kv: LayerNorm::new(d_model),
        }
    }

    pub fn forward(&self, query: &Tensor, key_value: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let q = self.norm_q.forward(query);
        let kv = self.norm_kv.forward(key_value);
        let attn = self.mha.forward_qkv(&q, &kv, &kv, mask);
        query.add(&attn)
    }
}

/// Linear attention (O(n) complexity approximation)
pub fn linear_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    let seq_q = q.shape[0];
    let seq_k = k.shape[0];
    let d_k = q.shape[1];
    let d_v = v.shape[1];

    // Apply ELU+1 feature map
    let phi_q = q.map(|x| if x > 0.0 { x + 1.0 } else { x.exp() });
    let phi_k = k.map(|x| if x > 0.0 { x + 1.0 } else { x.exp() });

    // S = sum_j phi(k_j) v_j^T : [d_k, d_v]
    let mut s_data = vec![0.0; d_k * d_v];
    for j in 0..seq_k {
        for dk in 0..d_k {
            for dv in 0..d_v {
                s_data[dk * d_v + dv] += phi_k.data[j * d_k + dk] * v.data[j * d_v + dv];
            }
        }
    }
    let s = Tensor::from_vec(s_data, &[d_k, d_v]);

    // z = sum_j phi(k_j) : [d_k]
    let mut z_data = vec![0.0; d_k];
    for j in 0..seq_k {
        for dk in 0..d_k {
            z_data[dk] += phi_k.data[j * d_k + dk];
        }
    }

    // output_i = phi(q_i) S / (phi(q_i) . z)
    let mut out_data = vec![0.0; seq_q * d_v];
    for i in 0..seq_q {
        let mut denom = 0.0;
        for dk in 0..d_k {
            denom += phi_q.data[i * d_k + dk] * z_data[dk];
        }
        denom = denom.max(1e-12);
        for dv in 0..d_v {
            let mut num = 0.0;
            for dk in 0..d_k {
                num += phi_q.data[i * d_k + dk] * s_data[dk * d_v + dv];
            }
            out_data[i * d_v + dv] = num / denom;
        }
    }

    Tensor::from_vec(out_data, &[seq_q, d_v])
}

/// Performer random feature attention
pub fn random_feature_attention(q: &Tensor, k: &Tensor, v: &Tensor, num_features: usize, seed: u64) -> Tensor {
    let d_k = q.shape[1];
    let seq_q = q.shape[0];
    let seq_k = k.shape[0];
    let d_v = v.shape[1];

    // generate random projection
    let mut rng = Rng64(seed);
    let omega: Vec<f64> = (0..d_k * num_features).map(|_| rng.next() * 2.0).collect();

    let project = |x: &Tensor| -> Tensor {
        let seq = x.shape[0];
        let mut out = vec![0.0; seq * num_features];
        let scale = 1.0 / (num_features as f64).sqrt();
        for t in 0..seq {
            for f in 0..num_features {
                let mut dot = 0.0;
                for d in 0..d_k {
                    dot += x.data[t * d_k + d] * omega[d * num_features + f];
                }
                out[t * num_features + f] = (dot.cos() + dot.sin()) * scale;
            }
        }
        Tensor::from_vec(out, &[seq, num_features])
    };

    let q_proj = project(q);
    let k_proj = project(k);

    // Linear attention on projected features
    linear_attention(&q_proj, &k_proj, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product() {
        let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (out, weights) = scaled_dot_product_attention(&q, &k, &v, None);
        assert_eq!(out.shape, vec![2, 2]);
        assert!(out.is_finite());
    }

    #[test]
    fn test_causal_mask() {
        let m = causal_mask(4);
        assert_eq!(m.data[0], 1.0); // [0,0]
        assert_eq!(m.data[1], 0.0); // [0,1]
        assert_eq!(m.data[5], 1.0); // [1,1]
    }

    #[test]
    fn test_positional_encoding() {
        let pe = sinusoidal_positional_encoding(10, 8);
        assert_eq!(pe.shape, vec![10, 8]);
        assert!(pe.is_finite());
    }

    #[test]
    fn test_multi_head_attention() {
        let mha = MultiHeadAttention::new(8, 2);
        let x = Tensor::from_vec((0..24).map(|i| i as f64 * 0.1).collect(), &[3, 8]);
        let out = mha.forward(&x, None);
        assert_eq!(out.shape, vec![3, 8]);
        assert!(out.is_finite());
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let layer = TransformerEncoderLayer::new(8, 2, 16, true);
        let x = Tensor::from_vec((0..24).map(|i| i as f64 * 0.1).collect(), &[3, 8]);
        let out = layer.forward(&x, None);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_transformer_encoder() {
        let enc = TransformerEncoder::new(8, 2, 16, 2, 100, true);
        let x = Tensor::from_vec((0..24).map(|i| i as f64 * 0.1).collect(), &[3, 8]);
        let out = enc.forward(&x, None);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_rope() {
        let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5], &[2, 4]);
        let out = apply_rope(&x, 0);
        assert_eq!(out.shape, vec![2, 4]);
        assert!(out.is_finite());
    }

    #[test]
    fn test_linear_attention() {
        let q = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[3, 2]);
        let k = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[3, 2]);
        let v = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let out = linear_attention(&q, &k, &v);
        assert_eq!(out.shape, vec![3, 2]);
        assert!(out.is_finite());
    }

    #[test]
    fn test_sliding_window() {
        let m = sliding_window_mask(5, 2);
        assert_eq!(m.data[0 * 5 + 0], 1.0);
        assert_eq!(m.data[4 * 5 + 1], 0.0); // outside window
        assert_eq!(m.data[4 * 5 + 2], 1.0); // inside window
    }

    #[test]
    fn test_cross_attention() {
        let ca = CrossAttention::new(8, 2);
        let q = Tensor::from_vec((0..16).map(|i| i as f64 * 0.1).collect(), &[2, 8]);
        let kv = Tensor::from_vec((0..24).map(|i| i as f64 * 0.1).collect(), &[3, 8]);
        let out = ca.forward(&q, &kv, None);
        assert_eq!(out.shape, vec![2, 8]);
    }
}
