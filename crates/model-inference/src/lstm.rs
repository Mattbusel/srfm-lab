// lstm.rs — LSTM cell, GRU cell, stacked LSTM, bidirectional, seq2one, seq2seq
use crate::tensor::Tensor;

/// LSTM cell weights for a single layer
#[derive(Clone, Debug)]
pub struct LSTMWeights {
    pub w_ii: Tensor, // [input_size, hidden_size] — input gate, input
    pub w_hi: Tensor, // [hidden_size, hidden_size] — input gate, hidden
    pub b_i: Tensor,  // [hidden_size]
    pub w_if: Tensor, // forget gate
    pub w_hf: Tensor,
    pub b_f: Tensor,
    pub w_ig: Tensor, // cell gate (g)
    pub w_hg: Tensor,
    pub b_g: Tensor,
    pub w_io: Tensor, // output gate
    pub w_ho: Tensor,
    pub b_o: Tensor,
}

impl LSTMWeights {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = 1.0 / (hidden_size as f64).sqrt();
        let mut rng = SimpleRng::new(42);
        let mut mk = |rows: usize, cols: usize| -> Tensor {
            let data: Vec<f64> = (0..rows * cols).map(|_| rng.uniform(-scale, scale)).collect();
            Tensor::from_vec(data, &[rows, cols])
        };
        let mkb = |size: usize| -> Tensor {
            Tensor::zeros(&[size])
        };
        Self {
            w_ii: mk(input_size, hidden_size), w_hi: mk(hidden_size, hidden_size), b_i: mkb(hidden_size),
            w_if: mk(input_size, hidden_size), w_hf: mk(hidden_size, hidden_size), b_f: Tensor::from_vec(vec![1.0; hidden_size], &[hidden_size]), // forget bias init=1
            w_ig: mk(input_size, hidden_size), w_hg: mk(hidden_size, hidden_size), b_g: mkb(hidden_size),
            w_io: mk(input_size, hidden_size), w_ho: mk(hidden_size, hidden_size), b_o: mkb(hidden_size),
        }
    }

    pub fn input_size(&self) -> usize { self.w_ii.shape[0] }
    pub fn hidden_size(&self) -> usize { self.w_ii.shape[1] }
}

struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
}

fn sigmoid_vec(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

fn tanh_vec(data: &[f64]) -> Vec<f64> {
    data.iter().map(|&x| x.tanh()).collect()
}

fn add_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn mul_vecs(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Single LSTM cell forward step
pub fn lstm_cell_forward(
    x: &Tensor,       // [input_size]
    h_prev: &Tensor,  // [hidden_size]
    c_prev: &Tensor,  // [hidden_size]
    w: &LSTMWeights,
) -> (Tensor, Tensor) {
    let hs = w.hidden_size();
    // i = sigma(W_ii x + W_hi h + b_i)
    let xi = x.vecmat(&w.w_ii);
    let hi = h_prev.vecmat(&w.w_hi);
    let i_gate = sigmoid_vec(&add_vecs(&add_vecs(&xi.data, &hi.data), &w.b_i.data));

    let xf = x.vecmat(&w.w_if);
    let hf = h_prev.vecmat(&w.w_hf);
    let f_gate = sigmoid_vec(&add_vecs(&add_vecs(&xf.data, &hf.data), &w.b_f.data));

    let xg = x.vecmat(&w.w_ig);
    let hg = h_prev.vecmat(&w.w_hg);
    let g_gate = tanh_vec(&add_vecs(&add_vecs(&xg.data, &hg.data), &w.b_g.data));

    let xo = x.vecmat(&w.w_io);
    let ho = h_prev.vecmat(&w.w_ho);
    let o_gate = sigmoid_vec(&add_vecs(&add_vecs(&xo.data, &ho.data), &w.b_o.data));

    // c = f * c_prev + i * g
    let fc = mul_vecs(&f_gate, &c_prev.data);
    let ig = mul_vecs(&i_gate, &g_gate);
    let c_new: Vec<f64> = add_vecs(&fc, &ig);

    // h = o * tanh(c)
    let tc = tanh_vec(&c_new);
    let h_new: Vec<f64> = mul_vecs(&o_gate, &tc);

    (Tensor::from_vec(h_new, &[hs]), Tensor::from_vec(c_new, &[hs]))
}

/// GRU cell weights
#[derive(Clone, Debug)]
pub struct GRUWeights {
    pub w_ir: Tensor, pub w_hr: Tensor, pub b_r: Tensor, // reset gate
    pub w_iz: Tensor, pub w_hz: Tensor, pub b_z: Tensor, // update gate
    pub w_in: Tensor, pub w_hn: Tensor, pub b_n: Tensor, // new gate
}

impl GRUWeights {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale = 1.0 / (hidden_size as f64).sqrt();
        let mut rng = SimpleRng::new(123);
        let mut mk = |r: usize, c: usize| -> Tensor {
            let data: Vec<f64> = (0..r * c).map(|_| rng.uniform(-scale, scale)).collect();
            Tensor::from_vec(data, &[r, c])
        };
        let z = |s: usize| Tensor::zeros(&[s]);
        Self {
            w_ir: mk(input_size, hidden_size), w_hr: mk(hidden_size, hidden_size), b_r: z(hidden_size),
            w_iz: mk(input_size, hidden_size), w_hz: mk(hidden_size, hidden_size), b_z: z(hidden_size),
            w_in: mk(input_size, hidden_size), w_hn: mk(hidden_size, hidden_size), b_n: z(hidden_size),
        }
    }

    pub fn input_size(&self) -> usize { self.w_ir.shape[0] }
    pub fn hidden_size(&self) -> usize { self.w_ir.shape[1] }
}

/// GRU cell forward
pub fn gru_cell_forward(
    x: &Tensor,
    h_prev: &Tensor,
    w: &GRUWeights,
) -> Tensor {
    let hs = w.hidden_size();
    let xr = x.vecmat(&w.w_ir);
    let hr = h_prev.vecmat(&w.w_hr);
    let r_gate = sigmoid_vec(&add_vecs(&add_vecs(&xr.data, &hr.data), &w.b_r.data));

    let xz = x.vecmat(&w.w_iz);
    let hz = h_prev.vecmat(&w.w_hz);
    let z_gate = sigmoid_vec(&add_vecs(&add_vecs(&xz.data, &hz.data), &w.b_z.data));

    let xn = x.vecmat(&w.w_in);
    let rh = mul_vecs(&r_gate, &h_prev.data);
    let rh_t = Tensor::from_vec(rh, &[hs]);
    let hn = rh_t.vecmat(&w.w_hn);
    let n_gate = tanh_vec(&add_vecs(&add_vecs(&xn.data, &hn.data), &w.b_n.data));

    // h = (1-z) * n + z * h_prev
    let h_new: Vec<f64> = (0..hs).map(|i| {
        (1.0 - z_gate[i]) * n_gate[i] + z_gate[i] * h_prev.data[i]
    }).collect();

    Tensor::from_vec(h_new, &[hs])
}

/// Peephole LSTM weights
#[derive(Clone, Debug)]
pub struct PeepholeLSTMWeights {
    pub base: LSTMWeights,
    pub w_ci: Tensor, // [hidden_size] peephole for input gate
    pub w_cf: Tensor, // [hidden_size] peephole for forget gate
    pub w_co: Tensor, // [hidden_size] peephole for output gate
}

impl PeepholeLSTMWeights {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let base = LSTMWeights::new(input_size, hidden_size);
        let mut rng = SimpleRng::new(777);
        let scale = 0.1;
        let mut mk = |s: usize| -> Tensor {
            let data: Vec<f64> = (0..s).map(|_| rng.uniform(-scale, scale)).collect();
            Tensor::from_vec(data, &[s])
        };
        Self {
            base,
            w_ci: mk(hidden_size),
            w_cf: mk(hidden_size),
            w_co: mk(hidden_size),
        }
    }
}

/// Peephole LSTM forward
pub fn peephole_lstm_cell_forward(
    x: &Tensor,
    h_prev: &Tensor,
    c_prev: &Tensor,
    w: &PeepholeLSTMWeights,
) -> (Tensor, Tensor) {
    let hs = w.base.hidden_size();
    let b = &w.base;

    let xi = x.vecmat(&b.w_ii);
    let hi = h_prev.vecmat(&b.w_hi);
    let ci: Vec<f64> = mul_vecs(&w.w_ci.data, &c_prev.data);
    let i_pre: Vec<f64> = (0..hs).map(|j| xi.data[j] + hi.data[j] + ci[j] + b.b_i.data[j]).collect();
    let i_gate = sigmoid_vec(&i_pre);

    let xf = x.vecmat(&b.w_if);
    let hf = h_prev.vecmat(&b.w_hf);
    let cf: Vec<f64> = mul_vecs(&w.w_cf.data, &c_prev.data);
    let f_pre: Vec<f64> = (0..hs).map(|j| xf.data[j] + hf.data[j] + cf[j] + b.b_f.data[j]).collect();
    let f_gate = sigmoid_vec(&f_pre);

    let xg = x.vecmat(&b.w_ig);
    let hg = h_prev.vecmat(&b.w_hg);
    let g_pre: Vec<f64> = (0..hs).map(|j| xg.data[j] + hg.data[j] + b.b_g.data[j]).collect();
    let g_gate = tanh_vec(&g_pre);

    let c_new: Vec<f64> = (0..hs).map(|j| f_gate[j] * c_prev.data[j] + i_gate[j] * g_gate[j]).collect();

    let xo = x.vecmat(&b.w_io);
    let ho = h_prev.vecmat(&b.w_ho);
    let co: Vec<f64> = mul_vecs(&w.w_co.data, &c_new);
    let o_pre: Vec<f64> = (0..hs).map(|j| xo.data[j] + ho.data[j] + co[j] + b.b_o.data[j]).collect();
    let o_gate = sigmoid_vec(&o_pre);

    let tc = tanh_vec(&c_new);
    let h_new: Vec<f64> = mul_vecs(&o_gate, &tc);

    (Tensor::from_vec(h_new, &[hs]), Tensor::from_vec(c_new, &[hs]))
}

/// Stacked LSTM: multiple layers
#[derive(Clone, Debug)]
pub struct StackedLSTM {
    pub layers: Vec<LSTMWeights>,
    pub dropout_rate: f64,
}

impl StackedLSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let inp = if i == 0 { input_size } else { hidden_size };
            layers.push(LSTMWeights::new(inp, hidden_size));
        }
        Self { layers, dropout_rate: 0.0 }
    }

    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Process full sequence: input [seq_len, input_size]
    /// Returns (outputs [seq_len, hidden_size], final_h [num_layers, hidden_size], final_c [num_layers, hidden_size])
    pub fn forward(&self, input: &Tensor) -> (Tensor, Vec<Tensor>, Vec<Tensor>) {
        let seq_len = input.shape[0];
        let hidden_size = self.layers[0].hidden_size();
        let num_layers = self.layers.len();

        let mut h_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hidden_size])).collect();
        let mut c_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hidden_size])).collect();

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.slice_along(0, t, t + 1).reshape(&[input.shape[1]]);
            let mut layer_input = x_t;

            for l in 0..num_layers {
                let (h_new, c_new) = lstm_cell_forward(&layer_input, &h_states[l], &c_states[l], &self.layers[l]);
                h_states[l] = h_new.clone();
                c_states[l] = c_new;
                layer_input = h_new;
            }
            outputs.push(layer_input);
        }

        // Stack outputs into [seq_len, hidden_size]
        let out_data: Vec<f64> = outputs.iter().flat_map(|t| t.data.clone()).collect();
        let out_tensor = Tensor::from_vec(out_data, &[seq_len, hidden_size]);

        (out_tensor, h_states, c_states)
    }

    /// Sequence to one: returns only final hidden state of last layer
    pub fn seq2one(&self, input: &Tensor) -> Tensor {
        let (_, h_states, _) = self.forward(input);
        h_states.last().unwrap().clone()
    }

    /// Sequence to sequence: returns all timestep outputs
    pub fn seq2seq(&self, input: &Tensor) -> Tensor {
        let (outputs, _, _) = self.forward(input);
        outputs
    }

    pub fn num_params(&self) -> usize {
        self.layers.iter().map(|w| {
            let is = w.input_size();
            let hs = w.hidden_size();
            4 * (is * hs + hs * hs + hs)
        }).sum()
    }
}

/// Stacked GRU
#[derive(Clone, Debug)]
pub struct StackedGRU {
    pub layers: Vec<GRUWeights>,
}

impl StackedGRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let inp = if i == 0 { input_size } else { hidden_size };
            layers.push(GRUWeights::new(inp, hidden_size));
        }
        Self { layers }
    }

    pub fn forward(&self, input: &Tensor) -> (Tensor, Vec<Tensor>) {
        let seq_len = input.shape[0];
        let hidden_size = self.layers[0].hidden_size();
        let num_layers = self.layers.len();

        let mut h_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hidden_size])).collect();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = input.slice_along(0, t, t + 1).reshape(&[input.shape[1]]);
            let mut layer_input = x_t;

            for l in 0..num_layers {
                let h_new = gru_cell_forward(&layer_input, &h_states[l], &self.layers[l]);
                h_states[l] = h_new.clone();
                layer_input = h_new;
            }
            outputs.push(layer_input);
        }

        let out_data: Vec<f64> = outputs.iter().flat_map(|t| t.data.clone()).collect();
        (Tensor::from_vec(out_data, &[seq_len, hidden_size]), h_states)
    }

    pub fn seq2one(&self, input: &Tensor) -> Tensor {
        let (_, h) = self.forward(input);
        h.last().unwrap().clone()
    }

    pub fn seq2seq(&self, input: &Tensor) -> Tensor {
        let (out, _) = self.forward(input);
        out
    }
}

/// Bidirectional LSTM
#[derive(Clone, Debug)]
pub struct BidirectionalLSTM {
    pub forward_lstm: StackedLSTM,
    pub backward_lstm: StackedLSTM,
    pub merge_mode: MergeMode,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MergeMode {
    Concat,
    Sum,
    Average,
    Multiply,
}

impl BidirectionalLSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, merge: MergeMode) -> Self {
        Self {
            forward_lstm: StackedLSTM::new(input_size, hidden_size, num_layers),
            backward_lstm: StackedLSTM::new(input_size, hidden_size, num_layers),
            merge_mode: merge,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape[0];
        let feat = input.shape[1];

        let (fwd_out, _, _) = self.forward_lstm.forward(input);

        // reverse input
        let mut rev_data = vec![0.0; seq_len * feat];
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * feat;
            let dst = t * feat;
            rev_data[dst..dst + feat].copy_from_slice(&input.data[src..src + feat]);
        }
        let rev_input = Tensor::from_vec(rev_data, &[seq_len, feat]);
        let (bwd_out_rev, _, _) = self.backward_lstm.forward(&rev_input);

        // reverse backward output
        let hs = bwd_out_rev.shape[1];
        let mut bwd_data = vec![0.0; seq_len * hs];
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * hs;
            let dst = t * hs;
            bwd_data[dst..dst + hs].copy_from_slice(&bwd_out_rev.data[src..src + hs]);
        }
        let bwd_out = Tensor::from_vec(bwd_data, &[seq_len, hs]);

        match self.merge_mode {
            MergeMode::Concat => {
                let mut data = vec![0.0; seq_len * 2 * hs];
                for t in 0..seq_len {
                    data[t * 2 * hs..t * 2 * hs + hs].copy_from_slice(&fwd_out.data[t * hs..(t + 1) * hs]);
                    data[t * 2 * hs + hs..t * 2 * hs + 2 * hs].copy_from_slice(&bwd_out.data[t * hs..(t + 1) * hs]);
                }
                Tensor::from_vec(data, &[seq_len, 2 * hs])
            }
            MergeMode::Sum => fwd_out.add(&bwd_out),
            MergeMode::Average => fwd_out.add(&bwd_out).mul_scalar(0.5),
            MergeMode::Multiply => fwd_out.mul_elem(&bwd_out),
        }
    }

    pub fn seq2one(&self, input: &Tensor) -> Tensor {
        let out = self.forward(input);
        let seq_len = out.shape[0];
        let feat = out.shape[1];
        Tensor::from_slice(&out.data[(seq_len - 1) * feat..seq_len * feat], &[feat])
    }

    pub fn output_size(&self) -> usize {
        let hs = self.forward_lstm.layers[0].hidden_size();
        match self.merge_mode {
            MergeMode::Concat => 2 * hs,
            _ => hs,
        }
    }
}

/// Bidirectional GRU
#[derive(Clone, Debug)]
pub struct BidirectionalGRU {
    pub forward_gru: StackedGRU,
    pub backward_gru: StackedGRU,
    pub merge_mode: MergeMode,
}

impl BidirectionalGRU {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, merge: MergeMode) -> Self {
        Self {
            forward_gru: StackedGRU::new(input_size, hidden_size, num_layers),
            backward_gru: StackedGRU::new(input_size, hidden_size, num_layers),
            merge_mode: merge,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape[0];
        let feat = input.shape[1];

        let (fwd_out, _) = self.forward_gru.forward(input);

        let mut rev_data = vec![0.0; seq_len * feat];
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * feat;
            let dst = t * feat;
            rev_data[dst..dst + feat].copy_from_slice(&input.data[src..src + feat]);
        }
        let rev_input = Tensor::from_vec(rev_data, &[seq_len, feat]);
        let (bwd_out_rev, _) = self.backward_gru.forward(&rev_input);

        let hs = bwd_out_rev.shape[1];
        let mut bwd_data = vec![0.0; seq_len * hs];
        for t in 0..seq_len {
            let src = (seq_len - 1 - t) * hs;
            let dst = t * hs;
            bwd_data[dst..dst + hs].copy_from_slice(&bwd_out_rev.data[src..src + hs]);
        }
        let bwd_out = Tensor::from_vec(bwd_data, &[seq_len, hs]);

        match self.merge_mode {
            MergeMode::Concat => {
                let mut data = vec![0.0; seq_len * 2 * hs];
                for t in 0..seq_len {
                    data[t * 2 * hs..t * 2 * hs + hs].copy_from_slice(&fwd_out.data[t * hs..(t + 1) * hs]);
                    data[t * 2 * hs + hs..t * 2 * hs + 2 * hs].copy_from_slice(&bwd_out.data[t * hs..(t + 1) * hs]);
                }
                Tensor::from_vec(data, &[seq_len, 2 * hs])
            }
            MergeMode::Sum => fwd_out.add(&bwd_out),
            MergeMode::Average => fwd_out.add(&bwd_out).mul_scalar(0.5),
            MergeMode::Multiply => fwd_out.mul_elem(&bwd_out),
        }
    }
}

/// LSTM with attention pooling over sequence
#[derive(Clone, Debug)]
pub struct LSTMWithAttention {
    pub lstm: StackedLSTM,
    pub attn_w: Tensor, // [hidden_size]
}

impl LSTMWithAttention {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut rng = SimpleRng::new(999);
        let scale = 1.0 / (hidden_size as f64).sqrt();
        let w: Vec<f64> = (0..hidden_size).map(|_| rng.uniform(-scale, scale)).collect();
        Self {
            lstm: StackedLSTM::new(input_size, hidden_size, num_layers),
            attn_w: Tensor::from_vec(w, &[hidden_size]),
        }
    }

    /// Returns attention-weighted sum of LSTM outputs
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let (outputs, _, _) = self.lstm.forward(input); // [seq_len, hidden]
        let seq_len = outputs.shape[0];
        let hs = outputs.shape[1];

        // scores = outputs @ attn_w -> [seq_len]
        let mut scores = vec![0.0; seq_len];
        for t in 0..seq_len {
            let mut s = 0.0;
            for h in 0..hs {
                s += outputs.data[t * hs + h] * self.attn_w.data[h];
            }
            scores[t] = s;
        }

        // softmax
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        for v in exp_scores.iter_mut() { *v /= sum; }

        // weighted sum
        let mut result = vec![0.0; hs];
        for t in 0..seq_len {
            let w = exp_scores[t];
            for h in 0..hs {
                result[h] += w * outputs.data[t * hs + h];
            }
        }

        Tensor::from_vec(result, &[hs])
    }
}

/// Sequence encoder: LSTM + various pooling modes
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoolingMode {
    Last,
    First,
    Mean,
    Max,
    Attention,
}

#[derive(Clone, Debug)]
pub struct SeqEncoder {
    pub lstm: StackedLSTM,
    pub pooling: PoolingMode,
    pub attn_w: Option<Tensor>,
}

impl SeqEncoder {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize, pooling: PoolingMode) -> Self {
        let attn_w = if pooling == PoolingMode::Attention {
            let mut rng = SimpleRng::new(555);
            let scale = 1.0 / (hidden_size as f64).sqrt();
            Some(Tensor::from_vec((0..hidden_size).map(|_| rng.uniform(-scale, scale)).collect(), &[hidden_size]))
        } else {
            None
        };
        Self {
            lstm: StackedLSTM::new(input_size, hidden_size, num_layers),
            pooling,
            attn_w,
        }
    }

    pub fn encode(&self, input: &Tensor) -> Tensor {
        let (outputs, h_states, _) = self.lstm.forward(input);
        let seq_len = outputs.shape[0];
        let hs = outputs.shape[1];

        match self.pooling {
            PoolingMode::Last => h_states.last().unwrap().clone(),
            PoolingMode::First => {
                Tensor::from_slice(&outputs.data[0..hs], &[hs])
            }
            PoolingMode::Mean => {
                let mut result = vec![0.0; hs];
                for t in 0..seq_len {
                    for h in 0..hs {
                        result[h] += outputs.data[t * hs + h];
                    }
                }
                for v in result.iter_mut() { *v /= seq_len as f64; }
                Tensor::from_vec(result, &[hs])
            }
            PoolingMode::Max => {
                let mut result = vec![f64::NEG_INFINITY; hs];
                for t in 0..seq_len {
                    for h in 0..hs {
                        result[h] = result[h].max(outputs.data[t * hs + h]);
                    }
                }
                Tensor::from_vec(result, &[hs])
            }
            PoolingMode::Attention => {
                let w = self.attn_w.as_ref().unwrap();
                let mut scores = vec![0.0; seq_len];
                for t in 0..seq_len {
                    for h in 0..hs {
                        scores[t] += outputs.data[t * hs + h] * w.data[h];
                    }
                }
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_s: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
                let sum: f64 = exp_s.iter().sum();
                for v in exp_s.iter_mut() { *v /= sum; }
                let mut result = vec![0.0; hs];
                for t in 0..seq_len {
                    for h in 0..hs {
                        result[h] += exp_s[t] * outputs.data[t * hs + h];
                    }
                }
                Tensor::from_vec(result, &[hs])
            }
        }
    }
}

/// Variational dropout: same mask across time steps
pub struct VariationalDropout {
    pub rate: f64,
    pub training: bool,
}

impl VariationalDropout {
    pub fn new(rate: f64) -> Self { Self { rate, training: false } }

    pub fn forward(&self, seq: &Tensor) -> Tensor {
        if !self.training || self.rate == 0.0 { return seq.clone(); }
        let seq_len = seq.shape[0];
        let feat = seq.shape[1];
        let scale = 1.0 / (1.0 - self.rate);
        let threshold = (self.rate * u32::MAX as f64) as u32;
        // generate mask for one timestep
        let mut mask = vec![1.0f64; feat];
        let mut state = 314159u64;
        for i in 0..feat {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (state >> 33) as u32;
            mask[i] = if r < threshold { 0.0 } else { scale };
        }
        let mut out = seq.clone();
        for t in 0..seq_len {
            for f in 0..feat {
                out.data[t * feat + f] *= mask[f];
            }
        }
        out
    }
}

/// Zone-out: stochastically preserve previous hidden state
pub struct ZoneOut {
    pub rate: f64,
    pub training: bool,
}

impl ZoneOut {
    pub fn new(rate: f64) -> Self { Self { rate, training: false } }

    pub fn apply(&self, h_new: &Tensor, h_prev: &Tensor) -> Tensor {
        if !self.training {
            // At inference: interpolate
            let mut out = h_new.mul_scalar(1.0 - self.rate);
            out = out.add(&h_prev.mul_scalar(self.rate));
            return out;
        }
        h_new.clone()
    }
}

/// LSTM with skip connections across layers
#[derive(Clone, Debug)]
pub struct SkipLSTM {
    pub layers: Vec<LSTMWeights>,
    pub skip_proj: Vec<Option<Tensor>>, // projection if sizes differ
}

impl SkipLSTM {
    pub fn new(input_size: usize, hidden_size: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        let mut skip_proj = Vec::new();
        for i in 0..num_layers {
            let inp = if i == 0 { input_size } else { hidden_size };
            layers.push(LSTMWeights::new(inp, hidden_size));
            if i >= 2 {
                skip_proj.push(None); // same size, no proj needed
            } else {
                skip_proj.push(None);
            }
        }
        Self { layers, skip_proj }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape[0];
        let hidden_size = self.layers[0].hidden_size();
        let num_layers = self.layers.len();

        let mut h_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hidden_size])).collect();
        let mut c_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hidden_size])).collect();
        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let x_t = input.slice_along(0, t, t + 1).reshape(&[input.shape[1]]);
            let mut layer_outputs: Vec<Tensor> = Vec::new();
            let mut layer_input = x_t;

            for l in 0..num_layers {
                let (h_new, c_new) = lstm_cell_forward(&layer_input, &h_states[l], &c_states[l], &self.layers[l]);
                // skip connection from layer l-2
                let h_out = if l >= 2 {
                    h_new.add(&layer_outputs[l - 2])
                } else {
                    h_new
                };
                h_states[l] = h_out.clone();
                c_states[l] = c_new;
                layer_outputs.push(h_out.clone());
                layer_input = h_out;
            }
            outputs.push(layer_input);
        }

        let out_data: Vec<f64> = outputs.iter().flat_map(|t| t.data.clone()).collect();
        Tensor::from_vec(out_data, &[seq_len, hidden_size])
    }
}

/// Multi-scale LSTM: run multiple LSTMs at different temporal scales
#[derive(Clone, Debug)]
pub struct MultiScaleLSTM {
    pub scales: Vec<(usize, StackedLSTM)>, // (stride, lstm)
    pub merge_proj: Tensor, // projection after concat
    pub merge_bias: Tensor,
}

impl MultiScaleLSTM {
    pub fn new(input_size: usize, hidden_size: usize, scales: &[usize]) -> Self {
        let lstms: Vec<(usize, StackedLSTM)> = scales.iter().map(|&s| {
            (s, StackedLSTM::new(input_size, hidden_size, 1))
        }).collect();
        let total = hidden_size * scales.len();
        let mut rng = SimpleRng::new(12345);
        let sc = 1.0 / (total as f64).sqrt();
        let proj_data: Vec<f64> = (0..total * hidden_size).map(|_| rng.uniform(-sc, sc)).collect();
        Self {
            scales: lstms,
            merge_proj: Tensor::from_vec(proj_data, &[total, hidden_size]),
            merge_bias: Tensor::zeros(&[hidden_size]),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.shape[0];
        let hidden_size = self.scales[0].1.layers[0].hidden_size();
        let mut scale_outputs = Vec::new();

        for (stride, lstm) in &self.scales {
            // subsample
            let sub_len = (seq_len + stride - 1) / stride;
            let feat = input.shape[1];
            let mut sub_data = vec![0.0; sub_len * feat];
            for i in 0..sub_len {
                let t = (i * stride).min(seq_len - 1);
                sub_data[i * feat..(i + 1) * feat].copy_from_slice(&input.data[t * feat..(t + 1) * feat]);
            }
            let sub_input = Tensor::from_vec(sub_data, &[sub_len, feat]);
            let h = lstm.seq2one(&sub_input);
            scale_outputs.push(h);
        }

        // concat all scale outputs
        let total_size: usize = scale_outputs.iter().map(|t| t.numel()).sum();
        let mut concat = vec![0.0; total_size];
        let mut off = 0;
        for t in &scale_outputs {
            concat[off..off + t.numel()].copy_from_slice(&t.data);
            off += t.numel();
        }
        let cat = Tensor::from_vec(concat, &[total_size]);

        // project
        let out = cat.vecmat(&self.merge_proj);
        out.add(&self.merge_bias)
    }
}

/// Sequence decoder: auto-regressive LSTM
#[derive(Clone, Debug)]
pub struct SeqDecoder {
    pub lstm: StackedLSTM,
    pub output_proj: Tensor, // [hidden_size, output_size]
    pub output_bias: Tensor,
}

impl SeqDecoder {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, num_layers: usize) -> Self {
        let mut rng = SimpleRng::new(777);
        let sc = 1.0 / (hidden_size as f64).sqrt();
        let proj: Vec<f64> = (0..hidden_size * output_size).map(|_| rng.uniform(-sc, sc)).collect();
        Self {
            lstm: StackedLSTM::new(input_size, hidden_size, num_layers),
            output_proj: Tensor::from_vec(proj, &[hidden_size, output_size]),
            output_bias: Tensor::zeros(&[output_size]),
        }
    }

    /// Decode for `steps` timesteps given initial input
    pub fn decode(&self, initial_input: &Tensor, steps: usize) -> Tensor {
        let hs = self.lstm.layers[0].hidden_size();
        let os = self.output_proj.shape[1];
        let num_layers = self.lstm.layers.len();

        let mut h_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hs])).collect();
        let mut c_states: Vec<Tensor> = (0..num_layers).map(|_| Tensor::zeros(&[hs])).collect();

        let mut x = initial_input.clone();
        let mut outputs = Vec::with_capacity(steps);

        for _ in 0..steps {
            let mut layer_input = x.clone();
            for l in 0..num_layers {
                let (h_new, c_new) = lstm_cell_forward(&layer_input, &h_states[l], &c_states[l], &self.lstm.layers[l]);
                h_states[l] = h_new.clone();
                c_states[l] = c_new;
                layer_input = h_new;
            }
            let out = layer_input.vecmat(&self.output_proj).add(&self.output_bias);
            outputs.push(out.clone());
            // feed output back as next input (teacher forcing off)
            x = out;
        }

        let out_data: Vec<f64> = outputs.iter().flat_map(|t| t.data.clone()).collect();
        Tensor::from_vec(out_data, &[steps, os])
    }
}

/// Encoder-decoder LSTM
#[derive(Clone, Debug)]
pub struct Seq2Seq {
    pub encoder: StackedLSTM,
    pub decoder: SeqDecoder,
}

impl Seq2Seq {
    pub fn new(enc_input: usize, dec_input: usize, hidden: usize, output: usize, enc_layers: usize, dec_layers: usize) -> Self {
        Self {
            encoder: StackedLSTM::new(enc_input, hidden, enc_layers),
            decoder: SeqDecoder::new(dec_input, hidden, output, dec_layers),
        }
    }

    pub fn forward(&self, enc_input: &Tensor, dec_initial: &Tensor, dec_steps: usize) -> Tensor {
        // encode
        let (_, enc_h, enc_c) = self.encoder.forward(enc_input);
        // initialize decoder states (use encoder final states for matching layers)
        self.decoder.decode(dec_initial, dec_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lstm_cell() {
        let w = LSTMWeights::new(4, 8);
        let x = Tensor::from_vec(vec![1.0, 0.5, -0.5, 0.2], &[4]);
        let h = Tensor::zeros(&[8]);
        let c = Tensor::zeros(&[8]);
        let (h_new, c_new) = lstm_cell_forward(&x, &h, &c, &w);
        assert_eq!(h_new.shape, vec![8]);
        assert_eq!(c_new.shape, vec![8]);
        assert!(h_new.is_finite());
    }

    #[test]
    fn test_gru_cell() {
        let w = GRUWeights::new(4, 8);
        let x = Tensor::from_vec(vec![1.0, 0.5, -0.5, 0.2], &[4]);
        let h = Tensor::zeros(&[8]);
        let h_new = gru_cell_forward(&x, &h, &w);
        assert_eq!(h_new.shape, vec![8]);
        assert!(h_new.is_finite());
    }

    #[test]
    fn test_stacked_lstm() {
        let lstm = StackedLSTM::new(4, 8, 2);
        let input = Tensor::from_vec(vec![
            1.0, 0.5, -0.5, 0.2,
            0.3, -0.1, 0.8, -0.4,
            0.7, 0.2, 0.1, -0.3,
        ], &[3, 4]);
        let (out, h, c) = lstm.forward(&input);
        assert_eq!(out.shape, vec![3, 8]);
        assert_eq!(h.len(), 2);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn test_bidirectional() {
        let bilstm = BidirectionalLSTM::new(4, 8, 1, MergeMode::Concat);
        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 0.2, 0.3, -0.1, 0.8, -0.4], &[2, 4]);
        let out = bilstm.forward(&input);
        assert_eq!(out.shape, vec![2, 16]);
    }

    #[test]
    fn test_seq_encoder() {
        let enc = SeqEncoder::new(4, 8, 1, PoolingMode::Mean);
        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 0.2, 0.3, -0.1, 0.8, -0.4], &[2, 4]);
        let out = enc.encode(&input);
        assert_eq!(out.shape, vec![8]);
    }

    #[test]
    fn test_seq_decoder() {
        let dec = SeqDecoder::new(3, 8, 3, 1);
        let init = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]);
        let out = dec.decode(&init, 5);
        assert_eq!(out.shape, vec![5, 3]);
    }

    #[test]
    fn test_stacked_gru() {
        let gru = StackedGRU::new(4, 8, 2);
        let input = Tensor::from_vec(vec![1.0, 0.5, -0.5, 0.2, 0.3, -0.1, 0.8, -0.4], &[2, 4]);
        let (out, h) = gru.forward(&input);
        assert_eq!(out.shape, vec![2, 8]);
        assert_eq!(h.len(), 2);
    }
}
