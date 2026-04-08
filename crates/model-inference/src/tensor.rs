// tensor.rs — N-dimensional tensor with matmul, broadcast, SIMD dot, reshape, slice, transpose
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, Index, IndexMut};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() { return strides; }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn flat_index(indices: &[usize], strides: &[usize]) -> usize {
    indices.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

fn multi_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    let mut rem = flat;
    for i in 0..shape.len() {
        let vol: usize = shape[i + 1..].iter().product();
        if vol == 0 { break; }
        idx[i] = rem / vol;
        rem %= vol;
    }
    idx
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape: shape.to_vec(), strides: compute_strides(shape) }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![1.0; n], shape: shape.to_vec(), strides: compute_strides(shape) }
    }

    pub fn full(shape: &[usize], val: f64) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![val; n], shape: shape.to_vec(), strides: compute_strides(shape) }
    }

    pub fn from_vec(data: Vec<f64>, shape: &[usize]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "data len mismatch shape");
        Self { data, shape: shape.to_vec(), strides: compute_strides(shape) }
    }

    pub fn from_slice(data: &[f64], shape: &[usize]) -> Self {
        Self::from_vec(data.to_vec(), shape)
    }

    pub fn scalar(v: f64) -> Self {
        Self { data: vec![v], shape: vec![], strides: vec![] }
    }

    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn numel(&self) -> usize { self.data.len() }
    pub fn is_scalar(&self) -> bool { self.shape.is_empty() }
    pub fn is_vector(&self) -> bool { self.shape.len() == 1 }
    pub fn is_matrix(&self) -> bool { self.shape.len() == 2 }

    pub fn get(&self, indices: &[usize]) -> f64 {
        self.data[flat_index(indices, &self.strides)]
    }

    pub fn set(&mut self, indices: &[usize], val: f64) {
        let idx = flat_index(indices, &self.strides);
        self.data[idx] = val;
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: size mismatch");
        Self::from_vec(self.data.clone(), new_shape)
    }

    pub fn reshape_inplace(&mut self, new_shape: &[usize]) {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape_inplace: size mismatch");
        self.shape = new_shape.to_vec();
        self.strides = compute_strides(new_shape);
    }

    pub fn flatten(&self) -> Self {
        self.reshape(&[self.numel()])
    }

    pub fn transpose(&self) -> Self {
        assert!(self.ndim() == 2, "transpose requires 2D");
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = Tensor::zeros(&[c, r]);
        for i in 0..r {
            for j in 0..c {
                out.set(&[j, i], self.get(&[i, j]));
            }
        }
        out
    }

    pub fn transpose_axes(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.ndim());
        let new_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let n = self.numel();
        let mut out = Tensor::zeros(&new_shape);
        for flat in 0..n {
            let src_idx = multi_index(flat, &self.shape);
            let dst_idx: Vec<usize> = axes.iter().map(|&a| src_idx[a]).collect();
            out.set(&dst_idx, self.data[flat]);
        }
        out
    }

    pub fn slice(&self, ranges: &[(usize, usize)]) -> Self {
        assert_eq!(ranges.len(), self.ndim());
        let new_shape: Vec<usize> = ranges.iter().map(|(s, e)| e - s).collect();
        let n: usize = new_shape.iter().product();
        let mut out = Tensor::zeros(&new_shape);
        for flat in 0..n {
            let out_idx = multi_index(flat, &new_shape);
            let src_idx: Vec<usize> = out_idx.iter().zip(ranges.iter()).map(|(o, (s, _))| o + s).collect();
            let val = self.get(&src_idx);
            out.data[flat] = val;
        }
        out
    }

    pub fn slice_along(&self, axis: usize, start: usize, end: usize) -> Self {
        let mut ranges: Vec<(usize, usize)> = self.shape.iter().map(|&s| (0, s)).collect();
        ranges[axis] = (start, end);
        self.slice(&ranges)
    }

    pub fn concat(tensors: &[&Tensor], axis: usize) -> Self {
        assert!(!tensors.is_empty());
        let ndim = tensors[0].ndim();
        for t in tensors.iter().skip(1) {
            assert_eq!(t.ndim(), ndim);
            for d in 0..ndim {
                if d != axis { assert_eq!(t.shape[d], tensors[0].shape[d]); }
            }
        }
        let mut new_shape = tensors[0].shape.clone();
        new_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();
        let mut out = Tensor::zeros(&new_shape);
        let mut offset = 0usize;
        for t in tensors {
            let n = t.numel();
            for flat in 0..n {
                let mut idx = multi_index(flat, &t.shape);
                idx[axis] += offset;
                out.set(&idx, t.data[flat]);
            }
            offset += t.shape[axis];
        }
        out
    }

    pub fn unsqueeze(&self, axis: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        self.reshape(&new_shape)
    }

    pub fn squeeze(&self, axis: usize) -> Self {
        assert_eq!(self.shape[axis], 1);
        let mut new_shape = self.shape.clone();
        new_shape.remove(axis);
        if new_shape.is_empty() { return Tensor::scalar(self.data[0]); }
        self.reshape(&new_shape)
    }

    pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let ndim = a.len().max(b.len());
        let mut result = vec![0usize; ndim];
        for i in 0..ndim {
            let da = if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
            let db = if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
            assert!(da == db || da == 1 || db == 1, "broadcast incompatible: {} vs {}", da, db);
            result[i] = da.max(db);
        }
        result
    }

    pub fn broadcast_to(&self, target: &[usize]) -> Self {
        let n: usize = target.iter().product();
        let mut out = Tensor::zeros(target);
        let ndim = target.len();
        let offset = ndim - self.ndim();
        for flat in 0..n {
            let out_idx = multi_index(flat, target);
            let src_idx: Vec<usize> = (0..self.ndim())
                .map(|d| if self.shape[d] == 1 { 0 } else { out_idx[d + offset] })
                .collect();
            out.data[flat] = self.get(&src_idx);
        }
        out
    }

    fn elementwise_binary<F: Fn(f64, f64) -> f64>(&self, other: &Tensor, op: F) -> Self {
        let target = Self::broadcast_shape(&self.shape, &other.shape);
        let a = self.broadcast_to(&target);
        let b = other.broadcast_to(&target);
        let data: Vec<f64> = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| op(x, y)).collect();
        Self::from_vec(data, &target)
    }

    pub fn add(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a + b) }
    pub fn sub(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a - b) }
    pub fn mul_elem(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a * b) }
    pub fn div_elem(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a / b) }
    pub fn pow_elem(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a.powf(b)) }
    pub fn max_elem(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a.max(b)) }
    pub fn min_elem(&self, other: &Tensor) -> Self { self.elementwise_binary(other, |a, b| a.min(b)) }

    pub fn add_scalar(&self, s: f64) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x + s).collect(), &self.shape)
    }
    pub fn sub_scalar(&self, s: f64) -> Self { self.add_scalar(-s) }
    pub fn mul_scalar(&self, s: f64) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x * s).collect(), &self.shape)
    }
    pub fn div_scalar(&self, s: f64) -> Self { self.mul_scalar(1.0 / s) }
    pub fn neg(&self) -> Self { self.mul_scalar(-1.0) }
    pub fn abs(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.abs()).collect(), &self.shape)
    }
    pub fn sqrt(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.sqrt()).collect(), &self.shape)
    }
    pub fn exp(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.exp()).collect(), &self.shape)
    }
    pub fn ln(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.ln()).collect(), &self.shape)
    }
    pub fn tanh_elem(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.tanh()).collect(), &self.shape)
    }
    pub fn sigmoid(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(), &self.shape)
    }
    pub fn relu(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.max(0.0)).collect(), &self.shape)
    }
    pub fn gelu(&self) -> Self {
        let k = (2.0 / std::f64::consts::PI).sqrt();
        Self::from_vec(self.data.iter().map(|&x| {
            0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
        }).collect(), &self.shape)
    }
    pub fn swish(&self) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x / (1.0 + (-x).exp())).collect(), &self.shape)
    }
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        let mut out = self.clone();
        let outer: usize = self.shape[..axis].iter().product();
        let dim = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        for o in 0..outer {
            for i in 0..inner {
                let mut max_v = f64::NEG_INFINITY;
                for d in 0..dim {
                    let flat = o * dim * inner + d * inner + i;
                    max_v = max_v.max(self.data[flat]);
                }
                let mut sum = 0.0;
                for d in 0..dim {
                    let flat = o * dim * inner + d * inner + i;
                    let e = (self.data[flat] - max_v).exp();
                    out.data[flat] = e;
                    sum += e;
                }
                for d in 0..dim {
                    let flat = o * dim * inner + d * inner + i;
                    out.data[flat] /= sum;
                }
            }
        }
        out
    }

    pub fn log_softmax(&self, axis: usize) -> Self {
        let sm = self.softmax(axis);
        sm.ln()
    }

    pub fn sum_all(&self) -> f64 { self.data.iter().sum() }
    pub fn mean_all(&self) -> f64 { self.sum_all() / self.numel() as f64 }
    pub fn max_all(&self) -> f64 { self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    pub fn min_all(&self) -> f64 { self.data.iter().cloned().fold(f64::INFINITY, f64::min) }
    pub fn var_all(&self) -> f64 {
        let m = self.mean_all();
        self.data.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / self.numel() as f64
    }
    pub fn std_all(&self) -> f64 { self.var_all().sqrt() }

    pub fn sum_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        let mut new_shape = self.shape.clone();
        new_shape[axis] = 1;
        let n: usize = new_shape.iter().product();
        let mut out = Tensor::zeros(&new_shape);
        let outer: usize = self.shape[..axis].iter().product();
        let dim = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        for o in 0..outer {
            for i in 0..inner {
                let mut s = 0.0;
                for d in 0..dim {
                    s += self.data[o * dim * inner + d * inner + i];
                }
                out.data[o * inner + i] = s;
            }
        }
        let reduced: Vec<usize> = self.shape.iter().enumerate()
            .filter(|&(i, _)| i != axis).map(|(_, &s)| s).collect();
        if reduced.is_empty() { Tensor::scalar(out.data[0]) } else { out.reshape(&reduced) }
    }

    pub fn mean_axis(&self, axis: usize) -> Self {
        let s = self.sum_axis(axis);
        s.div_scalar(self.shape[axis] as f64)
    }

    pub fn max_axis(&self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        let outer: usize = self.shape[..axis].iter().product();
        let dim = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let reduced: Vec<usize> = self.shape.iter().enumerate()
            .filter(|&(i, _)| i != axis).map(|(_, &s)| s).collect();
        let n: usize = reduced.iter().product();
        let mut out_data = vec![f64::NEG_INFINITY; n];
        for o in 0..outer {
            for i in 0..inner {
                let mut mx = f64::NEG_INFINITY;
                for d in 0..dim {
                    mx = mx.max(self.data[o * dim * inner + d * inner + i]);
                }
                out_data[o * inner + i] = mx;
            }
        }
        if reduced.is_empty() { Tensor::scalar(out_data[0]) } else { Tensor::from_vec(out_data, &reduced) }
    }

    pub fn argmax_axis(&self, axis: usize) -> Vec<usize> {
        let outer: usize = self.shape[..axis].iter().product();
        let dim = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let n = outer * inner;
        let mut result = vec![0usize; n];
        for o in 0..outer {
            for i in 0..inner {
                let mut mx = f64::NEG_INFINITY;
                let mut mi = 0usize;
                for d in 0..dim {
                    let v = self.data[o * dim * inner + d * inner + i];
                    if v > mx { mx = v; mi = d; }
                }
                result[o * inner + i] = mi;
            }
        }
        result
    }

    pub fn clip(&self, lo: f64, hi: f64) -> Self {
        Self::from_vec(self.data.iter().map(|&x| x.max(lo).min(hi)).collect(), &self.shape)
    }

    pub fn where_cond(&self, cond: &Tensor, other: &Tensor) -> Self {
        assert_eq!(self.shape, cond.shape);
        assert_eq!(self.shape, other.shape);
        let data: Vec<f64> = self.data.iter().zip(cond.data.iter()).zip(other.data.iter())
            .map(|((&a, &c), &b)| if c > 0.0 { a } else { b }).collect();
        Self::from_vec(data, &self.shape)
    }

    /// dot product of two 1D vectors, SIMD-style unrolled
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let chunks = n / 8;
        let mut s0 = 0.0f64; let mut s1 = 0.0f64;
        let mut s2 = 0.0f64; let mut s3 = 0.0f64;
        let mut s4 = 0.0f64; let mut s5 = 0.0f64;
        let mut s6 = 0.0f64; let mut s7 = 0.0f64;
        for c in 0..chunks {
            let base = c * 8;
            s0 += a[base]     * b[base];
            s1 += a[base + 1] * b[base + 1];
            s2 += a[base + 2] * b[base + 2];
            s3 += a[base + 3] * b[base + 3];
            s4 += a[base + 4] * b[base + 4];
            s5 += a[base + 5] * b[base + 5];
            s6 += a[base + 6] * b[base + 6];
            s7 += a[base + 7] * b[base + 7];
        }
        let mut total = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
        for i in (chunks * 8)..n {
            total += a[i] * b[i];
        }
        total
    }

    /// Matrix multiply: [M,K] x [K,N] -> [M,N]
    pub fn matmul(&self, other: &Tensor) -> Self {
        assert!(self.ndim() >= 2 && other.ndim() >= 2);
        let a_shape = &self.shape;
        let b_shape = &other.shape;
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        assert_eq!(k, k2, "matmul inner dim mismatch: {} vs {}", k, k2);

        if self.ndim() == 2 && other.ndim() == 2 {
            return Self::matmul_2d(&self.data, m, k, &other.data, n);
        }
        // batched matmul
        let batch_a: Vec<usize> = a_shape[..a_shape.len() - 2].to_vec();
        let batch_b: Vec<usize> = b_shape[..b_shape.len() - 2].to_vec();
        let batch_shape = Self::broadcast_shape(&batch_a, &batch_b);
        let batch_n: usize = batch_shape.iter().product();
        let a_mat_size = m * k;
        let b_mat_size = k * n;
        let o_mat_size = m * n;
        let a_batch_n: usize = batch_a.iter().product::<usize>().max(1);
        let b_batch_n: usize = batch_b.iter().product::<usize>().max(1);
        let mut out_shape = batch_shape.clone();
        out_shape.push(m);
        out_shape.push(n);
        let mut out_data = vec![0.0f64; batch_n * o_mat_size];
        for bi in 0..batch_n {
            let ai = if a_batch_n == 1 { 0 } else { bi % a_batch_n };
            let bbi = if b_batch_n == 1 { 0 } else { bi % b_batch_n };
            let a_off = ai * a_mat_size;
            let b_off = bbi * b_mat_size;
            let o_off = bi * o_mat_size;
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for p in 0..k {
                        s += self.data[a_off + i * k + p] * other.data[b_off + p * n + j];
                    }
                    out_data[o_off + i * n + j] = s;
                }
            }
        }
        Self::from_vec(out_data, &out_shape)
    }

    fn matmul_2d(a: &[f64], m: usize, k: usize, b: &[f64], n: usize) -> Self {
        let mut out = vec![0.0f64; m * n];
        // loop tiling for cache efficiency
        const TILE: usize = 32;
        for ii in (0..m).step_by(TILE) {
            let i_end = (ii + TILE).min(m);
            for jj in (0..n).step_by(TILE) {
                let j_end = (jj + TILE).min(n);
                for pp in (0..k).step_by(TILE) {
                    let p_end = (pp + TILE).min(k);
                    for i in ii..i_end {
                        for p in pp..p_end {
                            let a_val = a[i * k + p];
                            for j in jj..j_end {
                                out[i * n + j] += a_val * b[p * n + j];
                            }
                        }
                    }
                }
            }
        }
        Self::from_vec(out, &[m, n])
    }

    /// Batched matrix multiply: [...,M,K] x [...,K,N] -> [...,M,N]
    pub fn bmm(&self, other: &Tensor) -> Self { self.matmul(other) }

    /// Vector-matrix: [K] x [K,N] -> [N]
    pub fn vecmat(&self, mat: &Tensor) -> Self {
        assert!(self.ndim() == 1 && mat.ndim() == 2);
        let k = self.shape[0];
        assert_eq!(k, mat.shape[0]);
        let n = mat.shape[1];
        let mut out = vec![0.0; n];
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += self.data[p] * mat.data[p * n + j];
            }
            out[j] = s;
        }
        Self::from_vec(out, &[n])
    }

    /// Matrix-vector: [M,K] x [K] -> [M]
    pub fn matvec(&self, v: &Tensor) -> Self {
        assert!(self.ndim() == 2 && v.ndim() == 1);
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(k, v.shape[0]);
        let mut out = vec![0.0; m];
        for i in 0..m {
            out[i] = Self::dot(&self.data[i * k..(i + 1) * k], &v.data);
        }
        Self::from_vec(out, &[m])
    }

    /// Outer product: [M] x [N] -> [M,N]
    pub fn outer(&self, other: &Tensor) -> Self {
        assert!(self.ndim() == 1 && other.ndim() == 1);
        let m = self.shape[0];
        let n = other.shape[0];
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = self.data[i] * other.data[j];
            }
        }
        Self::from_vec(data, &[m, n])
    }

    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = 1.0; }
        Self::from_vec(data, &[n, n])
    }

    pub fn diag(v: &[f64]) -> Self {
        let n = v.len();
        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = v[i]; }
        Self::from_vec(data, &[n, n])
    }

    pub fn trace(&self) -> f64 {
        assert!(self.ndim() == 2 && self.shape[0] == self.shape[1]);
        let n = self.shape[0];
        (0..n).map(|i| self.data[i * n + i]).sum()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn l1_norm(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).sum()
    }

    pub fn linf_norm(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
    }

    /// Layer normalization along last axis
    pub fn layer_norm(&self, eps: f64) -> Self {
        assert!(self.ndim() >= 1);
        let last = *self.shape.last().unwrap();
        let batch: usize = self.numel() / last;
        let mut out = self.clone();
        for b in 0..batch {
            let off = b * last;
            let sl = &self.data[off..off + last];
            let mean: f64 = sl.iter().sum::<f64>() / last as f64;
            let var: f64 = sl.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / last as f64;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..last {
                out.data[off + i] = (sl[i] - mean) * inv_std;
            }
        }
        out
    }

    /// Batch normalization: input [N, C, ...], given running mean/var of shape [C]
    pub fn batch_norm(&self, mean: &[f64], var: &[f64], gamma: &[f64], beta: &[f64], eps: f64) -> Self {
        assert!(self.ndim() >= 2);
        let c = self.shape[1];
        assert_eq!(mean.len(), c);
        let n = self.shape[0];
        let spatial: usize = self.shape[2..].iter().product();
        let mut out = self.clone();
        for ch in 0..c {
            let inv_std = 1.0 / (var[ch] + eps).sqrt();
            let g = gamma[ch];
            let b = beta[ch];
            let m = mean[ch];
            for ni in 0..n {
                for s in 0..spatial {
                    let idx = ni * c * spatial + ch * spatial + s;
                    out.data[idx] = g * (self.data[idx] - m) * inv_std + b;
                }
            }
        }
        out
    }

    /// Repeat tensor along axis
    pub fn repeat(&self, axis: usize, times: usize) -> Self {
        let refs: Vec<&Tensor> = (0..times).map(|_| self).collect();
        Self::concat(&refs, axis)
    }

    /// Apply a function elementwise
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Self {
        Self::from_vec(self.data.iter().map(|&x| f(x)).collect(), &self.shape)
    }

    /// Zip two tensors
    pub fn zip_with<F: Fn(f64, f64) -> f64>(&self, other: &Tensor, f: F) -> Self {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| f(a, b)).collect();
        Self::from_vec(data, &self.shape)
    }

    pub fn gather(&self, axis: usize, indices: &[usize]) -> Self {
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let idx_len = indices.len();
        let mut new_shape = self.shape.clone();
        new_shape[axis] = idx_len;
        let mut out = Tensor::zeros(&new_shape);
        let dim = self.shape[axis];
        for o in 0..outer {
            for (di, &idx) in indices.iter().enumerate() {
                for i in 0..inner {
                    let src = o * dim * inner + idx * inner + i;
                    let dst = o * idx_len * inner + di * inner + i;
                    out.data[dst] = self.data[src];
                }
            }
        }
        out
    }

    pub fn scatter(&mut self, axis: usize, indices: &[usize], src: &Tensor) {
        let outer: usize = self.shape[..axis].iter().product();
        let inner: usize = self.shape[axis + 1..].iter().product();
        let dim = self.shape[axis];
        let idx_len = indices.len();
        for o in 0..outer {
            for (di, &idx) in indices.iter().enumerate() {
                for i in 0..inner {
                    let s = o * idx_len * inner + di * inner + i;
                    let d = o * dim * inner + idx * inner + i;
                    self.data[d] = src.data[s];
                }
            }
        }
    }

    pub fn arange(start: f64, end: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut v = start;
        while v < end { data.push(v); v += step; }
        let n = data.len();
        Self::from_vec(data, &[n])
    }

    pub fn linspace(start: f64, end: f64, n: usize) -> Self {
        if n <= 1 { return Self::from_vec(vec![start], &[1]); }
        let step = (end - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Self::from_vec(data, &[n])
    }

    /// Cumulative sum along axis
    pub fn cumsum(&self, axis: usize) -> Self {
        let outer: usize = self.shape[..axis].iter().product();
        let dim = self.shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        let mut out = self.clone();
        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0.0;
                for d in 0..dim {
                    let idx = o * dim * inner + d * inner + i;
                    acc += self.data[idx];
                    out.data[idx] = acc;
                }
            }
        }
        out
    }

    /// Tril: zero above diagonal
    pub fn tril(&self, diag_offset: i64) -> Self {
        assert_eq!(self.ndim(), 2);
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = self.clone();
        for i in 0..r {
            for j in 0..c {
                if j as i64 > i as i64 + diag_offset {
                    out.data[i * c + j] = 0.0;
                }
            }
        }
        out
    }

    pub fn triu(&self, diag_offset: i64) -> Self {
        assert_eq!(self.ndim(), 2);
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = self.clone();
        for i in 0..r {
            for j in 0..c {
                if (j as i64) < i as i64 + diag_offset {
                    out.data[i * c + j] = 0.0;
                }
            }
        }
        out
    }

    pub fn conv1d(&self, kernel: &Tensor, stride: usize) -> Self {
        assert_eq!(self.ndim(), 1);
        assert_eq!(kernel.ndim(), 1);
        let n = self.shape[0];
        let k = kernel.shape[0];
        let out_len = (n - k) / stride + 1;
        let mut out = vec![0.0; out_len];
        for i in 0..out_len {
            let start = i * stride;
            out[i] = Self::dot(&self.data[start..start + k], &kernel.data);
        }
        Self::from_vec(out, &[out_len])
    }

    pub fn pad(&self, axis: usize, before: usize, after: usize, val: f64) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape[axis] += before + after;
        let mut out = Tensor::full(&new_shape, val);
        let outer: usize = self.shape[..axis].iter().product();
        let dim = self.shape[axis];
        let new_dim = new_shape[axis];
        let inner: usize = self.shape[axis + 1..].iter().product();
        for o in 0..outer {
            for d in 0..dim {
                for i in 0..inner {
                    let src = o * dim * inner + d * inner + i;
                    let dst = o * new_dim * inner + (d + before) * inner + i;
                    out.data[dst] = self.data[src];
                }
            }
        }
        out
    }

    pub fn allclose(&self, other: &Tensor, atol: f64) -> bool {
        if self.shape != other.shape { return false; }
        self.data.iter().zip(other.data.iter()).all(|(&a, &b)| (a - b).abs() < atol)
    }

    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|x| x.is_finite())
    }

    /// Simple LU decomposition (no pivoting) for square matrices
    pub fn lu_solve(&self, b: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2);
        let n = self.shape[0];
        assert_eq!(self.shape[1], n);
        assert_eq!(b.shape, vec![n]);
        // forward: LU in place
        let mut a = self.data.clone();
        let mut x = b.data.clone();
        // Gaussian elimination
        for col in 0..n {
            let pivot = a[col * n + col];
            assert!(pivot.abs() > 1e-15, "singular matrix");
            for row in (col + 1)..n {
                let factor = a[row * n + col] / pivot;
                a[row * n + col] = factor;
                for j in (col + 1)..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
            }
        }
        // forward substitution
        for i in 1..n {
            for j in 0..i {
                x[i] -= a[i * n + j] * x[j];
            }
        }
        // back substitution
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= a[i * n + j] * x[j];
            }
            x[i] /= a[i * n + i];
        }
        Tensor::from_vec(x, &[n])
    }

    /// Cholesky decomposition for SPD matrices
    pub fn cholesky(&self) -> Tensor {
        assert_eq!(self.ndim(), 2);
        let n = self.shape[0];
        assert_eq!(self.shape[1], n);
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for k in 0..j { s += l[i * n + k] * l[j * n + k]; }
                if i == j {
                    l[i * n + j] = (self.data[i * n + i] - s).sqrt();
                } else {
                    l[i * n + j] = (self.data[i * n + j] - s) / l[j * n + j];
                }
            }
        }
        Tensor::from_vec(l, &[n, n])
    }

    /// Determinant via LU
    pub fn det(&self) -> f64 {
        assert_eq!(self.ndim(), 2);
        let n = self.shape[0];
        assert_eq!(self.shape[1], n);
        let mut a = self.data.clone();
        let mut sign = 1.0;
        for col in 0..n {
            // partial pivot
            let mut max_row = col;
            let mut max_val = a[col * n + col].abs();
            for row in (col + 1)..n {
                let v = a[row * n + col].abs();
                if v > max_val { max_val = v; max_row = row; }
            }
            if max_row != col {
                for j in 0..n { a.swap(col * n + j, max_row * n + j); }
                sign *= -1.0;
            }
            if a[col * n + col].abs() < 1e-15 { return 0.0; }
            for row in (col + 1)..n {
                let factor = a[row * n + col] / a[col * n + col];
                for j in (col + 1)..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
            }
        }
        let mut d = sign;
        for i in 0..n { d *= a[i * n + i]; }
        d
    }

    /// Matrix inverse via augmented row reduction
    pub fn inverse(&self) -> Tensor {
        assert_eq!(self.ndim(), 2);
        let n = self.shape[0];
        assert_eq!(self.shape[1], n);
        let mut aug = vec![0.0; n * 2 * n];
        for i in 0..n {
            for j in 0..n { aug[i * 2 * n + j] = self.data[i * n + j]; }
            aug[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n {
            let mut max_row = col;
            for row in (col + 1)..n {
                if aug[row * 2 * n + col].abs() > aug[max_row * 2 * n + col].abs() { max_row = row; }
            }
            if max_row != col {
                for j in 0..2 * n { aug.swap(col * 2 * n + j, max_row * 2 * n + j); }
            }
            let pivot = aug[col * 2 * n + col];
            assert!(pivot.abs() > 1e-15, "singular");
            for j in 0..2 * n { aug[col * 2 * n + j] /= pivot; }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row * 2 * n + col];
                for j in 0..2 * n { aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j]; }
            }
        }
        let mut inv = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n { inv[i * n + j] = aug[i * 2 * n + n + j]; }
        }
        Tensor::from_vec(inv, &[n, n])
    }

    /// QR decomposition (Gram-Schmidt)
    pub fn qr(&self) -> (Tensor, Tensor) {
        assert_eq!(self.ndim(), 2);
        let (m, n) = (self.shape[0], self.shape[1]);
        let k = m.min(n);
        let mut q_data = vec![0.0; m * k];
        let mut r_data = vec![0.0; k * n];
        for j in 0..k {
            // copy column j
            let mut v: Vec<f64> = (0..m).map(|i| self.data[i * n + j]).collect();
            for i in 0..j {
                let dot: f64 = (0..m).map(|r| q_data[r * k + i] * v[r]).sum();
                r_data[i * n + j] = dot;
                for r in 0..m { v[r] -= dot * q_data[r * k + i]; }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r_data[j * n + j] = norm;
            if norm > 1e-15 {
                for r in 0..m { q_data[r * k + j] = v[r] / norm; }
            }
        }
        (Tensor::from_vec(q_data, &[m, k]), Tensor::from_vec(r_data, &[k, n]))
    }

    /// Eigenvalues of symmetric matrix via QR iteration
    pub fn eig_symmetric(&self, max_iter: usize) -> Vec<f64> {
        assert_eq!(self.ndim(), 2);
        let n = self.shape[0];
        assert_eq!(n, self.shape[1]);
        let mut a = self.clone();
        for _ in 0..max_iter {
            let (q, r) = a.qr();
            a = r.matmul(&q);
        }
        (0..n).map(|i| a.data[i * n + i]).collect()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        let limit = 8.min(self.data.len());
        for i in 0..limit {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{:.4}", self.data[i])?;
        }
        if self.data.len() > limit { write!(f, ", ...")?; }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0] - 22.0).abs() < 1e-10);
    }
    #[test]
    fn test_broadcast() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3, 1]);
        let b = Tensor::from_vec(vec![10.0, 20.0], &[1, 2]);
        let c = a.add(&b);
        assert_eq!(c.shape, vec![3, 2]);
        assert!((c.get(&[0, 0]) - 11.0).abs() < 1e-10);
    }
    #[test]
    fn test_transpose() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert!((t.get(&[1, 0]) - 2.0).abs() < 1e-10);
    }
    #[test]
    fn test_softmax() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
        let s = a.softmax(1);
        let total: f64 = s.data.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_det() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!((a.det() - (-2.0)).abs() < 1e-10);
    }
    #[test]
    fn test_inverse() {
        let a = Tensor::from_vec(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]);
        let inv = a.inverse();
        let prod = a.matmul(&inv);
        assert!((prod.data[0] - 1.0).abs() < 1e-10);
        assert!((prod.data[3] - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_cholesky() {
        let a = Tensor::from_vec(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let l = a.cholesky();
        let lt = l.transpose();
        let prod = l.matmul(&lt);
        assert!(a.allclose(&prod, 1e-10));
    }
}
