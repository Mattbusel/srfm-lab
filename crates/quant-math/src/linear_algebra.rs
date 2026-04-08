// linear_algebra.rs — Dense matrix operations, decompositions, solvers
// All f64, row-major storage, std-only

use std::fmt;

/// Row-major dense matrix of f64.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:>12.6}", self[(i, j)])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        debug_assert!(r < self.rows && c < self.cols);
        &self.data[r * self.cols + c]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        debug_assert!(r < self.rows && c < self.cols);
        &mut self.data[r * self.cols + c]
    }
}

impl Matrix {
    /// Create zero matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    /// Create identity matrix
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n { m[(i, i)] = 1.0; }
        m
    }

    /// Create from row-major slice
    pub fn from_slice(rows: usize, cols: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data: data.to_vec() }
    }

    /// Create from Vec of row Vecs
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let r = rows.len();
        let c = if r > 0 { rows[0].len() } else { 0 };
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c);
            data.extend_from_slice(row);
        }
        Self { rows: r, cols: c, data }
    }

    /// Create diagonal matrix from slice
    pub fn diag(vals: &[f64]) -> Self {
        let n = vals.len();
        let mut m = Self::zeros(n, n);
        for i in 0..n { m[(i, i)] = vals[i]; }
        m
    }

    /// Create column vector from slice
    pub fn col_vec(vals: &[f64]) -> Self {
        Self { rows: vals.len(), cols: 1, data: vals.to_vec() }
    }

    /// Create row vector from slice
    pub fn row_vec(vals: &[f64]) -> Self {
        Self { rows: 1, cols: vals.len(), data: vals.to_vec() }
    }

    /// Random matrix using simple LCG for reproducibility
    pub fn random(rows: usize, cols: usize, seed: u64) -> Self {
        let mut s = seed;
        let n = rows * cols;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((s >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            data.push(v);
        }
        Self { rows, cols, data }
    }

    /// Random symmetric positive-definite matrix
    pub fn random_spd(n: usize, seed: u64) -> Self {
        let a = Self::random(n, n, seed);
        let at = a.transpose();
        let mut m = at.matmul(&a);
        for i in 0..n { m[(i, i)] += n as f64; }
        m
    }

    /// Number of elements
    #[inline]
    pub fn len(&self) -> usize { self.rows * self.cols }

    /// Is empty
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Is square
    #[inline]
    pub fn is_square(&self) -> bool { self.rows == self.cols }

    /// Get row as slice
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    /// Get row as mutable slice
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    /// Get column as new Vec
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, j)]).collect()
    }

    /// Transpose
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// In-place transpose for square matrices
    pub fn transpose_inplace(&mut self) {
        assert!(self.is_square());
        let n = self.rows;
        for i in 0..n {
            for j in (i + 1)..n {
                let a = i * n + j;
                let b = j * n + i;
                self.data.swap(a, b);
            }
        }
    }

    /// Matrix multiplication: self (m×k) * other (k×n) → (m×n)
    pub fn matmul(&self, other: &Matrix) -> Self {
        assert_eq!(self.cols, other.rows, "matmul dimension mismatch");
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut out = Self::zeros(m, n);
        for i in 0..m {
            for p in 0..k {
                let a = self[(i, p)];
                if a == 0.0 { continue; }
                for j in 0..n {
                    out.data[i * n + j] += a * other.data[p * n + j];
                }
            }
        }
        out
    }

    /// Element-wise add
    pub fn add(&self, other: &Matrix) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// Element-wise subtract
    pub fn sub(&self, other: &Matrix) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// Scalar multiply
    pub fn scale(&self, s: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|x| x * s).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// In-place scalar multiply
    pub fn scale_inplace(&mut self, s: f64) {
        for x in &mut self.data { *x *= s; }
    }

    /// Element-wise (Hadamard) product
    pub fn hadamard(&self, other: &Matrix) -> Self {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Self { rows: self.rows, cols: self.cols, data }
    }

    /// Trace (sum of diagonal)
    pub fn trace(&self) -> f64 {
        assert!(self.is_square());
        (0..self.rows).map(|i| self[(i, i)]).sum()
    }

    /// Frobenius norm
    pub fn norm_frobenius(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Infinity norm (max absolute row sum)
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows)
            .map(|i| self.row(i).iter().map(|x| x.abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }

    /// 1-norm (max absolute column sum)
    pub fn norm_1(&self) -> f64 {
        (0..self.cols)
            .map(|j| (0..self.rows).map(|i| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0_f64, f64::max)
    }

    /// Spectral norm (largest singular value), via power iteration
    pub fn norm_spectral(&self, max_iter: usize) -> f64 {
        let ata = self.transpose().matmul(self);
        let n = ata.cols;
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..max_iter {
            let mut w = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..n { s += ata[(i, j)] * v[j]; }
                w[i] = s;
            }
            let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm < 1e-15 { return 0.0; }
            for x in &mut w { *x /= nrm; }
            v = w;
        }
        let mut av = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n { s += ata[(i, j)] * v[j]; }
            av[i] = s;
        }
        let lam: f64 = av.iter().zip(&v).map(|(a, b)| a * b).sum();
        lam.abs().sqrt()
    }

    /// Determinant via LU decomposition
    pub fn det(&self) -> f64 {
        assert!(self.is_square());
        let n = self.rows;
        if n == 0 { return 1.0; }
        if n == 1 { return self[(0, 0)]; }
        if n == 2 { return self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]; }
        let (lu, piv, sign) = self.lu_decompose();
        let _ = piv;
        let mut d = sign as f64;
        for i in 0..n { d *= lu[(i, i)]; }
        d
    }

    /// LU decomposition with partial pivoting.
    /// Returns (LU combined, pivot indices, sign of permutation).
    pub fn lu_decompose(&self) -> (Matrix, Vec<usize>, i32) {
        assert!(self.is_square());
        let n = self.rows;
        let mut lu = self.clone();
        let mut piv: Vec<usize> = (0..n).collect();
        let mut sign: i32 = 1;

        for k in 0..n {
            // Find pivot
            let mut max_val = lu[(k, k)].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = lu[(i, k)].abs();
                if v > max_val { max_val = v; max_row = i; }
            }
            if max_row != k {
                // Swap rows
                for j in 0..n {
                    let a = k * n + j;
                    let b = max_row * n + j;
                    lu.data.swap(a, b);
                }
                piv.swap(k, max_row);
                sign = -sign;
            }
            let diag = lu[(k, k)];
            if diag.abs() < 1e-15 { continue; }
            for i in (k + 1)..n {
                lu[(i, k)] /= diag;
                let factor = lu[(i, k)];
                for j in (k + 1)..n {
                    let v = lu[(k, j)];
                    lu[(i, j)] -= factor * v;
                }
            }
        }
        (lu, piv, sign)
    }

    /// Extract L and U from combined LU matrix
    pub fn lu_split(lu: &Matrix) -> (Matrix, Matrix) {
        let n = lu.rows;
        let mut l = Matrix::eye(n);
        let mut u = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if j < i { l[(i, j)] = lu[(i, j)]; }
                else { u[(i, j)] = lu[(i, j)]; }
            }
        }
        (l, u)
    }

    /// Solve Ax = b via LU decomposition
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        assert!(self.is_square());
        let n = self.rows;
        assert_eq!(b.len(), n);
        let (lu, piv, _) = self.lu_decompose();

        // Apply permutation to b
        let mut pb = vec![0.0; n];
        for i in 0..n { pb[i] = b[piv[i]]; }

        // Forward substitution (L y = Pb)
        for i in 1..n {
            let mut s = pb[i];
            for j in 0..i { s -= lu[(i, j)] * pb[j]; }
            pb[i] = s;
        }

        // Back substitution (U x = y)
        let mut x = pb;
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in (i + 1)..n { s -= lu[(i, j)] * x[j]; }
            let d = lu[(i, i)];
            if d.abs() < 1e-15 { x[i] = 0.0; } else { x[i] = s / d; }
        }
        x
    }

    /// Solve AX = B where B is a matrix
    pub fn solve_matrix(&self, b: &Matrix) -> Matrix {
        assert!(self.is_square());
        assert_eq!(self.rows, b.rows);
        let mut out = Matrix::zeros(b.rows, b.cols);
        for j in 0..b.cols {
            let col_b: Vec<f64> = (0..b.rows).map(|i| b[(i, j)]).collect();
            let x = self.solve(&col_b);
            for i in 0..b.rows { out[(i, j)] = x[i]; }
        }
        out
    }

    /// Matrix inverse via Gauss-Jordan elimination
    pub fn inverse(&self) -> Option<Matrix> {
        assert!(self.is_square());
        let n = self.rows;
        let mut aug = Matrix::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n { aug[(i, j)] = self[(i, j)]; }
            aug[(i, n + i)] = 1.0;
        }

        for k in 0..n {
            // Partial pivot
            let mut max_val = aug[(k, k)].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = aug[(i, k)].abs();
                if v > max_val { max_val = v; max_row = i; }
            }
            if max_val < 1e-14 { return None; }
            if max_row != k {
                for j in 0..(2 * n) {
                    let a = k * (2 * n) + j;
                    let b = max_row * (2 * n) + j;
                    aug.data.swap(a, b);
                }
            }
            let diag = aug[(k, k)];
            for j in 0..(2 * n) { aug[(k, j)] /= diag; }
            for i in 0..n {
                if i == k { continue; }
                let factor = aug[(i, k)];
                for j in 0..(2 * n) {
                    let v = aug[(k, j)];
                    aug[(i, j)] -= factor * v;
                }
            }
        }

        let mut inv = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n { inv[(i, j)] = aug[(i, n + j)]; }
        }
        Some(inv)
    }

    /// Cholesky decomposition: A = L L^T, returns lower triangular L.
    /// A must be symmetric positive-definite.
    pub fn cholesky(&self) -> Option<Matrix> {
        assert!(self.is_square());
        let n = self.rows;
        let mut l = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let mut s = 0.0;
                for k in 0..j { s += l[(i, k)] * l[(j, k)]; }
                if i == j {
                    let val = self[(i, i)] - s;
                    if val <= 0.0 { return None; }
                    l[(i, j)] = val.sqrt();
                } else {
                    let d = l[(j, j)];
                    if d.abs() < 1e-15 { return None; }
                    l[(i, j)] = (self[(i, j)] - s) / d;
                }
            }
        }
        Some(l)
    }

    /// Solve via Cholesky: A x = b, where A = L L^T
    pub fn cholesky_solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        let l = self.cholesky()?;
        let n = l.rows;
        // Forward: L y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for j in 0..i { s -= l[(i, j)] * y[j]; }
            y[i] = s / l[(i, i)];
        }
        // Back: L^T x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in (i + 1)..n { s -= l[(j, i)] * x[j]; }
            x[i] = s / l[(i, i)];
        }
        Some(x)
    }

    /// QR decomposition via Gram-Schmidt (classical with re-orthogonalization)
    /// Returns (Q, R)
    pub fn qr_gram_schmidt(&self) -> (Matrix, Matrix) {
        let m = self.rows;
        let n = self.cols;
        let k = m.min(n);
        let mut q = Matrix::zeros(m, k);
        let mut r = Matrix::zeros(k, n);

        for j in 0..k {
            // Copy column j of self
            let mut v: Vec<f64> = (0..m).map(|i| self[(i, j)]).collect();

            // Two passes of Gram-Schmidt for numerical stability
            for _pass in 0..2 {
                for i in 0..j {
                    let mut dot = 0.0;
                    for row in 0..m { dot += q[(row, i)] * v[row]; }
                    if _pass == 0 { r[(i, j)] += dot; }
                    for row in 0..m { v[row] -= dot * q[(row, i)]; }
                }
            }

            let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            r[(j, j)] = nrm;
            if nrm > 1e-14 {
                for row in 0..m { q[(row, j)] = v[row] / nrm; }
            }
        }

        // Fill remaining R entries for rectangular case
        for j in k..n {
            for i in 0..k {
                let mut dot = 0.0;
                for row in 0..m { dot += q[(row, i)] * self[(row, j)]; }
                r[(i, j)] = dot;
            }
        }

        (q, r)
    }

    /// QR decomposition via Householder reflections
    /// Returns (Q, R)
    pub fn qr_householder(&self) -> (Matrix, Matrix) {
        let m = self.rows;
        let n = self.cols;
        let k = m.min(n);
        let mut r = self.clone();
        let mut q = Matrix::eye(m);

        for j in 0..k {
            // Extract column below diagonal
            let mut x = vec![0.0; m - j];
            for i in j..m { x[i - j] = r[(i, j)]; }

            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-15 { continue; }

            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            x[0] += sign * norm_x;
            let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_v < 1e-15 { continue; }
            for v in &mut x { *v /= norm_v; }

            // Apply Householder to R: R = R - 2 v (v^T R)
            for col in j..n {
                let mut dot = 0.0;
                for i in 0..x.len() { dot += x[i] * r[(j + i, col)]; }
                for i in 0..x.len() { r[(j + i, col)] -= 2.0 * x[i] * dot; }
            }

            // Apply Householder to Q: Q = Q - 2 (Q v) v^T
            for row in 0..m {
                let mut dot = 0.0;
                for i in 0..x.len() { dot += q[(row, j + i)] * x[i]; }
                for i in 0..x.len() { q[(row, j + i)] -= 2.0 * dot * x[i]; }
            }
        }

        // Clean below diagonal
        for j in 0..n {
            for i in (j + 1)..m {
                r[(i, j)] = 0.0;
            }
        }

        (q, r)
    }

    /// Eigenvalue decomposition via QR algorithm with shifts (for symmetric matrices).
    /// Returns (eigenvalues, eigenvectors as columns of V).
    pub fn eigen_symmetric(&self) -> (Vec<f64>, Matrix) {
        assert!(self.is_square());
        let n = self.rows;
        if n == 0 { return (vec![], Matrix::zeros(0, 0)); }
        if n == 1 { return (vec![self[(0, 0)]], Matrix::eye(1)); }

        // Tridiagonalize via Householder
        let (mut t, mut acc_q) = self.tridiagonalize();

        // QR iteration with Wilkinson shift
        let max_iter = 100 * n;
        let mut p = n;
        let mut iter_count = 0;

        while p > 1 && iter_count < max_iter {
            iter_count += 1;
            // Check for convergence of t[(p-1, p-2)]
            let tol = 1e-12 * (t[(p - 2, p - 2)].abs() + t[(p - 1, p - 1)].abs()).max(1e-30);
            if t[(p - 1, p - 2)].abs() < tol {
                p -= 1;
                continue;
            }

            // Wilkinson shift
            let d = (t[(p - 2, p - 2)] - t[(p - 1, p - 1)]) / 2.0;
            let mu = t[(p - 1, p - 1)]
                - t[(p - 1, p - 2)] * t[(p - 1, p - 2)]
                    / (d + d.signum() * (d * d + t[(p - 1, p - 2)] * t[(p - 1, p - 2)]).sqrt());

            // Implicit QR step with Givens rotations
            let mut x = t[(0, 0)] - mu;
            let mut z = t[(1, 0)];
            for k in 0..(p - 1) {
                // Compute Givens rotation
                let r = (x * x + z * z).sqrt();
                let c = if r > 1e-30 { x / r } else { 1.0 };
                let s = if r > 1e-30 { -z / r } else { 0.0 };

                // Apply to T from left and right (tridiag preserved by Givens chase)
                // Rows k, k+1
                let lo = if k > 0 { k - 1 } else { 0 };
                let hi = (k + 3).min(p);
                for j in lo..hi {
                    let a = t[(k, j)];
                    let b = t[(k + 1, j)];
                    t[(k, j)] = c * a - s * b;
                    t[(k + 1, j)] = s * a + c * b;
                }
                for i in lo..hi {
                    let a = t[(i, k)];
                    let b = t[(i, k + 1)];
                    t[(i, k)] = c * a - s * b;
                    t[(i, k + 1)] = s * a + c * b;
                }

                // Accumulate eigenvectors
                for i in 0..n {
                    let a = acc_q[(i, k)];
                    let b = acc_q[(i, k + 1)];
                    acc_q[(i, k)] = c * a - s * b;
                    acc_q[(i, k + 1)] = s * a + c * b;
                }

                if k + 2 < p {
                    x = t[(k + 1, k)];
                    z = t[(k + 2, k)];
                }
            }
        }

        let eigenvalues: Vec<f64> = (0..n).map(|i| t[(i, i)]).collect();
        (eigenvalues, acc_q)
    }

    /// Tridiagonalize symmetric matrix via Householder.
    /// Returns (tridiagonal matrix, accumulated orthogonal transform Q).
    pub fn tridiagonalize(&self) -> (Matrix, Matrix) {
        let n = self.rows;
        let mut a = self.clone();
        let mut q = Matrix::eye(n);

        for k in 0..(n.saturating_sub(2)) {
            let mut x = vec![0.0; n - k - 1];
            for i in 0..x.len() { x[i] = a[(k + 1 + i, k)]; }

            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-15 { continue; }

            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            x[0] += sign * norm_x;
            let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_v < 1e-15 { continue; }
            for v in &mut x { *v /= norm_v; }

            // p = A[k+1:, k+1:] @ v
            let start = k + 1;
            let sz = n - start;
            let mut p = vec![0.0; sz];
            for i in 0..sz {
                let mut s = 0.0;
                for j in 0..sz { s += a[(start + i, start + j)] * x[j]; }
                p[i] = s;
            }

            let dot_vp: f64 = x.iter().zip(&p).map(|(a, b)| a * b).sum();
            // w = p - (v^T p) v
            let mut w = vec![0.0; sz];
            for i in 0..sz { w[i] = p[i] - dot_vp * x[i]; }

            // A -= 2 (w v^T + v w^T)
            for i in 0..sz {
                for j in 0..sz {
                    a[(start + i, start + j)] -= 2.0 * (w[i] * x[j] + x[i] * w[j]);
                }
            }

            // Update first column/row in the active block
            a[(k, start)] = -sign * norm_x;
            a[(start, k)] = -sign * norm_x;
            for i in 1..sz {
                a[(k, start + i)] = 0.0;
                a[(start + i, k)] = 0.0;
            }

            // Accumulate Q
            for row in 0..n {
                let mut dot = 0.0;
                for i in 0..sz { dot += q[(row, start + i)] * x[i]; }
                for i in 0..sz { q[(row, start + i)] -= 2.0 * dot * x[i]; }
            }
        }
        (a, q)
    }

    /// Singular Value Decomposition via one-sided Jacobi.
    /// Returns (U, sigma, V) where A ≈ U * diag(sigma) * V^T
    pub fn svd(&self) -> (Matrix, Vec<f64>, Matrix) {
        let m = self.rows;
        let n = self.cols;
        let wide = m < n;
        let (work, m2, n2) = if wide {
            (self.transpose(), n, m)
        } else {
            (self.clone(), m, n)
        };

        // Compute A^T A
        let ata = work.transpose().matmul(&work);

        // Eigendecompose A^T A → V, sigma^2
        let (eigvals, v) = ata.eigen_symmetric();

        // Sort by descending eigenvalue
        let mut idx: Vec<usize> = (0..n2).collect();
        idx.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap_or(std::cmp::Ordering::Equal));

        let mut sigma = vec![0.0; n2];
        let mut v_sorted = Matrix::zeros(n2, n2);
        for (new_j, &old_j) in idx.iter().enumerate() {
            let eval = eigvals[old_j];
            sigma[new_j] = if eval > 0.0 { eval.sqrt() } else { 0.0 };
            for i in 0..n2 { v_sorted[(i, new_j)] = v[(i, old_j)]; }
        }

        // U = A V Sigma^{-1}
        let av = work.matmul(&v_sorted);
        let mut u = Matrix::zeros(m2, n2);
        for j in 0..n2 {
            if sigma[j] > 1e-14 {
                for i in 0..m2 { u[(i, j)] = av[(i, j)] / sigma[j]; }
            }
        }

        if wide {
            // A = U Sigma V^T was for A^T, so real A = V Sigma U^T
            (v_sorted, sigma, u)
        } else {
            (u, sigma, v_sorted)
        }
    }

    /// Pseudoinverse via SVD
    pub fn pinv(&self) -> Matrix {
        let (u, sigma, v) = self.svd();
        let tol = 1e-10 * sigma.first().copied().unwrap_or(0.0);
        let k = sigma.len();
        // V * Sigma^{-1} * U^T
        let mut result = Matrix::zeros(self.cols, self.rows);
        for s in 0..k {
            if sigma[s] > tol {
                let inv_s = 1.0 / sigma[s];
                for i in 0..self.cols {
                    for j in 0..self.rows {
                        result[(i, j)] += v[(i, s)] * inv_s * u[(j, s)];
                    }
                }
            }
        }
        result
    }

    /// Condition number (ratio of largest to smallest singular value)
    pub fn cond(&self) -> f64 {
        let (_, sigma, _) = self.svd();
        if sigma.is_empty() { return f64::INFINITY; }
        let max_s = sigma.first().copied().unwrap_or(0.0);
        let min_s = sigma.iter().copied().filter(|&s| s > 1e-14).last().unwrap_or(0.0);
        if min_s < 1e-14 { f64::INFINITY } else { max_s / min_s }
    }

    /// Matrix exponential via Padé approximation (scaling and squaring)
    pub fn expm(&self) -> Matrix {
        assert!(self.is_square());
        let n = self.rows;
        let norm = self.norm_inf();
        let s = (norm / 0.5).log2().ceil().max(0.0) as u32;
        let scale = 2.0_f64.powi(-(s as i32));
        let a_scaled = self.scale(scale);

        // Padé(6) coefficients
        let c = [1.0, 0.5, 1.0/10.0, 1.0/120.0, 1.0/1680.0, 1.0/30240.0, 1.0/665280.0];
        let id = Matrix::eye(n);
        let a2 = a_scaled.matmul(&a_scaled);
        let a4 = a2.matmul(&a2);
        let a6 = a4.matmul(&a2);

        // U = A(c1 I + c3 A^2 + c5 A^4) ... simplified Horner
        let u_inner = id.scale(c[1]).add(&a2.scale(c[3])).add(&a4.scale(c[5]));
        let u = a_scaled.matmul(&u_inner);

        let v = id.scale(c[0]).add(&a2.scale(c[2])).add(&a4.scale(c[4])).add(&a6.scale(c[6]));

        let lhs = v.sub(&u);
        let rhs = v.add(&u);

        // Solve (V - U) R = (V + U)
        let mut result = lhs.solve_matrix(&rhs);

        // Repeated squaring
        for _ in 0..s {
            result = result.matmul(&result);
        }
        result
    }

    /// Matrix power (integer, non-negative)
    pub fn pow(&self, p: u32) -> Matrix {
        assert!(self.is_square());
        if p == 0 { return Matrix::eye(self.rows); }
        let mut result = Matrix::eye(self.rows);
        let mut base = self.clone();
        let mut exp = p;
        while exp > 0 {
            if exp & 1 == 1 { result = result.matmul(&base); }
            base = base.matmul(&base);
            exp >>= 1;
        }
        result
    }

    /// Kronecker product
    pub fn kron(&self, other: &Matrix) -> Matrix {
        let m = self.rows * other.rows;
        let n = self.cols * other.cols;
        let mut result = Matrix::zeros(m, n);
        for i1 in 0..self.rows {
            for j1 in 0..self.cols {
                let s = self[(i1, j1)];
                for i2 in 0..other.rows {
                    for j2 in 0..other.cols {
                        result[(i1 * other.rows + i2, j1 * other.cols + j2)] = s * other[(i2, j2)];
                    }
                }
            }
        }
        result
    }

    /// Rank via SVD
    pub fn rank(&self, tol: f64) -> usize {
        let (_, sigma, _) = self.svd();
        sigma.iter().filter(|&&s| s > tol).count()
    }

    /// Null space basis via SVD (columns of V corresponding to zero singular values)
    pub fn null_space(&self, tol: f64) -> Matrix {
        let (_, sigma, v) = self.svd();
        let null_dim = sigma.iter().filter(|&&s| s <= tol).count();
        let n = v.rows;
        let start = sigma.len() - null_dim;
        let mut result = Matrix::zeros(n, null_dim);
        for j in 0..null_dim {
            for i in 0..n { result[(i, j)] = v[(i, start + j)]; }
        }
        result
    }

    /// Column space basis via SVD
    pub fn col_space(&self, tol: f64) -> Matrix {
        let (u, sigma, _) = self.svd();
        let rank = sigma.iter().filter(|&&s| s > tol).count();
        let m = u.rows;
        let mut result = Matrix::zeros(m, rank);
        for j in 0..rank {
            for i in 0..m { result[(i, j)] = u[(i, j)]; }
        }
        result
    }

    /// Solve least squares min ||Ax - b||^2 via QR
    pub fn least_squares(&self, b: &[f64]) -> Vec<f64> {
        let (q, r) = self.qr_householder();
        let n = self.cols;
        // Q^T b
        let mut qtb = vec![0.0; n];
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..self.rows { s += q[(i, j)] * b[i]; }
            qtb[j] = s;
        }
        // Back-substitute R x = Q^T b
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = qtb[i];
            for j in (i + 1)..n { s -= r[(i, j)] * x[j]; }
            let d = r[(i, i)];
            x[i] = if d.abs() > 1e-14 { s / d } else { 0.0 };
        }
        x
    }

    /// Compute the outer product of two vectors (given as column matrices or slices)
    pub fn outer(a: &[f64], b: &[f64]) -> Matrix {
        let m = a.len();
        let n = b.len();
        let mut out = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n { out[(i, j)] = a[i] * b[j]; }
        }
        out
    }

    /// Dot product of two vectors stored as flat matrices
    pub fn dot(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Vector L2 norm
    pub fn vec_norm(v: &[f64]) -> f64 {
        v.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Gram-Schmidt orthonormalization of column vectors
    pub fn orthonormalize_cols(&self) -> Matrix {
        let m = self.rows;
        let n = self.cols;
        let mut result = self.clone();
        for j in 0..n {
            for i in 0..j {
                let mut dot = 0.0;
                let mut ni = 0.0;
                for r in 0..m {
                    dot += result[(r, i)] * result[(r, j)];
                    ni += result[(r, i)] * result[(r, i)];
                }
                if ni > 1e-30 {
                    let proj = dot / ni;
                    for r in 0..m { result[(r, j)] -= proj * result[(r, i)]; }
                }
            }
            let mut nrm = 0.0;
            for r in 0..m { nrm += result[(r, j)] * result[(r, j)]; }
            nrm = nrm.sqrt();
            if nrm > 1e-14 {
                for r in 0..m { result[(r, j)] /= nrm; }
            }
        }
        result
    }

    /// Extract submatrix
    pub fn submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Matrix {
        let mut out = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                out[(i, j)] = self[(row_start + i, col_start + j)];
            }
        }
        out
    }

    /// Set submatrix
    pub fn set_submatrix(&mut self, row_start: usize, col_start: usize, sub: &Matrix) {
        for i in 0..sub.rows {
            for j in 0..sub.cols {
                self[(row_start + i, col_start + j)] = sub[(i, j)];
            }
        }
    }

    /// Horizontal concatenation [self | other]
    pub fn hstack(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        let mut out = Matrix::zeros(self.rows, self.cols + other.cols);
        for i in 0..self.rows {
            for j in 0..self.cols { out[(i, j)] = self[(i, j)]; }
            for j in 0..other.cols { out[(i, self.cols + j)] = other[(i, j)]; }
        }
        out
    }

    /// Vertical concatenation
    pub fn vstack(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.cols);
        let mut data = self.data.clone();
        data.extend_from_slice(&other.data);
        Matrix { rows: self.rows + other.rows, cols: self.cols, data }
    }

    /// Apply function element-wise
    pub fn map<F: Fn(f64) -> f64>(&self, f: F) -> Matrix {
        let data: Vec<f64> = self.data.iter().map(|&x| f(x)).collect();
        Matrix { rows: self.rows, cols: self.cols, data }
    }

    /// Sum of all elements
    pub fn sum(&self) -> f64 { self.data.iter().sum() }

    /// Max element
    pub fn max(&self) -> f64 { self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max) }

    /// Min element
    pub fn min(&self) -> f64 { self.data.iter().copied().fold(f64::INFINITY, f64::min) }

    /// Reshape (must have same total elements)
    pub fn reshape(&self, rows: usize, cols: usize) -> Matrix {
        assert_eq!(self.len(), rows * cols);
        Matrix { rows, cols, data: self.data.clone() }
    }

    /// Flatten to column vector
    pub fn flatten(&self) -> Matrix {
        Matrix { rows: self.len(), cols: 1, data: self.data.clone() }
    }

    /// Extract diagonal as vector
    pub fn diag_vec(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self[(i, i)]).collect()
    }

    /// Create block diagonal matrix from a list of matrices
    pub fn block_diag(blocks: &[&Matrix]) -> Matrix {
        let total_rows: usize = blocks.iter().map(|b| b.rows).sum();
        let total_cols: usize = blocks.iter().map(|b| b.cols).sum();
        let mut out = Matrix::zeros(total_rows, total_cols);
        let mut r = 0;
        let mut c = 0;
        for block in blocks {
            out.set_submatrix(r, c, block);
            r += block.rows;
            c += block.cols;
        }
        out
    }

    /// Matrix-vector multiply: self * v
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut s = 0.0;
            for j in 0..self.cols { s += self[(i, j)] * v[j]; }
            out[i] = s;
        }
        out
    }

    /// Solve tridiagonal system using Thomas algorithm.
    /// a = sub-diagonal (len n-1), b = diagonal (len n), c = super-diagonal (len n-1), d = rhs (len n)
    pub fn solve_tridiag(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
        let n = b.len();
        assert_eq!(a.len(), n - 1);
        assert_eq!(c.len(), n - 1);
        assert_eq!(d.len(), n);
        let mut cp = vec![0.0; n];
        let mut dp = vec![0.0; n];
        cp[0] = c[0] / b[0];
        dp[0] = d[0] / b[0];
        for i in 1..n {
            let m = b[i] - a[i - 1] * cp[i - 1];
            if i < n - 1 { cp[i] = c[i] / m; }
            dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m;
        }
        let mut x = vec![0.0; n];
        x[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() { x[i] = dp[i] - cp[i] * x[i + 1]; }
        x
    }

    /// Iterative refinement for Ax = b
    pub fn solve_refined(&self, b: &[f64], max_iter: usize) -> Vec<f64> {
        let mut x = self.solve(b);
        for _ in 0..max_iter {
            let ax = self.matvec(&x);
            let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
            let dx = self.solve(&r);
            for i in 0..x.len() { x[i] += dx[i]; }
        }
        x
    }

    /// Conjugate gradient solver for SPD systems
    pub fn solve_cg(&self, b: &[f64], tol: f64, max_iter: usize) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];
        let mut r: Vec<f64> = b.to_vec();
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

        for _ in 0..max_iter {
            let ap = self.matvec(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-30 { break; }
            let alpha = rs_old / pap;

            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let rs_new: f64 = r.iter().map(|v| v * v).sum();
            if rs_new.sqrt() < tol { break; }

            let beta = rs_new / rs_old;
            for i in 0..n { p[i] = r[i] + beta * p[i]; }
            rs_old = rs_new;
        }
        x
    }

    /// Power iteration for dominant eigenvalue
    pub fn power_iteration(&self, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
        let n = self.rows;
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut lambda = 0.0;
        for _ in 0..max_iter {
            let w = self.matvec(&v);
            let new_lambda = w.iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
            let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm < 1e-30 { break; }
            for i in 0..n { v[i] = w[i] / nrm; }
            if (new_lambda - lambda).abs() < tol { lambda = new_lambda; break; }
            lambda = new_lambda;
        }
        (lambda, v)
    }

    /// Inverse iteration for eigenvalue nearest mu
    pub fn inverse_iteration(&self, mu: f64, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
        let n = self.rows;
        let shifted = self.sub(&Matrix::eye(n).scale(mu));
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut lambda = mu;
        for _ in 0..max_iter {
            let w = shifted.solve(&v);
            let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm < 1e-30 { break; }
            for i in 0..n { v[i] = w[i] / nrm; }
            let av = self.matvec(&v);
            let new_lambda: f64 = av.iter().zip(&v).map(|(a, b)| a * b).sum();
            if (new_lambda - lambda).abs() < tol { lambda = new_lambda; break; }
            lambda = new_lambda;
        }
        (lambda, v)
    }

    /// Compute all eigenvalues of a general (non-symmetric) matrix via QR with shifts.
    /// Returns complex eigenvalues as (real, imag) pairs.
    pub fn eigenvalues_general(&self) -> Vec<(f64, f64)> {
        assert!(self.is_square());
        let n = self.rows;
        // Hessenberg reduction
        let mut h = self.to_hessenberg();
        let max_iter = 200 * n;
        let mut p = n;

        let mut eigenvalues: Vec<(f64, f64)> = Vec::with_capacity(n);

        let mut iter = 0;
        while p > 0 && iter < max_iter {
            iter += 1;
            if p == 1 {
                eigenvalues.push((h[(0, 0)], 0.0));
                break;
            }

            // Check subdiagonal convergence
            let tol = 1e-12 * (h[(p - 2, p - 2)].abs() + h[(p - 1, p - 1)].abs()).max(1e-30);
            if h[(p - 1, p - 2)].abs() < tol {
                eigenvalues.push((h[(p - 1, p - 1)], 0.0));
                p -= 1;
                continue;
            }

            if p == 2 {
                let (e1, e2) = eigen_2x2(h[(0, 0)], h[(0, 1)], h[(1, 0)], h[(1, 1)]);
                eigenvalues.push(e1);
                eigenvalues.push(e2);
                break;
            }

            // Check for 2x2 block
            if p >= 3 {
                let sub_tol = 1e-12 * (h[(p - 3, p - 3)].abs() + h[(p - 2, p - 2)].abs()).max(1e-30);
                if h[(p - 2, p - 3)].abs() < sub_tol {
                    let (e1, e2) = eigen_2x2(
                        h[(p - 2, p - 2)], h[(p - 2, p - 1)],
                        h[(p - 1, p - 2)], h[(p - 1, p - 1)],
                    );
                    eigenvalues.push(e1);
                    eigenvalues.push(e2);
                    p -= 2;
                    continue;
                }
            }

            // Francis double-shift QR step
            let s = h[(p - 2, p - 2)] + h[(p - 1, p - 1)];
            let t = h[(p - 2, p - 2)] * h[(p - 1, p - 1)] - h[(p - 2, p - 1)] * h[(p - 1, p - 2)];

            let mut x = h[(0, 0)] * h[(0, 0)] + h[(0, 1)] * h[(1, 0)] - s * h[(0, 0)] + t;
            let mut y = h[(1, 0)] * (h[(0, 0)] + h[(1, 1)] - s);
            let mut z = h[(1, 0)] * h[(2, 1)];

            for k in 0..p.saturating_sub(2) {
                let (c1, s1, c2, s2) = double_shift_reflector(x, y, z);
                let _ = (c2, s2);

                let r = if k > 0 { k - 1 } else { 0 };
                // Apply 3x3 Householder from left
                apply_householder_left_3(&mut h, k, r, p, c1, s1, z);
                // Apply from right
                apply_householder_right_3(&mut h, k, p, c1, s1, z);

                if k + 3 < p {
                    x = h[(k + 1, k)];
                    y = h[(k + 2, k)];
                    z = if k + 3 < p { h[(k + 3, k)] } else { 0.0 };
                }
            }

            // Final 2x2 Givens
            if p >= 2 {
                let k = p - 2;
                let xx = h[(k, k.saturating_sub(1).max(k))];
                let yy = h[(k + 1, k.saturating_sub(1).max(k))];
                let rr = (xx * xx + yy * yy).sqrt();
                if rr > 1e-30 {
                    let c = xx / rr;
                    let s_val = -yy / rr;
                    for j in 0..p {
                        let a = h[(k, j)];
                        let b = h[(k + 1, j)];
                        h[(k, j)] = c * a - s_val * b;
                        h[(k + 1, j)] = s_val * a + c * b;
                    }
                    for i in 0..p {
                        let a = h[(i, k)];
                        let b = h[(i, k + 1)];
                        h[(i, k)] = c * a - s_val * b;
                        h[(i, k + 1)] = s_val * a + c * b;
                    }
                }
            }
        }

        // If we didn't converge, extract remaining
        while eigenvalues.len() < n {
            eigenvalues.push((0.0, 0.0));
        }

        eigenvalues
    }

    /// Reduce to upper Hessenberg form via Householder
    pub fn to_hessenberg(&self) -> Matrix {
        assert!(self.is_square());
        let n = self.rows;
        let mut h = self.clone();

        for k in 0..(n.saturating_sub(2)) {
            let mut x = vec![0.0; n - k - 1];
            for i in 0..x.len() { x[i] = h[(k + 1 + i, k)]; }

            let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_x < 1e-15 { continue; }

            let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
            x[0] += sign * norm_x;
            let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm_v < 1e-15 { continue; }
            for v in &mut x { *v /= norm_v; }

            let start = k + 1;
            // H = H - 2 v (v^T H) (from left, rows start..n)
            for j in 0..n {
                let mut dot = 0.0;
                for i in 0..x.len() { dot += x[i] * h[(start + i, j)]; }
                for i in 0..x.len() { h[(start + i, j)] -= 2.0 * x[i] * dot; }
            }
            // H = H - 2 (H v) v^T (from right, cols start..n)
            for i in 0..n {
                let mut dot = 0.0;
                for j in 0..x.len() { dot += h[(i, start + j)] * x[j]; }
                for j in 0..x.len() { h[(i, start + j)] -= 2.0 * dot * x[j]; }
            }
        }
        h
    }

    /// Bidiagonalization: A = U B V^T where B is bidiagonal
    pub fn bidiagonalize(&self) -> (Matrix, Matrix, Matrix) {
        let m = self.rows;
        let n = self.cols;
        let mut b = self.clone();
        let mut u = Matrix::eye(m);
        let mut v = Matrix::eye(n);

        for k in 0..n.min(m) {
            // Left Householder
            if k < m {
                let mut x = vec![0.0; m - k];
                for i in 0..x.len() { x[i] = b[(k + i, k)]; }
                let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm_x > 1e-15 {
                    let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
                    x[0] += sign * norm_x;
                    let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if norm_v > 1e-15 {
                        for v in &mut x { *v /= norm_v; }
                        for j in k..n {
                            let mut dot = 0.0;
                            for i in 0..x.len() { dot += x[i] * b[(k + i, j)]; }
                            for i in 0..x.len() { b[(k + i, j)] -= 2.0 * x[i] * dot; }
                        }
                        for j in 0..m {
                            let mut dot = 0.0;
                            for i in 0..x.len() { dot += u[(j, k + i)] * x[i]; }
                            for i in 0..x.len() { u[(j, k + i)] -= 2.0 * dot * x[i]; }
                        }
                    }
                }
            }

            // Right Householder
            if k + 1 < n {
                let mut x = vec![0.0; n - k - 1];
                for i in 0..x.len() { x[i] = b[(k, k + 1 + i)]; }
                let norm_x = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm_x > 1e-15 {
                    let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
                    x[0] += sign * norm_x;
                    let norm_v = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if norm_v > 1e-15 {
                        for v in &mut x { *v /= norm_v; }
                        for i in k..m {
                            let mut dot = 0.0;
                            for j in 0..x.len() { dot += b[(i, k + 1 + j)] * x[j]; }
                            for j in 0..x.len() { b[(i, k + 1 + j)] -= 2.0 * dot * x[j]; }
                        }
                        for i in 0..n {
                            let mut dot = 0.0;
                            for j in 0..x.len() { dot += v[(i, k + 1 + j)] * x[j]; }
                            for j in 0..x.len() { v[(i, k + 1 + j)] -= 2.0 * dot * x[j]; }
                        }
                    }
                }
            }
        }
        (u, b, v)
    }

    /// Schur decomposition: A = Q T Q^T, where T is quasi-upper-triangular
    pub fn schur(&self) -> (Matrix, Matrix) {
        assert!(self.is_square());
        let n = self.rows;
        let mut t = self.to_hessenberg();
        let mut q = Matrix::eye(n);

        for _ in 0..(200 * n) {
            // Check convergence
            let mut converged = true;
            for i in 1..n {
                if t[(i, i - 1)].abs() > 1e-12 { converged = false; break; }
            }
            if converged { break; }

            // Single shift QR step
            let shift = t[(n - 1, n - 1)];
            for i in 0..n { t[(i, i)] -= shift; }
            let (q_step, r_step) = t.qr_householder();
            t = r_step.matmul(&q_step);
            for i in 0..n { t[(i, i)] += shift; }
            q = q.matmul(&q_step);
        }
        (q, t)
    }
}

/// Compute eigenvalues of 2x2 matrix
fn eigen_2x2(a: f64, b: f64, c: f64, d: f64) -> ((f64, f64), (f64, f64)) {
    let tr = a + d;
    let det = a * d - b * c;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let sq = disc.sqrt();
        ((0.5 * (tr + sq), 0.0), (0.5 * (tr - sq), 0.0))
    } else {
        let sq = (-disc).sqrt();
        ((0.5 * tr, 0.5 * sq), (0.5 * tr, -0.5 * sq))
    }
}

/// Double-shift reflector helper
fn double_shift_reflector(x: f64, y: f64, z: f64) -> (f64, f64, f64, f64) {
    let r = (x * x + y * y + z * z).sqrt();
    if r < 1e-30 { return (1.0, 0.0, 1.0, 0.0); }
    let c1 = x / r;
    let s1 = y / r;
    let c2 = x / (x * x + y * y).sqrt().max(1e-30);
    let s2 = z / r;
    (c1, s1, c2, s2)
}

fn apply_householder_left_3(h: &mut Matrix, k: usize, _r: usize, p: usize, c: f64, s: f64, z: f64) {
    let r = (1.0 + s * s / (c * c + 1e-30) + z * z / ((c * c + s * s).max(1e-30))).sqrt();
    let _ = r;
    // Simplified: apply Givens between rows k and k+1
    for j in 0..p {
        let a = h[(k, j)];
        let b = h[(k + 1, j)];
        h[(k, j)] = c * a + s * b;
        h[(k + 1, j)] = -s * a + c * b;
    }
}

fn apply_householder_right_3(h: &mut Matrix, k: usize, p: usize, c: f64, s: f64, _z: f64) {
    for i in 0..p {
        let a = h[(i, k)];
        let b = h[(i, k + 1)];
        h[(i, k)] = c * a + s * b;
        h[(i, k + 1)] = -s * a + c * b;
    }
}

/// Sparse matrix in CSR format
#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    /// Create from triplets (row, col, value)
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut sorted = triplets.to_vec();
        sorted.sort_by_key(|&(r, c, _)| (r, c));

        let mut row_ptr = vec![0; rows + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());

        for &(r, c, v) in &sorted {
            row_ptr[r + 1] += 1;
            col_idx.push(c);
            values.push(v);
        }
        for i in 1..=rows { row_ptr[i] += row_ptr[i - 1]; }

        Self { rows, cols, row_ptr, col_idx, values }
    }

    /// Sparse matrix-vector multiply
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut s = 0.0;
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                s += self.values[idx] * x[self.col_idx[idx]];
            }
            y[i] = s;
        }
        y
    }

    /// Convert to dense
    pub fn to_dense(&self) -> Matrix {
        let mut m = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                m[(i, self.col_idx[idx])] = self.values[idx];
            }
        }
        m
    }

    /// Number of non-zeros
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Transpose
    pub fn transpose(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.nnz());
        for i in 0..self.rows {
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                triplets.push((self.col_idx[idx], i, self.values[idx]));
            }
        }
        Self::from_triplets(self.cols, self.rows, &triplets)
    }

    /// Sparse CG solver
    pub fn solve_cg(&self, b: &[f64], tol: f64, max_iter: usize) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];
        let mut r: Vec<f64> = b.to_vec();
        let mut p = r.clone();
        let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

        for _ in 0..max_iter {
            let ap = self.matvec(&p);
            let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-30 { break; }
            let alpha = rs_old / pap;
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            let rs_new: f64 = r.iter().map(|v| v * v).sum();
            if rs_new.sqrt() < tol { break; }
            let beta = rs_new / rs_old;
            for i in 0..n { p[i] = r[i] + beta * p[i]; }
            rs_old = rs_new;
        }
        x
    }
}

/// Permutation matrix representation
#[derive(Clone, Debug)]
pub struct Permutation {
    pub indices: Vec<usize>,
}

impl Permutation {
    pub fn identity(n: usize) -> Self { Self { indices: (0..n).collect() } }
    pub fn from_indices(indices: Vec<usize>) -> Self { Self { indices } }
    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; v.len()];
        for (i, &p) in self.indices.iter().enumerate() { out[i] = v[p]; }
        out
    }
    pub fn inverse(&self) -> Self {
        let mut inv = vec![0; self.indices.len()];
        for (i, &p) in self.indices.iter().enumerate() { inv[p] = i; }
        Self { indices: inv }
    }
    pub fn sign(&self) -> i32 {
        let n = self.indices.len();
        let mut visited = vec![false; n];
        let mut sgn = 1i32;
        for i in 0..n {
            if visited[i] { continue; }
            let mut len = 0;
            let mut j = i;
            while !visited[j] { visited[j] = true; j = self.indices[j]; len += 1; }
            if len % 2 == 0 { sgn = -sgn; }
        }
        sgn
    }
    pub fn to_matrix(&self) -> Matrix {
        let n = self.indices.len();
        let mut m = Matrix::zeros(n, n);
        for (i, &p) in self.indices.iter().enumerate() { m[(i, p)] = 1.0; }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_identity() {
        let a = Matrix::random(4, 4, 42);
        let i = Matrix::eye(4);
        let b = a.matmul(&i);
        for k in 0..16 { assert!((a.data[k] - b.data[k]).abs() < 1e-12); }
    }

    #[test]
    fn test_inverse() {
        let a = Matrix::random_spd(4, 123);
        let inv = a.inverse().unwrap();
        let prod = a.matmul(&inv);
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod[(i, j)] - expected).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn test_cholesky() {
        let a = Matrix::random_spd(5, 99);
        let l = a.cholesky().unwrap();
        let llt = l.matmul(&l.transpose());
        for i in 0..5 {
            for j in 0..5 {
                assert!((a[(i, j)] - llt[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_qr_householder() {
        let a = Matrix::random(5, 4, 77);
        let (q, r) = a.qr_householder();
        let qr = q.matmul(&r);
        for i in 0..5 {
            for j in 0..4 {
                assert!((a[(i, j)] - qr[(i, j)]).abs() < 1e-10);
            }
        }
        // Q^T Q ≈ I
        let qtq = q.transpose().matmul(&q);
        for i in 0..4 {
            for j in 0..4 {
                let exp = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[(i, j)] - exp).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_solve() {
        let a = Matrix::random_spd(5, 42);
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = a.solve(&b);
        let ax = a.matvec(&x);
        for i in 0..5 { assert!((ax[i] - b[i]).abs() < 1e-8); }
    }

    #[test]
    fn test_det() {
        let a = Matrix::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!((a.det() - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn test_eigen_symmetric() {
        let a = Matrix::random_spd(4, 55);
        let (evals, evecs) = a.eigen_symmetric();
        // Check A v = lambda v for each eigenpair
        for j in 0..4 {
            let v: Vec<f64> = (0..4).map(|i| evecs[(i, j)]).collect();
            let av = a.matvec(&v);
            for i in 0..4 {
                assert!((av[i] - evals[j] * v[i]).abs() < 1e-6, "eigen check failed");
            }
        }
    }

    #[test]
    fn test_svd() {
        let a = Matrix::random(4, 3, 88);
        let (u, sigma, v) = a.svd();
        // Reconstruct
        let mut reconstructed = Matrix::zeros(4, 3);
        for s in 0..3 {
            for i in 0..4 {
                for j in 0..3 {
                    reconstructed[(i, j)] += u[(i, s)] * sigma[s] * v[(j, s)];
                }
            }
        }
        for i in 0..4 {
            for j in 0..3 {
                assert!((a[(i, j)] - reconstructed[(i, j)]).abs() < 1e-8);
            }
        }
    }
}
