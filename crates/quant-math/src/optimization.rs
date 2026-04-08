// optimization.rs — Gradient descent variants, Newton, BFGS, L-BFGS, CG, simplex, metaheuristics

/// Result of an optimization run
#[derive(Clone, Debug)]
pub struct OptResult {
    pub x: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub converged: bool,
    pub grad_norm: f64,
}

/// Gradient descent with constant learning rate
pub fn gradient_descent<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        for j in 0..n { x[j] -= lr * g[j]; }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// Gradient descent with momentum (heavy ball)
pub fn gradient_descent_momentum<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, momentum: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut v = vec![0.0; n];
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        for j in 0..n {
            v[j] = momentum * v[j] - lr * g[j];
            x[j] += v[j];
        }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// Nesterov accelerated gradient
pub fn nesterov_ag<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, momentum: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut v = vec![0.0; n];
    let mut val;
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let lookahead: Vec<f64> = (0..n).map(|j| x[j] + momentum * v[j]).collect();
        let g = grad(&lookahead);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        for j in 0..n {
            v[j] = momentum * v[j] - lr * g[j];
            x[j] += v[j];
        }
    }
    val = f(&x);
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// Adam optimizer
pub fn adam<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, beta1: f64, beta2: f64, eps: f64,
    max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        let t = (i + 1) as f64;
        for j in 0..n {
            m[j] = beta1 * m[j] + (1.0 - beta1) * g[j];
            v[j] = beta2 * v[j] + (1.0 - beta2) * g[j] * g[j];
            let m_hat = m[j] / (1.0 - beta1.powf(t));
            let v_hat = v[j] / (1.0 - beta2.powf(t));
            x[j] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// AdaGrad optimizer
pub fn adagrad<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, eps: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g_acc = vec![0.0; n];
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        for j in 0..n {
            g_acc[j] += g[j] * g[j];
            x[j] -= lr * g[j] / (g_acc[j].sqrt() + eps);
        }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// RMSProp optimizer
pub fn rmsprop<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, decay: f64, eps: f64,
    max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut eg2 = vec![0.0; n];
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        for j in 0..n {
            eg2[j] = decay * eg2[j] + (1.0 - decay) * g[j] * g[j];
            x[j] -= lr * g[j] / (eg2[j].sqrt() + eps);
        }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// AdamW (Adam with decoupled weight decay)
pub fn adamw<F, G>(
    f: F, grad: G, x0: &[f64], lr: f64, beta1: f64, beta2: f64, eps: f64,
    weight_decay: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }
        let t = (i + 1) as f64;
        for j in 0..n {
            m[j] = beta1 * m[j] + (1.0 - beta1) * g[j];
            v[j] = beta2 * v[j] + (1.0 - beta2) * g[j] * g[j];
            let m_hat = m[j] / (1.0 - beta1.powf(t));
            let v_hat = v[j] / (1.0 - beta2.powf(t));
            x[j] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * x[j]);
        }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// Newton's method (requires Hessian)
pub fn newton_method<F, G, H>(
    f: F, grad: G, hessian: H, x0: &[f64], max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
    H: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = 0.0;

    for i in 0..max_iter {
        it = i + 1;
        let g = grad(&x);
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }

        let h = hessian(&x);
        let hmat = crate::linear_algebra::Matrix::from_rows(&h);
        let dx = hmat.solve(&g);
        for j in 0..n { x[j] -= dx[j]; }
        val = f(&x);
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// BFGS quasi-Newton method
pub fn bfgs<F, G>(
    f: F, grad: G, x0: &[f64], max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;

    // Initial Hessian approximation: identity
    let mut h_inv = vec![vec![0.0; n]; n];
    for i in 0..n { h_inv[i][i] = 1.0; }

    for i in 0..max_iter {
        it = i + 1;
        if gn < tol { converged = true; break; }

        // Direction: d = -H^{-1} g
        let mut d = vec![0.0; n];
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n { s += h_inv[j][k] * g[k]; }
            d[j] = -s;
        }

        // Line search (backtracking Armijo)
        let mut alpha = 1.0;
        let c1 = 1e-4;
        let dg: f64 = d.iter().zip(&g).map(|(a, b)| a * b).sum();
        let fx = val;
        for _ in 0..50 {
            let x_new: Vec<f64> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
            let f_new = f(&x_new);
            if f_new <= fx + c1 * alpha * dg { break; }
            alpha *= 0.5;
        }

        let x_new: Vec<f64> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
        let g_new = grad(&x_new);
        let new_val = f(&x_new);

        // s = x_new - x, y = g_new - g
        let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();

        if sy > 1e-10 {
            // BFGS update
            let rho = 1.0 / sy;
            // H_inv = (I - rho s y^T) H_inv (I - rho y s^T) + rho s s^T
            let mut hy = vec![0.0; n];
            for j in 0..n {
                for k in 0..n { hy[j] += h_inv[j][k] * y[k]; }
            }
            let yhy: f64 = y.iter().zip(&hy).map(|(a, b)| a * b).sum();

            for j in 0..n {
                for k in 0..n {
                    h_inv[j][k] += rho * ((1.0 + rho * yhy) * s[j] * s[k]
                        - hy[j] * s[k] - s[j] * hy[k]);
                }
            }
        }

        x = x_new;
        g = g_new;
        val = new_val;
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// L-BFGS (limited-memory BFGS)
pub fn lbfgs<F, G>(
    f: F, grad: G, x0: &[f64], m: usize, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    for i in 0..max_iter {
        it = i + 1;
        if gn < tol { converged = true; break; }

        // Two-loop recursion
        let k = s_hist.len();
        let mut q = g.clone();
        let mut alpha_arr = vec![0.0; k];

        for j in (0..k).rev() {
            let dot: f64 = s_hist[j].iter().zip(&q).map(|(a, b)| a * b).sum();
            alpha_arr[j] = rho_hist[j] * dot;
            for l in 0..n { q[l] -= alpha_arr[j] * y_hist[j][l]; }
        }

        // Initial Hessian scaling
        let gamma = if k > 0 {
            let sy: f64 = s_hist[k - 1].iter().zip(&y_hist[k - 1]).map(|(a, b)| a * b).sum();
            let yy: f64 = y_hist[k - 1].iter().map(|v| v * v).sum();
            if yy > 1e-30 { sy / yy } else { 1.0 }
        } else { 1.0 };
        let mut r: Vec<f64> = q.iter().map(|v| gamma * v).collect();

        for j in 0..k {
            let dot: f64 = y_hist[j].iter().zip(&r).map(|(a, b)| a * b).sum();
            let beta = rho_hist[j] * dot;
            for l in 0..n { r[l] += (alpha_arr[j] - beta) * s_hist[j][l]; }
        }

        // Direction
        let d: Vec<f64> = r.iter().map(|v| -v).collect();

        // Backtracking line search
        let mut alpha = 1.0;
        let c1 = 1e-4;
        let dg: f64 = d.iter().zip(&g).map(|(a, b)| a * b).sum();
        for _ in 0..40 {
            let x_new: Vec<f64> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
            if f(&x_new) <= val + c1 * alpha * dg { break; }
            alpha *= 0.5;
        }

        let x_new: Vec<f64> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
        let g_new = grad(&x_new);
        let new_val = f(&x_new);

        let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        let sy: f64 = s.iter().zip(&y).map(|(a, b)| a * b).sum();

        if sy > 1e-10 {
            if s_hist.len() >= m {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s);
            y_hist.push(y);
        }

        x = x_new;
        g = g_new;
        val = new_val;
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// Nonlinear conjugate gradient (Fletcher-Reeves)
pub fn conjugate_gradient_fr<F, G>(
    f: F, grad: G, x0: &[f64], max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut d: Vec<f64> = g.iter().map(|v| -v).collect();
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();

    for i in 0..max_iter {
        it = i + 1;
        if gn < tol { converged = true; break; }

        // Line search
        let alpha = backtrack_line_search(&f, &x, &d, &g, val);
        for j in 0..n { x[j] += alpha * d[j]; }

        let g_new = grad(&x);
        val = f(&x);

        let gg_old: f64 = g.iter().map(|v| v * v).sum();
        let gg_new: f64 = g_new.iter().map(|v| v * v).sum();
        let beta = if gg_old > 1e-30 { gg_new / gg_old } else { 0.0 };

        for j in 0..n { d[j] = -g_new[j] + beta * d[j]; }
        g = g_new;
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Restart if direction is not descent
        let dg: f64 = d.iter().zip(&g).map(|(a, b)| a * b).sum();
        if dg >= 0.0 {
            for j in 0..n { d[j] = -g[j]; }
        }
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

/// Polak-Ribière conjugate gradient
pub fn conjugate_gradient_pr<F, G>(
    f: F, grad: G, x0: &[f64], max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);
    let mut d: Vec<f64> = g.iter().map(|v| -v).collect();
    let mut val = f(&x);
    let mut converged = false;
    let mut it = 0;
    let mut gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();

    for i in 0..max_iter {
        it = i + 1;
        if gn < tol { converged = true; break; }

        let alpha = backtrack_line_search(&f, &x, &d, &g, val);
        for j in 0..n { x[j] += alpha * d[j]; }
        let g_new = grad(&x);
        val = f(&x);

        let gg_old: f64 = g.iter().map(|v| v * v).sum();
        let gy: f64 = g_new.iter().zip(&g).map(|(gn, go)| (gn - go) * gn).sum();
        let beta = if gg_old > 1e-30 { (gy / gg_old).max(0.0) } else { 0.0 };

        for j in 0..n { d[j] = -g_new[j] + beta * d[j]; }
        g = g_new;
        gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    }
    OptResult { x, value: val, iterations: it, converged, grad_norm: gn }
}

fn backtrack_line_search<F: Fn(&[f64]) -> f64>(
    f: &F, x: &[f64], d: &[f64], g: &[f64], fx: f64,
) -> f64 {
    let n = x.len();
    let mut alpha = 1.0;
    let c1 = 1e-4;
    let dg: f64 = d.iter().zip(g).map(|(a, b)| a * b).sum();
    for _ in 0..50 {
        let x_new: Vec<f64> = (0..n).map(|j| x[j] + alpha * d[j]).collect();
        if f(&x_new) <= fx + c1 * alpha * dg { break; }
        alpha *= 0.5;
    }
    alpha
}

/// Simplex method for linear programming: min c^T x s.t. Ax <= b, x >= 0
/// Returns (optimal x, optimal value) or None if infeasible/unbounded.
pub fn simplex_lp(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Option<(Vec<f64>, f64)> {
    let m = a.len(); // constraints
    let n = c.len(); // variables
    let total = n + m; // with slack variables

    // Tableau: m+1 rows, total+1 cols (last col = RHS)
    let cols = total + 1;
    let rows = m + 1;
    let mut tab = vec![vec![0.0; cols]; rows];

    // Fill constraints
    for i in 0..m {
        for j in 0..n { tab[i][j] = a[i][j]; }
        tab[i][n + i] = 1.0; // slack
        tab[i][total] = b[i];
        if b[i] < 0.0 { return None; } // requires two-phase for negative RHS
    }
    // Objective row
    for j in 0..n { tab[m][j] = c[j]; }

    let mut basis: Vec<usize> = (n..total).collect();

    for _ in 0..(10 * (m + n)) {
        // Find entering variable (most negative in objective row)
        let mut pivot_col = 0;
        let mut min_val = -1e-10;
        for j in 0..total {
            if tab[m][j] < min_val { min_val = tab[m][j]; pivot_col = j; }
        }
        if min_val >= -1e-10 { break; } // optimal

        // Find leaving variable (minimum ratio test)
        let mut pivot_row = None;
        let mut min_ratio = f64::INFINITY;
        for i in 0..m {
            if tab[i][pivot_col] > 1e-10 {
                let ratio = tab[i][total] / tab[i][pivot_col];
                if ratio < min_ratio { min_ratio = ratio; pivot_row = Some(i); }
            }
        }
        let pivot_row = match pivot_row { Some(r) => r, None => return None }; // unbounded

        // Pivot
        let pivot_val = tab[pivot_row][pivot_col];
        for j in 0..cols { tab[pivot_row][j] /= pivot_val; }
        for i in 0..rows {
            if i == pivot_row { continue; }
            let factor = tab[i][pivot_col];
            for j in 0..cols { tab[i][j] -= factor * tab[pivot_row][j]; }
        }
        basis[pivot_row] = pivot_col;
    }

    let mut x = vec![0.0; n];
    for i in 0..m {
        if basis[i] < n { x[basis[i]] = tab[i][total]; }
    }
    let obj = -tab[m][total];
    Some((x, obj))
}

/// Golden section search for 1D minimization
pub fn golden_section<F: Fn(f64) -> f64>(
    f: F, mut a: f64, mut b: f64, tol: f64,
) -> (f64, f64) {
    let gr = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);

    while (b - a).abs() > tol {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    let x = 0.5 * (a + b);
    (x, f(x))
}

/// Brent's method for 1D minimization
pub fn brent_minimize<F: Fn(f64) -> f64>(
    f: F, ax: f64, bx: f64, cx: f64, tol: f64, max_iter: usize,
) -> (f64, f64) {
    let golden = 0.3819660;
    let mut a = ax.min(cx);
    let mut b = ax.max(cx);
    let mut x = bx;
    let mut w = bx;
    let mut v = bx;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut e = 0.0_f64;
    let mut d = 0.0_f64;

    for _ in 0..max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = tol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) { break; }

        let mut use_para = false;
        if e.abs() > tol1 {
            // Parabolic interpolation
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let p = (x - v) * q - (x - w) * r;
            let q = 2.0 * (q - r);
            let (p, q) = if q > 0.0 { (-p, q) } else { (p, -q) };
            if p.abs() < (0.5 * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = if x < xm { tol1 } else { -tol1 };
                }
                use_para = true;
            }
        }
        if !use_para {
            e = if x < xm { b - x } else { a - x };
            d = golden * e;
        }

        let u = if d.abs() >= tol1 { x + d } else { x + tol1 * d.signum() };
        let fu = f(u);

        if fu <= fx {
            if u < x { b = x; } else { a = x; }
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if u < x { a = u; } else { b = u; }
            if fu <= fw || (w - x).abs() < 1e-15 {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if fu <= fv || (v - x).abs() < 1e-15 || (v - w).abs() < 1e-15 {
                v = u; fv = fu;
            }
        }
        if !use_para { e = d; } else { e = d; }
    }
    (x, fx)
}

/// Nelder-Mead simplex (derivative-free)
pub fn nelder_mead<F: Fn(&[f64]) -> f64>(
    f: F, x0: &[f64], step: f64, max_iter: usize, tol: f64,
) -> OptResult {
    let n = x0.len();
    let np1 = n + 1;

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(np1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        simplex.push(v);
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;
    let mut it = 0;

    for i in 0..max_iter {
        it = i + 1;

        // Sort
        let mut idx: Vec<usize> = (0..np1).collect();
        idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_s: Vec<Vec<f64>> = idx.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_v: Vec<f64> = idx.iter().map(|&i| values[i]).collect();
        simplex = sorted_s;
        values = sorted_v;

        // Check convergence
        let range = values[np1 - 1] - values[0];
        if range < tol { break; }

        // Centroid (excluding worst)
        let mut centroid = vec![0.0; n];
        for j in 0..n {
            for k in 0..n { centroid[j] += simplex[k][j]; }
            centroid[j] /= n as f64;
        }

        // Reflection
        let xr: Vec<f64> = (0..n).map(|j| centroid[j] + alpha * (centroid[j] - simplex[n][j])).collect();
        let fr = f(&xr);

        if fr < values[0] {
            // Expansion
            let xe: Vec<f64> = (0..n).map(|j| centroid[j] + gamma * (xr[j] - centroid[j])).collect();
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe; values[n] = fe;
            } else {
                simplex[n] = xr; values[n] = fr;
            }
        } else if fr < values[n - 1] {
            simplex[n] = xr; values[n] = fr;
        } else {
            // Contraction
            let xc: Vec<f64> = if fr < values[n] {
                (0..n).map(|j| centroid[j] + rho * (xr[j] - centroid[j])).collect()
            } else {
                (0..n).map(|j| centroid[j] + rho * (simplex[n][j] - centroid[j])).collect()
            };
            let fc = f(&xc);
            if fc < values[n].min(fr) {
                simplex[n] = xc; values[n] = fc;
            } else {
                // Shrink
                for i in 1..np1 {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    values[i] = f(&simplex[i]);
                }
            }
        }
    }

    let best_idx = values.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap().0;
    OptResult {
        x: simplex[best_idx].clone(),
        value: values[best_idx],
        iterations: it,
        converged: it < max_iter,
        grad_norm: 0.0,
    }
}

/// Simulated annealing
pub fn simulated_annealing<F: Fn(&[f64]) -> f64>(
    f: F, x0: &[f64], temp_init: f64, temp_min: f64, cooling: f64,
    step_size: f64, max_iter: usize, seed: u64,
) -> OptResult {
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut best_x = x.clone();
    let mut best_f = fx;
    let mut temp = temp_init;
    let mut rng_state = seed;

    let mut lcg_next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / (u32::MAX as f64)
    };

    for i in 0..max_iter {
        // Random neighbor
        let mut x_new = x.clone();
        for j in 0..n {
            x_new[j] += (lcg_next(&mut rng_state) * 2.0 - 1.0) * step_size;
        }
        let f_new = f(&x_new);
        let delta = f_new - fx;

        if delta < 0.0 || lcg_next(&mut rng_state) < (-delta / temp).exp() {
            x = x_new;
            fx = f_new;
        }

        if fx < best_f {
            best_x = x.clone();
            best_f = fx;
        }

        temp *= cooling;
        if temp < temp_min { temp = temp_min; }
    }

    OptResult {
        x: best_x,
        value: best_f,
        iterations: max_iter,
        converged: true,
        grad_norm: 0.0,
    }
}

/// Differential evolution
pub fn differential_evolution<F: Fn(&[f64]) -> f64>(
    f: F, bounds: &[(f64, f64)], pop_size: usize, max_iter: usize,
    crossover_rate: f64, mutation_factor: f64, seed: u64,
) -> OptResult {
    let n = bounds.len();
    let mut rng_state = seed;
    let mut lcg_next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / (u32::MAX as f64)
    };
    let mut lcg_usize = |state: &mut u64, max: usize| -> usize {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % max
    };

    // Initialize population
    let mut pop: Vec<Vec<f64>> = (0..pop_size).map(|_| {
        bounds.iter().map(|&(lo, hi)| lo + lcg_next(&mut rng_state) * (hi - lo)).collect()
    }).collect();
    let mut fitness: Vec<f64> = pop.iter().map(|x| f(x)).collect();

    for _ in 0..max_iter {
        for i in 0..pop_size {
            // Select 3 distinct individuals != i
            let mut r = [0usize; 3];
            for k in 0..3 {
                loop {
                    let idx = lcg_usize(&mut rng_state, pop_size);
                    if idx != i && (k == 0 || (k == 1 && idx != r[0]) || (k == 2 && idx != r[0] && idx != r[1])) {
                        r[k] = idx;
                        break;
                    }
                }
            }

            // Mutation + crossover
            let j_rand = lcg_usize(&mut rng_state, n);
            let mut trial = pop[i].clone();
            for j in 0..n {
                if j == j_rand || lcg_next(&mut rng_state) < crossover_rate {
                    let mut v = pop[r[0]][j] + mutation_factor * (pop[r[1]][j] - pop[r[2]][j]);
                    v = v.clamp(bounds[j].0, bounds[j].1);
                    trial[j] = v;
                }
            }

            let f_trial = f(&trial);
            if f_trial <= fitness[i] {
                pop[i] = trial;
                fitness[i] = f_trial;
            }
        }
    }

    let best = fitness.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    OptResult {
        x: pop[best.0].clone(),
        value: *best.1,
        iterations: max_iter,
        converged: true,
        grad_norm: 0.0,
    }
}

/// Particle swarm optimization
pub fn particle_swarm<F: Fn(&[f64]) -> f64>(
    f: F, bounds: &[(f64, f64)], n_particles: usize, max_iter: usize,
    w: f64, c1: f64, c2: f64, seed: u64,
) -> OptResult {
    let n = bounds.len();
    let mut rng_state = seed;
    let mut lcg_next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f64) / (u32::MAX as f64)
    };

    let mut positions: Vec<Vec<f64>> = (0..n_particles).map(|_| {
        bounds.iter().map(|&(lo, hi)| lo + lcg_next(&mut rng_state) * (hi - lo)).collect()
    }).collect();
    let mut velocities: Vec<Vec<f64>> = (0..n_particles).map(|_| {
        bounds.iter().map(|&(lo, hi)| (lcg_next(&mut rng_state) - 0.5) * (hi - lo) * 0.1).collect()
    }).collect();

    let mut p_best = positions.clone();
    let mut p_best_f: Vec<f64> = positions.iter().map(|x| f(x)).collect();
    let g_idx = p_best_f.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).unwrap().0;
    let mut g_best = p_best[g_idx].clone();
    let mut g_best_f = p_best_f[g_idx];

    for _ in 0..max_iter {
        for i in 0..n_particles {
            for j in 0..n {
                let r1 = lcg_next(&mut rng_state);
                let r2 = lcg_next(&mut rng_state);
                velocities[i][j] = w * velocities[i][j]
                    + c1 * r1 * (p_best[i][j] - positions[i][j])
                    + c2 * r2 * (g_best[j] - positions[i][j]);
                positions[i][j] += velocities[i][j];
                positions[i][j] = positions[i][j].clamp(bounds[j].0, bounds[j].1);
            }
            let fi = f(&positions[i]);
            if fi < p_best_f[i] {
                p_best[i] = positions[i].clone();
                p_best_f[i] = fi;
                if fi < g_best_f {
                    g_best = positions[i].clone();
                    g_best_f = fi;
                }
            }
        }
    }

    OptResult {
        x: g_best,
        value: g_best_f,
        iterations: max_iter,
        converged: true,
        grad_norm: 0.0,
    }
}

/// Numerical gradient via central differences
pub fn numerical_gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<f64> {
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut x_plus = x.to_vec();
    let mut x_minus = x.to_vec();
    for i in 0..n {
        x_plus[i] = x[i] + h;
        x_minus[i] = x[i] - h;
        grad[i] = (f(&x_plus) - f(&x_minus)) / (2.0 * h);
        x_plus[i] = x[i];
        x_minus[i] = x[i];
    }
    grad
}

/// Numerical Hessian via central differences
pub fn numerical_hessian<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];
    let fx = f(x);
    for i in 0..n {
        for j in i..n {
            let mut xpp = x.to_vec();
            let mut xpm = x.to_vec();
            let mut xmp = x.to_vec();
            let mut xmm = x.to_vec();
            xpp[i] += h; xpp[j] += h;
            xpm[i] += h; xpm[j] -= h;
            xmp[i] -= h; xmp[j] += h;
            xmm[i] -= h; xmm[j] -= h;
            hess[i][j] = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h);
            hess[j][i] = hess[i][j];
        }
    }
    hess
}

/// Levenberg-Marquardt for nonlinear least squares
pub fn levenberg_marquardt<R, J>(
    residuals: R, jacobian: J, x0: &[f64], max_iter: usize, tol: f64,
) -> OptResult
where
    R: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut cost: f64 = r.iter().map(|v| v * v).sum::<f64>() * 0.5;
    let mut mu = 1e-3;
    let mut converged = false;
    let mut it = 0;

    for i in 0..max_iter {
        it = i + 1;
        let j = jacobian(&x);
        let m = r.len();

        // J^T J + mu I
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for k in 0..m {
            for a in 0..n {
                jtr[a] += j[k][a] * r[k];
                for b in 0..n {
                    jtj[a][b] += j[k][a] * j[k][b];
                }
            }
        }

        let gn: f64 = jtr.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gn < tol { converged = true; break; }

        for a in 0..n { jtj[a][a] += mu; }

        let mat = crate::linear_algebra::Matrix::from_rows(&jtj);
        let dx = mat.solve(&jtr);

        let x_new: Vec<f64> = (0..n).map(|j| x[j] - dx[j]).collect();
        let r_new = residuals(&x_new);
        let cost_new: f64 = r_new.iter().map(|v| v * v).sum::<f64>() * 0.5;

        if cost_new < cost {
            x = x_new;
            r = r_new;
            cost = cost_new;
            mu *= 0.5;
        } else {
            mu *= 2.0;
        }
    }

    OptResult {
        x,
        value: cost,
        iterations: it,
        converged,
        grad_norm: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_descent_quadratic() {
        // min (x-3)^2 + (y-5)^2
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] - 5.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] - 5.0)];
        let res = gradient_descent(f, g, &[0.0, 0.0], 0.1, 1000, 1e-8);
        assert!(res.converged);
        assert!((res.x[0] - 3.0).abs() < 1e-4);
        assert!((res.x[1] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let f = |x: &[f64]| 100.0 * (x[1] - x[0]*x[0]).powi(2) + (1.0 - x[0]).powi(2);
        let g = |x: &[f64]| vec![
            -400.0 * x[0] * (x[1] - x[0]*x[0]) + 2.0 * (x[0] - 1.0),
            200.0 * (x[1] - x[0]*x[0]),
        ];
        let res = bfgs(f, g, &[-1.0, 1.0], 5000, 1e-8);
        assert!((res.x[0] - 1.0).abs() < 1e-3);
        assert!((res.x[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_golden_section() {
        let f = |x: f64| (x - 2.0).powi(2);
        let (x, _) = golden_section(f, -5.0, 10.0, 1e-10);
        assert!((x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_nelder_mead() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let res = nelder_mead(f, &[0.0, 0.0], 1.0, 1000, 1e-10);
        assert!((res.x[0] - 1.0).abs() < 1e-4);
        assert!((res.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_simplex_lp() {
        // max 5x + 4y s.t. 6x + 4y <= 24, x + 2y <= 6, x,y >= 0
        // => min -5x - 4y
        let c = vec![-5.0, -4.0];
        let a = vec![vec![6.0, 4.0], vec![1.0, 2.0]];
        let b = vec![24.0, 6.0];
        let result = simplex_lp(&c, &a, &b);
        assert!(result.is_some());
        let (x, val) = result.unwrap();
        assert!((val - (-21.0)).abs() < 1e-6 || (val + 21.0).abs() < 1e-6);
    }
}
