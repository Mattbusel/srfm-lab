// numerical.rs — Integration, differentiation, root finding, ODE solvers, FFT

use std::f64::consts::PI;

// ============================================================
// Numerical Integration
// ============================================================

/// Trapezoidal rule
pub fn trapezoid<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

/// Simpson's rule
pub fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}

/// Simpson's 3/8 rule
pub fn simpson_38<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = n - (n % 3); // Make divisible by 3
    let n = n.max(3);
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 3 == 0 { 2.0 * f(x) } else { 3.0 * f(x) };
    }
    sum * 3.0 * h / 8.0
}

/// Boole's rule (5-point Newton-Cotes)
pub fn boole<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = (n / 4) * 4;
    let n = n.max(4);
    let h = (b - a) / n as f64;
    let mut sum = 0.0;
    for i in (0..n).step_by(4) {
        let x0 = a + i as f64 * h;
        sum += 7.0 * f(x0) + 32.0 * f(x0 + h) + 12.0 * f(x0 + 2.0 * h)
            + 32.0 * f(x0 + 3.0 * h) + 7.0 * f(x0 + 4.0 * h);
    }
    sum * 2.0 * h / 45.0
}

/// Gauss-Legendre quadrature (n-point, n = 2..20)
pub fn gauss_legendre<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let (nodes, weights) = gauss_legendre_nodes_weights(n);
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut sum = 0.0;
    for i in 0..n {
        let x = mid + half * nodes[i];
        sum += weights[i] * f(x);
    }
    sum * half
}

/// Compute Gauss-Legendre nodes and weights via eigenvalue method
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Use Golub-Welsch algorithm: eigenvalues of tridiagonal Jacobi matrix
    let mut diag = vec![0.0; n]; // all zeros for Legendre
    let mut sub = Vec::with_capacity(n);
    for i in 1..n {
        let k = i as f64;
        sub.push(k / (4.0 * k * k - 1.0).sqrt());
    }

    // QR iteration on tridiagonal
    let mut d = diag.clone();
    let mut e = sub.clone();
    let mut z = vec![vec![0.0; n]; n]; // eigenvectors
    for i in 0..n { z[i][i] = 1.0; }

    implicit_qr_tridiag(&mut d, &mut e, &mut z, n);

    let mut nodes = d;
    let mut weights: Vec<f64> = (0..n).map(|i| 2.0 * z[0][i] * z[0][i]).collect();

    // Sort by node value
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_nodes: Vec<f64> = idx.iter().map(|&i| nodes[i]).collect();
    let sorted_weights: Vec<f64> = idx.iter().map(|&i| weights[i]).collect();
    (sorted_nodes, sorted_weights)
}

fn implicit_qr_tridiag(d: &mut [f64], e: &mut [f64], z: &mut [Vec<f64>], n: usize) {
    for _ in 0..(30 * n) {
        let mut converged = true;
        for i in 0..n.saturating_sub(1) {
            if e[i].abs() > 1e-14 * (d[i].abs() + d[i + 1].abs()).max(1e-30) {
                converged = false;
                break;
            }
        }
        if converged { break; }

        // Find active block
        let mut p = n - 1;
        while p > 0 && e[p - 1].abs() < 1e-14 * (d[p - 1].abs() + d[p].abs()).max(1e-30) {
            p -= 1;
        }
        if p == 0 { break; }

        // Wilkinson shift
        let dd = (d[p - 1] - d[p]) / 2.0;
        let mu = d[p] - e[p - 1] * e[p - 1] / (dd + dd.signum() * (dd * dd + e[p - 1] * e[p - 1]).sqrt());

        // QR step with Givens
        let mut x = d[0] - mu;
        let mut zz = e[0];
        for k in 0..(p) {
            let r = (x * x + zz * zz).sqrt();
            let c = if r > 1e-30 { x / r } else { 1.0 };
            let s = if r > 1e-30 { -zz / r } else { 0.0 };

            let d0 = d[k];
            let d1 = d[k + 1];
            let e0 = e[k];

            d[k] = c * c * d0 - 2.0 * c * s * e0 + s * s * d1;
            d[k + 1] = s * s * d0 + 2.0 * c * s * e0 + c * c * d1;
            e[k] = c * s * (d0 - d1) + (c * c - s * s) * e0;

            if k + 1 < n - 1 {
                let e_next = e[k + 1];
                e[k + 1] = c * e_next;
                zz = -s * e_next;
            }
            x = e[k];
            if k + 1 < p { zz = if k + 2 <= n - 1 { -s * e[k + 1] } else { 0.0 }; }
            // Actually need to track properly
            if k + 1 < p {
                x = e[k + 1];
            }

            // Update eigenvectors
            for j in 0..n {
                let a = z[j][k];
                let b = z[j][k + 1];
                z[j][k] = c * a - s * b;
                z[j][k + 1] = s * a + c * b;
            }
        }
    }
}

/// Gauss-Hermite quadrature (for integrals ∫ f(x) exp(-x²) dx)
pub fn gauss_hermite<F: Fn(f64) -> f64>(f: F, n: usize) -> f64 {
    let (nodes, weights) = gauss_hermite_nodes_weights(n);
    let mut sum = 0.0;
    for i in 0..n { sum += weights[i] * f(nodes[i]); }
    sum
}

fn gauss_hermite_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d = vec![0.0; n];
    let mut e: Vec<f64> = (1..n).map(|i| (i as f64 / 2.0).sqrt()).collect();
    let mut z = vec![vec![0.0; n]; n];
    for i in 0..n { z[i][i] = 1.0; }
    implicit_qr_tridiag(&mut d, &mut e, &mut z, n);
    let weights: Vec<f64> = (0..n).map(|i| PI.sqrt() * z[0][i] * z[0][i]).collect();
    (d, weights)
}

/// Gauss-Laguerre quadrature (for integrals ∫₀^∞ f(x) exp(-x) dx)
pub fn gauss_laguerre<F: Fn(f64) -> f64>(f: F, n: usize) -> f64 {
    let (nodes, weights) = gauss_laguerre_nodes_weights(n);
    let mut sum = 0.0;
    for i in 0..n { sum += weights[i] * f(nodes[i]); }
    sum
}

fn gauss_laguerre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut d: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 1.0).collect();
    let mut e: Vec<f64> = (1..n).map(|i| i as f64).collect();
    let mut z = vec![vec![0.0; n]; n];
    for i in 0..n { z[i][i] = 1.0; }
    implicit_qr_tridiag(&mut d, &mut e, &mut z, n);
    let weights: Vec<f64> = (0..n).map(|i| z[0][i] * z[0][i]).collect();
    (d, weights)
}

/// Adaptive Simpson integration
pub fn adaptive_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
    let mid = 0.5 * (a + b);
    let h = b - a;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let whole = h / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(f, a, b, fa, fb, fm, whole, tol, 20)
}

fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F, a: f64, b: f64, fa: f64, fb: f64, fm: f64, whole: f64, tol: f64, depth: usize,
) -> f64 {
    let mid = 0.5 * (a + b);
    let lm = 0.5 * (a + mid);
    let rm = 0.5 * (mid + b);
    let flm = f(lm);
    let frm = f(rm);
    let h = b - a;
    let left = h / 12.0 * (fa + 4.0 * flm + fm);
    let right = h / 12.0 * (fm + 4.0 * frm + fb);
    let refined = left + right;
    if depth <= 0 || (refined - whole).abs() < 15.0 * tol {
        return refined + (refined - whole) / 15.0;
    }
    adaptive_simpson_rec(f, a, mid, fa, fm, flm, left, tol / 2.0, depth - 1)
        + adaptive_simpson_rec(f, mid, b, fm, fb, frm, right, tol / 2.0, depth - 1)
}

/// Romberg integration
pub fn romberg<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, max_steps: usize, tol: f64) -> f64 {
    let mut r = vec![vec![0.0; max_steps]; max_steps];
    let h = b - a;
    r[0][0] = 0.5 * h * (f(a) + f(b));

    for i in 1..max_steps {
        let h_i = h / (1u64 << i) as f64;
        let mut sum = 0.0;
        for k in 0..(1u64 << (i - 1)) {
            sum += f(a + (2 * k + 1) as f64 * h_i);
        }
        r[i][0] = 0.5 * r[i - 1][0] + h_i * sum;

        for j in 1..=i {
            let factor = (4u64.pow(j as u32)) as f64;
            r[i][j] = (factor * r[i][j - 1] - r[i - 1][j - 1]) / (factor - 1.0);
        }

        if i > 0 && (r[i][i] - r[i - 1][i - 1]).abs() < tol {
            return r[i][i];
        }
    }
    r[max_steps - 1][max_steps - 1]
}

/// Double integral ∫∫ f(x,y) dy dx over [ax,bx] × [ay,by]
pub fn double_integral<F: Fn(f64, f64) -> f64>(
    f: F, ax: f64, bx: f64, ay: f64, by: f64, nx: usize, ny: usize,
) -> f64 {
    let hx = (bx - ax) / nx as f64;
    let hy = (by - ay) / ny as f64;
    let mut sum = 0.0;
    for i in 0..nx {
        let x = ax + (i as f64 + 0.5) * hx;
        for j in 0..ny {
            let y = ay + (j as f64 + 0.5) * hy;
            sum += f(x, y);
        }
    }
    sum * hx * hy
}

/// Monte Carlo integration
pub fn monte_carlo_integrate<F: Fn(&[f64]) -> f64>(
    f: F, bounds: &[(f64, f64)], n_samples: usize, seed: u64,
) -> (f64, f64) {
    let dim = bounds.len();
    let mut rng = seed;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let vol: f64 = bounds.iter().map(|(a, b)| b - a).product();

    for _ in 0..n_samples {
        let point: Vec<f64> = bounds.iter().map(|&(a, b)| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (rng >> 33) as f64 / u32::MAX as f64;
            a + u * (b - a)
        }).collect();
        let val = f(&point);
        sum += val;
        sum2 += val * val;
    }
    let mean = sum / n_samples as f64;
    let var = sum2 / n_samples as f64 - mean * mean;
    let estimate = mean * vol;
    let error = (var / n_samples as f64).sqrt() * vol;
    (estimate, error)
}

// ============================================================
// Numerical Differentiation
// ============================================================

/// Forward difference derivative
pub fn deriv_forward<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x)) / h
}

/// Central difference derivative
pub fn deriv_central<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative via central difference
pub fn deriv2_central<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Richardson extrapolation for derivative
pub fn deriv_richardson<F: Fn(f64) -> f64>(f: F, x: f64, h: f64, order: usize) -> f64 {
    let mut d = vec![vec![0.0; order + 1]; order + 1];
    for i in 0..=order {
        let hi = h / (2u64.pow(i as u32) as f64);
        d[i][0] = (f(x + hi) - f(x - hi)) / (2.0 * hi);
    }
    for j in 1..=order {
        for i in j..=order {
            let factor = 4u64.pow(j as u32) as f64;
            d[i][j] = (factor * d[i][j - 1] - d[i - 1][j - 1]) / (factor - 1.0);
        }
    }
    d[order][order]
}

/// Five-point stencil derivative
pub fn deriv_5point<F: Fn(f64) -> f64>(f: F, x: f64, h: f64) -> f64 {
    (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
}

/// Partial derivative ∂f/∂xᵢ via central differences
pub fn partial_deriv<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], i: usize, h: f64) -> f64 {
    let mut xp = x.to_vec();
    let mut xm = x.to_vec();
    xp[i] += h;
    xm[i] -= h;
    (f(&xp) - f(&xm)) / (2.0 * h)
}

/// Gradient via central differences
pub fn gradient<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<f64> {
    (0..x.len()).map(|i| partial_deriv(f, x, i, h)).collect()
}

/// Jacobian of vector-valued function
pub fn jacobian<F: Fn(&[f64]) -> Vec<f64>>(f: &F, x: &[f64], h: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let f0 = f(x);
    let m = f0.len();
    let mut jac = vec![vec![0.0; n]; m];
    for j in 0..n {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[j] += h;
        xm[j] -= h;
        let fp = f(&xp);
        let fm = f(&xm);
        for i in 0..m {
            jac[i][j] = (fp[i] - fm[i]) / (2.0 * h);
        }
    }
    jac
}

/// Hessian of scalar function
pub fn hessian<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let mut hess = vec![vec![0.0; n]; n];
    let fx = f(x);
    for i in 0..n {
        for j in i..n {
            let mut xpp = x.to_vec(); xpp[i] += h; xpp[j] += h;
            let mut xpm = x.to_vec(); xpm[i] += h; xpm[j] -= h;
            let mut xmp = x.to_vec(); xmp[i] -= h; xmp[j] += h;
            let mut xmm = x.to_vec(); xmm[i] -= h; xmm[j] -= h;
            hess[i][j] = (f(&xpp) - f(&xpm) - f(&xmp) + f(&xmm)) / (4.0 * h * h);
            hess[j][i] = hess[i][j];
        }
    }
    hess
}

/// Laplacian of scalar function
pub fn laplacian<F: Fn(&[f64]) -> f64>(f: &F, x: &[f64], h: f64) -> f64 {
    let n = x.len();
    let fx = f(x);
    let mut lap = 0.0;
    for i in 0..n {
        let mut xp = x.to_vec(); xp[i] += h;
        let mut xm = x.to_vec(); xm[i] -= h;
        lap += (f(&xp) - 2.0 * fx + f(&xm)) / (h * h);
    }
    lap
}

// ============================================================
// Root Finding
// ============================================================

/// Bisection method
pub fn bisection<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut fa = f(a);
    if fa * f(b) > 0.0 { return None; }
    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        if (b - a) < tol { return Some(mid); }
        let fm = f(mid);
        if fm * fa < 0.0 { b = mid; } else { a = mid; fa = fm; }
    }
    Some(0.5 * (a + b))
}

/// Newton-Raphson method
pub fn newton<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(
    f: F, df: G, x0: f64, tol: f64, max_iter: usize,
) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol { return Some(x); }
        let dfx = df(x);
        if dfx.abs() < 1e-30 { return None; }
        x -= fx / dfx;
    }
    if f(x).abs() < tol * 100.0 { Some(x) } else { None }
}

/// Secant method
pub fn secant<F: Fn(f64) -> f64>(f: F, x0: f64, x1: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut xm1 = x0;
    let mut x = x1;
    let mut fm1 = f(xm1);
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol { return Some(x); }
        let denom = fx - fm1;
        if denom.abs() < 1e-30 { return None; }
        let x_new = x - fx * (x - xm1) / denom;
        xm1 = x;
        fm1 = fx;
        x = x_new;
    }
    if f(x).abs() < tol * 100.0 { Some(x) } else { None }
}

/// Brent's method for root finding
pub fn brent<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa * fb > 0.0 { return None; }
    if fa.abs() < fb.abs() { std::mem::swap(&mut a, &mut b); std::mem::swap(&mut fa, &mut fb); }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let mut mflag = true;

    for _ in 0..max_iter {
        if fb.abs() < tol { return Some(b); }
        if fa.abs() < tol { return Some(a); }

        let s;
        if (fa - fc).abs() > 1e-15 && (fb - fc).abs() > 1e-15 {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }

        let cond1 = !(s > ((3.0 * a + b) / 4.0).min(b) && s < ((3.0 * a + b) / 4.0).max(b));
        let cond2 = mflag && (s - b).abs() >= (b - c).abs() / 2.0;
        let cond3 = !mflag && (s - b).abs() >= (c - d).abs() / 2.0;
        let cond4 = mflag && (b - c).abs() < tol;
        let cond5 = !mflag && (c - d).abs() < tol;

        if cond1 || cond2 || cond3 || cond4 || cond5 {
            let s_new = 0.5 * (a + b);
            mflag = true;
            d = b - a;
            e = d;
            let _ = s;
            let fs = f(s_new);
            c = b; fc = fb;
            if fa * fs < 0.0 { b = s_new; fb = fs; }
            else { a = s_new; fa = fs; }
        } else {
            mflag = false;
            d = e;
            e = b - a;
            let fs = f(s);
            c = b; fc = fb;
            if fa * fs < 0.0 { b = s; fb = fs; }
            else { a = s; fa = fs; }
        }

        if fa.abs() < fb.abs() { std::mem::swap(&mut a, &mut b); std::mem::swap(&mut fa, &mut fb); }
    }
    Some(b)
}

/// Ridder's method
pub fn ridder<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut fa = f(a);
    let mut fb = f(b);
    if fa * fb > 0.0 { return None; }
    for _ in 0..max_iter {
        let mid = 0.5 * (a + b);
        let fm = f(mid);
        let s = (fm * fm - fa * fb).sqrt();
        if s < 1e-30 { return Some(mid); }
        let sign = if (fa - fb) > 0.0 { 1.0 } else { -1.0 };
        let x_new = mid + (mid - a) * sign * fm / s;
        let fx = f(x_new);
        if fx.abs() < tol { return Some(x_new); }
        if fm * fx < 0.0 {
            a = mid; fa = fm;
            b = x_new; fb = fx;
        } else if fa * fx < 0.0 {
            b = x_new; fb = fx;
        } else {
            a = x_new; fa = fx;
        }
        if (b - a).abs() < tol { return Some(0.5 * (a + b)); }
    }
    Some(0.5 * (a + b))
}

/// Multi-dimensional Newton-Raphson
pub fn newton_multi<F, J>(
    f: F, jac: J, x0: &[f64], tol: f64, max_iter: usize,
) -> Option<Vec<f64>>
where
    F: Fn(&[f64]) -> Vec<f64>,
    J: Fn(&[f64]) -> Vec<Vec<f64>>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    for _ in 0..max_iter {
        let fx = f(&x);
        let norm: f64 = fx.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < tol { return Some(x); }
        let j = jac(&x);
        let mat = crate::linear_algebra::Matrix::from_rows(&j);
        let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = mat.solve(&neg_fx);
        for i in 0..n { x[i] += dx[i]; }
    }
    Some(x)
}

// ============================================================
// ODE Solvers
// ============================================================

/// ODE solution: list of (t, y) pairs
pub type OdeSolution = Vec<(f64, Vec<f64>)>;

/// Forward Euler method
pub fn euler_method<F: Fn(f64, &[f64]) -> Vec<f64>>(
    f: F, t0: f64, y0: &[f64], t_end: f64, dt: f64,
) -> OdeSolution {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut sol = vec![(t, y.clone())];
    while t < t_end - 1e-12 {
        let h = dt.min(t_end - t);
        let dy = f(t, &y);
        for i in 0..n { y[i] += h * dy[i]; }
        t += h;
        sol.push((t, y.clone()));
    }
    sol
}

/// Improved Euler (Heun's method)
pub fn heun_method<F: Fn(f64, &[f64]) -> Vec<f64>>(
    f: F, t0: f64, y0: &[f64], t_end: f64, dt: f64,
) -> OdeSolution {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut sol = vec![(t, y.clone())];
    while t < t_end - 1e-12 {
        let h = dt.min(t_end - t);
        let k1 = f(t, &y);
        let y_euler: Vec<f64> = (0..n).map(|i| y[i] + h * k1[i]).collect();
        let k2 = f(t + h, &y_euler);
        for i in 0..n { y[i] += 0.5 * h * (k1[i] + k2[i]); }
        t += h;
        sol.push((t, y.clone()));
    }
    sol
}

/// Classical 4th-order Runge-Kutta
pub fn rk4<F: Fn(f64, &[f64]) -> Vec<f64>>(
    f: F, t0: f64, y0: &[f64], t_end: f64, dt: f64,
) -> OdeSolution {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut sol = vec![(t, y.clone())];
    while t < t_end - 1e-12 {
        let h = dt.min(t_end - t);
        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
        let k2 = f(t + 0.5 * h, &y2);
        let y3: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k2[i]).collect();
        let k3 = f(t + 0.5 * h, &y3);
        let y4: Vec<f64> = (0..n).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(t + h, &y4);
        for i in 0..n {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        sol.push((t, y.clone()));
    }
    sol
}

/// Dormand-Prince (RK45) adaptive step-size solver
pub fn rk45<F: Fn(f64, &[f64]) -> Vec<f64>>(
    f: F, t0: f64, y0: &[f64], t_end: f64, rtol: f64, atol: f64,
    h_init: f64, h_min: f64, h_max: f64,
) -> OdeSolution {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = h_init;
    let mut sol = vec![(t, y.clone())];

    // Dormand-Prince coefficients
    let a = [
        [0.0; 7],
        [1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0, 0.0],
        [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0, 0.0],
        [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0, 0.0, 0.0],
        [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0],
    ];
    let b5 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0];
    let b4 = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0];
    let c = [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0];

    let max_steps = 100_000;
    for _ in 0..max_steps {
        if t >= t_end - 1e-15 { break; }
        h = h.min(t_end - t).min(h_max).max(h_min);

        let mut k = vec![vec![0.0; n]; 7];
        k[0] = f(t, &y);
        for s in 1..7 {
            let ts = t + c[s] * h;
            let ys: Vec<f64> = (0..n).map(|i| {
                let mut v = y[i];
                for j in 0..s { v += h * a[s][j] * k[j][i]; }
                v
            }).collect();
            k[s] = f(ts, &ys);
        }

        // 5th order solution
        let y5: Vec<f64> = (0..n).map(|i| {
            let mut v = y[i];
            for s in 0..7 { v += h * b5[s] * k[s][i]; }
            v
        }).collect();

        // 4th order solution for error estimate
        let y4: Vec<f64> = (0..n).map(|i| {
            let mut v = y[i];
            for s in 0..7 { v += h * b4[s] * k[s][i]; }
            v
        }).collect();

        // Error
        let err: f64 = (0..n).map(|i| {
            let scale = atol + rtol * y[i].abs().max(y5[i].abs());
            ((y5[i] - y4[i]) / scale).powi(2)
        }).sum::<f64>().sqrt() / (n as f64).sqrt();

        if err <= 1.0 || h <= h_min * 1.01 {
            t += h;
            y = y5;
            sol.push((t, y.clone()));
            // Increase step
            if err > 1e-10 {
                h *= (0.9 * (1.0 / err).powf(0.2)).min(5.0);
            } else {
                h *= 5.0;
            }
        } else {
            // Decrease step
            h *= (0.9 * (1.0 / err).powf(0.25)).max(0.1);
        }
    }
    sol
}

/// Implicit Euler (backward Euler) for stiff systems — 1D only for simplicity
pub fn backward_euler_1d<F: Fn(f64, f64) -> f64>(
    f: F, t0: f64, y0: f64, t_end: f64, dt: f64,
) -> Vec<(f64, f64)> {
    let mut t = t0;
    let mut y = y0;
    let mut sol = vec![(t, y)];
    while t < t_end - 1e-12 {
        let h = dt.min(t_end - t);
        // Newton iteration: y_new = y + h * f(t+h, y_new)
        let mut yn = y + h * f(t, y); // initial guess
        for _ in 0..20 {
            let fn_val = f(t + h, yn);
            let g = yn - y - h * fn_val;
            // Approximate Jacobian
            let eps = 1e-8;
            let dg = 1.0 - h * (f(t + h, yn + eps) - fn_val) / eps;
            if dg.abs() < 1e-30 { break; }
            let dy = g / dg;
            yn -= dy;
            if dy.abs() < 1e-12 { break; }
        }
        y = yn;
        t += h;
        sol.push((t, y));
    }
    sol
}

/// Midpoint method
pub fn midpoint<F: Fn(f64, &[f64]) -> Vec<f64>>(
    f: F, t0: f64, y0: &[f64], t_end: f64, dt: f64,
) -> OdeSolution {
    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut sol = vec![(t, y.clone())];
    while t < t_end - 1e-12 {
        let h = dt.min(t_end - t);
        let k1 = f(t, &y);
        let ym: Vec<f64> = (0..n).map(|i| y[i] + 0.5 * h * k1[i]).collect();
        let k2 = f(t + 0.5 * h, &ym);
        for i in 0..n { y[i] += h * k2[i]; }
        t += h;
        sol.push((t, y.clone()));
    }
    sol
}

// ============================================================
// Fast Fourier Transform
// ============================================================

/// Complex number (simple pair)
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub fn zero() -> Self { Self { re: 0.0, im: 0.0 } }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }
    pub fn mag(&self) -> f64 { (self.re * self.re + self.im * self.im).sqrt() }
    pub fn phase(&self) -> f64 { self.im.atan2(self.re) }
    pub fn conj(&self) -> Self { Self { re: self.re, im: -self.im } }
    pub fn add(&self, other: &Complex) -> Complex {
        Complex { re: self.re + other.re, im: self.im + other.im }
    }
    pub fn sub(&self, other: &Complex) -> Complex {
        Complex { re: self.re - other.re, im: self.im - other.im }
    }
    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    pub fn scale(&self, s: f64) -> Complex {
        Complex { re: self.re * s, im: self.im * s }
    }
    pub fn div(&self, other: &Complex) -> Complex {
        let d = other.re * other.re + other.im * other.im;
        Complex {
            re: (self.re * other.re + self.im * other.im) / d,
            im: (self.im * other.re - self.re * other.im) / d,
        }
    }
}

/// FFT (Cooley-Tukey radix-2 DIT). Input length must be power of 2.
pub fn fft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    assert!(n.is_power_of_two(), "FFT input length must be power of 2");
    if n == 1 { return input.to_vec(); }

    let mut x = bit_reverse_copy(input);
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let wn = Complex::from_polar(1.0, angle);
        let half = len / 2;
        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let u = x[start + j];
                let t = w.mul(&x[start + j + half]);
                x[start + j] = u.add(&t);
                x[start + j + half] = u.sub(&t);
                w = w.mul(&wn);
            }
        }
        len *= 2;
    }
    x
}

/// Inverse FFT
pub fn ifft(input: &[Complex]) -> Vec<Complex> {
    let n = input.len();
    let conj_input: Vec<Complex> = input.iter().map(|c| c.conj()).collect();
    let mut result = fft(&conj_input);
    let scale = 1.0 / n as f64;
    for c in &mut result {
        c.re *= scale;
        c.im = -c.im * scale;
    }
    result
}

fn bit_reverse_copy(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    let bits = (n as f64).log2() as u32;
    let mut result = vec![Complex::zero(); n];
    for i in 0..n {
        let rev = bit_reverse(i as u32, bits) as usize;
        result[rev] = x[i];
    }
    result
}

fn bit_reverse(mut x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Real-valued FFT (returns N/2+1 complex values)
pub fn rfft(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    let complex_input: Vec<Complex> = input.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let result = fft(&complex_input);
    result[..n / 2 + 1].to_vec()
}

/// Inverse real FFT
pub fn irfft(input: &[Complex], n: usize) -> Vec<f64> {
    let mut full = Vec::with_capacity(n);
    for c in input { full.push(*c); }
    for i in 1..n / 2 {
        full.push(input[n / 2 - i].conj());
    }
    while full.len() < n { full.push(Complex::zero()); }
    let result = ifft(&full);
    result.iter().map(|c| c.re).collect()
}

/// Power spectrum from FFT
pub fn power_spectrum(data: &[f64]) -> Vec<f64> {
    let fft_result = rfft(data);
    fft_result.iter().map(|c| c.mag() * c.mag()).collect()
}

/// Convolution via FFT
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = (a.len() + b.len() - 1).next_power_of_two();
    let mut ca: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut cb: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    ca.resize(n, Complex::zero());
    cb.resize(n, Complex::zero());
    let fa = fft(&ca);
    let fb = fft(&cb);
    let fc: Vec<Complex> = fa.iter().zip(&fb).map(|(a, b)| a.mul(b)).collect();
    let result = ifft(&fc);
    result[..a.len() + b.len() - 1].iter().map(|c| c.re).collect()
}

/// Cross-correlation via FFT
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = (a.len() + b.len() - 1).next_power_of_two();
    let mut ca: Vec<Complex> = a.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let mut cb: Vec<Complex> = b.iter().map(|&x| Complex::new(x, 0.0)).collect();
    ca.resize(n, Complex::zero());
    cb.resize(n, Complex::zero());
    let fa = fft(&ca);
    let fb = fft(&cb);
    let fc: Vec<Complex> = fa.iter().zip(&fb).map(|(a, b)| a.mul(&b.conj())).collect();
    let result = ifft(&fc);
    result.iter().map(|c| c.re).collect()
}

/// Discrete cosine transform (DCT-II)
pub fn dct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut result = vec![0.0; n];
    for k in 0..n {
        let mut sum = 0.0;
        for i in 0..n {
            sum += input[i] * (PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos();
        }
        result[k] = sum;
    }
    result
}

/// Inverse DCT (DCT-III)
pub fn idct(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.5 * input[0];
        for k in 1..n {
            sum += input[k] * (PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos();
        }
        result[i] = 2.0 * sum / n as f64;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpson() {
        // ∫₀¹ x² dx = 1/3
        let result = simpson(|x| x * x, 0.0, 1.0, 100);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bisection() {
        let root = bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-10, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    #[test]
    fn test_newton() {
        let root = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.5, 1e-12, 50).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_rk4_exponential() {
        // y' = y, y(0) = 1 => y = e^t
        let sol = rk4(|_t, y| vec![y[0]], 0.0, &[1.0], 1.0, 0.01);
        let (t, y) = sol.last().unwrap();
        assert!((y[0] - std::f64::consts::E).abs() < 1e-6);
    }

    #[test]
    fn test_fft() {
        let n = 8;
        let input: Vec<Complex> = (0..n).map(|i| {
            Complex::new((2.0 * PI * i as f64 / n as f64).cos(), 0.0)
        }).collect();
        let result = fft(&input);
        // For cos, should have peaks at k=1 and k=n-1
        assert!(result[1].mag() > 1.0);
    }

    #[test]
    fn test_fft_inverse() {
        let input: Vec<Complex> = vec![
            Complex::new(1.0, 0.0), Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0), Complex::new(4.0, 0.0),
        ];
        let fwd = fft(&input);
        let inv = ifft(&fwd);
        for i in 0..4 {
            assert!((inv[i].re - input[i].re).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adaptive_simpson() {
        let result = adaptive_simpson(&|x: f64| x.sin(), 0.0, PI, 1e-10);
        assert!((result - 2.0).abs() < 1e-8);
    }
}
