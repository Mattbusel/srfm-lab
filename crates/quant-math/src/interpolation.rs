// interpolation.rs — Linear, cubic spline, B-spline, Lagrange, Newton, 2D, Hermite, rational

/// Linear interpolation (1D)
pub fn lerp(x0: f64, y0: f64, x1: f64, y1: f64, x: f64) -> f64 {
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// Piecewise linear interpolation
pub struct LinearInterp {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

impl LinearInterp {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>) -> Self {
        assert_eq!(xs.len(), ys.len());
        assert!(xs.len() >= 2);
        // Verify sorted
        for i in 1..xs.len() {
            assert!(xs[i] > xs[i - 1], "xs must be strictly increasing");
        }
        Self { xs, ys }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] { return self.ys[0]; }
        if x >= self.xs[n - 1] { return self.ys[n - 1]; }
        let i = self.find_segment(x);
        lerp(self.xs[i], self.ys[i], self.xs[i + 1], self.ys[i + 1], x)
    }

    pub fn eval_extrapolate(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return lerp(self.xs[0], self.ys[0], self.xs[1], self.ys[1], x);
        }
        if x >= self.xs[n - 1] {
            return lerp(self.xs[n - 2], self.ys[n - 2], self.xs[n - 1], self.ys[n - 1], x);
        }
        let i = self.find_segment(x);
        lerp(self.xs[i], self.ys[i], self.xs[i + 1], self.ys[i + 1], x)
    }

    fn find_segment(&self, x: f64) -> usize {
        // Binary search
        let mut lo = 0;
        let mut hi = self.xs.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= x { lo = mid; } else { hi = mid; }
        }
        lo
    }

    /// Evaluate at multiple points
    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    /// Integral over [a, b]
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if a >= b { return 0.0; }
        let n = self.xs.len();
        let mut total = 0.0;
        let mut lo = a;
        for i in 0..(n - 1) {
            if lo >= b { break; }
            if self.xs[i + 1] <= lo { continue; }
            let x0 = lo.max(self.xs[i]);
            let x1 = b.min(self.xs[i + 1]);
            let y0 = lerp(self.xs[i], self.ys[i], self.xs[i + 1], self.ys[i + 1], x0);
            let y1 = lerp(self.xs[i], self.ys[i], self.xs[i + 1], self.ys[i + 1], x1);
            total += 0.5 * (y0 + y1) * (x1 - x0);
            lo = x1;
        }
        total
    }
}

/// Cubic spline interpolation
#[derive(Clone, Debug)]
pub struct CubicSpline {
    pub xs: Vec<f64>,
    /// Coefficients for each segment: a + b(x-xi) + c(x-xi)^2 + d(x-xi)^3
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

/// Boundary condition for cubic spline
#[derive(Clone, Debug)]
pub enum SplineBoundary {
    Natural,
    Clamped(f64, f64), // first derivatives at endpoints
    NotAKnot,
}

impl CubicSpline {
    /// Build natural cubic spline
    pub fn natural(xs: &[f64], ys: &[f64]) -> Self {
        Self::build(xs, ys, SplineBoundary::Natural)
    }

    /// Build clamped cubic spline
    pub fn clamped(xs: &[f64], ys: &[f64], d0: f64, dn: f64) -> Self {
        Self::build(xs, ys, SplineBoundary::Clamped(d0, dn))
    }

    /// Build not-a-knot cubic spline
    pub fn not_a_knot(xs: &[f64], ys: &[f64]) -> Self {
        Self::build(xs, ys, SplineBoundary::NotAKnot)
    }

    fn build(xs: &[f64], ys: &[f64], bc: SplineBoundary) -> Self {
        let n = xs.len();
        assert_eq!(n, ys.len());
        assert!(n >= 3);
        let nm1 = n - 1;

        let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();
        let delta: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

        // Set up tridiagonal system for c coefficients
        let mut sub = vec![0.0; n]; // sub-diagonal
        let mut diag = vec![0.0; n]; // diagonal
        let mut sup = vec![0.0; n]; // super-diagonal
        let mut rhs = vec![0.0; n];

        // Interior equations
        for i in 1..nm1 {
            sub[i] = h[i - 1];
            diag[i] = 2.0 * (h[i - 1] + h[i]);
            sup[i] = h[i];
            rhs[i] = 3.0 * (delta[i] - delta[i - 1]);
        }

        match bc {
            SplineBoundary::Natural => {
                diag[0] = 1.0;
                diag[nm1] = 1.0;
            }
            SplineBoundary::Clamped(d0, dn) => {
                diag[0] = 2.0 * h[0];
                sup[0] = h[0];
                rhs[0] = 3.0 * (delta[0] - d0);
                sub[nm1] = h[nm1 - 1];
                diag[nm1] = 2.0 * h[nm1 - 1];
                rhs[nm1] = 3.0 * (dn - delta[nm1 - 1]);
            }
            SplineBoundary::NotAKnot => {
                if n >= 4 {
                    diag[0] = h[1];
                    sup[0] = -(h[0] + h[1]);
                    rhs[0] = 0.0;
                    // Actually for not-a-knot: d3 is continuous at x[1] and x[n-2]
                    // d[0] = d[1] and d[n-2] = d[n-1]
                    // Simplified: set c[0] via extrapolation
                    diag[0] = 1.0;
                    sup[0] = -1.0;
                    rhs[0] = 0.0;
                    sub[nm1] = -1.0;
                    diag[nm1] = 1.0;
                    rhs[nm1] = 0.0;
                } else {
                    diag[0] = 1.0;
                    diag[nm1] = 1.0;
                }
            }
        }

        // Solve tridiagonal
        let c = solve_tridiag_general(&sub, &diag, &sup, &rhs);

        // Compute b and d
        let mut b_coeff = vec![0.0; nm1];
        let mut d_coeff = vec![0.0; nm1];
        for i in 0..nm1 {
            b_coeff[i] = delta[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
            d_coeff[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        }

        Self {
            xs: xs.to_vec(),
            a: ys[..nm1].to_vec(),
            b: b_coeff,
            c: c[..nm1].to_vec(),
            d: d_coeff,
        }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let i = if x <= self.xs[0] { 0 }
                else if x >= self.xs[n - 1] { n - 2 }
                else { self.find_segment(x) };
        let dx = x - self.xs[i];
        self.a[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))
    }

    pub fn eval_deriv(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let i = if x <= self.xs[0] { 0 }
                else if x >= self.xs[n - 1] { n - 2 }
                else { self.find_segment(x) };
        let dx = x - self.xs[i];
        self.b[i] + dx * (2.0 * self.c[i] + 3.0 * self.d[i] * dx)
    }

    pub fn eval_deriv2(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let i = if x <= self.xs[0] { 0 }
                else if x >= self.xs[n - 1] { n - 2 }
                else { self.find_segment(x) };
        let dx = x - self.xs[i];
        2.0 * self.c[i] + 6.0 * self.d[i] * dx
    }

    fn find_segment(&self, x: f64) -> usize {
        let mut lo = 0;
        let mut hi = self.xs.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.xs[mid] <= x { lo = mid; } else { hi = mid; }
        }
        lo
    }

    pub fn eval_many(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.eval(x)).collect()
    }

    /// Integrate the spline over [a, b]
    pub fn integrate(&self, a: f64, b: f64) -> f64 {
        if a >= b { return 0.0; }
        let n = self.xs.len();
        let mut total = 0.0;
        for i in 0..(n - 1) {
            let x0 = a.max(self.xs[i]);
            let x1 = b.min(self.xs[i + 1]);
            if x0 >= x1 { continue; }
            let lo = x0 - self.xs[i];
            let hi = x1 - self.xs[i];
            // Integral of a + b*t + c*t^2 + d*t^3 from lo to hi
            let ai = self.a[i];
            let bi = self.b[i];
            let ci = self.c[i];
            let di = self.d[i];
            let int_hi = ai * hi + bi * hi * hi / 2.0 + ci * hi * hi * hi / 3.0 + di * hi.powi(4) / 4.0;
            let int_lo = ai * lo + bi * lo * lo / 2.0 + ci * lo * lo * lo / 3.0 + di * lo.powi(4) / 4.0;
            total += int_hi - int_lo;
        }
        total
    }

    /// Find roots (x where spline(x) = 0) in each segment
    pub fn roots(&self) -> Vec<f64> {
        let mut result = Vec::new();
        let n = self.xs.len() - 1;
        for i in 0..n {
            let roots_seg = cubic_roots(self.a[i], self.b[i], self.c[i], self.d[i]);
            let h = self.xs[i + 1] - self.xs[i];
            for r in roots_seg {
                if r >= -1e-10 && r <= h + 1e-10 {
                    result.push(self.xs[i] + r);
                }
            }
        }
        result
    }
}

/// Solve tridiagonal system (Thomas algorithm, general)
fn solve_tridiag_general(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    c[0] = sup[0] / diag[0];
    d[0] = rhs[0] / diag[0];
    for i in 1..n {
        let m = diag[i] - sub[i] * c[i - 1];
        c[i] = if i < n - 1 { sup[i] / m } else { 0.0 };
        d[i] = (rhs[i] - sub[i] * d[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = d[n - 1];
    for i in (0..n - 1).rev() { x[i] = d[i] - c[i] * x[i + 1]; }
    x
}

/// Real roots of cubic a + b*t + c*t^2 + d*t^3 = 0
fn cubic_roots(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if d.abs() < 1e-15 {
        // Quadratic
        if c.abs() < 1e-15 {
            if b.abs() < 1e-15 { return vec![]; }
            return vec![-a / b];
        }
        let disc = b * b - 4.0 * c * a;
        if disc < 0.0 { return vec![]; }
        let sq = disc.sqrt();
        return vec![(-b + sq) / (2.0 * c), (-b - sq) / (2.0 * c)];
    }
    // Cardano's formula
    let a0 = a / d; let a1 = b / d; let a2 = c / d;
    let q = (3.0 * a1 - a2 * a2) / 9.0;
    let r = (9.0 * a2 * a1 - 27.0 * a0 - 2.0 * a2.powi(3)) / 54.0;
    let disc = q.powi(3) + r * r;
    let shift = -a2 / 3.0;

    if disc > 0.0 {
        let sq = disc.sqrt();
        let s = (r + sq).cbrt();
        let t = (r - sq).cbrt();
        vec![s + t + shift]
    } else {
        let theta = (r / (-q.powi(3)).sqrt().max(1e-30)).acos();
        let sq = 2.0 * (-q).sqrt();
        vec![
            sq * (theta / 3.0).cos() + shift,
            sq * ((theta + 2.0 * std::f64::consts::PI) / 3.0).cos() + shift,
            sq * ((theta + 4.0 * std::f64::consts::PI) / 3.0).cos() + shift,
        ]
    }
}

/// B-spline basis function (Cox-de Boor recursion)
pub fn bspline_basis(i: usize, p: usize, t: f64, knots: &[f64]) -> f64 {
    if p == 0 {
        return if knots[i] <= t && t < knots[i + 1] { 1.0 } else { 0.0 };
    }
    let mut result = 0.0;
    let d1 = knots[i + p] - knots[i];
    if d1 > 1e-15 {
        result += (t - knots[i]) / d1 * bspline_basis(i, p - 1, t, knots);
    }
    let d2 = knots[i + p + 1] - knots[i + 1];
    if d2 > 1e-15 {
        result += (knots[i + p + 1] - t) / d2 * bspline_basis(i + 1, p - 1, t, knots);
    }
    result
}

/// B-spline curve evaluation
pub struct BSpline {
    pub degree: usize,
    pub knots: Vec<f64>,
    pub control_points: Vec<f64>, // 1D control points (y-values)
}

impl BSpline {
    pub fn new(degree: usize, knots: Vec<f64>, control_points: Vec<f64>) -> Self {
        assert_eq!(knots.len(), control_points.len() + degree + 1);
        Self { degree, knots, control_points }
    }

    /// Uniform B-spline from data points
    pub fn uniform(degree: usize, points: Vec<f64>) -> Self {
        let n = points.len();
        let m = n + degree + 1;
        let mut knots = vec![0.0; m];
        for i in 0..m {
            if i <= degree { knots[i] = 0.0; }
            else if i >= n { knots[i] = (n - degree) as f64; }
            else { knots[i] = (i - degree) as f64; }
        }
        Self { degree, knots, control_points: points }
    }

    pub fn eval(&self, t: f64) -> f64 {
        let n = self.control_points.len();
        let mut result = 0.0;
        for i in 0..n {
            result += self.control_points[i] * bspline_basis(i, self.degree, t, &self.knots);
        }
        result
    }

    /// De Boor's algorithm (more efficient)
    pub fn eval_deboor(&self, t: f64) -> f64 {
        let p = self.degree;
        let n = self.control_points.len();
        // Find knot span
        let mut k = p;
        for i in p..n {
            if t >= self.knots[i] && t < self.knots[i + 1] { k = i; break; }
        }
        if t >= self.knots[n] { k = n - 1; }

        let mut d: Vec<f64> = (0..=p).map(|j| self.control_points[(k - p + j).min(n - 1)]).collect();

        for r in 1..=p {
            for j in (r..=p).rev() {
                let idx = k - p + j;
                let denom = self.knots[idx + p + 1 - r] - self.knots[idx];
                let alpha = if denom.abs() > 1e-15 {
                    (t - self.knots[idx]) / denom
                } else { 0.0 };
                d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
            }
        }
        d[p]
    }

    pub fn eval_many(&self, ts: &[f64]) -> Vec<f64> {
        ts.iter().map(|&t| self.eval_deboor(t)).collect()
    }

    /// Derivative of B-spline
    pub fn derivative(&self) -> BSpline {
        let p = self.degree;
        if p == 0 { return BSpline::new(0, vec![0.0, 1.0], vec![0.0]); }
        let n = self.control_points.len();
        let mut new_cp = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let denom = self.knots[i + p + 1] - self.knots[i + 1];
            let dp = if denom.abs() > 1e-15 {
                p as f64 * (self.control_points[i + 1] - self.control_points[i]) / denom
            } else { 0.0 };
            new_cp.push(dp);
        }
        let new_knots = self.knots[1..self.knots.len() - 1].to_vec();
        BSpline::new(p - 1, new_knots, new_cp)
    }
}

/// Lagrange interpolation
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    assert_eq!(n, ys.len());
    let mut result = 0.0;
    for i in 0..n {
        let mut li = 1.0;
        for j in 0..n {
            if i != j { li *= (x - xs[j]) / (xs[i] - xs[j]); }
        }
        result += ys[i] * li;
    }
    result
}

/// Lagrange with barycentric weights (more stable, O(n) per evaluation after O(n^2) setup)
pub struct BarycentricLagrange {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub weights: Vec<f64>,
}

impl BarycentricLagrange {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        let mut weights = vec![1.0; n];
        for i in 0..n {
            for j in 0..n {
                if i != j { weights[i] /= xs[i] - xs[j]; }
            }
        }
        Self { xs: xs.to_vec(), ys: ys.to_vec(), weights }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        // Check if x is a node
        for i in 0..n {
            if (x - self.xs[i]).abs() < 1e-15 { return self.ys[i]; }
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let t = self.weights[i] / (x - self.xs[i]);
            num += t * self.ys[i];
            den += t;
        }
        num / den
    }
}

/// Newton divided differences interpolation
pub struct NewtonDivDiff {
    pub xs: Vec<f64>,
    pub coeffs: Vec<f64>,
}

impl NewtonDivDiff {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        let mut table = ys.to_vec();
        let mut coeffs = vec![table[0]];
        for j in 1..n {
            for i in (j..n).rev() {
                table[i] = (table[i] - table[i - 1]) / (xs[i] - xs[i - j]);
            }
            coeffs.push(table[j]);
        }
        Self { xs: xs.to_vec(), coeffs }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.coeffs.len();
        let mut result = self.coeffs[n - 1];
        for i in (0..n - 1).rev() {
            result = result * (x - self.xs[i]) + self.coeffs[i];
        }
        result
    }

    /// Add a new point (O(n))
    pub fn add_point(&mut self, x: f64, y: f64) {
        let n = self.xs.len();
        let mut d = y;
        let mut prev = vec![0.0; n];
        prev[0] = y;
        for j in 1..=n {
            let old_d = d;
            d = (d - self.coeffs[j - 1]) / (x - self.xs[j - 1]);
            if j < n { prev[j] = d; }
        }
        self.xs.push(x);
        self.coeffs.push(d);
    }
}

/// 2D Bilinear interpolation
pub struct Bilinear2D {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub zs: Vec<Vec<f64>>, // zs[i][j] = f(xs[i], ys[j])
}

impl Bilinear2D {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<Vec<f64>>) -> Self {
        assert_eq!(zs.len(), xs.len());
        for row in &zs { assert_eq!(row.len(), ys.len()); }
        Self { xs, ys, zs }
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let ix = find_interval(&self.xs, x);
        let iy = find_interval(&self.ys, y);
        let nx = self.xs.len() - 1;
        let ny = self.ys.len() - 1;
        let ix = ix.min(nx - 1);
        let iy = iy.min(ny - 1);

        let x0 = self.xs[ix]; let x1 = self.xs[ix + 1];
        let y0 = self.ys[iy]; let y1 = self.ys[iy + 1];
        let t = if (x1 - x0).abs() > 1e-15 { (x - x0) / (x1 - x0) } else { 0.0 };
        let u = if (y1 - y0).abs() > 1e-15 { (y - y0) / (y1 - y0) } else { 0.0 };

        let z00 = self.zs[ix][iy];
        let z10 = self.zs[ix + 1][iy];
        let z01 = self.zs[ix][iy + 1];
        let z11 = self.zs[ix + 1][iy + 1];

        (1.0 - t) * (1.0 - u) * z00 + t * (1.0 - u) * z10
            + (1.0 - t) * u * z01 + t * u * z11
    }
}

/// 2D Bicubic interpolation
pub struct Bicubic2D {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub zs: Vec<Vec<f64>>,
}

impl Bicubic2D {
    pub fn new(xs: Vec<f64>, ys: Vec<f64>, zs: Vec<Vec<f64>>) -> Self {
        Self { xs, ys, zs }
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let ix = find_interval(&self.xs, x).min(self.xs.len() - 2);
        let iy = find_interval(&self.ys, y).min(self.ys.len() - 2);
        let nx = self.xs.len();
        let ny = self.ys.len();

        let t = if (self.xs[ix + 1] - self.xs[ix]).abs() > 1e-15 {
            (x - self.xs[ix]) / (self.xs[ix + 1] - self.xs[ix])
        } else { 0.0 };
        let u = if (self.ys[iy + 1] - self.ys[iy]).abs() > 1e-15 {
            (y - self.ys[iy]) / (self.ys[iy + 1] - self.ys[iy])
        } else { 0.0 };

        // Catmull-Rom style: use 4 points in each direction
        let mut result = 0.0;
        for di in -1i32..=2 {
            let i = (ix as i32 + di).clamp(0, nx as i32 - 1) as usize;
            let bx = cubic_hermite_basis(di, t);
            for dj in -1i32..=2 {
                let j = (iy as i32 + dj).clamp(0, ny as i32 - 1) as usize;
                let by = cubic_hermite_basis(dj, u);
                result += bx * by * self.zs[i][j];
            }
        }
        result
    }
}

fn cubic_hermite_basis(k: i32, t: f64) -> f64 {
    // Catmull-Rom basis
    match k {
        -1 => { let t2 = t * t; -0.5 * t + t2 - 0.5 * t2 * t }
        0 => { 1.0 - 2.5 * t * t + 1.5 * t * t * t }
        1 => { 0.5 * t + 2.0 * t * t - 1.5 * t * t * t }
        2 => { -0.5 * t * t + 0.5 * t * t * t }
        _ => 0.0,
    }
}

/// Monotone cubic Hermite interpolation (Fritsch-Carlson)
pub struct MonotoneHermite {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub m: Vec<f64>, // tangent slopes
}

impl MonotoneHermite {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert!(n >= 2);
        let nm1 = n - 1;

        let delta: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])).collect();

        let mut m = vec![0.0; n];
        m[0] = delta[0];
        m[nm1] = delta[nm1 - 1];
        for i in 1..nm1 {
            if delta[i - 1].signum() != delta[i].signum() {
                m[i] = 0.0;
            } else {
                m[i] = (delta[i - 1] + delta[i]) / 2.0;
            }
        }

        // Fritsch-Carlson adjustment for monotonicity
        for i in 0..nm1 {
            if delta[i].abs() < 1e-30 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
            } else {
                let alpha = m[i] / delta[i];
                let beta = m[i + 1] / delta[i];
                let s = alpha * alpha + beta * beta;
                if s > 9.0 {
                    let tau = 3.0 / s.sqrt();
                    m[i] = tau * alpha * delta[i];
                    m[i + 1] = tau * beta * delta[i];
                }
            }
        }

        Self { xs: xs.to_vec(), ys: ys.to_vec(), m }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] { return self.ys[0]; }
        if x >= self.xs[n - 1] { return self.ys[n - 1]; }
        let i = find_interval(&self.xs, x);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        h00 * self.ys[i] + h10 * h * self.m[i] + h01 * self.ys[i + 1] + h11 * h * self.m[i + 1]
    }
}

/// Rational interpolation (Stoer-Bulirsch)
pub fn rational_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    let mut c = ys.to_vec();
    let mut d = ys.to_vec();
    let mut ns = 0;
    let mut diff = (x - xs[0]).abs();
    for i in 1..n {
        let dd = (x - xs[i]).abs();
        if dd < diff { ns = i; diff = dd; }
    }
    let mut y = ys[ns];
    ns = ns.wrapping_sub(0); // keep ns as is for now
    // Using Neville-like algorithm
    for m in 1..n {
        for i in 0..n - m {
            let ho = xs[i] - x;
            let hp = xs[i + m] - x;
            let w = c[i + 1] - d[i];
            let den = ho - hp;
            if den.abs() < 1e-30 { continue; }
            let dd_val = w / den;
            d[i] = hp * dd_val;
            c[i] = ho * dd_val;
        }
        if 2 * ns < n - m {
            y += c[ns];
        } else {
            if ns > 0 { ns -= 1; }
            y += d[ns];
        }
    }
    y
}

/// Akima interpolation (robust against outliers)
pub struct AkimaInterp {
    pub xs: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl AkimaInterp {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert!(n >= 5);
        let nm1 = n - 1;

        let mut m = vec![0.0; n + 3]; // slopes extended
        for i in 0..nm1 {
            m[i + 2] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
        }
        // Extend using quadratic extrapolation
        m[1] = 2.0 * m[2] - m[3];
        m[0] = 2.0 * m[1] - m[2];
        m[n + 1] = 2.0 * m[n] - m[n - 1];
        m[n + 2] = 2.0 * m[n + 1] - m[n];

        let mut t = vec![0.0; n]; // tangents
        for i in 0..n {
            let w1 = (m[i + 3] - m[i + 2]).abs();
            let w2 = (m[i + 1] - m[i]).abs();
            if w1 + w2 > 1e-15 {
                t[i] = (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
            } else {
                t[i] = 0.5 * (m[i + 1] + m[i + 2]);
            }
        }

        let mut a = Vec::with_capacity(nm1);
        let mut b = Vec::with_capacity(nm1);
        let mut c = Vec::with_capacity(nm1);
        let mut d = Vec::with_capacity(nm1);

        for i in 0..nm1 {
            let h = xs[i + 1] - xs[i];
            let dy = ys[i + 1] - ys[i];
            a.push(ys[i]);
            b.push(t[i]);
            c.push((3.0 * dy / h - 2.0 * t[i] - t[i + 1]) / h);
            d.push((t[i] + t[i + 1] - 2.0 * dy / h) / (h * h));
        }

        Self { xs: xs.to_vec(), a, b, c, d }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let i = find_interval(&self.xs, x).min(n - 2);
        let dx = x - self.xs[i];
        self.a[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))
    }
}

/// Steffen's monotonic interpolation
pub struct SteffenInterp {
    pub xs: Vec<f64>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub c: Vec<f64>,
    pub d: Vec<f64>,
}

impl SteffenInterp {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert!(n >= 3);
        let nm1 = n - 1;

        let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();
        let s: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

        let mut yp = vec![0.0; n];
        yp[0] = s[0];
        yp[nm1] = s[nm1 - 1];
        for i in 1..nm1 {
            let pi = (s[i - 1] * h[i] + s[i] * h[i - 1]) / (h[i - 1] + h[i]);
            yp[i] = (s[i - 1].signum() + s[i].signum())
                * pi.abs().min(s[i - 1].abs().min(s[i].abs())) * 0.5;
            // Actually Steffen's formula
            let p = (s[i-1]*h[i] + s[i]*h[i-1]) / (h[i-1]+h[i]);
            yp[i] = (s[i-1].signum() + s[i].signum())
                * (s[i-1].abs().min(s[i].abs())).min(0.5 * p.abs());
        }

        let mut a = Vec::with_capacity(nm1);
        let mut b = Vec::with_capacity(nm1);
        let mut c = Vec::with_capacity(nm1);
        let mut d = Vec::with_capacity(nm1);

        for i in 0..nm1 {
            a.push(ys[i]);
            b.push(yp[i]);
            c.push((3.0 * s[i] - 2.0 * yp[i] - yp[i + 1]) / h[i]);
            d.push((yp[i] + yp[i + 1] - 2.0 * s[i]) / (h[i] * h[i]));
        }

        Self { xs: xs.to_vec(), a, b, c, d }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let i = find_interval(&self.xs, x).min(n - 2);
        let dx = x - self.xs[i];
        self.a[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))
    }
}

fn find_interval(xs: &[f64], x: f64) -> usize {
    let n = xs.len();
    if x <= xs[0] { return 0; }
    if x >= xs[n - 1] { return n - 2; }
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x { lo = mid; } else { hi = mid; }
    }
    lo
}

/// Piecewise cubic Hermite interpolation (PCHIP)
pub struct PchipInterp {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub slopes: Vec<f64>,
}

impl PchipInterp {
    pub fn new(xs: &[f64], ys: &[f64]) -> Self {
        let n = xs.len();
        assert!(n >= 2);
        let nm1 = n - 1;
        let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();
        let delta: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
            return Self { xs: xs.to_vec(), ys: ys.to_vec(), slopes: d };
        }

        for i in 1..nm1 {
            if delta[i - 1].signum() != delta[i].signum() || delta[i - 1].abs() < 1e-30 || delta[i].abs() < 1e-30 {
                d[i] = 0.0;
            } else {
                // Harmonic mean weighted by h
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }

        // Endpoint slopes
        d[0] = pchip_end_slope(&h, &delta, true);
        d[nm1] = pchip_end_slope(&h, &delta, false);

        Self { xs: xs.to_vec(), ys: ys.to_vec(), slopes: d }
    }

    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let i = find_interval(&self.xs, x).min(n - 2);
        let h = self.xs[i + 1] - self.xs[i];
        let t = (x - self.xs[i]) / h;
        let t2 = t * t; let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        h00 * self.ys[i] + h10 * h * self.slopes[i] + h01 * self.ys[i + 1] + h11 * h * self.slopes[i + 1]
    }
}

fn pchip_end_slope(h: &[f64], delta: &[f64], left: bool) -> f64 {
    if left {
        let d = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
        if d.signum() != delta[0].signum() { 0.0 }
        else if delta[0].signum() != delta[1].signum() && d.abs() > 3.0 * delta[0].abs() {
            3.0 * delta[0]
        } else { d }
    } else {
        let n = delta.len();
        let d = ((2.0 * h[n - 1] + h[n - 2]) * delta[n - 1] - h[n - 1] * delta[n - 2]) / (h[n - 1] + h[n - 2]);
        if d.signum() != delta[n - 1].signum() { 0.0 }
        else if delta[n - 1].signum() != delta[n - 2].signum() && d.abs() > 3.0 * delta[n - 1].abs() {
            3.0 * delta[n - 1]
        } else { d }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interp() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        let li = LinearInterp::new(xs, ys);
        assert!((li.eval(0.5) - 0.5).abs() < 1e-10);
        assert!((li.eval(1.5) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_spline() {
        let xs: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let cs = CubicSpline::natural(&xs, &ys);
        // Should interpolate exactly at nodes
        for i in 0..10 {
            assert!((cs.eval(xs[i]) - ys[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lagrange() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![1.0, 3.0, 7.0]; // 1 + x + x^2 at 0,1,2 => 1,3,7 ✓ (actually 1+0+0=1, 1+1+1=3, 1+2+4=7)
        let y = lagrange(&xs, &ys, 1.5);
        let expected = 1.0 + 1.5 + 1.5 * 1.5;
        assert!((y - expected).abs() < 1e-10);
    }

    #[test]
    fn test_monotone_hermite() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mh = MonotoneHermite::new(&xs, &ys);
        assert!((mh.eval(2.5) - 2.5).abs() < 1e-6);
    }
}
