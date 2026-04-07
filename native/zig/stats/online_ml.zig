// online_ml.zig -- Online ML primitives for SRFM signal learning.
// Implements:
//   - LogisticSGD: logistic regression with SGD updates
//   - FTRLProximal: Follow-The-Regularized-Leader with L1+L2 proximal operator
//
// Both structs operate on 6-dimensional feature vectors and produce
// probability outputs in (0, 1).
// C ABI exports provided for all functions.
//
// Zig 0.12+ syntax.

const std = @import("std");
const math = std.math;

// ============================================================
// Sigmoid (logistic function) -- numerically stable
// ============================================================

inline fn sigmoid(x: f64) f64 {
    if (x >= 0.0) {
        const e = @exp(-x);
        return 1.0 / (1.0 + e);
    } else {
        const e = @exp(x);
        return e / (1.0 + e);
    }
}

// ============================================================
// Dot product of two 6-element vectors
// ============================================================

inline fn dot6(a: [6]f64, b: [6]f64) f64 {
    var s: f64 = 0.0;
    for (0..6) |i| s += a[i] * b[i];
    return s;
}

// ============================================================
// LogisticSGD -- online logistic regression with SGD
// ============================================================
//
// Model: p = sigmoid(dot(weights, features) + bias)
// Update rule (negative log-likelihood gradient):
//   grad = (p - label)
//   weights[i] -= lr * grad * features[i]
//   bias       -= lr * grad
//
// Features should be pre-normalized to roughly [-1, 1] for best convergence.
// label should be 0.0 or 1.0.

pub const LogisticSGD = struct {
    weights: [6]f64,
    bias:    f64,
    lr:      f64,    // learning rate

    // Default initialization: small random-like weights (deterministic)
    pub fn init(lr: f64) LogisticSGD {
        // Initialize weights with small values derived from golden ratio
        // to break symmetry without requiring an RNG.
        const phi: f64 = 0.6180339887;
        var w: [6]f64 = undefined;
        var x: f64 = phi;
        for (0..6) |i| {
            x = @mod(x * phi, 1.0);
            w[i] = (x - 0.5) * 0.02;  // small values in (-0.01, 0.01)
        }
        return .{
            .weights = w,
            .bias    = 0.0,
            .lr      = lr,
        };
    }

    // zero init
    pub fn initZero(lr: f64) LogisticSGD {
        return .{
            .weights = [6]f64{ 0, 0, 0, 0, 0, 0 },
            .bias    = 0.0,
            .lr      = lr,
        };
    }

    // predict -- returns probability in (0, 1).
    pub fn predict(self: *const LogisticSGD, features: [6]f64) f64 {
        const logit = dot6(self.weights, features) + self.bias;
        return sigmoid(logit);
    }

    // update -- SGD step given features and true label in {0.0, 1.0}.
    // Also accepts soft labels in (0, 1) for knowledge distillation.
    pub fn update(self: *LogisticSGD, features: [6]f64, label: f64) void {
        const p = self.predict(features);
        const grad = p - label;  // dL/dlogit = p - y for log-loss

        for (0..6) |i| {
            self.weights[i] -= self.lr * grad * features[i];
        }
        self.bias -= self.lr * grad;
    }

    // update_with_l2 -- SGD step with L2 weight decay.
    // lambda: L2 regularization coefficient.
    pub fn updateWithL2(
        self: *LogisticSGD,
        features: [6]f64,
        label: f64,
        lambda: f64,
    ) void {
        const p = self.predict(features);
        const grad = p - label;

        for (0..6) |i| {
            self.weights[i] -= self.lr * (grad * features[i] + lambda * self.weights[i]);
        }
        self.bias -= self.lr * grad;
    }

    // predict_class -- threshold at 0.5 => returns 0.0 or 1.0.
    pub fn predictClass(self: *const LogisticSGD, features: [6]f64) f64 {
        return if (self.predict(features) >= 0.5) 1.0 else 0.0;
    }

    // l2_norm -- ||weights||_2 for monitoring convergence.
    pub fn l2Norm(self: *const LogisticSGD) f64 {
        var s: f64 = 0.0;
        for (self.weights) |w| s += w * w;
        return @sqrt(s);
    }
};

// ============================================================
// FTRLProximal -- Follow-The-Regularized-Leader (proximal variant)
// ============================================================
//
// McMahan et al. (2013): Ad Click Prediction at Scale.
// Per-coordinate adaptive learning rates via per-coordinate gradient sums.
//
// State per coordinate:
//   z[i] -- accumulated gradient minus learning-rate-corrected weight
//   n[i] -- sum of squared gradients
//
// Update:
//   g[i]  = (p - label) * features[i]  (gradient component)
//   sigma = (sqrt(n[i] + g[i]^2) - sqrt(n[i])) / alpha
//   z[i] += g[i] - sigma * w[i]
//   n[i] += g[i]^2
//
// Weight (proximal operator -- L1 soft-threshold + L2):
//   if |z[i]| <= l1: w[i] = 0
//   else:            w[i] = -(z[i] - sign(z[i])*l1) / ((beta + sqrt(n[i]))/alpha + l2)

pub const FTRLProximal = struct {
    z:     [6]f64,   // gradient accumulator
    n:     [6]f64,   // squared gradient accumulator
    alpha: f64,      // initial learning rate
    beta:  f64,      // smoothing parameter (e.g., 1.0)
    l1:    f64,      // L1 regularization
    l2:    f64,      // L2 regularization

    pub fn init(alpha: f64, beta: f64, l1: f64, l2: f64) FTRLProximal {
        return .{
            .z     = [6]f64{ 0, 0, 0, 0, 0, 0 },
            .n     = [6]f64{ 0, 0, 0, 0, 0, 0 },
            .alpha = alpha,
            .beta  = beta,
            .l1    = l1,
            .l2    = l2,
        };
    }

    // computeWeight -- derive weight for coordinate i from z[i], n[i].
    pub fn computeWeight(self: *const FTRLProximal, i: usize) f64 {
        const zi = self.z[i];
        const ni = self.n[i];
        if (@abs(zi) <= self.l1) return 0.0;
        const sign_z: f64 = if (zi > 0.0) 1.0 else -1.0;
        const denom = (self.beta + @sqrt(ni)) / self.alpha + self.l2;
        if (@abs(denom) < 1e-15) return 0.0;
        return -(zi - sign_z * self.l1) / denom;
    }

    // predict -- compute current weights on-the-fly and return sigmoid output.
    pub fn predict(self: *const FTRLProximal, features: [6]f64) f64 {
        var logit: f64 = 0.0;
        for (0..6) |i| {
            logit += self.computeWeight(i) * features[i];
        }
        return sigmoid(logit);
    }

    // update -- FTRL-Proximal update step.
    // features: 6-element feature vector.
    // label: true label in {0.0, 1.0}.
    // grad: pre-computed gradient (p - label). If 0.0, will be computed internally.
    pub fn update(self: *FTRLProximal, features: [6]f64, label: f64, grad_in: f64) void {
        // Compute prediction (need current weights)
        const p = self.predict(features);
        const grad_base = if (grad_in == 0.0) (p - label) else grad_in;

        for (0..6) |i| {
            const g = grad_base * features[i];
            const g2 = g * g;

            // Sigma: learning-rate correction
            const sqrt_n_old = @sqrt(self.n[i]);
            self.n[i] += g2;
            const sqrt_n_new = @sqrt(self.n[i]);
            const sigma = (sqrt_n_new - sqrt_n_old) / self.alpha;

            // Compute current weight for z update
            const w_i = self.computeWeight(i);
            self.z[i] += g - sigma * w_i;
        }
    }

    // reset -- clear state (use for online re-training)
    pub fn reset(self: *FTRLProximal) void {
        self.z = [6]f64{ 0, 0, 0, 0, 0, 0 };
        self.n = [6]f64{ 0, 0, 0, 0, 0, 0 };
    }

    // effectiveWeights -- fill out[6] with the current effective weights.
    pub fn effectiveWeights(self: *const FTRLProximal, out: *[6]f64) void {
        for (0..6) |i| {
            out[i] = self.computeWeight(i);
        }
    }

    // sparsity -- count of zero weights (L1 drives sparsity)
    pub fn sparsity(self: *const FTRLProximal) usize {
        var count: usize = 0;
        for (0..6) |i| {
            if (@abs(self.z[i]) <= self.l1) count += 1;
        }
        return count;
    }
};

// ============================================================
// C ABI exports -- LogisticSGD
// ============================================================
//
// Memory management: caller allocates and owns the model struct.
// Pass pointer to a LogisticSGD via [*]u8 cast (opaque pointer).
// Alternatively, use the init/free pattern below.

// logistic_sgd_new -- allocate a new LogisticSGD on the heap and return pointer.
// lr: learning rate.
export fn logistic_sgd_new(lr: f64) ?*LogisticSGD {
    const p = std.heap.c_allocator.create(LogisticSGD) catch return null;
    p.* = LogisticSGD.init(lr);
    return p;
}

// logistic_sgd_free -- free a heap-allocated LogisticSGD.
export fn logistic_sgd_free(model: *LogisticSGD) void {
    std.heap.c_allocator.destroy(model);
}

// logistic_sgd_predict -- predict probability for 6 features.
// features: pointer to 6 f64 values.
export fn logistic_sgd_predict(model: *const LogisticSGD, features: [*]const f64) f64 {
    const f6: [6]f64 = .{
        features[0], features[1], features[2],
        features[3], features[4], features[5],
    };
    return model.predict(f6);
}

// logistic_sgd_update -- SGD update with label in {0.0, 1.0}.
export fn logistic_sgd_update(
    model: *LogisticSGD,
    features: [*]const f64,
    label: f64,
) void {
    const f6: [6]f64 = .{
        features[0], features[1], features[2],
        features[3], features[4], features[5],
    };
    model.update(f6, label);
}

// logistic_sgd_update_l2 -- SGD update with L2 regularization.
export fn logistic_sgd_update_l2(
    model: *LogisticSGD,
    features: [*]const f64,
    label: f64,
    lambda: f64,
) void {
    const f6: [6]f64 = .{
        features[0], features[1], features[2],
        features[3], features[4], features[5],
    };
    model.updateWithL2(f6, label, lambda);
}

// logistic_sgd_get_weights -- copy weights[6] and bias into out[7].
// out[0..6] = weights, out[6] = bias.
export fn logistic_sgd_get_weights(model: *const LogisticSGD, out: [*]f64) void {
    for (0..6) |i| out[i] = model.weights[i];
    out[6] = model.bias;
}

// logistic_sgd_set_lr -- update learning rate at runtime.
export fn logistic_sgd_set_lr(model: *LogisticSGD, lr: f64) void {
    model.lr = lr;
}

// ============================================================
// C ABI exports -- FTRLProximal
// ============================================================

// ftrl_new -- allocate FTRL model on the heap.
export fn ftrl_new(alpha: f64, beta: f64, l1: f64, l2: f64) ?*FTRLProximal {
    const p = std.heap.c_allocator.create(FTRLProximal) catch return null;
    p.* = FTRLProximal.init(alpha, beta, l1, l2);
    return p;
}

// ftrl_free -- free heap-allocated FTRL model.
export fn ftrl_free(model: *FTRLProximal) void {
    std.heap.c_allocator.destroy(model);
}

// ftrl_predict -- predict probability for 6 features.
export fn ftrl_predict(model: *const FTRLProximal, features: [*]const f64) f64 {
    const f6: [6]f64 = .{
        features[0], features[1], features[2],
        features[3], features[4], features[5],
    };
    return model.predict(f6);
}

// ftrl_update -- FTRL-Proximal update step.
// grad: pre-computed gradient; pass 0.0 to compute internally.
export fn ftrl_update(
    model: *FTRLProximal,
    features: [*]const f64,
    label: f64,
    grad: f64,
) void {
    const f6: [6]f64 = .{
        features[0], features[1], features[2],
        features[3], features[4], features[5],
    };
    model.update(f6, label, grad);
}

// ftrl_get_weights -- fill out[6] with current effective weights.
export fn ftrl_get_weights(model: *const FTRLProximal, out: [*]f64) void {
    var w: [6]f64 = undefined;
    model.effectiveWeights(&w);
    for (0..6) |i| out[i] = w[i];
}

// ftrl_sparsity -- return number of zero-weighted features.
export fn ftrl_sparsity(model: *const FTRLProximal) usize {
    return model.sparsity();
}

// ftrl_reset -- clear model state.
export fn ftrl_reset(model: *FTRLProximal) void {
    model.reset();
}

// ============================================================
// Tests (run with `zig test online_ml.zig`)
// ============================================================

test "sigmoid properties" {
    const testing = std.testing;
    try testing.expectApproxEqAbs(sigmoid(0.0), 0.5, 1e-9);
    try testing.expect(sigmoid(10.0) > 0.99);
    try testing.expect(sigmoid(-10.0) < 0.01);
    // Symmetry: sigmoid(x) + sigmoid(-x) = 1
    try testing.expectApproxEqAbs(sigmoid(3.0) + sigmoid(-3.0), 1.0, 1e-9);
}

test "logistic sgd learns XOR-like pattern" {
    const testing = std.testing;
    var model = LogisticSGD.init(0.1);

    // Train on: feature[0] > 0 => label 1, feature[0] < 0 => label 0
    for (0..200) |i| {
        const sign: f64 = if (i % 2 == 0) 1.0 else -1.0;
        const features: [6]f64 = .{ sign, 0, 0, 0, 0, 0 };
        const label: f64 = if (i % 2 == 0) 1.0 else 0.0;
        model.update(features, label);
    }

    // After training, positive feature should give p > 0.6
    const pos_pred = model.predict(.{ 1.0, 0, 0, 0, 0, 0 });
    const neg_pred = model.predict(.{ -1.0, 0, 0, 0, 0, 0 });
    try testing.expect(pos_pred > 0.6);
    try testing.expect(neg_pred < 0.4);
}

test "ftrl convergence" {
    const testing = std.testing;
    var model = FTRLProximal.init(0.1, 1.0, 0.001, 0.01);

    // Train on linearly separable data
    for (0..500) |i| {
        const sign: f64 = if (i % 2 == 0) 1.0 else -1.0;
        const features: [6]f64 = .{ sign, sign * 0.5, 0, 0, 0, 0 };
        const label: f64 = if (i % 2 == 0) 1.0 else 0.0;
        model.update(features, label, 0.0);
    }

    const pos_pred = model.predict(.{ 1.0, 0.5, 0, 0, 0, 0 });
    const neg_pred = model.predict(.{ -1.0, -0.5, 0, 0, 0, 0 });
    try testing.expect(pos_pred > 0.55);
    try testing.expect(neg_pred < 0.45);
}

test "ftrl l1 sparsity" {
    const testing = std.testing;
    // High L1 should zero out uninformative features
    var model = FTRLProximal.init(0.1, 1.0, 0.5, 0.01);

    // Train with only feature[0] informative
    for (0..200) |i| {
        const s: f64 = if (i % 2 == 0) 1.0 else -1.0;
        const features: [6]f64 = .{ s, 0.01, 0.01, 0.01, 0.01, 0.01 };
        const label: f64 = if (i % 2 == 0) 1.0 else 0.0;
        model.update(features, label, 0.0);
    }
    // At least some features should be zeroed by L1
    try testing.expect(model.sparsity() >= 1);
}

test "logistic sgd l2 norm finite" {
    const testing = std.testing;
    var model = LogisticSGD.init(0.01);
    for (0..50) |i| {
        const f: f64 = @as(f64, @floatFromInt(i)) * 0.1 - 2.5;
        model.update(.{ f, f*0.5, 0, 0, 0, 0 }, if (f > 0) 1.0 else 0.0);
    }
    const norm = model.l2Norm();
    try testing.expect(norm > 0.0);
    try testing.expect(!math.isNan(norm));
    try testing.expect(!math.isInf(norm));
}
