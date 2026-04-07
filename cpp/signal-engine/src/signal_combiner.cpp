// signal_combiner.cpp -- Out-of-line helpers for SignalCombiner and batch utilities.
// Includes:
//   - Online IC tracking stress test helper
//   - AdaHedge adaptive learning rate variant
//   - SRFM ensemble combination over an array of bar outputs
//   - Self-test under SRFM_COMBINER_SELFTEST

#include "signal_combiner.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <cassert>
#include <functional>

namespace srfm {
namespace combining {

// ============================================================
// AdaHedge -- adaptive learning rate hedge algorithm.
// Automatically tunes eta to minimize regret.
// ============================================================

class AdaHedge {
public:
    int    n_experts;
    double delta;        // hedge mixing parameter (typically 0.1)

    explicit AdaHedge(int n, double d = 0.1)
        : n_experts(n), delta(d),
          weights_(n, 1.0 / static_cast<double>(n)),
          cumulative_loss_(n, 0.0),
          Delta_(0.0),
          eta_(1.0)
    {}

    // predict -- return weighted average of expert predictions.
    double predict(const std::vector<double>& expert_preds) const noexcept {
        assert(static_cast<int>(expert_preds.size()) == n_experts);
        double result = 0.0;
        for (int i = 0; i < n_experts; ++i) {
            result += weights_[i] * expert_preds[i];
        }
        return result;
    }

    // update -- update weights given per-expert losses in [0,1].
    // loss[i] = instantaneous loss for expert i this round.
    void update(const std::vector<double>& losses) {
        assert(static_cast<int>(losses.size()) == n_experts);

        // Compute mixture loss
        double mix_loss = 0.0;
        for (int i = 0; i < n_experts; ++i) {
            mix_loss += weights_[i] * losses[i];
        }

        // Update cumulative losses and compute Delta increment
        double delta_inc = 0.0;
        for (int i = 0; i < n_experts; ++i) {
            cumulative_loss_[i] += losses[i];
            const double diff = losses[i] - mix_loss;
            delta_inc += weights_[i] * diff * diff;
        }
        Delta_ += delta_inc;

        // AdaHedge learning rate: eta = sqrt(ln(n) / Delta)
        const double log_n = std::log(static_cast<double>(n_experts));
        if (Delta_ > 1e-15) {
            eta_ = std::sqrt(log_n / Delta_);
        }

        // Update weights: w[i] *= exp(-eta * loss[i])
        double w_sum = 0.0;
        for (int i = 0; i < n_experts; ++i) {
            weights_[i] *= std::exp(-eta_ * losses[i]);
            w_sum += weights_[i];
        }
        // Normalize
        for (int i = 0; i < n_experts; ++i) {
            weights_[i] = (1.0 - delta) * weights_[i] / w_sum
                        + delta / static_cast<double>(n_experts);
        }
    }

    double learning_rate() const noexcept { return eta_; }
    const std::vector<double>& weights() const noexcept { return weights_; }

private:
    std::vector<double> weights_;
    std::vector<double> cumulative_loss_;
    double Delta_;
    double eta_;
};

// ============================================================
// Ensemble combination over a sequence of bar-level signals
// ============================================================

struct BarSignals {
    double bh_mass;
    double nav_curv;
    double hurst;
    double garch_vol;
    double pattern;
    double realized_return;  // filled in by caller after each bar; used for IC update
};

// Run SignalCombiner over a bar sequence, returning per-bar portfolio signals.
// Writes n_bars values into out_signals.
void run_ensemble_batch(
    const BarSignals* bars,
    int n_bars,
    CombineMethod method,
    double* out_signals) noexcept
{
    if (!bars || n_bars <= 0 || !out_signals) return;

    SignalCombiner comb;
    const std::vector<std::string> names = {
        "bh_mass", "nav_curv", "hurst", "garch_vol", "pattern"
    };

    for (int i = 0; i < n_bars; ++i) {
        const BarSignals& b = bars[i];

        // Build signal inputs -- normalize each to roughly [-1,1]
        std::vector<SignalInput> inputs;
        inputs.reserve(5);

        // BH mass: assume [0,1] input, center to [-1,1]
        inputs.push_back({"bh_mass",   2.0 * b.bh_mass - 1.0,   1.0, 0.0});
        // Nav curvature: sign indicates direction; clamp
        inputs.push_back({"nav_curv",  std::tanh(b.nav_curv),    0.8, 0.0});
        // Hurst: 2*(H-0.5) => [-1,1]
        inputs.push_back({"hurst",     2.0*(b.hurst - 0.5),      0.7, 0.0});
        // GARCH vol: invert and center (high vol => reduce)
        inputs.push_back({"garch_vol", -std::tanh(b.garch_vol),  0.5, 0.0});
        // Pattern: already in [-1,1]
        inputs.push_back({"pattern",   b.pattern,                 1.0, 0.0});

        out_signals[i] = comb.combine(inputs, method);

        // Online IC update using previous bar's realized return
        if (i > 0 && bars[i-1].realized_return != 0.0) {
            for (int j = 0; j < 5; ++j) {
                comb.update_ic(names[j], bars[i-1].realized_return, inputs[j].value);
            }
        }
    }
}

// ============================================================
// IC time series: compute rolling IC over a paired (signals, returns) array
// ============================================================

// rolling_rank_ic -- Spearman rank IC between signals and returns over a window.
// signals[i] and returns[i] must be paired.
// out[i] = IC at position i.
void rolling_rank_ic(
    const double* signals,
    const double* returns,
    int n,
    int window,
    double* out) noexcept
{
    if (!signals || !returns || !out || n <= 0 || window < 4) return;

    for (int i = 0; i < n; ++i) {
        out[i] = 0.0;
        const int start = std::max(0, i - window + 1);
        const int len   = i - start + 1;
        if (len < 4) continue;

        // Rank signals and returns within window
        std::vector<int> sig_idx(len), ret_idx(len);
        std::iota(sig_idx.begin(), sig_idx.end(), 0);
        std::iota(ret_idx.begin(), ret_idx.end(), 0);

        std::sort(sig_idx.begin(), sig_idx.end(), [&](int a, int b) {
            return signals[start + a] < signals[start + b];
        });
        std::sort(ret_idx.begin(), ret_idx.end(), [&](int a, int b) {
            return returns[start + a] < returns[start + b];
        });

        // Convert sort order to rank arrays
        std::vector<double> sig_rank(len), ret_rank(len);
        for (int r = 0; r < len; ++r) {
            sig_rank[sig_idx[r]] = static_cast<double>(r);
            ret_rank[ret_idx[r]] = static_cast<double>(r);
        }

        // Pearson correlation of ranks = Spearman IC
        double mean_s = 0, mean_r = 0;
        for (int k = 0; k < len; ++k) { mean_s += sig_rank[k]; mean_r += ret_rank[k]; }
        mean_s /= static_cast<double>(len);
        mean_r /= static_cast<double>(len);

        double cov = 0, var_s = 0, var_r = 0;
        for (int k = 0; k < len; ++k) {
            const double ds = sig_rank[k] - mean_s;
            const double dr = ret_rank[k] - mean_r;
            cov   += ds * dr;
            var_s += ds * ds;
            var_r += dr * dr;
        }
        const double denom = std::sqrt(var_s * var_r);
        out[i] = (denom > 1e-15) ? cov / denom : 0.0;
    }
}

// ============================================================
// Self-test
// ============================================================

#ifdef SRFM_COMBINER_SELFTEST

static void check(bool cond, const char* label) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::abort();
    }
    std::fprintf(stdout, "PASS: %s\n", label);
}

void run_combiner_selftest() {
    SignalCombiner comb;

    // Equal weight test: signals of equal weight
    {
        std::vector<SignalInput> inputs = {
            {"s1", 0.5,  1.0, 0.0},
            {"s2", -0.5, 1.0, 0.0},
            {"s3", 0.0,  1.0, 0.0},
        };
        double r = comb.combine(inputs, CombineMethod::EQUAL_WEIGHT);
        check(std::abs(r) < 0.01, "equal_weight_zero");
    }

    // Equal weight: all bullish
    {
        std::vector<SignalInput> inputs = {
            {"s1", 0.8, 1.0, 0.0},
            {"s2", 0.6, 1.0, 0.0},
            {"s3", 0.4, 1.0, 0.0},
        };
        double r = comb.combine(inputs, CombineMethod::EQUAL_WEIGHT);
        check(r > 0.5 && r <= 1.0, "equal_weight_bullish");
    }

    // IC update and retrieval
    {
        for (int i = 0; i < 30; ++i) {
            comb.update_ic("s1", (i % 2 == 0) ? 0.01 : -0.005, (i % 2 == 0) ? 0.6 : -0.3);
        }
        double ic = comb.get_ic("s1");
        check(ic > 0.0 && ic <= 1.0, "ic_range_after_update");
    }

    // IC weight test: s1 has higher IC => should dominate
    {
        // Give s_good many correct predictions, s_bad random
        SignalCombiner comb2;
        for (int i = 0; i < 50; ++i) {
            comb2.update_ic("s_good", 0.01, 0.7);   // always correct
            comb2.update_ic("s_bad",  0.01, (i % 3 == 0) ? 0.5 : -0.5);  // noisy
        }
        std::vector<SignalInput> inputs = {
            {"s_good", 0.8, 1.0, 0.0},
            {"s_bad",  0.2, 1.0, 0.0},
        };
        double r_ic    = comb2.combine(inputs, CombineMethod::IC_WEIGHT);
        double r_equal = comb2.combine(inputs, CombineMethod::EQUAL_WEIGHT);
        // IC weight should give higher value (pulls toward s_good=0.8)
        check(r_ic >= r_equal - 0.01, "ic_weight_favors_better_signal");
    }

    // Rank weight test
    {
        std::vector<SignalInput> inputs = {
            {"r1", 0.9,  1.0, 0.0},
            {"r2", -0.1, 1.0, 0.0},
            {"r3", 0.3,  1.0, 0.0},
        };
        double r = comb.combine(inputs, CombineMethod::RANK_WEIGHT);
        // Highest magnitude is r1=0.9 and gets top rank => positive
        check(r > 0.0 && r <= 1.0, "rank_weight_positive");
    }

    // Hedge algorithm: after training on correct signals, weights favor correct experts
    {
        SignalCombiner comb3;
        for (int i = 0; i < 40; ++i) {
            double ret = 0.005;
            comb3.update_ic("expert_good", ret, 0.8);
            comb3.update_ic("expert_bad",  ret, -0.6);  // consistently wrong
        }
        std::vector<SignalInput> inputs = {
            {"expert_good", 0.7, 1.0, 0.0},
            {"expert_bad", -0.3, 1.0, 0.0},
        };
        double r = comb3.combine(inputs, CombineMethod::HEDGE_ALGORITHM);
        check(r > 0.0, "hedge_favors_good_expert");
    }

    // portfolio_signal -- basic range check
    {
        SignalCombiner sc;
        double s = sc.portfolio_signal(0.7, 0.1, 0.6, 0.15, 0.4);
        check(s >= -1.0 && s <= 1.0, "portfolio_signal_range");
        check(s > 0.0, "portfolio_signal_positive_when_bullish");

        double s2 = sc.portfolio_signal(0.2, 0.5, 0.3, 0.40, -0.6);
        check(s2 >= -1.0 && s2 <= 1.0, "portfolio_signal_range_2");
    }

    // AdaHedge basic test
    {
        AdaHedge ah(3, 0.1);
        std::vector<double> preds = {0.8, 0.2, -0.3};
        double p = ah.predict(preds);
        check(std::abs(p - (0.8 + 0.2 - 0.3) / 3.0) < 0.01, "adahedge_equal_init");

        // Update with losses: expert 0 has lower loss
        for (int i = 0; i < 20; ++i) {
            std::vector<double> losses = {0.1, 0.5, 0.8};
            ah.update(losses);
        }
        const auto& w = ah.weights();
        check(w[0] > w[1] && w[1] > w[2], "adahedge_weight_ordering");
    }

    // Rolling IC test
    {
        std::vector<double> sigs    = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
        std::vector<double> rets    = {0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01};
        std::vector<double> ic_out(sigs.size(), 0.0);
        rolling_rank_ic(sigs.data(), rets.data(), static_cast<int>(sigs.size()),
                        6, ic_out.data());
        // Perfect signals => IC should be high (close to 1.0)
        check(ic_out[9] > 0.8, "perfect_ic");
    }

    std::fprintf(stdout, "All signal_combiner self-tests passed.\n");
}

#endif // SRFM_COMBINER_SELFTEST

} // namespace combining
} // namespace srfm
