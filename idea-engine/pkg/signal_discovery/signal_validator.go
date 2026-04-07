package signal_discovery

import (
	"fmt"
	"math"
)

// ValidationResult holds the outcome of validating a SignalCandidate.
type ValidationResult struct {
	// Passed is true when all validation gates are satisfied.
	Passed bool
	// IC is the full-sample Information Coefficient on the validation data.
	IC float64
	// ICIR is the IC Information Ratio on the validation data.
	ICIR float64
	// MaxDD is the maximum drawdown of a long-short portfolio built on the signal.
	MaxDD float64
	// Sharpe is the annualised Sharpe of the signal-driven long-short portfolio.
	Sharpe float64
	// OverfitScore is train_IC / test_IC -- values close to 1 are good.
	// Values > 1 indicate overfitting.
	OverfitScore float64
	// Reasons accumulates human-readable descriptions of each failed gate.
	Reasons []string
}

// ValidatorConfig holds tunable thresholds for SignalValidator.
type ValidatorConfig struct {
	// MinIC is the minimum IC required (default 0.03).
	MinIC float64
	// MinICIR is the minimum ICIR required (default 0.5).
	MinICIR float64
	// MaxDrawdown is the maximum allowed drawdown fraction (default 0.30).
	MaxDrawdown float64
	// MaxOverfitScore is the ceiling on train_IC / test_IC (default 0.3).
	// Wait -- the spec says OverfitScore < 0.3, meaning test_IC / train_IC
	// must be > 0.7 (i.e. test is no more than 30% degraded vs train).
	// We model OverfitScore = 1 - (test_IC / train_IC). Low is good.
	MaxOverfitScore float64
	// MaxCorrelation is the max |Pearson r| to any of the top-5 existing signals.
	MaxCorrelation float64
	// RollingICWindow is bars-per-window for the rolling ICIR computation.
	RollingICWindow int
	// AnnualisationFactor is sqrt(trading_periods_per_year) for Sharpe.
	AnnualisationFactor float64
}

// DefaultValidatorConfig returns production-grade defaults.
func DefaultValidatorConfig() ValidatorConfig {
	return ValidatorConfig{
		MinIC:               0.03,
		MinICIR:             0.5,
		MaxDrawdown:         0.30,
		MaxOverfitScore:     0.30,
		MaxCorrelation:      0.70,
		RollingICWindow:     20,
		AnnualisationFactor: math.Sqrt(252),
	}
}

// SignalValidator validates a SignalCandidate against a set of quality gates.
type SignalValidator struct {
	cfg ValidatorConfig
	// top5 is the series of the current top-5 portfolio signals, used for
	// the correlation gate.
	top5 [][]float64
}

// NewSignalValidator constructs a SignalValidator with the given config.
func NewSignalValidator(cfg ValidatorConfig) *SignalValidator {
	return &SignalValidator{cfg: cfg}
}

// SetTop5 sets the series of the top-5 existing signals for correlation checking.
func (v *SignalValidator) SetTop5(series [][]float64) {
	v.top5 = series
}

// Validate runs all gates against the candidate using the provided bars.
// bars must have the Return field populated. The candidate's Series must
// be pre-computed and aligned to bars.
func (v *SignalValidator) Validate(candidate SignalCandidate, bars []Bar) ValidationResult {
	res := ValidationResult{}
	var reasons []string

	n := len(bars)
	if n < 20 {
		res.Reasons = []string{"insufficient bars for validation"}
		return res
	}

	// Compute or reuse signal series.
	sig := candidate.Series
	if len(sig) != n {
		sig = computeSignalSeries(candidate, bars)
	}

	ret := make([]float64, n)
	for i, b := range bars {
		ret[i] = b.Return
	}

	// Gate 1: IC > MinIC
	res.IC = pearsonCorrelation(sig, ret)
	if res.IC < v.cfg.MinIC {
		reasons = append(reasons, fmt.Sprintf("IC %.4f < MinIC %.4f", res.IC, v.cfg.MinIC))
	}

	// Gate 2: ICIR > MinICIR
	res.ICIR = rollingICIR(sig, ret, v.cfg.RollingICWindow)
	if res.ICIR < v.cfg.MinICIR {
		reasons = append(reasons, fmt.Sprintf("ICIR %.4f < MinICIR %.4f", res.ICIR, v.cfg.MinICIR))
	}

	// Gate 3: MaxDD < MaxDrawdown
	ls := longShortReturns(sig, ret)
	res.MaxDD = maxDrawdown(ls)
	if res.MaxDD >= v.cfg.MaxDrawdown {
		reasons = append(reasons, fmt.Sprintf("MaxDD %.4f >= MaxDrawdown %.4f", res.MaxDD, v.cfg.MaxDrawdown))
	}

	// Compute Sharpe for informational purposes.
	res.Sharpe = sharpe(ls, v.cfg.AnnualisationFactor)

	// Gate 4: OverfitScore < MaxOverfitScore
	// Split the bars into train / test halves.
	trainSplit := n / 2
	if trainSplit >= 10 && (n-trainSplit) >= 10 {
		trainSig := sig[:trainSplit]
		trainRet := ret[:trainSplit]
		testSig := sig[trainSplit:]
		testRet := ret[trainSplit:]

		trainIC := pearsonCorrelation(trainSig, trainRet)
		testIC := pearsonCorrelation(testSig, testRet)
		res.OverfitScore = overfitScore(trainIC, testIC)
		if res.OverfitScore > v.cfg.MaxOverfitScore {
			reasons = append(reasons, fmt.Sprintf("OverfitScore %.4f > MaxOverfitScore %.4f", res.OverfitScore, v.cfg.MaxOverfitScore))
		}
	}

	// Gate 5: not correlated > MaxCorrelation with top-5 signals.
	for i, ex := range v.top5 {
		minLen := len(sig)
		if len(ex) < minLen {
			minLen = len(ex)
		}
		if minLen < 5 {
			continue
		}
		corr := pearsonCorrelation(sig[:minLen], ex[:minLen])
		if math.Abs(corr) > v.cfg.MaxCorrelation {
			reasons = append(reasons, fmt.Sprintf(
				"correlation %.4f with existing signal[%d] exceeds %.4f",
				math.Abs(corr), i, v.cfg.MaxCorrelation,
			))
		}
	}

	res.Passed = len(reasons) == 0
	res.Reasons = reasons
	return res
}

// DeflatedSharpe applies the Bailey-Lopez de Prado deflation to account for
// multiple testing. It returns the probability-adjusted expected maximum Sharpe
// under the null hypothesis of no skill.
//
// Formula:
//
//	E[max SR] = ((1 - euler_gamma) * Z(1 - 1/nTrials) + euler_gamma * Z(1 - 1/(nTrials*e))) * sigma_sr
//
// where sigma_sr = sqrt(1 + 0.5*SR^2) * sqrt(T) (approximation).
// We use the simpler Expected Maximum Sharpe approximation from Bailey & Lopez:
//
//	PSR(SR*) = Phi( (SR* - E[max SR]) / sigma_sr )
func (v *SignalValidator) DeflatedSharpe(trialSharpe float64, nTrials int) float64 {
	if nTrials <= 0 {
		nTrials = 1
	}
	// Expected maximum Sharpe under iid null with nTrials candidates.
	// E[max] approx sqrt(2 * log(nTrials)) for large nTrials.
	eulerGamma := 0.5772156649
	eMaxSR := (1-eulerGamma)*normalQuantile(1-1.0/float64(nTrials)) +
		eulerGamma*normalQuantile(1-1.0/(float64(nTrials)*math.E))

	// Variance of SR estimate -- assume T=252 observations with approx skew/kurtosis.
	// sigma_sr approx sqrt((1 + 0.5*SR^2) / T)
	T := 252.0
	sigSR := math.Sqrt((1 + 0.5*trialSharpe*trialSharpe) / T)

	// PSR: probability that the observed SR is above the expected max under null.
	z := (trialSharpe - eMaxSR) / sigSR
	return normalCDF(z)
}

// ---------------------------------------------------------------------------
// Signal-portfolio helpers
// ---------------------------------------------------------------------------

// longShortReturns converts signal values and forward returns into a long-short
// portfolio return series. Positive signal = long, negative = short.
// The position size is proportional to |signal| normalised to unit leverage.
func longShortReturns(sig, ret []float64) []float64 {
	n := len(sig)
	out := make([]float64, n)
	// Compute total absolute signal weight for normalisation.
	totalAbs := 0.0
	for _, v := range sig {
		totalAbs += math.Abs(v)
	}
	if totalAbs == 0 {
		return out
	}
	for i := range sig {
		weight := sig[i] / totalAbs
		out[i] = weight * ret[i]
	}
	return out
}

// maxDrawdown computes the maximum peak-to-trough drawdown of a return series.
func maxDrawdown(returns []float64) float64 {
	if len(returns) == 0 {
		return 0
	}
	peak := 1.0
	equity := 1.0
	maxDD := 0.0
	for _, r := range returns {
		equity *= (1 + r)
		if equity > peak {
			peak = equity
		}
		dd := (peak - equity) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

// sharpe computes the annualised Sharpe ratio of a return series.
func sharpe(returns []float64, annFactor float64) float64 {
	m := mean64(returns)
	s := stddev64(returns, m)
	if s == 0 {
		return 0
	}
	return (m / s) * annFactor
}

// overfitScore returns 1 - (testIC / trainIC). Zero means perfect
// generalisation; positive means test degraded vs train.
func overfitScore(trainIC, testIC float64) float64 {
	if trainIC == 0 {
		return 0
	}
	return 1 - (testIC / trainIC)
}

// ---------------------------------------------------------------------------
// Normal distribution helpers (no external imports)
// ---------------------------------------------------------------------------

// normalCDF approximates the standard normal CDF using Abramowitz & Stegun.
func normalCDF(x float64) float64 {
	if x < -8 {
		return 0
	}
	if x > 8 {
		return 1
	}
	return 0.5 * math.Erfc(-x/math.Sqrt2)
}

// normalQuantile approximates the standard normal quantile (inverse CDF)
// using the rational approximation by Peter J. Acklam.
func normalQuantile(p float64) float64 {
	if p <= 0 {
		return math.Inf(-1)
	}
	if p >= 1 {
		return math.Inf(1)
	}
	// Coefficients in rational approximation.
	a := [...]float64{
		-3.969683028665376e+01, 2.209460984245205e+02,
		-2.759285104469687e+02, 1.383577518672690e+02,
		-3.066479806614716e+01, 2.506628277459239e+00,
	}
	b := [...]float64{
		-5.447609879822406e+01, 1.615858368580409e+02,
		-1.556989798598866e+02, 6.680131188771972e+01,
		-1.328068155288572e+01,
	}
	c := [...]float64{
		-7.784894002430293e-03, -3.223964580411365e-01,
		-2.400758277161838e+00, -2.549732539343734e+00,
		4.374664141464968e+00, 2.938163982698783e+00,
	}
	d := [...]float64{
		7.784695709041462e-03, 3.224671290700398e-01,
		2.445134137142996e+00, 3.754408661907416e+00,
	}

	pLow := 0.02425
	pHigh := 1 - pLow
	var q float64

	switch {
	case p < pLow:
		q = math.Sqrt(-2 * math.Log(p))
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	case p <= pHigh:
		q = p - 0.5
		r := q * q
		return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
			(((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
	default:
		q = math.Sqrt(-2 * math.Log(1-p))
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) /
			((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
}
