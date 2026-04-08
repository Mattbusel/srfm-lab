package main

import (
	"math"
)

// ---------------------------------------------------------------------------
// Covariance estimation
// ---------------------------------------------------------------------------

// MeanReturns computes annualized mean returns per asset.
func MeanReturns(returns [][]float64, n int) []float64 {
	t := len(returns)
	if t == 0 {
		return make([]float64, n)
	}
	mu := make([]float64, n)
	for i := 0; i < t; i++ {
		for j := 0; j < n && j < len(returns[i]); j++ {
			mu[j] += returns[i][j]
		}
	}
	for j := 0; j < n; j++ {
		mu[j] = mu[j] / float64(t) * 252
	}
	return mu
}

// SampleCov computes the sample covariance matrix (annualized).
func SampleCov(returns [][]float64, n int) [][]float64 {
	t := len(returns)
	if t < 2 {
		cov := make([][]float64, n)
		for i := range cov {
			cov[i] = make([]float64, n)
		}
		return cov
	}
	mu := make([]float64, n)
	for i := 0; i < t; i++ {
		for j := 0; j < n && j < len(returns[i]); j++ {
			mu[j] += returns[i][j]
		}
	}
	for j := 0; j < n; j++ {
		mu[j] /= float64(t)
	}
	cov := make([][]float64, n)
	for i := range cov {
		cov[i] = make([]float64, n)
	}
	for k := 0; k < t; k++ {
		for i := 0; i < n; i++ {
			di := returns[k][i] - mu[i]
			for j := i; j < n; j++ {
				dj := returns[k][j] - mu[j]
				cov[i][j] += di * dj
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			cov[i][j] = cov[i][j] / float64(t-1) * 252
			cov[j][i] = cov[i][j]
		}
	}
	return cov
}

// LedoitWolfShrinkage computes the Ledoit-Wolf shrinkage covariance estimator.
func LedoitWolfShrinkage(returns [][]float64, n int) [][]float64 {
	t := len(returns)
	sample := SampleCov(returns, n)
	if t < 2 || n < 2 {
		return sample
	}
	// Target: scaled identity (average variance on diagonal)
	avgVar := 0.0
	for i := 0; i < n; i++ {
		avgVar += sample[i][i]
	}
	avgVar /= float64(n)
	// Compute shrinkage intensity using Ledoit-Wolf formula
	// Simplified: compute sum of squared off-diagonal sample cov
	sumSqSample := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i != j {
				sumSqSample += sample[i][j] * sample[i][j]
			}
		}
	}
	// Compute rho (asymptotic optimal shrinkage)
	// Simplified estimation of numerator
	mu := make([]float64, n)
	for i := 0; i < t; i++ {
		for j := 0; j < n && j < len(returns[i]); j++ {
			mu[j] += returns[i][j]
		}
	}
	for j := 0; j < n; j++ {
		mu[j] /= float64(t)
	}
	piSum := 0.0
	for k := 0; k < t; k++ {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				di := returns[k][i] - mu[i]
				dj := returns[k][j] - mu[j]
				piSum += (di*dj - sample[i][j]/252) * (di*dj - sample[i][j]/252)
			}
		}
	}
	piHat := piSum / float64(t*t) * 252 * 252
	gamma := sumSqSample
	if gamma == 0 {
		gamma = 1
	}
	shrinkage := piHat / gamma
	if shrinkage > 1 {
		shrinkage = 1
	}
	if shrinkage < 0 {
		shrinkage = 0
	}
	// Shrink
	result := make([][]float64, n)
	for i := range result {
		result[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				result[i][j] = (1-shrinkage)*sample[i][j] + shrinkage*avgVar
			} else {
				result[i][j] = (1 - shrinkage) * sample[i][j]
			}
		}
	}
	return result
}

// EWMACov computes exponentially weighted covariance matrix.
func EWMACov(returns [][]float64, n int, lambda float64) [][]float64 {
	t := len(returns)
	cov := make([][]float64, n)
	for i := range cov {
		cov[i] = make([]float64, n)
	}
	if t == 0 {
		return cov
	}
	mu := make([]float64, n)
	for i := 0; i < t; i++ {
		for j := 0; j < n && j < len(returns[i]); j++ {
			mu[j] += returns[i][j]
		}
	}
	for j := 0; j < n; j++ {
		mu[j] /= float64(t)
	}
	// EWMA: most recent observation has highest weight
	weightSum := 0.0
	for k := 0; k < t; k++ {
		w := math.Pow(lambda, float64(t-1-k))
		weightSum += w
		for i := 0; i < n; i++ {
			di := returns[k][i] - mu[i]
			for j := i; j < n; j++ {
				dj := returns[k][j] - mu[j]
				cov[i][j] += w * di * dj
			}
		}
	}
	if weightSum > 0 {
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				cov[i][j] = cov[i][j] / weightSum * 252
				cov[j][i] = cov[i][j]
			}
		}
	}
	return cov
}

// ---------------------------------------------------------------------------
// Matrix and vector utilities
// ---------------------------------------------------------------------------

func dotProduct(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		if i < len(b) {
			s += a[i] * b[i]
		}
	}
	return s
}

func matVecMul(mat [][]float64, vec []float64) []float64 {
	n := len(mat)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < len(vec) && j < len(mat[i]); j++ {
			result[i] += mat[i][j] * vec[j]
		}
	}
	return result
}

func portfolioVol(weights []float64, cov [][]float64) float64 {
	n := len(weights)
	variance := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			variance += weights[i] * weights[j] * cov[i][j]
		}
	}
	if variance < 0 {
		variance = 0
	}
	return math.Sqrt(variance)
}

func portfolioVariance(weights []float64, cov [][]float64) float64 {
	n := len(weights)
	variance := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			variance += weights[i] * weights[j] * cov[i][j]
		}
	}
	return variance
}

func sqrt(x float64) float64 {
	return math.Sqrt(math.Abs(x))
}

// invertSymmetric inverts a symmetric positive definite matrix using Cholesky.
func invertSymmetric(mat [][]float64, n int) [][]float64 {
	// Cholesky decomposition: A = L * L^T
	L := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := 0.0
			for k := 0; k < j; k++ {
				s += L[i][k] * L[j][k]
			}
			if i == j {
				val := mat[i][i] - s
				if val <= 0 {
					val = 1e-10
				}
				L[i][j] = math.Sqrt(val)
			} else {
				if L[j][j] != 0 {
					L[i][j] = (mat[i][j] - s) / L[j][j]
				}
			}
		}
	}
	// Invert L (lower triangular)
	Linv := make([][]float64, n)
	for i := range Linv {
		Linv[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		if L[i][i] != 0 {
			Linv[i][i] = 1.0 / L[i][i]
		}
		for j := i + 1; j < n; j++ {
			s := 0.0
			for k := i; k < j; k++ {
				s += L[j][k] * Linv[k][i]
			}
			if L[j][j] != 0 {
				Linv[j][i] = -s / L[j][j]
			}
		}
	}
	// A^-1 = (L^T)^-1 * L^-1 = Linv^T * Linv
	inv := make([][]float64, n)
	for i := range inv {
		inv[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			s := 0.0
			for k := i; k < n; k++ {
				s += Linv[k][i] * Linv[k][j]
			}
			inv[i][j] = s
			inv[j][i] = s
		}
	}
	return inv
}

func clampWeights(weights []float64, c ConstraintSet) []float64 {
	n := len(weights)
	out := make([]float64, n)
	copy(out, weights)
	// Box constraints
	for i := 0; i < n; i++ {
		if out[i] < c.MinWeight {
			out[i] = c.MinWeight
		}
		if out[i] > c.MaxWeight {
			out[i] = c.MaxWeight
		}
	}
	// Normalize to sum to 1
	s := 0.0
	for _, w := range out {
		s += w
	}
	if s > 0 {
		for i := range out {
			out[i] /= s
		}
	}
	return out
}

func normalizeWeights(weights []float64) []float64 {
	s := 0.0
	for _, w := range weights {
		s += w
	}
	out := make([]float64, len(weights))
	if s > 0 {
		for i, w := range weights {
			out[i] = w / s
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

// Optimizer implements portfolio optimization methods.
type Optimizer struct{}

// NewOptimizer creates a new optimizer.
func NewOptimizer() *Optimizer {
	return &Optimizer{}
}

// EqualWeight returns equal weights.
func (o *Optimizer) EqualWeight(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 1.0 / float64(n)
	}
	return w
}

// InverseVol returns inverse-volatility weighted portfolio.
func (o *Optimizer) InverseVol(cov [][]float64, n int) []float64 {
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		vol := math.Sqrt(cov[i][i])
		if vol > 0 {
			w[i] = 1.0 / vol
		} else {
			w[i] = 1.0
		}
	}
	return normalizeWeights(w)
}

// MinVariance finds the minimum variance portfolio.
func (o *Optimizer) MinVariance(cov [][]float64, n int, c ConstraintSet) []float64 {
	inv := invertSymmetric(cov, n)
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1.0
	}
	invOnes := matVecMul(inv, ones)
	denom := dotProduct(ones, invOnes)
	w := make([]float64, n)
	if denom > 0 {
		for i := range w {
			w[i] = invOnes[i] / denom
		}
	} else {
		return o.EqualWeight(n)
	}
	return clampWeights(w, c)
}

// MeanVariance implements mean-variance optimization with a target return.
func (o *Optimizer) MeanVariance(mu []float64, cov [][]float64, n int, targetReturn float64, c ConstraintSet) []float64 {
	inv := invertSymmetric(cov, n)
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1.0
	}
	invMu := matVecMul(inv, mu)
	invOnes := matVecMul(inv, ones)
	a := dotProduct(ones, invOnes)
	b := dotProduct(ones, invMu)
	cc := dotProduct(mu, invMu)
	d := a*cc - b*b
	if d <= 0 || a <= 0 {
		return o.MinVariance(cov, n, c)
	}
	// w = (1/d) * [(c - targetReturn*b) * inv*ones + (targetReturn*a - b) * inv*mu]
	g := (cc - targetReturn*b) / d
	h := (targetReturn*a - b) / d
	w := make([]float64, n)
	for i := 0; i < n; i++ {
		w[i] = g*invOnes[i] + h*invMu[i]
	}
	return clampWeights(w, c)
}

// MaxSharpe finds the maximum Sharpe ratio portfolio.
func (o *Optimizer) MaxSharpe(mu []float64, cov [][]float64, n int, rf float64, c ConstraintSet) []float64 {
	excessMu := make([]float64, n)
	for i := 0; i < n; i++ {
		excessMu[i] = mu[i] - rf
	}
	inv := invertSymmetric(cov, n)
	invExcess := matVecMul(inv, excessMu)
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1.0
	}
	denom := dotProduct(ones, invExcess)
	if denom <= 0 {
		return o.MinVariance(cov, n, c)
	}
	w := make([]float64, n)
	for i := range w {
		w[i] = invExcess[i] / denom
	}
	return clampWeights(w, c)
}

// RiskParity implements Equal Risk Contribution (ERC) via iterative method.
func (o *Optimizer) RiskParity(cov [][]float64, n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 1.0 / float64(n)
	}
	// Iterative Spinu (2013) approach
	for iter := 0; iter < 500; iter++ {
		// Portfolio vol
		pVar := portfolioVariance(w, cov)
		if pVar <= 0 {
			break
		}
		pVol := math.Sqrt(pVar)
		// Marginal risk contribution
		mrc := matVecMul(cov, w)
		// Risk contribution
		rc := make([]float64, n)
		targetRC := pVol / float64(n)
		maxDiff := 0.0
		for i := 0; i < n; i++ {
			rc[i] = w[i] * mrc[i] / pVol
			diff := math.Abs(rc[i] - targetRC)
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if maxDiff < 1e-10 {
			break
		}
		// Update weights: w_i *= (targetRC / rc_i)
		for i := 0; i < n; i++ {
			if rc[i] > 0 {
				w[i] *= targetRC / rc[i]
			}
		}
		w = normalizeWeights(w)
	}
	return w
}

// BlackLitterman implements the Black-Litterman model.
func (o *Optimizer) BlackLitterman(mu []float64, cov [][]float64, n int, views []BlackLittermanView, rf float64, c ConstraintSet) []float64 {
	if len(views) == 0 {
		return o.MaxSharpe(mu, cov, n, rf, c)
	}
	// Risk aversion parameter
	pVol := 0.0
	for i := 0; i < n; i++ {
		pVol += cov[i][i]
	}
	pVol = math.Sqrt(pVol / float64(n))
	delta := (dotProduct(mu, mu)/float64(n) - rf) / (pVol * pVol)
	if delta <= 0 {
		delta = 2.5
	}
	// Equilibrium excess returns: pi = delta * Sigma * w_mkt
	// Use equal weights as market cap proxy
	wMkt := make([]float64, n)
	for i := range wMkt {
		wMkt[i] = 1.0 / float64(n)
	}
	pi := matVecMul(cov, wMkt)
	for i := range pi {
		pi[i] *= delta
	}
	// tau (scaling factor for uncertainty)
	tau := 1.0 / float64(len(mu))
	// Build P matrix and Q vector from views
	k := len(views)
	P := make([][]float64, k)
	Q := make([]float64, k)
	omega := make([][]float64, k)
	for i := range omega {
		omega[i] = make([]float64, k)
	}
	for v, view := range views {
		P[v] = make([]float64, n)
		Q[v] = view.Return
		for j, asset := range view.Assets {
			for a := 0; a < n; a++ {
				// Find asset index (simplified: assume order matches)
				if a < len(view.Weights) && j < len(view.Assets) && asset == view.Assets[j] {
					P[v][a] = view.Weights[j]
				}
			}
		}
		// Omega = diag(confidence^-1)
		if view.Confidence > 0 {
			omega[v][v] = 1.0 / view.Confidence
		} else {
			omega[v][v] = 1.0
		}
	}
	// BL formula: mu_BL = pi + tau*Sigma*P^T * (P*tau*Sigma*P^T + Omega)^-1 * (Q - P*pi)
	// Simplified for small view count
	tauSigma := make([][]float64, n)
	for i := range tauSigma {
		tauSigma[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			tauSigma[i][j] = tau * cov[i][j]
		}
	}
	// P * tauSigma * P^T + Omega (k x k)
	pTauSigPt := make([][]float64, k)
	for i := 0; i < k; i++ {
		pTauSigPt[i] = make([]float64, k)
		tauSigPi := matVecMul(tauSigma, P[i])
		for j := 0; j < k; j++ {
			pTauSigPt[i][j] = dotProduct(tauSigPi, P[j]) + omega[i][j]
		}
	}
	// Invert (k x k)
	pTauSigPtInv := invertSymmetric(pTauSigPt, k)
	// Q - P*pi
	qMinusPpi := make([]float64, k)
	for i := 0; i < k; i++ {
		qMinusPpi[i] = Q[i] - dotProduct(P[i], pi)
	}
	// pTauSigPtInv * (Q - P*pi)
	adjustment := matVecMul(pTauSigPtInv, qMinusPpi)
	// tau*Sigma*P^T * adjustment
	blAdjust := make([]float64, n)
	for i := 0; i < n; i++ {
		for v := 0; v < k; v++ {
			tauSigPt_iv := 0.0
			for j := 0; j < n; j++ {
				tauSigPt_iv += tauSigma[i][j] * P[v][j]
			}
			blAdjust[i] += tauSigPt_iv * adjustment[v]
		}
	}
	// BL expected returns
	blMu := make([]float64, n)
	for i := 0; i < n; i++ {
		blMu[i] = pi[i] + blAdjust[i]
	}
	// Optimize with BL returns
	return o.MaxSharpe(blMu, cov, n, rf, c)
}

// MaxDiversification maximizes the diversification ratio.
func (o *Optimizer) MaxDiversification(cov [][]float64, n int, c ConstraintSet) []float64 {
	// Diversification ratio = sum(w_i * sigma_i) / sigma_p
	// Equivalent to max Sharpe with expected returns = individual volatilities
	vols := make([]float64, n)
	for i := 0; i < n; i++ {
		vols[i] = math.Sqrt(cov[i][i])
	}
	return o.MaxSharpe(vols, cov, n, 0, c)
}

// ---------------------------------------------------------------------------
// Transaction cost model
// ---------------------------------------------------------------------------

// TransactionCost computes the cost of a trade.
func TransactionCost(config CostModelConfig, tradeSize float64) float64 {
	absSize := math.Abs(tradeSize)
	switch config.Type {
	case "sqrt_impact":
		return config.FixedCost + config.PropCost*absSize + config.ImpactCoeff*math.Sqrt(absSize)
	default:
		return config.FixedCost + config.PropCost*absSize
	}
}

// TotalTransactionCost computes total cost for a set of trades.
func TotalTransactionCost(config CostModelConfig, trades []float64) float64 {
	total := 0.0
	for _, t := range trades {
		total += TransactionCost(config, t)
	}
	return total
}

// ---------------------------------------------------------------------------
// Utility: portfolio turnover
// ---------------------------------------------------------------------------

// Turnover computes one-way turnover between old and new weights.
func Turnover(oldWeights, newWeights []float64) float64 {
	n := len(oldWeights)
	if len(newWeights) < n {
		n = len(newWeights)
	}
	t := 0.0
	for i := 0; i < n; i++ {
		t += math.Abs(newWeights[i] - oldWeights[i])
	}
	return t / 2
}

// TrackingError computes ex-ante tracking error vs benchmark.
func TrackingError(weights, benchmark []float64, cov [][]float64) float64 {
	n := len(weights)
	diff := make([]float64, n)
	for i := 0; i < n; i++ {
		bw := 0.0
		if i < len(benchmark) {
			bw = benchmark[i]
		}
		diff[i] = weights[i] - bw
	}
	return portfolioVol(diff, cov)
}

// ---------------------------------------------------------------------------
// Constraint validation
// ---------------------------------------------------------------------------

// ValidateConstraints checks if weights satisfy all constraints.
func ValidateConstraints(weights []float64, assets []string, c ConstraintSet, cov [][]float64) []string {
	var violations []string
	n := len(weights)
	// Box constraints
	for i := 0; i < n; i++ {
		if weights[i] < c.MinWeight-1e-6 {
			violations = append(violations, fmt.Sprintf("%s: weight %.4f below min %.4f", assets[i], weights[i], c.MinWeight))
		}
		if weights[i] > c.MaxWeight+1e-6 {
			violations = append(violations, fmt.Sprintf("%s: weight %.4f above max %.4f", assets[i], weights[i], c.MaxWeight))
		}
	}
	// Sum to 1
	s := 0.0
	for _, w := range weights {
		s += w
	}
	if math.Abs(s-1.0) > 1e-4 {
		violations = append(violations, fmt.Sprintf("weights sum to %.6f, expected 1.0", s))
	}
	// Group constraints
	assetIdx := make(map[string]int, n)
	for i, a := range assets {
		assetIdx[a] = i
	}
	for _, gc := range c.GroupConstraints {
		groupSum := 0.0
		for _, a := range gc.Assets {
			if idx, ok := assetIdx[a]; ok {
				groupSum += weights[idx]
			}
		}
		if groupSum < gc.MinWeight-1e-6 {
			violations = append(violations, fmt.Sprintf("group weight %.4f below min %.4f", groupSum, gc.MinWeight))
		}
		if groupSum > gc.MaxWeight+1e-6 {
			violations = append(violations, fmt.Sprintf("group weight %.4f above max %.4f", groupSum, gc.MaxWeight))
		}
	}
	// Factor exposure
	for _, fc := range c.FactorExposure {
		exposure := 0.0
		for i := 0; i < n && i < len(fc.Loadings); i++ {
			exposure += weights[i] * fc.Loadings[i]
		}
		if exposure < fc.MinExposure-1e-6 {
			violations = append(violations, fmt.Sprintf("factor %s exposure %.4f below min %.4f", fc.FactorName, exposure, fc.MinExposure))
		}
		if exposure > fc.MaxExposure+1e-6 {
			violations = append(violations, fmt.Sprintf("factor %s exposure %.4f above max %.4f", fc.FactorName, exposure, fc.MaxExposure))
		}
	}
	return violations
}

// ---------------------------------------------------------------------------
// Efficient frontier
// ---------------------------------------------------------------------------

// EfficientFrontierPoint is a point on the efficient frontier.
type EfficientFrontierPoint struct {
	TargetReturn float64            `json:"target_return"`
	Volatility   float64            `json:"volatility"`
	Sharpe       float64            `json:"sharpe"`
	Weights      map[string]float64 `json:"weights"`
}

// EfficientFrontier computes points along the efficient frontier.
func (o *Optimizer) EfficientFrontier(mu []float64, cov [][]float64, n int, assets []string, rf float64, c ConstraintSet, points int) []EfficientFrontierPoint {
	// Find min and max achievable returns
	minRet := math.MaxFloat64
	maxRet := -math.MaxFloat64
	for _, m := range mu {
		if m < minRet {
			minRet = m
		}
		if m > maxRet {
			maxRet = m
		}
	}
	if points < 2 {
		points = 20
	}
	step := (maxRet - minRet) / float64(points-1)
	frontier := make([]EfficientFrontierPoint, 0, points)
	for i := 0; i < points; i++ {
		target := minRet + step*float64(i)
		w := o.MeanVariance(mu, cov, n, target, c)
		vol := portfolioVol(w, cov)
		ret := dotProduct(w, mu)
		sharpe := 0.0
		if vol > 0 {
			sharpe = (ret - rf) / vol
		}
		wMap := make(map[string]float64, n)
		for j, a := range assets {
			wMap[a] = w[j]
		}
		frontier = append(frontier, EfficientFrontierPoint{
			TargetReturn: target,
			Volatility:   vol,
			Sharpe:       sharpe,
			Weights:      wMap,
		})
	}
	return frontier
}
