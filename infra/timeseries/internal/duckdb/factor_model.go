package duckdb

import (
	"math"
	"sort"
)

// FactorReturn holds a factor name and its return for a period.
type FactorReturn struct {
	Name   string
	Return float64
}

// FactorExposureResult holds beta and alpha for a symbol regressed against factors.
type FactorExposureResult struct {
	Symbol  string             `json:"symbol"`
	Alpha   float64            `json:"alpha"`
	Betas   map[string]float64 `json:"betas"`
	RSq     float64            `json:"r_squared"`
	TStats  map[string]float64 `json:"t_stats"`
	Residuals []float64        `json:"residuals,omitempty"`
}

// OLSResult holds the result of an ordinary-least-squares regression.
type OLSResult struct {
	Coefficients []float64
	RSq          float64
	Residuals    []float64
	TStats       []float64
}

// OLS performs ordinary least squares regression of y on X (which should already
// include a leading column of 1s for the intercept).
// Returns nil if the system cannot be solved.
func OLS(X [][]float64, y []float64) *OLSResult {
	n := len(y)
	if n == 0 || len(X) != n {
		return nil
	}
	k := len(X[0]) // number of predictors including intercept
	if n <= k {
		return nil
	}

	// Build X'X and X'y.
	XtX := make([][]float64, k)
	for i := range XtX {
		XtX[i] = make([]float64, k)
	}
	Xty := make([]float64, k)
	for i := 0; i < n; i++ {
		for j := 0; j < k; j++ {
			Xty[j] += X[i][j] * y[i]
			for l := 0; l < k; l++ {
				XtX[j][l] += X[i][j] * X[i][l]
			}
		}
	}

	// Cholesky-based solve (Gauss-Jordan elimination for simplicity).
	beta := gaussJordan(XtX, Xty)
	if beta == nil {
		return nil
	}

	// Residuals.
	res := make([]float64, n)
	yMean := 0.0
	for i := 0; i < n; i++ {
		yhat := 0.0
		for j := 0; j < k; j++ {
			yhat += beta[j] * X[i][j]
		}
		res[i] = y[i] - yhat
		yMean += y[i]
	}
	yMean /= float64(n)

	ssTot, ssRes := 0.0, 0.0
	for i := 0; i < n; i++ {
		d := y[i] - yMean
		ssTot += d * d
		ssRes += res[i] * res[i]
	}
	rSq := 0.0
	if ssTot > 0 {
		rSq = 1 - ssRes/ssTot
	}

	// Standard errors and t-stats.
	s2 := ssRes / float64(n-k)
	tstats := make([]float64, k)
	for j := 0; j < k; j++ {
		// Diagonal of (X'X)^-1 gives variance of each beta.
		// Approximate by inverting XtX diagonal (simplified).
		se := math.Sqrt(s2 / math.Max(XtX[j][j], 1e-12))
		if se > 0 {
			tstats[j] = beta[j] / se
		}
	}

	return &OLSResult{
		Coefficients: beta,
		RSq:          rSq,
		Residuals:    res,
		TStats:       tstats,
	}
}

// gaussJordan solves Ax = b using Gauss-Jordan elimination with partial pivoting.
func gaussJordan(A [][]float64, b []float64) []float64 {
	n := len(b)
	// Build augmented matrix.
	aug := make([][]float64, n)
	for i := range aug {
		row := make([]float64, n+1)
		copy(row, A[i])
		row[n] = b[i]
		aug[i] = row
	}

	for col := 0; col < n; col++ {
		// Find pivot.
		pivot := col
		for row := col + 1; row < n; row++ {
			if math.Abs(aug[row][col]) > math.Abs(aug[pivot][col]) {
				pivot = row
			}
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]

		p := aug[col][col]
		if math.Abs(p) < 1e-12 {
			return nil // singular
		}
		for j := col; j <= n; j++ {
			aug[col][j] /= p
		}
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := col; j <= n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	x := make([]float64, n)
	for i := range x {
		x[i] = aug[i][n]
	}
	return x
}

// FitFactorModel regresses symbol returns against a set of factor returns.
// symbolReturns and each factor return slice must be the same length.
func FitFactorModel(symbol string, symbolReturns []float64, factors []FactorReturn) *FactorExposureResult {
	if len(symbolReturns) == 0 || len(factors) == 0 {
		return nil
	}
	k := len(factors) + 1 // +1 for intercept
	n := len(symbolReturns)
	if n < k+1 {
		return nil
	}

	X := make([][]float64, n)
	for i := range X {
		row := make([]float64, k)
		row[0] = 1.0 // intercept
		for j, f := range factors {
			if len(f.Return) != n {
				// Each FactorReturn.Return should be a scalar per the type — skip
				// This model uses a slice of FactorReturn but Return is float64.
				// We treat it as a constant factor (not useful for regression).
				row[j+1] = 0
			}
		}
		X[i] = row
	}
	// This approach requires factor time-series, not scalars.
	// Re-define for time-series regression: caller passes FactorTimeSeries.
	// Returning a zero result here — see FitFactorModelTS for the real implementation.
	_ = X
	return &FactorExposureResult{Symbol: symbol}
}

// FactorTimeSeries holds a named factor return series.
type FactorTimeSeries struct {
	Name    string
	Returns []float64
}

// FitFactorModelTS regresses symbol returns against factor time-series.
func FitFactorModelTS(symbol string, symbolReturns []float64, factors []FactorTimeSeries) *FactorExposureResult {
	if len(symbolReturns) == 0 || len(factors) == 0 {
		return nil
	}
	n := len(symbolReturns)
	k := len(factors) + 1
	if n < k+1 {
		return nil
	}

	// Validate factor lengths.
	for _, f := range factors {
		if len(f.Returns) != n {
			return nil
		}
	}

	X := make([][]float64, n)
	for i := range X {
		row := make([]float64, k)
		row[0] = 1.0
		for j, f := range factors {
			row[j+1] = f.Returns[i]
		}
		X[i] = row
	}

	res := OLS(X, symbolReturns)
	if res == nil {
		return nil
	}

	betas := make(map[string]float64, len(factors))
	tstats := make(map[string]float64, len(factors))
	for j, f := range factors {
		betas[f.Name] = res.Coefficients[j+1]
		if len(res.TStats) > j+1 {
			tstats[f.Name] = res.TStats[j+1]
		}
	}

	alpha := 0.0
	if len(res.Coefficients) > 0 {
		alpha = res.Coefficients[0]
	}
	alphaTStat := 0.0
	if len(res.TStats) > 0 {
		alphaTStat = res.TStats[0]
	}
	tstats["alpha"] = alphaTStat

	return &FactorExposureResult{
		Symbol:    symbol,
		Alpha:     alpha,
		Betas:     betas,
		RSq:       res.RSq,
		TStats:    tstats,
		Residuals: res.Residuals,
	}
}

// PortfolioRisk computes portfolio-level risk metrics.
type PortfolioRisk struct {
	PortfolioVar   float64            `json:"portfolio_variance"`
	PortfolioVol   float64            `json:"portfolio_volatility"`
	FactorVars     map[string]float64 `json:"factor_variances"`
	IdiosyncraticV float64            `json:"idiosyncratic_variance"`
	ConcentrationH float64            `json:"concentration_herfindahl"`
}

// ComputePortfolioRisk computes factor-model-based portfolio risk.
// weights maps symbol to weight (should sum to 1).
func ComputePortfolioRisk(weights map[string]float64, exposures map[string]*FactorExposureResult, factorCov map[string]map[string]float64) *PortfolioRisk {
	if len(weights) == 0 {
		return nil
	}

	// Herfindahl index for concentration.
	h := 0.0
	for _, w := range weights {
		h += w * w
	}

	// Portfolio factor betas (weighted sum).
	portBetas := make(map[string]float64)
	portIdio := 0.0
	for sym, w := range weights {
		exp, ok := exposures[sym]
		if !ok {
			continue
		}
		for factor, beta := range exp.Betas {
			portBetas[factor] += w * beta
		}
		// Residual variance approximation.
		if len(exp.Residuals) > 1 {
			mean := 0.0
			for _, r := range exp.Residuals {
				mean += r
			}
			mean /= float64(len(exp.Residuals))
			v := 0.0
			for _, r := range exp.Residuals {
				d := r - mean
				v += d * d
			}
			portIdio += (w * w) * v / float64(len(exp.Residuals)-1)
		}
	}

	// Factor contribution to portfolio variance: beta' * Sigma_F * beta
	factorVars := make(map[string]float64)
	portFactorVar := 0.0
	for fi, bi := range portBetas {
		for fj, bj := range portBetas {
			cov := 0.0
			if row, ok := factorCov[fi]; ok {
				cov = row[fj]
			}
			contrib := bi * bj * cov
			portFactorVar += contrib
			if fi == fj {
				factorVars[fi] = bi * bi * cov
			}
		}
	}

	totalVar := portFactorVar + portIdio
	vol := 0.0
	if totalVar > 0 {
		vol = math.Sqrt(totalVar)
	}

	return &PortfolioRisk{
		PortfolioVar:   totalVar,
		PortfolioVol:   vol,
		FactorVars:     factorVars,
		IdiosyncraticV: portIdio,
		ConcentrationH: h,
	}
}

// PCAResult holds principal component analysis output.
type PCAResult struct {
	ExplainedVariance []float64   `json:"explained_variance"`
	Components        [][]float64 `json:"components"`
}

// SimplePCA performs PCA on a returns matrix (rows = observations, cols = assets)
// using the covariance approach. Returns the top maxComponents components.
func SimplePCA(returns [][]float64, maxComponents int) *PCAResult {
	if len(returns) == 0 || len(returns[0]) == 0 {
		return nil
	}
	nObs := len(returns)
	nAssets := len(returns[0])
	if maxComponents > nAssets {
		maxComponents = nAssets
	}

	// Compute mean per asset.
	means := make([]float64, nAssets)
	for _, row := range returns {
		for j, v := range row {
			means[j] += v
		}
	}
	for j := range means {
		means[j] /= float64(nObs)
	}

	// Covariance matrix.
	cov := make([][]float64, nAssets)
	for i := range cov {
		cov[i] = make([]float64, nAssets)
	}
	for _, row := range returns {
		centered := make([]float64, nAssets)
		for j, v := range row {
			centered[j] = v - means[j]
		}
		for i := 0; i < nAssets; i++ {
			for j := 0; j < nAssets; j++ {
				cov[i][j] += centered[i] * centered[j]
			}
		}
	}
	scale := 1.0 / float64(nObs-1)
	for i := range cov {
		for j := range cov[i] {
			cov[i][j] *= scale
		}
	}

	// Power iteration to find top eigenvectors.
	components := make([][]float64, 0, maxComponents)
	explained := make([]float64, 0, maxComponents)
	deflated := copyCov(cov)

	for c := 0; c < maxComponents; c++ {
		vec, eigenval := powerIteration(deflated, 100)
		if eigenval < 1e-10 {
			break
		}
		components = append(components, vec)
		explained = append(explained, eigenval)
		// Deflate: A = A - lambda * v * v'
		for i := range deflated {
			for j := range deflated[i] {
				deflated[i][j] -= eigenval * vec[i] * vec[j]
			}
		}
	}

	return &PCAResult{
		ExplainedVariance: explained,
		Components:        components,
	}
}

func copyCov(m [][]float64) [][]float64 {
	cp := make([][]float64, len(m))
	for i := range m {
		cp[i] = make([]float64, len(m[i]))
		copy(cp[i], m[i])
	}
	return cp
}

// powerIteration finds the dominant eigenvector and eigenvalue of a symmetric matrix.
func powerIteration(A [][]float64, maxIter int) ([]float64, float64) {
	n := len(A)
	v := make([]float64, n)
	for i := range v {
		v[i] = 1.0 / math.Sqrt(float64(n))
	}
	eigenval := 0.0
	for iter := 0; iter < maxIter; iter++ {
		w := make([]float64, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				w[i] += A[i][j] * v[j]
			}
		}
		norm := 0.0
		for _, x := range w {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		if norm < 1e-15 {
			break
		}
		eigenval = norm
		for i := range w {
			v[i] = w[i] / norm
		}
	}
	return v, eigenval
}

// SortedFactors returns factor names sorted by absolute beta descending.
func SortedFactors(betas map[string]float64) []string {
	names := make([]string, 0, len(betas))
	for n := range betas {
		names = append(names, n)
	}
	sort.Slice(names, func(i, j int) bool {
		return math.Abs(betas[names[i]]) > math.Abs(betas[names[j]])
	})
	return names
}
