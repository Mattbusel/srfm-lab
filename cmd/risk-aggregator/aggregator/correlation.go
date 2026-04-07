package aggregator

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/srfm-lab/risk-aggregator/model"
)

// returnsAPIResponse is the JSON returned by the risk API at
// GET /returns/history?window=30
type returnsAPIResponse struct {
	Instruments []string    `json:"instruments"`
	// Dates in ascending order, ISO-8601 strings
	Dates       []string    `json:"dates"`
	// Returns matrix: rows = dates, cols = instruments (row-major)
	Returns     [][]float64 `json:"returns"`
}

// defaultCorrelationWindow is the rolling window in trading days.
const defaultCorrelationWindow = 30

// GetCorrelationMatrix fetches 30-day rolling return history from the risk
// API and computes the Pearson correlation matrix for all instruments.
func (a *Aggregator) GetCorrelationMatrix(ctx context.Context) (model.CorrelationMatrix, error) {
	url := fmt.Sprintf("%s/returns/history?window=%d", a.cfg.RiskAPIBase, defaultCorrelationWindow)

	resp, err := a.httpClient.R().SetContext(ctx).Get(url)
	if err != nil {
		return model.CorrelationMatrix{}, fmt.Errorf("GET returns/history: %w", err)
	}
	if resp.IsError() {
		return model.CorrelationMatrix{}, fmt.Errorf("risk API returned %d", resp.StatusCode())
	}

	var apiResp returnsAPIResponse
	if err := json.Unmarshal(resp.Body(), &apiResp); err != nil {
		return model.CorrelationMatrix{}, fmt.Errorf("unmarshal returns: %w", err)
	}

	if len(apiResp.Returns) < 2 {
		return model.CorrelationMatrix{}, fmt.Errorf("insufficient return history: %d rows", len(apiResp.Returns))
	}

	n := len(apiResp.Instruments)
	corrValues := pearsonMatrix(apiResp.Returns, n)

	log.Debug().Int("n_instruments", n).Int("n_dates", len(apiResp.Dates)).
		Msg("correlation matrix computed")

	return model.CorrelationMatrix{
		AsOf:        time.Now().UTC(),
		WindowDays:  defaultCorrelationWindow,
		Instruments: apiResp.Instruments,
		Values:      corrValues,
	}, nil
}

// pearsonMatrix computes an n x n Pearson correlation matrix from a T x n
// returns matrix supplied as a slice of row slices.
// The result is stored row-major in a flat slice of length n*n.
func pearsonMatrix(returns [][]float64, n int) []float64 {
	T := len(returns)

	// Extract column slices.
	cols := make([][]float64, n)
	for j := 0; j < n; j++ {
		cols[j] = make([]float64, T)
		for i := 0; i < T; i++ {
			if j < len(returns[i]) {
				cols[j][i] = returns[i][j]
			}
		}
	}

	result := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				result[i*n+j] = 1.0
				continue
			}
			if j < i {
				// Symmetric -- copy from upper triangle.
				result[i*n+j] = result[j*n+i]
				continue
			}
			result[i*n+j] = pearson(cols[i], cols[j])
		}
	}
	return result
}

// pearson computes the Pearson correlation coefficient between two equal-length
// slices. Returns 0 if either slice has zero variance.
func pearson(x, y []float64) float64 {
	n := len(x)
	if n == 0 {
		return 0
	}

	var sumX, sumY float64
	for i := 0; i < n; i++ {
		sumX += x[i]
		sumY += y[i]
	}
	meanX := sumX / float64(n)
	meanY := sumY / float64(n)

	var cov, varX, varY float64
	for i := 0; i < n; i++ {
		dx := x[i] - meanX
		dy := y[i] - meanY
		cov += dx * dy
		varX += dx * dx
		varY += dy * dy
	}

	denom := math.Sqrt(varX * varY)
	if denom == 0 {
		return 0
	}
	r := cov / denom
	// Clamp to [-1, 1] for floating-point safety.
	if r > 1 {
		return 1
	}
	if r < -1 {
		return -1
	}
	return r
}

// GetPCA fetches the correlation matrix and extracts principal components via
// the Jacobi eigenvalue algorithm. Returns eigenvalues in descending order
// with loadings and cumulative variance explained.
func (a *Aggregator) GetPCA(ctx context.Context) (model.PCAResult, error) {
	cm, err := a.GetCorrelationMatrix(ctx)
	if err != nil {
		return model.PCAResult{}, err
	}

	n := len(cm.Instruments)
	eigenvalues, eigenvectors := jacobiEigen(cm.Values, n)

	// Sort by descending eigenvalue.
	type ev struct {
		val float64
		vec []float64
	}
	pairs := make([]ev, n)
	for i := 0; i < n; i++ {
		vec := make([]float64, n)
		for j := 0; j < n; j++ {
			vec[j] = eigenvectors[j*n+i]
		}
		pairs[i] = ev{val: eigenvalues[i], vec: vec}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].val > pairs[j].val })

	totalVar := 0.0
	for _, p := range pairs {
		totalVar += p.val
	}

	varExp := make([]float64, n)
	cumVar := make([]float64, n)
	loadings := make([][]float64, n)
	eigenvals := make([]float64, n)
	running := 0.0
	for i, p := range pairs {
		eigenvals[i] = p.val
		ve := 0.0
		if totalVar > 0 {
			ve = p.val / totalVar
		}
		varExp[i] = ve
		running += ve
		cumVar[i] = running
		loadings[i] = p.vec
	}

	return model.PCAResult{
		AsOf:              time.Now().UTC(),
		Instruments:       cm.Instruments,
		Eigenvalues:       eigenvals,
		VarianceExplained: varExp,
		CumulativeVar:     cumVar,
		Loadings:          loadings,
	}, nil
}

// jacobiEigen computes eigenvalues and eigenvectors of a symmetric n x n
// matrix supplied as a row-major flat slice of length n*n.
// Uses the classic Jacobi rotation method; converges well for small n.
// Returns eigenvalues slice and column eigenvectors in a flat row-major slice.
func jacobiEigen(flatA []float64, n int) ([]float64, []float64) {
	const maxIter = 100
	const tol = 1e-10

	// Copy A so we don't mutate the caller's slice.
	a := make([]float64, n*n)
	copy(a, flatA)

	// V starts as identity -- columns will become eigenvectors.
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1.0
	}

	for iter := 0; iter < maxIter; iter++ {
		// Find the largest off-diagonal element.
		maxVal := 0.0
		p, q := 0, 1
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				if math.Abs(a[i*n+j]) > maxVal {
					maxVal = math.Abs(a[i*n+j])
					p, q = i, j
				}
			}
		}
		if maxVal < tol {
			break
		}

		// Compute rotation angle.
		theta := 0.0
		app := a[p*n+p]
		aqq := a[q*n+q]
		apq := a[p*n+q]
		if app != aqq {
			theta = 0.5 * math.Atan2(2*apq, app-aqq)
		} else {
			theta = math.Pi / 4
		}
		c := math.Cos(theta)
		s := math.Sin(theta)

		// Apply Jacobi rotation to A.
		newA := make([]float64, n*n)
		copy(newA, a)

		for r := 0; r < n; r++ {
			if r != p && r != q {
				arp := c*a[r*n+p] + s*a[r*n+q]
				arq := -s*a[r*n+p] + c*a[r*n+q]
				newA[r*n+p] = arp
				newA[p*n+r] = arp
				newA[r*n+q] = arq
				newA[q*n+r] = arq
			}
		}
		newA[p*n+p] = c*c*app + 2*s*c*apq + s*s*aqq
		newA[q*n+q] = s*s*app - 2*s*c*apq + c*c*aqq
		newA[p*n+q] = 0
		newA[q*n+p] = 0

		a = newA

		// Accumulate rotations into V.
		for r := 0; r < n; r++ {
			vrp := c*v[r*n+p] + s*v[r*n+q]
			vrq := -s*v[r*n+p] + c*v[r*n+q]
			v[r*n+p] = vrp
			v[r*n+q] = vrq
		}
	}

	// Extract diagonal eigenvalues.
	eigenvalues := make([]float64, n)
	for i := 0; i < n; i++ {
		eigenvalues[i] = a[i*n+i]
	}
	return eigenvalues, v
}
