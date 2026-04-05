package scheduler

import (
	"fmt"
	"sync"
)

// ModuleBudget describes the CPU-time allocation for a single module type.
type ModuleBudget struct {
	// Fraction is the share of total parallel experiment slots (0–1).
	Fraction float64
	// MaxConcurrent is the derived maximum number of simultaneous experiments.
	MaxConcurrent int
}

// defaultBudgets defines the per-module CPU time budget fractions.
// They must sum to 1.0.
var defaultBudgets = map[string]float64{
	"genome":         0.40,
	"counterfactual": 0.20,
	"shadow":         0.20,
	"causal":         0.10,
	"academic":       0.05,
	"serendipity":    0.05,
}

// BudgetManager tracks running experiment counts per module type and enforces
// the per-module CPU budget fractions.  It is safe for concurrent use.
//
// The total parallelism is set at construction time (totalSlots). Each module
// receives floor(fraction * totalSlots) concurrent slots, with at least one
// slot guaranteed for every registered module.
type BudgetManager struct {
	mu       sync.Mutex
	budgets  map[string]ModuleBudget
	running  map[string]int
	totalSlots int
}

// NewBudgetManager constructs a BudgetManager with the given total parallelism.
// totalSlots is typically the number of CPU cores available for experiments.
func NewBudgetManager(totalSlots int) *BudgetManager {
	if totalSlots <= 0 {
		totalSlots = 8
	}

	budgets := make(map[string]ModuleBudget, len(defaultBudgets))
	for module, fraction := range defaultBudgets {
		max := int(fraction * float64(totalSlots))
		if max < 1 {
			max = 1
		}
		budgets[module] = ModuleBudget{
			Fraction:      fraction,
			MaxConcurrent: max,
		}
	}

	return &BudgetManager{
		budgets:    budgets,
		running:    make(map[string]int),
		totalSlots: totalSlots,
	}
}

// CanRun reports whether the given moduleType has a free slot.
// Returns false for unknown module types.
func (bm *BudgetManager) CanRun(moduleType string) bool {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	budget, ok := bm.budgets[moduleType]
	if !ok {
		return false
	}
	return bm.running[moduleType] < budget.MaxConcurrent
}

// Acquire reserves a slot for moduleType. Returns an error if no slot is
// available or the module type is unknown. The caller must call Release when
// the experiment finishes.
func (bm *BudgetManager) Acquire(moduleType string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	budget, ok := bm.budgets[moduleType]
	if !ok {
		return fmt.Errorf("budget_manager: unknown module type %q", moduleType)
	}
	if bm.running[moduleType] >= budget.MaxConcurrent {
		return fmt.Errorf("budget_manager: no slot available for %q (%d/%d running)",
			moduleType, bm.running[moduleType], budget.MaxConcurrent)
	}
	bm.running[moduleType]++
	return nil
}

// Release frees a slot previously reserved by Acquire.
func (bm *BudgetManager) Release(moduleType string) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if bm.running[moduleType] > 0 {
		bm.running[moduleType]--
	}
}

// Stats returns a snapshot of current usage per module type.
func (bm *BudgetManager) Stats() map[string]interface{} {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	out := make(map[string]interface{}, len(bm.budgets))
	for module, budget := range bm.budgets {
		out[module] = map[string]interface{}{
			"fraction":       budget.Fraction,
			"max_concurrent": budget.MaxConcurrent,
			"running":        bm.running[module],
			"available":      budget.MaxConcurrent - bm.running[module],
		}
	}
	return out
}

// TotalRunning returns the total number of running experiments across all modules.
func (bm *BudgetManager) TotalRunning() int {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	total := 0
	for _, n := range bm.running {
		total += n
	}
	return total
}

// SetBudget overrides the budget fraction for a module. maxConcurrent is
// recomputed from the new fraction and the original totalSlots.  This is
// provided for runtime tuning without restarting the scheduler.
func (bm *BudgetManager) SetBudget(moduleType string, fraction float64) error {
	if fraction < 0 || fraction > 1 {
		return fmt.Errorf("budget_manager: fraction must be between 0 and 1, got %f", fraction)
	}
	bm.mu.Lock()
	defer bm.mu.Unlock()

	max := int(fraction * float64(bm.totalSlots))
	if max < 1 {
		max = 1
	}
	bm.budgets[moduleType] = ModuleBudget{
		Fraction:      fraction,
		MaxConcurrent: max,
	}
	return nil
}
