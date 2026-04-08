package persistence

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

// ──────────────────────────────────────────────────────────────────────────────
// Domain types
// ──────────────────────────────────────────────────────────────────────────────

type StrategyStatus int

const (
	StatusCandidate StrategyStatus = iota
	StatusActive
	StatusChampion
	StatusRetired
	StatusArchived
)

func (s StrategyStatus) String() string {
	return [...]string{"candidate", "active", "champion", "retired", "archived"}[s]
}

func ParseStatus(s string) StrategyStatus {
	switch strings.ToLower(s) {
	case "active":
		return StatusActive
	case "champion":
		return StatusChampion
	case "retired":
		return StatusRetired
	case "archived":
		return StatusArchived
	default:
		return StatusCandidate
	}
}

type PerformanceSnapshot struct {
	Timestamp   time.Time          `json:"timestamp"`
	Cycle       int                `json:"cycle"`
	Sharpe      float64            `json:"sharpe"`
	Calmar      float64            `json:"calmar"`
	MaxDrawdown float64            `json:"max_drawdown"`
	Returns     float64            `json:"returns"`
	Volatility  float64            `json:"volatility"`
	Sortino     float64            `json:"sortino"`
	WinRate     float64            `json:"win_rate"`
	Turnover    float64            `json:"turnover"`
	Extra       map[string]float64 `json:"extra,omitempty"`
}

type MutationRecord struct {
	Timestamp   time.Time          `json:"timestamp"`
	Type        string             `json:"type"` // crossover, mutation, parameter_tweak
	ParentIDs   []string           `json:"parent_ids"`
	Description string             `json:"description"`
	Parameters  map[string]float64 `json:"parameters,omitempty"`
}

type StrategyConfig struct {
	Type       string             `json:"type"` // momentum, mean_rev, breakout, composite
	Symbols    []string           `json:"symbols"`
	Parameters map[string]float64 `json:"parameters"`
	Signals    []string           `json:"signals,omitempty"`
	Weights    map[string]float64 `json:"weights,omitempty"`
}

type StrategyRecord struct {
	ID          string                `json:"id"`
	Name        string                `json:"name"`
	Version     int                   `json:"version"`
	Config      StrategyConfig        `json:"config"`
	Performance []PerformanceSnapshot `json:"performance_history"`
	Status      StrategyStatus        `json:"status"`
	Tags        []string              `json:"tags"`
	Lineage     []MutationRecord      `json:"lineage"`
	ParentID    string                `json:"parent_id,omitempty"`
	ChildIDs    []string              `json:"child_ids,omitempty"`
	CreatedAt   time.Time             `json:"created_at"`
	UpdatedAt   time.Time             `json:"updated_at"`
	Notes       string                `json:"notes,omitempty"`
}

// LatestPerformance returns the most recent performance snapshot.
func (sr *StrategyRecord) LatestPerformance() PerformanceSnapshot {
	if len(sr.Performance) == 0 {
		return PerformanceSnapshot{}
	}
	return sr.Performance[len(sr.Performance)-1]
}

// AvgSharpe returns the average Sharpe across all evaluation cycles.
func (sr *StrategyRecord) AvgSharpe() float64 {
	if len(sr.Performance) == 0 {
		return 0
	}
	s := 0.0
	for _, p := range sr.Performance {
		s += p.Sharpe
	}
	return s / float64(len(sr.Performance))
}

// SharpeStability returns the standard deviation of Sharpe across cycles.
func (sr *StrategyRecord) SharpeStability() float64 {
	if len(sr.Performance) < 2 {
		return 0
	}
	sharpes := make([]float64, len(sr.Performance))
	for i, p := range sr.Performance {
		sharpes[i] = p.Sharpe
	}
	return stddevF(sharpes)
}

// HasTag checks if strategy has a given tag.
func (sr *StrategyRecord) HasTag(tag string) bool {
	for _, t := range sr.Tags {
		if t == tag {
			return true
		}
	}
	return false
}

// ──────────────────────────────────────────────────────────────────────────────
// Strategy store (in-memory with SQLite-compatible serialization)
// ──────────────────────────────────────────────────────────────────────────────

type StrategyStore struct {
	mu         sync.RWMutex
	strategies map[string]*StrategyRecord
	byTag      map[string]map[string]bool // tag -> set of IDs
	versions   map[string][]string        // base_name -> [id_v1, id_v2, ...]
	nextID     int64
}

func NewStrategyStore() *StrategyStore {
	return &StrategyStore{
		strategies: make(map[string]*StrategyRecord),
		byTag:      make(map[string]map[string]bool),
		versions:   make(map[string][]string),
	}
}

// Create adds a new strategy record.
func (ss *StrategyStore) Create(rec StrategyRecord) string {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	if rec.ID == "" {
		ss.nextID++
		rec.ID = fmt.Sprintf("strat_%d_%d", time.Now().UnixNano(), ss.nextID)
	}
	if rec.Version == 0 {
		rec.Version = 1
	}
	now := time.Now()
	rec.CreatedAt = now
	rec.UpdatedAt = now

	ss.strategies[rec.ID] = &rec
	ss.indexTags(&rec)
	ss.versions[rec.Name] = append(ss.versions[rec.Name], rec.ID)

	return rec.ID
}

// Get retrieves a strategy by ID.
func (ss *StrategyStore) Get(id string) (*StrategyRecord, bool) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	r, ok := ss.strategies[id]
	if !ok {
		return nil, false
	}
	cp := *r
	return &cp, true
}

// Update replaces a strategy record (same ID).
func (ss *StrategyStore) Update(rec StrategyRecord) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	if _, ok := ss.strategies[rec.ID]; !ok {
		return fmt.Errorf("strategy %s not found", rec.ID)
	}
	rec.UpdatedAt = time.Now()
	// Remove old tags
	ss.deindexTags(rec.ID)
	ss.strategies[rec.ID] = &rec
	ss.indexTags(&rec)
	return nil
}

// Delete removes a strategy.
func (ss *StrategyStore) Delete(id string) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	ss.deindexTags(id)
	delete(ss.strategies, id)
}

// List returns all strategies.
func (ss *StrategyStore) List() []*StrategyRecord {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	out := make([]*StrategyRecord, 0, len(ss.strategies))
	for _, r := range ss.strategies {
		cp := *r
		out = append(out, &cp)
	}
	return out
}

// Count returns total strategies.
func (ss *StrategyStore) Count() int {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	return len(ss.strategies)
}

// ──────────────────────────────────────────────────────────────────────────────
// Versioning
// ──────────────────────────────────────────────────────────────────────────────

// NewVersion creates a new version of an existing strategy.
func (ss *StrategyStore) NewVersion(parentID string, config StrategyConfig, mutation MutationRecord) (string, error) {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	parent, ok := ss.strategies[parentID]
	if !ok {
		return "", fmt.Errorf("parent %s not found", parentID)
	}

	ss.nextID++
	newID := fmt.Sprintf("strat_%d_%d", time.Now().UnixNano(), ss.nextID)
	now := time.Now()

	child := &StrategyRecord{
		ID:        newID,
		Name:      parent.Name,
		Version:   parent.Version + 1,
		Config:    config,
		Status:    StatusCandidate,
		Tags:      append([]string{}, parent.Tags...),
		Lineage:   append(append([]MutationRecord{}, parent.Lineage...), mutation),
		ParentID:  parentID,
		CreatedAt: now,
		UpdatedAt: now,
	}

	ss.strategies[newID] = child
	parent.ChildIDs = append(parent.ChildIDs, newID)
	ss.indexTags(child)
	ss.versions[child.Name] = append(ss.versions[child.Name], newID)

	return newID, nil
}

// GetVersionHistory returns all versions of a strategy by name.
func (ss *StrategyStore) GetVersionHistory(name string) []*StrategyRecord {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	ids := ss.versions[name]
	out := make([]*StrategyRecord, 0, len(ids))
	for _, id := range ids {
		if r, ok := ss.strategies[id]; ok {
			cp := *r
			out = append(out, &cp)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version < out[j].Version })
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Tagging
// ──────────────────────────────────────────────────────────────────────────────

func (ss *StrategyStore) AddTag(id, tag string) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	r, ok := ss.strategies[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	for _, t := range r.Tags {
		if t == tag {
			return nil // already tagged
		}
	}
	r.Tags = append(r.Tags, tag)
	r.UpdatedAt = time.Now()
	if ss.byTag[tag] == nil {
		ss.byTag[tag] = make(map[string]bool)
	}
	ss.byTag[tag][id] = true
	return nil
}

func (ss *StrategyStore) RemoveTag(id, tag string) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	r, ok := ss.strategies[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	for i, t := range r.Tags {
		if t == tag {
			r.Tags = append(r.Tags[:i], r.Tags[i+1:]...)
			break
		}
	}
	if ss.byTag[tag] != nil {
		delete(ss.byTag[tag], id)
	}
	r.UpdatedAt = time.Now()
	return nil
}

func (ss *StrategyStore) indexTags(r *StrategyRecord) {
	for _, tag := range r.Tags {
		if ss.byTag[tag] == nil {
			ss.byTag[tag] = make(map[string]bool)
		}
		ss.byTag[tag][r.ID] = true
	}
}

func (ss *StrategyStore) deindexTags(id string) {
	for tag, ids := range ss.byTag {
		delete(ids, id)
		if len(ids) == 0 {
			delete(ss.byTag, tag)
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Performance history
// ──────────────────────────────────────────────────────────────────────────────

// AppendPerformance adds a new performance snapshot to a strategy.
func (ss *StrategyStore) AppendPerformance(id string, snap PerformanceSnapshot) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	r, ok := ss.strategies[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	if snap.Timestamp.IsZero() {
		snap.Timestamp = time.Now()
	}
	r.Performance = append(r.Performance, snap)
	r.UpdatedAt = time.Now()
	return nil
}

// GetPerformanceHistory returns performance history for a strategy.
func (ss *StrategyStore) GetPerformanceHistory(id string) []PerformanceSnapshot {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	r, ok := ss.strategies[id]
	if !ok {
		return nil
	}
	out := make([]PerformanceSnapshot, len(r.Performance))
	copy(out, r.Performance)
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Search and query
// ──────────────────────────────────────────────────────────────────────────────

type SearchCriteria struct {
	Tags        []string        `json:"tags,omitempty"`
	Status      *StrategyStatus `json:"status,omitempty"`
	MinSharpe   *float64        `json:"min_sharpe,omitempty"`
	MaxDrawdown *float64        `json:"max_drawdown,omitempty"`
	MinWinRate  *float64        `json:"min_win_rate,omitempty"`
	StrategyType string         `json:"strategy_type,omitempty"`
	SortBy      string          `json:"sort_by,omitempty"` // sharpe, calmar, returns, created_at
	SortDesc    bool            `json:"sort_desc"`
	Limit       int             `json:"limit,omitempty"`
}

func (ss *StrategyStore) Search(criteria SearchCriteria) []*StrategyRecord {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	var candidates []*StrategyRecord

	// If searching by tags, start from tag index
	if len(criteria.Tags) > 0 {
		idSet := make(map[string]bool)
		for _, tag := range criteria.Tags {
			if ids, ok := ss.byTag[tag]; ok {
				for id := range ids {
					idSet[id] = true
				}
			}
		}
		for id := range idSet {
			if r, ok := ss.strategies[id]; ok {
				candidates = append(candidates, r)
			}
		}
	} else {
		for _, r := range ss.strategies {
			candidates = append(candidates, r)
		}
	}

	// Filter
	var results []*StrategyRecord
	for _, r := range candidates {
		if criteria.Status != nil && r.Status != *criteria.Status {
			continue
		}
		if criteria.StrategyType != "" && r.Config.Type != criteria.StrategyType {
			continue
		}
		latest := r.LatestPerformance()
		if criteria.MinSharpe != nil && latest.Sharpe < *criteria.MinSharpe {
			continue
		}
		if criteria.MaxDrawdown != nil && latest.MaxDrawdown < *criteria.MaxDrawdown {
			continue
		}
		if criteria.MinWinRate != nil && latest.WinRate < *criteria.MinWinRate {
			continue
		}
		cp := *r
		results = append(results, &cp)
	}

	// Sort
	sortFn := func(i, j int) bool {
		a := results[i].LatestPerformance()
		b := results[j].LatestPerformance()
		var va, vb float64
		switch criteria.SortBy {
		case "calmar":
			va, vb = a.Calmar, b.Calmar
		case "returns":
			va, vb = a.Returns, b.Returns
		case "created_at":
			if criteria.SortDesc {
				return results[i].CreatedAt.After(results[j].CreatedAt)
			}
			return results[i].CreatedAt.Before(results[j].CreatedAt)
		default:
			va, vb = a.Sharpe, b.Sharpe
		}
		if criteria.SortDesc {
			return va > vb
		}
		return va < vb
	}
	sort.Slice(results, sortFn)

	if criteria.Limit > 0 && len(results) > criteria.Limit {
		results = results[:criteria.Limit]
	}
	return results
}

// SearchByPerformance finds strategies matching a performance threshold.
func (ss *StrategyStore) SearchByPerformance(metric string, threshold float64, above bool) []*StrategyRecord {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	var results []*StrategyRecord
	for _, r := range ss.strategies {
		latest := r.LatestPerformance()
		var val float64
		switch metric {
		case "sharpe":
			val = latest.Sharpe
		case "calmar":
			val = latest.Calmar
		case "max_drawdown":
			val = latest.MaxDrawdown
		case "returns":
			val = latest.Returns
		case "sortino":
			val = latest.Sortino
		case "win_rate":
			val = latest.WinRate
		case "volatility":
			val = latest.Volatility
		default:
			if latest.Extra != nil {
				val = latest.Extra[metric]
			}
		}
		if above && val >= threshold {
			cp := *r
			results = append(results, &cp)
		} else if !above && val <= threshold {
			cp := *r
			results = append(results, &cp)
		}
	}
	return results
}

// ByRegime finds strategies tagged with a specific regime label.
func (ss *StrategyStore) ByRegime(regime string) []*StrategyRecord {
	tag := "regime:" + regime
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	ids := ss.byTag[tag]
	out := make([]*StrategyRecord, 0, len(ids))
	for id := range ids {
		if r, ok := ss.strategies[id]; ok {
			cp := *r
			out = append(out, &cp)
		}
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// Comparison
// ──────────────────────────────────────────────────────────────────────────────

type ComparisonResult struct {
	StrategyA      string              `json:"strategy_a"`
	StrategyB      string              `json:"strategy_b"`
	PerformanceA   PerformanceSnapshot `json:"perf_a"`
	PerformanceB   PerformanceSnapshot `json:"perf_b"`
	SharpeAWins    int                 `json:"sharpe_a_wins"`
	SharpeBWins    int                 `json:"sharpe_b_wins"`
	ReturnCorr     float64             `json:"return_correlation"`
	BetterOverall  string              `json:"better_overall"`
}

func (ss *StrategyStore) Compare(idA, idB string) (*ComparisonResult, error) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	a, okA := ss.strategies[idA]
	b, okB := ss.strategies[idB]
	if !okA || !okB {
		return nil, fmt.Errorf("strategy not found")
	}

	result := &ComparisonResult{
		StrategyA:    idA,
		StrategyB:    idB,
		PerformanceA: a.LatestPerformance(),
		PerformanceB: b.LatestPerformance(),
	}

	// Compare cycle-by-cycle Sharpe
	minCycles := len(a.Performance)
	if len(b.Performance) < minCycles {
		minCycles = len(b.Performance)
	}
	sharpeA := make([]float64, minCycles)
	sharpeB := make([]float64, minCycles)
	for i := 0; i < minCycles; i++ {
		sharpeA[i] = a.Performance[i].Sharpe
		sharpeB[i] = b.Performance[i].Sharpe
		if sharpeA[i] > sharpeB[i] {
			result.SharpeAWins++
		} else if sharpeB[i] > sharpeA[i] {
			result.SharpeBWins++
		}
	}

	// Correlation of return series
	retsA := make([]float64, minCycles)
	retsB := make([]float64, minCycles)
	for i := 0; i < minCycles; i++ {
		retsA[i] = a.Performance[i].Returns
		retsB[i] = b.Performance[i].Returns
	}
	if minCycles >= 3 {
		result.ReturnCorr = pearsonCorr(retsA, retsB)
	}

	// Overall winner (multi-metric)
	scoreA := result.PerformanceA.Sharpe*0.4 + result.PerformanceA.Calmar*0.3 + (1+result.PerformanceA.MaxDrawdown)*0.3
	scoreB := result.PerformanceB.Sharpe*0.4 + result.PerformanceB.Calmar*0.3 + (1+result.PerformanceB.MaxDrawdown)*0.3
	if scoreA >= scoreB {
		result.BetterOverall = idA
	} else {
		result.BetterOverall = idB
	}
	return result, nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Export / serialization
// ──────────────────────────────────────────────────────────────────────────────

// ExportJSON serializes a strategy to JSON bytes.
func (ss *StrategyStore) ExportJSON(id string) ([]byte, error) {
	ss.mu.RLock()
	r, ok := ss.strategies[id]
	ss.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("not found: %s", id)
	}
	return json.MarshalIndent(r, "", "  ")
}

// ExportAll exports all strategies as JSON array.
func (ss *StrategyStore) ExportAll() ([]byte, error) {
	ss.mu.RLock()
	defer ss.mu.RUnlock()
	all := make([]*StrategyRecord, 0, len(ss.strategies))
	for _, r := range ss.strategies {
		all = append(all, r)
	}
	return json.MarshalIndent(all, "", "  ")
}

// ImportJSON deserializes and stores a strategy from JSON.
func (ss *StrategyStore) ImportJSON(data []byte) (string, error) {
	var rec StrategyRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return "", err
	}
	return ss.Create(rec), nil
}

// ImportBatch imports multiple strategies.
func (ss *StrategyStore) ImportBatch(data []byte) ([]string, error) {
	var recs []StrategyRecord
	if err := json.Unmarshal(data, &recs); err != nil {
		return nil, err
	}
	ids := make([]string, len(recs))
	for i, rec := range recs {
		ids[i] = ss.Create(rec)
	}
	return ids, nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Pruning and promotion
// ──────────────────────────────────────────────────────────────────────────────

type PruneConfig struct {
	MinSharpe      float64 `json:"min_sharpe"`
	MaxDrawdown    float64 `json:"max_drawdown"` // e.g., -0.20
	MinCycles      int     `json:"min_cycles"`
	InactiveDays   int     `json:"inactive_days"`
	PromoteSharpe  float64 `json:"promote_sharpe"`
}

func DefaultPruneConfig() PruneConfig {
	return PruneConfig{
		MinSharpe:     0.3,
		MaxDrawdown:   -0.25,
		MinCycles:     10,
		InactiveDays:  90,
		PromoteSharpe: 1.5,
	}
}

type PruneResult struct {
	Archived  []string `json:"archived"`
	Promoted  []string `json:"promoted"`
	Unchanged []string `json:"unchanged"`
}

func (ss *StrategyStore) Prune(cfg PruneConfig) PruneResult {
	ss.mu.Lock()
	defer ss.mu.Unlock()

	var result PruneResult
	now := time.Now()

	for id, r := range ss.strategies {
		if r.Status == StatusArchived {
			continue
		}

		latest := r.LatestPerformance()
		nCycles := len(r.Performance)
		daysSinceUpdate := int(now.Sub(r.UpdatedAt).Hours() / 24)

		shouldArchive := false
		shouldPromote := false

		// Archive underperformers
		if nCycles >= cfg.MinCycles {
			avgSharpe := r.AvgSharpe()
			if avgSharpe < cfg.MinSharpe {
				shouldArchive = true
			}
			if latest.MaxDrawdown < cfg.MaxDrawdown {
				shouldArchive = true
			}
		}

		// Archive inactive strategies
		if daysSinceUpdate > cfg.InactiveDays && r.Status == StatusCandidate {
			shouldArchive = true
		}

		// Promote high performers
		if nCycles >= cfg.MinCycles && latest.Sharpe >= cfg.PromoteSharpe && r.AvgSharpe() >= cfg.PromoteSharpe*0.7 {
			shouldPromote = true
		}

		if shouldArchive {
			r.Status = StatusArchived
			r.UpdatedAt = now
			result.Archived = append(result.Archived, id)
		} else if shouldPromote {
			r.Status = StatusChampion
			r.UpdatedAt = now
			result.Promoted = append(result.Promoted, id)
		} else {
			result.Unchanged = append(result.Unchanged, id)
		}
	}
	return result
}

// PromoteToActive sets a strategy's status to active.
func (ss *StrategyStore) PromoteToActive(id string) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	r, ok := ss.strategies[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	r.Status = StatusActive
	r.UpdatedAt = time.Now()
	return nil
}

// Retire marks a strategy as retired.
func (ss *StrategyStore) Retire(id string) error {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	r, ok := ss.strategies[id]
	if !ok {
		return fmt.Errorf("not found: %s", id)
	}
	r.Status = StatusRetired
	r.UpdatedAt = time.Now()
	return nil
}

// ──────────────────────────────────────────────────────────────────────────────
// Lineage tracking
// ──────────────────────────────────────────────────────────────────────────────

// GetLineage returns the full ancestry chain for a strategy.
func (ss *StrategyStore) GetLineage(id string) []*StrategyRecord {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	var chain []*StrategyRecord
	current := id
	seen := make(map[string]bool)

	for current != "" && !seen[current] {
		seen[current] = true
		r, ok := ss.strategies[current]
		if !ok {
			break
		}
		cp := *r
		chain = append(chain, &cp)
		current = r.ParentID
	}
	// Reverse to get oldest-first
	for i, j := 0, len(chain)-1; i < j; i, j = i+1, j-1 {
		chain[i], chain[j] = chain[j], chain[i]
	}
	return chain
}

// GetDescendants returns all descendants of a strategy.
func (ss *StrategyStore) GetDescendants(id string) []*StrategyRecord {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	var result []*StrategyRecord
	queue := []string{id}
	seen := map[string]bool{id: true}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		r, ok := ss.strategies[current]
		if !ok {
			continue
		}
		if current != id {
			cp := *r
			result = append(result, &cp)
		}
		for _, childID := range r.ChildIDs {
			if !seen[childID] {
				seen[childID] = true
				queue = append(queue, childID)
			}
		}
	}
	return result
}

// ──────────────────────────────────────────────────────────────────────────────
// Statistics and analytics
// ──────────────────────────────────────────────────────────────────────────────

type StoreStats struct {
	TotalStrategies int                `json:"total_strategies"`
	ByStatus        map[string]int     `json:"by_status"`
	ByType          map[string]int     `json:"by_type"`
	AvgSharpe       float64            `json:"avg_sharpe"`
	MedianSharpe    float64            `json:"median_sharpe"`
	TopPerformers   []*StrategyRecord  `json:"top_performers"`
	WorstPerformers []*StrategyRecord  `json:"worst_performers"`
	AvgVersions     float64            `json:"avg_versions"`
}

func (ss *StrategyStore) Stats(topN int) StoreStats {
	ss.mu.RLock()
	defer ss.mu.RUnlock()

	stats := StoreStats{
		TotalStrategies: len(ss.strategies),
		ByStatus:        make(map[string]int),
		ByType:          make(map[string]int),
	}

	var sharpes []float64
	all := make([]*StrategyRecord, 0, len(ss.strategies))

	for _, r := range ss.strategies {
		stats.ByStatus[r.Status.String()]++
		stats.ByType[r.Config.Type]++
		latest := r.LatestPerformance()
		sharpes = append(sharpes, latest.Sharpe)
		cp := *r
		all = append(all, &cp)
	}

	if len(sharpes) > 0 {
		stats.AvgSharpe = meanF(sharpes)
		sorted := make([]float64, len(sharpes))
		copy(sorted, sharpes)
		sort.Float64s(sorted)
		stats.MedianSharpe = sorted[len(sorted)/2]
	}

	// Top and worst
	sort.Slice(all, func(i, j int) bool {
		return all[i].LatestPerformance().Sharpe > all[j].LatestPerformance().Sharpe
	})
	if topN > len(all) {
		topN = len(all)
	}
	stats.TopPerformers = all[:topN]
	if len(all) >= topN {
		stats.WorstPerformers = all[len(all)-topN:]
	}

	// Average versions per strategy name
	totalVersions := 0
	for _, ids := range ss.versions {
		totalVersions += len(ids)
	}
	if len(ss.versions) > 0 {
		stats.AvgVersions = float64(totalVersions) / float64(len(ss.versions))
	}

	return stats
}

// ──────────────────────────────────────────────────────────────────────────────
// Bulk operations
// ──────────────────────────────────────────────────────────────────────────────

// SetStatus sets the status for multiple strategies.
func (ss *StrategyStore) SetStatus(ids []string, status StrategyStatus) int {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	count := 0
	now := time.Now()
	for _, id := range ids {
		if r, ok := ss.strategies[id]; ok {
			r.Status = status
			r.UpdatedAt = now
			count++
		}
	}
	return count
}

// BulkAddTag adds a tag to multiple strategies.
func (ss *StrategyStore) BulkAddTag(ids []string, tag string) int {
	count := 0
	for _, id := range ids {
		if ss.AddTag(id, tag) == nil {
			count++
		}
	}
	return count
}

// CleanupOldArchived removes archived strategies older than days.
func (ss *StrategyStore) CleanupOldArchived(days int) int {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	cutoff := time.Now().AddDate(0, 0, -days)
	removed := 0
	for id, r := range ss.strategies {
		if r.Status == StatusArchived && r.UpdatedAt.Before(cutoff) {
			ss.deindexTags(id)
			delete(ss.strategies, id)
			removed++
		}
	}
	return removed
}

// ──────────────────────────────────────────────────────────────────────────────
// Math utilities
// ──────────────────────────────────────────────────────────────────────────────

func meanF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func stddevF(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := meanF(xs)
	s := 0.0
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	return math.Sqrt(s / float64(len(xs)-1))
}

func pearsonCorr(xs, ys []float64) float64 {
	n := len(xs)
	if n < 2 || n != len(ys) {
		return 0
	}
	mx := meanF(xs)
	my := meanF(ys)
	var num, dx2, dy2 float64
	for i := 0; i < n; i++ {
		dx := xs[i] - mx
		dy := ys[i] - my
		num += dx * dy
		dx2 += dx * dx
		dy2 += dy * dy
	}
	denom := math.Sqrt(dx2 * dy2)
	if denom == 0 {
		return 0
	}
	return num / denom
}
