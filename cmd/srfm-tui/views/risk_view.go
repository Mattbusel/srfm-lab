// risk_view.go -- RiskView displays risk metrics, circuit breakers, and alerts.
// Color scheme: green < 50%, yellow 50-80%, red > 80% of limits.
package views

import (
	"fmt"
	"math"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ── Styles ────────────────────────────────────────────────────────────────────

var (
	rvStyleTitle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#58a6ff")).Bold(true)
	rvStyleGreen   = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88"))
	rvStyleRed     = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444"))
	rvStyleYellow  = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffcc00"))
	rvStyleNeutral = lipgloss.NewStyle().Foreground(lipgloss.Color("#c9d1d9"))
	rvStyleDim     = lipgloss.NewStyle().Foreground(lipgloss.Color("#666688"))
	rvStyleHeader  = lipgloss.NewStyle().Foreground(lipgloss.Color("#8888cc")).Bold(true)
	rvStyleBorder  = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#4444aa")).
			Padding(0, 1)
	rvStyleBreached = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#ff4444")).
			Bold(true)
	rvStyleClosed    = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88"))
	rvStyleOpen      = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444")).Bold(true)
	rvStyleHalfOpen  = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffcc00"))
)

// ── Data types ────────────────────────────────────────────────────────────────

// RiskMetrics holds all top-level risk numbers.
type RiskMetrics struct {
	VaR95        float64 // Value at Risk 95%, as dollar amount
	VaRLimit     float64 // configured VaR limit
	Drawdown     float64 // current drawdown from peak, as fraction (0-1)
	DrawdownLimit float64
	MarginUsed   float64 // dollar margin currently used
	MarginLimit  float64
	UpdatedAt    time.Time
}

// CircuitBreaker represents a single circuit breaker's state.
type CircuitBreaker struct {
	Name      string
	State     string // CLOSED / OPEN / HALF_OPEN
	Reason    string // why it tripped, if any
	TrippedAt time.Time
	TripCount int
}

// RiskAlert is a single risk alert event.
type RiskAlert struct {
	Timestamp time.Time
	Level     string // INFO / WARN / BREACH
	Message   string
}

// PositionLimit holds per-symbol position limit info.
type PositionLimit struct {
	Symbol      string
	Position    float64 // current absolute position
	Limit       float64
	Utilization float64 // Position / Limit * 100
}

// RiskViewData bundles all risk display data.
type RiskViewData struct {
	Metrics         RiskMetrics
	CircuitBreakers []CircuitBreaker
	Alerts          []RiskAlert   // last 10
	PositionLimits  []PositionLimit
}

// RiskView is the Bubble Tea component for risk display.
type RiskView struct {
	Data        RiskViewData
	alertScroll int
	limitScroll int
}

// ── Messages ──────────────────────────────────────────────────────────────────

// RiskDataMsg carries fresh risk data.
type RiskDataMsg struct {
	Data RiskViewData
}

// ── Constructor ───────────────────────────────────────────────────────────────

// NewRiskView creates a default RiskView.
func NewRiskView() RiskView {
	return RiskView{}
}

// ── Update ────────────────────────────────────────────────────────────────────

// Update handles keyboard and data messages.
func (v RiskView) Update(msg tea.Msg) (RiskView, tea.Cmd) {
	switch m := msg.(type) {

	case RiskDataMsg:
		v.Data = m.Data

	case tea.KeyMsg:
		switch m.String() {
		case "j", "down":
			maxScroll := len(v.Data.Alerts) - 10
			if maxScroll < 0 {
				maxScroll = 0
			}
			if v.alertScroll < maxScroll {
				v.alertScroll++
			}
		case "k", "up":
			if v.alertScroll > 0 {
				v.alertScroll--
			}
		}
	}

	return v, nil
}

// ── Progress bar helpers ──────────────────────────────────────────────────────

// utilizationColor returns the appropriate style for a utilization percentage.
func utilizationColor(pct float64) lipgloss.Style {
	if pct >= 80 {
		return rvStyleRed
	}
	if pct >= 50 {
		return rvStyleYellow
	}
	return rvStyleGreen
}

// progressBar renders a filled progress bar of given width.
func progressBar(pct float64, width int) string {
	if pct > 100 {
		pct = 100
	}
	filled := int(math.Round(pct / 100.0 * float64(width)))
	if filled > width {
		filled = width
	}
	if filled < 0 {
		filled = 0
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	return utilizationColor(pct).Render(bar)
}

// labeledGauge renders: Label  [██████░░░░░░]  XX.X%  (value / limit)
func labeledGauge(label string, value, limit float64, width int) string {
	pct := 0.0
	if limit > 0 {
		pct = value / limit * 100
	}
	bar := progressBar(pct, width)
	col := utilizationColor(pct)
	return fmt.Sprintf("%-20s [%s] %s  ($%.0f / $%.0f)",
		label,
		bar,
		col.Render(fmt.Sprintf("%5.1f%%", pct)),
		value,
		limit,
	)
}

// ── Circuit breakers ──────────────────────────────────────────────────────────

func circuitBreakerStyle(state string) lipgloss.Style {
	switch state {
	case "CLOSED":
		return rvStyleClosed
	case "OPEN":
		return rvStyleOpen
	case "HALF_OPEN":
		return rvStyleHalfOpen
	default:
		return rvStyleDim
	}
}

func renderCircuitBreakers(cbs []CircuitBreaker) string {
	if len(cbs) == 0 {
		return rvStyleDim.Render("  -- no circuit breakers registered --")
	}
	colWidths := []int{24, 10, 8, 30}
	header := fmt.Sprintf("  %-*s  %-*s  %-*s  %-*s",
		colWidths[0], "Breaker",
		colWidths[1], "State",
		colWidths[2], "Trips",
		colWidths[3], "Reason / Tripped At",
	)
	lines := []string{rvStyleHeader.Render(header)}
	lines = append(lines, rvStyleDim.Render("  "+strings.Repeat("-", 78)))
	for _, cb := range cbs {
		stateStr := circuitBreakerStyle(cb.State).Render(fmt.Sprintf("%-*s", colWidths[1], cb.State))
		reason := cb.Reason
		if !cb.TrippedAt.IsZero() {
			reason += fmt.Sprintf(" @%s", cb.TrippedAt.Format("15:04:05"))
		}
		line := fmt.Sprintf("  %-*s  %s  %-*d  %-*s",
			colWidths[0], cb.Name,
			stateStr,
			colWidths[2], cb.TripCount,
			colWidths[3], reason,
		)
		lines = append(lines, rvStyleNeutral.Render(line))
	}
	return strings.Join(lines, "\n")
}

// ── Alerts ────────────────────────────────────────────────────────────────────

func alertLevelStyle(level string) lipgloss.Style {
	switch level {
	case "BREACH":
		return rvStyleBreached
	case "WARN":
		return rvStyleYellow
	default:
		return rvStyleDim
	}
}

func renderAlerts(alerts []RiskAlert, offset int) string {
	if len(alerts) == 0 {
		return rvStyleDim.Render("  -- no recent alerts --")
	}
	// show up to 10 from offset
	end := offset + 10
	if end > len(alerts) {
		end = len(alerts)
	}
	visible := alerts[offset:end]
	lines := []string{rvStyleHeader.Render(
		fmt.Sprintf("  %-19s  %-7s  %s", "Time", "Level", "Message"))}
	lines = append(lines, rvStyleDim.Render("  "+strings.Repeat("-", 70)))
	for _, a := range visible {
		lStyle := alertLevelStyle(a.Level)
		line := fmt.Sprintf("  %s  %s  %s",
			rvStyleDim.Render(a.Timestamp.Format("01-02 15:04:05")),
			lStyle.Render(fmt.Sprintf("%-7s", a.Level)),
			rvStyleNeutral.Render(a.Message),
		)
		lines = append(lines, line)
	}
	if len(alerts) > 10 {
		lines = append(lines, rvStyleDim.Render(
			fmt.Sprintf("  [%d-%d of %d]  j/k to scroll", offset+1, end, len(alerts))))
	}
	return strings.Join(lines, "\n")
}

// ── Position limits table ─────────────────────────────────────────────────────

func renderPositionLimits(limits []PositionLimit) string {
	if len(limits) == 0 {
		return rvStyleDim.Render("  -- no position limit data --")
	}
	header := fmt.Sprintf("  %-8s  %12s  %12s  %-26s  %s",
		"Symbol", "Position", "Limit", "Utilization", "Bar (40 chars)")
	lines := []string{rvStyleHeader.Render(header)}
	lines = append(lines, rvStyleDim.Render("  "+strings.Repeat("-", 80)))
	for _, pl := range limits {
		pct := pl.Utilization
		col := utilizationColor(pct)
		bar := progressBar(pct, 20)
		line := fmt.Sprintf("  %-8s  %12.0f  %12.0f  %s  [%s]",
			rvStyleNeutral.Render(pl.Symbol),
			math.Abs(pl.Position),
			pl.Limit,
			col.Render(fmt.Sprintf("%5.1f%%", pct)),
			bar,
		)
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

// ── Section header ────────────────────────────────────────────────────────────

func rvSectionHeader(title string) string {
	return rvStyleTitle.Render(title) + "\n" + rvStyleDim.Render(strings.Repeat("─", 72))
}

// ── View ──────────────────────────────────────────────────────────────────────

// View renders the full risk view.
func (v RiskView) View() string {
	m := v.Data.Metrics
	var sb strings.Builder

	// ── Header ────────────────────────────────────────────────────────────────
	ts := m.UpdatedAt.Format("15:04:05")
	if m.UpdatedAt.IsZero() {
		ts = "--:--:--"
	}
	sb.WriteString(rvStyleTitle.Render("RISK MONITOR") +
		rvStyleDim.Render(fmt.Sprintf("  updated %s", ts)) + "\n")
	sb.WriteString(rvStyleDim.Render(strings.Repeat("═", 72)) + "\n\n")

	// ── Gauges ────────────────────────────────────────────────────────────────
	sb.WriteString(rvSectionHeader("GAUGES") + "\n")
	sb.WriteString(labeledGauge("VaR 95%", m.VaR95, m.VaRLimit, 30) + "\n")
	drawdownDollar := m.Drawdown * m.VaRLimit // rough proxy if no NAV; use as fraction
	_ = drawdownDollar
	// drawdown is a fraction 0-1, convert to pct for progress bar
	ddPct := m.Drawdown * 100
	ddLimitPct := m.DrawdownLimit * 100
	ddBar := progressBar(ddPct/ddLimitPct*100, 30)
	ddCol := utilizationColor(ddPct / ddLimitPct * 100)
	sb.WriteString(fmt.Sprintf("%-20s [%s] %s  (%.2f%% / %.2f%% limit)\n",
		"Drawdown",
		ddBar,
		ddCol.Render(fmt.Sprintf("%5.1f%%", ddPct/ddLimitPct*100)),
		ddPct,
		ddLimitPct,
	))
	sb.WriteString(labeledGauge("Margin Utilized", m.MarginUsed, m.MarginLimit, 30) + "\n")

	sb.WriteString("\n")

	// ── Circuit breakers ──────────────────────────────────────────────────────
	sb.WriteString(rvSectionHeader("CIRCUIT BREAKERS") + "\n")
	sb.WriteString(renderCircuitBreakers(v.Data.CircuitBreakers) + "\n")
	sb.WriteString("\n")

	// ── Recent alerts ─────────────────────────────────────────────────────────
	sb.WriteString(rvSectionHeader("RECENT ALERTS (last 10)") + "\n")
	sb.WriteString(renderAlerts(v.Data.Alerts, v.alertScroll) + "\n")
	sb.WriteString("\n")

	// ── Position limits ───────────────────────────────────────────────────────
	sb.WriteString(rvSectionHeader("POSITION LIMITS") + "\n")
	sb.WriteString(renderPositionLimits(v.Data.PositionLimits) + "\n")

	return rvStyleBorder.Render(sb.String())
}
