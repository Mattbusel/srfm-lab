// execution_view.go -- ExecutionView displays recent fills, pending orders, and TCA stats.
// Pending orders colored yellow if age > 10s, red if > 30s.
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
	evStyleTitle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#58a6ff")).Bold(true)
	evStyleGreen   = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88"))
	evStyleRed     = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444"))
	evStyleYellow  = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffcc00"))
	evStyleNeutral = lipgloss.NewStyle().Foreground(lipgloss.Color("#c9d1d9"))
	evStyleDim     = lipgloss.NewStyle().Foreground(lipgloss.Color("#666688"))
	evStyleHeader  = lipgloss.NewStyle().Foreground(lipgloss.Color("#8888cc")).Bold(true)
	evStyleBorder  = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#4444aa")).
			Padding(0, 1)
	evStyleBuy  = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88")).Bold(true)
	evStyleSell = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444")).Bold(true)
)

// ── Data types ────────────────────────────────────────────────────────────────

// Fill represents a completed trade execution.
type Fill struct {
	Time      time.Time
	Symbol    string
	Side      string  // BUY / SELL
	Qty       float64
	Price     float64
	Slippage  float64 // in bps
	Strategy  string
}

// PendingOrder is an order that has been submitted but not yet filled.
type PendingOrder struct {
	OrderID   string
	Symbol    string
	Side      string
	Qty       float64
	LimitPrice float64
	SubmittedAt time.Time
	Strategy   string
}

// ExecQualityStats holds aggregate execution quality numbers.
type ExecQualityStats struct {
	AvgSlippage24h float64 // bps
	FillRate       float64 // fraction 0-1
	AvgFillTimeSec float64 // seconds
	TodayTCACostBps float64
}

// VenueStats holds per-venue execution stats.
type VenueStats struct {
	Venue    string
	FillRate float64 // fraction 0-1
	Fills    int
	Volume   float64 // notional
}

// ExecutionViewData bundles all data for the execution view.
type ExecutionViewData struct {
	RecentFills []Fill         // last ~50
	PendingOrders []PendingOrder
	QualityStats  ExecQualityStats
	Venues        []VenueStats
	UpdatedAt     time.Time
}

// ExecutionView is the Bubble Tea component for execution display.
type ExecutionView struct {
	Data       ExecutionViewData
	fillScroll int
	viewHeight int
	now        time.Time // injected for age calculation; falls back to time.Now()
}

// ── Messages ──────────────────────────────────────────────────────────────────

// ExecutionDataMsg carries fresh execution data.
type ExecutionDataMsg struct {
	Data ExecutionViewData
}

// ── Constructor ───────────────────────────────────────────────────────────────

// NewExecutionView creates an ExecutionView with defaults.
func NewExecutionView() ExecutionView {
	return ExecutionView{viewHeight: 15}
}

// ── Update ────────────────────────────────────────────────────────────────────

// Update handles keyboard navigation and data messages.
func (v ExecutionView) Update(msg tea.Msg) (ExecutionView, tea.Cmd) {
	switch m := msg.(type) {

	case ExecutionDataMsg:
		v.Data = m.Data

	case tea.KeyMsg:
		switch m.String() {
		case "j", "down":
			max := len(v.Data.RecentFills) - v.viewHeight
			if max < 0 {
				max = 0
			}
			if v.fillScroll < max {
				v.fillScroll++
			}
		case "k", "up":
			if v.fillScroll > 0 {
				v.fillScroll--
			}
		}
	}

	return v, nil
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func (v ExecutionView) currentTime() time.Time {
	if !v.now.IsZero() {
		return v.now
	}
	return time.Now()
}

// sideStyle returns buy/sell colored style.
func sideStyle(side string) lipgloss.Style {
	if side == "BUY" {
		return evStyleBuy
	}
	return evStyleSell
}

// slippageStyle colors slippage: green <= 1bps, yellow <= 5bps, red > 5bps.
func slippageStyle(bps float64) lipgloss.Style {
	absBps := math.Abs(bps)
	if absBps <= 1 {
		return evStyleGreen
	}
	if absBps <= 5 {
		return evStyleYellow
	}
	return evStyleRed
}

// orderAgeStyle colors order age indicator.
func orderAgeStyle(age time.Duration) lipgloss.Style {
	if age > 30*time.Second {
		return evStyleRed
	}
	if age > 10*time.Second {
		return evStyleYellow
	}
	return evStyleGreen
}

// evPadRight pads s to width w using visible width.
func evPadRight(s string, w int) string {
	vis := lipgloss.Width(s)
	if vis < w {
		return s + strings.Repeat(" ", w-vis)
	}
	return s
}

// fillRate bar: [████░░░░] pct%
func fillRateBar(rate float64) string {
	pct := rate * 100
	filled := int(pct / 100 * 12)
	if filled > 12 {
		filled = 12
	}
	bar := strings.Repeat("▪", filled) + strings.Repeat("░", 12-filled)
	style := evStyleGreen
	if pct < 80 {
		style = evStyleYellow
	}
	if pct < 60 {
		style = evStyleRed
	}
	return fmt.Sprintf("[%s] %.0f%%", style.Render(bar), pct)
}

// ── Section header ────────────────────────────────────────────────────────────

func evSectionHeader(title string) string {
	return evStyleTitle.Render(title) + "\n" + evStyleDim.Render(strings.Repeat("─", 72))
}

// ── Fills table ───────────────────────────────────────────────────────────────

func (v ExecutionView) renderFills() string {
	fills := v.Data.RecentFills
	colW := []int{10, 8, 5, 10, 10, 9, 14}
	header := fmt.Sprintf("  %-*s  %-*s  %-*s  %-*s  %-*s  %-*s  %-*s",
		colW[0], "Time",
		colW[1], "Symbol",
		colW[2], "Side",
		colW[3], "Qty",
		colW[4], "Price",
		colW[5], "Slip(bps)",
		colW[6], "Strategy",
	)
	lines := []string{evStyleHeader.Render(header)}
	lines = append(lines, evStyleDim.Render("  "+strings.Repeat("-", 80)))

	if len(fills) == 0 {
		lines = append(lines, evStyleDim.Render("  -- no fills --"))
		return strings.Join(lines, "\n")
	}

	end := v.fillScroll + v.viewHeight
	if end > len(fills) {
		end = len(fills)
	}
	visible := fills[v.fillScroll:end]

	for _, f := range visible {
		sStyle := sideStyle(f.Side)
		slipCol := slippageStyle(f.Slippage)
		line := fmt.Sprintf("  %-*s  %-*s  %s  %-*s  %-*s  %s  %-*s",
			colW[0], f.Time.Format("15:04:05.0"),
			colW[1], f.Symbol,
			evPadRight(sStyle.Render(fmt.Sprintf("%-5s", f.Side)), colW[2]+2),
			colW[3], fmt.Sprintf("%.2f", f.Qty),
			colW[4], fmt.Sprintf("%.4f", f.Price),
			evPadRight(slipCol.Render(fmt.Sprintf("%+.2f", f.Slippage)), colW[5]+2),
			colW[6], f.Strategy,
		)
		lines = append(lines, evStyleNeutral.Render(line))
	}

	if len(fills) > v.viewHeight {
		lines = append(lines, evStyleDim.Render(
			fmt.Sprintf("  [%d-%d of %d fills]  j/k=scroll",
				v.fillScroll+1, end, len(fills))))
	}
	return strings.Join(lines, "\n")
}

// ── Pending orders ────────────────────────────────────────────────────────────

func (v ExecutionView) renderPending() string {
	orders := v.Data.PendingOrders
	if len(orders) == 0 {
		return evStyleDim.Render("  -- no pending orders --")
	}
	header := fmt.Sprintf("  %-12s  %-8s  %-5s  %-10s  %-10s  %-8s  %-14s",
		"OrderID", "Symbol", "Side", "Qty", "LimitPrice", "Age", "Strategy")
	lines := []string{evStyleHeader.Render(header)}
	lines = append(lines, evStyleDim.Render("  "+strings.Repeat("-", 80)))

	now := v.currentTime()
	for _, o := range orders {
		age := now.Sub(o.SubmittedAt)
		ageStr := fmt.Sprintf("%.0fs", age.Seconds())
		ageStyle := orderAgeStyle(age)
		sStyle := sideStyle(o.Side)
		line := fmt.Sprintf("  %-12s  %-8s  %s  %-10s  %-10s  %s  %-14s",
			o.OrderID,
			o.Symbol,
			evPadRight(sStyle.Render(fmt.Sprintf("%-5s", o.Side)), 7),
			fmt.Sprintf("%.2f", o.Qty),
			fmt.Sprintf("%.4f", o.LimitPrice),
			evPadRight(ageStyle.Render(fmt.Sprintf("%-8s", ageStr)), 10),
			o.Strategy,
		)
		lines = append(lines, evStyleNeutral.Render(line))
	}
	return strings.Join(lines, "\n")
}

// ── Quality stats ─────────────────────────────────────────────────────────────

func renderQualityStats(s ExecQualityStats) string {
	slipStyle := slippageStyle(s.AvgSlippage24h)
	tcaStyle := evStyleGreen
	if s.TodayTCACostBps > 10 {
		tcaStyle = evStyleRed
	} else if s.TodayTCACostBps > 5 {
		tcaStyle = evStyleYellow
	}
	lines := []string{
		fmt.Sprintf("  Avg Slippage 24h:  %s",
			slipStyle.Render(fmt.Sprintf("%+.2f bps", s.AvgSlippage24h))),
		fmt.Sprintf("  Fill Rate:         %s",
			fillRateBar(s.FillRate)),
		fmt.Sprintf("  Avg Fill Time:     %s",
			evStyleNeutral.Render(fmt.Sprintf("%.2fs", s.AvgFillTimeSec))),
		fmt.Sprintf("  Today TCA Cost:    %s",
			tcaStyle.Render(fmt.Sprintf("%.2f bps", s.TodayTCACostBps))),
	}
	return strings.Join(lines, "\n")
}

// ── Venue stats ───────────────────────────────────────────────────────────────

func renderVenues(venues []VenueStats) string {
	if len(venues) == 0 {
		return evStyleDim.Render("  -- no venue data today --")
	}
	header := fmt.Sprintf("  %-16s  %-14s  %-8s  %s",
		"Venue", "Fill Rate", "Fills", "Volume")
	lines := []string{evStyleHeader.Render(header)}
	lines = append(lines, evStyleDim.Render("  "+strings.Repeat("-", 56)))
	for _, vs := range venues {
		line := fmt.Sprintf("  %-16s  %-14s  %-8d  $%.0f",
			vs.Venue,
			fillRateBar(vs.FillRate),
			vs.Fills,
			vs.Volume,
		)
		lines = append(lines, evStyleNeutral.Render(line))
	}
	return strings.Join(lines, "\n")
}

// ── View ──────────────────────────────────────────────────────────────────────

// View renders the full execution view.
func (v ExecutionView) View() string {
	var sb strings.Builder

	ts := v.Data.UpdatedAt.Format("15:04:05")
	if v.Data.UpdatedAt.IsZero() {
		ts = "--:--:--"
	}
	sb.WriteString(evStyleTitle.Render("EXECUTION") +
		evStyleDim.Render(fmt.Sprintf("  updated %s", ts)) + "\n")
	sb.WriteString(evStyleDim.Render(strings.Repeat("═", 72)) + "\n\n")

	// ── Recent fills ──────────────────────────────────────────────────────────
	sb.WriteString(evSectionHeader("RECENT FILLS") + "\n")
	sb.WriteString(v.renderFills() + "\n\n")

	// ── Pending orders ────────────────────────────────────────────────────────
	pendingCount := len(v.Data.PendingOrders)
	pendingTitle := fmt.Sprintf("PENDING ORDERS (%d)", pendingCount)
	if pendingCount > 0 {
		pendingTitle = evStyleYellow.Render(pendingTitle)
	}
	sb.WriteString(evStyleTitle.Render("") + pendingTitle + "\n")
	sb.WriteString(evStyleDim.Render(strings.Repeat("─", 72)) + "\n")
	sb.WriteString(v.renderPending() + "\n\n")

	// ── Execution quality ─────────────────────────────────────────────────────
	sb.WriteString(evSectionHeader("EXECUTION QUALITY") + "\n")
	sb.WriteString(renderQualityStats(v.Data.QualityStats) + "\n\n")

	// ── Venue utilization ─────────────────────────────────────────────────────
	sb.WriteString(evSectionHeader("VENUE UTILIZATION (today)") + "\n")
	sb.WriteString(renderVenues(v.Data.Venues) + "\n")

	return evStyleBorder.Render(sb.String())
}
