// portfolio_view.go -- PortfolioView displays current portfolio positions and metrics.
// Layout: two-column (positions table | metrics panel)
// Keyboard: j/k scroll, s cycle sort column
package views

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ── Styles ────────────────────────────────────────────────────────────────────

var (
	pvStyleTitle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#58a6ff")).Bold(true)
	pvStyleGain    = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88"))
	pvStyleLoss    = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444"))
	pvStyleNeutral = lipgloss.NewStyle().Foreground(lipgloss.Color("#c9d1d9"))
	pvStyleDim     = lipgloss.NewStyle().Foreground(lipgloss.Color("#666688"))
	pvStyleWarn    = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffcc00"))
	pvStyleHeader  = lipgloss.NewStyle().Foreground(lipgloss.Color("#8888cc")).Bold(true)
	pvStyleBorder  = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#4444aa")).
			Padding(0, 1)
	pvStyleSelected = lipgloss.NewStyle().
			Background(lipgloss.Color("#1a1a3a")).
			Foreground(lipgloss.Color("#ffffff"))
)

// ── Data types ────────────────────────────────────────────────────────────────

// Position holds a single portfolio position row.
type Position struct {
	Symbol    string
	Qty       float64 // number of contracts / shares
	AvgCost   float64 // average fill price
	Mark      float64 // current mark price
	UnrealPnL float64 // unrealized P&L in dollars
	PctPnL    float64 // unrealized P&L as percent
	DailyPnL  float64 // today's P&L contribution
}

// PortfolioMetrics holds aggregate portfolio stats.
type PortfolioMetrics struct {
	NAV           float64
	Cash          float64
	GrossExposure float64 // sum of abs(position * mark)
	NetExposure   float64 // signed sum
	Leverage      float64 // gross / NAV
	DailyPnL      float64
	YtdPnL        float64
	UpdatedAt     time.Time
}

// SortColumn identifies which column the positions table is sorted by.
type SortColumn int

const (
	SortBySymbol SortColumn = iota
	SortByQty
	SortByUnrealPnL
	SortByPctPnL
	SortByDailyPnL
	sortColumnCount
)

func (s SortColumn) String() string {
	switch s {
	case SortBySymbol:
		return "Symbol"
	case SortByQty:
		return "Qty"
	case SortByUnrealPnL:
		return "Unreal P&L"
	case SortByPctPnL:
		return "%"
	case SortByDailyPnL:
		return "Daily P&L"
	default:
		return "Symbol"
	}
}

// PortfolioView is the Bubble Tea component for portfolio display.
type PortfolioView struct {
	Positions []Position
	Metrics   PortfolioMetrics

	// scroll / sort state
	cursor     int
	sortCol    SortColumn
	sortAsc    bool
	viewOffset int
	viewHeight int // visible rows in table
}

// ── Constructor ───────────────────────────────────────────────────────────────

// NewPortfolioView creates an empty PortfolioView with sensible defaults.
func NewPortfolioView() PortfolioView {
	return PortfolioView{
		sortCol:    SortBySymbol,
		sortAsc:    true,
		viewHeight: 12,
	}
}

// ── Messages ──────────────────────────────────────────────────────────────────

// PortfolioDataMsg carries fresh portfolio data from the API client.
type PortfolioDataMsg struct {
	Positions []Position
	Metrics   PortfolioMetrics
}

// ── Update ────────────────────────────────────────────────────────────────────

// Update handles keyboard events and incoming data messages.
func (v PortfolioView) Update(msg tea.Msg) (PortfolioView, tea.Cmd) {
	switch m := msg.(type) {

	case tea.KeyMsg:
		switch m.String() {
		case "j", "down":
			if v.cursor < len(v.Positions)-1 {
				v.cursor++
				if v.cursor >= v.viewOffset+v.viewHeight {
					v.viewOffset++
				}
			}
		case "k", "up":
			if v.cursor > 0 {
				v.cursor--
				if v.cursor < v.viewOffset {
					v.viewOffset--
				}
			}
		case "s":
			// cycle to next sort column
			v.sortCol = (v.sortCol + 1) % sortColumnCount
			v.sortAsc = true
			v.cursor = 0
			v.viewOffset = 0
		case "S":
			// toggle sort direction
			v.sortAsc = !v.sortAsc
		}

	case PortfolioDataMsg:
		v.Positions = m.Positions
		v.Metrics = m.Metrics
		// clamp cursor after data refresh
		if v.cursor >= len(v.Positions) && len(v.Positions) > 0 {
			v.cursor = len(v.Positions) - 1
		}
	}

	return v, nil
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func (v PortfolioView) sortedPositions() []Position {
	cp := make([]Position, len(v.Positions))
	copy(cp, v.Positions)
	sort.SliceStable(cp, func(i, j int) bool {
		var less bool
		switch v.sortCol {
		case SortByQty:
			less = cp[i].Qty < cp[j].Qty
		case SortByUnrealPnL:
			less = cp[i].UnrealPnL < cp[j].UnrealPnL
		case SortByPctPnL:
			less = cp[i].PctPnL < cp[j].PctPnL
		case SortByDailyPnL:
			less = cp[i].DailyPnL < cp[j].DailyPnL
		default:
			less = cp[i].Symbol < cp[j].Symbol
		}
		if v.sortAsc {
			return less
		}
		return !less
	})
	return cp
}

// fmtPnL formats a dollar P&L value with color.
func fmtPnLColored(v float64) string {
	if v >= 0 {
		return pvStyleGain.Render(fmt.Sprintf("+$%.0f", v))
	}
	return pvStyleLoss.Render(fmt.Sprintf("-$%.0f", math.Abs(v)))
}

// fmtPct formats a percent P&L value with color.
func fmtPctColored(v float64) string {
	if v >= 0 {
		return pvStyleGain.Render(fmt.Sprintf("+%.2f%%", v))
	}
	return pvStyleLoss.Render(fmt.Sprintf("%.2f%%", v))
}

// padRight pads/truncates a string to exactly w visible characters.
func padRight(s string, w int) string {
	vis := lipgloss.Width(s)
	if vis < w {
		return s + strings.Repeat(" ", w-vis)
	}
	// truncate the raw runes (not ANSI), rough approximation
	r := []rune(s)
	if len(r) > w {
		return string(r[:w])
	}
	return s
}

// colWidth defines widths for each column in the positions table.
var posColWidths = []int{8, 10, 11, 11, 13, 9, 13}

// posHeader builds the header row string.
func posHeader(sortCol SortColumn) string {
	labels := []string{"Symbol", "Qty", "Avg Cost", "Mark", "Unreal P&L", "%", "Daily P&L"}
	var parts []string
	for i, lbl := range labels {
		w := posColWidths[i]
		s := lbl
		if (i == 0 && sortCol == SortBySymbol) ||
			(i == 1 && sortCol == SortByQty) ||
			(i == 4 && sortCol == SortByUnrealPnL) ||
			(i == 5 && sortCol == SortByPctPnL) ||
			(i == 6 && sortCol == SortByDailyPnL) {
			s = pvStyleWarn.Render(s)
		} else {
			s = pvStyleHeader.Render(s)
		}
		parts = append(parts, padRight(s, w))
	}
	return strings.Join(parts, " ")
}

// posRow builds a single position row string, highlighted if selected.
func posRow(p Position, selected bool) string {
	sym := padRight(pvStyleNeutral.Render(p.Symbol), posColWidths[0])
	qty := padRight(pvStyleNeutral.Render(fmt.Sprintf("%.2f", p.Qty)), posColWidths[1])
	avgCost := padRight(pvStyleNeutral.Render(fmt.Sprintf("%.2f", p.AvgCost)), posColWidths[2])
	mark := padRight(pvStyleNeutral.Render(fmt.Sprintf("%.2f", p.Mark)), posColWidths[3])
	unreal := padRight(fmtPnLColored(p.UnrealPnL), posColWidths[4])
	pct := padRight(fmtPctColored(p.PctPnL), posColWidths[5])
	daily := padRight(fmtPnLColored(p.DailyPnL), posColWidths[6])

	row := strings.Join([]string{sym, qty, avgCost, mark, unreal, pct, daily}, " ")
	if selected {
		return pvStyleSelected.Render("> " + row)
	}
	return "  " + row
}

// ── Metrics Panel ─────────────────────────────────────────────────────────────

func (v PortfolioView) renderMetrics() string {
	m := v.Metrics

	leverageColor := pvStyleGain
	if m.Leverage > 3.0 {
		leverageColor = pvStyleLoss
	} else if m.Leverage > 1.5 {
		leverageColor = pvStyleWarn
	}

	lines := []string{
		pvStyleTitle.Render("PORTFOLIO METRICS"),
		strings.Repeat("-", 28),
		fmt.Sprintf("%-16s %s", "NAV:",
			pvStyleGain.Render(fmt.Sprintf("$%.0f", m.NAV))),
		fmt.Sprintf("%-16s %s", "Cash:",
			pvStyleNeutral.Render(fmt.Sprintf("$%.0f", m.Cash))),
		"",
		fmt.Sprintf("%-16s %s", "Gross Exposure:",
			pvStyleNeutral.Render(fmt.Sprintf("$%.0f", m.GrossExposure))),
		fmt.Sprintf("%-16s %s", "Net Exposure:",
			fmtPnLColored(m.NetExposure)),
		fmt.Sprintf("%-16s %s", "Leverage:",
			leverageColor.Render(fmt.Sprintf("%.2fx", m.Leverage))),
		"",
		fmt.Sprintf("%-16s %s", "Daily P&L:",
			fmtPnLColored(m.DailyPnL)),
		fmt.Sprintf("%-16s %s", "YTD P&L:",
			fmtPnLColored(m.YtdPnL)),
		"",
		pvStyleDim.Render(fmt.Sprintf("Updated: %s",
			m.UpdatedAt.Format("15:04:05"))),
	}

	return pvStyleBorder.Render(strings.Join(lines, "\n"))
}

// ── View ──────────────────────────────────────────────────────────────────────

// View renders the full portfolio view as a string.
func (v PortfolioView) View() string {
	sorted := v.sortedPositions()

	// ── Left: positions table ─────────────────────────────────────────────────
	tableTitle := pvStyleTitle.Render("POSITIONS") +
		pvStyleDim.Render(fmt.Sprintf("  sort: %s  [s]=next  [S]=reverse  [j/k]=scroll",
			v.sortCol.String()))
	header := posHeader(v.sortCol)
	sepLine := pvStyleDim.Render(strings.Repeat("-", 80))

	var rowLines []string
	end := v.viewOffset + v.viewHeight
	if end > len(sorted) {
		end = len(sorted)
	}
	for i := v.viewOffset; i < end; i++ {
		rowLines = append(rowLines, posRow(sorted[i], i == v.cursor))
	}

	// scroll indicator
	scrollInfo := ""
	if len(sorted) > v.viewHeight {
		scrollInfo = pvStyleDim.Render(fmt.Sprintf("  [%d-%d of %d]",
			v.viewOffset+1, end, len(sorted)))
	}

	if len(sorted) == 0 {
		rowLines = []string{pvStyleDim.Render("  -- no open positions --")}
	}

	tableLines := []string{tableTitle, header, sepLine}
	tableLines = append(tableLines, rowLines...)
	tableLines = append(tableLines, scrollInfo)

	tableContent := strings.Join(tableLines, "\n")
	leftPanel := pvStyleBorder.Width(82).Render(tableContent)

	// ── Right: metrics panel ──────────────────────────────────────────────────
	rightPanel := v.renderMetrics()

	// ── Join side by side ─────────────────────────────────────────────────────
	return lipgloss.JoinHorizontal(lipgloss.Top, leftPanel, "  ", rightPanel)
}
