// signals_view.go -- SignalsView displays live signal state with sparklines and gauges.
// Auto-refreshes every 15 seconds via RefreshSignalsCmd.
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
	svStyleTitle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#58a6ff")).Bold(true)
	svStyleGreen   = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88"))
	svStyleRed     = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444"))
	svStyleYellow  = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffcc00"))
	svStyleNeutral = lipgloss.NewStyle().Foreground(lipgloss.Color("#c9d1d9"))
	svStyleDim     = lipgloss.NewStyle().Foreground(lipgloss.Color("#666688"))
	svStyleHeader  = lipgloss.NewStyle().Foreground(lipgloss.Color("#8888cc")).Bold(true)
	svStyleBorder  = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#4444aa")).
			Padding(0, 1)
	svStyleActive = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88")).Bold(true)
)

// ── Data types ────────────────────────────────────────────────────────────────

// SignalHistory holds the last N bar values for a signal.
type SignalHistory [5]float64

// BHMassSignal represents the Black-Hole mass indicator per symbol.
type BHMassSignal struct {
	Symbol  string
	Mass    float64 // 0-5 scale
	History SignalHistory
	Active  bool // mass >= threshold
}

// HurstSignal holds the current Hurst exponent.
type HurstSignal struct {
	Value   float64
	History SignalHistory
}

// GARCHSignal holds the GARCH volatility forecast.
type GARCHSignal struct {
	CurrentForecast float64 // annualized vol %
	HistoricalMean  float64
	History         SignalHistory
}

// NavCurvature holds the quaternion navigation state.
type NavCurvature struct {
	W, X, Y, Z    float64
	CurvatureAngle float64 // derived magnitude in degrees
	Regime         string
}

// RLQValue holds a state-action pair and its Q-value.
type RLQValue struct {
	Action  string
	QValue  float64
	IsTop   bool
}

// SignalState holds all current signal data.
type SignalState struct {
	BHMasses    []BHMassSignal
	Hurst       HurstSignal
	GARCH       GARCHSignal
	NavCurve    NavCurvature
	RLQValues   []RLQValue // top 3 state-action values
	Regime      string
	Confidence  float64
	UpdatedAt   time.Time
}

// SignalsView is the Bubble Tea component for live signal display.
type SignalsView struct {
	State         SignalState
	lastRefresh   time.Time
	refreshTicker int // countdown to next auto-refresh attempt
	loading       bool
	err           string
}

// ── Messages ──────────────────────────────────────────────────────────────────

// SignalsDataMsg carries fresh signal data.
type SignalsDataMsg struct {
	State SignalState
}

// SignalsRefreshTickMsg is sent every second to count down auto-refresh.
type SignalsRefreshTickMsg time.Time

// SignalsErrorMsg carries a fetch error string.
type SignalsErrorMsg struct{ Err string }

// ── Commands ──────────────────────────────────────────────────────────────────

// SignalsTickCmd returns a command that sends SignalsRefreshTickMsg every second.
func SignalsTickCmd() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg {
		return SignalsRefreshTickMsg(t)
	})
}

// NewSignalsView constructs an empty SignalsView.
func NewSignalsView() SignalsView {
	return SignalsView{
		refreshTicker: 15,
	}
}

// ── Update ────────────────────────────────────────────────────────────────────

// Update handles messages for the signals view.
func (v SignalsView) Update(msg tea.Msg) (SignalsView, tea.Cmd) {
	switch m := msg.(type) {

	case SignalsDataMsg:
		v.State = m.State
		v.lastRefresh = time.Now()
		v.refreshTicker = 15
		v.loading = false
		v.err = ""
		return v, SignalsTickCmd()

	case SignalsErrorMsg:
		v.err = m.Err
		v.loading = false
		v.refreshTicker = 15
		return v, SignalsTickCmd()

	case SignalsRefreshTickMsg:
		v.refreshTicker--
		if v.refreshTicker <= 0 {
			v.loading = true
			v.refreshTicker = 15
			// caller should listen for this and fire the actual API fetch
		}
		return v, SignalsTickCmd()

	case tea.KeyMsg:
		switch m.String() {
		case "R":
			// manual refresh request
			v.loading = true
			v.refreshTicker = 15
		}
	}

	return v, nil
}

// ── Sparkline ─────────────────────────────────────────────────────────────────

func svSparkline(hist SignalHistory, minV, maxV float64) string {
	chars := []rune("▁▂▃▄▅▆▇█")
	span := maxV - minV
	if span < 1e-9 {
		span = 1
	}
	var sb strings.Builder
	for _, v := range hist {
		norm := (v - minV) / span
		idx := int(norm * float64(len(chars)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(chars) {
			idx = len(chars) - 1
		}
		sb.WriteRune(chars[idx])
	}
	return sb.String()
}

// ── BH Mass bar ───────────────────────────────────────────────────────────────

// bhMassBar renders an ASCII bar proportional to mass on a 0-5 scale, width=20.
func bhMassBar(mass float64) string {
	const barWidth = 20
	filled := int(math.Round(mass / 5.0 * barWidth))
	if filled > barWidth {
		filled = barWidth
	}
	if filled < 0 {
		filled = 0
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
	if mass >= 1.5 {
		return svStyleActive.Render(bar)
	}
	return svStyleDim.Render(bar)
}

// ── Hurst gauge ───────────────────────────────────────────────────────────────

func hurstColor(h float64) lipgloss.Style {
	if h > 0.58 {
		return svStyleGreen // trending
	}
	if h >= 0.42 {
		return svStyleYellow // random walk
	}
	return svStyleRed // mean-reverting
}

func hurstLabel(h float64) string {
	if h > 0.58 {
		return "TREND"
	}
	if h >= 0.42 {
		return "RW"
	}
	return "MR"
}

// ── GARCH render ─────────────────────────────────────────────────────────────

func renderGARCH(g GARCHSignal) string {
	ratio := g.CurrentForecast / math.Max(g.HistoricalMean, 0.001)
	ratioStr := ""
	if ratio > 1.5 {
		ratioStr = svStyleRed.Render(fmt.Sprintf("%.1fx mean (ELEVATED)", ratio))
	} else if ratio > 1.1 {
		ratioStr = svStyleYellow.Render(fmt.Sprintf("%.1fx mean", ratio))
	} else {
		ratioStr = svStyleGreen.Render(fmt.Sprintf("%.1fx mean", ratio))
	}

	sparkHist := svSparkline(g.History, 0, g.HistoricalMean*2.5)
	return fmt.Sprintf("GARCH Vol  %s%s  cur=%.1f%%  mean=%.1f%%  %s",
		svStyleDim.Render("["),
		svStyleNeutral.Render(sparkHist),
		svStyleDim.Render("]"),
		g.CurrentForecast*100,
		g.HistoricalMean*100,
		ratioStr,
	)
}

// ── NAV Curvature render ──────────────────────────────────────────────────────

func renderNavCurvature(nc NavCurvature) string {
	regime := svStyleNeutral.Render(nc.Regime)
	angle := nc.CurvatureAngle
	angleStr := ""
	if angle > 45 {
		angleStr = svStyleRed.Render(fmt.Sprintf("%.1f deg (HIGH CURVATURE)", angle))
	} else if angle > 20 {
		angleStr = svStyleYellow.Render(fmt.Sprintf("%.1f deg", angle))
	} else {
		angleStr = svStyleGreen.Render(fmt.Sprintf("%.1f deg", angle))
	}

	quat := svStyleDim.Render(fmt.Sprintf("q=[%.3f %.3f %.3f %.3f]",
		nc.W, nc.X, nc.Y, nc.Z))
	return fmt.Sprintf("NAV Curv   %s  regime=%s  %s",
		angleStr, regime, quat)
}

// ── RL Q-values render ────────────────────────────────────────────────────────

func renderRLQValues(vals []RLQValue) string {
	if len(vals) == 0 {
		return svStyleDim.Render("RL Q-vals  -- no data --")
	}
	// find max for relative bar
	maxQ := vals[0].QValue
	for _, v := range vals {
		if v.QValue > maxQ {
			maxQ = v.QValue
		}
	}
	var parts []string
	parts = append(parts, svStyleHeader.Render("RL Q-vals  top-3 actions:"))
	for i, rv := range vals {
		if i >= 3 {
			break
		}
		rel := 0.0
		if maxQ != 0 {
			rel = math.Abs(rv.QValue / maxQ)
		}
		barW := int(rel * 12)
		if barW < 0 {
			barW = 0
		}
		bar := strings.Repeat("▪", barW) + strings.Repeat(" ", 12-barW)
		style := svStyleNeutral
		if i == 0 {
			style = svStyleGreen
		}
		parts = append(parts, fmt.Sprintf("  %d. %-12s Q=%+.4f [%s]",
			i+1,
			style.Render(rv.Action),
			rv.QValue,
			svStyleDim.Render(bar),
		))
	}
	return strings.Join(parts, "\n")
}

// ── Section header ────────────────────────────────────────────────────────────

func svSectionHeader(title string) string {
	sep := strings.Repeat("─", 60)
	return svStyleTitle.Render(title) + "\n" + svStyleDim.Render(sep)
}

// ── View ──────────────────────────────────────────────────────────────────────

// View renders the full signals view.
func (v SignalsView) View() string {
	var sb strings.Builder

	// ── Status bar ────────────────────────────────────────────────────────────
	statusLine := svStyleTitle.Render("SIGNALS") + "  "
	if v.loading {
		statusLine += svStyleYellow.Render("[ refreshing... ]")
	} else if v.err != "" {
		statusLine += svStyleRed.Render("[ ERR: "+v.err+" ]")
	} else {
		statusLine += svStyleDim.Render(fmt.Sprintf("next refresh in %ds  updated %s  [R]=manual",
			v.refreshTicker,
			v.lastRefresh.Format("15:04:05")))
	}

	// ── Regime / confidence ───────────────────────────────────────────────────
	regimeStyle := svStyleNeutral
	switch v.State.Regime {
	case "BULL":
		regimeStyle = svStyleGreen
	case "BEAR":
		regimeStyle = svStyleRed
	case "HV":
		regimeStyle = svStyleYellow
	}
	confBar := int(v.State.Confidence * 20)
	if confBar > 20 {
		confBar = 20
	}
	confStr := svStyleGreen.Render(strings.Repeat("▪", confBar)) +
		svStyleDim.Render(strings.Repeat("░", 20-confBar))
	regimeLine := fmt.Sprintf("Regime: %s  Confidence: [%s] %.0f%%",
		regimeStyle.Render(fmt.Sprintf("%-9s", v.State.Regime)),
		confStr,
		v.State.Confidence*100,
	)

	sb.WriteString(statusLine + "\n")
	sb.WriteString(regimeLine + "\n")
	sb.WriteString(svStyleDim.Render(strings.Repeat("═", 72)) + "\n\n")

	// ── BH Mass panel ─────────────────────────────────────────────────────────
	sb.WriteString(svSectionHeader("BH MASS") + "\n")
	sb.WriteString(svStyleHeader.Render(
		fmt.Sprintf("  %-8s  %-22s  %-7s  %-14s  %s",
			"Symbol", "Bar (0-5 scale)", "Value", "5-bar history", "Status")) + "\n")

	if len(v.State.BHMasses) == 0 {
		sb.WriteString(svStyleDim.Render("  -- no BH mass data --") + "\n")
	}
	for _, bh := range v.State.BHMasses {
		spark := svSparkline(bh.History, 0, 5)
		status := svStyleDim.Render("dormant")
		if bh.Active {
			status = svStyleActive.Render("ACTIVE")
		}
		sb.WriteString(fmt.Sprintf("  %-8s  %s  %-7.3f  %s  %s\n",
			svStyleNeutral.Render(bh.Symbol),
			bhMassBar(bh.Mass),
			bh.Mass,
			svStyleNeutral.Render(spark),
			status,
		))
	}

	// ── Hurst panel ───────────────────────────────────────────────────────────
	sb.WriteString("\n" + svSectionHeader("HURST EXPONENT") + "\n")
	h := v.State.Hurst
	hurstSpark := svSparkline(h.History, 0, 1)
	hStyle := hurstColor(h.Value)
	sb.WriteString(fmt.Sprintf("  Value: %s  [%s]  5-bar: %s\n",
		hStyle.Render(fmt.Sprintf("%.4f", h.Value)),
		hStyle.Render(hurstLabel(h.Value)),
		svStyleNeutral.Render(hurstSpark),
	))
	// color-coded gauge bar
	gaugeW := 40
	hurstPos := int(h.Value * float64(gaugeW))
	if hurstPos > gaugeW {
		hurstPos = gaugeW
	}
	gauge := strings.Repeat("█", hurstPos) + strings.Repeat("░", gaugeW-hurstPos)
	sb.WriteString(fmt.Sprintf("  [%s]  0.42|0.58 bands\n",
		hStyle.Render(gauge)))

	// ── GARCH panel ───────────────────────────────────────────────────────────
	sb.WriteString("\n" + svSectionHeader("GARCH VOLATILITY") + "\n")
	sb.WriteString("  " + renderGARCH(v.State.GARCH) + "\n")

	// ── NAV Curvature ─────────────────────────────────────────────────────────
	sb.WriteString("\n" + svSectionHeader("NAV CURVATURE (quaternion)") + "\n")
	sb.WriteString("  " + renderNavCurvature(v.State.NavCurve) + "\n")

	// ── RL Q-values ───────────────────────────────────────────────────────────
	sb.WriteString("\n" + svSectionHeader("RL EXIT POLICY") + "\n")
	sb.WriteString(renderRLQValues(v.State.RLQValues) + "\n")

	return svStyleBorder.Render(sb.String())
}
