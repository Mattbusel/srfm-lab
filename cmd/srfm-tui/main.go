// srfm-tui: Real-time SRFM strategy terminal dashboard.
// "The Bloomberg Terminal for SRFM"
//
// Usage:
//
//	go run cmd/srfm-tui/main.go
//	go run cmd/srfm-tui/main.go --json results/latest_run.json
//
// Shows: BH mass per instrument, regime, position, equity, drawdown
// Updates: every second (reads from JSON file or simulated data)
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// ── Styles ────────────────────────────────────────────────────────────────────

var (
	styleBorder = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("#4444aa")).
			Padding(0, 1)

	styleBHActive = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88")).Bold(true)
	styleInactive = lipgloss.NewStyle().Foreground(lipgloss.Color("#666688"))
	styleBull     = lipgloss.NewStyle().Foreground(lipgloss.Color("#00dd66"))
	styleBear     = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff4444"))
	styleSide     = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffcc00"))
	styleHV       = lipgloss.NewStyle().Foreground(lipgloss.Color("#cc44ff"))
	styleTitle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#58a6ff")).Bold(true)
	styleDD       = lipgloss.NewStyle().Foreground(lipgloss.Color("#ff6666"))
	styleProfit   = lipgloss.NewStyle().Foreground(lipgloss.Color("#00ff88"))
	styleNeutral  = lipgloss.NewStyle().Foreground(lipgloss.Color("#c9d1d9"))
	styleKey      = lipgloss.NewStyle().Foreground(lipgloss.Color("#888899"))
	styleConv     = lipgloss.NewStyle().Foreground(lipgloss.Color("#ffdd00")).Bold(true)
)

// ── Data types ────────────────────────────────────────────────────────────────

type InstrumentState struct {
	Symbol   string
	BHMass   float64
	BHActive bool
	CTL      int
	Regime   string // BULL/BEAR/SIDEWAYS/HV
	Position float64
	PnL      float64
	// internal sim state
	massVel   float64
	massTgt   float64
	regimeTik int
}

type Model struct {
	instruments    map[string]*InstrumentState
	instrumentKeys []string
	portfolioValue float64
	peakValue      float64
	drawdown       float64
	barCount       int
	lastUpdate     time.Time
	simMode        bool
	speed          int // ticks per "bar"
	tickCount      int
	massHistory    map[string][]float64
	histLen        int
	jsonPath       string
	quitting       bool
	snapshotMsg    string
	snapshotTick   int
}

// ── Tea messages ─────────────────────────────────────────────────────────────

type tickMsg time.Time
type snapshotDoneMsg string

// ── Model init ────────────────────────────────────────────────────────────────

func initialModel(jsonPath string) Model {
	syms := []string{"ES", "NQ", "YM"}
	insts := make(map[string]*InstrumentState, 3)
	hist := make(map[string][]float64, 3)

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	regimes := []string{"BULL", "SIDEWAYS", "BULL", "BEAR", "HV"}
	for i, s := range syms {
		mass := 0.8 + rng.Float64()*0.8
		insts[s] = &InstrumentState{
			Symbol:    s,
			BHMass:    mass,
			BHActive:  mass >= 1.5,
			CTL:       rng.Intn(10) + 1,
			Regime:    regimes[i%len(regimes)],
			Position:  (rng.Float64()*0.6 + 0.1) * 100,
			PnL:       (rng.Float64()*2 - 0.5) * 20000,
			massVel:   0,
			massTgt:   1.2 + rng.Float64()*0.8,
			regimeTik: rng.Intn(50) + 20,
		}
		hist[s] = make([]float64, 20)
		for j := range hist[s] {
			hist[s][j] = mass * (0.7 + rng.Float64()*0.6)
		}
	}

	return Model{
		instruments:    insts,
		instrumentKeys: syms,
		portfolioValue: 1_000_000 + rng.Float64()*300_000,
		peakValue:      1_300_000,
		barCount:       0,
		lastUpdate:     time.Now(),
		simMode:        jsonPath == "",
		speed:          1,
		histLen:        20,
		massHistory:    hist,
		jsonPath:       jsonPath,
	}
}

// ── BubbleTea interface ───────────────────────────────────────────────────────

func (m Model) Init() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			m.quitting = true
			return m, tea.Quit
		case "r":
			newM := initialModel(m.jsonPath)
			newM.speed = m.speed
			return newM, tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
		case "+", "=":
			if m.speed < 10 {
				m.speed++
			}
		case "-":
			if m.speed > 1 {
				m.speed--
			}
		case "s":
			return m, func() tea.Msg {
				return snapshotDoneMsg(m.saveSnapshot())
			}
		}

	case snapshotDoneMsg:
		m.snapshotMsg = string(msg)
		m.snapshotTick = 3

	case tickMsg:
		m.lastUpdate = time.Time(msg)
		m.tickCount++

		if m.simMode {
			for _ = range m.speed {
				m.simStep()
			}
		} else {
			m.loadJSON()
		}

		if m.snapshotTick > 0 {
			m.snapshotTick--
		} else {
			m.snapshotMsg = ""
		}

		return m, tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
	}

	return m, nil
}

// ── Simulation step ───────────────────────────────────────────────────────────

var regimeList = []string{"BULL", "BULL", "BULL", "SIDEWAYS", "BEAR", "HV"}

func (m *Model) simStep() {
	rng := rand.New(rand.NewSource(time.Now().UnixNano() + int64(m.barCount)))
	m.barCount++

	totalPnL := 0.0
	for _, k := range m.instrumentKeys {
		inst := m.instruments[k]

		// Regime transitions
		inst.regimeTik--
		if inst.regimeTik <= 0 {
			inst.Regime = regimeList[rng.Intn(len(regimeList))]
			inst.regimeTik = rng.Intn(80) + 20
			// When bull, aim for higher mass
			switch inst.Regime {
			case "BULL":
				inst.massTgt = 1.4 + rng.Float64()*0.8
			case "BEAR":
				inst.massTgt = 0.4 + rng.Float64()*0.6
			case "HV":
				inst.massTgt = 1.8 + rng.Float64()*0.5
			default:
				inst.massTgt = 0.8 + rng.Float64()*0.6
			}
		}

		// BH mass spring-damper toward target
		force := (inst.massTgt - inst.BHMass) * 0.05
		noise := (rng.Float64()*2 - 1) * 0.04
		inst.massVel = inst.massVel*0.85 + force + noise
		inst.BHMass = math.Max(0.0, inst.BHMass+inst.massVel)
		inst.BHActive = inst.BHMass >= 1.5

		// CTL random walk
		if rng.Float64() < 0.1 {
			inst.CTL += rng.Intn(3) - 1
			if inst.CTL < 0 {
				inst.CTL = 0
			}
		}

		// Position driven by BH mass + regime
		tgtPos := 0.0
		if inst.BHActive {
			switch inst.Regime {
			case "BULL":
				tgtPos = 65.0
			case "BEAR":
				tgtPos = -32.5
			case "HV":
				tgtPos = 32.5
			default:
				tgtPos = 32.5
			}
		}
		inst.Position = inst.Position*0.9 + tgtPos*0.1

		// P&L tick
		pnlNoise := (rng.Float64()*2-1)*500 + inst.Position*50
		if inst.BHActive {
			if inst.Regime == "BULL" {
				pnlNoise += 200
			} else if inst.Regime == "BEAR" {
				pnlNoise -= 100
			}
		}
		inst.PnL += pnlNoise
		totalPnL += pnlNoise

		// Update history
		hist := m.massHistory[k]
		hist = append(hist[1:], inst.BHMass)
		m.massHistory[k] = hist
	}

	m.portfolioValue += totalPnL
	if m.portfolioValue > m.peakValue {
		m.peakValue = m.portfolioValue
	}
	m.drawdown = (m.peakValue - m.portfolioValue) / m.peakValue * 100
}

// ── JSON loading ──────────────────────────────────────────────────────────────

type jsonSnapshot struct {
	Instruments map[string]struct {
		BHMass   float64 `json:"bh_mass"`
		Regime   string  `json:"regime"`
		CTL      int     `json:"ctl"`
		Position float64 `json:"position"`
		PnL      float64 `json:"pnl"`
	} `json:"instruments"`
	PortfolioValue float64 `json:"portfolio_value"`
}

func (m *Model) loadJSON() {
	if m.jsonPath == "" {
		return
	}
	b, err := os.ReadFile(m.jsonPath)
	if err != nil {
		return
	}
	var snap jsonSnapshot
	if err := json.Unmarshal(b, &snap); err != nil {
		return
	}
	for k, v := range snap.Instruments {
		if inst, ok := m.instruments[k]; ok {
			inst.BHMass = v.BHMass
			inst.BHActive = v.BHMass >= 1.5
			inst.Regime = v.Regime
			inst.CTL = v.CTL
			inst.Position = v.Position
			inst.PnL = v.PnL
		}
	}
	if snap.PortfolioValue > 0 {
		m.portfolioValue = snap.PortfolioValue
	}
}

// ── Snapshot ──────────────────────────────────────────────────────────────────

func (m *Model) saveSnapshot() string {
	os.MkdirAll("results", 0o755)
	path := fmt.Sprintf("results/tui_snapshot_%s.txt",
		time.Now().Format("20060102_150405"))
	lines := []string{
		"SRFM TUI Snapshot",
		fmt.Sprintf("Time: %s", m.lastUpdate.Format("2006-01-02 15:04:05")),
		fmt.Sprintf("Portfolio: $%.0f  Peak: $%.0f  DD: %.2f%%",
			m.portfolioValue, m.peakValue, m.drawdown),
		"",
	}
	for _, k := range m.instrumentKeys {
		inst := m.instruments[k]
		bh := "○"
		if inst.BHActive {
			bh = "●"
		}
		lines = append(lines, fmt.Sprintf("%s %s BHMass=%.3f CTL=%d Regime=%s Pos=%.1f%% PnL=$%.0f",
			bh, k, inst.BHMass, inst.CTL, inst.Regime, inst.Position, inst.PnL))
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		return "snapshot failed: " + err.Error()
	}
	return "snapshot saved: " + path
}

// ── View ──────────────────────────────────────────────────────────────────────

const BH_THRESHOLD = 1.5

func regimeStyle(r string) lipgloss.Style {
	switch r {
	case "BULL":
		return styleBull
	case "BEAR":
		return styleBear
	case "HV":
		return styleHV
	default:
		return styleSide
	}
}

func massBar(mass float64, width int) string {
	filled := int(math.Round(mass / 2.5 * float64(width)))
	if filled > width {
		filled = width
	}
	bar := strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
	if mass >= BH_THRESHOLD {
		return styleBHActive.Render(bar)
	}
	return styleInactive.Render(bar)
}

func sparkline(hist []float64) string {
	chars := []rune("▁▂▃▄▅▆▇█")
	maxV := 0.0
	for _, v := range hist {
		if v > maxV {
			maxV = v
		}
	}
	if maxV == 0 {
		maxV = 1
	}
	var sb strings.Builder
	for _, v := range hist {
		idx := int(v / maxV * float64(len(chars)-1))
		if idx >= len(chars) {
			idx = len(chars) - 1
		}
		if v >= BH_THRESHOLD {
			sb.WriteString(styleBHActive.Render(string(chars[idx])))
		} else {
			sb.WriteString(styleInactive.Render(string(chars[idx])))
		}
	}
	return sb.String()
}

func fmtPnL(v float64) string {
	s := fmt.Sprintf("%+.0f", v)
	if v >= 0 {
		return styleProfit.Render("$" + s[1:] + " ▲")
	}
	return styleDD.Render("$-" + s[2:] + " ▼")
}

func (m Model) View() string {
	if m.quitting {
		return styleTitle.Render("SRFM Terminal exited.\n")
	}

	width := 72

	// ── Header ────────────────────────────────────────────────────────────────
	ts := m.lastUpdate.Format("2006-01-02 15:04:05")
	title := styleTitle.Render("SRFM TERMINAL")
	header := fmt.Sprintf(" %s %s Bar #%d ", title, styleNeutral.Render(ts), m.barCount)
	topLine := "┌" + strings.Repeat("─", width-2) + "┐"
	botLine := "└" + strings.Repeat("─", width-2) + "┘"
	midLine := "├" + strings.Repeat("─", width-2) + "┤"
	pad := func(s string, w int) string {
		vis := lipgloss.Width(s)
		if vis < w {
			return s + strings.Repeat(" ", w-vis)
		}
		return s
	}
	row := func(content string) string {
		return "│ " + pad(content, width-4) + " │"
	}

	var sb strings.Builder
	sb.WriteString(topLine + "\n")
	sb.WriteString(row(header) + "\n")
	sb.WriteString(midLine + "\n")

	// ── Instrument panels ─────────────────────────────────────────────────────
	panelW := 20
	var cols [3]string
	for i, k := range m.instrumentKeys {
		inst := m.instruments[k]
		bhInd := "○"
		bhStyle := styleInactive
		if inst.BHActive {
			bhInd = "■"
			bhStyle = styleBHActive
		}
		regStr := regimeStyle(inst.Regime).Render(fmt.Sprintf("%-8s", inst.Regime))
		posStr := ""
		if inst.Position >= 0 {
			posStr = styleProfit.Render(fmt.Sprintf("%+.1f%%", inst.Position))
		} else {
			posStr = styleDD.Render(fmt.Sprintf("%+.1f%%", inst.Position))
		}

		lines := []string{
			styleBorder.Render(fmt.Sprintf("  %s  ", styleTitle.Render(k))),
			bhStyle.Render(fmt.Sprintf("BH: %.3f %s", inst.BHMass, bhInd)),
			"Regime: " + regStr,
			fmt.Sprintf("CTL:    %d", inst.CTL),
			"Pos: " + posStr,
			"PnL: " + fmtPnL(inst.PnL),
		}
		_ = i
		cols[i] = strings.Join(lines, "\n")
	}

	// Render columns side by side
	col0 := strings.Split(cols[0], "\n")
	col1 := strings.Split(cols[1], "\n")
	col2 := strings.Split(cols[2], "\n")
	maxLines := len(col0)
	if len(col1) > maxLines {
		maxLines = len(col1)
	}
	if len(col2) > maxLines {
		maxLines = len(col2)
	}
	for len(col0) < maxLines {
		col0 = append(col0, "")
	}
	for len(col1) < maxLines {
		col1 = append(col1, "")
	}
	for len(col2) < maxLines {
		col2 = append(col2, "")
	}

	for i := range maxLines {
		line := pad(col0[i], panelW) + "  " + pad(col1[i], panelW) + "  " + pad(col2[i], panelW)
		sb.WriteString(row(line) + "\n")
	}

	sb.WriteString(midLine + "\n")

	// ── Portfolio summary ─────────────────────────────────────────────────────
	ddStr := fmt.Sprintf("DD: %.2f%%", m.drawdown)
	if m.drawdown > 5 {
		ddStr = styleDD.Render(ddStr)
	} else {
		ddStr = styleProfit.Render(ddStr)
	}
	portLine := fmt.Sprintf("Portfolio: %s  Peak: %s  %s",
		styleProfit.Render(fmt.Sprintf("$%,.0f", m.portfolioValue)),
		styleNeutral.Render(fmt.Sprintf("$%,.0f", m.peakValue)),
		ddStr)
	sb.WriteString(row(portLine) + "\n")

	// ── Convergence indicator ─────────────────────────────────────────────────
	active := 0
	convSyms := []string{}
	for _, k := range m.instrumentKeys {
		if m.instruments[k].BHActive {
			active++
			convSyms = append(convSyms, k)
		}
	}
	convDots := strings.Repeat("■", active) + strings.Repeat("□", 3-active)
	convMsg := ""
	if active >= 2 {
		convMsg = styleConv.Render(fmt.Sprintf("CONV: %s (%d/3 BH active → CONVERGENCE MULTIPLIER ACTIVE [%s])",
			convDots, active, strings.Join(convSyms, "+")))
	} else {
		convMsg = styleInactive.Render(fmt.Sprintf("CONV: %s (%d/3 BH active)", convDots, active))
	}
	sb.WriteString(row(convMsg) + "\n")

	sb.WriteString(midLine + "\n")

	// ── Sparkline chart ───────────────────────────────────────────────────────
	sb.WriteString(row(styleTitle.Render("BH Mass History (last 20 bars):")) + "\n")
	for _, k := range m.instrumentKeys {
		hist := m.massHistory[k]
		spark := sparkline(hist)
		label := fmt.Sprintf("%-3s │%s│ %.3f", k, spark, m.instruments[k].BHMass)
		sb.WriteString(row(label) + "\n")
	}
	threshLine := fmt.Sprintf("    %s threshold=1.5",
		styleNeutral.Render("────────────────────"))
	sb.WriteString(row(threshLine) + "\n")

	sb.WriteString(midLine + "\n")

	// ── Key hints ─────────────────────────────────────────────────────────────
	keyHints := styleKey.Render(fmt.Sprintf(
		"[q]uit  [r]eset  [+/-] speed(×%d)  [s]napshot", m.speed))
	sb.WriteString(row(keyHints) + "\n")

	if m.snapshotMsg != "" {
		sb.WriteString(row(styleProfit.Render(m.snapshotMsg)) + "\n")
	}

	sb.WriteString(botLine + "\n")

	return sb.String()
}

// ── Entry point ───────────────────────────────────────────────────────────────

func main() {
	jsonPath := ""
	for i, arg := range os.Args[1:] {
		if arg == "--json" && i+1 < len(os.Args)-1 {
			jsonPath = os.Args[i+2]
		}
	}

	p := tea.NewProgram(initialModel(jsonPath), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error running TUI: %v\n", err)
		os.Exit(1)
	}
}
