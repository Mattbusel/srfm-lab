//go:build ignore

// snapshot.go — Static ASCII snapshot fallback (no external deps)
// Run with: go run snapshot.go
//
// Use this if bubbletea cannot be installed:
//   go run cmd/srfm-tui/snapshot.go
package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"time"
)

type InstSnap struct {
	Symbol   string
	BHMass   float64
	BHActive bool
	Regime   string
	CTL      int
	Position float64
	PnL      float64
}

type PortSnap struct {
	Instruments    []InstSnap
	PortfolioValue float64
	PeakValue      float64
	Drawdown       float64
	BarCount       int
	Timestamp      string
}

func loadOrSim(jsonPath string) PortSnap {
	if jsonPath != "" {
		b, err := os.ReadFile(jsonPath)
		if err == nil {
			var raw map[string]interface{}
			if json.Unmarshal(b, &raw) == nil {
				// parse if matches expected schema
			}
		}
	}
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	insts := []InstSnap{
		{Symbol: "ES", BHMass: 1.0 + rng.Float64()*1.2, Regime: "BULL", CTL: rng.Intn(10) + 1, Position: 65, PnL: rng.Float64() * 20000},
		{Symbol: "NQ", BHMass: 0.5 + rng.Float64()*1.0, Regime: "SIDEWAYS", CTL: rng.Intn(5) + 1, Position: 32.5, PnL: rng.Float64()*10000 - 2000},
		{Symbol: "YM", BHMass: 1.2 + rng.Float64()*0.9, Regime: "BULL", CTL: rng.Intn(8) + 2, Position: 65, PnL: rng.Float64() * 15000},
	}
	for i := range insts {
		insts[i].BHActive = insts[i].BHMass >= 1.5
	}
	pv := 1_000_000.0 + rng.Float64()*300_000
	return PortSnap{
		Instruments:    insts,
		PortfolioValue: pv,
		PeakValue:      1_300_000,
		Drawdown:       (1_300_000 - pv) / 1_300_000 * 100,
		BarCount:       rng.Intn(5000),
		Timestamp:      time.Now().Format("2006-01-02 15:04:05"),
	}
}

func massBar(mass float64, w int) string {
	filled := int(mass / 2.5 * float64(w))
	if filled > w {
		filled = w
	}
	return strings.Repeat("#", filled) + strings.Repeat("-", w-filled)
}

func main() {
	jsonPath := ""
	for i, arg := range os.Args[1:] {
		if arg == "--json" && i+1 < len(os.Args)-1 {
			jsonPath = os.Args[i+2]
		}
	}

	snap := loadOrSim(jsonPath)

	width := 68
	line := strings.Repeat("─", width)
	fmt.Printf("┌%s┐\n", line)
	fmt.Printf("│  SRFM TERMINAL SNAPSHOT   %s  Bar #%d%s│\n",
		snap.Timestamp, snap.BarCount,
		strings.Repeat(" ", width-len(snap.Timestamp)-28-len(fmt.Sprintf("%d", snap.BarCount))))
	fmt.Printf("├%s┤\n", line)

	for _, inst := range snap.Instruments {
		bh := "○"
		if inst.BHActive {
			bh = "■"
		}
		bar := massBar(inst.BHMass, 20)
		posSign := "+"
		if inst.Position < 0 {
			posSign = ""
		}
		pnlSign := "+"
		if inst.PnL < 0 {
			pnlSign = ""
		}
		fmt.Printf("│  %s %s  BH: %.3f [%s]  Regime: %-8s CTL:%2d%s│\n",
			bh, inst.Symbol, inst.BHMass, bar, inst.Regime, inst.CTL,
			strings.Repeat(" ", width-60))
		fmt.Printf("│     Pos: %s%.1f%%    P&L: %s$%.0f%s│\n",
			posSign, inst.Position, pnlSign, inst.PnL,
			strings.Repeat(" ", width-32))
		fmt.Printf("│%s│\n", strings.Repeat(" ", width))
	}

	fmt.Printf("├%s┤\n", line)

	active := 0
	for _, i := range snap.Instruments {
		if i.BHActive {
			active++
		}
	}
	convDots := strings.Repeat("■", active) + strings.Repeat("□", 3-active)
	convNote := ""
	if active >= 2 {
		convNote = " → CONVERGENCE MULTIPLIER ACTIVE"
	}
	fmt.Printf("│  CONV: %s  (%d/3 BH active%s)%s│\n",
		convDots, active, convNote,
		strings.Repeat(" ", width-30-len(convNote)))
	fmt.Printf("│  Portfolio: $%,.0f   Peak: $%,.0f   DD: %.2f%%%s│\n",
		snap.PortfolioValue, snap.PeakValue, snap.Drawdown,
		strings.Repeat(" ", 5))
	fmt.Printf("└%s┘\n", line)

	// Save snapshot
	os.MkdirAll("results", 0o755)
	path := fmt.Sprintf("results/tui_snapshot_%s.txt",
		time.Now().Format("20060102_150405"))
	var out strings.Builder
	out.WriteString(fmt.Sprintf("SRFM TUI Snapshot\nTime: %s\n", snap.Timestamp))
	out.WriteString(fmt.Sprintf("Portfolio: $%.0f  Peak: $%.0f  DD: %.2f%%\n",
		snap.PortfolioValue, snap.PeakValue, snap.Drawdown))
	for _, i := range snap.Instruments {
		bh := "○"
		if i.BHActive {
			bh = "●"
		}
		out.WriteString(fmt.Sprintf("%s %s BHMass=%.3f CTL=%d Regime=%s Pos=%.1f%% PnL=$%.0f\n",
			bh, i.Symbol, i.BHMass, i.CTL, i.Regime, i.Position, i.PnL))
	}
	os.WriteFile(path, []byte(out.String()), 0o644)
	fmt.Printf("\nSnapshot saved: %s\n", path)
}
