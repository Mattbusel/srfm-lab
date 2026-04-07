// Command genome-inspector is a CLI tool for inspecting the IAE genome
// population stored in a SQLite database.
//
// Usage:
//
//	genome-inspector --db=./iae.db <command> [args]
//
// Commands:
//
//	list                    -- show all genomes in the current (latest) generation
//	best                    -- show top 10 genomes with gene values and fitness
//	compare <id1> <id2>     -- gene-by-gene diff between two genomes
//	history <id>            -- fitness trajectory of one genome across generations
//	export <id>             -- export genome as JSON
//	stats                   -- population diversity, entropy, convergence index
package main

import (
	"database/sql"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"srfm-lab/idea-engine/pkg/analysis"
	"srfm-lab/idea-engine/pkg/persistence"
)

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

const usageText = `genome-inspector -- IAE genome population inspector

Usage:
  genome-inspector --db=<path> [--gen=<N|latest>] <command> [args]

Commands:
  list                  List all genomes in the target generation
  best                  Show top 10 genomes with full gene detail
  compare <id1> <id2>   Gene-by-gene diff between two genomes
  history <id>          Fitness trajectory of one genome
  export <id>           Export genome as JSON to stdout
  stats                 Population diversity, entropy, convergence

Flags:
`

type cliConfig struct {
	dbPath    string
	genStr    string
	noColor   bool
	topN      int
}

// ---------------------------------------------------------------------------
// ANSI color helpers
// ---------------------------------------------------------------------------

const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorCyan   = "\033[36m"
	colorBold   = "\033[1m"
)

func colorize(cfg cliConfig, color, s string) string {
	if cfg.noColor {
		return s
	}
	return color + s + colorReset
}

// fitnessColor returns the color appropriate for a fitness value relative to
// the threshold: green = good, yellow = borderline, red = below.
func fitnessColor(cfg cliConfig, f, threshold float64) string {
	switch {
	case f >= threshold*1.1:
		return colorize(cfg, colorGreen, fmt.Sprintf("%.6f", f))
	case f >= threshold:
		return colorize(cfg, colorYellow, fmt.Sprintf("%.6f", f))
	default:
		return colorize(cfg, colorRed, fmt.Sprintf("%.6f", f))
	}
}

// ---------------------------------------------------------------------------
// Database helpers
// ---------------------------------------------------------------------------

func openDB(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_foreign_keys=on")
	if err != nil {
		return nil, fmt.Errorf("open db %q: %w", path, err)
	}
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("ping db %q: %w", path, err)
	}
	return db, nil
}

func latestGeneration(gs *persistence.GenomeStore) int {
	gens := gs.ListGenerations()
	if len(gens) == 0 {
		return 0
	}
	return gens[len(gens)-1]
}

func parseGeneration(gs *persistence.GenomeStore, s string) int {
	if s == "latest" || s == "" {
		return latestGeneration(gs)
	}
	var n int
	if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
		return latestGeneration(gs)
	}
	return n
}

// ---------------------------------------------------------------------------
// Command implementations
// ---------------------------------------------------------------------------

// cmdList prints all genomes in the target generation in a table.
func cmdList(cfg cliConfig, gs *persistence.GenomeStore) {
	gen := parseGeneration(gs, cfg.genStr)
	records := gs.LoadGeneration(gen)
	if len(records) == 0 {
		fmt.Fprintf(os.Stderr, "No genomes found for generation %d\n", gen)
		os.Exit(1)
	}

	fmt.Printf("%s-- Generation %d  (%d individuals) %s\n",
		colorize(cfg, colorBold, ""),
		gen, len(records),
		colorReset,
	)

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ID\tFitness\tOperator\tParents\tCreatedAt")
	fmt.Fprintln(w, strings.Repeat("-", 80))

	// fitness threshold heuristic: median fitness
	fits := make([]float64, len(records))
	for i, r := range records {
		fits[i] = r.Fitness
	}
	threshold := median(fits)

	for _, r := range records {
		parentStr := strings.Join(r.ParentIDs, ",")
		if len(parentStr) > 24 {
			parentStr = parentStr[:21] + "..."
		}
		shortID := r.ID
		if len(shortID) > 16 {
			shortID = shortID[:16]
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n",
			shortID,
			fitnessColor(cfg, r.Fitness, threshold),
			r.Operator,
			parentStr,
			r.CreatedAt.Format(time.RFC3339),
		)
	}
	w.Flush()
}

// cmdBest prints the top N genomes with full gene detail.
func cmdBest(cfg cliConfig, gs *persistence.GenomeStore) {
	records := gs.GetTopN(cfg.topN)
	if len(records) == 0 {
		fmt.Fprintln(os.Stderr, "No genomes found in database")
		os.Exit(1)
	}

	fits := make([]float64, len(records))
	for i, r := range records {
		fits[i] = r.Fitness
	}
	threshold := median(fits)

	fmt.Printf("%s-- Top %d Genomes (all generations) %s\n",
		colorize(cfg, colorBold, ""), cfg.topN, colorReset)

	for rank, r := range records {
		fmt.Printf("\n%s#%d  ID: %s  Gen: %d  Fitness: %s  Op: %s%s\n",
			colorize(cfg, colorCyan, ""),
			rank+1, r.ID, r.Generation,
			fitnessColor(cfg, r.Fitness, threshold),
			r.Operator,
			colorReset,
		)
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "  Gene\tValue")
		fmt.Fprintln(w, "  "+strings.Repeat("-", 30))
		for k, v := range r.Genes {
			fmt.Fprintf(w, "  gene[%03d]\t%.8f\n", k, v)
		}
		w.Flush()
	}
}

// cmdCompare prints a gene-by-gene diff between two genomes.
func cmdCompare(cfg cliConfig, gs *persistence.GenomeStore, id1, id2 string) {
	r1 := gs.GetByID(id1)
	r2 := gs.GetByID(id2)

	if r1 == nil {
		fmt.Fprintf(os.Stderr, "Genome %q not found\n", id1)
		os.Exit(1)
	}
	if r2 == nil {
		fmt.Fprintf(os.Stderr, "Genome %q not found\n", id2)
		os.Exit(1)
	}

	fmt.Printf("-- Comparing genomes\n")
	fmt.Printf("   A: %s  (gen %d, fitness %.6f)\n", r1.ID, r1.Generation, r1.Fitness)
	fmt.Printf("   B: %s  (gen %d, fitness %.6f)\n", r2.ID, r2.Generation, r2.Fitness)
	fmt.Println()

	maxLen := len(r1.Genes)
	if len(r2.Genes) > maxLen {
		maxLen = len(r2.Genes)
	}

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Gene\tValue A\tValue B\tDelta\tDelta%")
	fmt.Fprintln(w, strings.Repeat("-", 60))

	for k := 0; k < maxLen; k++ {
		var a, b float64
		if k < len(r1.Genes) {
			a = r1.Genes[k]
		}
		if k < len(r2.Genes) {
			b = r2.Genes[k]
		}
		delta := b - a
		var pct float64
		if a != 0 {
			pct = delta / math.Abs(a) * 100
		}
		deltaStr := fmt.Sprintf("%+.8f", delta)
		if math.Abs(delta) > 0.01 {
			deltaStr = colorize(cfg, colorYellow, deltaStr)
		}
		fmt.Fprintf(w, "gene[%03d]\t%.8f\t%.8f\t%s\t%+.2f%%\n",
			k, a, b, deltaStr, pct)
	}
	w.Flush()

	// Summary
	l2 := 0.0
	for k := 0; k < maxLen; k++ {
		var a, b float64
		if k < len(r1.Genes) {
			a = r1.Genes[k]
		}
		if k < len(r2.Genes) {
			b = r2.Genes[k]
		}
		d := a - b
		l2 += d * d
	}
	fmt.Printf("\nL2 distance: %s\n",
		colorize(cfg, colorCyan, fmt.Sprintf("%.6f", math.Sqrt(l2))))
	fmt.Printf("Fitness delta: %+.6f\n", r2.Fitness-r1.Fitness)
}

// cmdHistory prints the fitness trajectory of one genome.
func cmdHistory(cfg cliConfig, gs *persistence.GenomeStore, id string) {
	r := gs.GetByID(id)
	if r == nil {
		fmt.Fprintf(os.Stderr, "Genome %q not found\n", id)
		os.Exit(1)
	}
	history := gs.GetFitnessHistory(id)
	if len(history) == 0 {
		fmt.Fprintf(os.Stderr, "No fitness history found for %q\n", id)
		os.Exit(1)
	}

	fmt.Printf("-- Fitness history for %s\n\n", r.ID)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Index\tFitness\tChange")
	fmt.Fprintln(w, strings.Repeat("-", 40))

	for i, f := range history {
		changeStr := "--"
		if i > 0 {
			delta := f - history[i-1]
			if delta > 0 {
				changeStr = colorize(cfg, colorGreen, fmt.Sprintf("+%.6f", delta))
			} else if delta < 0 {
				changeStr = colorize(cfg, colorRed, fmt.Sprintf("%.6f", delta))
			} else {
				changeStr = "0.000000"
			}
		}
		fmt.Fprintf(w, "%d\t%.6f\t%s\n", i, f, changeStr)
	}
	w.Flush()
}

// cmdExport writes a genome as JSON to stdout.
func cmdExport(gs *persistence.GenomeStore, id string) {
	r := gs.GetByID(id)
	if r == nil {
		fmt.Fprintf(os.Stderr, "Genome %q not found\n", id)
		os.Exit(1)
	}
	out := map[string]interface{}{
		"id":         r.ID,
		"generation": r.Generation,
		"genes":      r.Genes,
		"fitness":    r.Fitness,
		"parent_ids": r.ParentIDs,
		"operator":   r.Operator,
		"created_at": r.CreatedAt.Format(time.RFC3339),
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		fmt.Fprintf(os.Stderr, "export encode: %v\n", err)
		os.Exit(1)
	}
}

// cmdStats prints population diversity, entropy, and convergence index.
func cmdStats(cfg cliConfig, gs *persistence.GenomeStore) {
	gen := parseGeneration(gs, cfg.genStr)
	records := gs.LoadGeneration(gen)
	if len(records) == 0 {
		fmt.Fprintf(os.Stderr, "No genomes found for generation %d\n", gen)
		os.Exit(1)
	}

	genomes := make([][]float64, len(records))
	fits := make([]float64, len(records))
	for i, r := range records {
		genomes[i] = r.Genes
		fits[i] = r.Fitness
	}

	analyzer := analysis.NewGenomeAnalyzer(nil)
	stats := analyzer.ComputeStats(genomes, fits)
	stats.Generation = gen

	div := gs.ComputeGeneticDiversity(gen)

	fmt.Printf("%s-- Population Statistics  Generation %d%s\n\n",
		colorize(cfg, colorBold, ""), gen, colorReset)

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "Metric\tValue")
	fmt.Fprintln(w, strings.Repeat("-", 40))
	fmt.Fprintf(w, "Population size\t%d\n", len(records))
	fmt.Fprintf(w, "Best fitness\t%s\n", colorize(cfg, colorGreen, fmt.Sprintf("%.6f", stats.BestFitness)))
	fmt.Fprintf(w, "Average fitness\t%.6f\n", stats.AvgFitness)
	fmt.Fprintf(w, "Worst fitness\t%s\n", colorize(cfg, colorRed, fmt.Sprintf("%.6f", stats.WorstFitness)))
	fmt.Fprintf(w, "Diversity (mean pairwise dist)\t%.6f\n", div)
	fmt.Fprintf(w, "Shannon entropy\t%.6f bits\n", stats.Entropy)
	fmt.Fprintf(w, "Convergence index\t%.4f\n", stats.ConvergenceIdx)
	w.Flush()

	// Niche summary
	niches := analyzer.FindNiches(genomes, div*0.5)
	fmt.Printf("\n-- Niches (radius = %.4f)\n", div*0.5)
	for i, niche := range niches {
		fmt.Printf("  Niche %d: %d members\n", i+1, len(niche))
	}
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	cfg := cliConfig{}
	flag.StringVar(&cfg.dbPath, "db", "./iae.db", "Path to the IAE SQLite database")
	flag.StringVar(&cfg.genStr, "gen", "latest", "Generation number, or 'latest'")
	flag.BoolVar(&cfg.noColor, "no-color", false, "Disable ANSI color output")
	flag.IntVar(&cfg.topN, "top", 10, "Number of top genomes shown by the 'best' command")
	flag.Usage = func() {
		fmt.Fprint(os.Stderr, usageText)
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(1)
	}

	db, err := openDB(cfg.dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()

	gs, err := persistence.NewGenomeStore(db)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening genome store: %v\n", err)
		os.Exit(1)
	}

	cmd := strings.ToLower(args[0])
	switch cmd {
	case "list":
		cmdList(cfg, gs)

	case "best":
		cmdBest(cfg, gs)

	case "compare":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: genome-inspector compare <id1> <id2>")
			os.Exit(1)
		}
		cmdCompare(cfg, gs, args[1], args[2])

	case "history":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: genome-inspector history <id>")
			os.Exit(1)
		}
		cmdHistory(cfg, gs, args[1])

	case "export":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: genome-inspector export <id>")
			os.Exit(1)
		}
		cmdExport(gs, args[1])

	case "stats":
		cmdStats(cfg, gs)

	default:
		fmt.Fprintf(os.Stderr, "Unknown command %q\n\n", cmd)
		flag.Usage()
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Internal math helpers
// ---------------------------------------------------------------------------

// median returns the median of a float64 slice. Does not modify the input.
func median(xs []float64) float64 {
	if len(xs) == 0 {
		return 0.0
	}
	c := make([]float64, len(xs))
	copy(c, xs)
	sort.Float64s(c)
	n := len(c)
	if n%2 == 0 {
		return (c[n/2-1] + c[n/2]) / 2.0
	}
	return c[n/2]
}
