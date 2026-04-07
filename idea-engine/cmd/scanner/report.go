package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"srfm-lab/idea-engine/pkg/signal_discovery"
)

// GenerateScanReport writes both a Markdown summary and a JSON export for the
// given signal candidates. outputPath should end in ".json"; the Markdown file
// will be written alongside it with a ".md" extension.
func GenerateScanReport(candidates []signal_discovery.SignalCandidate, outputPath string) error {
	// Write JSON.
	if err := exportJSON(candidates, outputPath); err != nil {
		return fmt.Errorf("export JSON: %w", err)
	}

	// Derive Markdown path.
	mdPath := outputPath
	if strings.HasSuffix(mdPath, ".json") {
		mdPath = mdPath[:len(mdPath)-5] + ".md"
	} else {
		mdPath += ".md"
	}

	if err := writeMarkdown(candidates, mdPath); err != nil {
		return fmt.Errorf("write markdown: %w", err)
	}
	return nil
}

// writeMarkdown generates a Markdown report sorted by ICIR descending.
func writeMarkdown(candidates []signal_discovery.SignalCandidate, path string) error {
	// Sort by ICIR descending (may already be sorted, but ensure it).
	sorted := make([]signal_discovery.SignalCandidate, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].TestICIR > sorted[j].TestICIR
	})

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create markdown file: %w", err)
	}
	defer f.Close()

	w := func(format string, args ...interface{}) {
		fmt.Fprintf(f, format+"\n", args...)
	}

	w("# Signal Discovery Scan Report")
	w("")
	w("Generated: %s", time.Now().UTC().Format(time.RFC3339))
	w("")
	w("Total candidates passing gates: **%d**", len(sorted))
	w("")

	if len(sorted) == 0 {
		w("No candidates passed the quality gates.")
		return nil
	}

	w("## Candidates (sorted by ICIR desc)")
	w("")
	w("| Rank | Name | Formula | IC | ICIR | Novel |")
	w("|------|------|---------|-----|------|-------|")

	for i, c := range sorted {
		novel := "no"
		if c.IsNovel {
			novel = "yes"
		}
		w("| %d | %s | %s | %.4f | %.4f | %s |",
			i+1,
			markdownEscape(c.Name),
			markdownEscape(c.Formula),
			c.TestIC,
			c.TestICIR,
			novel,
		)
	}
	w("")

	// Parameter details section.
	w("## Parameter Details")
	w("")
	for i, c := range sorted {
		w("### %d. %s", i+1, c.Name)
		w("")
		if len(c.Params) > 0 {
			w("| Parameter | Value |")
			w("|-----------|-------|")
			// Sort params for deterministic output.
			keys := make([]string, 0, len(c.Params))
			for k := range c.Params {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				w("| %s | %.6g |", k, c.Params[k])
			}
		} else {
			w("No parameters.")
		}
		w("")
	}

	return nil
}

// markdownEscape replaces pipe characters to avoid breaking Markdown tables.
func markdownEscape(s string) string {
	return strings.ReplaceAll(s, "|", "\\|")
}
