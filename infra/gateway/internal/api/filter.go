package api

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/srfm/gateway/internal/feed"
)

// BarFilter holds parsed query parameters for bar queries.
type BarFilter struct {
	Symbol    string
	Timeframe string
	From      time.Time
	To        time.Time
	Limit     int
	Ascending bool
}

// ParseBarFilter extracts BarFilter fields from a request's URL query params.
// Returns an error string if any parameter is invalid.
func ParseBarFilter(r *http.Request, symbol, timeframe string) (BarFilter, string) {
	q := r.URL.Query()
	bf := BarFilter{
		Symbol:    symbol,
		Timeframe: timeframe,
		Limit:     500,
		Ascending: true,
	}

	if v := q.Get("limit"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 1 {
			return bf, "limit must be a positive integer"
		}
		if n > 10000 {
			n = 10000
		}
		bf.Limit = n
	}

	if v := q.Get("order"); v != "" {
		switch strings.ToLower(v) {
		case "desc", "descending":
			bf.Ascending = false
		case "asc", "ascending":
			bf.Ascending = true
		default:
			return bf, "order must be asc or desc"
		}
	}

	if v := q.Get("from"); v != "" {
		t, err := parseFlexTime(v)
		if err != nil {
			return bf, "from: " + err.Error()
		}
		bf.From = t
	}

	if v := q.Get("to"); v != "" {
		t, err := parseFlexTime(v)
		if err != nil {
			return bf, "to: " + err.Error()
		}
		bf.To = t
	}

	if !bf.From.IsZero() && !bf.To.IsZero() && bf.To.Before(bf.From) {
		return bf, "to must be after from"
	}
	return bf, ""
}

// parseFlexTime parses a time string that may be RFC3339, Unix seconds (integer),
// or a date-only string in YYYY-MM-DD format.
func parseFlexTime(s string) (time.Time, error) {
	// Try Unix timestamp first.
	if n, err := strconv.ParseInt(s, 10, 64); err == nil {
		return time.Unix(n, 0).UTC(), nil
	}
	// Try RFC3339.
	if t, err := time.Parse(time.RFC3339, s); err == nil {
		return t, nil
	}
	// Try date only.
	if t, err := time.Parse("2006-01-02", s); err == nil {
		return t, nil
	}
	return time.Time{}, fmt.Errorf("unrecognised time format: %q", s)
}

// BarSorter sorts bars by timestamp.
type BarSorter []feed.Bar

func (s BarSorter) Len() int           { return len(s) }
func (s BarSorter) Less(i, j int) bool { return s[i].Timestamp.Before(s[j].Timestamp) }
func (s BarSorter) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// ApplyFilter applies limit and ordering to a sorted (ascending) bar slice.
func ApplyFilter(bars []feed.Bar, bf BarFilter) []feed.Bar {
	if len(bars) == 0 {
		return bars
	}

	// Apply time range.
	if !bf.From.IsZero() || !bf.To.IsZero() {
		var filtered []feed.Bar
		for _, b := range bars {
			if !bf.From.IsZero() && b.Timestamp.Before(bf.From) {
				continue
			}
			if !bf.To.IsZero() && b.Timestamp.After(bf.To) {
				continue
			}
			filtered = append(filtered, b)
		}
		bars = filtered
	}

	// Apply limit (take last N if ascending, first N if descending).
	if bf.Limit > 0 && len(bars) > bf.Limit {
		if bf.Ascending {
			bars = bars[len(bars)-bf.Limit:]
		} else {
			bars = bars[:bf.Limit]
		}
	}

	// Reverse if descending.
	if !bf.Ascending {
		for i, j := 0, len(bars)-1; i < j; i, j = i+1, j-1 {
			bars[i], bars[j] = bars[j], bars[i]
		}
	}

	return bars
}

// QuoteFilter holds parsed query parameters for quote queries.
type QuoteFilter struct {
	Symbols   []string
	StaleMax  time.Duration
	IncludeOB bool // include order book snapshot
}

// ParseQuoteFilter extracts QuoteFilter from request.
func ParseQuoteFilter(r *http.Request) QuoteFilter {
	q := r.URL.Query()
	qf := QuoteFilter{StaleMax: 60 * time.Second}

	if v := q.Get("symbols"); v != "" {
		for _, s := range strings.Split(v, ",") {
			s = strings.TrimSpace(strings.ToUpper(s))
			if s != "" {
				qf.Symbols = append(qf.Symbols, s)
			}
		}
	}
	if v := q.Get("stale_max"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			qf.StaleMax = d
		}
	}
	if q.Get("include_ob") == "true" || q.Get("include_ob") == "1" {
		qf.IncludeOB = true
	}
	return qf
}

// SymbolFilter parses and normalises a comma-separated symbols query parameter.
func SymbolFilter(r *http.Request, param string) []string {
	v := r.URL.Query().Get(param)
	if v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(strings.ToUpper(p))
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

