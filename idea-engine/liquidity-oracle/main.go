// Command liquidity-oracle monitors real-time liquidity conditions and
// suppresses entries when liquidity is too thin.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"srfm-lab/idea-engine/liquidity-oracle/alerts"
	"srfm-lab/idea-engine/liquidity-oracle/historical"
	"srfm-lab/idea-engine/liquidity-oracle/monitors"
	"srfm-lab/idea-engine/liquidity-oracle/scorer"
	"srfm-lab/idea-engine/liquidity-oracle/store"
)

// trackedSymbols are the Binance symbols monitored by all subsystems.
var trackedSymbols = []string{
	"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
	"ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
}

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(log)

	dbPath := envOrDefault("DB_PATH", "liquidity_oracle.db")
	busURL := envOrDefault("BUS_URL", "http://localhost:8768")
	listenAddr := envOrDefault("LISTEN_ADDR", ":8772")

	db, err := store.Open(dbPath)
	if err != nil {
		log.Error("failed to open store", "err", err)
		os.Exit(1)
	}
	defer db.Close()

	spreadAlerts := make(chan monitors.SpreadAlert, 128)
	spreadMon := monitors.NewSpreadMonitor(log, spreadAlerts)
	depthMon := monitors.NewDepthMonitor(trackedSymbols, log)
	volumeMon := monitors.NewVolumeMonitor(trackedSymbols, log)
	tradeRateMon := monitors.NewTradeRateMonitor(trackedSymbols, log)
	sc := scorer.New(spreadMon, depthMon, volumeMon, tradeRateMon)
	profiles := historical.NewProfileStore()
	alertMgr := alerts.New(busURL, log)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	go spreadMon.Run(ctx)
	go depthMon.Run(ctx)
	go volumeMon.Run(ctx)
	go tradeRateMon.Run(ctx)

	// Background scoring loop.
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				for _, sym := range trackedSymbols {
					score := sc.Score(sym)
					if err := db.InsertScore(score); err != nil {
						log.Warn("persist score", "symbol", sym, "err", err)
					}
					// Record into hourly profiles.
					if obs, ok := spreadMon.Current(sym); ok {
						if depth, dok := depthMon.Current(sym); dok {
							volSnap, _ := volumeMon.Current(sym)
							trSnap, _ := tradeRateMon.Current(sym)
							profiles.Record(sym, obs.EffectiveSpread, depth.TotalDepth,
								volSnap.CurrentVolume, trSnap.TradesPerMinute)
						}
					}
					alertMgr.MaybeAlert(ctx, score)
				}
			}
		}
	}()

	// Drain spread alerts channel.
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case alert := <-spreadAlerts:
				log.Info("spread alert",
					"symbol", alert.Symbol,
					"spread", alert.CurrentSpread,
					"avg", alert.HourlyAverage,
					"ratio", alert.Ratio,
				)
			}
		}
	}()

	mux := http.NewServeMux()

	mux.HandleFunc("GET /liquidity/{symbol}", func(w http.ResponseWriter, r *http.Request) {
		symbol := strings.ToUpper(r.PathValue("symbol"))
		score := sc.Score(symbol)
		writeJSON(w, score)
	})

	mux.HandleFunc("GET /liquidity/all", func(w http.ResponseWriter, r *http.Request) {
		scores := sc.ScoreAll(trackedSymbols)
		writeJSON(w, scores)
	})

	mux.HandleFunc("GET /alert", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, alertMgr.ActiveAlerts())
	})

	mux.HandleFunc("POST /subscribe", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			URL string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.URL == "" {
			http.Error(w, "body must contain {\"url\":\"...\"}", http.StatusBadRequest)
			return
		}
		alertMgr.AddSubscriber(alerts.Subscriber{URL: req.URL})
		writeJSON(w, map[string]string{"status": "subscribed", "url": req.URL})
	})

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]string{"status": "ok"})
	})

	srv := &http.Server{
		Addr:         listenAddr,
		Handler:      mux,
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		log.Info("liquidity-oracle: listening", "addr", listenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", "err", err)
			cancel()
		}
	}()

	<-ctx.Done()
	log.Info("liquidity-oracle: shutting down")
	shutCtx, shutCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutCancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		log.Error("shutdown error", "err", err)
	}
}

func writeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(v); err != nil {
		http.Error(w, "encode error", http.StatusInternalServerError)
	}
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
