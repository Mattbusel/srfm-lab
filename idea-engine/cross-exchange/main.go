// Command cross-exchange polls multiple crypto exchanges simultaneously,
// detects price divergences, computes futures/spot basis, and publishes
// quality signals to the IAE event bus.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"srfm-lab/idea-engine/cross-exchange/aggregator"
	"srfm-lab/idea-engine/cross-exchange/basis"
	"srfm-lab/idea-engine/cross-exchange/divergence"
	"srfm-lab/idea-engine/cross-exchange/exchanges"
	"srfm-lab/idea-engine/cross-exchange/publisher"
	"srfm-lab/idea-engine/cross-exchange/store"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(log)

	dbPath := envOrDefault("DB_PATH", "cross_exchange.db")
	busURL := envOrDefault("BUS_URL", "http://localhost:8768")
	listenAddr := envOrDefault("LISTEN_ADDR", ":8771")

	db, err := store.Open(dbPath)
	if err != nil {
		log.Error("failed to open store", "err", err)
		os.Exit(1)
	}
	defer db.Close()

	agg := aggregator.New(60 * time.Second)
	det := divergence.New()
	pub := publisher.New(busURL, log)

	binancePoller := exchanges.NewBinancePoller(agg, log)
	coinbasePoller := exchanges.NewCoinbasePoller(agg, log)
	krakenPoller := exchanges.NewKrakenPoller(agg, log)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	go binancePoller.Run(ctx)
	go coinbasePoller.Run(ctx)
	go krakenPoller.Run(ctx)

	// Background loop: detect divergences, compute basis, persist, and publish.
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				consensus := agg.Consensus()
				signals := det.Detect(consensus)

				for _, sig := range signals {
					if err := db.InsertDivergenceSignal(sig); err != nil {
						log.Warn("persist divergence signal", "err", err)
					}
					if sig.Zscore >= 3.0 {
						if err := pub.PublishDivergence(ctx, sig); err != nil {
							log.Warn("publish divergence signal", "err", err)
						}
					}
				}

				// Compute basis from Binance futures/spot.
				fp := binancePoller.FuturesPrices()
				fr := binancePoller.FundingRates()
				spotPrices := make(map[string]float64, len(consensus))
				for _, cp := range consensus {
					spotPrices[cp.Symbol] = cp.ConsensusMid
				}
				futPrices := make(map[string]float64, len(fp))
				for sym, f := range fp {
					futPrices[sym] = f.Price
				}
				fundingMap := make(map[string]float64, len(fr))
				for sym, f := range fr {
					fundingMap[sym] = f.FundingRate
				}
				basisSignals := basis.ComputeAllBasis(spotPrices, futPrices, fundingMap)
				for _, sig := range basisSignals {
					if err := pub.PublishBasis(ctx, sig); err != nil {
						log.Warn("publish basis signal", "err", err)
					}
				}
			}
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /prices", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, agg.AllPrices())
	})
	mux.HandleFunc("GET /divergence", func(w http.ResponseWriter, r *http.Request) {
		consensus := agg.Consensus()
		signals := det.Detect(consensus)
		writeJSON(w, signals)
	})
	mux.HandleFunc("GET /basis", func(w http.ResponseWriter, r *http.Request) {
		fp := binancePoller.FuturesPrices()
		fr := binancePoller.FundingRates()
		consensus := agg.Consensus()
		spotPrices := make(map[string]float64, len(consensus))
		for _, cp := range consensus {
			spotPrices[cp.Symbol] = cp.ConsensusMid
		}
		futPrices := make(map[string]float64, len(fp))
		for sym, f := range fp {
			futPrices[sym] = f.Price
		}
		fundingMap := make(map[string]float64, len(fr))
		for sym, f := range fr {
			fundingMap[sym] = f.FundingRate
		}
		signals := basis.ComputeAllBasis(spotPrices, futPrices, fundingMap)
		writeJSON(w, signals)
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
		log.Info("cross-exchange: listening", "addr", listenAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error("server error", "err", err)
			cancel()
		}
	}()

	<-ctx.Done()
	log.Info("cross-exchange: shutting down")
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
