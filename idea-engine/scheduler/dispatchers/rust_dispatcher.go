// Package dispatchers implements experiment runners for the scheduler.
package dispatchers

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"go.uber.org/zap"
)

// RustDispatcher runs experiments implemented as a compiled Rust binary.
// It writes a JSON config file to a temp directory, launches the binary as a
// subprocess, waits for completion, and reads the JSON result from another
// temp file.
//
// Contract with the Rust binary:
//
//	idea-genome-engine --config <path> --output <path>
//
// The config file contains the full experiment configuration JSON.
// The output file contains the result JSON on success; the binary must exit 0
// on success and non-zero on failure.
type RustDispatcher struct {
	// BinaryPath is the absolute path to the Rust binary.
	BinaryPath string
	// Timeout is the maximum time to wait for the binary to complete.
	Timeout time.Duration
	// TempDir is the directory used for config/output temp files.
	TempDir string
	log     *zap.Logger
}

// ExperimentConfig is the JSON written to the config temp file.
type ExperimentConfig struct {
	ExperimentID   string          `json:"experiment_id"`
	HypothesisID   string          `json:"hypothesis_id"`
	ExperimentType string          `json:"experiment_type"`
	Config         json.RawMessage `json:"config"`
}

// NewRustDispatcher constructs a RustDispatcher.
// If timeout is zero it defaults to 2 hours (genome evaluations can be long).
func NewRustDispatcher(binaryPath string, timeout time.Duration, tempDir string, log *zap.Logger) *RustDispatcher {
	if timeout <= 0 {
		timeout = 2 * time.Hour
	}
	if tempDir == "" {
		tempDir = os.TempDir()
	}
	return &RustDispatcher{
		BinaryPath: binaryPath,
		Timeout:    timeout,
		TempDir:    tempDir,
		log:        log,
	}
}

// Dispatch satisfies the scheduler.DispatchFunc signature.
// It runs the Rust binary synchronously and returns the JSON result.
func (d *RustDispatcher) Dispatch(ctx context.Context, item interface{ GetExperimentID() string; GetHypothesisID() string; GetExperimentType() string; GetConfig() []byte }) (json.RawMessage, error) {
	expID := item.GetExperimentID()

	// Write config to a temp file.
	cfg := ExperimentConfig{
		ExperimentID:   expID,
		HypothesisID:   item.GetHypothesisID(),
		ExperimentType: item.GetExperimentType(),
		Config:         json.RawMessage(item.GetConfig()),
	}
	cfgBytes, err := json.Marshal(cfg)
	if err != nil {
		return nil, fmt.Errorf("rust_dispatcher: marshal config: %w", err)
	}

	cfgFile, err := os.CreateTemp(d.TempDir, "genome-cfg-*.json")
	if err != nil {
		return nil, fmt.Errorf("rust_dispatcher: create config temp file: %w", err)
	}
	defer os.Remove(cfgFile.Name())
	if _, err := cfgFile.Write(cfgBytes); err != nil {
		cfgFile.Close()
		return nil, fmt.Errorf("rust_dispatcher: write config: %w", err)
	}
	cfgFile.Close()

	// Create output temp file path (do not create yet; the binary will write it).
	outPath := filepath.Join(d.TempDir, fmt.Sprintf("genome-out-%s-%d.json", expID, time.Now().UnixNano()))
	defer os.Remove(outPath)

	// Build the subprocess context with timeout.
	runCtx, cancel := context.WithTimeout(ctx, d.Timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, d.BinaryPath, "--config", cfgFile.Name(), "--output", outPath)
	cmd.Stdout = os.Stdout // pipe binary stdout to our stdout for logging
	cmd.Stderr = os.Stderr

	d.log.Info("rust_dispatcher: launching binary",
		zap.String("binary", d.BinaryPath),
		zap.String("experiment_id", expID),
		zap.String("config", cfgFile.Name()),
		zap.String("output", outPath),
	)

	start := time.Now()
	if err := cmd.Run(); err != nil {
		if runCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("rust_dispatcher: timed out after %s", d.Timeout)
		}
		return nil, fmt.Errorf("rust_dispatcher: binary exited with error: %w", err)
	}

	d.log.Info("rust_dispatcher: binary completed",
		zap.String("experiment_id", expID),
		zap.Duration("elapsed", time.Since(start)),
	)

	// Read the output file.
	resultBytes, err := os.ReadFile(outPath)
	if err != nil {
		return nil, fmt.Errorf("rust_dispatcher: read output file %q: %w", outPath, err)
	}

	// Validate JSON.
	var raw json.RawMessage = resultBytes
	if !json.Valid(resultBytes) {
		return nil, fmt.Errorf("rust_dispatcher: output is not valid JSON")
	}

	return raw, nil
}

// DispatchItem is an adapter that wraps a raw struct to satisfy the interface
// required by Dispatch. Used when calling Dispatch directly with named fields.
type DispatchItem struct {
	ExperimentID   string
	HypothesisID   string
	ExperimentType string
	Config         []byte
}

func (i *DispatchItem) GetExperimentID() string   { return i.ExperimentID }
func (i *DispatchItem) GetHypothesisID() string   { return i.HypothesisID }
func (i *DispatchItem) GetExperimentType() string { return i.ExperimentType }
func (i *DispatchItem) GetConfig() []byte         { return i.Config }

// RunnerFunc returns a function with the scheduler.DispatchFunc signature
// that can be registered with ExperimentLifecycle.RegisterDispatcher.
func (d *RustDispatcher) RunnerFunc() func(ctx context.Context, item *DispatchItem) (json.RawMessage, error) {
	return func(ctx context.Context, item *DispatchItem) (json.RawMessage, error) {
		return d.Dispatch(ctx, item)
	}
}
