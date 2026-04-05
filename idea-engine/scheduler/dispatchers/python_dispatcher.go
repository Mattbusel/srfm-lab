package dispatchers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"go.uber.org/zap"
)

// PythonDispatcher runs experiments implemented as Python modules.
// It launches a subprocess:
//
//	python -m idea_engine.<module> --experiment-id <id> [--config-json <json>]
//
// The module writes a JSON result object to stdout; anything else goes to
// stderr and is captured for the error log.
//
// A configurable timeout prevents runaway Python processes from tying up
// budget slots indefinitely.
type PythonDispatcher struct {
	// PythonBin is the path to the Python interpreter (default: "python").
	PythonBin string
	// ModulePrefix is the Python package prefix, e.g. "idea_engine".
	ModulePrefix string
	// Timeout is the maximum duration allowed for a Python experiment.
	Timeout time.Duration
	// Env is extra environment variables injected into the subprocess.
	Env []string
	log *zap.Logger
}

// moduleForType maps an experiment type to the Python sub-module name.
var moduleForType = map[string]string{
	"counterfactual": "counterfactual.runner",
	"shadow":         "shadow_runner.runner",
	"causal":         "causal.runner",
	"academic":       "academic_miner.runner",
	"serendipity":    "serendipity.runner",
	"hypothesis":     "hypothesis.generator",
}

// NewPythonDispatcher constructs a PythonDispatcher.
// If timeout is zero it defaults to 30 minutes.
func NewPythonDispatcher(pythonBin, modulePrefix string, timeout time.Duration, log *zap.Logger) *PythonDispatcher {
	if pythonBin == "" {
		pythonBin = "python"
	}
	if modulePrefix == "" {
		modulePrefix = "idea_engine"
	}
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	return &PythonDispatcher{
		PythonBin:    pythonBin,
		ModulePrefix: modulePrefix,
		Timeout:      timeout,
		log:          log,
	}
}

// Dispatch runs the Python experiment identified by item.ExperimentType.
// Returns the parsed JSON result or an error.
func (d *PythonDispatcher) Dispatch(ctx context.Context, item *DispatchItem) (json.RawMessage, error) {
	subModule, ok := moduleForType[item.ExperimentType]
	if !ok {
		return nil, fmt.Errorf("python_dispatcher: no module mapped for type %q", item.ExperimentType)
	}

	modulePath := d.ModulePrefix + "." + subModule

	// Encode config as an inline JSON string argument.
	configArg := "{}"
	if len(item.Config) > 0 {
		configArg = string(item.Config)
	}

	args := []string{
		"-m", modulePath,
		"--experiment-id", item.ExperimentID,
		"--hypothesis-id", item.HypothesisID,
		"--config-json", configArg,
	}

	runCtx, cancel := context.WithTimeout(ctx, d.Timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, d.PythonBin, args...)

	// Capture stdout (result JSON) and stderr (logs).
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Inject any extra environment variables.
	if len(d.Env) > 0 {
		cmd.Env = append(os.Environ(), d.Env...)
	}

	d.log.Info("python_dispatcher: launching module",
		zap.String("module", modulePath),
		zap.String("experiment_id", item.ExperimentID),
		zap.String("type", item.ExperimentType),
	)

	start := time.Now()
	runErr := cmd.Run()
	elapsed := time.Since(start)

	// Log stderr regardless of success/failure for debugging.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		d.log.Debug("python_dispatcher: stderr",
			zap.String("experiment_id", item.ExperimentID),
			zap.String("module", modulePath),
			zap.String("stderr", stderrStr),
		)
	}

	if runErr != nil {
		if runCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("python_dispatcher: timed out after %s (module: %s)", d.Timeout, modulePath)
		}
		stderrMsg := strings.TrimSpace(stderr.String())
		if stderrMsg == "" {
			stderrMsg = runErr.Error()
		}
		return nil, fmt.Errorf("python_dispatcher: module %s failed: %s", modulePath, stderrMsg)
	}

	d.log.Info("python_dispatcher: module completed",
		zap.String("experiment_id", item.ExperimentID),
		zap.String("module", modulePath),
		zap.Duration("elapsed", elapsed),
	)

	// Parse stdout as JSON result.
	output := bytes.TrimSpace(stdout.Bytes())
	if len(output) == 0 {
		// Empty stdout: return a minimal success marker.
		return json.RawMessage(`{"status":"done"}`), nil
	}
	if !json.Valid(output) {
		return nil, fmt.Errorf("python_dispatcher: stdout is not valid JSON: %q", string(output))
	}
	return json.RawMessage(output), nil
}

// AddEnv adds an environment variable (KEY=VALUE) injected into all Python
// subprocesses launched by this dispatcher.
func (d *PythonDispatcher) AddEnv(kv string) {
	d.Env = append(d.Env, kv)
}
