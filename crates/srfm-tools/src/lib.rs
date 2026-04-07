// lib.rs -- srfm-tools library root
// Exposes diagnostic, profiling, config validation, and benchmark utilities.

pub mod benchmark_runner;
pub mod config_validator;
pub mod diagnostic;
pub mod profiler;

pub use benchmark_runner::{
    run_benchmark, run_suite, BenchmarkResult, BenchmarkSuite, Benchmark,
    to_markdown_table, regressions_to_markdown, check_regressions,
    load_baseline, save_baseline,
};
pub use config_validator::{
    validate_all_configs, validate_event_calendar, validate_generic_json,
    validate_param_schema, validate_toml_config, ValidationError, ValidationResult,
};
pub use diagnostic::{
    check_config_validity, check_db_connectivity, check_env_vars,
    check_native_libs, check_signal_freshness, run_diagnostics,
    CheckStatus, DiagnosticCheck, DiagnosticConfig, DiagnosticReport,
};
pub use profiler::{
    global_record, global_reset, global_summary, ExecutionProfiler,
    LabelSummary, ProfileSample, ProfileSummary, ScopedTimer,
};
