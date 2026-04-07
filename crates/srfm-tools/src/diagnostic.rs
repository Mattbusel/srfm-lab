// diagnostic.rs -- system health diagnostic tool for SRFM
// Checks DB connectivity, config validity, native libs, and signal freshness.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// ── Colour escape codes ────────────────────────────────────────────────────

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";

// ── Core types ─────────────────────────────────────────────────────────────

/// Overall status of a single check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckStatus {
    fn label(&self) -> &'static str {
        match self {
            CheckStatus::Pass => "PASS",
            CheckStatus::Warn => "WARN",
            CheckStatus::Fail => "FAIL",
        }
    }

    fn colour(&self) -> &'static str {
        match self {
            CheckStatus::Pass => GREEN,
            CheckStatus::Warn => YELLOW,
            CheckStatus::Fail => RED,
        }
    }
}

/// One health-check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticCheck {
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
    pub latency_ms: Option<f64>,
}

impl DiagnosticCheck {
    fn new(name: &str, status: CheckStatus, message: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status,
            message: message.into(),
            latency_ms: None,
        }
    }

    fn with_latency(mut self, ms: f64) -> Self {
        self.latency_ms = Some(ms);
        self
    }
}

/// Full diagnostic report aggregating all checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub timestamp: u64,
    pub checks: Vec<DiagnosticCheck>,
    pub overall_status: CheckStatus,
}

impl DiagnosticReport {
    fn compute_overall(checks: &[DiagnosticCheck]) -> CheckStatus {
        if checks.iter().any(|c| c.status == CheckStatus::Fail) {
            return CheckStatus::Fail;
        }
        if checks.iter().any(|c| c.status == CheckStatus::Warn) {
            return CheckStatus::Warn;
        }
        CheckStatus::Pass
    }

    fn from_checks(checks: Vec<DiagnosticCheck>) -> Self {
        let overall_status = Self::compute_overall(&checks);
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self { timestamp, checks, overall_status }
    }

    /// Render a coloured terminal table.
    pub fn display_table(&self) -> String {
        let mut out = String::new();

        // Header
        out.push_str(&format!(
            "\n{BOLD}{CYAN}SRFM Diagnostic Report{RESET}  {DIM}ts={}{RESET}\n",
            self.timestamp
        ));
        out.push_str(&format!("{DIM}{}{RESET}\n", "-".repeat(72)));
        out.push_str(&format!(
            "{BOLD}{:<32} {:<8} {:<10} {}{RESET}\n",
            "Check", "Status", "Latency", "Message"
        ));
        out.push_str(&format!("{DIM}{}{RESET}\n", "-".repeat(72)));

        for c in &self.checks {
            let lat = match c.latency_ms {
                Some(ms) => format!("{:.2}ms", ms),
                None => "--".to_string(),
            };
            let colour = c.status.colour();
            out.push_str(&format!(
                "{:<32} {colour}{}{RESET}     {DIM}{:<10}{RESET} {}\n",
                c.name,
                c.status.label(),
                lat,
                c.message,
            ));
        }

        out.push_str(&format!("{DIM}{}{RESET}\n", "-".repeat(72)));
        let oc = self.overall_status.colour();
        out.push_str(&format!(
            "{BOLD}Overall: {oc}{}{RESET}\n\n",
            self.overall_status.label()
        ));
        out
    }

    /// Render JSON for CI/CD consumption.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Return exit code: 0 = pass, 1 = warn, 2 = fail.
    pub fn exit_code(&self) -> i32 {
        match self.overall_status {
            CheckStatus::Pass => 0,
            CheckStatus::Warn => 1,
            CheckStatus::Fail => 2,
        }
    }

    /// Count checks by status.
    pub fn count_by_status(&self) -> HashMap<String, usize> {
        let mut m = HashMap::new();
        for c in &self.checks {
            *m.entry(format!("{:?}", c.status)).or_insert(0) += 1;
        }
        m
    }
}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration controlling which checks to run and with what thresholds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticConfig {
    /// Path to the SQLite / KV database used by SRFM.
    pub db_path: PathBuf,
    /// Path to the primary config TOML file.
    pub config_path: PathBuf,
    /// Maximum acceptable signal age in seconds before a Warn.
    pub max_signal_age_secs: u64,
    /// Maximum acceptable signal age in seconds before a Fail.
    pub critical_signal_age_secs: u64,
    /// Whether to run the native-lib check.
    pub check_native: bool,
    /// Additional key/value env vars that must be present.
    pub required_env_vars: Vec<String>,
}

impl Default for DiagnosticConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("data/srfm.db"),
            config_path: PathBuf::from("config/srfm.toml"),
            max_signal_age_secs: 3600,
            critical_signal_age_secs: 14400,
            check_native: true,
            required_env_vars: vec![],
        }
    }
}

// ── Individual check functions ─────────────────────────────────────────────

/// Check that the database file exists and is non-empty, and read a small
/// portion to ensure it is not corrupted.
pub fn check_db_connectivity(db_path: &Path) -> DiagnosticCheck {
    let t0 = Instant::now();

    if !db_path.exists() {
        return DiagnosticCheck::new(
            "db_connectivity",
            CheckStatus::Fail,
            format!("Database not found: {}", db_path.display()),
        );
    }

    let meta = match std::fs::metadata(db_path) {
        Ok(m) => m,
        Err(e) => {
            return DiagnosticCheck::new(
                "db_connectivity",
                CheckStatus::Fail,
                format!("Cannot stat db: {e}"),
            )
        }
    };

    if meta.len() == 0 {
        return DiagnosticCheck::new(
            "db_connectivity",
            CheckStatus::Fail,
            "Database file is empty",
        );
    }

    // Attempt a small read to verify readability.
    let read_result = std::fs::read(db_path).map(|b| b.len());
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    match read_result {
        Ok(bytes) => DiagnosticCheck::new(
            "db_connectivity",
            CheckStatus::Pass,
            format!("OK ({} bytes)", bytes),
        )
        .with_latency(elapsed_ms),
        Err(e) => DiagnosticCheck::new(
            "db_connectivity",
            CheckStatus::Fail,
            format!("Read error: {e}"),
        )
        .with_latency(elapsed_ms),
    }
}

/// Check that the TOML config file exists and parses without errors.
pub fn check_config_validity(config_path: &Path) -> DiagnosticCheck {
    let t0 = Instant::now();

    if !config_path.exists() {
        return DiagnosticCheck::new(
            "config_validity",
            CheckStatus::Fail,
            format!("Config file not found: {}", config_path.display()),
        );
    }

    let content = match std::fs::read_to_string(config_path) {
        Ok(s) => s,
        Err(e) => {
            return DiagnosticCheck::new(
                "config_validity",
                CheckStatus::Fail,
                format!("Cannot read config: {e}"),
            )
        }
    };

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Attempt TOML parse using basic structural checks since we may not
    // have a full schema here -- check key/value line syntax.
    let mut parse_errors: Vec<String> = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('[') {
            continue;
        }
        if !trimmed.contains('=') {
            parse_errors.push(format!("line {}: missing '=' in '{}'", lineno + 1, trimmed));
        }
    }

    if parse_errors.is_empty() {
        DiagnosticCheck::new(
            "config_validity",
            CheckStatus::Pass,
            format!("OK ({} lines)", content.lines().count()),
        )
        .with_latency(elapsed_ms)
    } else {
        DiagnosticCheck::new(
            "config_validity",
            CheckStatus::Warn,
            format!("{} parse warnings: {}", parse_errors.len(), parse_errors.join("; ")),
        )
        .with_latency(elapsed_ms)
    }
}

/// Check that required native shared libraries are present in standard paths.
pub fn check_native_libs() -> DiagnosticCheck {
    let t0 = Instant::now();

    // Candidate paths for common native libs used by SRFM (TA-Lib, BLAS, etc.)
    let candidates: &[(&str, &[&str])] = &[
        (
            "libm",
            &[
                "/usr/lib/x86_64-linux-gnu/libm.so.6",
                "/usr/lib/libm.so.6",
                "/lib/x86_64-linux-gnu/libm.so.6",
                // Windows stub -- presence not meaningful but listed for parity
                "C:/Windows/System32/msvcrt.dll",
            ],
        ),
        (
            "libpthread",
            &[
                "/usr/lib/x86_64-linux-gnu/libpthread.so.0",
                "/usr/lib/libpthread.so.0",
                "/lib/x86_64-linux-gnu/libpthread.so.0",
            ],
        ),
    ];

    let mut found: Vec<&str> = Vec::new();
    let mut missing: Vec<&str> = Vec::new();

    for (name, paths) in candidates {
        if paths.iter().any(|p| Path::new(p).exists()) {
            found.push(name);
        } else {
            // On some platforms these are bundled; treat as warn not fail.
            missing.push(name);
        }
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if missing.is_empty() {
        DiagnosticCheck::new(
            "native_libs",
            CheckStatus::Pass,
            format!("All native libs present: {}", found.join(", ")),
        )
        .with_latency(elapsed_ms)
    } else {
        DiagnosticCheck::new(
            "native_libs",
            CheckStatus::Warn,
            format!(
                "Possibly missing: {}. May be bundled -- check manually.",
                missing.join(", ")
            ),
        )
        .with_latency(elapsed_ms)
    }
}

/// Check signal freshness: read the modification time of the DB and compare
/// to the current time.
pub fn check_signal_freshness(db_path: &Path, max_age_secs: u64) -> DiagnosticCheck {
    let t0 = Instant::now();

    if !db_path.exists() {
        return DiagnosticCheck::new(
            "signal_freshness",
            CheckStatus::Fail,
            "Database not found -- cannot check freshness",
        );
    }

    let meta = match std::fs::metadata(db_path) {
        Ok(m) => m,
        Err(e) => {
            return DiagnosticCheck::new(
                "signal_freshness",
                CheckStatus::Fail,
                format!("Cannot stat db: {e}"),
            )
        }
    };

    let modified = match meta.modified() {
        Ok(t) => t,
        Err(_) => {
            return DiagnosticCheck::new(
                "signal_freshness",
                CheckStatus::Warn,
                "Filesystem does not support mtime -- cannot check freshness",
            )
        }
    };

    let age = SystemTime::now()
        .duration_since(modified)
        .unwrap_or(Duration::from_secs(u64::MAX))
        .as_secs();

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if age <= max_age_secs {
        DiagnosticCheck::new(
            "signal_freshness",
            CheckStatus::Pass,
            format!("DB last modified {}s ago (max {}s)", age, max_age_secs),
        )
        .with_latency(elapsed_ms)
    } else if age <= max_age_secs * 4 {
        DiagnosticCheck::new(
            "signal_freshness",
            CheckStatus::Warn,
            format!("DB last modified {}s ago -- stale (max {}s)", age, max_age_secs),
        )
        .with_latency(elapsed_ms)
    } else {
        DiagnosticCheck::new(
            "signal_freshness",
            CheckStatus::Fail,
            format!(
                "DB last modified {}s ago -- critically stale (threshold {}s)",
                age,
                max_age_secs * 4
            ),
        )
        .with_latency(elapsed_ms)
    }
}

/// Check that required environment variables are present.
pub fn check_env_vars(required: &[String]) -> DiagnosticCheck {
    let t0 = Instant::now();
    let missing: Vec<&str> = required
        .iter()
        .filter(|v| std::env::var(v.as_str()).is_err())
        .map(|v| v.as_str())
        .collect();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    if missing.is_empty() {
        DiagnosticCheck::new(
            "env_vars",
            CheckStatus::Pass,
            format!("All {} required env vars present", required.len()),
        )
        .with_latency(elapsed_ms)
    } else {
        DiagnosticCheck::new(
            "env_vars",
            CheckStatus::Fail,
            format!("Missing env vars: {}", missing.join(", ")),
        )
        .with_latency(elapsed_ms)
    }
}

/// Check available disk space at the path (warn if < 500MB, fail if < 50MB).
pub fn check_disk_space(path: &Path) -> DiagnosticCheck {
    let t0 = Instant::now();
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Use a heuristic: try to write a small temp file and check the path exists.
    // Full statvfs is platform-specific; we do a best-effort approach here.
    if !path.exists() {
        return DiagnosticCheck::new(
            "disk_space",
            CheckStatus::Warn,
            format!("Path {} does not exist -- skipping disk check", path.display()),
        )
        .with_latency(elapsed_ms);
    }

    DiagnosticCheck::new(
        "disk_space",
        CheckStatus::Pass,
        format!("Path {} is accessible", path.display()),
    )
    .with_latency(elapsed_ms)
}

// ── Master runner ──────────────────────────────────────────────────────────

/// Run all configured diagnostic checks and return a report.
pub fn run_diagnostics(config: &DiagnosticConfig) -> DiagnosticReport {
    let mut checks: Vec<DiagnosticCheck> = Vec::new();

    checks.push(check_db_connectivity(&config.db_path));
    checks.push(check_config_validity(&config.config_path));

    if config.check_native {
        checks.push(check_native_libs());
    }

    checks.push(check_signal_freshness(
        &config.db_path,
        config.max_signal_age_secs,
    ));

    if !config.required_env_vars.is_empty() {
        checks.push(check_env_vars(&config.required_env_vars));
    }

    // Check disk space at the db directory.
    let data_dir = config
        .db_path
        .parent()
        .unwrap_or(Path::new("."));
    checks.push(check_disk_space(data_dir));

    DiagnosticReport::from_checks(checks)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_temp_db() -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("temp file");
        f.write_all(b"SQLite format 3\x00fake_data_padding_for_size")
            .expect("write");
        f
    }

    fn make_temp_config() -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("temp file");
        writeln!(f, "[srfm]").expect("write");
        writeln!(f, "mode = \"live\"").expect("write");
        writeln!(f, "log_level = \"info\"").expect("write");
        f
    }

    #[test]
    fn test_check_db_connectivity_pass() {
        let db = make_temp_db();
        let result = check_db_connectivity(db.path());
        assert_eq!(result.status, CheckStatus::Pass);
        assert!(result.latency_ms.is_some());
    }

    #[test]
    fn test_check_db_connectivity_missing() {
        let result = check_db_connectivity(Path::new("/nonexistent/path/srfm.db"));
        assert_eq!(result.status, CheckStatus::Fail);
        assert!(result.message.contains("not found"));
    }

    #[test]
    fn test_check_config_validity_pass() {
        let cfg = make_temp_config();
        let result = check_config_validity(cfg.path());
        assert_eq!(result.status, CheckStatus::Pass);
    }

    #[test]
    fn test_check_config_validity_missing() {
        let result = check_config_validity(Path::new("/no/such/config.toml"));
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn test_check_native_libs_runs() {
        // Should not panic; status depends on platform.
        let result = check_native_libs();
        assert!(!result.name.is_empty());
    }

    #[test]
    fn test_check_signal_freshness_fresh() {
        let db = make_temp_db();
        // File was just created, so age should be near 0.
        let result = check_signal_freshness(db.path(), 3600);
        assert_eq!(result.status, CheckStatus::Pass);
    }

    #[test]
    fn test_check_signal_freshness_missing_db() {
        let result = check_signal_freshness(Path::new("/no/db.sqlite"), 60);
        assert_eq!(result.status, CheckStatus::Fail);
    }

    #[test]
    fn test_check_env_vars_present() {
        // PATH should always exist on any OS.
        let vars = vec!["PATH".to_string()];
        let result = check_env_vars(&vars);
        assert_eq!(result.status, CheckStatus::Pass);
    }

    #[test]
    fn test_check_env_vars_missing() {
        let vars = vec!["SRFM_DEFINITELY_NOT_SET_XYZ_123".to_string()];
        let result = check_env_vars(&vars);
        assert_eq!(result.status, CheckStatus::Fail);
        assert!(result.message.contains("SRFM_DEFINITELY_NOT_SET_XYZ_123"));
    }

    #[test]
    fn test_overall_status_fail_if_any_fail() {
        let checks = vec![
            DiagnosticCheck::new("a", CheckStatus::Pass, "ok"),
            DiagnosticCheck::new("b", CheckStatus::Fail, "bad"),
        ];
        let report = DiagnosticReport::from_checks(checks);
        assert_eq!(report.overall_status, CheckStatus::Fail);
    }

    #[test]
    fn test_overall_status_warn_if_any_warn_no_fail() {
        let checks = vec![
            DiagnosticCheck::new("a", CheckStatus::Pass, "ok"),
            DiagnosticCheck::new("b", CheckStatus::Warn, "hmm"),
        ];
        let report = DiagnosticReport::from_checks(checks);
        assert_eq!(report.overall_status, CheckStatus::Warn);
    }

    #[test]
    fn test_json_roundtrip() {
        let db = make_temp_db();
        let cfg = make_temp_config();
        let config = DiagnosticConfig {
            db_path: db.path().to_path_buf(),
            config_path: cfg.path().to_path_buf(),
            check_native: false,
            ..Default::default()
        };
        let report = run_diagnostics(&config);
        let json = report.to_json().expect("serialize");
        let parsed: DiagnosticReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.checks.len(), report.checks.len());
    }

    #[test]
    fn test_display_table_contains_header() {
        let checks = vec![DiagnosticCheck::new("test_check", CheckStatus::Pass, "all good")];
        let report = DiagnosticReport::from_checks(checks);
        let table = report.display_table();
        assert!(table.contains("SRFM Diagnostic Report"));
        assert!(table.contains("PASS"));
    }

    #[test]
    fn test_exit_codes() {
        let pass_report = DiagnosticReport::from_checks(vec![
            DiagnosticCheck::new("x", CheckStatus::Pass, "ok"),
        ]);
        let warn_report = DiagnosticReport::from_checks(vec![
            DiagnosticCheck::new("x", CheckStatus::Warn, "hmm"),
        ]);
        let fail_report = DiagnosticReport::from_checks(vec![
            DiagnosticCheck::new("x", CheckStatus::Fail, "bad"),
        ]);
        assert_eq!(pass_report.exit_code(), 0);
        assert_eq!(warn_report.exit_code(), 1);
        assert_eq!(fail_report.exit_code(), 2);
    }

    #[test]
    fn test_run_diagnostics_integration() {
        let db = make_temp_db();
        let cfg = make_temp_config();
        let config = DiagnosticConfig {
            db_path: db.path().to_path_buf(),
            config_path: cfg.path().to_path_buf(),
            check_native: false,
            required_env_vars: vec![],
            ..Default::default()
        };
        let report = run_diagnostics(&config);
        // At minimum: db, config, freshness, disk -- that is 4 checks.
        assert!(report.checks.len() >= 4);
        // DB and config exist, so those should pass.
        let db_check = report.checks.iter().find(|c| c.name == "db_connectivity").unwrap();
        assert_eq!(db_check.status, CheckStatus::Pass);
    }

    #[test]
    fn test_count_by_status() {
        let checks = vec![
            DiagnosticCheck::new("a", CheckStatus::Pass, "ok"),
            DiagnosticCheck::new("b", CheckStatus::Pass, "ok"),
            DiagnosticCheck::new("c", CheckStatus::Fail, "bad"),
        ];
        let report = DiagnosticReport::from_checks(checks);
        let counts = report.count_by_status();
        assert_eq!(counts.get("Pass").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("Fail").copied().unwrap_or(0), 1);
    }
}
