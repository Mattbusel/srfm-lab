// config_validator.rs -- validates SRFM config JSON/TOML files
// Checks param schemas, event calendars, and TOML configs for correctness.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ── Error / warning types ──────────────────────────────────────────────────

/// A structured error found during validation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationError {
    /// Dot-separated path to the offending field (e.g. "params.cf.min").
    pub field_path: String,
    /// Human-readable description of the problem.
    pub message: String,
    /// Expected type or format string (e.g. "f64", "ISO-8601 date").
    pub expected_type: String,
}

impl ValidationError {
    fn new(
        field_path: impl Into<String>,
        message: impl Into<String>,
        expected_type: impl Into<String>,
    ) -> Self {
        Self {
            field_path: field_path.into(),
            message: message.into(),
            expected_type: expected_type.into(),
        }
    }
}

/// Result of validating one file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub file_path: PathBuf,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    fn new(file_path: PathBuf) -> Self {
        Self { file_path, errors: vec![], warnings: vec![] }
    }

    /// True if no errors were found (warnings allowed).
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    fn push_error(
        &mut self,
        field: impl Into<String>,
        msg: impl Into<String>,
        expected: impl Into<String>,
    ) {
        self.errors.push(ValidationError::new(field, msg, expected));
    }

    fn push_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

// ── param_schema.json validation ───────────────────────────────────────────

/// A single parameter definition as it appears in param_schema.json.
#[derive(Debug, Clone, Deserialize)]
pub struct ParamSchemaEntry {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub default: Option<f64>,
    pub description: Option<String>,
    #[serde(rename = "type")]
    pub param_type: Option<String>,
}

/// Validate param_schema.json: all ranges valid, defaults in range.
pub fn validate_param_schema(path: &Path) -> ValidationResult {
    let mut result = ValidationResult::new(path.to_path_buf());

    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            result.push_error("", format!("Cannot read file: {e}"), "readable file");
            return result;
        }
    };

    let schema: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            result.push_error("", format!("JSON parse error: {e}"), "valid JSON");
            return result;
        }
    };

    let obj = match schema.as_object() {
        Some(o) => o,
        None => {
            result.push_error("root", "param_schema.json must be a JSON object", "object");
            return result;
        }
    };

    for (param_name, entry_val) in obj {
        // Try to deserialize each entry.
        let entry: ParamSchemaEntry = match serde_json::from_value(entry_val.clone()) {
            Ok(e) => e,
            Err(e) => {
                result.push_error(
                    param_name,
                    format!("Cannot parse param entry: {e}"),
                    "ParamSchemaEntry",
                );
                continue;
            }
        };

        // Check min < max if both are present.
        if let (Some(min), Some(max)) = (entry.min, entry.max) {
            if min >= max {
                result.push_error(
                    format!("{param_name}.range"),
                    format!("min ({min}) must be less than max ({max})"),
                    "min < max",
                );
            }
        }

        // Check default is within [min, max].
        if let Some(default) = entry.default {
            if let Some(min) = entry.min {
                if default < min {
                    result.push_error(
                        format!("{param_name}.default"),
                        format!("default ({default}) is below min ({min})"),
                        "default >= min",
                    );
                }
            }
            if let Some(max) = entry.max {
                if default > max {
                    result.push_error(
                        format!("{param_name}.default"),
                        format!("default ({default}) exceeds max ({max})"),
                        "default <= max",
                    );
                }
            }
        } else {
            result.push_warning(format!("{param_name}: no default value specified"));
        }

        // Warn if no description.
        if entry.description.as_deref().unwrap_or("").trim().is_empty() {
            result.push_warning(format!("{param_name}: missing description"));
        }
    }

    result
}

// ── event_calendar.json validation ─────────────────────────────────────────

const DATE_PATTERN_LEN: usize = 10; // "YYYY-MM-DD"

fn is_valid_iso_date(s: &str) -> bool {
    if s.len() != DATE_PATTERN_LEN {
        return false;
    }
    let bytes = s.as_bytes();
    // Check format: YYYY-MM-DD
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let year: u32 = s[0..4].parse().unwrap_or(0);
    let month: u32 = s[5..7].parse().unwrap_or(0);
    let day: u32 = s[8..10].parse().unwrap_or(0);
    year >= 1900 && year <= 2200 && month >= 1 && month <= 12 && day >= 1 && day <= 31
}

/// Validate event_calendar.json: check date formats, no duplicate dates.
pub fn validate_event_calendar(path: &Path) -> ValidationResult {
    let mut result = ValidationResult::new(path.to_path_buf());

    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            result.push_error("", format!("Cannot read file: {e}"), "readable file");
            return result;
        }
    };

    let calendar: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            result.push_error("", format!("JSON parse error: {e}"), "valid JSON");
            return result;
        }
    };

    // Accept either an array of event objects or an object keyed by date.
    let events: Vec<serde_json::Value> = if let Some(arr) = calendar.as_array() {
        arr.clone()
    } else if let Some(obj) = calendar.as_object() {
        // Object keyed by date -> treat each value as an event entry.
        obj.iter()
            .map(|(k, v)| {
                let mut m = serde_json::Map::new();
                m.insert("date".to_string(), serde_json::Value::String(k.clone()));
                m.insert("event".to_string(), v.clone());
                serde_json::Value::Object(m)
            })
            .collect()
    } else {
        result.push_error("root", "event_calendar.json must be array or object", "array|object");
        return result;
    };

    let mut seen_dates: HashSet<String> = HashSet::new();

    for (idx, event) in events.iter().enumerate() {
        let prefix = format!("events[{idx}]");
        let obj = match event.as_object() {
            Some(o) => o,
            None => {
                result.push_error(format!("{prefix}"), "each event must be an object", "object");
                continue;
            }
        };

        // Validate date field.
        match obj.get("date") {
            None => {
                result.push_error(format!("{prefix}.date"), "missing 'date' field", "ISO-8601 date");
            }
            Some(date_val) => {
                let date_str = match date_val.as_str() {
                    Some(s) => s,
                    None => {
                        result.push_error(
                            format!("{prefix}.date"),
                            "date field must be a string",
                            "ISO-8601 date",
                        );
                        continue;
                    }
                };

                if !is_valid_iso_date(date_str) {
                    result.push_error(
                        format!("{prefix}.date"),
                        format!("invalid date format: '{date_str}'"),
                        "YYYY-MM-DD",
                    );
                } else if !seen_dates.insert(date_str.to_string()) {
                    result.push_error(
                        format!("{prefix}.date"),
                        format!("duplicate date: '{date_str}'"),
                        "unique YYYY-MM-DD",
                    );
                }
            }
        }

        // Warn if event has no name/title/description.
        let has_label = obj.contains_key("name")
            || obj.contains_key("title")
            || obj.contains_key("description")
            || obj.contains_key("event");
        if !has_label {
            result.push_warning(format!("{prefix}: no name/title/description/event field"));
        }
    }

    result
}

// ── TOML config validation ─────────────────────────────────────────────────

/// Required top-level sections that must appear in an SRFM TOML config.
const REQUIRED_TOML_SECTIONS: &[&str] = &["[srfm]", "[risk]", "[execution]"];

/// Required key=value fields expected in a complete SRFM TOML config.
/// Format: (section_prefix, field_name)
const REQUIRED_TOML_FIELDS: &[(&str, &str)] = &[
    ("srfm", "mode"),
    ("srfm", "log_level"),
    ("risk", "max_drawdown"),
    ("execution", "exchange"),
];

/// Validate a single TOML config file.
pub fn validate_toml_config(path: &Path) -> ValidationResult {
    let mut result = ValidationResult::new(path.to_path_buf());

    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            result.push_error("", format!("Cannot read file: {e}"), "readable file");
            return result;
        }
    };

    // Check required sections are present.
    for section in REQUIRED_TOML_SECTIONS {
        if !content.contains(section) {
            result.push_warning(format!("Missing expected section: {section}"));
        }
    }

    // Parse into a simple key=value map, tracking current section.
    let mut current_section = String::new();
    let mut kv_map: HashMap<String, String> = HashMap::new();
    let mut line_count = 0usize;

    for (lineno, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        line_count += 1;

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len() - 1].trim().to_string();
            continue;
        }

        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim().to_string();
            let val = line[eq_pos + 1..].trim().to_string();
            if key.is_empty() {
                result.push_error(
                    format!("line_{}", lineno + 1),
                    "empty key before '='",
                    "non-empty identifier",
                );
                continue;
            }
            let full_key = if current_section.is_empty() {
                key.clone()
            } else {
                format!("{}.{}", current_section, key)
            };
            kv_map.insert(full_key, val);
        } else {
            result.push_warning(format!(
                "line {}: unparseable line (no '='): '{}'",
                lineno + 1,
                line
            ));
        }
    }

    // Check required fields.
    for (section, field) in REQUIRED_TOML_FIELDS {
        let key = format!("{section}.{field}");
        if !kv_map.contains_key(&key) {
            result.push_warning(format!("Expected field not found: {key}"));
        }
    }

    if line_count == 0 {
        result.push_error("root", "Config file is empty", "non-empty TOML");
    }

    result
}

// ── Validate all configs in a directory ────────────────────────────────────

/// Validate all supported config files found in `config_dir`.
/// Recurses one level into subdirectories.
pub fn validate_all_configs(config_dir: &Path) -> Vec<ValidationResult> {
    let mut results: Vec<ValidationResult> = Vec::new();

    let entries = match std::fs::read_dir(config_dir) {
        Ok(e) => e,
        Err(e) => {
            let mut r = ValidationResult::new(config_dir.to_path_buf());
            r.push_error("", format!("Cannot read config dir: {e}"), "readable directory");
            return vec![r];
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        // Handle subdirectories (one level deep).
        if path.is_dir() {
            let sub_results = validate_all_configs(&path);
            results.extend(sub_results);
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        let result = match ext.as_str() {
            "toml" => validate_toml_config(&path),
            "json" if file_name.contains("param_schema") => validate_param_schema(&path),
            "json" if file_name.contains("event_calendar") => validate_event_calendar(&path),
            "json" => {
                // Generic JSON: just check it parses.
                validate_generic_json(&path)
            }
            _ => continue,
        };

        results.push(result);
    }

    results
}

/// Validate any JSON file: just check it parses and is non-empty.
pub fn validate_generic_json(path: &Path) -> ValidationResult {
    let mut result = ValidationResult::new(path.to_path_buf());

    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            result.push_error("", format!("Cannot read file: {e}"), "readable file");
            return result;
        }
    };

    if content.trim().is_empty() {
        result.push_error("root", "JSON file is empty", "non-empty JSON");
        return result;
    }

    if let Err(e) = serde_json::from_str::<serde_json::Value>(&content) {
        result.push_error("root", format!("JSON parse error: {e}"), "valid JSON");
    }

    result
}

// ── Summary helpers ────────────────────────────────────────────────────────

/// Print a summary of all validation results to stderr.
pub fn print_validation_summary(results: &[ValidationResult]) {
    let total = results.len();
    let passing = results.iter().filter(|r| r.is_valid()).count();
    let failing = total - passing;
    let total_errors: usize = results.iter().map(|r| r.errors.len()).sum();
    let total_warnings: usize = results.iter().map(|r| r.warnings.len()).sum();

    eprintln!("\n=== Config Validation Summary ===");
    eprintln!("Files checked:  {total}");
    eprintln!("Passing:        {passing}");
    eprintln!("Failing:        {failing}");
    eprintln!("Total errors:   {total_errors}");
    eprintln!("Total warnings: {total_warnings}");

    for r in results {
        if !r.is_valid() {
            eprintln!("\nFAIL: {}", r.file_path.display());
            for e in &r.errors {
                eprintln!("  ERROR [{}]: {} (expected: {})", e.field_path, e.message, e.expected_type);
            }
        }
        for w in &r.warnings {
            eprintln!("  WARN: {}", w);
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    fn write_temp_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("temp file");
        write!(f, "{}", content).expect("write");
        f
    }

    fn write_temp_toml(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().expect("temp file");
        write!(f, "{}", content).expect("write");
        f
    }

    #[test]
    fn test_valid_param_schema() {
        let json = r#"{
            "cf": { "min": 0.001, "max": 0.01, "default": 0.005, "description": "Critical flow" },
            "bh_decay": { "min": 0.9, "max": 0.999, "default": 0.95, "description": "BH decay" }
        }"#;
        let f = write_temp_json(json);
        // Rename so path has .json extension -- use path directly.
        let result = validate_param_schema(f.path());
        assert!(result.is_valid(), "{:?}", result.errors);
    }

    #[test]
    fn test_param_schema_min_exceeds_max() {
        let json = r#"{
            "bad_param": { "min": 0.9, "max": 0.1, "default": 0.5, "description": "bad" }
        }"#;
        let f = write_temp_json(json);
        let result = validate_param_schema(f.path());
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| e.message.contains("min")));
    }

    #[test]
    fn test_param_schema_default_out_of_range() {
        let json = r#"{
            "p": { "min": 0.0, "max": 1.0, "default": 2.5, "description": "param" }
        }"#;
        let f = write_temp_json(json);
        let result = validate_param_schema(f.path());
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| e.message.contains("exceeds max")));
    }

    #[test]
    fn test_valid_event_calendar_array() {
        let json = r#"[
            { "date": "2024-01-01", "name": "New Year" },
            { "date": "2024-07-04", "name": "Independence Day" }
        ]"#;
        let f = write_temp_json(json);
        let result = validate_event_calendar(f.path());
        assert!(result.is_valid(), "{:?}", result.errors);
    }

    #[test]
    fn test_event_calendar_duplicate_date() {
        let json = r#"[
            { "date": "2024-03-15", "name": "Event A" },
            { "date": "2024-03-15", "name": "Event B" }
        ]"#;
        let f = write_temp_json(json);
        let result = validate_event_calendar(f.path());
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| e.message.contains("duplicate")));
    }

    #[test]
    fn test_event_calendar_bad_date_format() {
        let json = r#"[
            { "date": "2024/13/99", "name": "Bad Date" }
        ]"#;
        let f = write_temp_json(json);
        let result = validate_event_calendar(f.path());
        assert!(!result.is_valid());
        assert!(result.errors.iter().any(|e| e.message.contains("invalid date format")));
    }

    #[test]
    fn test_is_valid_iso_date() {
        assert!(is_valid_iso_date("2024-01-15"));
        assert!(is_valid_iso_date("2000-12-31"));
        assert!(!is_valid_iso_date("2024/01/15"));
        assert!(!is_valid_iso_date("24-01-15"));
        assert!(!is_valid_iso_date("2024-13-01")); // month 13
        assert!(!is_valid_iso_date("2024-00-01")); // month 0
    }

    #[test]
    fn test_valid_toml_config() {
        let toml = "[srfm]\nmode = \"live\"\nlog_level = \"info\"\n[risk]\nmax_drawdown = 0.15\n[execution]\nexchange = \"binance\"\n";
        let f = write_temp_toml(toml);
        let result = validate_toml_config(f.path());
        // Required fields are present, so no errors expected.
        assert!(result.errors.is_empty(), "{:?}", result.errors);
    }

    #[test]
    fn test_toml_config_empty_file() {
        let f = write_temp_toml("");
        let result = validate_toml_config(f.path());
        assert!(!result.is_valid());
    }

    #[test]
    fn test_generic_json_valid() {
        let json = r#"{"key": "value", "number": 42}"#;
        let f = write_temp_json(json);
        let result = validate_generic_json(f.path());
        assert!(result.is_valid());
    }

    #[test]
    fn test_generic_json_invalid() {
        let json = r#"{"key": "value", broken"#;
        let f = write_temp_json(json);
        let result = validate_generic_json(f.path());
        assert!(!result.is_valid());
    }

    #[test]
    fn test_validate_all_configs_empty_dir() {
        let dir = TempDir::new().expect("temp dir");
        let results = validate_all_configs(dir.path());
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_all_configs_mixed() {
        let dir = TempDir::new().expect("temp dir");

        // Write a valid param schema.
        let schema = r#"{"cf": {"min": 0.001, "max": 0.01, "default": 0.005, "description": "cf"}}"#;
        std::fs::write(dir.path().join("param_schema.json"), schema).unwrap();

        // Write a valid TOML.
        let toml = "[srfm]\nmode = \"live\"\nlog_level = \"info\"\n[risk]\nmax_drawdown = 0.15\n[execution]\nexchange = \"kraken\"\n";
        std::fs::write(dir.path().join("srfm.toml"), toml).unwrap();

        let results = validate_all_configs(dir.path());
        assert_eq!(results.len(), 2);
        let all_valid = results.iter().all(|r| r.is_valid());
        assert!(all_valid, "{:?}", results);
    }

    #[test]
    fn test_validation_result_is_valid() {
        let mut r = ValidationResult::new(PathBuf::from("test.json"));
        assert!(r.is_valid());
        r.push_error("field", "something wrong", "f64");
        assert!(!r.is_valid());
    }

    #[test]
    fn test_event_calendar_object_format() {
        // Object keyed by date.
        let json = r#"{
            "2024-01-01": "New Year",
            "2024-07-04": "Independence Day"
        }"#;
        let f = write_temp_json(json);
        let result = validate_event_calendar(f.path());
        assert!(result.is_valid(), "{:?}", result.errors);
    }
}
