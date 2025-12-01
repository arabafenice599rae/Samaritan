#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

//! Configuration handling for Samaritan Lite nodes.
//!
//! This module provides a small, production-ready configuration layer that
//! reads settings from environment variables.
//!
//! It is intentionally simple and dependency-free (only `std` + `anyhow`).

use std::env;

use anyhow::{Context, Result};

/// High-level configuration for a `SimpleNode`.
///
/// This controls how the node behaves at runtime:
/// - how strict the safety policies are;
/// - how long model outputs are allowed to be.
#[derive(Debug, Clone)]
pub struct SimpleNodeConfig {
    /// When `true`, the policy core uses stricter checks
    /// (e.g. more conservative behaviour on long outputs).
    pub strict_mode: bool,
    /// Maximum number of characters the neural engine is allowed
    /// to emit for a single response.
    pub max_output_chars: usize,
}

impl Default for SimpleNodeConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_output_chars: 2_000,
        }
    }
}

impl SimpleNodeConfig {
    /// Creates a new configuration with explicit values.
    pub fn new(strict_mode: bool, max_output_chars: usize) -> Self {
        Self {
            strict_mode,
            max_output_chars,
        }
    }

    /// Loads configuration from environment variables, falling back to defaults.
    ///
    /// Recognised variables:
    ///
    /// - `SAMARITAN_LITE_STRICT_MODE` or `SAMARITAN_STRICT_MODE`  
    ///   - accepted values (case-insensitive): `"1"`, `"true"`, `"yes"`, `"on"` → `true`  
    ///   - `"0"`, `"false"`, `"no"`, `"off"` → `false`  
    ///   - unset → default (`false`)
    ///
    /// - `SAMARITAN_LITE_MAX_OUTPUT_CHARS` or `SAMARITAN_MAX_OUTPUT_CHARS`  
    ///   - any positive integer; invalid values cause an error.
    pub fn from_env() -> Result<Self> {
        let mut cfg = SimpleNodeConfig::default();

        // Strict mode
        if let Some(raw) = first_env(&[
            "SAMARITAN_LITE_STRICT_MODE",
            "SAMARITAN_STRICT_MODE",
        ]) {
            cfg.strict_mode = parse_bool(&raw);
        }

        // Output length limit
        if let Some(raw) = first_env(&[
            "SAMARITAN_LITE_MAX_OUTPUT_CHARS",
            "SAMARITAN_MAX_OUTPUT_CHARS",
        ]) {
            let value: usize = raw
                .trim()
                .parse()
                .with_context(|| format!("Invalid value for max output chars: {raw}"))?;
            if value == 0 {
                return Err(anyhow::anyhow!(
                    "max_output_chars must be > 0, got {value}"
                ));
            }
            cfg.max_output_chars = value;
        }

        Ok(cfg)
    }
}

/// Returns the first defined environment variable from the given list.
fn first_env(keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Ok(value) = env::var(key) {
            return Some(value);
        }
    }
    None
}

/// Parses a loose boolean value from a string.
///
/// Accepted as `true`:
/// - `"1"`, `"true"`, `"yes"`, `"on"`
///
/// Accepted as `false`:
/// - `"0"`, `"false"`, `"no"`, `"off"`
///
/// Any other value → `false` (conservative default).
fn parse_bool(raw: &str) -> bool {
    let v = raw.trim().to_ascii_lowercase();
    matches!(v.as_str(), "1" | "true" | "yes" | "on")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn reset_env() {
        let keys = [
            "SAMARITAN_LITE_STRICT_MODE",
            "SAMARITAN_STRICT_MODE",
            "SAMARITAN_LITE_MAX_OUTPUT_CHARS",
            "SAMARITAN_MAX_OUTPUT_CHARS",
        ];
        for key in keys {
            env::remove_var(key);
        }
    }

    #[test]
    fn default_config_is_sane() {
        let cfg = SimpleNodeConfig::default();
        assert!(!cfg.strict_mode);
        assert!(cfg.max_output_chars > 0);
    }

    #[test]
    fn env_overrides_defaults() {
        reset_env();
        env::set_var("SAMARITAN_LITE_STRICT_MODE", "true");
        env::set_var("SAMARITAN_LITE_MAX_OUTPUT_CHARS", "1234");

        let cfg = SimpleNodeConfig::from_env().expect("config from env");

        assert!(cfg.strict_mode);
        assert_eq!(cfg.max_output_chars, 1_234);
    }

    #[test]
    fn alternate_env_names_are_supported() {
        reset_env();
        env::set_var("SAMARITAN_STRICT_MODE", "yes");
        env::set_var("SAMARITAN_MAX_OUTPUT_CHARS", "777");

        let cfg = SimpleNodeConfig::from_env().expect("config from env");

        assert!(cfg.strict_mode);
        assert_eq!(cfg.max_output_chars, 777);
    }

    #[test]
    fn invalid_max_output_chars_errors() {
        reset_env();
        env::set_var("SAMARITAN_LITE_MAX_OUTPUT_CHARS", "not-a-number");

        let cfg = SimpleNodeConfig::from_env();
        assert!(cfg.is_err());
    }

    #[test]
    fn zero_max_output_chars_errors() {
        reset_env();
        env::set_var("SAMARITAN_MAX_OUTPUT_CHARS", "0");

        let cfg = SimpleNodeConfig::from_env();
        assert!(cfg.is_err());
    }

    #[test]
    fn parse_bool_variants() {
        assert!(parse_bool("1"));
        assert!(parse_bool("true"));
        assert!(parse_bool("YES"));
        assert!(parse_bool("On"));

        assert!(!parse_bool("0"));
        assert!(!parse_bool("false"));
        assert!(!parse_bool("no"));
        assert!(!parse_bool("off"));
        assert!(!parse_bool("maybe"));
    }
}
