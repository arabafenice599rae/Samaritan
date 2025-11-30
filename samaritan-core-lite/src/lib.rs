//! Crate di base per Samaritan Core Lite.
//!
//! Per ora espone solo il motore di policy (`PolicyCore`) usato per
//! decidere se un output del modello è ammesso, da ammorbidire o da
//! rifiutare del tutto.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

/// Modulo che implementa le regole di policy di Samaritan Lite.
pub mod policy_core;

// Re-export comodi, così da fuori puoi fare:
// use samaritan_core_lite::{PolicyCore, PolicyDecision, PolicyDecisionKind};
pub use crate::policy_core::{PolicyCore, PolicyDecision, PolicyDecisionKind};
