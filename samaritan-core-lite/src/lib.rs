// samaritan-core-lite/src/lib.rs
//
// Crate core di Samaritan Lite:
// - PolicyCore: regole di sicurezza basilari
// - NeuralEngine: motore "neurale" minimale (per ora pseudo-LLM)
// - SimpleNode: orchestratore tra engine + policy + meta-observer
// - MetaObserverLite: contatore semplice di statistiche sulle decisioni di policy

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

/// Modulo che contiene il core delle policy di sicurezza.
pub mod policy_core;
/// Modulo che contiene il motore neurale minimale.
pub mod neural_engine;
/// Modulo che contiene il nodo semplice che orchestra engine + policy.
pub mod node;
/// Modulo che contiene il meta-observer leggero.
pub mod meta_observer;

pub use policy_core::{PolicyCore, PolicyDecision, PolicyDecisionKind};
pub use neural_engine::NeuralEngine;
pub use node::SimpleNode;
pub use meta_observer::{MetaObserverLite, MetaSnapshot};
