//! Crate core "lite" di Samaritan.
//!
//! Questo crate contiene i mattoncini minimi e sicuri per costruire
//! un nodo semplice:
//!
//! - `policy_core`: regole di sicurezza (Allow / SafeRespond / Refuse)
//! - `meta_observer`: contatore banale di messaggi
//! - `neural_engine_lite`: un engine testuale minimale (mock LLM)

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

/// Modulo con le policy di sicurezza (self-harm, hacking, dati sensibili…).
pub mod policy_core;

/// Modulo con un osservatore minimale per metriche di alto livello.
pub mod meta_observer;

/// Modulo con un "motore neurale" estremamente semplice, usato come
/// placeholder per un LLM vero.
pub mod neural_engine_lite;
