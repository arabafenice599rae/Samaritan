//! Modulo MetaObserverLite
//!
//! Tiene traccia di quante volte la policy ha deciso:
//! - Allow
//! - SafeRespond
//! - Refuse
//!
//! È una versione ultra-semplificata del MetaObserver di Samaritan.

/// Snapshot immutabile delle statistiche raccolte dal `MetaObserverLite`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetaSnapshot {
    /// Numero totale di richieste processate.
    pub total_requests: u64,
    /// Numero di decisioni `Allow`.
    pub allow_count: u64,
    /// Numero di decisioni `SafeRespond`.
    pub safe_respond_count: u64,
    /// Numero di decisioni `Refuse`.
    pub refuse_count: u64,
}

/// Osservatore minimale che conta le decisioni della policy.
#[derive(Debug, Default)]
pub struct MetaObserverLite {
    total_requests: u64,
    allow_count: u64,
    safe_respond_count: u64,
    refuse_count: u64,
}

impl MetaObserverLite {
    /// Crea un nuovo `MetaObserverLite` con contatori azzerati.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una nuova decisione della policy.
    pub fn record_decision_allow(&mut self) {
        self.total_requests = self.total_requests.saturating_add(1);
        self.allow_count = self.allow_count.saturating_add(1);
    }

    /// Registra una nuova decisione di tipo `SafeRespond`.
    pub fn record_decision_safe_respond(&mut self) {
        self.total_requests = self.total_requests.saturating_add(1);
        self.safe_respond_count = self.safe_respond_count.saturating_add(1);
    }

    /// Registra una nuova decisione di tipo `Refuse`.
    pub fn record_decision_refuse(&mut self) {
        self.total_requests = self.total_requests.saturating_add(1);
        self.refuse_count = self.refuse_count.saturating_add(1);
    }

    /// Restituisce uno snapshot immutabile delle statistiche correnti.
    pub fn snapshot(&self) -> MetaSnapshot {
        MetaSnapshot {
            total_requests: self.total_requests,
            allow_count: self.allow_count,
            safe_respond_count: self.safe_respond_count,
            refuse_count: self.refuse_count,
        }
    }

    /// Azzera tutte le statistiche.
    pub fn reset(&mut self) {
        self.total_requests = 0;
        self.allow_count = 0;
        self.safe_respond_count = 0;
        self.refuse_count = 0;
    }
}
