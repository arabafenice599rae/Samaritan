// samaritan-core-lite/src/meta_observer.rs
//
// MetaObserverLite: raccoglie statistiche semplici sulle decisioni del PolicyCore.

use crate::PolicyDecision;
use crate::PolicyDecisionKind;

/// Snapshot immutabile dello stato del `MetaObserverLite`.
///
/// Questo tipo viene usato per leggere, in modo sicuro, le statistiche
/// correnti senza esporre i campi mutabili interni del meta-observer.
#[derive(Debug, Clone, Copy)]
pub struct MetaSnapshot {
    /// Numero totale di turni gestiti dal nodo.
    pub total_turns: u64,
    /// Numero di decisioni `Allow`.
    pub allow_count: u64,
    /// Numero di decisioni `SafeRespond`.
    pub safe_respond_count: u64,
    /// Numero di decisioni `Refuse`.
    pub refuse_count: u64,
}

impl MetaSnapshot {
    /// Restituisce la percentuale di turni con decisione `Allow` (0.0–100.0).
    pub fn allow_ratio_percent(&self) -> f64 {
        self.ratio_percent(self.allow_count)
    }

    /// Restituisce la percentuale di turni con decisione `SafeRespond` (0.0–100.0).
    pub fn safe_respond_ratio_percent(&self) -> f64 {
        self.ratio_percent(self.safe_respond_count)
    }

    /// Restituisce la percentuale di turni con decisione `Refuse` (0.0–100.0).
    pub fn refuse_ratio_percent(&self) -> f64 {
        self.ratio_percent(self.refuse_count)
    }

    fn ratio_percent(&self, count: u64) -> f64 {
        if self.total_turns == 0 {
            0.0
        } else {
            (count as f64 / self.total_turns as f64) * 100.0
        }
    }
}

/// Meta-observer minimale per Samaritan Lite.
///
/// Questo componente:
/// - conta quanti turni totali sono stati gestiti,
/// - conta quante volte il PolicyCore ha risposto con:
///   - `Allow`
///   - `SafeRespond`
///   - `Refuse`
///
/// Non fa nulla di "intelligente": è solo un contatore robusto e semplice,
/// utile per debugging, metriche base o comandi come `/stats` nella CLI.
#[derive(Debug, Default)]
pub struct MetaObserverLite {
    total_turns: u64,
    allow_count: u64,
    safe_respond_count: u64,
    refuse_count: u64,
}

impl MetaObserverLite {
    /// Crea un nuovo `MetaObserverLite` vuoto.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una decisione di policy.
    ///
    /// Va chiamato ogni volta che il nodo ha gestito un turno e
    /// ha ottenuto una `PolicyDecision` dal `PolicyCore`.
    pub fn record_decision(&mut self, decision: &PolicyDecision) {
        self.total_turns = self
            .total_turns
            .saturating_add(1);

        match decision.kind {
            PolicyDecisionKind::Allow => {
                self.allow_count = self.allow_count.saturating_add(1);
            }
            PolicyDecisionKind::SafeRespond => {
                self.safe_respond_count = self.safe_respond_count.saturating_add(1);
            }
            PolicyDecisionKind::Refuse => {
                self.refuse_count = self.refuse_count.saturating_add(1);
            }
        }
    }

    /// Restituisce uno snapshot immutabile delle statistiche correnti.
    pub fn snapshot(&self) -> MetaSnapshot {
        MetaSnapshot {
            total_turns: self.total_turns,
            allow_count: self.allow_count,
            safe_respond_count: self.safe_respond_count,
            refuse_count: self.refuse_count,
        }
    }

    /// Resetta tutte le statistiche a zero.
    pub fn reset(&mut self) {
        self.total_turns = 0;
        self.allow_count = 0;
        self.safe_respond_count = 0;
        self.refuse_count = 0;
    }
}
