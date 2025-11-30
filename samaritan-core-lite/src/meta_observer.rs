//! MetaObserverLite – telemetria minima per Samaritan Lite.
//!
//! In questa versione tiene solo un contatore dei messaggi
//! e quante volte è stato richiesto un `SafeRespond` o un `Refuse`
//! dal `PolicyCore`.

use crate::policy_core::{PolicyDecision, PolicyDecisionKind};

/// Statistiche minime raccolte dal MetaObserverLite.
#[derive(Debug, Default, Clone)]
pub struct MetaStats {
    /// Numero totale di richieste elaborate.
    pub total_requests: u64,
    /// Quante volte la policy ha chiesto una risposta "safe".
    pub safe_respond_count: u64,
    /// Quante volte la policy ha rifiutato la risposta.
    pub refuse_count: u64,
}

/// Osservatore minimale che aggiorna solo qualche contatore.
#[derive(Debug, Default)]
pub struct MetaObserverLite {
    stats: MetaStats,
}

impl MetaObserverLite {
    /// Crea un nuovo `MetaObserverLite` vuoto.
    pub fn new() -> Self {
        Self {
            stats: MetaStats::default(),
        }
    }

    /// Registra una decisione di policy e aggiorna le statistiche interne.
    pub fn record_policy_decision(&mut self, decision: &PolicyDecision) {
        self.stats.total_requests += 1;

        match decision.kind {
            PolicyDecisionKind::SafeRespond => {
                self.stats.safe_respond_count += 1;
            }
            PolicyDecisionKind::Refuse => {
                self.stats.refuse_count += 1;
            }
            PolicyDecisionKind::Allow => {
                // niente da fare
            }
        }
    }

    /// Restituisce una copia delle statistiche correnti.
    pub fn stats(&self) -> MetaStats {
        self.stats.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_core::PolicyDecisionKind;

    #[test]
    fn stats_are_updated_correctly() {
        let mut observer = MetaObserverLite::new();

        let allow = PolicyDecision {
            kind: PolicyDecisionKind::Allow,
            reason: "ok".to_string(),
        };
        let safe = PolicyDecision {
            kind: PolicyDecisionKind::SafeRespond,
            reason: "safe".to_string(),
        };
        let refuse = PolicyDecision {
            kind: PolicyDecisionKind::Refuse,
            reason: "no".to_string(),
        };

        observer.record_policy_decision(&allow);
        observer.record_policy_decision(&safe);
        observer.record_policy_decision(&refuse);

        let stats = observer.stats();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.safe_respond_count, 1);
        assert_eq!(stats.refuse_count, 1);
    }
}
