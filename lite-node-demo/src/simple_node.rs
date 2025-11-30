//! Implementazione di un nodo estremamente semplice per Samaritan Lite.
//!
//! - `NeuralEngineLite`: modello finto che genera una risposta base.
//! - `PolicyCore`: applica regole di sicurezza minimali.
//! - `MetaObserverLite`: conta quante decisioni Allow/Safe/Refuse sono state prese.

use crate::meta_observer_lite::MetaObserverLite;
use crate::policy_core::{PolicyCore, PolicyDecision, PolicyDecisionKind};

/// Motore neurale finto: per ora restituisce una risposta banalissima.
///
/// In futuro questo verrà sostituito con un vero backend ONNX / LLM.
#[derive(Debug, Default)]
pub struct NeuralEngineLite;

impl NeuralEngineLite {
    /// Crea un nuovo motore neurale finto.
    pub fn new() -> Self {
        Self
    }

    /// Genera una risposta "di esempio" a partire dall'input.
    pub fn generate(&self, user_input: &str) -> String {
        format!("Ho ricevuto: \"{user_input}\". (risposta finta del modello)")
    }
}

/// Nodo minimale: motore neurale finto + policy + meta-observer.
#[derive(Debug)]
pub struct SimpleNode {
    engine: NeuralEngineLite,
    policy: PolicyCore,
    meta: MetaObserverLite,
}

impl SimpleNode {
    /// Crea un nuovo `SimpleNode`.
    ///
    /// - `strict_mode = false` → policy più permissiva.
    /// - `strict_mode = true`  → policy più severa (es. taglia risposte lunghe).
    pub fn new(strict_mode: bool) -> Self {
        Self {
            engine: NeuralEngineLite::new(),
            policy: PolicyCore::new(strict_mode),
            meta: MetaObserverLite::new(),
        }
    }

    /// Gestisce un singolo turno:
    /// - passa l'input al motore neurale,
    /// - fa valutare la risposta dalla policy,
    /// - registra la decisione nel MetaObserverLite,
    /// - restituisce (output_del_modello, decisione_della_policy).
    pub fn handle_turn(&mut self, user_input: &str) -> (String, PolicyDecision) {
        let model_output = self.engine.generate(user_input);
        let decision = self.policy.evaluate_text(user_input, &model_output);

        // Registra la decisione nel MetaObserverLite.
        match decision.kind {
            PolicyDecisionKind::Allow => self.meta.record_decision_allow(),
            PolicyDecisionKind::SafeRespond => self.meta.record_decision_safe_respond(),
            PolicyDecisionKind::Refuse => self.meta.record_decision_refuse(),
        }

        (model_output, decision)
    }

    /// Restituisce un riferimento immutabile al MetaObserverLite.
    ///
    /// Utile per leggere le statistiche correnti (es. comando `/stats`).
    pub fn meta(&self) -> &MetaObserverLite {
        &self.meta
    }

    /// Azzera le statistiche del MetaObserverLite.
    ///
    /// Usato dal comando `/reset_stats`.
    pub fn reset_stats(&mut self) {
        self.meta.reset();
    }
}
