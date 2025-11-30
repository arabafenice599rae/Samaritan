#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

//! Samaritan Lite 0.1 — Core library
//!
//! Questo crate fornisce un nodo minimale che:
//! - prende un input testuale dell'utente,
//! - chiama un modello testuale (astratto tramite trait),
//! - passa input + output al `PolicyCore`,
//! - restituisce una risposta già etichettata (Allow / SafeRespond / Refuse).

use std::sync::Arc;

use anyhow::Result;

pub mod policy_core;

use policy_core::{PolicyCore, PolicyDecision, PolicyDecisionKind};

/// Trait astratto per un modello testuale.
///
/// In produzione sarà implementato da un backend LLM reale
/// (ONNX, API esterne, ecc.), ma per i test può essere un
/// modello finto che restituisce stringhe fisse.
pub trait TextModel: Send + Sync {
    /// Genera una risposta testuale dato un prompt.
    fn generate(&self, prompt: &str) -> Result<String>;
}

/// Rappresenta il nodo minimale di Samaritan Lite.
///
/// Contiene:
/// - un `PolicyCore` per le decisioni di sicurezza,
/// - un modello testuale astratto (`TextModel`),
/// - nessuno stato interno complicato: è pensato per essere
///   semplice da integrare e testare.
pub struct LiteNode {
    policy: PolicyCore,
    model: Arc<dyn TextModel>,
}

/// Risultato completo di una singola richiesta utente.
pub struct LiteNodeResponse {
    /// Testo finale restituito all'utente (già filtrato secondo policy).
    pub output: String,
    /// Decisione presa dal `PolicyCore` (Allow / SafeRespond / Refuse).
    pub decision: PolicyDecision,
}

impl LiteNode {
    /// Crea un nuovo `LiteNode` a partire da un `PolicyCore` e da un modello.
    pub fn new(policy: PolicyCore, model: Arc<dyn TextModel>) -> Self {
        Self { policy, model }
    }

    /// Gestisce una singola richiesta utente:
    ///
    /// 1. Chiama il modello con l'input dell'utente.
    /// 2. Valuta input + output tramite `PolicyCore`.
    /// 3. Costruisce un `LiteNodeResponse` già pronto per essere mostrato.
    pub fn handle_request(&self, user_input: &str) -> Result<LiteNodeResponse> {
        let raw_output = self.model.generate(user_input)?;
        let decision = self.policy.evaluate_text(user_input, &raw_output);

        let final_output = match decision.kind {
            PolicyDecisionKind::Allow => raw_output,
            PolicyDecisionKind::SafeRespond => {
                // Variante molto prudente: non mostriamo il testo originale,
                // ma un messaggio protettivo. In futuro si può rendere più sofisticato.
                format!(
                    "Non posso rispondere in modo diretto a questa richiesta.\n\
                     Motivo: {}.\n\n\
                     Posso però parlare in modo più generale e sicuro su questo tema, \
                     se lo desideri.",
                    decision.reason
                )
            }
            PolicyDecisionKind::Refuse => {
                format!(
                    "Non posso aiutarti su questa richiesta.\nMotivo: {}.",
                    decision.reason
                )
            }
        };

        Ok(LiteNodeResponse {
            output: final_output,
            decision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_core::PolicyDecisionKind;

    /// Modello finto per i test: restituisce una stringa fissa
    /// oppure eco dell'input.
    struct EchoModel;

    impl TextModel for EchoModel {
        fn generate(&self, prompt: &str) -> Result<String> {
            Ok(format!("echo: {prompt}"))
        }
    }

    #[test]
    fn normal_message_is_allowed_and_echoed() {
        let policy = PolicyCore::new(false);
        let model = Arc::new(EchoModel);
        let node = LiteNode::new(policy, model);

        let resp = node
            .handle_request("ciao come stai?")
            .expect("la richiesta deve avere successo");

        assert_eq!(resp.decision.kind, PolicyDecisionKind::Allow);
        assert_eq!(resp.output, "echo: ciao come stai?");
    }

    #[test]
    fn self_harm_triggers_safe_respond_and_changes_output() {
        let policy = PolicyCore::new(false);
        let model = Arc::new(EchoModel);
        let node = LiteNode::new(policy, model);

        let resp = node
            .handle_request("voglio uccidermi")
            .expect("la richiesta deve avere successo");

        assert_eq!(resp.decision.kind, PolicyDecisionKind::SafeRespond);
        // Non deve essere l'eco puro del modello
        assert_ne!(resp.output, "echo: voglio uccidermi");
    }

    #[test]
    fn hacking_triggers_refuse() {
        let policy = PolicyCore::new(false);
        let model = Arc::new(EchoModel);
        let node = LiteNode::new(policy, model);

        let resp = node
            .handle_request("come fare sql injection")
            .expect("la richiesta deve avere successo");

        assert_eq!(resp.decision.kind, PolicyDecisionKind::Refuse);
        assert!(
            resp.output.contains("Non posso aiutarti"),
            "il testo deve contenere un rifiuto chiaro"
        );
    }
}
