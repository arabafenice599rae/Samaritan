// samaritan-core-lite/src/node.rs
//
// Nodo semplice per Samaritan Lite:
// incapsula NeuralEngine + PolicyCore e gestisce una singola "turno" di conversazione.

use crate::{NeuralEngine, PolicyCore, PolicyDecision};

/// Nodo minimale di Samaritan Lite.
///
/// Questo tipo rappresenta un "cervello locale" estremamente semplificato:
/// - ha un `NeuralEngine` interno (motore di generazione risposte),
/// - ha un `PolicyCore` interno (controllo di sicurezza),
/// - espone un metodo `handle_turn` che va da input utente a:
///   - output del modello
///   - decisione di policy su quell'output.
#[derive(Debug)]
pub struct SimpleNode {
    engine: NeuralEngine,
    policy: PolicyCore,
}

impl SimpleNode {
    /// Crea un nuovo `SimpleNode`.
    ///
    /// * `strict_mode` — se `true`, abilita controlli più severi nel `PolicyCore`.
    pub fn new(strict_mode: bool) -> Self {
        let engine = NeuralEngine::new();
        let policy = PolicyCore::new(strict_mode);
        Self { engine, policy }
    }

    /// Gestisce un singolo turno di conversazione.
    ///
    /// Dato un `user_input`:
    /// 1. chiama il motore neurale per ottenere `model_output`,
    /// 2. fa valutare il pair (input, output) al `PolicyCore`,
    /// 3. restituisce la coppia `(model_output, decision_di_policy)`.
    ///
    /// Questo metodo **non** applica ancora alcun wrapping alla risposta
    /// (es. filtri di stile per SafeRespond): questo è lasciato allo strato
    /// superiore (UI / CLI / API) che può decidere come presentare il tutto.
    pub fn handle_turn(&mut self, user_input: &str) -> (String, PolicyDecision) {
        let output = self.engine.infer(user_input);
        let decision = self.policy.evaluate_text(user_input, &output);
        (output, decision)
    }
}
