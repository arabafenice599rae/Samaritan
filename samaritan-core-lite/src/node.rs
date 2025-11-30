// samaritan-core-lite/src/node.rs
//
// Nodo semplice per Samaritan Lite:
// incapsula NeuralEngine + PolicyCore + MetaObserverLite
// e gestisce un singolo "turno" di conversazione.

use crate::meta_observer::MetaObserverLite;
use crate::{NeuralEngine, PolicyCore, PolicyDecision};

/// Nodo minimale di Samaritan Lite.
///
/// Questo tipo rappresenta un "cervello locale" estremamente semplificato:
/// - ha un `NeuralEngine` interno (motore di generazione risposte),
/// - ha un `PolicyCore` interno (controllo di sicurezza),
/// - ha un `MetaObserverLite` interno (contatori di statistiche),
/// - espone un metodo `handle_turn` che:
///   1. prende input utente (`&str`),
///   2. genera un output testuale,
///   3. fa valutare la coppia (input, output) al `PolicyCore`,
///   4. registra la decisione nel `MetaObserverLite`,
///   5. restituisce `(output, decision_di_policy)`.
#[derive(Debug)]
pub struct SimpleNode {
    engine: NeuralEngine,
    policy: PolicyCore,
    meta: MetaObserverLite,
}

impl SimpleNode {
    /// Crea un nuovo `SimpleNode`.
    ///
    /// * `strict_mode` — se `true`, abilita controlli più severi nel `PolicyCore`.
    pub fn new(strict_mode: bool) -> Self {
        let engine = NeuralEngine::new();
        let policy = PolicyCore::new(strict_mode);
        let meta = MetaObserverLite::new();
        Self { engine, policy, meta }
    }

    /// Gestisce un singolo turno di conversazione.
    ///
    /// Dato un `user_input`:
    /// 1. chiama il motore neurale per ottenere `model_output`,
    /// 2. fa valutare il pair (input, output) al `PolicyCore`,
    /// 3. registra la decisione nel meta-observer,
    /// 4. restituisce la coppia `(model_output, decision_di_policy)`.
    pub fn handle_turn(&mut self, user_input: &str) -> (String, PolicyDecision) {
        let output = self.engine.infer(user_input);
        let decision = self.policy.evaluate_text(user_input, &output);
        self.meta.record_decision(&decision);
        (output, decision)
    }

    /// Restituisce un riferimento al meta-observer interno.
    ///
    /// Questo è utile, ad esempio, per leggere uno snapshot di statistiche
    /// e mostrarlo in una UI, in una CLI oppure esporlo via API.
    pub fn meta(&self) -> &MetaObserverLite {
        &self.meta
    }

    /// Resetta il meta-observer interno.
    pub fn reset_stats(&mut self) {
        self.meta.reset();
    }
}
