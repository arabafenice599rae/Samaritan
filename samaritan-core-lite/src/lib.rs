#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

//! Samaritan Lite core library.
//!
//! Questo crate contiene:
//!
//! - [`policy_core`] – regole di sicurezza / safety policy.
//! - [`neural_engine_lite`] – motore neurale deterministico e sicuro.
//! - [`meta_observer`] – raccolta di metriche e semplici insight.
//! - [`differential_privacy`] – meccanismi DP di base (clip + noise + accountant).
//! - [`dp_trainer`] – esempio di trainer locale DP (usabile in CLI / sim).
//! - [`config`] – caricamento configurazione da variabili d'ambiente.
//!
//! Sopra questi moduli c'è [`SimpleNode`], una facciata stabile che rappresenta
//! un “neurone” completo: prende input utente, genera output, applica policy,
//! manda tutto a MetaObserver, e restituisce il testo finale da mostrare.

use std::time::Instant;

use anyhow::Result;

pub mod config;
pub mod differential_privacy;
pub mod dp_trainer;
pub mod meta_observer;
pub mod neural_engine_lite;
pub mod policy_core;

pub use config::SimpleNodeConfig;
pub use differential_privacy::{DPConfig, DPEngine, PrivacyAccountant};
pub use meta_observer::MetaObserver;
pub use neural_engine_lite::{NeuralEngineLite, NeuralEngineLiteConfig, NeuralOutput};
pub use policy_core::{PolicyCore, PolicyDecision, PolicyDecisionKind};

/// High-level facade: a complete Samaritan Lite node.
///
/// This type wires together:
///
/// - [`NeuralEngineLite`] – deterministic text generator;
/// - [`PolicyCore`] – safety layer on top of raw model output;
/// - [`MetaObserver`] – metrics + lightweight insights.
///
/// Il flusso di base è:
///
/// 1. `handle_request(user_input)`
/// 2. neural engine → output grezzo
/// 3. policy → decide Allow / SafeRespond / Refuse
/// 4. meta observer → registra metriche
/// 5. restituisce il testo finale per la UI
#[derive(Debug)]
pub struct SimpleNode {
    policy: PolicyCore,
    engine: NeuralEngineLite,
    observer: MetaObserver,
}

impl SimpleNode {
    /// Costruisce un nuovo nodo a partire da una [`SimpleNodeConfig`].
    ///
    /// Non legge nulla dall'ambiente: è puro, ideale per test e embed.
    pub fn new(config: SimpleNodeConfig) -> Self {
        let policy = PolicyCore::new(config.strict_mode);

        let engine_cfg = NeuralEngineLiteConfig {
            max_output_chars: config.max_output_chars,
            ..NeuralEngineLiteConfig::default()
        };
        let engine = NeuralEngineLite::new(engine_cfg);

        let observer = MetaObserver::new();

        Self {
            policy,
            engine,
            observer,
        }
    }

    /// Costruisce un nodo leggendo la configurazione dall'ambiente.
    ///
    /// Usa [`SimpleNodeConfig::from_env`] come sorgente di verità; in caso di
    /// valori non validi (es. `SAMARITAN_MAX_OUTPUT_CHARS="abc"`), ritorna errore.
    pub fn from_env() -> Result<Self> {
        let cfg = SimpleNodeConfig::from_env()?;
        Ok(Self::new(cfg))
    }

    /// Gestisce una singola richiesta utente end-to-end.
    ///
    /// Flusso:
    ///
    /// 1. genera un output con [`NeuralEngineLite`];
    /// 2. passa `(input, output)` a [`PolicyCore::evaluate_text`];
    /// 3. registra la richiesta in [`MetaObserver`];
    /// 4. applica la decisione di policy e restituisce il testo finale.
    pub fn handle_request(&mut self, user_input: &str) -> Result<String> {
        let started_at = Instant::now();

        let raw_output: NeuralOutput = self.engine.generate(user_input)?;

        let decision: PolicyDecision = self
            .policy
            .evaluate_text(user_input, &raw_output.text);

        self.observer
            .observe_inference(user_input, &raw_output, &decision, started_at.elapsed());

        let final_text = match decision.kind {
            PolicyDecisionKind::Allow => raw_output.text,
            PolicyDecisionKind::SafeRespond => Self::wrap_safe_response(&raw_output.text),
            PolicyDecisionKind::Refuse => Self::refusal_message(),
        };

        Ok(final_text)
    }

    /// Restituisce un messaggio di rifiuto standard.
    ///
    /// Viene usato quando la policy decide che è più sicuro non rispondere.
    fn refusal_message() -> String {
        "I’m not able to answer this request safely. \
         You can try rephrasing it or focusing on a different aspect."
            .to_string()
    }

    /// Avvolge una risposta con un “layer” di sicurezza.
    ///
    /// Scenario tipico:
    /// - l’output di modello è potenzialmente sensibile / borderline;
    /// - invece di buttare via tutto, lo si incapsula in un framing più prudente.
    fn wrap_safe_response(text: &str) -> String {
        let mut out = String::new();
        out.push_str("⚠️ This answer has been safety-filtered.\n\n");
        out.push_str("Here is a safer, more general version of the response:\n\n");
        out.push_str(text);
        out
    }

    /// Espone un riferimento al [`MetaObserver`] interno.
    ///
    /// Utile per:
    /// - esportare metriche in un’API HTTP;
    /// - integrare con un sistema di logging esterno;
    /// - collezionare insight sul comportamento del nodo.
    pub fn meta_observer(&self) -> &MetaObserver {
        &self.observer
    }

    /// Espone un riferimento al [`NeuralEngineLite`] interno.
    ///
    /// Può essere utile per debug avanzato o per arricchire i log con
    /// statistiche interne del motore.
    pub fn neural_engine(&self) -> &NeuralEngineLite {
        &self.engine
    }

    /// Espone un riferimento al [`PolicyCore`] interno.
    ///
    /// Non permette di modificare le policy, ma è utile leggere lo stato
    /// (se aggiungerai campi interni in futuro).
    pub fn policy_core(&self) -> &PolicyCore {
        &self.policy
    }
}
