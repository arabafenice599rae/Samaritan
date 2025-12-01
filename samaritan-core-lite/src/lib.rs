//! Samaritan Core Lite
//!
//! Libreria centrale minimale per Samaritan Lite, pensata per:
//!
//! - avere un piccolo **motore neurale locale** (`NeuralEngineLite`),
//! - applicare **policy di sicurezza** centralizzate (`PolicyCore`),
//! - raccogliere **metriche per tick** (`MetaObserverLite`),
//! - supportare **Differential Privacy** (`differential_privacy` + `DpTrainer`),
//! - offrire un nodo semplice da usare nei demo (`SimpleNode`).
//!
//! Questo crate è pensato come “nodo lite” coerente con la visione Samaritan
//! 1.5, ma senza dipendenze esterne pesanti: solo `anyhow` e `regex`.

#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

mod policy_core;
mod neural_engine_lite;
mod meta_observer_lite;
mod differential_privacy;
mod dp_trainer;

use std::time::Instant;

use anyhow::Result;

pub use crate::differential_privacy::{l2_norm, DPConfig, DPEngine, PrivacyAccountant};
pub use crate::dp_trainer::{
    DpOptimizable, DpTrainer, DpTrainingConfig, RoundStats, TrainerStats,
};
pub use crate::meta_observer_lite::{AggregatedStats, MetaObserverLite, NodeTickEvent};
pub use crate::neural_engine_lite::{
    NeuralEngineLite, NeuralEngineLiteConfig, NeuralOutput, ResponseMode,
};
pub use crate::policy_core::{PolicyCore, PolicyDecision, PolicyDecisionKind};

/// Configurazione di alto livello per un `SimpleNode`.
///
/// Per il Lite demo teniamo solo poche opzioni:
/// - `strict_policy` per rendere il PolicyCore più conservativo,
/// - configurazione del motore neurale.
#[derive(Debug, Clone)]
pub struct SimpleNodeConfig {
    /// Modalità strict per il PolicyCore (più conservativo).
    pub strict_policy: bool,
    /// Configurazione del motore neurale lite.
    pub engine: NeuralEngineLiteConfig,
}

impl Default for SimpleNodeConfig {
    fn default() -> Self {
        Self {
            strict_policy: false,
            engine: NeuralEngineLiteConfig::default(),
        }
    }
}

/// Output di alto livello del `SimpleNode` per una singola richiesta.
///
/// È esattamente ciò che serve al frontend / demo:
/// - testo finale già filtrato dalla policy,
/// - decisione di policy applicata (per logging o debug),
/// - metadati utili (modalità risposta, token stimati).
#[derive(Debug, Clone)]
pub struct SimpleNodeOutput {
    /// Testo finale che il nodo restituisce all’utente.
    pub text: String,
    /// Decisione di policy applicata a questa risposta.
    pub policy_decision: PolicyDecisionKind,
    /// Modalità logica della risposta (Answer / Summary / Coaching).
    pub mode: ResponseMode,
    /// Stima approssimativa dei token usati.
    pub tokens_used: usize,
}

/// Nodo Lite completo: motore neurale + policy + meta-osservabilità.
///
/// Questo è il “cervello unico” usato dal binario `lite-node-demo`.
/// Non fa rete, non fa federated learning: è solo il nucleo locale.
///
/// Flusso per ogni messaggio:
///
/// 1. `NeuralEngineLite` genera una risposta grezza.
/// 2. `PolicyCore` valuta input+output e decide Allow / SafeRespond / Refuse.
/// 3. Applichiamo la decisione (testo finale).
/// 4. `MetaObserverLite` registra latenza e token.
/// 5. Ritorniamo un [`SimpleNodeOutput`] adatto ad essere mostrato al terminale/UI.
#[derive(Debug)]
pub struct SimpleNode {
    id: String,
    policy_core: PolicyCore,
    engine: NeuralEngineLite,
    meta: MetaObserverLite,
}

impl SimpleNode {
    /// Crea un nuovo nodo con configurazione di default.
    ///
    /// È comodo per demo / test veloci:
    /// ```ignore
    /// let mut node = SimpleNode::new(false)?;
    /// let out = node.handle_user_message("ciao")?;
    /// println!("{}", out.text);
    /// ```
    pub fn new(strict_policy: bool) -> Result<Self> {
        let cfg = SimpleNodeConfig {
            strict_policy,
            ..SimpleNodeConfig::default()
        };
        Self::from_config("node-lite-1".to_string(), cfg)
    }

    /// Costruisce un nodo a partire da un id e una configurazione esplicita.
    pub fn from_config(id: String, cfg: SimpleNodeConfig) -> Result<Self> {
        let policy_core = PolicyCore::new(cfg.strict_policy);
        let engine = NeuralEngineLite::new(cfg.engine.max_output_chars);
        let meta = MetaObserverLite::new(id.clone());

        Ok(Self {
            id,
            policy_core,
            engine,
            meta,
        })
    }

    /// Restituisce l’identificatore del nodo.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Restituisce un riferimento al meta-observer interno.
    pub fn meta_observer(&self) -> &MetaObserverLite {
        &self.meta
    }

    /// Restituisce un riferimento mutabile al meta-observer interno.
    pub fn meta_observer_mut(&mut self) -> &mut MetaObserverLite {
        &mut self.meta
    }

    /// Gestisce un singolo messaggio utente e produce un output finale.
    ///
    /// Passi:
    /// - genera testo grezzo con `NeuralEngineLite`,
    /// - applica `PolicyCore` per decidere cosa fare,
    /// - registra metriche nel `MetaObserverLite`,
    /// - restituisce un [`SimpleNodeOutput`] pronto per il frontend.
    pub fn handle_user_message(&mut self, user_input: &str) -> Result<SimpleNodeOutput> {
        let started = Instant::now();

        // 1) Generazione neurale grezza
        let raw = self.engine.generate(user_input);
        let latency_ms = started.elapsed().as_secs_f32() * 1_000.0;

        // 2) Valutazione policy
        let decision = self
            .policy_core
            .evaluate_text(user_input, &raw.text);

        // 3) Applichiamo la policy al testo
        let final_text = match decision.kind {
            PolicyDecisionKind::Allow => raw.text.clone(),
            PolicyDecisionKind::SafeRespond => {
                // Qui potresti collegare una pipeline di "riscrittura safe".
                // Per ora ci limitiamo a segnalare la modalità di sicurezza.
                format!(
                    "[safe-mode] {}\n\n(Questa risposta è stata adattata per maggiore sicurezza.)",
                    raw.text
                )
            }
            PolicyDecisionKind::Refuse => {
                "I’m not able to provide a safe answer to this request.".to_string()
            }
        };

        // 4) Aggiorniamo il MetaObserverLite
        let event = NodeTickEvent {
            node_id: self.id.clone(),
            latency_ms,
            tokens_used: raw.tokens_used as u64,
        };
        self.meta.on_tick(event);

        // 5) Ritorniamo un output pulito
        Ok(SimpleNodeOutput {
            text: final_text,
            policy_decision: decision.kind,
            mode: raw.mode,
            tokens_used: raw.tokens_used,
        })
    }
}
