//! Samaritan Core Lite
//!
//! Questo crate fornisce il “cuore” minimale di Samaritan in versione Lite:
//! - un motore neurale euristico (`NeuralEngineLite`),
//! - un nucleo di policy di sicurezza (`PolicyCore`),
//! - un semplice orchestratore di nodo (`SimpleNode`),
//! - un modulo di Differential Privacy standalone per federated learning
//!   (`differential_privacy`).
//!
//! Il focus è avere codice **reale**, compilabile, senza dipendenze esterne
//! oltre a `anyhow` e `regex`, mantenendo però uno standard pulito e chiaro.

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use anyhow::Result;

pub mod policy_core;
pub mod neural_engine_lite;
pub mod meta_observer;
pub mod differential_privacy;

use crate::neural_engine_lite::{ModelOutput, NeuralEngineLite, NeuralEngineLiteConfig};
use crate::policy_core::{PolicyCore, PolicyDecisionKind};

/// Riesporta i componenti di Differential Privacy per uso esterno.
///
/// Questo modulo è pensato per essere utilizzato da pipeline di training
/// federato/lite che vogliono applicare DP a gradienti o pesi.
pub use crate::differential_privacy::{l2_norm, DPConfig, DPEngine, PrivacyAccountant};

/// Riesporta alcuni tipi principali del motore neurale, per comodità.
pub use crate::neural_engine_lite::{NeuralEngineLite as LiteEngine, NeuralEngineLiteConfig as LiteEngineConfig};

/// Riesporta il nucleo di policy, per eventuali usi avanzati.
pub use crate::policy_core::{PolicyCore as LitePolicyCore, PolicyDecision, PolicyDecisionKind as LitePolicyDecisionKind};

/// Configurazione di alto livello per un `SimpleNode`.
///
/// Questa struttura definisce:
/// - se usare la modalità strict delle policy,
/// - come configurare il motore neurale lite (limite caratteri, ecc.).
#[derive(Debug, Clone)]
pub struct SimpleNodeConfig {
    /// Se `true`, abilita controlli più conservativi nel `PolicyCore`
    /// (es. limiti aggiuntivi sulla lunghezza dell'output).
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

/// Nodo logico minimale di Samaritan Lite.
///
/// Si occupa di:
/// - ricevere testo utente,
/// - generare una risposta con `NeuralEngineLite`,
/// - far passare input/output attraverso il `PolicyCore`,
/// - restituire all'esterno solo testi considerati sicuri.
///
/// Tutta la logica di policy e neural engine è incapsulata, in modo che
/// `lite-node-demo` (o altri binari) possano limitarsi a chiamare
/// `handle_input`.
#[derive(Debug)]
pub struct SimpleNode {
    policy: PolicyCore,
    engine: NeuralEngineLite,
}

impl SimpleNode {
    /// Crea un nuovo nodo a partire da una configurazione esplicita.
    ///
    /// Non usa I/O, non legge variabili d'ambiente: si limita a
    /// inizializzare i componenti interni in modo deterministico
    /// rispetto alla `config` passata.
    pub fn new(config: SimpleNodeConfig) -> Self {
        let policy = PolicyCore::new(config.strict_policy);
        let engine = NeuralEngineLite::new(config.engine);

        Self { policy, engine }
    }

    /// Crea un nodo con configurazione di default.
    ///
    /// Utile per test, esempi, o casi in cui non serve personalizzare
    /// `strict_policy` e i parametri del motore neurale.
    pub fn new_default() -> Self {
        Self::new(SimpleNodeConfig::default())
    }

    /// Accesso in sola lettura alla configurazione corrente del motore neurale.
    pub fn engine_config(&self) -> &NeuralEngineLiteConfig {
        self.engine.config()
    }

    /// Gestisce un singolo input utente, applicando:
    /// 1. generazione neurale con `NeuralEngineLite`,
    /// 2. valutazione di policy con `PolicyCore`,
    /// 3. trasformazione finale in risposta testuale sicura.
    ///
    /// Ritorna **sempre** una `String` pronta per essere mostrata
    /// all’utente (mai errori di policy come `Err`), salvo errori
    /// interni del motore neurale o I/O.
    pub fn handle_input(&mut self, user_input: &str) -> Result<String> {
        // 1. Generazione neurale (euristica, deterministica)
        let raw_output: ModelOutput = self.engine.generate(user_input)?;

        // 2. Valutazione delle policy su coppia (input, output)
        let decision = self
            .policy
            .evaluate_text(user_input, &raw_output.text);

        // 3. Costruzione della risposta finale in base alla decisione
        let final_text = match decision.kind {
            PolicyDecisionKind::Allow => {
                // Nessun problema rilevato: restituiamo il testo così com'è.
                raw_output.text
            }
            PolicyDecisionKind::SafeRespond => {
                // Contenuto potenzialmente sensibile o delicato.
                //
                // In modalità Lite, applichiamo una risposta estremamente prudente,
                // senza riportare direttamente il contenuto problematico.
                let mut out = String::new();
                out.push_str("For safety reasons I will answer in a very general and protective way.\n\n");
                out.push_str("I cannot go into technical, harmful or overly detailed instructions here.\n");
                out.push_str("If you are dealing with something risky or sensitive, consider talking with a qualified professional or trusted person.\n");
                out
            }
            PolicyDecisionKind::Refuse => {
                // Contenuto esplicitamente vietato (es. hacking, crimini, ecc.).
                //
                // In questo caso rifiutiamo in modo chiaro, senza fornire
                // alcun dettaglio operativo.
                "I’m not allowed to help with this type of request in Samaritan Lite."
                    .to_string()
            }
        };

        Ok(final_text)
    }

    /// Variante che restituisce anche i metadati neurali, prima delle policy.
    ///
    /// Utile per strumenti avanzati, logging o meta-analisi. **Attenzione**:
    /// la risposta contenuta in `ModelOutput` non è stata ancora filtrata
    /// dalle policy, quindi non va mostrata direttamente all’utente finale.
    pub fn generate_raw(&mut self, user_input: &str) -> Result<ModelOutput> {
        self.engine.generate(user_input)
    }
}
