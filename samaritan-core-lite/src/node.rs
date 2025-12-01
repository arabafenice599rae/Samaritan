//! Node runner & config glue per Samaritan 1.5 Heavy/Core.
//!
//! Questo modulo fornisce:
//!
//! - la struttura [`NodeConfig`] caricata da `samaritan.yaml`,
//! - una funzione [`build_node`] che costruisce un [`NeuroNode`] pronto all'uso,
//! - una funzione [`run_node`] che esegue il loop a tick infinito del nodo.
//!
//! È il punto di collegamento tra:
//! - file di configurazione,
//! - crate `samaritan-core` ([`NeuroNode`]),
//! - binario `node_daemon` (che chiama `build_node` + `run_node`).

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::node_profile::{NodeProfile, NodeProfileDetector};
use crate::NeuroNode;

/// Configurazione di alto livello del nodo, caricata da `samaritan.yaml`.
///
/// Questa è la sorgente di verità “umana” per un’installazione Samaritan.
/// Viene tipicamente versionata insieme al binario / pacchetto.
///
/// Esempio minimale di `samaritan.yaml`:
///
/// ```yaml
/// data_dir: "/var/lib/samaritan"
/// model_path: "/opt/samaritan/models/global_teacher.onnx"
///
/// # (opzionale)
/// profile_override: "HeavyCpu"   # oppure "HeavyGpu", "Desktop"
///
/// # (opzionale) endpoint di federated server
/// federated:
///   enabled: true
///   endpoint: "https://fl.samaritan.example.com"
///
/// # (opzionale) policy strict
/// policy:
///   strict_mode: true
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct NodeConfig {
    /// Directory radice dei dati persistenti del nodo.
    pub data_dir: PathBuf,

    /// Percorso del modello ONNX globale (teacher heavy o student heavy).
    pub model_path: PathBuf,

    /// Override esplicito del profilo (opzionale).
    ///
    /// Se assente, il profilo viene auto-rilevato da [`NodeProfileDetector`]
    /// in base a CPU, RAM, GPU, ecc.
    #[serde(default)]
    pub profile_override: Option<NodeProfile>,

    /// Blocco di configurazione per Federated Learning.
    #[serde(default)]
    pub federated: FederatedConfig,

    /// Blocco di configurazione per il core delle policy.
    #[serde(default)]
    pub policy: PolicyConfig,
}

/// Configurazione federated ad alto livello.
#[derive(Debug, Clone, Deserialize)]
pub struct FederatedConfig {
    /// Abilitazione del federated learning lato nodo.
    #[serde(default = "FederatedConfig::default_enabled")]
    pub enabled: bool,

    /// Endpoint dell’orchestratore federato (server).
    #[serde(default)]
    pub endpoint: Option<String>,
}

impl FederatedConfig {
    fn default_enabled() -> bool {
        true
    }
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: None,
        }
    }
}

/// Configurazione del core di policy.
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyConfig {
    /// Se true, il nodo usa una modalità di policy più restrittiva.
    #[serde(default)]
    pub strict_mode: bool,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self { strict_mode: false }
    }
}

impl NodeConfig {
    /// Carica la configurazione da file YAML.
    ///
    /// Tipicamente il file di default è `./samaritan.yaml` nella directory
    /// di lavoro del processo `node_daemon`.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("Unable to read config file {:?}", path))?;

        let mut cfg: NodeConfig = serde_yaml::from_str(&raw)
            .with_context(|| format!("Invalid YAML format in {:?}", path))?;

        // Auto-detection del profilo se non specificato.
        if cfg.profile_override.is_none() {
            cfg.profile_override = Some(NodeProfileDetector::detect());
        }

        Ok(cfg)
    }

    /// Versione conveniente che carica direttamente da `./samaritan.yaml`.
    pub fn load_default() -> Result<Self> {
        Self::from_yaml(Path::new("samaritan.yaml"))
    }

    /// Restituisce il profilo effettivo usato dal nodo.
    ///
    /// Se `profile_override` è `Some`, viene usato quello. Altrimenti,
    /// viene lanciata una detezione runtime (questo è ridondante rispetto
    /// a quanto fatto in [`from_yaml`], ma utile in caso di uso programmatico).
    pub fn effective_profile(&self) -> NodeProfile {
        self.profile_override
            .unwrap_or_else(NodeProfileDetector::detect)
    }
}

/// Costruisce un [`NeuroNode`] pronto all'uso a partire da una [`NodeConfig`].
///
/// Questa funzione è il punto di ingresso consigliato per qualunque processo
/// che voglia istanziare un nodo Samaritan (see `node_daemon/src/main.rs`).
pub async fn build_node(config: &NodeConfig) -> Result<NeuroNode> {
    let data_dir = config.data_dir.clone();
    let model_path = config.model_path.clone();
    let profile_override = config.profile_override;

    info!(
        "Bootstrapping Samaritan NeuroNode — data_dir={:?}, model_path={:?}, profile_override={:?}",
        data_dir, model_path, profile_override
    );

    let mut node = NeuroNode::bootstrap(data_dir, model_path, profile_override).await?;

    // Applichiamo alcune configurazioni ad alto livello ai sottosistemi.
    //
    // Nota: qui *non* modifichiamo i tipi interni (PolicyCore, FederatedState),
    // ma chiamiamo i loro metodi di configurazione. Se in futuro servirà,
    // potremo arricchire questo step.
    {
        // Configura strict mode delle policy.
        if config.policy.strict_mode {
            node.policy_core.enable_strict_mode();
            info!("PolicyCore strict_mode: ENABLED");
        } else {
            info!("PolicyCore strict_mode: disabled");
        }

        // Configura Federated lato stato interno / net_client.
        if !config.federated.enabled {
            node.federated.write().await.disable_training();
            info!("Federated learning: DISABLED via config");
        } else {
            info!("Federated learning: enabled");
        }

        if let Some(endpoint) = &config.federated.endpoint {
            node.net_client.set_endpoint(endpoint.clone());
            info!("Federated endpoint set to: {}", endpoint);
        } else {
            warn!("No federated.endpoint configured — node will not participate in remote rounds until configured");
        }
    }

    Ok(node)
}

/// Loop principale del nodo — tick infinito con graceful CPU yield.
///
/// Questo è il cuore del processo `node_daemon`.
///
/// Tipico `main`:
///
/// ```rust,no_run
/// use anyhow::Result;
/// use samaritan_core::node::{NodeConfig, build_node, run_node};
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     tracing_subscriber::fmt::init();
///
///     let cfg = NodeConfig::load_default()?;
///     let node = build_node(&cfg).await?;
///     run_node(node).await
/// }
/// ```
pub async fn run_node(mut node: NeuroNode) -> ! {
    info!("Starting NeuroNode main loop…");

    loop {
        if let Err(e) = node.tick().await {
            // In produzione: qui si potrebbe decidere di:
            // - incrementare un contatore di errori consecutivi,
            // - attivare una modalità degradata,
            // - notificare un sistema esterno.
            //
            // Per ora, logghiamo e continuiamo: il cervello non deve fermarsi.
            warn!("Fatal error in NeuroNode::tick: {e:?}");
        }

        // Evita busy-loop al 100% quando non c'è lavoro.
        tokio::task::yield_now().await;
        sleep(Duration::from_millis(1)).await;
    }
}
