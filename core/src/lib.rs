//! Samaritan 1.5 — Heavy/Core NeuroNode library.
//!
//! Questo crate implementa il **NeuroNode** completo per il profilo Heavy/Core:
//!
//! - runtime a tick con corsie (critical / background),
//! - motore neurale (`NeuralEngine<OnnxBackend>`),
//! - `PolicyCore` per sicurezza e governance,
//! - stato federato con DP e Secure Aggregation,
//! - `MetaObserver` e `MetaBrain` per meta-livello,
//! - snapshot periodici e aggiornamenti binari via `UpdateAgent`.
//!
//! Ogni processo che esegue un nodo Samaritan Heavy/Core istanzia un [`NeuroNode`]
//! e chiama ciclicamente [`NeuroNode::tick`] (direttamente o tramite
//! [`crate::node::run_node`]).
//!
//! # Panoramica
//!
//! ```text
//! ┌───────────────────────────┐
//! │        NeuroNode          │
//! │ ┌───────────────────────┐ │
//! │ │   NeuralEngine       │ │  inferenza modello
//! │ └───────────────────────┘ │
//! │ ┌───────────────────────┐ │
//! │ │   PolicyCore         │ │  policy + safety
//! │ └───────────────────────┘ │
//! │ ┌───────────────────────┐ │
//! │ │   FederatedState     │ │  DP-SGD + delta + SecAgg
//! │ └───────────────────────┘ │
//! │ ┌───────────────────────┐ │
//! │ │   MetaObserver       │ │  metriche + eventi
//! │ └───────────────────────┘ │
//! │ ┌───────────────────────┐ │
//! │ │   MetaBrain          │ │  ADR, distillazione
//! │ └───────────────────────┘ │
//! └───────────────────────────┘
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use tokio::sync::RwLock;
use tracing::{info, warn};
use uuid::Uuid;

/// Modulo per la rilevazione del profilo hardware del nodo (CPU, RAM, GPU).
pub mod node_profile;
/// Modulo che implementa il motore neurale (ONNX backend Heavy/Core).
pub mod neural_engine;
/// Modulo che contiene il core delle policy di sicurezza e governance.
pub mod policy_core;
/// Modulo per I/O verso l'utente (chat, stream, ecc.).
pub mod io_layer;
/// Modulo per throttling adattivo basato su latenza / carico.
pub mod adaptive_throttle;
/// Modulo che implementa lo scheduler a priorità (critical/normal/background).
pub mod scheduler;
/// Modulo per Federated Learning con DP-SGD e Secure Aggregation.
pub mod federated;
/// Modulo di networking (client per invio/recezione delta).
pub mod net;
/// Modulo per osservabilità e metriche strutturate.
pub mod meta_observer;
/// Modulo per il meta-livello (ADR, distillazione, pruning, ecc.).
pub mod meta_brain;
/// Modulo per gestione snapshot del modello e rollback.
pub mod snapshot_store;
/// Modulo per gestione aggiornamenti binari (update agent).
pub mod update_agent;
/// Modulo di glue per configurazione e loop di esecuzione del nodo.
pub mod node;

use adaptive_throttle::AdaptiveThrottle;
use federated::FederatedState;
use io_layer::IOLayer;
use meta_brain::MetaBrain;
use meta_observer::MetaObserver;
use net::{DeltaMessage, NetClient};
use neural_engine::{NeuralEngine, OnnxBackend};
use node_profile::{NodeProfile, NodeProfileDetector};
use policy_core::PolicyCore;
use scheduler::PriorityScheduler;
use snapshot_store::SnapshotStore;
use update_agent::UpdateAgent;

/// Identificativo univoco di un nodo.
///
/// È un array di 32 byte derivato da un UUID v4 duplicato (16 + 16 byte),
/// sufficiente per gli use-case di Samaritan 1.5.
pub type NodeId = [u8; 32];

/// Tipo di risultato per un tick del [`NeuroNode`].
pub type TickResult = Result<()>;

/// Struttura principale che rappresenta un cervello Samaritan locale.
///
/// Un [`NeuroNode`] incapsula:
/// - modello neurale globale + adapter proprietario,
/// - policy di sicurezza,
/// - stato federato con DP,
/// - layer di I/O verso l'utente,
/// - meta-livello (observer + brain),
/// - gestione snapshot e aggiornamenti.
pub struct NeuroNode {
    /// Identificativo del nodo, persistente su disco.
    pub id: NodeId,
    /// Profilo del nodo (HeavyGpu, HeavyCpu, Desktop, ecc.).
    pub profile: NodeProfile,

    /// Core delle policy (sicurezza, privacy, governance).
    pub policy_core: PolicyCore,
    /// Meta-brain (ADR, distillazione, pruning, ecc.).
    pub meta_brain: MetaBrain,
    /// Motore neurale principale basato su ONNX.
    pub neural_engine: NeuralEngine<OnnxBackend>,
    /// Strato di I/O verso l’utente (input/output chat, ecc.).
    pub io_layer: IOLayer,

    /// Throttle adattivo basato su latenza / carico.
    pub adaptive_throttle: AdaptiveThrottle,
    /// Scheduler a priorità per le corsie di esecuzione.
    pub scheduler: PriorityScheduler,

    /// Stato federato (DP-SGD, delta, privacy accountant, ecc.).
    pub federated: Arc<RwLock<FederatedState>>,
    /// Client di rete per invio/recezione delta e messaggi.
    pub net_client: NetClient,

    /// Store degli snapshot (modelli, adapter, metadati).
    pub snapshot_store: SnapshotStore,
    /// Meta-Observer per metriche e insight.
    pub meta_observer: MetaObserver,
    /// Agent per aggiornamenti binari (download, verifica, switch).
    pub update_agent: UpdateAgent,

    /// Contatore di tick eseguiti.
    pub tick_counter: u64,
    /// Istant di avvio del nodo (per uptime).
    pub start_time: Instant,
}

impl NeuroNode {
    /// Bootstrappa un [`NeuroNode`] Heavy/Core a partire da:
    ///
    /// - una directory dati (persistenza locale),
    /// - un percorso a modello ONNX globale (teacher o student heavy),
    /// - un profilo opzionale (se `None`, viene auto-rilevato).
    ///
    /// Questa funzione:
    /// 1. carica o genera il `NodeId`,
    /// 2. determina il [`NodeProfile`],
    /// 3. carica il modello ONNX,
    /// 4. inizializza tutti i sottosistemi.
    pub async fn bootstrap(
        data_dir: PathBuf,
        model_path: PathBuf,
        profile_override: Option<NodeProfile>,
    ) -> Result<Self> {
        let id = Self::load_or_create_node_id(&data_dir).await?;
        let profile = profile_override.unwrap_or_else(NodeProfileDetector::detect);

        info!(
            "Samaritan 1.5 NeuroNode {} — profile: {:?}",
            hex::encode(id),
            profile
        );

        // In una versione futura, qui si potrà scegliere backend diverso
        // in base a NodeProfile (es. GPU vs CPU).
        let backend = OnnxBackend::load(&model_path)
            .with_context(|| format!("Unable to load global ONNX model from {:?}", model_path))?;

        Ok(Self {
            id,
            profile,

            policy_core: PolicyCore::load_or_default(&data_dir).await?,
            meta_brain: MetaBrain::new(),
            neural_engine: NeuralEngine::new(backend),
            io_layer: IOLayer::new(data_dir.join("io")).await?,

            adaptive_throttle: AdaptiveThrottle::new(),
            scheduler: PriorityScheduler::new(),

            federated: Arc::new(RwLock::new(
                FederatedState::new(data_dir.join("federated")).await?,
            )),
            net_client: NetClient::new(id),

            snapshot_store: SnapshotStore::open(data_dir.join("snapshots")).await?,
            meta_observer: MetaObserver::new(),
            update_agent: UpdateAgent::new(data_dir.join("updates")),

            tick_counter: 0,
            start_time: Instant::now(),
        })
    }

    /// Carica un `NodeId` persistito, oppure ne crea uno nuovo se assente.
    ///
    /// Il file è salvato in `data_dir/node_id.bin` ed è un array di 32 byte.
    async fn load_or_create_node_id(data_dir: &Path) -> Result<NodeId> {
        let id_path = data_dir.join("node_id.bin");

        if id_path.exists() {
            let bytes = tokio::fs::read(&id_path)
                .await
                .with_context(|| format!("Unable to read node id from {:?}", id_path))?;

            bytes
                .try_into()
                .map_err(|_| anyhow!("NodeId corrupted or wrong size (expected 32 bytes)"))
        } else {
            let uuid = Uuid::new_v4();
            let bytes = uuid.into_bytes(); // 16 bytes
            let mut array = [0_u8; 32];
            // duplico il UUID nei 32 byte
            array[0..16].copy_from_slice(&bytes);
            array[16..32].copy_from_slice(&bytes);

            tokio::fs::create_dir_all(data_dir)
                .await
                .with_context(|| format!("Unable to create data dir {:?}", data_dir))?;
            tokio::fs::write(&id_path, &array)
                .await
                .with_context(|| format!("Unable to persist node id to {:?}", id_path))?;

            Ok(array)
        }
    }

    /// Esegue **un singolo tick** del cervello locale.
    ///
    /// Questo metodo è pensato per essere invocato in un loop (vedi
    /// [`crate::node::run_node`]) e implementa:
    ///
    /// - **corsia critical**: gestione input utente + inferenza + policy;
    /// - **corsia background**: training federato DP, meta-observer,
    ///   snapshot, aggiornamenti binari;
    /// - aggiornamento di `AdaptiveThrottle` e metriche.
    pub async fn tick(&mut self) -> TickResult {
        // Aggiorna il throttle in base al profilo (in futuro: anche system load).
        self.adaptive_throttle.update(&self.profile);

        // ────────────────────────────────────────────────────────────────
        // CORSIA CRITICAL — I/O utente + inferenza + policy
        // ────────────────────────────────────────────────────────────────
        if let Some(user_input) = self.io_layer.try_recv_user_input() {
            // Prepara gli input per il modello neurale.
            let model_inputs = self.io_layer.prepare_model_inputs(user_input)?;
            // Inferenzia con il motore neurale (bloccante per la corsia critical).
            let raw_output = self.neural_engine.infer(&model_inputs).await?;
            // Applica le policy di sicurezza / governance.
            let decision = self.policy_core.evaluate(&raw_output)?;
            // Consegna la risposta all’utente.
            self.io_layer.deliver_to_user(decision).await?;
        }

        // ────────────────────────────────────────────────────────────────
        // CORSIA BACKGROUND — training, meta, snapshot, update
        // ────────────────────────────────────────────────────────────────
        if self.adaptive_throttle.allow_background() {
            let intensity = self.adaptive_throttle.current_intensity();

            // Scheduling a livello concettuale: qui potremmo chiedere al
            // PriorityScheduler quali "neuroni / moduli" eseguire.
            let _scheduled = self.scheduler.schedule_tick(self.tick_counter);

            // 1) Federated training locale (solo nodi Heavy / Desktop forti)
            if self.profile.is_heavy() {
                let mut fed = self.federated.write().await;
                if fed.is_training_enabled().await? {
                    fed.run_local_epoch(intensity).await?;
                    if fed.should_submit_delta() {
                        let delta: DeltaMessage = fed.compute_and_package_delta().await?;
                        // La failure di rete non è fatale per il tick.
                        if let Err(err) = self.net_client.submit_delta(delta).await {
                            warn!("NetClient.submit_delta failed: {err:?}");
                        }
                    }
                }
            }

            // 2) Meta-observer (metriche di inferenza / training)
            if self.profile.is_heavy() {
                self.meta_observer.sample(&self.neural_engine).await;
            }

            // 3) Snapshot periodici del modello
            if self.tick_counter % 10_000 == 0 {
                self.snapshot_store.create_snapshot(&self.neural_engine).await?;
            }

            // 4) Controllo aggiornamenti binari
            if self.tick_counter % 50_000 == 0 {
                if let Err(e) = self.update_agent.check_for_updates().await {
                    warn!("UpdateAgent error: {e:?}");
                }
            }
        }

        // ────────────────────────────────────────────────────────────────
        // Accounting interno e logging di stato
        // ────────────────────────────────────────────────────────────────
        self.tick_counter = self.tick_counter.wrapping_add(1);

        if self.tick_counter % 5_000 == 0 {
            info!(
                "Tick {:>10} │ uptime {:>8.0?} │ {:?} │ throttle {:?}",
                self.tick_counter,
                self.start_time.elapsed(),
                self.profile,
                self.adaptive_throttle.current_level()
            );
        }

        Ok(())
    }
}
