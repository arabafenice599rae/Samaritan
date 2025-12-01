//! Samaritan 1.5 — Heavy/Core NeuroNode library
//!
//! Questo crate implementa il **NeuroNode Heavy/Core** di Samaritan:
//! ogni processo è un cervello locale completo, privacy-by-design, che
//! partecipa al cervello globale tramite Federated Learning con
//! Differential Privacy.
//!
//! # Componenti principali
//!
//! - [`NeuroNode`]: istanza completa del cervello locale
//! - [`policy_core`]: Costituzione di sicurezza e privacy
//! - [`neural_engine`]: backend neurale (ONNX / GPU, ecc.)
//! - [`federated`]: stato federato, DP-SGD, secure aggregation
//! - [`differential_privacy`]: meccanica DP di base (epsilon, delta, clipping)
//! - [`dp_trainer`]: orchestratore del training DP locale
//! - [`adaptive_throttle`] + [`scheduler`]: runtime a tick con corsie
//! - [`meta_observer`] + [`meta_brain`]: meta-livello (metriche, ADR)
//! - [`snapshot_store`] + [`update_agent`]: snapshot/rollback e aggiornamenti
//! - [`io_layer`]: interfaccia I/O con l’utente
//! - [`net`]: rete per invio/recv dei delta federati
//! - [`node_profile`]: rilevamento profilo macchina (HeavyGpu, HeavyCpu…)
//!
//! Questo file è l’orchestratore: coordina i moduli e definisce il ciclo
//! di vita del NeuroNode (bootstrap + tick).

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

/// Adaptive runtime throttle (PID + guardrail CPU/RAM).
pub mod adaptive_throttle;
/// Federated learning, DP-SGD, delta & secure aggregation.
pub mod federated;
/// User I/O (chat, voce, API locali).
pub mod io_layer;
/// Meta-livello decisionale (ADR, model slimming, ecc.).
pub mod meta_brain;
/// Metriche, detector, eventi meta.
pub mod meta_observer;
/// Networking (DeltaMessage, protocollo federato).
pub mod net;
/// Motore neurale (ONNX backend, ecc.).
pub mod neural_engine;
/// Rilevamento profilo macchina (HeavyGpu, HeavyCpu, Desktop, Edge…).
pub mod node_profile;
/// Costituzione di sicurezza e privacy.
pub mod policy_core;
/// Scheduler a priorità (Critical / Normal / Background).
pub mod scheduler;
/// Snapshot, rollback, versioni modello.
pub mod snapshot_store;
/// Aggiornamenti binari, manifest, firma.
pub mod update_agent;
/// Differential Privacy di base (config DP, engine, accountant).
pub mod differential_privacy;
/// Trainer locale DP-SGD che usa il motore neurale e la DP.
pub mod dp_trainer;

use adaptive_throttle::AdaptiveThrottle;
use differential_privacy::DPConfig;
use dp_trainer::DpTrainer;
use federated::{FederatedState, FedRoundDecision};
use io_layer::IOLayer;
use meta_brain::MetaBrain;
use meta_observer::{MetaObserver, TickMetrics};
use net::{DeltaMessage, NetClient};
use neural_engine::{NeuralEngine, OnnxBackend};
use node_profile::{NodeProfile, NodeProfileDetector};
use policy_core::PolicyCore;
use scheduler::{PriorityScheduler, ScheduledSet};
use snapshot_store::SnapshotStore;
use update_agent::UpdateAgent;

/// Identificativo unico e persistente del nodo (256 bit).
pub type NodeId = [u8; 32];

/// Risultato di un singolo tick del NeuroNode.
pub type TickResult = Result<()>;

/// Profilo di runtime del nodo, semplificato per l’orchestrazione.
///
/// Qui esponiamo solo ciò che serve al `NeuroNode`. I dettagli vivono
/// nei rispettivi moduli (`node_profile`, `adaptive_throttle`, ecc.).
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Configurazione DP di default per il trainer locale.
    pub dp: DPConfig,
    /// Numero di tick tra uno snapshot e l’altro (approx).
    pub snapshot_interval_ticks: u64,
    /// Numero di tick tra un controllo aggiornamenti e il successivo.
    pub update_interval_ticks: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            dp: DPConfig::moderate(),
            snapshot_interval_ticks: 10_000,
            update_interval_ticks: 50_000,
        }
    }
}

/// Struttura centrale del NeuroNode Heavy/Core.
///
/// Tutta la complessità è delegata ai moduli; qui coordiniamo:
/// - corsie Critical / Normal / Background
/// - DP Trainer + FederatedState
/// - MetaObserver + MetaBrain
/// - SnapshotStore + UpdateAgent
pub struct NeuroNode {
    /// Identità unica del nodo.
    pub id: NodeId,
    /// Profilo hardware/logico rilevato (HeavyGpu, HeavyCpu, …).
    pub profile: NodeProfile,

    /// Costituzione di sicurezza e privacy.
    policy_core: PolicyCore,
    /// Meta-livello (ADR, model slimming, suggerimenti config).
    meta_brain: MetaBrain,
    /// Motore neurale (modello globale + adapter privato).
    neural_engine: NeuralEngine<OnnxBackend>,
    /// I/O utente (chat, voce, API locali).
    io_layer: IOLayer,

    /// Throttle adattivo (PID + guardrail risorse).
    adaptive_throttle: AdaptiveThrottle,
    /// Scheduler a priorità (Critical / Normal / Background).
    scheduler: PriorityScheduler,

    /// Stato federato (DP-SGD locale, delta, secure aggregation).
    federated: Arc<RwLock<FederatedState>>,
    /// Client di rete per comunicare delta & round info.
    net_client: NetClient,

    /// Trainer locale DP-SGD (usa NeuralEngine + DPConfig).
    dp_trainer: DpTrainer,

    /// Snapshot e rollback del cervello locale.
    snapshot_store: SnapshotStore,
    /// Osservatore meta (metriche e detector).
    meta_observer: MetaObserver,
    /// Gestione aggiornamenti binari/manifest.
    update_agent: UpdateAgent,

    /// Config runtime (DP, intervalli snapshot/update).
    runtime_cfg: RuntimeConfig,

    /// Contatore di tick globali.
    pub tick_counter: u64,
    /// Istante di bootstrap per calcolare uptime.
    start_time: Instant,
}

impl NeuroNode {
    /// Bootstrap completo del nodo — crea o riprende uno stato esistente.
    ///
    /// Questo è il punto di ingresso principale per Heavy/Core:
    /// - genera o carica un `NodeId` persistente,
    /// - rileva il profilo macchina (`NodeProfileDetector`),
    /// - carica il modello ONNX globale,
    /// - inizializza PolicyCore, FederatedState, SnapshotStore, ecc.
    pub async fn bootstrap(
        data_dir: PathBuf,
        model_path: PathBuf,
        profile_override: Option<NodeProfile>,
        runtime_cfg: Option<RuntimeConfig>,
    ) -> Result<Self> {
        let id = Self::load_or_create_node_id(&data_dir).await?;
        let profile = profile_override.unwrap_or_else(NodeProfileDetector::detect);
        let runtime_cfg = runtime_cfg.unwrap_or_default();

        info!(
            "Samaritan 1.5 NeuroNode Heavy/Core {} — profile: {:?}",
            hex::encode(id),
            profile
        );

        let backend = OnnxBackend::load(&model_path)
            .with_context(|| format!("Unable to load global ONNX model from {:?}", model_path))?;

        let policy_core = PolicyCore::load_or_default(&data_dir).await?;
        let neural_engine = NeuralEngine::new(backend);
        let io_layer = IOLayer::new(data_dir.join("io")).await?;
        let adaptive_throttle = AdaptiveThrottle::new(&profile);
        let scheduler = PriorityScheduler::new();

        let federated_state =
            FederatedState::new(data_dir.join("federated"), &policy_core, &runtime_cfg.dp).await?;
        let net_client = NetClient::new(id);

        let dp_trainer = DpTrainer::new(runtime_cfg.dp);

        let snapshot_store = SnapshotStore::open(data_dir.join("snapshots")).await?;
        let meta_observer = MetaObserver::new(data_dir.join("metrics")).await?;
        let meta_brain = MetaBrain::new(data_dir.join("meta")).await?;
        let update_agent = UpdateAgent::new(data_dir.join("updates")).await?;

        Ok(Self {
            id,
            profile,
            policy_core,
            meta_brain,
            neural_engine,
            io_layer,
            adaptive_throttle,
            scheduler,
            federated: Arc::new(RwLock::new(federated_state)),
            net_client,
            dp_trainer,
            snapshot_store,
            meta_observer,
            update_agent,
            runtime_cfg,
            tick_counter: 0,
            start_time: Instant::now(),
        })
    }

    /// Carica o genera un NodeId persistente su disco.
    async fn load_or_create_node_id(data_dir: &Path) -> Result<NodeId> {
        let id_path = data_dir.join("node_id.bin");

        if id_path.exists() {
            let bytes = tokio::fs::read(&id_path).await?;
            let array: [u8; 32] = bytes
                .try_into()
                .map_err(|_| anyhow!("NodeId corrupted or invalid size"))?;
            Ok(array)
        } else {
            let uuid = Uuid::new_v4();
            let bytes = uuid.into_bytes();
            let mut array = [0u8; 32];
            array[0..16].copy_from_slice(&bytes);
            array[16..32].copy_from_slice(&bytes);

            tokio::fs::create_dir_all(data_dir).await?;
            tokio::fs::write(&id_path, &array).await?;
            Ok(array)
        }
    }

    /// Tick principale del NeuroNode — runtime a corsie.
    ///
    /// Ogni tick:
    /// 1. aggiorna `AdaptiveThrottle` in base al profilo e alle metriche,
    /// 2. costruisce uno [`ScheduledSet`] dal [`PriorityScheduler`],
    /// 3. esegue:
    ///    - corsia Critical (sempre, no-compromise UX),
    ///    - corsia Normal (se risorse OK),
    ///    - corsia Background (se consentito dal throttle),
    /// 4. aggiorna metriche e le manda al [`MetaObserver`].
    pub async fn tick(&mut self) -> TickResult {
        let tick_start = Instant::now();

        // 1) Aggiorna throttle in base al profilo (CPU/GPU/RAM letti internamente).
        self.adaptive_throttle.update(&self.profile);

        // 2) Chiedi al scheduler cosa deve essere processato in questo tick.
        let scheduled: ScheduledSet = self.scheduler.schedule_tick(
            self.tick_counter,
            self.adaptive_throttle.current_level(),
        );

        // ====================== CRITICAL LANE ======================
        //
        // I/O utente, inferenza neurale, PolicyCore. Nessun training qui.
        //
        if scheduled.has_critical() {
            if let Some(user_input) = self.io_layer.try_recv_user_input() {
                let model_inputs = self.io_layer.prepare_model_inputs(user_input)?;
                let raw_output = self.neural_engine.infer(&model_inputs).await?;
                let decision = self.policy_core.evaluate(&raw_output)?;
                self.io_layer.deliver_to_user(decision).await?;
            }
        }

        // ====================== NORMAL LANE ========================
        //
        // Task di servizio importanti ma non real-time (es. flush cache, I/O secondario).
        //
        if scheduled.has_normal() {
            // Qui puoi aggiungere task Normal (log flush, housekeeping, ecc.).
            // L’implementazione reale vive nei moduli specifici; il core non
            // introduce logica ad-hoc per restare pulito.
        }

        // ====================== BACKGROUND LANE ====================
        //
        // Training locale DP-SGD, federated, meta-brain, snapshot, update check.
        //
        if scheduled.has_background() && self.adaptive_throttle.allow_background() {
            let intensity = self.adaptive_throttle.current_intensity();

            // Federated & DP Trainer: solo su nodi Heavy.
            if self.profile.is_heavy() {
                let mut fed = self.federated.write().await;

                // Chiedi al FederatedState se ha senso fare un round DP locale ora.
                match fed.decide_round(&self.policy_core)? {
                    FedRoundDecision::Skip => {
                        // Nessun training in questo tick.
                    }
                    FedRoundDecision::TrainLocal { max_steps } => {
                        // Esegui DP-SGD locale tramite DpTrainer.
                        self.dp_trainer
                            .run_local_round(
                                &mut self.neural_engine,
                                &mut *fed,
                                &self.policy_core,
                                intensity,
                                max_steps,
                            )
                            .await?;

                        if fed.should_submit_delta() {
                            let delta: DeltaMessage = fed.compute_and_package_delta().await?;
                            // Invio best-effort, errori loggati ma non fatali.
                            if let Err(e) = self.net_client.submit_delta(delta).await {
                                warn!("Failed to submit federated delta: {e:?}");
                            }
                        }
                    }
                    FedRoundDecision::TrainingBlocked { reason } => {
                        warn!("Local DP training blocked by PolicyCore: {reason}");
                    }
                }
            }

            // Meta-observer: metriche e detector (solo Heavy).
            if self.profile.is_heavy() {
                self.meta_observer
                    .sample_engine(&self.neural_engine)
                    .await?;
            }

            // Snapshot periodici del modello e stato.
            if self.tick_counter % self.runtime_cfg.snapshot_interval_ticks == 0 {
                self.snapshot_store
                    .create_snapshot(&self.neural_engine)
                    .await?;
            }

            // Update agent: controllo aggiornamenti binari/manifest.
            if self.tick_counter % self.runtime_cfg.update_interval_ticks == 0 {
                if let Err(e) = self.update_agent.check_for_updates().await {
                    warn!("UpdateAgent error: {e:?}");
                }
            }
        }

        self.tick_counter += 1;

        // ====================== META-OBSERVATION ===================
        let tick_duration = tick_start.elapsed();

        let metrics = TickMetrics {
            tick: self.tick_counter,
            duration: tick_duration,
            throttle_level: self.adaptive_throttle.current_level(),
            background_intensity: self.adaptive_throttle.current_intensity(),
            profile: self.profile,
        };

        self.meta_observer.on_tick_end(&metrics).await?;

        if self.tick_counter % 5_000 == 0 {
            info!(
                "Tick {:>10} │ uptime {:>8.0?} │ {:?} │ throttle {:?} (bg {:.2})",
                self.tick_counter,
                self.start_time.elapsed(),
                self.profile,
                self.adaptive_throttle.current_level(),
                self.adaptive_throttle.current_intensity()
            );
        }

        Ok(())
    }
}
