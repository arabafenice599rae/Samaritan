//! Differentially Private Trainer for Samaritan Lite.
//!
//! Questo modulo fornisce un **trainer generico** che applica
//! Differential Privacy a gradienti di modello, pronto per essere
//! usato in un contesto di Federated Learning / training locale.
//!
//! # Obiettivo
//!
//! - Nessun dato grezzo esce dal nodo: solo gradienti/pesi già privatizzati.
//! - DP applicata a livello di **gradiente** tramite:
//!   1. Clipping L2 a `max_grad_norm`,
//!   2. Rumore gaussiano calibrato per (ε, δ)-DP.
//! - Tracciamento lineare del consumo di privacy tramite `PrivacyAccountant`.
//!
//! # Integrazione
//!
//! Il modulo è agnostico rispetto all'architettura del modello: richiede
//! solo che il modello implementi il trait [`DpOptimizable`].
//!
//! Esempio concettuale (fuori da questo modulo):
//!
//! ```ignore
//! use samaritan_core_lite::dp_trainer::{DpTrainer, DpTrainingConfig, DpOptimizable};
//! use samaritan_core_lite::differential_privacy::DPConfig;
//!
//! struct MyModel { /* pesi, stato, ecc. */ }
//!
//! impl DpOptimizable for MyModel {
//!     type Batch = MyBatchType;
//!
//!     fn compute_gradients(&mut self, batch: &Self::Batch) -> Vec<f32> {
//!         // backprop → gradienti flat
//!     }
//!
//!     fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
//!         // aggiorna i pesi con SGD / Adam / ecc.
//!     }
//! }
//!
//! let dp_cfg = DPConfig::moderate();
//! let trainer_cfg = DpTrainingConfig::from_dp(dp_cfg, 0.01, 50.0);
//! let model = MyModel { /* ... */ };
//!
//! let mut trainer = DpTrainer::new(model, trainer_cfg);
//!
//! // per ogni round federato:
//! let round_stats = trainer.train_round(batches_iterable)?;
//! ```

use anyhow::{anyhow, Result};

use crate::differential_privacy::{l2_norm, DPConfig, DPEngine, PrivacyAccountant};

/// Modello ottimizzabile con Differential Privacy.
///
/// Questo trait astratto rappresenta un modello qualsiasi (rete neurale,
/// regressione, ecc.) per cui è possibile:
///
/// - calcolare un gradiente flat (come `Vec<f32>`) a partire da un batch,
/// - applicare un gradiente con un certo learning rate.
///
/// Il trainer DP **non** conosce la struttura interna del modello: lavora
/// unicamente sui gradienti.
pub trait DpOptimizable {
    /// Tipo di batch gestito dal modello (dataset locale).
    type Batch;

    /// Calcola i gradienti per un batch.
    ///
    /// Il gradiente deve essere restituito come vettore flat di `f32`,
    /// con un layout consistente rispetto a `apply_gradients`.
    fn compute_gradients(&mut self, batch: &Self::Batch) -> Vec<f32>;

    /// Applica i gradienti al modello.
    ///
    /// Il trainer si occupa di DP (clipping + rumore) *prima* di chiamare
    /// questo metodo. Qui si può implementare una qualsiasi regola di
    /// aggiornamento (SGD classico, momentum, ecc.).
    ///
    /// * `gradients` – gradiente DP già privatizzato
    /// * `learning_rate` – passo di aggiornamento
    fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32);
}

/// Configurazione di alto livello per un ciclo di training DP.
///
/// Questa struttura unisce:
/// - parametri DP puri (`DPConfig`),
/// - parametri di training (learning rate),
/// - limiti operativi (budget di privacy, limiti di round/batch).
#[derive(Debug, Clone)]
pub struct DpTrainingConfig {
    /// Configurazione del meccanismo di Differential Privacy.
    pub dp: DPConfig,
    /// Learning rate usato dal trainer quando applica i gradienti.
    pub learning_rate: f32,
    /// Budget totale di privacy (ε massimo consentito).
    ///
    /// Se `epsilon_budget <= 0.0`, il budget è considerato illimitato.
    pub epsilon_budget: f32,
    /// ε addebitato per ogni round di training.
    ///
    /// Di default coincide con `dp.epsilon`, ma può essere personalizzato
    /// se si usa un accounting più sofisticato a livello esterno.
    pub epsilon_per_round: f32,
    /// Numero massimo di batch che il trainer processerà in un singolo round.
    ///
    /// Se `None`, non c'è limite interno (oltre al budget).
    pub max_batches_per_round: Option<u32>,
}

impl DpTrainingConfig {
    /// Costruisce una configurazione a partire da un [`DPConfig`],
    /// learning rate e budget di privacy totale.
    pub fn from_dp(dp: DPConfig, learning_rate: f32, epsilon_budget: f32) -> Self {
        Self {
            epsilon_per_round: dp.epsilon,
            dp,
            learning_rate,
            epsilon_budget,
            max_batches_per_round: None,
        }
    }

    /// Versione "safe" di default: DP moderata, LR modesto, budget ε=10.
    pub fn safe_default() -> Self {
        let dp = DPConfig::moderate();
        Self::from_dp(dp, 0.01, 10.0)
    }
}

/// Statistiche di un singolo round di training DP.
#[derive(Debug, Clone)]
pub struct RoundStats {
    /// Round indice (0-based).
    pub round_index: u32,
    /// Numero di batch effettivamente processati.
    pub batches_processed: u32,
    /// Media delle norme L2 dei gradienti **prima** del DP.
    pub mean_grad_norm_before: f32,
    /// Media delle norme L2 dei gradienti **dopo** il DP (clip+rumore).
    pub mean_grad_norm_after: f32,
    /// ε speso in questo round.
    pub epsilon_spent: f32,
    /// ε totale speso finora.
    pub epsilon_total: f32,
    /// ε residuo (non negativo).
    pub epsilon_remaining: f32,
    /// Indica se il budget è stato esaurito dopo questo round.
    pub budget_exhausted: bool,
}

/// Statistiche cumulative del trainer.
#[derive(Debug, Clone)]
pub struct TrainerStats {
    /// Numero di round completati.
    pub rounds_completed: u32,
    /// Numero totale di batch processati.
    pub total_batches: u64,
    /// ε totale consumato.
    pub total_epsilon: f32,
}

impl TrainerStats {
    fn new() -> Self {
        Self {
            rounds_completed: 0,
            total_batches: 0,
            total_epsilon: 0.0,
        }
    }
}

/// Trainer con Differential Privacy per modelli [`DpOptimizable`].
///
/// Questo oggetto è pensato per essere posseduto da ogni NeuroNode
/// (o sottocomponente) che effettua training locale.
///
/// Flusso logico di `train_round`:
/// 1. verifica del budget di privacy (`PrivacyAccountant`),
/// 2. per ogni batch:
///    - calcolo gradienti grezzi tramite `DpOptimizable::compute_gradients`,
///    - calcolo norma L2,
///    - applicazione `DPEngine::privatize_gradients` (clip + rumore),
///    - calcolo norma L2 post-DP,
///    - aggiornamento modello con `apply_gradients`,
/// 3. aggiornamento dell'accountant con `epsilon_per_round`,
/// 4. ritorno di [`RoundStats`].
pub struct DpTrainer<M: DpOptimizable> {
    model: M,
    cfg: DpTrainingConfig,
    dp_engine: DPEngine,
    accountant: PrivacyAccountant,
    stats: TrainerStats,
    next_round_index: u32,
}

impl<M> DpTrainer<M>
where
    M: DpOptimizable,
{
    /// Crea un nuovo trainer DP a partire da un modello e una configurazione.
    pub fn new(model: M, cfg: DpTrainingConfig) -> Self {
        let accountant = if cfg.epsilon_budget > 0.0 {
            PrivacyAccountant::new(cfg.epsilon_budget)
        } else {
            // Budget "illimitato": impostiamo un budget enorme, ma
            // di fatto `is_budget_exhausted` non scatterà mai in pratica.
            PrivacyAccountant::new(f32::MAX / 2.0)
        };

        let dp_engine = DPEngine::new(cfg.dp);

        Self {
            model,
            cfg,
            dp_engine,
            accountant,
            stats: TrainerStats::new(),
            next_round_index: 0,
        }
    }

    /// Restituisce un riferimento al modello interno.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Consuma il trainer e restituisce il modello (per salvarlo o usarlo altrove).
    pub fn into_inner(self) -> M {
        self.model
    }

    /// Restituisce le statistiche cumulative attuali.
    pub fn stats(&self) -> &TrainerStats {
        &self.stats
    }

    /// Indica se il budget di privacy è esaurito.
    pub fn is_budget_exhausted(&self) -> bool {
        self.accountant.is_budget_exhausted()
    }

    /// Esegue un singolo round di training DP su una sequenza di batch.
    ///
    /// Il round:
    /// - può essere vuoto (nessun batch) → `batches_processed = 0`,
    /// - rispetta `max_batches_per_round` se impostato,
    /// - rifiuta l'esecuzione se il budget è già esaurito.
    pub fn train_round<I>(&mut self, batches: I) -> Result<RoundStats>
    where
        I: IntoIterator<Item = M::Batch>,
    {
        if self.accountant.is_budget_exhausted() {
            return Err(anyhow!(
                "privacy budget exhausted: no further DP training is allowed"
            ));
        }

        let round_index = self.next_round_index;
        self.next_round_index = self
            .next_round_index
            .saturating_add(1);

        let mut batches_processed: u32 = 0;
        let mut sum_norm_before = 0.0_f32;
        let mut sum_norm_after = 0.0_f32;

        let max_batches = self.cfg.max_batches_per_round.unwrap_or(u32::MAX);

        for batch in batches.into_iter() {
            if batches_processed >= max_batches {
                break;
            }

            // 1) Gradienti grezzi dal modello
            let mut gradients = self.model.compute_gradients(&batch);
            if gradients.is_empty() {
                // Batch "vuoto": saltiamo senza interrompere il round.
                continue;
            }

            let norm_before = l2_norm(&gradients);

            // 2) Applica DP (clip + rumore)
            self.dp_engine.privatize_gradients(&mut gradients);

            let norm_after = l2_norm(&gradients);

            // 3) Aggiorna modello
            self.model
                .apply_gradients(&gradients, self.cfg.learning_rate);

            // 4) Statistiche
            sum_norm_before += norm_before;
            sum_norm_after += norm_after;
            batches_processed = batches_processed.saturating_add(1);
            self.stats.total_batches = self.stats.total_batches.saturating_add(1);
        }

        // Se nessun batch è stato processato, non consumiamo privacy
        // e non aggiorniamo l'accountant (ε rimane invariato).
        if batches_processed == 0 {
            return Ok(RoundStats {
                round_index,
                batches_processed,
                mean_grad_norm_before: 0.0,
                mean_grad_norm_after: 0.0,
                epsilon_spent: 0.0,
                epsilon_total: self.accountant.total_epsilon(),
                epsilon_remaining: self.accountant.remaining_budget(),
                budget_exhausted: self.accountant.is_budget_exhausted(),
            });
        }

        // Aggiorna accountant con ε per round
        let eps_round = self.cfg.epsilon_per_round;
        self.accountant.record_round(eps_round);

        let epsilon_total = self.accountant.total_epsilon();
        let epsilon_remaining = self.accountant.remaining_budget();
        let budget_exhausted = self.accountant.is_budget_exhausted();

        self.stats.rounds_completed = self.stats.rounds_completed.saturating_add(1);
        self.stats.total_epsilon = epsilon_total;

        let mean_before = sum_norm_before / batches_processed as f32;
        let mean_after = sum_norm_after / batches_processed as f32;

        Ok(RoundStats {
            round_index,
            batches_processed,
            mean_grad_norm_before: mean_before,
            mean_grad_norm_after: mean_after,
            epsilon_spent: eps_round,
            epsilon_total,
            epsilon_remaining,
            budget_exhausted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy model di test:
    /// - ha un vettore di pesi,
    /// - compute_gradients restituisce semplicemente (weights - target_val),
    /// - apply_gradients fa un update SGD.
    #[derive(Debug, Clone)]
    struct DummyModel {
        weights: Vec<f32>,
        target: f32,
    }

    impl DpOptimizable for DummyModel {
        type Batch = ();

        fn compute_gradients(&mut self, _batch: &Self::Batch) -> Vec<f32> {
            self.weights
                .iter()
                .map(|&w| w - self.target)
                .collect()
        }

        fn apply_gradients(&mut self, gradients: &[f32], learning_rate: f32) {
            for (w, g) in self.weights.iter_mut().zip(gradients) {
                *w -= learning_rate * g;
            }
        }
    }

    #[test]
    fn trainer_respects_budget_and_updates_model() -> Result<()> {
        let dp_cfg = DPConfig::moderate();
        let trainer_cfg = DpTrainingConfig::from_dp(dp_cfg, 0.1, 0.5);
        let model = DummyModel {
            weights: vec![1.0, 1.0, 1.0],
            target: 0.0,
        };

        let mut trainer = DpTrainer::new(model, trainer_cfg);

        // Round 0: un po' di training
        let stats_round_0 = trainer.train_round(std::iter::repeat(()).take(5))?;
        assert!(stats_round_0.batches_processed > 0);
        assert!(stats_round_0.epsilon_spent > 0.0);
        assert!(stats_round_0.epsilon_total > 0.0);

        let weights_after_0 = trainer.model().weights.clone();

        // Round 1: altro training
        let stats_round_1 = trainer.train_round(std::iter::repeat(()).take(5))?;
        assert_eq!(stats_round_1.round_index, 1);
        assert!(stats_round_1.epsilon_total >= stats_round_0.epsilon_total);
        let weights_after_1 = trainer.model().weights.clone();

        // I pesi devono essere cambiati tra i round
        assert_ne!(weights_after_0, weights_after_1);

        Ok(())
    }

    #[test]
    fn trainer_fails_when_budget_exhausted() -> Result<()> {
        let dp_cfg = DPConfig::moderate();
        // Budget minuscolo: dopo un round siamo già praticamente al limite.
        let trainer_cfg = DpTrainingConfig::from_dp(dp_cfg, 0.01, 0.05);
        let model = DummyModel {
            weights: vec![0.5, 0.5],
            target: 0.0,
        };

        let mut trainer = DpTrainer::new(model, trainer_cfg);

        // Primo round: deve passare
        let _ = trainer.train_round(std::iter::repeat(()).take(3))?;

        // Probabilmente il budget è già esaurito o quasi
        assert!(trainer.is_budget_exhausted());

        // Secondo round: deve restituire errore
        let res = trainer.train_round(std::iter::repeat(()).take(3));
        assert!(res.is_err());

        Ok(())
    }

    #[test]
    fn zero_batches_do_not_consume_epsilon() -> Result<()> {
        let dp_cfg = DPConfig::moderate();
        let trainer_cfg = DpTrainingConfig::from_dp(dp_cfg, 0.01, 1.0);
        let model = DummyModel {
            weights: vec![1.0, 2.0],
            target: 0.0,
        };

        let mut trainer = DpTrainer::new(model, trainer_cfg);

        let stats = trainer.train_round(std::iter::empty())?;
        assert_eq!(stats.batches_processed, 0);
        assert_eq!(stats.epsilon_spent, 0.0);
        assert_eq!(stats.epsilon_total, 0.0);

        Ok(())
    }
}
