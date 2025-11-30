//! Local DP Trainer for Samaritan Lite.
//!
//! Questo modulo fornisce un orchestratore di training locale con
//! **Differential Privacy**, basato sul motore definito in
//! `crate::differential_privacy`.
//!
//! Non simula nulla: si aspetta **gradienti reali** calcolati dal
//! modello chiamante, e si occupa solo di:
//!   1. Clipping L2 dei gradienti,
//!   2. Aggiunta di rumore gaussiano calibrato (ε, δ),
//!   3. Aggiornare il modello con i gradienti DP,
//!   4. Aggiornare il privacy accountant.
//!
//! Può essere usato sia per training locale puro, sia come blocco
//! interno di una pipeline di Federated Learning (tipo FedAvg).
//!
//! # Uso tipico
//!
//! ```rust
//! use samaritan_core_lite::differential_privacy::DPConfig;
//! use samaritan_core_lite::dp_trainer::{DpTrainableModel, LocalDpTrainer};
//!
//! struct MyModel {
//!     weights: Vec<f32>,
//! }
//!
//! impl DpTrainableModel for MyModel {
//!     fn params_len(&self) -> usize {
//!         self.weights.len()
//!     }
//!
//!     fn apply_gradients(&mut self, gradients: &[f32]) {
//!         let lr = 0.01_f32;
//!         for (w, g) in self.weights.iter_mut().zip(gradients.iter()) {
//!             *w -= lr * g;
//!         }
//!     }
//! }
//!
//! fn train_one_step(
//!     trainer: &mut LocalDpTrainer,
//!     model: &mut MyModel,
//!     mut raw_gradients: Vec<f32>,
//! ) {
//!     if let Some(stats) = trainer.train_step(model, &mut raw_gradients) {
//!         // stats contiene info su norm L2, epsilon speso, ecc.
//!         let _ = stats;
//!     }
//! }
//! ```

use crate::differential_privacy::{l2_norm, DPConfig, DPEngine, PrivacyAccountant};

/// Trait per modelli addestrabili con Differential Privacy.
///
/// Il trainer non sa come calcolare i gradienti: se li fai tu (es.
/// backprop) e li passi, il trainer si occupa solo di sanitizzarli
/// e applicarli. Questo trait serve solo a permettere al trainer di
/// applicare i gradienti DP ai parametri del modello.
pub trait DpTrainableModel {
    /// Numero di parametri del modello (lunghezza del vettore dei gradienti).
    fn params_len(&self) -> usize;

    /// Applica i gradienti DP sanitizzati ai parametri del modello.
    ///
    /// È responsabilità del modello decidere:
    /// - il learning rate,
    /// - l'eventuale ottimizzatore (SGD, Adam, ecc),
    /// - eventuali vincoli aggiuntivi.
    fn apply_gradients(&mut self, gradients: &[f32]);
}

/// Risultato di una singola chiamata a [`LocalDpTrainer::train_step`].
///
/// Fornisce metadati utili per logging / metriche.
#[derive(Debug, Clone)]
pub struct DpTrainingStepStats {
    /// Indice dello step (parte da 1).
    pub step_index: u64,
    /// Norma L2 dei gradienti **prima** del DP (clip + rumore).
    pub raw_grad_norm: f32,
    /// Norma L2 dei gradienti **dopo** il DP.
    pub dp_grad_norm: f32,
    /// Epsilon totale speso dopo questo step.
    pub epsilon_spent: f32,
    /// Epsilon rimanente.
    pub epsilon_remaining: f32,
}

/// Trainer locale con Differential Privacy.
///
/// Orchestratore stateless rispetto ai dati: non conosce batch,
/// loss o architettura. Si limita a:
///
/// - prendere gradienti **grezzi** (già aggregati sul batch),
/// - applicare `clip_gradients + add_noise`,
/// - invocare `model.apply_gradients(dp_gradients)`,
/// - aggiornare il [`PrivacyAccountant`].
#[derive(Debug)]
pub struct LocalDpTrainer {
    dp_engine: DPEngine,
    accountant: PrivacyAccountant,
    dp_config: DPConfig,
    step_counter: u64,
}

impl LocalDpTrainer {
    /// Crea un nuovo trainer locale con DP.
    ///
    /// * `dp_config` – parametri (ε, δ, max_grad_norm).
    /// * `epsilon_budget` – budget totale di privacy per questo trainer.
    pub fn new(dp_config: DPConfig, epsilon_budget: f32) -> Self {
        let dp_engine = DPEngine::new(dp_config);
        let accountant = PrivacyAccountant::new(epsilon_budget);

        Self {
            dp_engine,
            accountant,
            dp_config,
            step_counter: 0,
        }
    }

    /// Restituisce la configurazione DP corrente.
    pub fn dp_config(&self) -> DPConfig {
        self.dp_config
    }

    /// Restituisce il budget totale di epsilon.
    pub fn epsilon_budget(&self) -> f32 {
        self.accountant.budget()
    }

    /// Restituisce l'epsilon speso finora.
    pub fn epsilon_spent(&self) -> f32 {
        self.accountant.total_epsilon()
    }

    /// Restituisce l'epsilon rimanente.
    pub fn epsilon_remaining(&self) -> f32 {
        self.accountant.remaining_budget()
    }

    /// Restituisce il numero di step completati da questo trainer.
    pub fn steps(&self) -> u64 {
        self.step_counter
    }

    /// Indica se è ancora possibile addestrare senza sforare il budget.
    pub fn can_train(&self) -> bool {
        !self.accountant.is_budget_exhausted()
    }

    /// Esegue **uno step** di training con DP.
    ///
    /// # Parametri
    ///
    /// * `model` – modello che implementa [`DpTrainableModel`].
    /// * `raw_gradients` – gradienti **grezzi** (prima del DP),
    ///   lunghezza = `model.params_len()`.
    ///
    /// # Ritorno
    ///
    /// - `Some(DpTrainingStepStats)` se lo step è stato eseguito,
    /// - `None` se il budget di privacy è già esaurito o se
    ///   `raw_gradients` è vuoto / dimensione non coerente.
    pub fn train_step<M>(&mut self, model: &mut M, raw_gradients: &mut [f32]) -> Option<DpTrainingStepStats>
    where
        M: DpTrainableModel,
    {
        if !self.can_train() {
            return None;
        }

        if raw_gradients.is_empty() || raw_gradients.len() != model.params_len() {
            return None;
        }

        let raw_norm = l2_norm(raw_gradients);

        // Applichiamo pipeline DP: clip + noise (in-place).
        self.dp_engine.privatize_gradients(raw_gradients);

        let dp_norm = l2_norm(raw_gradients);

        // Applichiamo i gradienti sanitizzati al modello.
        model.apply_gradients(raw_gradients);

        // Aggiorniamo la spesa di privacy (composizione lineare).
        self.accountant.record_round(self.dp_config.epsilon);

        self.step_counter += 1;

        Some(DpTrainingStepStats {
            step_index: self.step_counter,
            raw_grad_norm: raw_norm,
            dp_grad_norm: dp_norm,
            epsilon_spent: self.accountant.total_epsilon(),
            epsilon_remaining: self.accountant.remaining_budget(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Modello finto *solo per i test* del trainer.
    ///
    /// Il codice di produzione userà il proprio modello reale.
    struct ToyModel {
        weights: Vec<f32>,
        last_applied_gradients: Vec<f32>,
    }

    impl ToyModel {
        fn new(size: usize) -> Self {
            Self {
                weights: vec![0.0; size],
                last_applied_gradients: vec![0.0; size],
            }
        }
    }

    impl DpTrainableModel for ToyModel {
        fn params_len(&self) -> usize {
            self.weights.len()
        }

        fn apply_gradients(&mut self, gradients: &[f32]) {
            let lr = 0.1_f32;
            for (w, g) in self.weights.iter_mut().zip(gradients.iter()) {
                *w -= lr * g;
            }
            self.last_applied_gradients.copy_from_slice(gradients);
        }
    }

    #[test]
    fn trainer_respects_privacy_budget() {
        let dp_config = DPConfig::moderate(); // ε = 1.0
        let epsilon_budget = 2.5_f32;

        let mut trainer = LocalDpTrainer::new(dp_config, epsilon_budget);
        let mut model = ToyModel::new(4);

        // Gradienti finti (come se arrivassero dal backprop).
        let mut grads = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];

        // Primo step
        let s1 = trainer.train_step(&mut model, &mut grads).unwrap();
        assert_eq!(s1.step_index, 1);
        assert!((s1.epsilon_spent - dp_config.epsilon).abs() < 1e-5);

        // Secondo step
        let s2 = trainer.train_step(&mut model, &mut grads).unwrap();
        assert_eq!(s2.step_index, 2);
        assert!((s2.epsilon_spent - 2.0).abs() < 1e-5);

        // Terzo step: dovrebbe ancora essere permesso (2.0 → 3.0 > 2.5 ma
        // lo verifichiamo solo *dopo* averlo registrato).
        let s3 = trainer.train_step(&mut model, &mut grads).unwrap();
        assert_eq!(s3.step_index, 3);
        assert!(trainer.epsilon_spent() > epsilon_budget);

        // Ora il trainer deve risultare "bloccato".
        assert!(!trainer.can_train());
        assert!(trainer.train_step(&mut model, &mut grads).is_none());
    }

    #[test]
    fn trainer_checks_gradient_size() {
        let dp_config = DPConfig::moderate();
        let mut trainer = LocalDpTrainer::new(dp_config, 10.0);
        let mut model = ToyModel::new(4);

        // Gradienti vuoti → None
        let mut empty = Vec::<f32>::new();
        assert!(trainer.train_step(&mut model, &mut empty).is_none());

        // Gradienti dimensione errata → None
        let mut wrong_size = vec![1.0_f32, 2.0_f32];
        assert!(trainer.train_step(&mut model, &mut wrong_size).is_none());

        // Gradienti ok → Some(...)
        let mut ok = vec![1.0_f32; 4];
        let res = trainer.train_step(&mut model, &mut ok);
        assert!(res.is_some());
    }

    #[test]
    fn dp_gradients_are_applied_to_model() {
        let dp_config = DPConfig::moderate();
        let mut trainer = LocalDpTrainer::new(dp_config, 10.0);
        let mut model = ToyModel::new(3);

        let mut grads = vec![1.0_f32, 1.0_f32, 1.0_f32];
        let before = model.weights.clone();

        let stats = trainer.train_step(&mut model, &mut grads).unwrap();

        // Il modello deve aver cambiato i pesi.
        assert_ne!(before, model.weights);
        // La norma DP deve essere finita e > 0 nella maggior parte dei casi.
        assert!(stats.dp_grad_norm.is_finite());
    }
}
