//! Differential Privacy core for Samaritan.
//!
//! Modulo self-contained, zero-dependency, pensato per:
//! - applicare clipping L2 ai gradienti,
//! - aggiungere rumore gaussiano calibrato (meccanismo (ε, δ)-DP),
//! - tracciare il consumo di privacy nel tempo tramite un accountant lineare.
//!
//! Questo componente può essere usato come building block per:
//! - DP-SGD locale,
//! - training federato con DP per ogni nodo,
//! - simulazioni di budget di privacy.
//!
//! API principali:
//! - [`DPConfig`]: configurazione (ε, δ, max_grad_norm);
//! - [`DPEngine`]: engine che clippa e noisa i gradienti;
//! - [`PrivacyAccountant`]: tracking lineare di epsilon sui round;
//! - [`l2_norm`]: utility per la norma L2.
//!
//! Tutto è scritto solo con `std`, senza RNG crittografici esterni:
//! se si vuole una variante hardened basta sostituire l'RNG interno con
//! un CSPRNG mantenendo la stessa interfaccia.

// Non usiamo attributi di crate qui: questo file è un modulo interno.

use std::f32::consts::PI;
use std::time::{SystemTime, UNIX_EPOCH};

/// Parametri di configurazione per il meccanismo DP.
///
/// Il meccanismo gaussiano aggiunge rumore calibrato per ottenere
/// (ε, δ)-differential privacy. Epsilon più piccolo ⇒ più privacy ⇒ più rumore.
#[derive(Debug, Clone, Copy)]
pub struct DPConfig {
    /// Parametro di privacy epsilon (ε). Più piccolo = privacy più forte.
    /// Valori tipici: 0.1 (forte) fino a ~8.0 (debole).
    pub epsilon: f32,

    /// Parametro di privacy delta (δ). Probabilità di fallimento della privacy.
    /// Tipicamente < 1/n, dove n è la dimensione del dataset.
    pub delta: f32,

    /// Norma L2 massima dopo il clipping dei gradienti.
    pub max_grad_norm: f32,
}

impl DPConfig {
    /// Crea una nuova configurazione DP con parametri custom.
    ///
    /// # Parametri
    ///
    /// * `epsilon` - budget di privacy per round;
    /// * `delta` - probabilità di fallimento;
    /// * `max_grad_norm` - bound di norma L2 per il clipping.
    #[must_use]
    pub fn new(epsilon: f32, delta: f32, max_grad_norm: f32) -> Self {
        Self {
            epsilon,
            delta,
            max_grad_norm,
        }
    }

    /// Preset di privacy forte: ε = 0.1, δ = 1e-5, clip = 1.0.
    ///
    /// Rumore elevato, adatto a dati molto sensibili.
    #[must_use]
    pub fn strong() -> Self {
        Self::new(0.1, 1.0e-5, 1.0)
    }

    /// Preset di privacy bilanciata: ε = 1.0, δ = 1e-5, clip = 1.0.
    ///
    /// Compromesso tra privacy e utilità.
    #[must_use]
    pub fn moderate() -> Self {
        Self::new(1.0, 1.0e-5, 1.0)
    }

    /// Preset di privacy debole: ε = 8.0, δ = 1e-5, clip = 1.0.
    ///
    /// Rumore più contenuto, massimizza l’utilità.
    #[must_use]
    pub fn weak() -> Self {
        Self::new(8.0, 1.0e-5, 1.0)
    }

    /// Calcola la scala del rumore (σ) per il meccanismo gaussiano.
    ///
    /// Formula:
    /// `σ = (sensibilità × √(2 × ln(1.25/δ))) / ε`
    ///
    /// La sensibilità è `max_grad_norm` (norma L2 dopo il clipping).
    #[must_use]
    pub fn noise_scale(&self) -> f32 {
        let sensitivity = self.max_grad_norm;
        let ln_term = (1.25_f32 / self.delta).ln();
        (sensitivity * (2.0 * ln_term).sqrt()) / self.epsilon
    }
}

impl Default for DPConfig {
    fn default() -> Self {
        Self::moderate()
    }
}

/// Motore di Differential Privacy che applica clipping + rumore ai gradienti.
///
/// Pipeline standard:
/// 1. Clipping L2 a `max_grad_norm`;
/// 2. Rumore gaussiano N(0, σ²) con σ calcolato da [`DPConfig::noise_scale`].
pub struct DPEngine {
    config: DPConfig,
    rng_state: u64,
}

impl DPEngine {
    /// Crea un nuovo motore DP con la configurazione indicata.
    ///
    /// L’RNG interno è seedato dall’orologio di sistema (con fallback fisso).
    #[must_use]
    pub fn new(config: DPConfig) -> Self {
        let rng_state = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xDEAD_BEEF_F0F0_1234);

        Self { config, rng_state }
    }

    /// Crea un motore DP con un seed esplicito (utile per test riproducibili).
    #[must_use]
    pub fn with_seed(config: DPConfig, seed: u64) -> Self {
        Self {
            config,
            rng_state: seed,
        }
    }

    /// Restituisce la configurazione corrente.
    #[must_use]
    pub fn config(&self) -> &DPConfig {
        &self.config
    }

    /// Restituisce l’epsilon configurato.
    #[must_use]
    pub fn epsilon(&self) -> f32 {
        self.config.epsilon
    }

    /// Restituisce il delta configurato.
    #[must_use]
    pub fn delta(&self) -> f32 {
        self.config.delta
    }

    /// Restituisce la deviazione standard σ usata per il rumore gaussiano.
    #[must_use]
    pub fn noise_std(&self) -> f32 {
        self.config.noise_scale()
    }

    /// Genera il prossimo `u64` pseudo-casuale via LCG.
    ///
    /// *Non* è un CSPRNG, è intenzionalmente semplice per rimanere zero-dep.
    fn next_u64(&mut self) -> u64 {
        // Parametri LCG standard (Numerical Recipes)
        self.rng_state = self
            .rng_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.rng_state
    }

    /// Genera un `f32` uniforme in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }

    /// Campiona una variabile gaussiana N(0, 1) via Box–Muller.
    fn sample_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1.0e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Applica il clipping L2 in-place sui gradienti.
    ///
    /// Se la norma L2 supera `max_grad_norm`, i gradienti sono scalati
    /// preservando la direzione.
    pub fn clip_gradients(&self, gradients: &mut [f32]) {
        let norm = l2_norm(gradients);
        let max_norm = self.config.max_grad_norm;

        if norm > max_norm && norm > 0.0 {
            let scale = max_norm / norm;
            for g in gradients.iter_mut() {
                *g *= scale;
            }
        }
    }

    /// Aggiunge rumore gaussiano calibrato ai gradienti.
    pub fn add_noise(&mut self, gradients: &mut [f32]) {
        let sigma = self.config.noise_scale();

        for g in gradients.iter_mut() {
            *g += sigma * self.sample_gaussian();
        }
    }

    /// Applica l’intera pipeline DP: clipping + rumore.
    ///
    /// Questa funzione è quella da usare in un loop di training DP-SGD.
    pub fn privatize_gradients(&mut self, gradients: &mut [f32]) {
        self.clip_gradients(gradients);
        self.add_noise(gradients);
    }
}

/// Accountant per il tracciamento del consumo di privacy (epsilon) nel tempo.
///
/// Implementa una composizione **lineare**:
/// `ε_tot = ε₁ + ε₂ + ... + εₙ`.
///
/// In molti casi reali si usa una composizione più sofisticata (RDP, moments
/// accountant, ecc.), ma questa struttura è sufficiente come punto centrale
/// per il controllo del budget.
pub struct PrivacyAccountant {
    budget: f32,
    total_epsilon: f32,
    rounds: usize,
}

impl PrivacyAccountant {
    /// Crea un nuovo accountant con budget totale `budget`.
    #[must_use]
    pub fn new(budget: f32) -> Self {
        Self {
            budget,
            total_epsilon: 0.0,
            rounds: 0,
        }
    }

    /// Registra un round di training con costo `epsilon`.
    pub fn record_round(&mut self, epsilon: f32) {
        self.total_epsilon += epsilon;
        self.rounds += 1;
    }

    /// Restituisce `true` se il budget è esaurito o superato.
    #[must_use]
    pub fn is_budget_exhausted(&self) -> bool {
        self.total_epsilon >= self.budget
    }

    /// Restituisce il budget residuo (minimo 0).
    #[must_use]
    pub fn remaining_budget(&self) -> f32 {
        (self.budget - self.total_epsilon).max(0.0)
    }

    /// Restituisce l’epsilon totale speso finora.
    #[must_use]
    pub fn total_epsilon(&self) -> f32 {
        self.total_epsilon
    }

    /// Restituisce il numero di round registrati.
    #[must_use]
    pub fn rounds(&self) -> usize {
        self.rounds
    }

    /// Restituisce il budget iniziale configurato.
    #[must_use]
    pub fn budget(&self) -> f32 {
        self.budget
    }

    /// Reset completo (per iniziare un nuovo ciclo di training).
    pub fn reset(&mut self) {
        self.total_epsilon = 0.0;
        self.rounds = 0;
    }
}

/// Calcola la norma L2 (euclidea) di un vettore.
///
/// `‖v‖₂ = √(x₁² + x₂² + ... + xₙ²)`
#[must_use]
pub fn l2_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0e-5;

    #[test]
    fn l2_norm_basic() {
        let v = [3.0_f32, 4.0_f32];
        let n = l2_norm(&v);
        assert!((n - 5.0).abs() < EPS);

        let zero = [0.0_f32, 0.0_f32];
        let n0 = l2_norm(&zero);
        assert!(n0.abs() < EPS);
    }

    #[test]
    fn noise_scale_reasonable() {
        let cfg = DPConfig::new(1.0, 1.0e-5, 1.0);
        let sigma = cfg.noise_scale();
        // σ ≈ 4.7 in questo settaggio, verifichiamo che sia > 0
        assert!(sigma > 0.0);
    }

    #[test]
    fn strong_vs_weak_noise_scale() {
        let strong = DPConfig::strong();
        let weak = DPConfig::weak();

        let s_sigma = strong.noise_scale();
        let w_sigma = weak.noise_scale();

        // Strong privacy (eps più piccolo) ⇒ rumore maggiore.
        assert!(s_sigma > w_sigma);
    }

    #[test]
    fn clip_gradients_respects_max_norm() {
        let cfg = DPConfig::new(1.0, 1.0e-5, 5.0); // max_norm = 5
        let engine = DPEngine::new(cfg);

        let mut g = [3.0_f32, 4.0_f32]; // norma 5, al limite
        engine.clip_gradients(&mut g);
        let n = l2_norm(&g);
        assert!((n - 5.0).abs() < EPS);

        let mut g_big = [6.0_f32, 8.0_f32]; // norma 10, va clippato
        engine.clip_gradients(&mut g_big);
        let n_big = l2_norm(&g_big);
        assert!((n_big - 5.0).abs() < 1.0e-3);

        // Direzione approssimativamente invariata (rapporto 3:4 ≈ 0.75)
        let ratio = g_big[0] / g_big[1];
        assert!((ratio - 0.75).abs() < 1.0e-3);
    }

    #[test]
    fn add_noise_changes_values() {
        let cfg = DPConfig::moderate();
        let mut engine = DPEngine::with_seed(cfg, 42);

        let mut grads = [1.0_f32, 2.0_f32, 3.0_f32];
        let original = grads;

        engine.add_noise(&mut grads);

        // Almeno un valore deve cambiare
        let changed = grads
            .iter()
            .zip(original.iter())
            .any(|(a, b)| (a - b).abs() > EPS);
        assert!(changed);
    }

    #[test]
    fn privatize_pipeline_runs_and_produces_finite_values() {
        let cfg = DPConfig::new(1.0, 1.0e-5, 1.0);
        let mut engine = DPEngine::with_seed(cfg, 123);

        let mut grads = [10.0_f32, 0.0_f32];
        engine.privatize_gradients(&mut grads);

        assert!(grads[0].is_finite());
        assert!(grads[1].is_finite());
    }

    #[test]
    fn accountant_tracks_budget() {
        let mut acc = PrivacyAccountant::new(1.0);

        assert!(!acc.is_budget_exhausted());
        assert!((acc.remaining_budget() - 1.0).abs() < EPS);
        assert_eq!(acc.rounds(), 0);

        acc.record_round(0.3);
        acc.record_round(0.3);
        acc.record_round(0.3);

        assert_eq!(acc.rounds(), 3);
        assert!((acc.total_epsilon() - 0.9).abs() < EPS);
        assert!((acc.remaining_budget() - 0.1).abs() < EPS);
        assert!(!acc.is_budget_exhausted());

        acc.record_round(0.2);
        assert!(acc.is_budget_exhausted());
        assert!(acc.remaining_budget() <= EPS);
    }

    #[test]
    fn accountant_reset_restores_budget() {
        let mut acc = PrivacyAccountant::new(5.0);
        acc.record_round(2.0);
        acc.record_round(2.0);

        assert_eq!(acc.rounds(), 2);
        assert!((acc.total_epsilon() - 4.0).abs() < EPS);

        acc.reset();

        assert_eq!(acc.rounds(), 0);
        assert!(acc.total_epsilon().abs() < EPS);
        assert!((acc.remaining_budget() - 5.0).abs() < EPS);
    }

    #[test]
    fn presets_are_consistent() {
        let strong = DPConfig::strong();
        let moderate = DPConfig::moderate();
        let weak = DPConfig::weak();

        assert!(strong.epsilon < moderate.epsilon);
        assert!(moderate.epsilon < weak.epsilon);

        assert!((strong.delta - moderate.delta).abs() < 1.0e-10);
        assert!((moderate.delta - weak.delta).abs() < 1.0e-10);
    }

    #[test]
    fn deterministic_with_seed() {
        let cfg = DPConfig::moderate();
        let mut e1 = DPEngine::with_seed(cfg, 999);
        let mut e2 = DPEngine::with_seed(cfg, 999);

        let mut g1 = [1.0_f32, 2.0_f32, 3.0_f32];
        let mut g2 = [1.0_f32, 2.0_f32, 3.0_f32];

        e1.add_noise(&mut g1);
        e2.add_noise(&mut g2);

        for (a, b) in g1.iter().zip(g2.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }
}
