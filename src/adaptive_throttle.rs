//! Adaptive throttling system for Samaritan NeuroNode.
//!
//! Questo modulo implementa un sistema di throttling adattivo che:
//!
//! - monitora la latenza dei tick del cervello locale,
//! - applica un PID controller per regolare l'intensità computazionale,
//! - definisce livelli di throttle (Normal / Throttled / Survival),
//! - protegge il sistema da overload tramite guardrail su CPU/RAM.
//!
//! # Livelli di throttle
//!
//! - **Normal**: nessun throttle, massime prestazioni
//! - **Throttled**: riduzione parziale delle operazioni background
//! - **Survival**: solo operazioni critiche (inferenza utente)
//!
//! # Algoritmo
//!
//! Il throttle usa un PID controller semplificato:
//!
//! ```text
//! error = target_latency - actual_latency
//! output = Kp * error + Ki * integral + Kd * derivative
//! intensity = clamp(1.0 + output, 0.0, 1.0)
//! ```
//!
//! L'intensità controlla quante operazioni background vengono eseguite
//! per tick (1.0 = tutte, 0.5 = metà, 0.0 = nessuna).

use std::time::{Duration, Instant};

use crate::node_profile::NodeProfile;

/// Livello di throttling corrente del nodo.
///
/// Definisce quanto aggressivamente il sistema deve ridurre il carico
/// per mantenere la latenza target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThrottleLevel {
    /// Operazione normale: tutte le corsie attive.
    ///
    /// La latenza è entro il target, il sistema è stabile.
    Normal,

    /// Throttling parziale: riduzione delle operazioni background.
    ///
    /// La latenza sta superando il target, riduciamo il carico.
    Throttled,

    /// Modalità sopravvivenza: solo operazioni critiche.
    ///
    /// La latenza è molto alta o le risorse sono al limite.
    /// Solo inferenza utente, nessun training/snapshot/meta.
    Survival,
}

impl ThrottleLevel {
    /// Restituisce `true` se le operazioni background sono permesse.
    #[must_use]
    pub const fn allows_background(&self) -> bool {
        matches!(self, Self::Normal | Self::Throttled)
    }

    /// Restituisce una stringa human-readable del livello.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Throttled => "Throttled",
            Self::Survival => "Survival",
        }
    }
}

impl Default for ThrottleLevel {
    fn default() -> Self {
        Self::Normal
    }
}

impl std::fmt::Display for ThrottleLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Configurazione del throttle adattivo.
///
/// Definisce i parametri del PID controller e i threshold per i livelli.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Latenza target per tick, in millisecondi.
    ///
    /// Se la latenza supera questo valore, il throttle aumenta.
    /// Valori tipici: 10-50ms per nodi Heavy, 50-200ms per Desktop/Mobile.
    pub target_latency_ms: f64,

    /// Coefficiente proporzionale del PID (Kp).
    ///
    /// Controlla la reazione immediata all'errore di latenza.
    pub kp: f64,

    /// Coefficiente integrale del PID (Ki).
    ///
    /// Accumula l'errore nel tempo per eliminare offset steady-state.
    pub ki: f64,

    /// Coefficiente derivativo del PID (Kd).
    ///
    /// Reagisce alla velocità di cambiamento dell'errore (damping).
    pub kd: f64,

    /// Latenza massima prima di passare a Survival mode, in ms.
    pub survival_threshold_ms: f64,

    /// Soglia di intensità sotto la quale si va in Throttled mode.
    pub throttled_threshold: f64,
}

impl ThrottleConfig {
    /// Preset per nodi Heavy: latenza target bassa, reazione aggressiva.
    #[must_use]
    pub fn heavy() -> Self {
        Self {
            target_latency_ms: 10.0,
            kp: 0.05,
            ki: 0.001,
            kd: 0.01,
            survival_threshold_ms: 100.0,
            throttled_threshold: 0.7,
        }
    }

    /// Preset per nodi Desktop: latenza target moderata, reazione equilibrata.
    #[must_use]
    pub fn desktop() -> Self {
        Self {
            target_latency_ms: 50.0,
            kp: 0.03,
            ki: 0.0005,
            kd: 0.005,
            survival_threshold_ms: 200.0,
            throttled_threshold: 0.6,
        }
    }

    /// Preset per nodi Mobile: latenza target alta, conservazione risorse.
    #[must_use]
    pub fn mobile() -> Self {
        Self {
            target_latency_ms: 200.0,
            kp: 0.02,
            ki: 0.0002,
            kd: 0.002,
            survival_threshold_ms: 500.0,
            throttled_threshold: 0.5,
        }
    }

    /// Seleziona automaticamente il preset in base al profilo del nodo.
    #[must_use]
    pub fn for_profile(profile: &NodeProfile) -> Self {
        match profile {
            NodeProfile::HeavyGpu | NodeProfile::HeavyCpu => Self::heavy(),
            NodeProfile::Desktop => Self::desktop(),
            NodeProfile::Mobile => Self::mobile(),
        }
    }
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self::desktop()
    }
}

/// Sistema di throttling adattivo basato su PID controller.
///
/// Monitora la latenza dei tick e regola l'intensità computazionale
/// per mantenere il sistema stabile e responsivo.
#[derive(Debug)]
pub struct AdaptiveThrottle {
    config: ThrottleConfig,
    current_level: ThrottleLevel,
    intensity: f64,

    // PID state
    integral: f64,
    last_error: f64,
    last_update: Instant,

    // Metriche
    tick_count: u64,
    last_latency_ms: f64,
    avg_latency_ms: f64,
}

impl AdaptiveThrottle {
    /// Crea un nuovo throttle con configurazione di default (Desktop).
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ThrottleConfig::default())
    }

    /// Crea un nuovo throttle con configurazione custom.
    #[must_use]
    pub fn with_config(config: ThrottleConfig) -> Self {
        Self {
            config,
            current_level: ThrottleLevel::Normal,
            intensity: 1.0,
            integral: 0.0,
            last_error: 0.0,
            last_update: Instant::now(),
            tick_count: 0,
            last_latency_ms: 0.0,
            avg_latency_ms: 0.0,
        }
    }

    /// Aggiorna il throttle in base al profilo del nodo.
    ///
    /// Chiamato all'inizio di ogni tick per aggiornare la configurazione
    /// e calcolare il nuovo livello di throttle.
    ///
    /// In una versione futura, potrebbe anche leggere metriche di sistema
    /// (CPU%, RAM%) per decisioni più informate.
    pub fn update(&mut self, profile: &NodeProfile) {
        // Aggiorna config se il profilo è cambiato (raro ma possibile)
        let expected_config = ThrottleConfig::for_profile(profile);
        if (self.config.target_latency_ms - expected_config.target_latency_ms).abs() > 1.0 {
            self.config = expected_config;
        }

        self.tick_count = self.tick_count.wrapping_add(1);
    }

    /// Registra la latenza di un tick e aggiorna il PID controller.
    ///
    /// Questo metodo dovrebbe essere chiamato alla fine di ogni tick
    /// per alimentare il feedback loop del throttle.
    pub fn record_tick_latency(&mut self, latency: Duration) {
        let latency_ms = duration_to_ms(latency);
        self.last_latency_ms = latency_ms;

        // Aggiorna media mobile esponenziale (EMA)
        const ALPHA: f64 = 0.1;
        if self.tick_count == 1 {
            self.avg_latency_ms = latency_ms;
        } else {
            self.avg_latency_ms = (1.0 - ALPHA) * self.avg_latency_ms + ALPHA * latency_ms;
        }

        // Calcola nuovo livello e intensità via PID
        self.update_pid_controller(latency_ms);
        self.update_throttle_level();
    }

    /// Aggiorna il PID controller in base alla latenza misurata.
    fn update_pid_controller(&mut self, latency_ms: f64) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        // Evita divisione per zero o dt troppo piccoli
        if dt < 1e-6 {
            return;
        }

        // Errore: positivo se siamo più veloci del target (buono),
        //         negativo se siamo più lenti (cattivo)
        let error = self.config.target_latency_ms - latency_ms;

        // Termine proporzionale
        let p = self.config.kp * error;

        // Termine integrale (con anti-windup: clamp a ±10)
        self.integral += error * dt;
        self.integral = self.integral.clamp(-10.0, 10.0);
        let i = self.config.ki * self.integral;

        // Termine derivativo
        let derivative = (error - self.last_error) / dt;
        let d = self.config.kd * derivative;

        self.last_error = error;

        // Output PID: quanto dobbiamo aggiustare l'intensità
        let output = p + i + d;

        // Intensità: 1.0 = piena potenza, 0.0 = nulla
        // Partiamo da 1.0 e sottraiamo l'output se negativo (latenza alta)
        self.intensity = (1.0 + output * 0.1).clamp(0.0, 1.0);
    }

    /// Aggiorna il livello di throttle in base all'intensità e alle soglie.
    fn update_throttle_level(&mut self) {
        // Survival: latenza oltre la soglia critica
        if self.last_latency_ms > self.config.survival_threshold_ms {
            self.current_level = ThrottleLevel::Survival;
            self.intensity = 0.0; // forza intensità a zero
            return;
        }

        // Throttled: intensità sotto soglia
        if self.intensity < self.config.throttled_threshold {
            self.current_level = ThrottleLevel::Throttled;
            return;
        }

        // Normal: tutto ok
        self.current_level = ThrottleLevel::Normal;
    }

    /// Restituisce `true` se le operazioni background sono permesse.
    ///
    /// Usato dal [`NeuroNode::tick`] per decidere se eseguire training,
    /// snapshot, meta-observer, ecc.
    #[must_use]
    pub fn allow_background(&self) -> bool {
        self.current_level.allows_background()
    }

    /// Restituisce il livello di throttle corrente.
    #[must_use]
    pub const fn current_level(&self) -> ThrottleLevel {
        self.current_level
    }

    /// Restituisce l'intensità computazionale corrente (0.0-1.0).
    ///
    /// Può essere usata per modulare quanti batch processare,
    /// quanto parallelismo usare, ecc.
    #[must_use]
    pub const fn current_intensity(&self) -> f64 {
        self.intensity
    }

    /// Restituisce la latenza dell'ultimo tick, in millisecondi.
    #[must_use]
    pub const fn last_latency_ms(&self) -> f64 {
        self.last_latency_ms
    }

    /// Restituisce la media mobile della latenza, in millisecondi.
    #[must_use]
    pub const fn avg_latency_ms(&self) -> f64 {
        self.avg_latency_ms
    }

    /// Restituisce il numero di tick processati.
    #[must_use]
    pub const fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Reset completo del throttle (utile per test o dopo un restart).
    pub fn reset(&mut self) {
        self.current_level = ThrottleLevel::Normal;
        self.intensity = 1.0;
        self.integral = 0.0;
        self.last_error = 0.0;
        self.last_update = Instant::now();
        self.tick_count = 0;
        self.last_latency_ms = 0.0;
        self.avg_latency_ms = 0.0;
    }
}

impl Default for AdaptiveThrottle {
    fn default() -> Self {
        Self::new()
    }
}

/// Converte una [`Duration`] in millisecondi (f64).
fn duration_to_ms(d: Duration) -> f64 {
    let secs = d.as_secs() as f64;
    let nanos = f64::from(d.subsec_nanos());
    secs * 1_000.0 + nanos / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn new_throttle_starts_normal() {
        let throttle = AdaptiveThrottle::new();
        assert_eq!(throttle.current_level(), ThrottleLevel::Normal);
        assert_eq!(throttle.current_intensity(), 1.0);
        assert!(throttle.allow_background());
    }

    #[test]
    fn throttle_level_allows_background_correctly() {
        assert!(ThrottleLevel::Normal.allows_background());
        assert!(ThrottleLevel::Throttled.allows_background());
        assert!(!ThrottleLevel::Survival.allows_background());
    }

    #[test]
    fn config_presets_differ_by_profile() {
        let heavy = ThrottleConfig::heavy();
        let desktop = ThrottleConfig::desktop();
        let mobile = ThrottleConfig::mobile();

        // Heavy ha target latenza più basso
        assert!(heavy.target_latency_ms < desktop.target_latency_ms);
        assert!(desktop.target_latency_ms < mobile.target_latency_ms);

        // Survival threshold più basso per Heavy
        assert!(heavy.survival_threshold_ms < desktop.survival_threshold_ms);
        assert!(desktop.survival_threshold_ms < mobile.survival_threshold_ms);
    }

    #[test]
    fn for_profile_selects_correct_preset() {
        let heavy_cfg = ThrottleConfig::for_profile(&NodeProfile::HeavyGpu);
        let desktop_cfg = ThrottleConfig::for_profile(&NodeProfile::Desktop);
        let mobile_cfg = ThrottleConfig::for_profile(&NodeProfile::Mobile);

        assert_eq!(heavy_cfg.target_latency_ms, ThrottleConfig::heavy().target_latency_ms);
        assert_eq!(desktop_cfg.target_latency_ms, ThrottleConfig::desktop().target_latency_ms);
        assert_eq!(mobile_cfg.target_latency_ms, ThrottleConfig::mobile().target_latency_ms);
    }

    #[test]
    fn low_latency_keeps_normal_level() {
        let mut throttle = AdaptiveThrottle::new();
        let low_latency = Duration::from_millis(5); // ben sotto i 50ms di default

        throttle.record_tick_latency(low_latency);

        assert_eq!(throttle.current_level(), ThrottleLevel::Normal);
        assert!(throttle.current_intensity() >= 0.9); // vicino a 1.0
    }

    #[test]
    fn high_latency_triggers_throttling() {
        let mut throttle = AdaptiveThrottle::new();
        let high_latency = Duration::from_millis(100); // sopra i 50ms di default

        // Simula diversi tick con latenza alta
        for _ in 0..10 {
            throttle.record_tick_latency(high_latency);
            std::thread::sleep(Duration::from_millis(10));
        }

        // Dovremmo essere in throttled o survival
        assert_ne!(throttle.current_level(), ThrottleLevel::Normal);
        assert!(throttle.current_intensity() < 0.7);
    }

    #[test]
    fn extreme_latency_triggers_survival() {
        let config = ThrottleConfig::desktop(); // survival_threshold = 200ms
        let mut throttle = AdaptiveThrottle::with_config(config);

        let extreme_latency = Duration::from_millis(500); // >> 200ms
        throttle.record_tick_latency(extreme_latency);

        assert_eq!(throttle.current_level(), ThrottleLevel::Survival);
        assert!(!throttle.allow_background());
        assert_eq!(throttle.current_intensity(), 0.0);
    }

    #[test]
    fn reset_restores_initial_state() {
        let mut throttle = AdaptiveThrottle::new();

        // Porta il throttle in survival
        let bad_latency = Duration::from_millis(1000);
        throttle.record_tick_latency(bad_latency);
        assert_eq!(throttle.current_level(), ThrottleLevel::Survival);

        // Reset
        throttle.reset();

        assert_eq!(throttle.current_level(), ThrottleLevel::Normal);
        assert_eq!(throttle.current_intensity(), 1.0);
        assert_eq!(throttle.tick_count(), 0);
        assert_eq!(throttle.last_latency_ms(), 0.0);
    }

    #[test]
    fn duration_to_ms_conversion() {
        let d = Duration::from_millis(123);
        let ms = duration_to_ms(d);
        assert!((ms - 123.0).abs() < 0.1);

        let d2 = Duration::from_secs(2);
        let ms2 = duration_to_ms(d2);
        assert!((ms2 - 2000.0).abs() < 0.1);
    }

    #[test]
    fn pid_controller_reacts_to_consistent_error() {
        let mut throttle = AdaptiveThrottle::new();

        // Simula 20 tick con latenza moderatamente alta
        for _ in 0..20 {
            let latency = Duration::from_millis(70); // sopra target di 50ms
            throttle.record_tick_latency(latency);
            std::thread::sleep(Duration::from_millis(5)); // dt realistico
        }

        // L'intensità dovrebbe essere diminuita dal PID
        assert!(throttle.current_intensity() < 1.0);
        // La media dovrebbe convergere verso 70ms
        assert!((throttle.avg_latency_ms() - 70.0).abs() < 5.0);
    }
}
