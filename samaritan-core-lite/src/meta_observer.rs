#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

//! MetaObserver – metrics & insights per Samaritan Lite.
//!
//! Questo modulo raccoglie metriche strutturate su:
//! - conteggio richieste,
//! - modalità di risposta del motore neurale,
//! - decisioni del `PolicyCore`,
//! - latenza (EMA, min, max),
//! - dimensioni input/output e token.
//!
//! Obiettivo: dare al nodo (o a un'API HTTP) una vista compatta ma
//! completa del comportamento del cervello locale, senza dipendenze esterne.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::neural_engine_lite::{NeuralOutput, ResponseMode};
use crate::policy_core::{PolicyDecision, PolicyDecisionKind};

/// Statistiche di latenza per le inferenze.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Latenza dell'ultima richiesta, in millisecondi.
    pub last_ms: f64,
    /// Media esponenziale (EMA) della latenza, in millisecondi.
    pub avg_ms_ema: f64,
    /// Latenza massima osservata, in millisecondi.
    pub max_ms: f64,
    /// Latenza minima osservata, in millisecondi.
    pub min_ms: f64,
}

/// Metriche sul traffico testuale del nodo.
#[derive(Debug, Clone)]
pub struct TrafficStats {
    /// Totale caratteri di input ricevuti.
    pub total_input_chars: u64,
    /// Totale caratteri di output generati.
    pub total_output_chars: u64,
    /// Totale token stimati (somma `tokens_used`).
    pub total_tokens: u64,
}

/// Breakdown delle decisioni prese dal `PolicyCore`.
#[derive(Debug, Clone)]
pub struct DecisionBreakdown {
    /// Numero di risposte `Allow`.
    pub allow: u64,
    /// Numero di risposte `SafeRespond`.
    pub safe_respond: u64,
    /// Numero di risposte `Refuse`.
    pub refuse: u64,
}

/// Breakdown delle modalità di risposta del motore neurale.
#[derive(Debug, Clone)]
pub struct ModeBreakdown {
    /// Numero di risposte in modalità `Answer`.
    pub answer: u64,
    /// Numero di risposte in modalità `Summary`.
    pub summary: u64,
    /// Numero di risposte in modalità `Coaching`.
    pub coaching: u64,
}

/// Snapshot completo delle metriche osservate.
///
/// Questo è il DTO pensato per essere serializzato/esposto verso l'esterno.
#[derive(Debug, Clone)]
pub struct InferenceMetricsSnapshot {
    /// Timestamp (millisecondi dal 1970-01-01 UTC) dell'ultima update.
    pub last_update_ms: u128,
    /// Numero totale di richieste processate.
    pub total_requests: u64,
    /// Statistiche di latenza.
    pub latency: LatencyStats,
    /// Statistiche di traffico (input/output/token).
    pub traffic: TrafficStats,
    /// Breakdown di quante volte sono state prese le varie decisioni.
    pub decisions: DecisionBreakdown,
    /// Breakdown delle modalità di risposta del motore neurale.
    pub modes: ModeBreakdown,
}

/// Osservatore meta-livello per inferenze del nodo.
///
/// Viene chiamato dal `SimpleNode` dopo ogni richiesta per:
/// - aggiornare contatori,
/// - calcolare EMA di latenza,
/// - misurare dimensioni input/output,
/// - classificare modalità e decisioni.
///
/// Non fa logging su disco, non blocca, non panica: è pensato per stare
/// nel percorso hot di ogni inferenza.
#[derive(Debug, Default)]
pub struct MetaObserver {
    total_requests: u64,

    latency_last_ms: f64,
    latency_avg_ema_ms: f64,
    latency_max_ms: f64,
    latency_min_ms: f64,

    total_input_chars: u64,
    total_output_chars: u64,
    total_tokens: u64,

    decision_allow: u64,
    decision_safe_respond: u64,
    decision_refuse: u64,

    mode_answer: u64,
    mode_summary: u64,
    mode_coaching: u64,

    last_update_ms: u128,
}

impl MetaObserver {
    /// Crea un nuovo `MetaObserver` senza metriche (tutti gli zeri).
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una singola inferenza nel sistema di metriche.
    ///
    /// Chiamato tipicamente da [`SimpleNode::handle_request`] subito dopo
    /// aver ricevuto:
    ///
    /// - l'output del motore neurale,
    /// - la decisione di policy,
    /// - la durata dell'inferenza.
    ///
    /// # Parametri
    ///
    /// * `user_input` – testo fornito dall'utente;
    /// * `output` – output del motore neurale;
    /// * `decision` – decisione di policy applicata a quell'output;
    /// * `latency` – durata dell'intera inferenza (dal punto di vista nodo).
    pub fn observe_inference(
        &mut self,
        user_input: &str,
        output: &NeuralOutput,
        decision: &PolicyDecision,
        latency: Duration,
    ) {
        // 1. Aggiorna contatore richieste
        self.total_requests = self.total_requests.saturating_add(1);

        // 2. Latenza
        let latency_ms = duration_to_ms(latency);
        self.latency_last_ms = latency_ms;
        if self.total_requests == 1 {
            // Prima osservazione: init EMA, min, max
            self.latency_avg_ema_ms = latency_ms;
            self.latency_min_ms = latency_ms;
            self.latency_max_ms = latency_ms;
        } else {
            // EMA con alpha fisso (0.1) – reattivo ma non troppo rumoroso
            const ALPHA: f64 = 0.1;
            self.latency_avg_ema_ms =
                (1.0 - ALPHA) * self.latency_avg_ema_ms + ALPHA * latency_ms;
            if latency_ms > self.latency_max_ms {
                self.latency_max_ms = latency_ms;
            }
            if latency_ms < self.latency_min_ms {
                self.latency_min_ms = latency_ms;
            }
        }

        // 3. Traffico: caratteri & token
        let input_chars = user_input.chars().count() as u64;
        let output_chars = output.text.chars().count() as u64;
        let tokens = output.tokens_used as u64;

        self.total_input_chars = self.total_input_chars.saturating_add(input_chars);
        self.total_output_chars = self.total_output_chars.saturating_add(output_chars);
        self.total_tokens = self.total_tokens.saturating_add(tokens);

        // 4. Decisioni di policy
        match decision.kind {
            PolicyDecisionKind::Allow => {
                self.decision_allow = self.decision_allow.saturating_add(1);
            }
            PolicyDecisionKind::SafeRespond => {
                self.decision_safe_respond = self.decision_safe_respond.saturating_add(1);
            }
            PolicyDecisionKind::Refuse => {
                self.decision_refuse = self.decision_refuse.saturating_add(1);
            }
        }

        // 5. Modalità del motore neurale
        match output.mode {
            ResponseMode::Answer => {
                self.mode_answer = self.mode_answer.saturating_add(1);
            }
            ResponseMode::Summary => {
                self.mode_summary = self.mode_summary.saturating_add(1);
            }
            ResponseMode::Coaching => {
                self.mode_coaching = self.mode_coaching.saturating_add(1);
            }
        }

        // 6. Timestamp ultimo aggiornamento
        self.last_update_ms = current_unix_time_ms();
    }

    /// Restituisce uno snapshot immutabile delle metriche correnti.
    ///
    /// Lo snapshot è pensato per:
    /// - essere serializzato in JSON per un'API HTTP,
    /// - essere loggato in modo strutturato,
    /// - essere mostrato in una dashboard locale.
    pub fn snapshot(&self) -> InferenceMetricsSnapshot {
        InferenceMetricsSnapshot {
            last_update_ms: self.last_update_ms,
            total_requests: self.total_requests,
            latency: LatencyStats {
                last_ms: self.latency_last_ms,
                avg_ms_ema: self.latency_avg_ema_ms,
                max_ms: self.latency_max_ms,
                min_ms: self.latency_min_ms,
            },
            traffic: TrafficStats {
                total_input_chars: self.total_input_chars,
                total_output_chars: self.total_output_chars,
                total_tokens: self.total_tokens,
            },
            decisions: DecisionBreakdown {
                allow: self.decision_allow,
                safe_respond: self.decision_safe_respond,
                refuse: self.decision_refuse,
            },
            modes: ModeBreakdown {
                answer: self.mode_answer,
                summary: self.mode_summary,
                coaching: self.mode_coaching,
            },
        }
    }

    /// Resetta tutte le metriche a zero.
    ///
    /// Utile per:
    /// - iniziare una nuova sessione di misura,
    /// - separare ambienti/periodi (es. "ultimo giorno").
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Converte una [`Duration`] in millisecondi (f64).
fn duration_to_ms(d: Duration) -> f64 {
    let secs = d.as_secs() as f64;
    let nanos = d.subsec_nanos() as f64;
    secs * 1_000.0 + nanos / 1_000_000.0
}

/// Restituisce l'ora corrente in millisecondi dal 1970-01-01 UTC.
///
/// In caso di errore, ritorna 0 (non blocca mai il percorso hot).
fn current_unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_core::PolicyDecision;

    #[test]
    fn new_observer_has_zero_metrics() {
        let obs = MetaObserver::new();
        let snap = obs.snapshot();

        assert_eq!(snap.total_requests, 0);
        assert_eq!(snap.traffic.total_input_chars, 0);
        assert_eq!(snap.traffic.total_output_chars, 0);
        assert_eq!(snap.traffic.total_tokens, 0);
        assert_eq!(snap.decisions.allow, 0);
        assert_eq!(snap.decisions.safe_respond, 0);
        assert_eq!(snap.decisions.refuse, 0);
        assert_eq!(snap.modes.answer, 0);
        assert_eq!(snap.modes.summary, 0);
        assert_eq!(snap.modes.coaching, 0);
    }

    #[test]
    fn observe_inference_updates_counters() {
        let mut obs = MetaObserver::new();

        let input = "Hello, Samaritan Lite!";
        let output = NeuralOutput {
            text: "Hi! This is a test answer.".to_string(),
            mode: ResponseMode::Answer,
            tokens_used: 7,
        };
        let decision = PolicyDecision {
            kind: PolicyDecisionKind::Allow,
            reason: "test".to_string(),
        };

        obs.observe_inference(input, &output, &decision, Duration::from_millis(42));

        let snap = obs.snapshot();

        assert_eq!(snap.total_requests, 1);
        assert_eq!(
            snap.traffic.total_input_chars,
            input.chars().count() as u64
        );
        assert_eq!(
            snap.traffic.total_output_chars,
            output.text.chars().count() as u64
        );
        assert_eq!(snap.traffic.total_tokens, 7);
        assert_eq!(snap.decisions.allow, 1);
        assert_eq!(snap.decisions.safe_respond, 0);
        assert_eq!(snap.decisions.refuse, 0);
        assert_eq!(snap.modes.answer, 1);
        assert_eq!(snap.modes.summary, 0);
        assert_eq!(snap.modes.coaching, 0);
        assert!(snap.latency.last_ms > 0.0);
        assert!(snap.latency.avg_ms_ema > 0.0);
    }

    #[test]
    fn reset_clears_metrics() {
        let mut obs = MetaObserver::new();

        let output = NeuralOutput {
            text: "something".to_string(),
            mode: ResponseMode::Coaching,
            tokens_used: 3,
        };
        let decision = PolicyDecision {
            kind: PolicyDecisionKind::Refuse,
            reason: "test".to_string(),
        };

        obs.observe_inference("x", &output, &decision, Duration::from_millis(10));
        assert_eq!(obs.snapshot().total_requests, 1);

        obs.reset();
        let snap = obs.snapshot();
        assert_eq!(snap.total_requests, 0);
        assert_eq!(snap.traffic.total_tokens, 0);
        assert_eq!(snap.decisions.refuse, 0);
        assert_eq!(snap.modes.coaching, 0);
    }
}
