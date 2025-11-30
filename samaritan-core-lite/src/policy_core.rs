//! Modulo di policy di sicurezza per Samaritan Lite.
//!
//! Qui vivono le regole base che decidono se un output del modello
//! può essere restituito così com'è, va "ammorbidito" oppure
//! va rifiutato del tutto.

use regex::Regex;

/// Tipo di decisione presa dal `PolicyCore` per una risposta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDecisionKind {
    /// La risposta può essere restituita così com'è.
    Allow,
    /// La risposta va resa più sicura, astratta o protettiva.
    SafeRespond,
    /// La risposta va rifiutata del tutto.
    Refuse,
}

/// Decisione completa del `PolicyCore` su una coppia (input, output).
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// Tipo di decisione (Allow, SafeRespond, Refuse).
    pub kind: PolicyDecisionKind,
    /// Motivazione testuale, utile per logging e debugging.
    pub reason: String,
}

/// Core delle policy di sicurezza per Samaritan Lite.
///
/// Per ora è intenzionalmente semplice:
/// - qualche parola chiave su self-harm / hacking,
/// - un controllo molto grezzo su possibili numeri di carta di credito,
/// - una modalità "strict" che rende più conservativa la risposta.
#[derive(Debug, Default)]
pub struct PolicyCore {
    strict_mode: bool,
}

impl PolicyCore {
    /// Crea un nuovo `PolicyCore`.
    ///
    /// * `strict_mode` – se `true`, applica controlli più conservativi
    ///   (per esempio limita la lunghezza massima delle risposte).
    pub fn new(strict_mode: bool) -> Self {
        Self { strict_mode }
    }

    /// Valuta input utente e output del modello e decide se sono accettabili.
    ///
    /// Questa funzione **non** modifica l'output, si limita a descrivere
    /// cosa andrebbe fatto (Allow / SafeRespond / Refuse).
    pub fn evaluate_text(&self, user_input: &str, model_output: &str) -> PolicyDecision {
        let trimmed_output = model_output.trim();

        // 1) Output completamente vuoto → rifiuta
        if trimmed_output.is_empty() {
            return PolicyDecision {
                kind: PolicyDecisionKind::Refuse,
                reason: "output vuoto o non valido".to_string(),
            };
        }

        // Normalizza in minuscolo per i controlli lessicali
        let lower_input = user_input.to_lowercase();
        let lower_output = model_output.to_lowercase();

        // 2) Contenuto self-harm (super grezzo ma sufficiente per la demo)
        if lower_input.contains("voglio uccidermi")
            || lower_input.contains("farla finita")
            || lower_output.contains("voglio uccidermi")
            || lower_output.contains("farla finita")
        {
            return PolicyDecision {
                kind: PolicyDecisionKind::SafeRespond,
                reason: "rilevato contenuto self-harm".to_string(),
            };
        }

        // 3) Hacking / crimine informatico
        let crime_keywords = [
            "ddos",
            "sql injection",
            "come bucare",
            "exploit 0day",
            "zero-day",
            "ransomware",
        ];

        if crime_keywords
            .iter()
            .any(|kw| lower_input.contains(kw) || lower_output.contains(kw))
        {
            return PolicyDecision {
                kind: PolicyDecisionKind::Refuse,
                reason: "rilevato contenuto di hacking o crimine".to_string(),
            };
        }

        // 4) Possibili dati sensibili (numero di carta molto grezzo).
        //    Se la regex dovesse mai essere invalida (bug nostro),
        //    semplicemente saltiamo questo controllo e non andiamo in panic.
        if let Ok(cc_regex) = Regex::new(r"\b(?:\d[ -]*?){13,16}\b") {
            if cc_regex.is_match(user_input) || cc_regex.is_match(model_output) {
                return PolicyDecision {
                    kind: PolicyDecisionKind::SafeRespond,
                    reason: "possibile dato sensibile rilevato".to_string(),
                };
            }
        }

        // 5) Modalità strict: limita risposte troppo lunghe
        if self.strict_mode && model_output.len() > 10_000 {
            return PolicyDecision {
                kind: PolicyDecisionKind::SafeRespond,
                reason: "risposta troppo lunga in modalità strict".to_string(),
            };
        }

        // 6) Nessuna violazione rilevata
        PolicyDecision {
            kind: PolicyDecisionKind::Allow,
            reason: "nessuna violazione rilevata".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_normal_message() {
        let core = PolicyCore::new(false);
        let decision = core.evaluate_text("ciao", "risposta normale");
        assert_eq!(decision.kind, PolicyDecisionKind::Allow);
    }

    #[test]
    fn detect_self_harm() {
        let core = PolicyCore::new(false);
        let decision = core.evaluate_text("voglio uccidermi", "...");
        assert_eq!(decision.kind, PolicyDecisionKind::SafeRespond);
    }

    #[test]
    fn detect_crime_keyword() {
        let core = PolicyCore::new(false);
        let decision = core.evaluate_text("come faccio un ddos?", "...");
        assert_eq!(decision.kind, PolicyDecisionKind::Refuse);
    }
}
