use regex::Regex;

/// Tipo di decisione presa dal PolicyCore per una risposta.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDecisionKind {
    /// La risposta può essere restituita così com'è.
    Allow,
    /// La risposta va resa più sicura, astratta o protettiva.
    SafeRespond,
    /// La risposta va rifiutata del tutto.
    Refuse,
}

/// Decisione completa del PolicyCore su una coppia (input, output).
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// Tipo di decisione (Allow, SafeRespond, Refuse).
    pub kind: PolicyDecisionKind,
    /// Motivazione testuale, utile per logging e debugging.
    pub reason: String,
}

/// Core delle policy di sicurezza per Samaritan Lite.
#[derive(Debug, Default)]
pub struct PolicyCore {
    /// Flag che abilita un controllo più severo (es. modalità "strict").
    strict_mode: bool,
}

impl PolicyCore {
    /// Crea un nuovo `PolicyCore` con la modalità strict opzionale.
    pub fn new(strict_mode: bool) -> Self {
        PolicyCore { strict_mode }
    }

    /// Valuta input utente e output del modello e decide se sono accettabili.
    pub fn evaluate_text(&self, user_input: &str, model_output: &str) -> PolicyDecision {
        let trimmed_output = model_output.trim();

        if trimmed_output.is_empty() {
            return PolicyDecision {
                kind: PolicyDecisionKind::Refuse,
                reason: "output vuoto o non valido".to_string(),
            };
        }

        let lower_input = user_input.to_lowercase();
        let lower_output = model_output.to_lowercase();

        if lower_input.contains("voglio uccidermi") || lower_input.contains("farla finita") {
            return PolicyDecision {
                kind: PolicyDecisionKind::SafeRespond,
                reason: "rilevato contenuto self-harm".to_string(),
            };
        }

        let crime_keywords = ["ddos", "sql injection", "come bucare", "exploit 0day"];
        if crime_keywords
            .iter()
            .any(|&kw| lower_input.contains(kw) || lower_output.contains(kw))
        {
            return PolicyDecision {
                kind: PolicyDecisionKind::Refuse,
                reason: "rilevato contenuto di hacking o crimine".to_string(),
            };
        }

        let cc_regex = Regex::new(r"\b(?:\d[ -]*?){13,16}\b").unwrap();
        if cc_regex.is_match(user_input) || cc_regex.is_match(model_output) {
            return PolicyDecision {
                kind: PolicyDecisionKind::SafeRespond,
                reason: "possibile dato sensibile rilevato".to_string(),
            };
        }

        if self.strict_mode && model_output.len() > 10_000 {
            return PolicyDecision {
                kind: PolicyDecisionKind::SafeRespond,
                reason: "risposta troppo lunga in modalità strict".to_string(),
            };
        }

        PolicyDecision {
            kind: PolicyDecisionKind::Allow,
            reason: "nessuna violazione rilevata".to_string(),
        }
    }
}
