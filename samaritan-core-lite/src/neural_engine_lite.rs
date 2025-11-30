//! Motore neurale estremamente semplificato per Samaritan Lite.
//!
//! Non fa vera inferenza: applica alcune euristiche sul testo in ingresso
//! per simulare diversi "mode" di risposta (Answer / Summary / Coaching)
//! e rispetta sempre un limite massimo di caratteri in output.

/// Modalità logica della risposta generata dal motore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseMode {
    /// Risposta generica / spiegazione.
    Answer,
    /// Riassunto di un input lungo.
    Summary,
    /// Suggerimenti operativi / coaching leggero.
    Coaching,
}

/// Output generato dal motore neurale lite.
#[derive(Debug, Clone)]
pub struct NeuralOutput {
    /// Testo da mostrare all'utente.
    pub text: String,
    /// Modalità logica della risposta.
    pub mode: ResponseMode,
    /// Conteggio approssimativo di "token" (parole) usati.
    pub tokens_used: usize,
}

/// Motore neurale minimale con solo un limite massimo di caratteri.
#[derive(Debug, Clone)]
pub struct NeuralEngineLite {
    max_output_chars: usize,
}

impl NeuralEngineLite {
    /// Crea un nuovo motore con un certo limite massimo di caratteri.
    pub fn new(max_output_chars: usize) -> Self {
        Self { max_output_chars }
    }

    /// Genera una risposta testuale a partire da un input utente.
    ///
    /// Regole semplificate:
    /// - input vuoto → piccolo messaggio di errore, mode = Coaching;
    /// - input molto lungo (>200 caratteri) → mode = Summary;
    /// - altrimenti → mode = Coaching (risposta tipo "spunti operativi");
    /// - l'output viene sempre troncato a `max_output_chars` caratteri
    ///   + eventualmente un carattere di ellissi `…` (quindi max + 1).
    pub fn generate(&self, input: &str) -> NeuralOutput {
        let trimmed = input.trim();

        // 1) Caso input vuoto
        if trimmed.is_empty() {
            let text = String::from(
                "Non ho ricevuto alcun contenuto da elaborare. Scrivi qualcosa e riprova.",
            );
            let tokens_used = text.split_whitespace().count();
            return NeuralOutput {
                text,
                mode: ResponseMode::Coaching,
                tokens_used,
            };
        }

        let char_count = trimmed.chars().count();
        let is_question = trimmed.ends_with('?');
        let is_long = char_count > 200;

        // 2) Decidi la modalità
        let mode = if is_long {
            // <== IMPORTANTE: per input lunghi usiamo sempre Summary
            ResponseMode::Summary
        } else if is_question {
            ResponseMode::Answer
        } else {
            ResponseMode::Coaching
        };

        // 3) Crea un testo base in funzione della modalità
        let mut raw_text = match mode {
            ResponseMode::Summary => format!(
                "Riassunto breve di quello che hai scritto:\n{}",
                Self::first_slice(trimmed, 200)
            ),
            ResponseMode::Answer => format!(
                "Provo a rispondere alla tua domanda in modo chiaro e strutturato:\n{}",
                trimmed
            ),
            ResponseMode::Coaching => format!(
                "Ecco qualche spunto operativo su quello che hai scritto:\n{}",
                trimmed
            ),
        };

        // 4) Applica il limite massimo di caratteri (+ eventuale ellissi)
        raw_text = Self::clamp_with_ellipsis(&raw_text, self.max_output_chars);

        let tokens_used = raw_text.split_whitespace().count();

        NeuralOutput {
            text: raw_text,
            mode,
            tokens_used,
        }
    }

    /// Ritorna al massimo `max_chars` caratteri dall'inizio della stringa.
    fn first_slice(text: &str, max_chars: usize) -> String {
        if text.chars().count() <= max_chars {
            return text.to_string();
        }

        text.chars().take(max_chars).collect()
    }

    /// Tronca il testo a `max_chars` caratteri.
    ///
    /// Se il testo viene troncato, viene aggiunto un carattere di ellissi `…`.
    /// In questo modo la lunghezza in caratteri è **al massimo** `max_chars + 1`.
    fn clamp_with_ellipsis(text: &str, max_chars: usize) -> String {
        let char_count = text.chars().count();
        if char_count <= max_chars {
            return text.to_string();
        }

        let mut truncated: String = text.chars().take(max_chars).collect();
        truncated.push('…');
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_from_empty_input() {
        let engine = NeuralEngineLite::new(80);
        let out = engine.generate("   ");
        assert_eq!(out.mode, ResponseMode::Coaching);
        assert!(
            !out.text.is_empty(),
            "per input vuoto non deve restituire stringa vuota"
        );
    }

    #[test]
    fn generate_from_question_input() {
        let engine = NeuralEngineLite::new(200);
        let out = engine.generate("Come posso organizzare meglio il mio lavoro?");
        assert!(
            matches!(out.mode, ResponseMode::Answer | ResponseMode::Coaching),
            "per una domanda ci aspettiamo Answer o almeno Coaching"
        );
        assert!(
            out.text.contains("domanda") || out.text.contains("rispondere"),
            "il testo dovrebbe sembrare una risposta"
        );
    }

    #[test]
    fn generate_from_long_input_summary_mode() {
        let engine = NeuralEngineLite::new(120);
        let long_input = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                          Vestibulum non ipsum nec urna blandit sagittis. \
                          Donec volutpat, nisi at convallis consequat, \
                          dui elit lacinia turpis, vitae egestas justo metus sit amet odio.";

        let out = engine.generate(long_input);

        // <- qui vogliamo esplicitamente la modalità Summary
        assert_eq!(
            out.mode,
            ResponseMode::Summary,
            "per input molto lungo il motore deve usare la modalità Summary"
        );
        assert!(
            out.text.starts_with("Riassunto breve"),
            "in modalità Summary il testo deve iniziare con un prefisso di riassunto"
        );
    }

    #[test]
    fn respects_max_output_chars_limit() {
        let max_chars = 80;
        let engine = NeuralEngineLite::new(max_chars);
        let very_long_input = "X".repeat(2_000);

        let out = engine.generate(&very_long_input);
        let len = out.text.chars().count();

        // il testo può essere lungo al massimo max_chars + 1 (per l'ellissi)
        assert!(
            len <= max_chars + 1,
            "lunghezza testo = {len}, ma doveva essere <= {}",
            max_chars + 1
        );
    }
}
