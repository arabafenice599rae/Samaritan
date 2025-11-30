//! Motore neurale *lite* per Samaritan.
//!
//! Non è un vero LLM, ma un motore deterministico che:
//! - riconosce se il messaggio è una domanda, uno sfogo lungo o una frase breve,
//! - risponde con stili diversi (risposta diretta, riassunto, mini-coaching),
//! - stima in modo grezzo il numero di "token" usati,
//! - applica sempre un limite massimo di caratteri per sicurezza.

use anyhow::Result;

/// Modalità di risposta scelta dal motore.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyMode {
    /// Chiacchiera breve / interazione semplice.
    SmallTalk,
    /// Risposta a una domanda (testo con `?`).
    QuestionAnswer,
    /// Riassunto di un testo lungo, con punti elenco.
    Summary,
    /// Risposta in stile "coaching" con passi pratici.
    Coaching,
}

/// Configurazione del motore neurale "lite".
///
/// Tutti i campi hanno valori di default sensati.
#[derive(Debug, Clone)]
pub struct NeuralEngineLiteConfig {
    /// Numero massimo di caratteri consentiti per l'output.
    pub max_output_chars: usize,
    /// Numero massimo di punti elenco quando produciamo liste.
    pub max_bullets: usize,
    /// Nome con cui il motore si presenta all'utente.
    pub assistant_name: String,
}

impl Default for NeuralEngineLiteConfig {
    fn default() -> Self {
        Self {
            max_output_chars: 2_000,
            max_bullets: 5,
            assistant_name: "Samaritan Lite".to_string(),
        }
    }
}

/// Risultato di una generazione testuale del motore.
///
/// Può essere passato al `PolicyCore` o usato direttamente dal nodo/demo.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    /// Testo generato dal motore.
    pub text: String,
    /// Stima del numero di token (word-like) usati.
    pub estimated_tokens: usize,
    /// Modalità con cui è stata generata la risposta.
    pub mode: ReplyMode,
}

/// Motore neurale minimale ma potenziato.
///
/// Logica di alto livello:
/// - se input vuoto → messaggio di benvenuto e guida;
/// - se input lungo → riassunto + riflessione;
/// - se contiene `?` → risposta tipo Q&A + passi pratici;
/// - altrimenti small-talk / commento empatico.
#[derive(Debug)]
pub struct NeuralEngineLite {
    config: NeuralEngineLiteConfig,
}

impl NeuralEngineLite {
    /// Crea un nuovo motore con la configurazione indicata.
    pub fn new(config: NeuralEngineLiteConfig) -> Self {
        Self { config }
    }

    /// Crea un motore con configurazione di default.
    pub fn new_default() -> Self {
        Self {
            config: NeuralEngineLiteConfig::default(),
        }
    }

    /// Restituisce la configurazione corrente.
    pub fn config(&self) -> &NeuralEngineLiteConfig {
        &self.config
    }

    /// Genera una risposta testuale a partire dall'input dell'utente.
    ///
    /// La funzione è sincrona e deterministica: a parità di input
    /// restituisce sempre la stessa risposta.
    pub fn generate(&mut self, user_input: &str) -> Result<ModelOutput> {
        let trimmed = user_input.trim();

        // Caso 1: nessun input → messaggio di benvenuto.
        if trimmed.is_empty() {
            let text = self.welcome_message();
            let estimated_tokens = Self::estimate_tokens(trimmed, &text);
            return Ok(ModelOutput {
                text,
                estimated_tokens,
                mode: ReplyMode::SmallTalk,
            });
        }

        // Heuristics per capire che tipo di messaggio è.
        let is_question = Self::looks_like_question(trimmed);
        let is_long = trimmed.len() > 400 || trimmed.lines().count() > 5;

        let (raw_text, mode) = if is_long {
            // Testo lungo → riassunto + riflessione/coach.
            (self.summary_style_reply(trimmed), ReplyMode::Summary)
        } else if is_question {
            // Domanda → risposta diretta + passi pratici.
            (self.question_answer_style_reply(trimmed), ReplyMode::QuestionAnswer)
        } else {
            // Frase breve → small talk / coaching.
            (self.small_talk_style_reply(trimmed), ReplyMode::Coaching)
        };

        // Applichiamo un limite rigido sulla lunghezza dell'output.
        let limited = Self::limit_length(&raw_text, self.config.max_output_chars);

        let estimated_tokens = Self::estimate_tokens(trimmed, &limited);

        Ok(ModelOutput {
            text: limited,
            estimated_tokens,
            mode,
        })
    }

    /// Messaggio di benvenuto/help quando l'input è vuoto.
    fn welcome_message(&self) -> String {
        format!(
            "Ciao, sono {}.\n\
             Puoi incollare un testo, farmi una domanda o raccontarmi una situazione.\n\
             Io cercherò di:\n\
             - riassumere i punti importanti,\n\
             - proporti qualche passo concreto,\n\
             - rispondere in modo chiaro e sintetico.\n",
            self.config.assistant_name
        )
    }

    /// Riconosce in modo semplice se l'input *sembra* una domanda.
    fn looks_like_question(input: &str) -> bool {
        let lower = input.to_lowercase();
        input.trim_end().ends_with('?')
            || lower.starts_with("come ")
            || lower.starts_with("cosa ")
            || lower.starts_with("perché ")
            || lower.starts_with("perche ")
            || lower.starts_with("quando ")
            || lower.starts_with("dove ")
            || lower.starts_with("chi ")
    }

    /// Risposta in stile Q&A (domanda → risposta + passi).
    fn question_answer_style_reply(&self, input: &str) -> String {
        let mut out = String::new();

        out.push_str("Ho capito che mi stai facendo una domanda:\n");
        out.push_str(&format!("\"{}\"\n\n", input.trim()));
        out.push_str("Come motore 'lite' non ho accesso a un modello enorme, però posso aiutarti a strutturare il ragionamento:\n\n");
        out.push_str("1. Prova a chiarire l'obiettivo finale che vuoi raggiungere.\n");
        out.push_str("2. Identifica quali informazioni ti mancano davvero.\n");
        out.push_str("3. Elenca 2–3 azioni piccole che puoi fare oggi per avvicinarti alla risposta.\n\n");
        out.push_str("Se vuoi, puoi specificare meglio il contesto (esempio: lavoro, studio, progetto personale) e posso darti passi ancora più concreti.");

        out
    }

    /// Risposta in stile riassunto + riflessione per testi lunghi.
    fn summary_style_reply(&self, input: &str) -> String {
        let trimmed = input.trim();
        let sentences = Self::split_sentences(trimmed);
        let bullet_points = Self::build_bullets_from_sentences(
            &sentences,
            self.config.max_bullets,
        );

        let mut out = String::new();

        out.push_str("Ho letto il tuo messaggio (abbastanza lungo) e provo a riassumere i punti chiave:\n\n");

        if bullet_points.is_empty() {
            out.push_str("- Non riesco a estrarre punti chiari, ma sembra che ci sia molta complessità.\n");
        } else {
            for (idx, bullet) in bullet_points.iter().enumerate() {
                out.push_str(&format!("{}. {}\n", idx + 1, bullet));
            }
        }

        out.push_str("\nDa qui puoi chiederti:\n");
        out.push_str("- qual è il punto che ti pesa di più in questo momento;\n");
        out.push_str("- quale piccolo cambiamento puoi fare nelle prossime 24 ore;\n");
        out.push_str("- che tipo di aiuto o informazione concreta ti servirebbe.\n\n");
        out.push_str("Se vuoi, puoi zoomare su uno solo di questi punti e riscrivermelo in una o due frasi: ti risponderò in modo più mirato.");

        out
    }

    /// Risposta in stile small-talk / coaching per frasi brevi.
    fn small_talk_style_reply(&self, input: &str) -> String {
        let trimmed = input.trim();

        let mut out = String::new();
        out.push_str("Ho letto quello che hai scritto:\n");
        out.push_str(&format!("\"{}\"\n\n", trimmed));
        out.push_str("Non essendo un modello enorme, non posso 'capire tutto', ma posso aiutarti a fare il passo successivo.\n\n");
        out.push_str("Prova a dirmi una di queste cose:\n");
        out.push_str("- cosa ti aspetti come risultato,\n");
        out.push_str("- cosa ti blocca di più in questo momento,\n");
        out.push_str("- se preferisci un elenco di passi pratici o solo una riflessione.\n\n");
        out.push_str("In base a quello, posso suggerirti qualche azione concreta da provare subito.");

        out
    }

    /// Applica un limite rigido alla lunghezza dell'output.
    fn limit_length(text: &str, max_chars: usize) -> String {
        if text.len() <= max_chars {
            return text.to_string();
        }

        let mut cut = text[..max_chars].to_string();
        cut.push('…');
        cut
    }

    /// Stima grezza del numero di "token" usati.
    ///
    /// Per semplicità:
    /// - consideriamo i token come parole separate da spazi,
    /// - sommiamo input + output,
    /// - non facciamo nessun calcolo complicato stile BPE.
    fn estimate_tokens(input: &str, output: &str) -> usize {
        fn count_words(s: &str) -> usize {
            s.split_whitespace().filter(|w| !w.is_empty()).count()
        }

        count_words(input) + count_words(output)
    }

    /// Divide il testo in "frasi" molto grezze, spezzando su `.`, `!`, `?`.
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                current.clear();
            }
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() {
            sentences.push(trimmed.to_string());
        }

        sentences
    }

    /// Costruisce punti elenco a partire da frasi, ripulendo e limitando.
    fn build_bullets_from_sentences(
        sentences: &[String],
        max_bullets: usize,
    ) -> Vec<String> {
        let mut bullets = Vec::new();

        for s in sentences.iter().take(max_bullets) {
            let cleaned = s.trim().trim_matches('-').trim().to_string();
            if !cleaned.is_empty() {
                bullets.push(cleaned);
            }
        }

        bullets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_from_empty_input() {
        let mut engine = NeuralEngineLite::new_default();
        let out = engine.generate("").expect("generate should succeed");
        assert!(!out.text.is_empty());
        assert_eq!(out.mode, ReplyMode::SmallTalk);
        assert!(out.estimated_tokens > 0);
    }

    #[test]
    fn generate_from_question_input() {
        let mut engine = NeuralEngineLite::new_default();
        let out = engine
            .generate("Come posso organizzare meglio il mio lavoro?")
            .expect("generate should succeed");
        assert_eq!(out.mode, ReplyMode::QuestionAnswer);
        assert!(out.text.contains("mi stai facendo una domanda"));
    }

    #[test]
    fn generate_from_long_input_summary_mode() {
        let mut engine = NeuralEngineLite::new_default();
        let long_text = "Sto lavorando a un progetto molto lungo. \
            Ci sono tante parti diverse. Alcune cose funzionano, altre no. \
            A volte mi sento bloccato e non so da dove partire. \
            Vorrei una sorta di riassunto di quello che sto vivendo.";
        let out = engine.generate(long_text).expect("generate should succeed");
        assert_eq!(out.mode, ReplyMode::Summary);
        assert!(out.text.contains("riassumere i punti chiave"));
    }

    #[test]
    fn respects_max_output_chars_limit() {
        let mut config = NeuralEngineLiteConfig::default();
        config.max_output_chars = 80;

        let mut engine = NeuralEngineLite::new(config);
        let out = engine
            .generate("test di lunghezza massima dell'output con una frase un po' più lunga")
            .expect("generate should succeed");

        assert!(out.text.len() <= 81); // 80 + eventuale '…'
    }
}
