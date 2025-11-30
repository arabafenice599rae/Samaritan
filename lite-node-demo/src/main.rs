#![forbid(unsafe_code)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

//! Eseguibile di demo per Samaritan Lite 0.1.
//!
//! Usa il `LiteNode` definito nel crate `samaritan-core` con un
//! modello finto che fa solo echo dell'input. Serve per verificare
//! il flusso completo: input → modello → PolicyCore → output.

use std::error::Error;
use std::io::{self, Write};
use std::sync::Arc;

use samaritan_core::{LiteNode, TextModel};
use samaritan_core::policy_core::PolicyCore;

/// Modello di esempio estremamente semplice:
/// restituisce sempre `echo: <prompt>`.
struct EchoModel;

impl TextModel for EchoModel {
    fn generate(&self, prompt: &str) -> anyhow::Result<String> {
        Ok(format!("echo: {prompt}"))
    }
}

/// Entry-point dell'eseguibile di demo.
///
/// Avvia un piccolo REPL:
/// - legge una riga da stdin,
/// - la passa al `LiteNode`,
/// - stampa output + decisione di policy.
/// Digita `exit` o `quit` per uscire.
fn main() -> Result<(), Box<dyn Error>> {
    // Logging minimale (userai `tracing_subscriber` se vuoi vedere log avanzati).
    // Per ora non inizializziamo nulla: la demo stampa solo su stdout/stderr.

    // Modalità non strict per iniziare.
    let policy = PolicyCore::new(false);
    let model = Arc::new(EchoModel);
    let node = LiteNode::new(policy, model);

    println!("Samaritan Lite 0.1 — demo CLI");
    println!("Scrivi un messaggio e premi invio.");
    println!("Digita 'exit' o 'quit' per uscire.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut buffer = String::new();

    loop {
        buffer.clear();
        print!("> ");
        stdout.flush()?;

        let bytes_read = stdin.read_line(&mut buffer)?;
        if bytes_read == 0 {
            // EOF (Ctrl+D / fine input)
            println!("\nEOF rilevato, esco.");
            break;
        }

        let line = buffer.trim();
        if line.eq_ignore_ascii_case("exit") || line.eq_ignore_ascii_case("quit") {
            println!("Ciao! 👋");
            break;
        }

        let response = match node.handle_request(line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Errore durante la gestione della richiesta: {e:?}");
                continue;
            }
        };

        println!("---");
        println!("Decisione policy: {:?}", response.decision.kind);
        println!("Motivo: {}", response.decision.reason);
        println!("Risposta:\n{}\n", response.output);
    }

    Ok(())
}
