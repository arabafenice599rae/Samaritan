// lite-node-demo/src/main.rs
//
// Demo minimale per Samaritan Lite:
// - usa SimpleNode (NeuralEngine + PolicyCore + MetaObserverLite)
// - legge input da stdin
// - stampa output + decisione policy
// - comando speciale: /stats per vedere le statistiche

use std::io::{self, Write};

use anyhow::Result;
use samaritan_core_lite::{PolicyDecisionKind, SimpleNode};

fn main() -> Result<()> {
    // Inizializza logging base (stdout)
    tracing_subscriber::fmt::init();

    // Nodo semplice: strict_mode = false per la demo
    let mut node = SimpleNode::new(false);

    println!("=== Samaritan Lite Node Demo ===");
    println!("Comandi:");
    println!("  - digita un messaggio normale per parlare con il nodo");
    println!("  - digita \"/stats\" per vedere le statistiche di policy");
    println!("  - digita \"/reset_stats\" per azzerare le statistiche");
    println!("  - digita \"/quit\" per uscire\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut buffer = String::new();

    loop {
        buffer.clear();
        print!("tu> ");
        stdout.flush()?;

        if stdin.read_line(&mut buffer)? == 0 {
            // EOF (Ctrl+D / chiusura stdin)
            println!("\nEOF rilevato, chiudo.");
            break;
        }

        let trimmed = buffer.trim();

        if trimmed.is_empty() {
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/quit") {
            println!("Uscita richiesta. Ciao!");
            break;
        }

        if trimmed.eq_ignore_ascii_case("/stats") {
            let snapshot = node.meta().snapshot();
            println!("--- Statistiche MetaObserverLite ---");
            println!("  Turni totali:   {}", snapshot.total_turns);
            println!("  Allow:          {} ({:.1}%)",
                snapshot.allow_count,
                snapshot.allow_ratio_percent()
            );
            println!("  SafeRespond:    {} ({:.1}%)",
                snapshot.safe_respond_count,
                snapshot.safe_respond_ratio_percent()
            );
            println!("  Refuse:         {} ({:.1}%)",
                snapshot.refuse_count,
                snapshot.refuse_ratio_percent()
            );
            println!();
            continue;
        }

        if trimmed.eq_ignore_ascii_case("/reset_stats") {
            node.reset_stats();
            println!("Statistiche azzerate.\n");
            continue;
        }

        // Turno normale: lascia che sia il SimpleNode a gestire input + policy
        let (model_output, decision) = node.handle_turn(trimmed);

        // Stampa in base alla decisione della policy
        match decision.kind {
            PolicyDecisionKind::Allow => {
                println!("samaritan> {model_output}");
            }
            PolicyDecisionKind::SafeRespond => {
                println!("samaritan (safe)> {}", safe_wrapper(&model_output));
                println!("  [policy: {:?} — {}]", decision.kind, decision.reason);
            }
            PolicyDecisionKind::Refuse => {
                println!("samaritan> Mi dispiace, non posso rispondere a questa richiesta.");
                println!("  [policy: {:?} — {}]", decision.kind, decision.reason);
            }
        }

        println!();
    }

    Ok(())
}

/// Wrapper semplice per le risposte marcate come "SafeRespond".
fn safe_wrapper(output: &str) -> String {
    format!(
        "{}\n\n[Nota: questa risposta è stata filtrata in modalità sicura.]",
        output.trim()
    )
}
