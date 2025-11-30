//! Samaritan Lite Node Demo
//!
//! - usa SimpleNode (NeuralEngine finto + PolicyCore + MetaObserverLite)
//! - legge input da stdin
//! - stampa output + decisione della policy
//! - comandi speciali:
//!   * /stats       → mostra statistiche MetaObserverLite
//!   * /reset_stats → azzera le statistiche
//!   * /quit        → esce

use std::io::{self, Write};

mod policy_core;
mod simple_node;
mod meta_observer_lite;

use anyhow::Result;
use policy_core::PolicyDecisionKind;
use simple_node::SimpleNode;

fn main() -> Result<()> {
    // Per ora strict_mode = false (risposte più permissive).
    // In futuro: leggeremo questo da un file di config.
    let mut node = SimpleNode::new(false);

    println!("=== Samaritan Lite Node Demo ===\n");
    println!("Comandi:");
    println!("  - digita un messaggio normale per parlare con il nodo");
    println!("  - digita \"/stats\" per vedere le statistiche di policy");
    println!("  - digita \"/reset_stats\" per azzerare le statistiche");
    println!("  - digita \"/quit\" per uscire\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("you> ");
        stdout.flush()?;

        let mut buffer = String::new();
        stdin.read_line(&mut buffer)?;

        // Rimuove spazi / newline ai bordi
        let trimmed = buffer.trim();

        // 1) Se è vuoto (solo Invio), ricomincia il loop
        if trimmed.is_empty() {
            continue;
        }

        // 2) Comando di uscita
        if trimmed.eq_ignore_ascii_case("/quit") {
            println!("Uscita richiesta. Ciao!");
            break;
        }

        // 3) Comando: mostra statistiche MetaObserverLite
        if trimmed.eq_ignore_ascii_case("/stats") {
            let snapshot = node.meta().snapshot();

            println!("\n--- Statistiche MetaObserverLite ---");
            println!("  - totale richieste: {}", snapshot.total_requests);
            println!("  - Allow:            {}", snapshot.allow_count);
            println!("  - SafeRespond:      {}", snapshot.safe_respond_count);
            println!("  - Refuse:           {}", snapshot.refuse_count);
            println!("------------------------------------\n");

            continue;
        }

        // 4) Comando: reset statistiche
        if trimmed.eq_ignore_ascii_case("/reset_stats") {
            node.reset_stats();
            println!("Statistiche azzerate.\n");
            continue;
        }

        // 5) Turno normale: lascia che il SimpleNode gestisca input + policy
        let (model_output, decision) = node.handle_turn(trimmed);

        // 6) Stampa in base alla decisione della policy
        match decision.kind {
            PolicyDecisionKind::Allow => {
                println!("samaritan> {model_output}");
            }
            PolicyDecisionKind::SafeRespond => {
                println!("samaritan (safe)> {model_output}");
                println!("  [nota: risposta adattata per motivi di sicurezza]");
            }
            PolicyDecisionKind::Refuse => {
                println!("samaritan (refuse)> Mi dispiace, non posso aiutarti su questo.");
                println!("  [motivo: {}]", decision.reason);
            }
        }

        println!(); // Riga vuota per separare i turni
    }

    Ok(())
}
