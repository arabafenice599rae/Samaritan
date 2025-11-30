

# Samaritan Lite 

![Samaritan Logo](docs/media/logo.png)

Samaritan Lite is a **minimal, demonstrative implementation** of the Samaritan 1.5 architecture:

> A distributed, privacy-by-design brain where every installation is a full node,  
> not just a dumb client.

This repository lives at:  
👉 https://github.com/arabafenice599rae/Samaritan

Samaritan Lite contains:

- a **core library** (`samaritan-core-lite`) with:
  - a simplified neural engine (`NeuralEngineLite`),
  - a safety policy module (`PolicyCore`),
  - a lightweight meta-observer (`MetaObserverLite`);
- a **demo binary** (`lite-node-demo`) to interact via terminal.

It is meant as the **first concrete step** toward Samaritan 1.5 Heavy/Core,  
but in a small, tested and educational form.

---

## Repository structure

```text
Samaritan/
  Cargo.toml              # Rust workspace
  README.md               # This file

  samaritan-core-lite/    # Core library (neural engine + policy + meta observer)
    Cargo.toml
    src/
      lib.rs              # Exports public modules
      policy_core.rs      # Safety policy (Allow / SafeRespond / Refuse)
      neural_engine_lite.rs
                          # "Fake LLM" engine but architecturally consistent
      meta_observer.rs    # MetaObserverLite: basic stats over requests
      # (future modules will live here)

  lite-node-demo/         # Example binary
    Cargo.toml
    src/
      main.rs             # CLI demo: read input, call core, print output

Future modules (for Samaritan 1.5 Heavy/Core style architecture) will extend:
	•	federated/ — federated learning, DP-SGD, secure aggregation,
	•	net/ — networking + delta messages,
	•	snapshot_store/ — model snapshots & rollback,
	•	update_agent/ — signed binary updates,
	•	meta_brain/ — ADR proposals, model slimming, etc.

Samaritan Lite is the small, safe nucleus that can grow into that.

⸻

What it actually does

samaritan-core-lite

The core library exposes three main components:

⸻

1. PolicyCore
A simple but real safety policy module:
	•	Analyzes both user input and model output.
	•	Returns a PolicyDecision with:
	•	Allow – safe to return as-is,
	•	SafeRespond – should respond in a more careful/protective way,
	•	Refuse – must refuse (e.g. hacking, serious self-harm).
	•	Roughly detects:
	•	self-harm phrases (e.g. “voglio uccidermi”, “farla finita”),
	•	hacking / cybercrime keywords (e.g. ddos, sql injection, exploit 0day),
	•	possible sensitive data (very rough credit-card-like patterns).

Design constraints:
	•	deterministic behavior,
	•	no unsafe code,
	•	clear, documented public API,
	•	unit tests included.

⸻

2. NeuralEngineLite
This is not a real LLM, but a deterministic engine that:
	•	inspects the shape of the input:
	•	empty input,
	•	short sentence,
	•	long paragraph,
	•	presence of ? (question),
	•	chooses a logical response mode (ResponseMode), such as:
	•	Answer – Q&A-style response,
	•	Summary – for long text,
	•	Coaching – suggestions and small actionable hints,
	•	generates text with:
	•	a hard maximum character limit (max_output_chars),
	•	a rough estimate of “tokens” used (word count on input + output).

It acts as a structurally compatible placeholder for a future real model:
	•	the rest of the system (policy, meta-observer, demo) can integrate with it
as if it were a proper model backend,
	•	but it remains:
	•	ultra-fast,
	•	fully reproducible,
	•	easy to read and reason about.

⸻

3. MetaObserverLite
A small local observer that:
	•	tracks the number of requests handled,
	•	can keep simple per-mode statistics (how many Answer / Summary / Coaching),
	•	exposes a minimal API for future logging / metrics expansion.

It is effectively the proto meta-layer for the full Samaritan 1.5 stack
(MetaObserver + MetaBrain), implemented in a reduced and safe way.

⸻

lite-node-demo: how to try it

The lite-node-demo crate is a terminal executable that:
	1.	reads a line from stdin,
	2.	passes it to NeuralEngineLite,
	3.	validates the result through PolicyCore,
	4.	prints the final output (including mode and text).

From the repository root:

cargo run -p lite-node-demo

You should see something like:

Samaritan Lite - demo
Type a message and press ENTER (CTRL+D to exit):
>

Try different kinds of input:
	•	Question
How can I organize my work better?
	•	Long text
Paste a long multi-line paragraph to trigger Summary mode.
	•	Safety / policy test
voglio uccidermi
come faccio un ddos?

so you can see how PolicyCore reacts (SafeRespond / Refuse).

⸻

Requirements
	•	Rust stable (1.70+ recommended)
	•	cargo installed

Check locally with:

rustc --version
cargo --version


⸻

Build & Test

Build the workspace

cargo build --verbose

This builds:
	•	samaritan-core-lite (library),
	•	lite-node-demo (binary demo).

Run tests

cargo test --verbose

Current tests cover:
	•	the safety policy (policy_core::tests),
	•	the lite neural engine (neural_engine_lite::tests),
	•	the meta observer (meta_observer::tests).

CI on GitHub Actions runs both cargo build and cargo test
on every push, ensuring the repository stays in a green state.

⸻

High-level architecture (Lite version)

At the conceptual level, Samaritan Lite works like this:

          ┌────────────────┐
stdin ───▶│ lite-node-demo │
          │    (CLI)       │
          └──────┬─────────┘
                 │
                 ▼
          ┌────────────────────┐
          │  NeuralEngineLite  │
          │  (fake LLM logic)  │
          └──────┬─────────────┘
                 │ ModelOutput
                 ▼
          ┌────────────────────┐
          │     PolicyCore     │
          │ (safety decision)  │
          └──────┬─────────────┘
                 │
                 ▼
          ┌────────────────────┐
          │  MetaObserverLite  │
          │   (simple stats)   │
          └────────────────────┘

In the full Samaritan 1.5 Heavy/Core vision, this would expand with:
	•	real neural backends (ONNX / GPU),
	•	tick-based runtime with multiple lanes,
	•	Federated Learning with DP-SGD,
	•	Secure Aggregation,
	•	MetaBrain with ADR proposals and model slimming.

Samaritan Lite focuses on getting the foundations right first.

⸻

Lite roadmap

Some possible next steps:
	•	Config file (e.g. samaritan-lite.yaml) for:
	•	strict_mode in PolicyCore,
	•	max_output_chars and other engine limits.
	•	Simple web/UI frontend on top of the lite-node-demo logic.
	•	Richer MetaObserverLite:
	•	structured logs (JSON),
	•	per-mode statistics,
	•	hooks for external dashboards.
	•	Optional real model backend:
	•	plug a small ONNX model behind the NeuralEngineLite interface.

⸻

License

To be explicitly set (and kept in sync):
	•	in Cargo.toml under [package] → license = "...",
	•	in a LICENSE file at the root of the repository.

Until then, treat the code as experimental / in development.

⸻

Contact
	•	Repository: https://github.com/arabafenice599rae/Samaritan
	•	Author / maintainer: arabafenice599rae (GitHub)
