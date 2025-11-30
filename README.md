
<p align="center">
  <img src="logo.svg" alt="Samaritan Lite Logo" width="220" />
</p>

<h1 align="center">Samaritan Lite</h1>

<p align="center">
  A tiny, opinionated playground for the Samaritan&nbsp;1.5 distributed brain.
</p>

<p align="center">
  <a href="https://github.com/arabafenice599rae/Samaritan/actions/workflows/build.yml">
    <img src="https://github.com/arabafenice599rae/Samaritan/actions/workflows/build.yml/badge.svg" alt="Build status" />
  </a>
  <img src="https://img.shields.io/badge/status-experimental-orange.svg" alt="Status: experimental" />
  <img src="https://img.shields.io/badge/language-Rust%202021-DEA584.svg" alt="Rust 2021" />
</p>

---

## ✨ What is Samaritan Lite?

Samaritan Lite is a **minimal, demonstrative implementation** of the Samaritan 1.5 architecture:

> A distributed, privacy-by-design brain where every installation is a full node,  
> not just a dumb client.

This repository lives at:

👉 <https://github.com/arabafenice599rae/Samaritan>

Samaritan Lite focuses on a **tiny but clean core** that is easy to read, test and extend.

---

## 🧩 Repository layout

This is a Cargo workspace with two members:

```text
Samaritan/
├── Cargo.toml                # workspace
├── samaritan-core-lite/      # core library
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── neural_engine_lite.rs
│       ├── policy_core.rs
│       └── meta_observer.rs
└── lite-node-demo/           # demo binary
    ├── Cargo.toml
    └── src/
        ├── main.rs
        ├── simple_node.rs
        ├── policy_core.rs
        └── meta_observer_lite.rs

samaritan-core-lite (library)

The core library currently contains:
	•	a simplified neural engine (NeuralEngineLite) that:
	•	classifies input as question / long text / short text,
	•	generates deterministic answers (Q&A, summary, or light coaching),
	•	enforces a max output length for safety.
	•	a safety policy module (PolicyCore) that:
	•	inspects user input and model output,
	•	flags obvious self-harm and crime / hacking patterns,
	•	crudely detects possible sensitive numbers (e.g. credit card-like strings),
	•	can run in a more conservative strict mode.
	•	a tiny meta-observer (MetaObserverLite) that:
	•	tracks number of turns, average length of messages,
	•	is meant as a playground for future MetaBrain ideas.

lite-node-demo (binary)

The demo binary wraps all of this into a small interactive node:
	•	reads text from stdin,
	•	sends it through NeuralEngineLite + PolicyCore,
	•	prints the model output together with the policy decision,
	•	exposes a few commands (for example, /stats to inspect metrics).

⸻

🚀 Getting started

1. Prerequisites
	•	Rust (stable) with edition 2021 support
You can install it via rustup￼.

2. Clone the repo

git clone https://github.com/arabafenice599rae/Samaritan.git
cd Samaritan

3. Build the whole workspace

cargo build --workspace

4. Run tests

cargo test --workspace

5. Run the demo node

From the repository root:

cargo run -p lite-node-demo

You should see something like:

=== Samaritan Lite Node Demo ===
Commands:
  - type a normal message to talk with the node
  - type "/stats" to inspect metrics
  - type "/reset_stats" to reset metrics
  - type "/quit" to exit


⸻

🛠 Development workflow

Some useful commands while hacking on Samaritan Lite:

# Format all Rust code
cargo fmt --all

# Run Clippy with warnings as errors
cargo clippy --all-targets --all-features -- -D warnings

# Run tests only for the core library
cargo test -p samaritan-core-lite

# Run the demo only
cargo run -p lite-node-demo

CI is configured to run build + tests on every push.

⸻

🧭 Roadmap (Lite)

Samaritan Lite is intentionally small, but opinionated.
Some next steps that fit this repository:
	•	Config file for the demo (YAML):
	•	switch strict_mode on/off for PolicyCore,
	•	configure max_output_chars for NeuralEngineLite.
	•	Expose a tiny HTTP API on top of the demo node
(/api/chat, /api/stats) using a lightweight web framework.
	•	Extend MetaObserverLite with more metrics:
	•	per-session statistics,
	•	simple anomaly flags (e.g. “many refused answers in a row”).
	•	Add more fine-grained safety rules in PolicyCore
in a way that mirrors the future “full” Samaritan 1.5 policy core.
	•	Publish a v0.1.0 tagged release once the interface stabilises.

⸻

🧠 Vision

Samaritan Lite is not the full Samaritan 1.5 node.
It is a playground that captures the spirit of the architecture:
	•	a self-contained node,
	•	with a clear separation between:
	•	neural engine,
	•	safety policy,
	•	meta-observer / metrics.

______

The long-term plan is to evolve these ideas into the full Samaritan Core:
federated learning, differential privacy, secure aggregation and all the heavy
runtime machinery.

For now, this repository is the smallest possible “brain” that still feels
like Samaritan. 🧠✨

