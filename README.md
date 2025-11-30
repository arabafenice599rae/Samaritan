
<p align="center">
  <img src="./logo.svg" alt="Samaritan Logo" width="260" />
</p>

<h1 align="center">Samaritan Lite</h1>

<p align="center">
  <em>A tiny, opinionated playground for the Samaritan 1.5 distributed brain.</em>
</p>

<p align="center">
  <a href="https://github.com/arabafenice599rae/Samaritan/actions/workflows/build.yml">
    <img src="https://github.com/arabafenice599rae/Samaritan/actions/workflows/build.yml/badge.svg" alt="Build Status">
  </a>
</p>

---

## ✨ What is Samaritan Lite?

Samaritan Lite is a **minimal, demonstrative implementation** of the Samaritan 1.5 architecture:

> A distributed, privacy-by-design brain where every installation is a full node,  
> not just a dumb client.

This repo is **not** a full FL / DP production stack.  
It’s a **small but realistic core** that shows:

- how a **neural engine** can be wrapped with
- a **safety policy core**, plus
- a tiny **meta-observer** that collects stats about the node.

Perfect for:

- experimenting locally,
- reviewing ideas for Samaritan 1.5,
- or using as a skeleton for a richer node later.

---

## 🧱 Repository layout

```text
Samaritan/
├─ Cargo.toml              # Workspace: samaritan-core-lite + lite-node-demo
├─ logo.svg                # Transparent brain logo
├─ README.md               # You are here
│
├─ samaritan-core-lite/    # Core library (NeuralEngineLite + PolicyCore + MetaObserverLite)
│  ├─ Cargo.toml
│  └─ src/
│     ├─ lib.rs
│     ├─ neural_engine_lite.rs
│     ├─ policy_core.rs
│     └─ meta_observer.rs
│
└─ lite-node-demo/         # Small CLI node using the core library
   ├─ Cargo.toml
   └─ src/
      ├─ main.rs
      ├─ simple_node.rs
      ├─ policy_core.rs        # wired into the demo
      └─ meta_observer_lite.rs


⸻

🚀 Quick start

Requirements: recent Rust toolchain (rustup + stable).

Clone the repo:

git clone https://github.com/arabafenice599rae/Samaritan.git
cd Samaritan

Build everything:

cargo build

Run the CLI demo node:

cargo run -p lite-node-demo

You’ll see a prompt like:

=== Samaritan Lite Node Demo ===
Commands:
  - type a normal message to talk to the node
  - type "/stats" to see MetaObserverLite statistics
  - type "/reset_stats" to reset the statistics
  - type "/quit" to exit

Then:
	•	type a normal message → the node runs NeuralEngineLite + PolicyCore,
	•	type /stats → the node prints aggregated stats (turns, average length, etc.),
	•	type /reset_stats → counters are cleared,
	•	type /quit → exit.

⸻

🧠 Core concepts

1. NeuralEngineLite

A deterministic, rule-based “neural engine” that simulates different response modes:
	•	detects:
	•	empty input,
	•	long wall-of-text,
	•	questions (?),
	•	chooses a style:
	•	Small talk / coaching,
	•	Question answer,
	•	Summary for long text,
	•	always applies a hard maximum output length for safety.

It doesn’t do real LLM inference.
It’s deliberately simple and testable, but structured like a real engine:
	•	clear config struct (NeuralEngineLiteConfig),
	•	pure, deterministic generate(...),
	•	unit tests that verify:
	•	mode selection,
	•	length limits,
	•	basic behavior.

⸻

2. PolicyCore

A tiny safety / policy module that inspects:
	•	user input, and
	•	model output,

and returns a PolicyDecision:

enum PolicyDecisionKind {
    Allow,
    SafeRespond,
    Refuse,
}

Current hard-coded rules (for the demo):
	•	detects self-harm phrases → SafeRespond,
	•	detects obvious crime / hacking keywords → Refuse,
	•	very rough check for possible credit-card-like numbers → SafeRespond,
	•	in strict_mode, can enforce stricter limits (e.g. very long outputs).

The idea: in the real Samaritan 1.5, PolicyCore becomes the Constitution.
Here you have a tiny, readable starting point.

⸻

3. MetaObserverLite

A minimal observer wired inside the demo node that tracks things like:
	•	number of turns,
	•	how many times each PolicyDecisionKind was used,
	•	average input / output length.

From the CLI you can:
	•	/stats → dump the current snapshot,
	•	/reset_stats → clear all counters.

It’s intentionally tiny, but keeps the same spirit as the full Meta-Observer:

observe the brain, don’t just run it.

⸻

🧪 Tests & CI

Run all tests locally:

cargo test

The repo ships with:
	•	unit tests for:
	•	NeuralEngineLite,
	•	PolicyCore,
	•	MetaObserverLite,
	•	a GitHub Actions workflow (.github/workflows/build.yml) that:
	•	builds the workspace,
	•	runs the full test suite on every push / PR.

If the badge on top is green, the lite node and core library compile and all tests pass.

⸻

🧭 Roadmap / Ideas

This repository is intentionally small, but it can grow in several directions:
	•	add a simple YAML config for:
	•	strict_mode,
	•	max_output_chars,
	•	maybe toggles for different policy profiles;
	•	plug in a real LLM backend (local / remote) behind NeuralEngineLite;
	•	expand PolicyCore into a proper policy engine:
	•	more categories,
	•	per-rule logging,
	•	configuration and tests;
	•	turn MetaObserverLite into a tiny metrics exporter (Prometheus / JSON over HTTP);
	•	experiment with multi-node setups later, reusing the same API surface.

⸻

🤝 Contributing

Right now this is a personal / experimental project.

If you want to play with it:
	1.	fork the repo,
	2.	make a small, focused change,
	3.	run:

cargo fmt
cargo clippy --all-targets --all-features
cargo test


	4.	open a Pull Request with a short description of what you changed and why.

⸻

📄 License

This repository is currently experimental.
See the LICENSE file (or future updates) for license details once stabilized.

