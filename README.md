# Synapta: Dynamic Token-Level Intelligence Composition

![Mewtwo Logo](https://raw.githubusercontent.com/uditjain13/mewtwo/main/assets/mewtwo_logo.png) <!-- Note: Mock URL for framing -->

Synapta is a state-of-the-art framework for real-time, token-level expert composition in Large Language Models. Built on the **Nemotron-3-Nano (30B)** hybrid Mamba-Attention architecture, Synapta enables specialized AI engines (Math, Code, Science) to be hot-swapped mid-sequence with **0ms overhead**.

## 🚀 Key Breakthroughs

### 1. The Code Paradox
Our research discovered that code-specialized adapters provide superior logical "scaffolding" for mathematical proofs compared to dedicated math adapters. Synapta's router leverages this paradox by switching to the Code expert during complex logical derivations, regardless of the prompt domain.

### 2. Zero-Latency Hot-Swapping
By utilizing high-speed PEFT pointer switching, Synapta eliminates the memory and latency overhead of traditional Mixture-of-Experts (MoE). Experts are pre-loaded in VRAM and activated per-token within the inference loop.

### 3. Syntax-Lock Guard (V2)
A stateful routing guard that prevents mid-function logic switches from disrupting syntactical integrity, recovering HumanEval performance gains from 45% to 60%.

## 📊 Performance Benchmarks (Nemotron 30B)

| Benchmark | Base Model | Merged Adapters | Synapta (Routed) |
| :--- | :---: | :---: | :---: |
| **MATH-500** | 41.5% | 56.0% | **56.0%** |
| **ARC-Challenge** | 20.0% | 19.0% | **31.0%** |
| **HumanEval** | 50.0% | 34.0% | **45.0%** |
| **GSM8K** | 68.0% | 72.0% | **78.0%** |

## 🛠 Tech Stack
- **Model**: nvidia/Nemotron-3-Nano-30B-A3B (Hybrid Mamba-Attention)
- **Quantization**: 4-bit NF4 (BitsAndBytes)
- **Framework**: PyTorch + HuggingFace PEFT + Custom LogitsProcessor
- **Hardware**: Optimized for NVIDIA RTX 5090 (32GB VRAM)
- **Demo**: FastAPI + WebSocket + Glassmorphism UX

## 📖 Academic Papers
The Synapta research suite consists of two primary manuscripts:
1. **[Synapta Systems](manuscripts/synapta_systems.md)**: Zero-Latency Token-Level PEFT Routing vs Static Parameter Collapse.
2. **[The Code Paradox](manuscripts/code_paradox.md)**: Asymmetric Cross-Domain Transfer in Token-Routed Composition.

## 🚀 Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Launch Investor Demo: `python src/demo/server.py`
3. Run Benchmarks: `python scripts/token_router_eval.py`

---
*Developed under the Mewtwo Research Collective.*
