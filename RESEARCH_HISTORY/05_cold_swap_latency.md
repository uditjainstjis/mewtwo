# Cold-Swap Latency Profile

**Date:** 2026-04
**Source artefact:** `results/cold_swap_metrics.json`

## Setup
44 LoRA-adapter swaps observed under realistic generation traffic on Nemotron-Nano-30B-A3B (4-bit NF4 base on RTX 5090). Adapters stored on local NVMe SSD.

## Two paths

### Cold swap (load adapter from disk on demand)
- Avg latency: **315.9 ms / swap**.
- Cost dominated by safetensors deserialization + PEFT integration.
- Acceptable for prototyping, **prohibitive for token-level routing in production** (would burn 31s of latency over a 100-token generation).

### Warm swap (all adapters pre-loaded into VRAM)
- `model.load_adapter(...)` once at startup loads all $N$ adapter weights into VRAM.
- `model.set_adapter(name)` becomes a dictionary pointer assignment — no memory transfer.
- Per-swap cost: $O(1)$ (microseconds, dominated by Python dispatch).

## Memory budget
- Base 4-bit Nemotron-30B: $\approx 17$ GB VRAM.
- Each $r=16$ LoRA adapter (bf16 safetensors): $\approx 1.7$ GB.
- 4-adapter Format Guard process: $\approx 18.0$ GB peak VRAM.
- Fits on a single 32 GB consumer GPU (RTX 5090).
- 5-adapter (with bfsi_recall) variant: $\approx 19.7$ GB peak — still single-GPU.

## Implication for production
**Use warm-swap.** Pre-load all customer adapters at process boot; route via `set_adapter()` at every token-window decision. The Format Guard mechanism (`04_format_guard.md`) uses warm-swap exclusively.

## Files
- `results/cold_swap_metrics.json`
- `docs/MASTER_KNOWLEDGE_BASE.md` §4.6 (narrative)
