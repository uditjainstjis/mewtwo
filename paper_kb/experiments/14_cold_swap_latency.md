# Experiment Card 14 — Cold-Swap Latency Profile

PKB wrapper around `RESEARCH_HISTORY/05_cold_swap_latency.md`.

## 1. Research question
What is the per-swap latency cost of switching the active LoRA adapter on Nemotron-30B 4-bit? Two settings: (a) cold (load adapter from NVMe SSD on demand), (b) warm (all adapters pre-loaded into VRAM, only the pointer changes).

## 2. Dataset / setup
- 44 adapter swaps observed under realistic generation traffic.

## 3. Model
- Base Nemotron-Nano-30B-A3B 4-bit on RTX 5090 (32 GB).
- Adapters $r=16$ ($\sim$1.7 GB each in bf16).

## 4. Evaluation
- Swap latency in ms, per call.

## 5. Results

| Path | Avg latency / swap |
|---|---|
| Cold (load from NVMe) | **315.9 ms** |
| Warm (all in VRAM, `set_adapter` pointer flip) | $O(1)$ (microseconds) |

Memory budget for warm 4-adapter Format Guard process: $\approx 18$ GB peak (base 17 GB + delta from extra adapters). Fits 32 GB consumer GPU.

## 6. Negatives + caveats
- Cold latency is prohibitive for token-level routing; warm swap is the only viable option.
- 5-adapter (with bfsi_recall) variant: $\approx 19.7$ GB peak — still single-GPU.

## 7. Artifact map
PRIMARY: `results/cold_swap_metrics.json`
SECONDARY: `RESEARCH_HISTORY/05_cold_swap_latency.md`, `docs/MASTER_KNOWLEDGE_BASE.md` §4.6
