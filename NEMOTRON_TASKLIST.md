# Nemotron × GC-LoRI Task List

Updated: 2026-04-16

This is the working checklist for the Nemotron track. Rewritten to center on the Gate-Conditioned LoRI (GC-LoRI) innovation rather than generic scale-up.

---

## ✅ Done — Environment & Architecture Mapping

- [x] Nemotron weights extracted in `models/nemotron/` (13 safetensors shards + config)
- [x] Architecture mapped by source inspection: 23 Mamba + 23 MoE + 6 GQA layers
- [x] Candidate target modules identified: `q_proj, k_proj, v_proj, o_proj` (GQA), `in_proj, out_proj` (Mamba), `up_proj, down_proj` (MLP/expert)
- [x] `mamba_ssm==2.3.1` installed and validated (`rmsnorm_fn` import succeeds)
- [x] `scripts/nemotron_probe.py` hardened for source-backed output without CUDA
- [x] `train_lori_adapter.py` upgraded to template-derived assistant-prefix masking
- [x] `src/lori_moe/configs/nemotron_config.py` created (attention-only default)
- [x] `models/nemotron/architecture_notes.md` written
- [x] `scripts/reformat_data_for_nemotron.py` exists
- [x] `scripts/nemotron_pipeline.sh` exists (draft)
- [x] Nemotron chat template available locally

## ✅ Done — Research Framing

- [x] Identified the core innovation: GC-LoRI (Gate-Conditioned LoRI)
- [x] Defined three concrete hypotheses (GC-LoRI, Shared-Expert Aug, Routing-Entropy Detector)
- [x] Defined falsification criteria
- [x] Wrote complete NEMOTRON_PLAN.md with GC-LoRI architecture and code specifications

---

## ✅ Phase 0 — Model Validation & GPU
*Status: Complete* — GPU driver validated, VRAM tested, and novel hypothesis verified.

- [x] **0.1** GPU checked: RTX 5090 healthy, 33.6 GB VRAM accessible.
- [x] **0.2-0.4** Mamba/MoE architecture probed and standard generation verified.
- [x] **0.5** **ROUTER ANALYSIS (NOVEL)**: 2000-prompt analysis proved internal MoE signal correlates with domain (Math/Code/Science) with p<0.0001.

---

---

## ✅ Phase 1 — Data + Config Preparation
*Status: Complete* — Data reformatted with max limits scaling to 50k for math.

- [x] **1.1** Built math (50k), code (20k), science (11k) sets.
- [x] **1.2** Nemotron config locked format.
- [x] **1.3/1.4** Shared `B` projection generated and mathematically verified orthogonal.

---

## [/] Phase 2 & 3 — Baseline Eval & Adapter Training
*Status: Proceeding in Background (Pipeline)* — Currently evaluating base Nemotron on smoke test scale, then training 3 domain adapters.

- [x] **2.1.a** Calibrated smoke test benchmarks (50-200 samples) to ensure pipeline health before full runs.
- [x] **Baseline Progress:** GSM8K Baseline Complete (**82.00% Accuracy**).
- [/] **Baseline Progress:** ARC-Challenge Baseline In-Progress (~20% done).
- [ ] **Upcoming:** MMLU (200 samples) & HumanEval (50 samples) Baseline.
- [x] **Phase 3:** Execute autonomous `train_lori_adapter.py` for **Mathematics Reasoning** (50k examples).
- [x] **Phase 4:** Execute training for Code (20k).
- [/] **Phase 5:** Execute training for Science (11k) sets — **CURRENTLY RUNNING**.
- [ ] **Phase 6:** Expanded Step 7 Evaluations (Math, HumanEval, ARC).
- [ ] **Phase 7:** Execute autonomous `train_gc_router.py` (Innovation Step).

---

## [/] Phase 3.5 & 4 — GC-LoRI Innovation & Ablation
*Status: Code built, queued in Pipeline* — The custom GC-Router and hook architecture is built and waiting to trigger.

- [x] **3.1-3.4** Engineered GC-Router, internal hooks, compose engine, and trainer script.
- [ ] **3.5/3.6** Train GC-LoRI using internal state tracking to outpredict blind-routed.
- [ ] **4.1** Run massive GC comparison (`gc_compose.py`) vs Blind.

---

## 🟡 Phase 5 — Comparison + Paper
*Status: Pending Pipeline Data*

- [ ] **5.1** Fill Qwen vs Nemotron scaling comparison table
- [ ] **5.2** Fill Blind vs GC-LoRI ablation table
- [ ] **5.3** Write honest verdict: did composition work? Did GC-LoRI help?
- [ ] **5.4** Draft paper with appropriate framing (positive or negative result)
- [ ] **5.5** Push results and code to GitHub

---

## 🚫 Hard Truths To Preserve

- [ ] Do NOT claim Nemotron LoRI-MoE works until at least one adapter successfully trains and evaluates
- [ ] Do NOT claim composition breakthrough unless it beats single-adapter on multi-domain evaluation
- [ ] Do NOT claim GC-LoRI is novel unless internal router signals actually correlate with domain/reasoning
- [ ] Do NOT conflate scale gains with method gains — if Nemotron alone explains the improvement, that's scale, not LoRI-MoE
- [ ] If GC-LoRI shows no benefit over blind routing, publish the negative result honestly

---

## 🎯 Breakthrough Bets (Ordered by Risk)

| # | Bet | Risk | Potential Impact |
|---|---|---|---|
| 1 | **Shared-Expert Augmentation** — adapt only the always-on shared expert | Low | Clean isolation of coordination mechanism |
| 2 | **Routing-Entropy Reasoning Detection** — entropy predicts reasoning tokens | Low | Novel diagnostic, publishable standalone |
| 3 | **GC-LoRI** — internal routing supervises external composition | Medium | If it works, this is the paper |
| 4 | **Mamba-Targeted Adapters** — adapt `in_proj/out_proj` | High | Could unlock sequence-state adaptation |
| 5 | **Full Hybrid (Mamba + GQA + SharedExpert)** | High | Most ambitious, most fragile |

---

## New Files Created/To Create

| File | Status | Purpose |
|---|---|---|
| `scripts/nemotron_router_analysis.py` | ✅ Created | Internal routing signal analysis |
| `src/lori_moe/model/gc_router.py` | ✅ Created | GC-LoRI Router module |
| `src/lori_moe/model/internal_hook.py` | ✅ Created | Internal MoE hook extractor |
| `src/lori_moe/inference/gc_compose.py` | ✅ Created | GC-LoRI inference engine |
| `src/lori_moe/training/train_gc_router.py` | ✅ Created | GC-LoRI router trainer |
| `results/nemotron/router_analysis/` | 📝 To create | Routing analysis outputs (Generated dynamically) |

---

## Execution Priority (For Efficient Compute Use)

> The user has limited access. Maximize signal per GPU-hour.

1. **Fix GPU** (0 compute, pure driver work)
2. **Run probe** (5 min, validates everything downstream)
3. **Run router analysis** (10 min, determines whether GC-LoRI is viable — HIGHEST ROI EXPERIMENT)
4. **Reformat data** (2 min, no GPU)
5. **Generate shared B** (1 min)
6. **Train Math adapter** (~30 min, proves pipeline works)
7. **Evaluate Math adapter** (10 min, proves adapters help)
8. **Train Code + Science** (~60 min)
9. **Build GC-LoRI code** (no compute, pure coding)
10. **Train GC-LoRI Router** (~20 min)
11. **Run ablation table** (~30 min)
12. **Write comparison + paper framing** (no compute)

**Total estimated GPU time: ~3 hours for the complete innovation.**
