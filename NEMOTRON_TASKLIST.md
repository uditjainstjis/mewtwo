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

## 🔴 BLOCKED — GPU Environment (MUST FIX FIRST)

Everything below is gated on GPU access.

- [ ] **B1.** Verify `torch.cuda.is_available() == True`
- [ ] **B2.** Verify `nvidia-smi` shows healthy driver + RTX 5090
- [ ] **B3.** Re-run `scripts/nemotron_probe.py` with CUDA and save `module_map.json`
- [ ] **B4.** Confirm 4-bit Nemotron loading works and record VRAM usage
- [ ] **B5.** Run short generation test to prove Nemotron works on this GPU

---

## 🟡 Phase 0 — Model Validation (After GPU Fix)

- [ ] **0.1** Run GPU health check script
- [ ] **0.2** Run `scripts/nemotron_probe.py` → save `module_map.json`
- [ ] **0.3** Record VRAM: 4-bit loading should use ~16GB, leaving ~16GB headroom
- [ ] **0.4** Run short generation test (5 prompts, verify coherent output)
- [ ] **0.5** 🔬 **NOVEL:** Run `scripts/nemotron_router_analysis.py` — analyze internal MoE routing patterns across domain types
  - Does router entropy differ between math vs code vs science tokens?
  - Do top-K expert selections cluster by domain?
  - This determines whether GC-LoRI is viable

---

## 🟡 Phase 1 — Data + Config Preparation

- [ ] **1.1** Run `scripts/reformat_data_for_nemotron.py` → `data/nemotron/*.jsonl`
- [ ] **1.2** Verify Nemotron config: `rank=64, alpha=128.0, attention_only targets`
- [ ] **1.3** Generate shared B projection: `checkpoints/nemotron_lori/shared_projection_B.pt`
- [ ] **1.4** Verify orthogonality of shared projection: `mean_cos_sim < 0.01`

---

## 🟡 Phase 2 — Baseline + Single-Adapter Training

- [ ] **2.1** Evaluate Nemotron baseline: GSM8K, ARC, MMLU (no adapters)
- [ ] **2.2** Train Math adapter (attention-only, QLoRA): ~30 min expected
- [ ] **2.3** Evaluate Math adapter on GSM8K — **GATE: must improve ≥2% over baseline**
- [ ] **2.4** Train Code adapter (same config)
- [ ] **2.5** Train Science adapter (same config)
- [ ] **2.6** Evaluate each single adapter on domain benchmarks
- [ ] **2.7** Verify cross-adapter orthogonality: `mean_cos_sim < 0.01`
- [ ] **2.8** Run interference test: linear merge should cause PPL explosion

---

## 🟡 Phase 3 — GC-LoRI Innovation (THE KEY WORK)

> This is the novel contribution. Everything before was setup for this.

- [ ] **3.1** Create `src/lori_moe/model/gc_router.py` — Gate-Conditioned Router
- [ ] **3.2** Create `src/lori_moe/model/internal_hook.py` — Nemotron internal router hook extractor
- [ ] **3.3** Create `src/lori_moe/inference/gc_compose.py` — GC-LoRI inference engine
- [ ] **3.4** Create `src/lori_moe/training/train_gc_router.py` — GC-LoRI router training
- [ ] **3.5** Train GC-LoRI Router using internal signals + hidden states
- [ ] **3.6** Validate GC-LoRI routing entropy is healthy (> 0.3, not collapsed)

---

## 🟡 Phase 4 — Ablation Experiments

| ID | Experiment | Purpose | Status |
|---|---|---|---|
| **4A** | Blind External Router (no internal signals) | Control — what does standard external MoE give? | ⬜ |
| **4B** | GC-LoRI Router (with internal signals) | **Innovation** — does conditioning help? | ⬜ |
| **4C** | Shared-Expert-Only Adapters | Test always-active path for coordination | ⬜ |
| **4D** | Routing-Entropy Reasoning Detection | Diagnostic — does entropy predict reasoning tokens? | ⬜ |

### Key Comparison

| Metric | Baseline | Single Best | Blind Router | **GC-LoRI** |
|---|---|---|---|---|
| GSM8K | ??? | ??? | ??? | ??? |
| ARC | ??? | ??? | ??? | ??? |
| MMLU | ??? | ??? | ??? | ??? |
| MD Gain | — | — | ??? | **≥ +3%** |

**SUCCESS CRITERION:** GC-LoRI Composition Δ > +3% over single adapter on multi-domain tasks.

---

## 🟡 Phase 5 — Comparison + Paper

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
| `scripts/nemotron_router_analysis.py` | 📝 To create | Internal routing signal analysis |
| `src/lori_moe/model/gc_router.py` | 📝 To create | GC-LoRI Router module |
| `src/lori_moe/model/internal_hook.py` | 📝 To create | Internal MoE hook extractor |
| `src/lori_moe/inference/gc_compose.py` | 📝 To create | GC-LoRI inference engine |
| `src/lori_moe/training/train_gc_router.py` | 📝 To create | GC-LoRI router trainer |
| `results/nemotron/router_analysis/` | 📝 To create | Routing analysis outputs |

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
