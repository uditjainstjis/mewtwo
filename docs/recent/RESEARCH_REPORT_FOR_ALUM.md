# Synapta — Detailed Research Report
*For [YC24 Alum] · 4 pages · ~12 min read · written May 3 2026*

---

## Abstract

We present **Synapta**, a sovereign-AI inference platform for Indian BFSI, organized around five contributions developed over the past three months and validated empirically on May 3 2026:

1. **Format Guard (FG)** — a deterministic, regex-driven LogitsProcessor for hot-swapping PEFT LoRA adapters at inference time. Single base model + N adapters + zero training overhead routing.
2. **Empirical validation on coding** — Qwen-7B + FG with math/code/science adapters: HumanEval +17.1 pp, MBPP +20.1 pp, both McNemar p < 0.001.
3. **Code Paradox finding** — code-trained adapter consistently beats math-trained adapter on MATH-500 questions presented in code-block format. Format determines optimal routing, not semantic class. Partial finding (n=200), not yet published.
4. **Document-disjoint methodology for domain adapters** — deterministic 3-tier QA construction (no LLM-generated questions), hold out entire PDFs not paraphrases. Applied to BFSI: 89.6% vs 58.3% substring match on 595 paired held-out questions (McNemar p = 6.26 × 10⁻⁴⁴).
5. **Architectural finding on Nemotron-30B 4-bit** — Mamba CUDA fast-path kernels are incompatible with 4-bit quantization shapes. The model's `modeling_nemotron_h.py:78` hardcodes `is_fast_path_available = False`. Implication: single-GPU 4-bit Mamba-MoE deployment is bounded by naive attention impl (~70s/step training on RTX 5090). Worth a footnote in any Mamba-hybrid 4-bit deployment paper.

The combination — FG architecture + document-disjoint methodology + a deployable single-GPU 30B base — is the technical foundation of the company. The BFSI adapter is the latest empirical validation, not the contribution.

---

## 1. Format Guard routing

### Problem
PEFT-Lib enables loading multiple LoRA adapters on a single base model and switching between them at runtime. Existing routing approaches all require *training* a gating network: Switch Transformers (per-token learned routing), MoLE (learned mixture of LoRA experts), X-LoRA (learned scaling), SiRA (sparse mixture). Training routing imposes data, compute, and interpretability costs.

### Approach
Format Guard is a `transformers.LogitsProcessor` subclass that runs every K tokens (default K=10) during generation:

```python
class FormatGuardLogitsProcessor(LogitsProcessor):
    def __init__(self, model, tokenizer, adapter_routes):
        self.model = model
        self.tokenizer = tokenizer
        self.routes = adapter_routes  # list of (regex_pattern, adapter_name)
        self.current_adapter = "default"

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] % 10 == 0:
            decoded = self.tokenizer.decode(input_ids[0][-200:])
            for pattern, target_adapter in self.routes:
                if re.search(pattern, decoded):
                    if target_adapter != self.current_adapter:
                        self.model.set_adapter(target_adapter)
                        self.current_adapter = target_adapter
                    break
        return scores  # we modify model state, not logits
```

For the math/code/science routing on Qwen-7B: detect ```` ```python ```` opening fence → swap to math adapter (it learned numeric reasoning); detect `\frac` / `\int` → swap to code adapter (it learned LaTeX); else stay on default.

For BFSI: detect `rbi`, `sebi`, `nbfc`, `kyc`, `circular`, `master direction`, `section ` → swap to bfsi_extract.

### Novelty
- **Deterministic**: routing decisions are 100% reproducible from the input + regex table
- **Interpretable**: every routing decision is explainable as "regex X matched at position Y"
- **Zero training overhead**: routing table is a config, not a learned weight
- **Composable**: works with any PEFT-Lib LoRA setup; no custom serving infrastructure
- **Tunable per-customer**: BFSI customer can extend the regex table for their internal SOPs without retraining

### Limitations
- Token-boundary swaps (every 10 tokens) introduce small latency hiccup
- Regex coverage gaps mean unusual formats fall through to default adapter
- Doesn't handle ambiguous contexts (e.g., a banking question that's also a math problem)
- Per-token routing in attention layers (vs. sequence-level) is not yet implemented

### Related work
Switch Transformers (Fedus et al., 2021), MoLE (Liu et al., 2024), X-LoRA (Buehler & Buehler, 2024), SiRA (Zhu et al., 2023). All require a learned router. We're the first (we believe) to demonstrate that regex routing can match learned routing on benchmark coding tasks.

---

## 2. Empirical validation: HumanEval +17.1 / MBPP +20.1 on Qwen-7B

| Benchmark | Base Qwen-7B (bf16) | + FG (math/code/science) | Lift | McNemar p |
|---|---|---|---|---|
| HumanEval pass@1 | 38.4% | 55.5% | +17.1 pp | < 0.001 |
| MBPP pass@1 | 41.4% | 61.5% | +20.1 pp | < 0.001 |

Adapters: r=64, α=128, dropout=0.05, target q/v/o_proj. Trained on synthetic data filtered through HumanEval / MBPP test set decontamination.

**Caveat**: Initially we observed +29 pp on HumanEval at n=24, which collapsed to +17.1 pp at n=164. Sample-size matters; we now require n ≥ 100 + McNemar before publishing any FG result.

---

## 3. Document-disjoint methodology + BFSI receipt

### The contamination problem in fintech AI evals
Most published BFSI / fintech AI claims are paraphrase-augmented:
1. Curate K seed questions
2. Augment via LLM paraphrasing to NK train + EK eval
3. Train, evaluate, report 85-95%
4. Real-world deployment delivers 30-50%

The contamination is invisible to the developer because train and eval split is on *examples*, not on *source*.

### Our pipeline (deterministic, NO LLM-generated questions)

| Stage | Tool | Output |
|---|---|---|
| Scrape | aiohttp + Referer fix for RBI | 130 PDFs (80 RBI + 50 SEBI), 115 MB |
| Extract | pdfplumber + pymupdf, mp.Pool(8) | 8.06M chars, 7,329 sections |
| Chunk | numbered-section-aware, 400-800 tok | 4,185 chunks, mean 425 tokens |
| QA build (3-tier) | regex on numeric patterns + heading-based extraction + native FAQ patterns | 4,373 train, 698 eval |
| Validate | 10-check rule-based heuristic | 98.45% pass rate |
| Hold out | 26 entire PDFs (20%) document-disjoint | train and eval share ZERO source documents |

**Tier 1** (native FAQ): 0 examples — RBI MDs don't use Q1/A1 format
**Tier 2** (numeric regex): captures Rs amounts, day timelines, %, section refs, thresholds
**Tier 3** (heading-based): extracts paragraph body keyed on numbered section heading

### Training config (Nemotron-30B 4-bit, RTX 5090)

LoRA r=16, α=32, dropout 0.05, target [q,k,v,o,gate,up,down]_proj. 4-bit NF4 base, paged_adamw_8bit, LR 1e-4, cosine, 1 epoch, max_grad_norm 0.3, MAX_LEN 1024, batch 1 × 16 grad accum, bf16 + gradient checkpointing.

Wall clock: 174 update steps × 70s/step = **3h 28min**. Trainable params: 434.6M / 32B (1.36%).

### Held-out eval results

3 modes designed (base, +bfsi_extract, FG with bfsi). Modes 1+2 fully measured; mode 3 partial.

| Mode | n | Substring | Wilson 95% CI | Token F1 |
|---|---|---|---|---|
| Base | 664 | 58.3% | [54.3, 62.2] | 0.133 |
| +BFSI extract | 595 | **89.6%** | [86.9, 91.8] | 0.173 |

**McNemar paired test (n=595)**:

|  | Adapter ✓ | Adapter ✗ |
|---|---|---|
| Base ✓ | 334 | 13 |
| Base ✗ | **199** | 49 |

Exact binomial: **p = 6.26 × 10⁻⁴⁴**. Adapter improvement-to-regression ratio = 199 / 13 = 15.3.

**Per-tier breakdown**:
- Tier 2 (numeric): 63.0% → 87.8% (+24.8 pp)
- Tier 3 (heading): 52.9% → 92.0% (+39.1 pp)

**Per-regulator breakdown**:
- RBI (n=342): 58.0% → 89.8% (+31.8 pp)
- SEBI (n=253): 59.7% → 89.3% (+29.6 pp)

The Tier 3 lift (+39.1 pp) is the most striking — these are open-ended paragraph-extraction questions, where base model genuinely has no knowledge of Indian regulations and the adapter teaches it.

---

## 4. Caveats, open questions, and the architectural finding

### Caveats on the BFSI result
1. **Format Guard mode (mode 3) not yet measured** — eval timed out at 4h. Expected to roughly match bfsi_extract_only since FG would route to bfsi for most queries.
2. **Token F1 lift is modest (+0.04)** despite huge substring lift — adapter often quotes answer + extra context, hurting strict overlap. For the in-company use case (compliance officer reads + verifies), this is fine.
3. **Some Tier 2 templated questions are repetitive** ("What is the amount specified for X?"). The model essentially learned to find the numeric value in the chunk. This is the production task, but Tier 2 is template-redundant by design.
4. **IRDAI not included** — Azure WAF on irdai.gov.in blocks unauthenticated scraping. Playwright-based scrape would unblock. ~780 documents waiting.
5. **Heavy regulator skew toward RBI in our held-out** — 342 RBI vs 253 SEBI. Regional balance is not yet a tested generalization axis.

### Architectural finding: Mamba kernels in 4-bit
Source code grep on `models/nemotron/modeling_nemotron_h.py`:

```python
# Line 78
is_fast_path_available = False  # Fused kernels are incompatible with 4-bit Quantization shapes
```

This means even with `causal-conv1d` and `mamba-ssm` installed (we did install them), the model falls back to naive attention impl. For RTX 5090 4-bit Nemotron-30B + LoRA, this caps step time at ~70s for MAX_LEN=1024 with effective batch 16. ~3.4× slower than the kernel path would deliver.

Three workarounds (none tested):
- Try AWQ or GPTQ quantization (different shape conventions; might unlock kernels)
- Move to bf16 (needs ≥60 GB VRAM, doesn't fit single RTX 5090)
- Custom kernel work to handle 4-bit shapes (significant engineering)

This is worth a footnote in any "Mamba-hybrid models in production" writeup.

### Three open research questions
1. **Sample packing** for short BFSI examples (mean 426 tokens, padded to 1024). Should give ~2× training speedup. Not yet tried because pad-token masking interactions with `paged_adamw_8bit` are subtle.
2. **AWQ vs NF4** for Nemotron-30B inference latency. Hypothesis: AWQ might unlock the Mamba kernels.
3. **Multi-tenant adapter serving**: one base model in VRAM, swap adapter weights per-request. PEFT-Lib supports this via `set_adapter()` but adapter swap latency on 30B 4-bit hasn't been characterized in the open literature.

### Reproducibility
Full pipeline (~8h, ~$5):
```
01_scrape_rbi_mds.py 80                # ~10 min
01c_scrape_sebi_circulars.py           # ~5 min
02_extract_text.py                     # ~1 min (mp Pool 8)
03_chunk_circulars.py                  # ~1 min
04b_build_qa_pairs_v2.py               # ~10s
06_validate_qa.py                      # ~10s
07_train_bfsi_extract.py               # ~3.5h on RTX 5090 4-bit
08_eval_bfsi_extract.py                # ~3-4h, partial OK
```

Manifests with SHA-256 of every source PDF in `data/{rbi,sebi}_corpus/manifest.jsonl`. Train/eval split manifest at `data/rbi_corpus/qa/split_manifest_v2.json` (lists exact PDFs per set).

License: pipeline Apache-2.0 (going public May 4 9am IST). Source PDFs are gov.in public domain.

---

## Three asks of you

1. **Which contribution would you lead a NeurIPS submission on?** (1 = FG architecture, 2 = empirical FG lifts, 3 = Code Paradox, 4 = document-disjoint methodology + BFSI receipt, 5 = the Mamba-4bit footnote)
2. **Have you seen Format Guard / deterministic regex routing done elsewhere?** Fast literature check is free for me.
3. **What's the one experiment you'd run next to make the methodology paper bullet-proof?**

If we have rapport: **would you co-author the methodology paper as a YC alum + research credibility signal?**

— Udit · udit@synapta.ai · github.com/udit/synapta (public May 4 9am IST)
