# Synapta — Skim Deck for [YC24 Alum]
*6 slides · ~5 min read · designed for export to Google Slides or print*

---

## Slide 1 — Cover

# Synapta
## Sovereign AI for India's regulated $400M-$1B inference market

> **+31.3 pp held-out lift on RBI/SEBI extractive QA. McNemar p < 10⁻⁴³. Trained in 3.5h on $1.50/day GPU.**
> *5 contributions. 1 base model. 1 RTX 5090.*

Udit Jain · solo founder · YC W26 applicant
[github.com/udit/synapta] · [demo.loom] · udit@synapta.ai

**Suggested visual**: simple black-on-white, the headline number set in 80pt as the only visual element. Minimal.

---

## Slide 2 — The architectural insight: Format Guard routing

# One base model, four adapters, regex-driven hot-swap every 10 tokens

```
                  ┌─────────────────────────────────────┐
                  │   Nemotron-30B (4-bit NF4 base)     │
                  └────────┬───────────┬───────────┬────┘
                           │           │           │
                  ┌────────┴───┐ ┌─────┴───┐ ┌─────┴────┐ ┌────────────┐
                  │ math LoRA  │ │ code    │ │ science  │ │ bfsi_extr  │
                  │   r=64     │ │  r=64   │ │  r=64    │ │  r=16      │
                  └────────────┘ └─────────┘ └──────────┘ └────────────┘
                           ▲
                           │
                  ┌────────┴────────────────────────────────────┐
                  │  FormatGuardLogitsProcessor                 │
                  │  every 10 tokens: regex(decoded ctx) →      │
                  │  swap adapter via model.set_adapter(target) │
                  └─────────────────────────────────────────────┘
```

**Why novel**:
- Prior PEFT routing (Switch / X-LoRA / MoLE / SiRA) uses *learned* gates
- Ours is **deterministic + interpretable + zero training overhead**
- Adapter selection is auditable (regex match → which adapter, when, why)
- Works at inference time on any base model with PEFT-Lib LoRA support

**Suggested visual**: the ASCII above as an actual diagram (rectangles + arrows in Slides shapes).

---

## Slide 3 — The proof FG actually works (pre-BFSI)

# HumanEval +17.1 pp · MBPP +20.1 pp on Qwen-7B
*McNemar paired tests, p < 0.001 both*

| Benchmark | Base Qwen-7B | + FG (math/code/sci) | Lift | Significance |
|---|---|---|---|---|
| HumanEval pass@1 | 38.4% | 55.5% | **+17.1 pp** | McNemar p < 0.001 |
| MBPP pass@1 | 41.4% | 61.5% | **+20.1 pp** | McNemar p < 0.001 |
| MATH-500 (n=200) | varies by adapter | (Code Paradox finding — see report) | — | — |

**Surprising side-finding**: code-trained adapter beats math-trained adapter on MATH-500 questions presented as `python` code blocks. Format determines optimal routing target, not the math/code semantic axis.

This was the architecture proof that warranted scaling to a regulated-domain task. Yesterday's BFSI work is the next data point.

**Suggested visual**: a 2-bar comparison chart (base vs +FG) on HumanEval and MBPP side-by-side.

---

## Slide 4 — The methodology moat: document-disjoint eval

# Most fintech AI vendors silently overstate accuracy by 3-5×

**The shortcut competitors take**:
1. Curate 100 BFSI questions
2. Paraphrase-augment to 1000 train + 100 eval
3. Train, evaluate, report 90%+ accuracy
4. Customer pilots in production: 30-40% accuracy

**The contamination**: train and eval are paraphrases of the same source questions. Model memorizes the templates, not the regulations.

**What we do instead**:
- Scraped 130 RBI + SEBI master directions (public domain)
- 3-tier *deterministic* QA construction (numeric regex + heading-based + native FAQ)
- **NO LLM-generated questions** (no self-distillation poison)
- Hold out **26 entire PDFs** for eval — model never sees these documents during training, even as paraphrases
- 98.45% validator pass rate on 4373 train + 698 eval

**The methodology IS the moat.** When a customer evaluates us in their dataroom on day one, the number stays the same. When competitors ship, theirs collapses.

**Suggested visual**: side-by-side diagram showing "their pipeline" (paraphrase loop) vs "our pipeline" (document-disjoint hold-out).

---

## Slide 5 — Yesterday's BFSI receipt

# +31.3 pp on 595 paired held-out questions, McNemar p = 6.26 × 10⁻⁴⁴

| Metric | Base Nemotron-30B (4-bit) | +BFSI Adapter | Lift |
|---|---|---|---|
| **Substring match** | **58.3%** [54.3, 62.2] | **89.6%** [86.9, 91.8] | **+31.3 pp** |
| Token F1 (mean) | 0.133 | 0.173 | +0.040 |
| Exact match | 0.0% | 0.0% | (model adds context) |

*[Wilson 95% CI in brackets, non-overlapping intervals]*

**McNemar contingency** (n=595 paired):

|  | Adapter ✓ | Adapter ✗ |
|---|---|---|
| **Base ✓** | 334 | 13 |
| **Base ✗** | **199** | 49 |

**15× improvement-to-regression ratio.** McNemar exact binomial p = 6.26 × 10⁻⁴⁴.

| Slice | Base | +BFSI | Lift |
|---|---|---|---|
| Tier 2 (numeric) | 63.0% | 87.8% | +24.8 pp |
| Tier 3 (heading-extractive) | 52.9% | 92.0% | **+39.1 pp** |
| RBI (n=342) | 58.0% | 89.8% | +31.8 pp |
| SEBI (n=253) | 59.7% | 89.3% | +29.6 pp |

**Cost**: $1.50/day GPU · 3h 28min training · LoRA r=16, α=32, 1 epoch · single RTX 5090

**Suggested visual**: the contingency table as a 2x2 colored grid (199 cell highlighted green, 13 cell highlighted red), McNemar p as a footer.

---

## Slide 6 — Why now + what we're asking

# RBI compute localization + open frontier models + PEFT maturity = first window

**Market timing**:
- RBI Master Direction on Outsourcing IT Services (April 2023) tightening sovereignty mandates
- DPDP Act 2023 raising data-residency stakes
- Nemotron-30B / Qwen-3.6 / Llama-4 closing the frontier gap on small-context tasks
- ~200 mid-tier Indian banks, NBFCs, insurers spending $1-5M/yr on compliance tech each

**Customer flywheel**:
1. Customer brings their proprietary regulatory + SOP corpus
2. We run the deterministic pipeline (8h, $5)
3. They get an adapter trained on their data, deployed on their hardware
4. They own the IP. We own the platform. They never re-license; we keep selling the platform updates.

**The ask of YC**: $500K SAFE, 18 months runway to first 3 paid pilots ($150-300K each).

**The ask of you, [FIRST_NAME]**:
1. Application opener — does it land in the first sentence?
2. Methodology vs BFSI numbers — which leads?
3. Anything I'm proud of that a partner won't care about?

**Bonus ask if you have rapport**: review the 60-sec pitch on a 10-min call before YC interview.

**Suggested visual**: timeline showing the 3 market converges + a single arrow pointing to "now".

---

## Closing pages (not slides — appendix for Google Slides export)

**Speaker notes for whoever delivers this**:
- Total read time: ~5 min
- Total speak time: ~10-12 min if presented out loud
- Each slide is self-contained; if [FIRST_NAME] only reads slide 1 + 5, he gets the headline
- Slide 4 is the most YC-partner-relevant; slide 2 is the most researcher-relevant; you should know which audience you're optimizing for

**Linked artifacts** (for him to follow up):
- Detailed report: `docs/recent/RESEARCH_REPORT_FOR_ALUM.md`
- Full methodology: `docs/recent/BFSI_ADAPTER_FINAL.md`
- Reproducer: `synapta_src/data_pipeline/01_*.py` through `08_*.py`
- Repo: github.com/udit/synapta (going public May 4 9am IST)
