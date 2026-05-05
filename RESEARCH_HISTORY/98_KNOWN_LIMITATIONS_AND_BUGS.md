# Known Limitations, Bugs, and Methodology Corrections

**Purpose:** every honest gap, every retracted claim, every methodology correction. The reviewer-defensible record of "what we got wrong and how we fixed it."

## Methodology corrections (resolved)

### 1. HumanEval scoring bug (discovered 2026-05-02, corrected)
**Bug:** `extract_code` had two issues — import-stripping (lost `from typing import` headers) and indent-stripping (`body.strip()` removed leading 4-space indent).

**Effect:** v1 scoring under-counted both base and Format Guard pass@1 by $\approx 30$ pp absolute. Inflated the reported delta from "+24 pp at n=25" to "+48.7 pp at n=164 v1" — both wrong directions.

**Correction:** rescored both modes from saved JSONL outputs. Final corrected values (v2): base 56.1\%, FG 73.2\%, **delta +17.1 pp** with McNemar $p < 10^{-3}$.

**Where:** `09_humaneval_scoring_bug.md`, `synapta_src/overnight_scripts/rescore_humaneval.py`.

### 2. Code Paradox cross-family overclaim (discovered 2026-04-29, rolled back)
**Claim that was rolled back:** "Code Paradox replicates across 3 base models from $n=50$ subsets each."

**Why wrong:** $n=50$ is too small for cross-family comparison. Subsequent $n=200$ follow-up showed only one base (Qwen-3.5-0.8B) replicated robustly, and only the in-domain regression (code-on-code), not the asymmetric positive transfer.

**Correction:** Headline is now "Code Paradox: Nemotron-30B Phase 1 ($n=100/200$); in-domain regression replicates on Qwen-3.5-0.8B ($n=200$); rank scaling on Nemotron-Mini-4B does not rescue code adapter." We do not claim cross-family replication of the asymmetric positive transfer.

**Where:** `07_code_paradox_replication.md`.

### 3. Demo "crap answers" bug (discovered 2026-05-02, fixed)
**Bug:** original `src/demo/server.py` had 5 issues that produced incoherent outputs (chat template not applied; max_new_tokens too low; FG router not actually swapping; eos token mishandled; system prompt dropped under certain modes).

**Effect:** early demos showed Synapta as much weaker than its actual evaluation metrics suggested.

**Correction:** Fix shipped 2026-05-02. Smoke test ($n=20$) post-fix: 95\% pass across all 4 modes (base, single_best, token_routing, format_guard).

**Where:** `synapta_src/demo/synapta_live_demo.py` (current canonical demo); backup at `src/demo/server_original_backup.py`.

### 4. n=595 mid-run snapshot vs n=664 full sweep (briefly cited as final)
**Mid-run claim:** Some early-iteration docs cited `n=595 paired, McNemar p = 6.26 \times 10^{-44}` as the headline. This was the eval state at the moment a 4-hour timeout fired.

**Correction:** Full $n=664$ sweep completed; canonical numbers are now $b_{10}=14$, $b_{01}=219$, $p = 1.66 \times 10^{-48}$. All 5 live docs (YC application, SUBMIT_CHECKLIST, INDEX, BFSI_ADAPTER_FINAL, SYNAPTA_FINAL_SYNTHESIS) updated 2026-05-04.

**Where:** `11_bfsi_extract_eval.md`, `99_HEADLINE_NUMBERS.md`.

## Open limitations (acknowledged in paper)

### A. Single 30B base for Format Guard headline
Format Guard's main numbers are reported on Nemotron-Nano-30B-A3B only. The Code Paradox replicates on smaller bases; the FG mechanism does not. Reviewer concern: "is Format Guard a Nemotron-specific routing trick?"

**Mitigation in paper:** explicit limitation statement in §10.

### B. Heuristic router
The Format Guard router is regex over decoded suffix. Replacing with a learned router (e.g., classifier over suffix embedding) is natural follow-up. The MBPP $-3$ pp regression is partly attributable to over-triggering.

### C. Static-merge baselines tested at uniform weights only
DARE/TIES/linear merges are at uniform weights ($w_i = 1/N$). LoRAHub-style learned-weight composition not tested at 30B scale.

### D. Frontier API comparison is $n=15$ only
The Synapta-vs-Claude finding is directional, not statistically conclusive at $p<0.05$. Larger frontier comparison gated on API budget.

### E. Substring is not always the right metric
On the Synapta Benchmark v1 token-F1 split, both base and adapter score 0\% — F1$\geq$0.5 cutoff is too strict for our verbose answer style. The substring cleaner-win (80\% → 100\% on the substring half) is real, but the published benchmark may want F1$\geq$0.3 in v2.

### F. Two regulators only
RBI + SEBI. IRDAI (insurance), PFRDA (pensions), FEMA (cross-border) not represented.

### G. Synapta Benchmark v1 is 60 questions
Hand-curated, but small. Future versions should grow with community contributions.

## Bugs that have been on the radar but not investigated

### H. Format Guard $b_{10}=6, b_{01}=0$ asymmetry on BFSI
On the $n=664$ paired eval, FG differs from dedicated adapter on exactly 6 questions, all in $b_{10}$ (FG worse). $b_{01}=0$ — FG never improves over the dedicated adapter on BFSI. The 6 cases are documented in the paper appendix; root cause is the BFSI router falsely firing on `Section 24(1)(a)` style citations briefly misread as code-block openers.

**Possible fix (untried):** require two consecutive 10-token windows to disagree before swapping (debounce).

### I. PPL-Reactive router has best perplexity but no accuracy gain
A perplexity-reactive router achieved lowest avg perplexity on math (0.115) and HumanEval (0.102). However, **better perplexity did not translate to better accuracy**. This suggests adapter selection is not bottlenecked by confidence calibration on these tasks.

**Where:** `docs/MASTER_RESEARCH_CHRONICLES.md` §7.5.

## Files
- `docs/findings/humaneval_n164.md`
- `docs/findings/humaneval_n164_critical.md`
- `docs/findings/code_paradox_replication.md`
- `docs/findings/demo_diagnosis.md`
- `docs/findings/demo_server_bugs.md`
- `docs/findings/demo_verification.md`
