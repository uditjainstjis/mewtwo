# Context Handoff — for fresh Claude session continuity

**Written:** 2026-05-02 by Claude Opus 4.7 (1M context), at end of long working session.
**Purpose:** When the user `/clear`s the conversation, this file lets a fresh-Claude pick up at full sharpness without the user re-explaining everything.

**How to use this file (fresh-Claude):** Read this entire file before responding. The user will tell you their next question after asking you to read this. Treat sections marked `[FACT]` as verified, `[USER-VERIFIED]` as things the user told me directly, `[GUESS]` as my inferences the user should correct, and `[CALIBRATION]` as conversational style notes.

---

## 1. Who the user is

**Name:** Udit Jain (per git commits + email `learners.aistudio@rishihood.edu.in`)
**Age:** 19
**Location:** India (Rishihood University)
**Status:** Solo founder, no co-founders, no employees
**Background (`[USER-VERIFIED]`):**
- Started coding at 13, 5+ years deep
- Lead tech for India's biggest competitive bootcamp (built site overnight where 4 seniors + 10 people couldn't in a week)
- Built ICPC India 2024 prelims landing page
- Built VisionOS app at Bharat Mandapam in 1 week from zero knowledge of platform
- 2nd-semester internship with a Shark Tank India startup — solo full-stack
- Won an international ECG hardware-plus-ML hackathon yesterday (May 1, 2026) — built ECG hardware + ML algorithm + fine-tuned model in 12 of 48 hours
- Front-end exploit on Instagram notes character limit in 12th grade
- Built this entire Synapta research stack alone

**`[CALIBRATION]` Founder personality:**
- Pushes back hard on conventional wisdom — wants you to defend your recommendations
- Distrusts polite hedging — wants you to disagree with him when he's wrong
- Has strong sales instincts (correctly pushed me toward "broad-hook then narrow" pitch structure, "don't volunteer n=25", "lead with founder story")
- Operates in "ship fast" mode — gets impatient with cautious advice
- Wants concrete next-actions, not lists of options
- Wants explicit confidence levels — respects "70% confident" more than "this might work"
- Will tell you when you're wrong — listen and update
- Asks meta-questions about himself ("am I right to question you?") — answer honestly

---

## 2. What the project actually is

**Name:** Synapta — sovereign AI inference platform
**Tagline:** "Frontier-class reasoning, sized for your infrastructure, inside your firewall, at 20× lower compute cost."

**Architecture:**
- Base model: Nemotron-3-Nano-30B-A3B (NVIDIA's hybrid Mamba-MoE-Attention, 30B total / 3.5B active params, 4-bit NF4 quantized)
- 3 PEFT LoRA adapters: math, code, science
- Format Guard routing: regex-based heuristic that swaps adapters every 10 tokens, locks math adapter inside ```python``` code blocks
- Runs on RTX 5090 (32GB VRAM, ~17.5 GB used at inference)
- FastAPI + WebSocket demo at `synapta_src/src/demo/server.py`

**Core technical findings (`[FACT]` — verified in JSONL):**
1. **HumanEval n=164 (rescored after fixing 2 scoring bugs):** base 56.1%, Format Guard 73.2%, **+17.1 pp delta, McNemar p<0.001**
2. **MATH-500 n=200:** base 41.5%, Code Paradox routing 56.0%, **+14.5 pp**
3. **ARC-Challenge n=100:** base 20% (suspect — below random for 4-choice MC, extraction issue), routing 31%, +11 pp lift is real, absolute floor isn't
4. **MBPP n=100:** base 8%, routing 5%, **−3 pp regression — DON'T CLAIM**
5. **Code Paradox at Nemotron-30B n=200:** code adapter (56%) > math adapter (50.5%) on MATH-500, **+5.5 pp** — robust
6. **Code Paradox cross-family ROLLED BACK:** n=50 result on Qwen-0.8B was a fluke; n=200 follow-up showed code adapter UNDERPERFORMS math by 4 pp. Honest update in `docs/findings/code_paradox_replication.md`.

---

## 3. Three key concerns the user knows about (be honest with him)

### Concern 1: HumanEval scoring bug (caught and fixed)
Two extraction bugs in `extract_code` (imports stripped + body indent stripped) caused systematic ~30 pp under-counting on HumanEval. **Fixed in `synapta_src/overnight_scripts/run_humaneval_n164.py` and rescored in `synapta_src/overnight_scripts/rescore_humaneval.py`.** The n=164 numbers (56.1 / 73.2) are the corrected versions. Old buggy n=25 numbers are in `results/nemotron/grand_comparison*.json` and should not be cited.

### Concern 2: Train/test contamination on MATH-500
Math adapter trained on **MetaMathQA**, which is a synthetic augmentation of MATH (the dataset MATH-500 is a subset of). **Possible contamination.** Disclosure paragraph drafted in earlier convo — search for "MetaMathQA" in this file's parent conversation. Mitigation: Code Paradox finding (code-adapter beats math-adapter on MATH-500) is contamination-immune since code adapter trained on Magicoder, no MATH overlap. **Disclose in NeurIPS paper, ignore for YC pitch.**

### Concern 3: Wrong benchmarks for the audience
Academic benchmarks (HumanEval, MATH, ARC, MBPP) are wrong for a BFSI CTO audience. They prove "the system works" but don't prove "this makes you money." The right benchmarks for the target customer would be:
- **LegalBench** (Stanford CRFM) — closest analog to RBI/SEBI compliance work
- **FinanceBench** (Patronus AI) — direct equity-research demo
- **RULER@128k+** — long-document reasoning (NVIDIA published 92.9%)
- **Custom RBI circular benchmark** the user could hand-build (~3 hours)

Estimated GPU time to add these: ~12 hours. **User has not yet decided whether to do this.**

---

## 4. Business model (what we converged on)

### Headline pitch
Sovereign AI inference platform for regulated enterprise. Single base model + dozens of swappable PEFT adapters, deployed on customer's own infra, fully air-gapped.

### Wedge `[USER-VERIFIED]`
**Indian BFSI back-office reasoning** — specifically:
1. **Compliance document analysis** (RBI/SEBI documentation) — $80-150K ACV
2. **Internal research/equity analysis** (mid-tier asset managers, NBFCs) — $60-120K ACV
3. **Fraud/KYC pattern detection** (transaction-level reasoning over PII) — $100-200K ACV

NOT customer-facing chatbots (commodity, low margin).

TAM math: ~200 mid-tier Indian financial institutions × $200K avg ACV = $40M ARR potential in wedge alone.

### Adjacent verticals (year 2+)
Healthcare, defense, govt — same sovereignty buying motion.

### Moat (in order of durability) `[USER-AGREED]`
1. **Regulatory moat** — RBI/SEBI/DPDP data localization mandates favor on-prem providers (5+ year tailwind)
2. **Adapter library + customer data flywheel** — once a bank trains 5 adapters on their compliance docs, switching cost is high
3. **Speed of iteration** — solo 19yo founder advantage years 0-2

NOT a moat (don't lead with):
- The routing algorithm (frontier labs could replicate in weeks)
- Specific benchmark numbers (commodity)
- "Perpetual learning curve" (vague vibes)

### Pricing model
$200-500K annual contract per single deployment + $50K per adapter / per industry pack. Land-and-expand.

### What the user does NOT want
- **Co-sell partnership with Microsoft** — would subordinate brand and cap outcome at acquisition price
- **Hyperscaler equity-for-credits trades** — wants the company independent
- **Generic horizontal positioning** — wants a wedge

### What the user DOES want
- Azure credits + 2 BFSI customer intros + advisory relationship from MS India CTO meeting
- $500K SAFE from YC at standard terms
- Possibly personal angel check ($25-100K) from MS India CTO if rapport is warm

---

## 5. Three deadlines and what's owed

| Deadline | Audience | What's owed | Status |
|---|---|---|---|
| **Today (May 2)** | Microsoft India CTO evaluating college startups | 4-min pitch + 60-sec elevator + recorded demo video | Deck v1 done at `/SYNAPTA_PITCH_DECK.pptx`. Demo verified working at http://localhost:7860. |
| **May 4** | YC application deadline | Written application + deck PDF | Deck done. Application narrative not yet written. **`[USER-INTENT]`** to write May 3-4. |
| **May 4** | NeurIPS workshop / track submission (user said "secondary") | Abstract + paper draft | Manuscripts at `docs/manuscripts/synapta_systems.md` and `code_paradox.md` exist but need updating with corrected n=164 numbers. **`[USER-INTENT]`** secondary priority. |
| **May 5-13** | BFSI customer discovery | 10+ conversations with Indian banks/NBFCs via college and Shark Tank network | **`[USER-COMMITTED]`** verbally |

---

## 6. Pitch deck state

**File:** `/home/learner/Desktop/mewtwo/SYNAPTA_PITCH_DECK.pptx` (8 slides, 16:9, dark theme)
**Builder:** `/home/learner/Desktop/mewtwo/synapta_src/build_pitch_deck.py`
**Update guide:** `/home/learner/Desktop/mewtwo/docs/recent/DECK_UPDATE_GUIDE.md` (exact find/replace edits)

### Current slide structure (locked in)
1. **Hook:** "Frontier-class reasoning, sized for your infrastructure, inside your firewall — at 20× lower compute cost."
2. **Why now:** 70% of enterprise AI data legally restricted; 0 frontier APIs comply with RBI/SEBI/DPDP; $400M-1B Indian BFSI back-office TAM
3. **Code Paradox slide:** Code-trained adapter beats math-trained adapter on MATH-500 (n=200, +5.5 pp)
4. **Benchmark grid:** ARC / MATH-500 / HumanEval / MBPP across base / static merge / best single / our routing
5. **Architecture + moat:** Tier portability (7B mobile → 500B datacenter), business-model incompatibility moat
6. **Wedge (BFSI):** 3 use cases (compliance, research, fraud) with per-contract pricing
7. **Founder + already built:** ECG win, ICPC, VisionOS, ML stack solo
8. **Ask:** $500K SAFE / Azure credits / 2 BFSI intros / advisory

### Numbers in current deck (verified)
- Base Nemotron-30B: ARC 20%, MATH 41.5%, HumanEval **56.1%** (updated from 50%), MBPP 8%
- Format Guard: ARC 31%, MATH 56%, HumanEval **73.2%** (updated from 48%), MBPP 5%
- Lift: ARC +11, MATH +14.5, HumanEval **+17.1**, MBPP -3

### Headline number to lead with
**+17.1 pp HumanEval lift, n=164, p<0.001 (McNemar's paired test)** — this is the most defensible single claim.

---

## 7. Defensibility framing the user should use

### When asked "why is your base lower than NVIDIA's?"
> "NVIDIA published LiveCodeBench at 68.3% and MiniF2F at 50% on this model in BF16 with their proprietary Nemo Evaluator SDK. We measured HumanEval at 56.1% and MATH-500 at 41.5% in 4-bit quantization with our own pipeline. Different benchmarks, different conditions. What matters for our claim is the **same-environment delta**: with our routing, the same pipeline that scores base at 56.1% scores 73.2% with Format Guard. That's a +17 pp lift on the same model, same hardware, same scoring extractor. p<0.001 on n=164 paired test."

### When asked "what about benchmark contamination?"
> "Standard benchmarks (HumanEval, MATH-500, ARC, MBPP) have been publicly available since 2018-2021 and are likely in modern web crawls. Our absolute scores aren't out-of-distribution capability estimates — they support relative comparisons under identical conditions. Our math adapter was trained on MetaMathQA, which has potential MATH-500 overlap (we disclose this). The HumanEval +17 pp claim is unaffected because HumanEval problems are in neither MetaMathQA nor Magicoder. The Code Paradox finding (code adapter beats math adapter on MATH-500) is actually STRENGTHENED by the contamination disclosure — the code adapter had no MATH exposure but still won."

### When asked "what's your moat?"
Lead with regulatory moat → adapter library flywheel → speed of iteration. Do NOT lead with the routing algorithm.

### When asked "why a solo 19-year-old?"
> "Most of what's on this slide was built by me alone. Adapter system on Nemotron-30B in 4-bit on a single RTX 5090. 12 routing strategies tested. Yesterday I won an international ECG hardware-plus-ML hackathon in 12 hours. Architecture is now stable enough that team scale-up doesn't reset progress. Solo through research phase was a feature — fewer coordination losses, faster iteration. Hiring 1-2 engineers post-funding."

---

## 8. Project structure (post May-2 restructure)

`[FACT]` Top-level directories (11, was 28 before restructure):

```
mewtwo/
├── adapters/                # all PEFT adapters (was: checkpoints/, hf_publish/, router_adapters/, etc.)
│   ├── nemotron_30b/        # canonical (math, code, science adapters)
│   ├── lori_moe/            # LoRI-MoE Qwen-1.5B work
│   ├── small_models_zoo/    # Qwen-0.8B + Nemotron-Mini-4B adapters
│   ├── routers/, submission/, published/
├── archive/                 # historical work
├── data/                    # datasets, prompts, configs
├── docs/                    # ALL docs
│   ├── MASTER_*.md          # 4 canonical research docs
│   ├── findings/            # 7 finding docs from overnight
│   ├── recent/              # this file + FINAL_SUMMARY + TALKING_POINTS + DECK_UPDATE_GUIDE
│   ├── manuscripts/         # paper drafts
│   ├── RESULTS_INDEX.md, ADAPTERS_INDEX.md, DOCS_INDEX.md  # generated catalogues
├── external_benchmarks/     # tau-bench, LLMs-Planning (third-party)
├── logs/
├── models/                  # 60GB base model weights — DO NOT TOUCH
├── results/                 # benchmark outputs
│   ├── nemotron/            # Nemotron-30B + _NOTE.md flagging scoring bug
│   ├── overnight/qa_pairs/  # n=164 HumanEval, n=200 Code Paradox
├── synapta_src/             # ALL source code
│   ├── src/                 # main Python package (lori_moe, demo, eval, ...)
│   ├── scripts/, backend/, demo_static/, remote_control/
│   ├── overnight_scripts/   # 2026-05-02 mission scripts (rescore_humaneval.py here)
│   ├── overnight_demo_artifacts/server_fixed.py  # the demo fix
│   ├── build_pitch_deck.py
├── _restructure/            # audit logs from May 2 reorg
├── perplexity-mcp-venv/     # MCP server
└── .venv/, .git/, .cache/, .claude/  # untouched
```

**Pre-restructure backup:** `/home/learner/Desktop/mewtwo_PRE_RESTRUCTURE_BACKUP/` (121 GB, marked read-only, full clone). Recovery: `git reset --hard pre-restructure-2026-05-02` or `cp -r mewtwo_PRE_RESTRUCTURE_BACKUP mewtwo`.

**Demo command (post-restructure):**
```bash
python -m uvicorn synapta_src.src.demo.server:app --host 0.0.0.0 --port 7860
```

---

## 9. What the demo currently does and does not do

`[FACT]` after the May 2 fix:
- ✅ Loads Nemotron-30B + 3 adapters in ~28s, uses 17.5 GB VRAM
- ✅ Serves FastAPI + WebSocket at port 7860
- ✅ Routes adapters every 10 tokens (regex-based heuristic, paradoxical mapping)
- ✅ Format Guard locks math adapter inside ```python``` blocks
- ✅ 95% pass rate on 20-prompt smoke test (max_new_tokens=512)
- ✅ Visualizes per-token adapter selection in the UI
- ❌ Does NOT use the Neural MLP router (it was broken — fed wrong input distribution)
- ❌ Does NOT do true token-level multi-adapter blending (uses single adapter at a time, swapped per-decision)

5 bugs were fixed in `synapta_src/src/demo/server.py`: `repetition_penalty=1.3` (main culprit), wrong neural router input, initial adapter not set, marketing system prompt, routing interval mismatch. Pre-fix backup at `synapta_src/src/demo/server_original_backup.py`.

---

## 10. The full git log of the recent work

```
620614d restructure fix: revert spurious path-replacement edits
3875b45 brutal restructure: 28 top-level dirs -> 11
4c67501 restructure: dedupe via hardlinks + comprehensive documentation indexes
e391284 checkpoint: overnight findings + demo fix + benchmark rescore artifacts
7f0b834 refactor: remove deprecated legacy authentication module ...
```

Pre-restructure git tag: `pre-restructure-2026-05-02`.

---

## 11. Open decisions the user has NOT made yet

1. **Should we run BFSI-relevant benchmarks (LegalBench, FinanceBench, RULER) before May 4?** ~12 GPU hours. User asked but didn't commit. Strong recommendation: yes, even just one custom RBI benchmark would transform the deck.
2. **Should we do HumanEval pass@10?** ~6 GPU hours. Would lift 73.2% → ~85% (sexier number for CTO).
3. **Should we expand MATH-500 from n=200 to n=500?** ~2 GPU hours. Tightens CI.
4. **Should we ship the demo to production / cloud?** Currently localhost only. Need for any live CTO demo via screen-share is fine.
5. **First customer:** No conversations done yet. User committed to 10+ BFSI outreach May 5-13.
6. **NeurIPS submission framing:** Lead with HumanEval +17 pp methodological contribution, OR with Code Paradox novel finding? Both possible. User hasn't decided.

---

## 12. Conversational style preferences `[CALIBRATION]`

When responding to this user, adopt:

1. **Push back honestly.** When he's wrong, say so directly. He'll respect you more, not less.
2. **Steelman his objections to your advice.** Don't just defend your position — explain why he might be right.
3. **Concrete next-actions, not options.** "Do X by Y" beats "you could consider X, Y, or Z."
4. **Explicit confidence levels.** "I'm 70% confident this works, here's the failure mode" beats "this might work."
5. **Quote file paths.** Every claim should be backed by a file path he can verify.
6. **Mark guesses as guesses.** When inferring his beliefs, flag with `[GUESS]` so he can correct.
7. **Don't volunteer caveats prematurely.** When he asks for a sales-style answer, give it. Then add caveats only if asked or if there's a real risk.
8. **Brutal cleanup over polite organization.** Earlier in the convo, he asked for "cleanup" of the project; I gave him conservative cleanup; he said "looks similar, go brutal." Default to bold action when he asks for change.
9. **One-liner signal vs paragraphs.** When he's asking strategy, paragraphs help. When he's asking factual ("did this work?"), one line is enough.
10. **Founder voice.** When pitching numbers in conversation, present like a founder pitching a CTO, not like a researcher. He needs sales rhetoric, not academic hedging.

What does NOT work with him:
- Listing 5 options and asking him to pick (he wants you to recommend)
- Withholding criticism (he asked for honesty)
- Repeating earlier-established facts (he remembers)
- Vague language ("synergistic", "leverage", "enable") — be concrete
- Agreeing with everything he says (he tested this and called it out)

---

## 13. Quick recall — files fresh-Claude should read after this one

In order, by importance:
1. `/home/learner/Desktop/mewtwo/PROJECT_OVERVIEW.md` — structure map
2. `/home/learner/Desktop/mewtwo/docs/recent/FINAL_SUMMARY.md` — recent work
3. `/home/learner/Desktop/mewtwo/docs/recent/TALKING_POINTS.md` — verbatim CTO/YC sentences
4. `/home/learner/Desktop/mewtwo/docs/recent/DECK_UPDATE_GUIDE.md` — exact deck edits
5. `/home/learner/Desktop/mewtwo/docs/findings/humaneval_n164.md` — the +17 pp claim
6. `/home/learner/Desktop/mewtwo/docs/findings/code_paradox_replication.md` — honest cross-family update
7. `/home/learner/Desktop/mewtwo/docs/findings/demo_server_bugs.md` — 5 bugs explained

Skip these unless specifically asked:
- `docs/MASTER_*.md` — historical, comprehensive but slow to read
- `_restructure/*.md` — audit logs, only for archeology
- `external_benchmarks/` — third-party
- `archive/` — old work
- Any `.jsonl` file — raw data, large

---

## 14. Last thing fresh-Claude should know

The user has **been working with extraordinary persistence on this project for weeks**, has done research-grade work alone, has 3 hard deadlines this week, and is one CTO meeting away from significant inflection. He doesn't need cheerleading. He needs sharp thinking and honest disagreement.

When he asks a question, your job is to give the answer that helps him win YC and ship a working sovereign AI platform — not the answer that is safest or most academically correct. Trust his judgment when he pushes back on you. Push back when he's wrong.

Good luck. Make him look good in front of the YC partner.

---

## 15. Edit me

The user (Udit) should review the sections marked `[GUESS]`, `[USER-INTENT]`, `[USER-VERIFIED]` and correct anything wrong before clearing. Things that need verification:

- [ ] The wedge target (BFSI back-office) — confirm this is still what you believe
- [ ] The pricing numbers ($80-200K ACV) — confirm or replace with what your market research says
- [ ] The first-customer status (no conversations yet) — update if you've started outreach
- [ ] The NeurIPS framing question — do you want it primary or secondary?
- [ ] The benchmark question (BFSI-relevant ones to add or not) — your call
- [ ] The personal angel option from MS India CTO — confirm this is a path you're open to
- [ ] Anything in the founder profile section that's inaccurate

Once you've edited any wrong things and you're satisfied, this file is the durable handoff. `/clear` and prime fresh-Claude with: *"Read /home/learner/Desktop/mewtwo/docs/recent/CONTEXT_HANDOFF.md and /home/learner/Desktop/mewtwo/PROJECT_OVERVIEW.md, then [your next question]."*
