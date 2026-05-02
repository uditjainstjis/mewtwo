# Wake-up Summary — FINAL

**Status:** All planned work done. GPU released. Loop scheduled to wake once more then idle.
**Last update:** 2026-05-02 04:03 UTC (~4.5 hours after start)

## **READ FIRST: `FINAL_SUMMARY.md`**

## Three wins, one honest rollback

### ✅ HumanEval n=164 with corrected scoring
- Base **56.1%**, Format Guard **73.2%**, **delta +17.1 pts**
- Replaces deck's n=25 (24% / 48% / +24)
- Both absolute numbers ~doubled — base 56% on a 30B is competitive with CodeLlama-34B class

### ✅ Demo bugs fixed
- 5 bugs in `src/demo/server.py`. Drop-in fix at `overnight_run/demo_artifacts/server_fixed.py`
- Polish smoke verified all 4 modes at 95% pass rate (was 85% with smaller token budget)

### ✅ Token routing efficiency (new deck claim)
- 35% shorter outputs, 35% faster wall-clock at same correctness
- "Same accuracy, 35% less compute cost per query"

### ❌ Code Paradox cross-family rolled back (HONEST)
- n=50 said code beats math on Qwen-0.8B (+6 pp)
- **n=200 says code is WORSE by 4 pp** — n=50 was lucky sample
- Code Paradox holds robustly only at **n=200 on Nemotron-30B (+5.5 pp)**
- Updated `findings/code_paradox_replication.md` with honest version
- The +5.5 pp single-model claim still stands — drop the cross-family overclaim

## Updated benchmark grid for the deck

| Method | ARC | MATH-500 | HumanEval | MBPP |
|---|---|---|---|---|
| Base Nemotron-30B | 20% | 41.5% | **56.1%** | 8% |
| Static Merge (DARE/TIES) | 19% | 56% | 34% | 0% |
| Best Single Adapter | 31% | 56% | 60% | 6% |
| **Our Format Guard** | **31%** | **56%** | **73.2%** | 5% |

Lift vs base: ARC +11 / MATH +14.5 / **HumanEval +17.1** / MBPP -3.

## Three things to do when you wake

1. **Update deck slide 4** with the new HumanEval row (56.1% / 73.2%). Run `python build_pitch_deck.py` to regenerate.
2. **Soften Code Paradox slide** — claim n=200 Nemotron-30B only. Drop cross-family.
3. **Deploy demo fix** — two cp commands (see above).

## Read in this order

1. **`FINAL_SUMMARY.md`** — master summary
2. `findings/humaneval_n164.md` — defensible HumanEval numbers + grid
3. `findings/code_paradox_replication.md` — **HONEST n=200 update**
4. `findings/demo_server_bugs.md` — server fix details
5. `findings/demo_diagnosis.md` — efficiency story
6. `findings/demo_verification.md` — side-by-side Q&A (35KB, browsable)
7. `STATUS.md` — chronological log

## To stop the loop

Type STOP / PAUSE / WAKE.

## Findings index

- demo_diagnosis.md (Iter 2)
- demo_server_bugs.md (Iter 4) — drop-in fix at demo_artifacts/server_fixed.py
- humaneval_n164_critical.md (Iter 5) — bug discovery
- humaneval_n164.md (Iter 7) — base 56.1%, FG 73.2%, +17.1 at n=164
- code_paradox_replication.md (Iter 11, **honest update**) — robust only at n=200 Nemotron-30B
- demo_verification.md (Iter 9) — side-by-side Q&A
- FINAL_SUMMARY.md (Iter 11) — master summary
