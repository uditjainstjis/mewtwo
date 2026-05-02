# Overnight Run — Status Log

Started: 2026-05-02 23:29 UTC
Final iter: 11 (04:03 UTC, ~4.5 hours elapsed)

## Iteration Log (condensed)

- **Iter 1** — bootstrap
- **Iter 2** — smoke test 80 gens: base 95%, adapters 85%
- **Iter 3** — HumanEval midcheck
- **Iter 4** — caught extract_code import-stripping bug; wrote server_fixed.py
- **Iter 5** — caught indent-stripping bug; partial base rescored 25%→65%
- **Iter 6** — partial big result (FG early sample 84%)
- **Iter 7** — HumanEval COMPLETE n=164: base 56.1%, FG 73.2%, +17.1 delta
- **Iter 8** — Code Paradox n=50 replication (apparent 3/3)
- **Iter 9** — polish smoke (max=512) all 4 modes 95%; FINAL_SUMMARY.md written
- **Iter 10** — Qwen-0.8B n=200 launch
- **Iter 11** — Qwen-0.8B n=200 COMPLETE → **n=50 cross-family Code Paradox does NOT replicate at n=200**. Updated findings honestly. Final docs.

## Completed jobs

| job | result |
|---|---|
| smoke (80 gens, max=256) | base 95% / adapters 85% |
| humaneval_n164 v2 (328 gens) | base 56.1%, FG 73.2%, **+17.1 delta** at n=164 |
| code_paradox n=50 (300 gens) | n=50 fluke — does not survive scaling |
| demo_polish (80 gens, max=512) | **all 4 modes 95%** |
| code_paradox_qwen_n200 (600 gens) | base 15%, math 16%, code 12% — **paradox does NOT replicate at n=200** |

## Findings (final)

- demo_diagnosis.md (Iter 2)
- demo_server_bugs.md (Iter 4) — 5 bugs + drop-in fix at demo_artifacts/server_fixed.py
- humaneval_n164_critical.md (Iter 5) — bug discovery
- humaneval_n164.md (Iter 7) — base 56.1%, FG 73.2%, +17.1 at n=164
- demo_verification.md (Iter 9) — side-by-side Q&A 80 generations
- code_paradox_replication.md (Iter 11) — **honest update; cross-family rolled back**

## Master files

- FINAL_SUMMARY.md — master summary
- WAKE_UP_README.md — quick start
- STATUS.md — this file

## Loop status

GPU is free. All planned work done. Loop will continue auto-waking but with no more substantive work to do — will either stop on user signal or idle until user types STOP.

## Stop signal

Listening for: WAKE / STOP / PAUSE.
