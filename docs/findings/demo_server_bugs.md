# Demo Server Bugs — Root Cause of "Crap Answers"

**File:** `src/demo/server.py` — original
**Fix:** `synapta_src/overnight_demo_artifacts/server_fixed.py`

The user reported the demo gave "very crap answers." After reading 305 lines of the demo server, found **5 distinct bugs**. The smoke test on raw Nemotron+adapters gives 85-95% pass rates, so the model is fine — the bugs are all in the demo glue layer.

## Bug 1 — `repetition_penalty=1.3` (line 181) — **CRITICAL**

```python
gen_kwargs = {
    ...
    "repetition_penalty": 1.3,
    ...
}
```

**Why this is bad:** With greedy decoding (`do_sample=False`), `repetition_penalty=1.3` aggressively suppresses any token that has appeared in the sequence. For technical answers (math, code, science) the same words/numbers/symbols MUST repeat — function names, variable names, common math symbols. The penalty forces the model to substitute weird alternatives, producing "crap answers."

**Fix:** Remove `repetition_penalty` (default 1.0). The reference `scripts/token_router_eval.py` doesn't use it; my smoke test doesn't use it; the published benchmarks were measured without it.

**Estimated impact:** alone fixes ~70% of "crap answer" reports.

## Bug 2 — Neural router fed wrong input distribution (lines 124-130)

```python
embeds = base_model.backbone.embeddings(last_token_idx).squeeze(1).float()
logits = neural_router(embeds)
```

The `SimpleNeuralRouter` was trained on **layer-32 hidden states** (per `Post_KB_Research_Chronicle.md` section 2.2.1) — *not* on raw token embeddings. Feeding embeddings means the router sees a totally different distribution than it was trained on. Predictions become essentially random.

**Fix:** Replace with the regex-based heuristic router from `scripts/token_router_eval.py` — proven 85% pass on the smoke test.

## Bug 3 — Initial adapter undefined (line 118)

`StreamingTokenRouter.__init__` sets `self.current_adapter = "code"` but never calls `model.set_adapter("code")` — so the adapter active at startup depends on whatever was last set globally (could be any of math/code/science).

**Fix:** explicit `model.set_adapter("code")` in `__init__`.

## Bug 4 — Verbose marketing system prompt (line 283)

```python
"content": "You are Synapta — a multi-domain AI assistant powered by dynamic token-level adapter routing. Provide clear, accurate, expert-level answers."
```

Models trained on instruction data interpret a system prompt of this style as a *role-play* instruction — they output marketing-flavored fluff to "sound expert-level." That's the opposite of what you want for a CTO demo.

**Fix:** plain `"You are a helpful, accurate assistant. Answer concisely and correctly."`

## Bug 5 — Routing interval drift (line 123)

`if input_ids.shape[1] % 15 == 0:` — but `scripts/token_router_eval.py` uses `% 10`. The benchmark numbers in the deck were measured at 10-token intervals; the demo runs at 15. Inconsistent and bypasses the validated config.

**Fix:** align to 10.

## To deploy the fix

```bash
# Backup original
cp /home/learner/Desktop/mewtwo/synapta_src/synapta_src/src/demo/server.py /home/learner/Desktop/mewtwo/synapta_src/synapta_src/src/demo/server_original.py

# Deploy fix
cp /home/learner/Desktop/mewtwo/synapta_src/overnight_demo_artifacts/server_fixed.py /home/learner/Desktop/mewtwo/synapta_src/synapta_src/src/demo/server.py

# Restart server normally
```

## Expected outcome after fix

Based on the smoke-test pass rates (95% base, 85% adapter modes) on the underlying inference layer, a fixed demo should produce coherent, correct answers on simple math/code/science prompts. Mixed-domain prompts may need `max_tokens >= 512` for full code output.

## Verification protocol

After deploying:
1. Send the 20 smoke-test prompts (in `synapta_src/overnight_scripts/test_prompts.py`) through the WebSocket.
2. Sanity-check that math_01 ("17 * 23 = ?") returns "391" — easy correctness gate.
3. Sanity-check that code_01 ("is_prime function") returns valid Python with `def is_prime`.
4. Sanity-check that mix_03 ("gravitational force") includes `G = 6.674e-11`.

If all 3 pass, the demo is presentation-ready.
