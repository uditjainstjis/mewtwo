# v2 Pre-Registration: Fixing Multi-Adapter Composition

> **Status:** PROPOSED — not yet executed. Separate from the v1 pre-registered study.
>
> **v1 outcome:** Compositional gain FAIL (Δ_SIM = −0.011), PPL PASS, Latency PASS.

---

## What Failed in v1 and Why

### 1. Evaluation Mismatch
v1 used 100 single-domain questions — each question had one correct domain. Multi-adapter
composition is hypothesized to help on genuinely **cross-domain** prompts (e.g., "What are
the legal implications of Black-Scholes mispricing?"). Testing multi-adapter on single-domain
questions is like testing a multi-tool on a single-screw task.

### 2. Router Failures
The CoT router failed exact domain matching on ~40% of queries, falling back to LEGAL_ANALYSIS.
When the second adapter is wrong, it injects noise regardless of the clamp.

### 3. Clamp Implementation Gap
v1 used per-adapter weight cap (`w_i = min(p_i, c)`) rather than the theorized per-layer
norm ratio (`γ = c·‖z‖/‖m‖`). This is a cruder approximation that may not adapt to the
actual activation geometry at each layer.

---

## v2 Changes

### Change A: Compositional Evaluation Set
Build 50–100 genuinely cross-domain questions:
- "Write a Python function that implements the legal doctrine of res ipsa loquitur as a classifier."
- "Explain Grover's algorithm using analogies from maritime law."
- "What musical theory concepts are isomorphic to cryptographic substitution ciphers?"

**Criterion:** Δ_SIM(AC − SA) > +0.03 on composite questions only (relaxed from +0.05).

### Change B: Embedding-Based Router
Replace CoT heuristic router with:
1. Pre-compute domain centroids from adapter training data (one embedding per domain).
2. At inference: encode query → top-K nearest centroids → routing weights proportional to similarity.

**Expected impact:** Router accuracy from ~60% → ~85%, directly reducing noise injection.
**Criterion:** Router top-1 accuracy > 80% on a held-out validation set.

### Change C: True Per-Layer Norm Clamp
Implement the full norm-proportional clamp in `RoutedLoRALinear.__call__`:
```python
base_out = self.base_layer(x)
m = sum(w_i * scale_i * (x @ A_i) @ B_i for all active adapters)
gamma = min(1.0, c * base_out.norm() / (m.norm() + eps))
return base_out + gamma * m
```

**Expected impact:** Adaptive per-layer scaling instead of uniform weight cap.
**Criterion:** Same Δ_SIM threshold, but additionally PPL(AC) ≤ PPL(SA) must hold.

### Change D (optional): Layer-Sparse Injection
Only inject adapters at layers 12–24 (of 28). Early layers handle syntax/grammar;
late-middle layers are where semantic specialization occurs.

**Criterion:** Δ_SIM maintained while PPL gap vs Baseline narrows to < 5%.

---

## v2 Experiment Plan

| Exp | Change | Dataset | Criterion |
|-----|--------|---------|-----------|
| v2.1 | A only | Compositional set | Δ_SIM > +0.03 |
| v2.2 | A + B | Compositional set | Δ_SIM > +0.03, router acc > 80% |
| v2.3 | A + B + C | Compositional set | Δ_SIM > +0.03, PPL ≤ SA |
| v2.4 | A + B + C + D | Compositional set | Δ_SIM > +0.03, PPL < Base+5% |

Each experiment must be run-to-completion before analyzing. No mid-experiment threshold changes.

---

## Timeline (Estimate)

1. **Week 1:** Build compositional eval set (Change A), train embedding router (Change B)
2. **Week 2:** Implement true norm clamp (Change C), run v2.1–v2.3
3. **Week 3:** Layer-sparse ablation (Change D), analyze, write v2 paper update
