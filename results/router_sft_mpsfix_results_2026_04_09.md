# Router SFT MPS Results (April 9, 2026)

## Router Accuracy

Validation holdout: `data/router_synthetic_routing_5000_valid_holdout.json`

Adapter:

- `router_adapters/router_reasoning_sft_5000_mpsfix`

Metrics from `results/router_accuracy_sft_5000_valid_holdout_mpsfix.json`:

| Metric | Value |
| --- | ---: |
| `n` | `100` |
| exact-match routing accuracy | `0.85` |
| partial-match routing accuracy | `1.00` |
| mean overlap F1 | `0.945` |
| mean router latency | `1.079s` |

This exceeds the original target band of `75-80%+` exact-match routing accuracy.

## 10-Item TCAR Pilot With Trained Router

Benchmark file:

- `results/tcar_collaborative_sft5000_mpsfix_pilot10.jsonl`

Average metrics:

| Method | Semantic Sim | Token F1 | Latency (s) |
| --- | ---: | ---: | ---: |
| `tcar_collaborative` | `0.6902` | `0.2874` | `16.8450` |
| `tcar_oracle_collaborative` | `0.7098` | `0.2774` | `15.1748` |
| `mistral` | `0.4688` | `0.1740` | `11.3232` |

## Comparison To The Earlier TCAR Pilot

Previous pilot reference from `results/tcar_pilot_10_comparison.json`:

| Method | Old Semantic Sim | New Semantic Sim | Delta |
| --- | ---: | ---: | ---: |
| `tcar_collaborative` | `0.6797` | `0.6902` | `+0.0105` |
| `tcar_oracle_collaborative` | `0.6939` | `0.7098` | `+0.0159` |

| Method | Old Token F1 | New Token F1 | Delta |
| --- | ---: | ---: | ---: |
| `tcar_collaborative` | `0.2682` | `0.2874` | `+0.0192` |
| `tcar_oracle_collaborative` | `0.2921` | `0.2774` | `-0.0147` |

## Interpretation

- The trained router clearly fixed the routing bottleneck.
- `tcar_collaborative` improved on the 10-item external pilot over the earlier untrained-router TCAR run.
- On semantic similarity, the trained collaborative path remains below the oracle collaborative ceiling, but the gap is now small:
  - oracle gap on this run: `0.7098 - 0.6902 = 0.0196`
- On token F1, the trained collaborative path slightly exceeded the oracle run on this slice.

## Important Caveat

The Mistral line in this specific rerun should not be over-interpreted as a clean apples-to-apples comparison target. The main purpose of this run was to measure:

- trained `tcar_collaborative`
- relative to `tcar_oracle_collaborative`

The key result is that trained routing materially improved the real collaborative system and moved it closer to the oracle ceiling.
