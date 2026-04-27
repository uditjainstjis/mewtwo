#!/usr/bin/env python3
"""
Fast routed-profile evaluator against the live demo server.

Runs a small benchmark slice plus three hard OOD prompts with concrete grading:
- MATH-500 slice
- HumanEval slice
- OOD math exact-answer prompt
- OOD science numeric/rule-based prompt
- OOD code prompt with executable tests
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT = Path("/home/learner/Desktop/mewtwo")
sys.path.insert(0, str(PROJECT))

from scripts.checkpointed_server_benchmarks import (
    extract_humaneval_code,
    extract_math_answer,
    generate_once,
    load_benchmark_examples,
    load_status,
    math_equal,
    normalize_ws_url,
    run_humaneval_check,
    score_example,
)


RESULTS_ROOT = PROJECT / "results" / "routing_profile_eval"


PROFILES: dict[str, dict[str, Any]] = {
    "base": {
        "mode": "naked",
        "adapter": "code",
        "chunk_size": 16,
    },
    "single_math": {
        "mode": "single",
        "adapter": "math",
        "chunk_size": 16,
    },
    "single_code": {
        "mode": "single",
        "adapter": "code",
        "chunk_size": 16,
    },
    "single_science": {
        "mode": "single",
        "adapter": "science",
        "chunk_size": 16,
    },
    "routed_baseline": {
        "mode": "routed",
        "adapter": "code",
        "chunk_size": 16,
        "routing_interval": 16,
        "min_tokens_before_swap": 0,
        "domain_hint_strength": 0.0,
        "swap_margin": 0.15,
        "lock_prompt_domain": False,
        "prompt_anchor": False,
    },
    "routed_anchored": {
        "mode": "routed",
        "adapter": "code",
        "chunk_size": 8,
        "routing_interval": 24,
        "min_tokens_before_swap": 64,
        "domain_hint_strength": 0.18,
        "swap_margin": 0.25,
        "lock_prompt_domain": True,
        "prompt_anchor": True,
    },
    "routed_anchored_strict": {
        "mode": "routed",
        "adapter": "code",
        "chunk_size": 8,
        "routing_interval": 32,
        "min_tokens_before_swap": 96,
        "domain_hint_strength": 0.28,
        "swap_margin": 0.3,
        "lock_prompt_domain": True,
        "prompt_anchor": True,
    },
}


OOD_MATH_PROMPT = (
    "Let a cyclic word of length 15 over the alphabet {0,1,2,3} be called harmonic if\n\n"
    "|x1-x2| + |x2-x3| + ... + |x15-x1|\n\n"
    "is divisible by 5.\n\n"
    "A cyclic word is primitive if it is fixed by no nontrivial rotation.\n\n"
    "How many primitive harmonic cyclic words of length 15 exist?\n\n"
    "Give only the final answer in \\boxed{}."
)
OOD_MATH_GOLD = "14640260"

OOD_SCIENCE_PROMPT = (
    "A 3-die inference module has a logic die, an SRAM die, and a SerDes die stacked vertically.\n\n"
    "Ambient temperature: 35 C\n\n"
    "Thermal resistances:\n"
    "- logic -> SRAM: 0.18 C/W\n"
    "- SRAM -> SerDes: 0.22 C/W\n"
    "- SerDes -> lid: 0.15 C/W\n"
    "- lid -> ambient: 0.55 C/W\n\n"
    "At core clock f (GHz) and lane rate r (Gb/s per lane), powers are:\n"
    "- P_logic = 12 + 6f + 0.5f^2\n"
    "- P_SRAM  = 4 + 1.2f\n"
    "- P_IO    = 2.5 + 0.30r\n\n"
    "Nominal supply is 1.02 V.\n"
    "Current draw is I = (P_logic + P_SRAM + P_IO) / 1.02.\n"
    "Package-path resistance is 3 mOhm, so logic-die voltage is:\n"
    "- V_logic = 1.02 - 0.003 I\n\n"
    "Timing limit:\n"
    "- f <= 8.0 * (V_logic - 0.62) * exp(-0.005 * (T_logic - 55))\n\n"
    "Signal-integrity limit:\n"
    "- eye_ps = 40 - (r - 18) - 0.08 * (T_IO - 55)\n"
    "- require eye_ps >= 12\n\n"
    "There are 64 lanes with 128/130 encoding.\n\n"
    "Question:\n"
    "If the core clock must be at least 1.90 GHz, is a payload bandwidth of 160 GB/s feasible?\n"
    "If yes, what is the maximum payload bandwidth at the 1.90 GHz floor, and which constraint binds first?\n\n"
    "Answer concisely."
)

OOD_CODE_PROMPT = (
    "Write Python code only.\n\n"
    "Implement:\n\n"
    "def min_cost_phase_echo_path(\n"
    "    n: int,\n"
    "    edges: list[tuple[int, int, int, int, int]],\n"
    "    L: int,\n"
    "    target_reg: int\n"
    ") -> tuple[int, list[int]]:\n\n"
    "You start at vertex 0 and must end at vertex n-1 after exactly L edge traversals.\n\n"
    "Each directed edge is (u, v, w, t, b):\n"
    "- w is a positive cost\n"
    "- t is a 5-bit integer in [0, 31]\n"
    "- b is a phase bit in {0, 1}\n\n"
    "State starts as reg = 0 and phase = 0.\n"
    "When traversing an edge:\n"
    "- new_reg = (((reg ^ t) << 1) & 31) | phase\n"
    "- new_phase = phase ^ b\n\n"
    "A walk is valid iff after exactly L edges:\n"
    "- current vertex is n-1\n"
    "- reg == target_reg\n"
    "- phase == 0\n\n"
    "Return the minimum total cost and the lexicographically smallest vertex sequence among all minimum-cost valid walks.\n"
    "If impossible, return (-1, []).\n"
)


def artifact_flags(text: str) -> dict[str, bool]:
    return {
        "think": bool(re.search(r"<think>|</think>", text, re.I)),
        "role_leak": bool(re.search(r"(^|\n)(User:|System:)", text, re.I)),
        "self_correct": bool(re.search(r"(let's check|verify|double-check|reconsider|actually|wait)", text, re.I)),
        "textbook": bool(re.search(r"(Figure\s+\d|Chapter\s+\d|Chapter summary|Example\s+\d+\.\d+)", text, re.I)),
    }


def score_ood_math(response: str) -> tuple[bool, str]:
    pred, _ = extract_math_answer(response)
    return math_equal(pred, OOD_MATH_GOLD), pred


def score_ood_science(response: str) -> tuple[bool, dict[str, Any]]:
    text = response.lower()
    feasible = "feasible" in text or bool(re.search(r"\byes\b", text))
    timing = "timing" in text
    signal = "signal" in text or "eye" in text
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", response)]
    near_max = any(abs(x - 297.67) <= 2.0 for x in numbers)
    return feasible and near_max and timing and not ("not feasible" in text), {
        "numbers": numbers[:10],
        "feasible": feasible,
        "near_297_67": near_max,
        "mentions_timing": timing,
        "mentions_signal": signal,
    }


def brute_force_phase_echo_path(
    n: int,
    edges: list[tuple[int, int, int, int, int]],
    L: int,
    target_reg: int,
) -> tuple[int, list[int]]:
    best_cost = None
    best_path: list[int] | None = None

    outgoing: dict[int, list[tuple[int, int, int, int, int]]] = {i: [] for i in range(n)}
    for edge in edges:
        outgoing[edge[0]].append(edge)

    def dfs(step: int, node: int, reg: int, phase: int, cost: int, path: list[int]) -> None:
        nonlocal best_cost, best_path
        if step == L:
            if node == n - 1 and reg == target_reg and phase == 0:
                if best_cost is None or cost < best_cost or (cost == best_cost and path < best_path):
                    best_cost = cost
                    best_path = path[:]
            return
        for u, v, w, t, b in outgoing.get(node, []):
            new_reg = (((reg ^ t) << 1) & 31) | phase
            new_phase = phase ^ b
            path.append(v)
            dfs(step + 1, v, new_reg, new_phase, cost + w, path)
            path.pop()

    dfs(0, 0, 0, 0, 0, [0])
    if best_cost is None:
        return (-1, [])
    return best_cost, best_path or []


def score_ood_code(response: str) -> tuple[bool, dict[str, Any]]:
    code, method = extract_humaneval_code(response, "min_cost_phase_echo_path", "")
    tests = [
        (3, [(0, 1, 1, 1, 0), (1, 2, 2, 0, 0), (0, 2, 10, 0, 0)], 2, 4),
        (4, [(0, 1, 3, 1, 1), (1, 3, 4, 2, 1), (0, 2, 1, 0, 0), (2, 3, 8, 1, 0), (1, 2, 1, 3, 1)], 2, 3),
        (3, [(0, 1, 2, 0, 1), (1, 2, 2, 0, 1)], 2, 1),
    ]
    expected = [brute_force_phase_echo_path(*case) for case in tests]

    harness = [
        code,
        "",
        "CASES = " + repr(tests),
        "EXPECTED = " + repr(expected),
        "for case, exp in zip(CASES, EXPECTED):",
        "    got = min_cost_phase_echo_path(*case)",
        "    assert got == exp, (case, exp, got)",
        "print('ok')",
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as handle:
        handle.write("\n".join(harness))
        handle.flush()
        import subprocess
        proc = subprocess.run(["python3", handle.name], capture_output=True, text=True, timeout=20)
    return proc.returncode == 0, {
        "method": method,
        "stderr": proc.stderr[-500:],
        "stdout": proc.stdout[-200:],
    }


async def eval_profile(
    ws_url: str,
    profile_name: str,
    payload: dict[str, Any],
    math_n: int,
    humaneval_n: int,
    ood_cases_enabled: set[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mode = payload["mode"]
    adapter = payload["adapter"]
    extra = {k: v for k, v in payload.items() if k not in {"mode", "adapter"}}

    for benchmark, limit in [("math500", math_n), ("humaneval", humaneval_n)]:
        for example in load_benchmark_examples(benchmark, limit):
            result = await generate_once(
                ws_url=ws_url,
                prompt=example["prompt"],
                mode=mode,
                adapter=adapter,
                max_tokens={"math500": 384, "humaneval": 512}[benchmark],
                extra_payload=extra,
            )
            parsed_prediction, method, correct = score_example(benchmark, example, result["response"])
            rows.append(
                {
                    "kind": "benchmark",
                    "profile": profile_name,
                    "benchmark": benchmark,
                    "index": example["index"],
                    "correct": correct,
                    "parsed_prediction": parsed_prediction if benchmark == "math500" else method,
                    "elapsed_s": result["elapsed_s"],
                    "total_tokens": result["total_tokens"],
                    "swaps": result["swaps"],
                    "response": result["response"],
                    "flags": artifact_flags(result["response"]),
                }
            )

    ood_cases = [
        ("ood_math", OOD_MATH_PROMPT, score_ood_math, 128),
        ("ood_science", OOD_SCIENCE_PROMPT, score_ood_science, 320),
        ("ood_code", OOD_CODE_PROMPT, score_ood_code, 768),
    ]
    for case_id, prompt, scorer, max_tokens in ood_cases:
        if case_id not in ood_cases_enabled:
            continue
        result = await generate_once(
            ws_url=ws_url,
            prompt=prompt,
            mode=mode,
            adapter=adapter,
            max_tokens=max_tokens,
            extra_payload=extra,
        )
        correct, details = scorer(result["response"])
        rows.append(
            {
                "kind": "ood",
                "profile": profile_name,
                "benchmark": case_id,
                "index": 0,
                "correct": bool(correct),
                "parsed_prediction": details,
                "elapsed_s": result["elapsed_s"],
                "total_tokens": result["total_tokens"],
                "swaps": result["swaps"],
                "response": result["response"],
                "flags": artifact_flags(result["response"]),
            }
        )

    bench_rows = [r for r in rows if r["kind"] == "benchmark"]
    ood_rows = [r for r in rows if r["kind"] == "ood"]
    summary = {
        "profile": profile_name,
        "benchmark_accuracy": round(sum(r["correct"] for r in bench_rows) / len(bench_rows), 4) if bench_rows else 0.0,
        "ood_pass_rate": round(sum(r["correct"] for r in ood_rows) / len(ood_rows), 4) if ood_rows else 0.0,
        "avg_elapsed_s": round(statistics.mean(r["elapsed_s"] for r in rows), 3),
        "avg_tokens": round(statistics.mean(r["total_tokens"] for r in rows), 2),
        "avg_swaps": round(statistics.mean(r["swaps"] for r in rows), 2),
        "artifact_rates": {
            name: round(sum(1 for r in rows if r["flags"][name]) / len(rows), 4)
            for name in ["think", "role_leak", "self_correct", "textbook"]
        },
    }
    return {"summary": summary, "rows": rows}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fast routed-profile evaluator against demo server")
    parser.add_argument("--server", default="ws://localhost:8765")
    parser.add_argument("--profiles", nargs="+", default=["routed_baseline", "routed_anchored", "routed_anchored_strict"])
    parser.add_argument("--math-n", type=int, default=8)
    parser.add_argument("--humaneval-n", type=int, default=5)
    parser.add_argument("--ood-cases", nargs="*", default=["ood_math", "ood_science", "ood_code"])
    args = parser.parse_args()

    ws_url, status_url = normalize_ws_url(args.server)
    status = load_status(status_url)
    if not status.get("ready"):
        raise RuntimeError(f"Server is not ready: {status}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_ROOT / f"profile_eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for profile_name in args.profiles:
        print(f"running {profile_name}", flush=True)
        result = await eval_profile(
            ws_url,
            profile_name,
            PROFILES[profile_name],
            args.math_n,
            args.humaneval_n,
            set(args.ood_cases),
        )
        all_results.append(result)
        (out_dir / f"{profile_name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2), flush=True)

    leaderboard = [r["summary"] for r in all_results]
    leaderboard.sort(key=lambda x: (x["ood_pass_rate"], x["benchmark_accuracy"], -x["artifact_rates"]["self_correct"]), reverse=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    md_lines = ["# Routed Profile Eval", ""]
    for row in leaderboard:
        md_lines.extend(
            [
                f"## {row['profile']}",
                "",
                f"- Benchmark accuracy: `{row['benchmark_accuracy']:.4f}`",
                f"- OOD pass rate: `{row['ood_pass_rate']:.4f}`",
                f"- Avg latency: `{row['avg_elapsed_s']:.3f}s`",
                f"- Avg tokens: `{row['avg_tokens']:.2f}`",
                f"- Avg swaps: `{row['avg_swaps']:.2f}`",
                f"- Artifact rates: `{json.dumps(row['artifact_rates'])}`",
                "",
            ]
        )
    (out_dir / "leaderboard.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    asyncio.run(main())
