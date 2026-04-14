#!/usr/bin/env python3
"""
One-shot pipeline: bootstrap Track B JSON, Mistral (A+B), Qwen injection (A+B), aggregates, summary.

Track B defaults to a clone of multidomain_eval_v2 with provenance flags until you replace it with
proxy/expert-authored items (metrics will match Track A until then).

Usage:
  PYTHONUNBUFFERED=1 python3 src/eval/run_full_showcase_pipeline.py --real
  PYTHONUNBUFFERED=1 python3 src/eval/run_full_showcase_pipeline.py --real --limit 3   # smoke

Optional proxy bootstrap (replaces clone when any item succeeds):
  CLUSTER_USE_PERPLEXITY_PROXY=1 python3 src/eval/run_full_showcase_pipeline.py --real --proxy-items 5

SATS trajectory eval (adds ~3x MD rows of GPU time):
  PYTHONUNBUFFERED=1 python3 src/eval/run_full_showcase_pipeline.py --real --with-sats

Strict cluster vs standard (~2x MD rows, slow):
  PYTHONUNBUFFERED=1 python3 src/eval/run_full_showcase_pipeline.py --real --with-cluster

Ollama URL (default http://127.0.0.1:11434): set OLLAMA_HOST
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TRACK_A_DATA = "data/multidomain_eval_v2.json"
TRACK_B_DATA = "data/multidomain_eval_external.json"
MISTRAL_A_OUT = "results/mistral_track_a.json"
MISTRAL_B_OUT = "results/mistral_track_b.json"
INJ_A_OUT = "results/injection_track_a.jsonl"
INJ_B_OUT = "results/injection_track_b.jsonl"
SATS_OUT = "results/sats_eval_showcase.jsonl"
CLUSTER_OUT = "results/cluster_strict_showcase.jsonl"
SUMMARY_JSON = "results/showcase_pipeline_summary.json"
SUMMARY_TXT = "results/showcase_pipeline_summary.txt"


def run_cmd(
    desc: str,
    argv: list[str],
    *,
    cwd: Path | None = None,
    allow_fail: bool = False,
) -> dict:
    cwd = cwd or PROJECT_ROOT
    print(f"\n{'='*70}\n>>> {desc}\n{'='*70}\n", flush=True)
    r = subprocess.run(argv, cwd=str(cwd), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    ok = r.returncode == 0
    if not ok and not allow_fail:
        print(f"FAILED ({r.returncode}): {' '.join(argv)}", file=sys.stderr, flush=True)
        sys.exit(r.returncode)
    if not ok:
        print(f"WARN: step failed ({r.returncode}), continuing: {desc}", flush=True)
    return {"argv": argv, "returncode": r.returncode, "ok": ok}


def bootstrap_track_b_from_v2(v2_path: Path, out_path: Path) -> int:
    items = json.loads(v2_path.read_text())
    out = []
    for it in items:
        doms = it.get("required_adapters") or it.get("domains") or []
        nid = it.get("id", "")
        if not str(nid).startswith("ext_"):
            nid = f"ext_{nid}"
        row = {
            "id": nid,
            "domains": list(it.get("domains") or doms),
            "required_adapters": list(it.get("required_adapters") or doms),
            "question": it["question"],
            "reference_answer": it["reference_answer"],
            "provenance": {
                "bootstrap": "cloned_from_multidomain_eval_v2",
                "replace_before_investor_showcase": True,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        out.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return len(out)


def try_proxy_bootstrap(n: int, v2_path: Path, out_path: Path) -> int:
    """Return number of items written (0 if proxy unavailable or failed)."""
    if n <= 0:
        return 0
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    try:
        from proxy_bridge import ProxyBridge
    except Exception:
        return 0
    bridge = ProxyBridge()
    if not bridge.enabled:
        print("CLUSTER_USE_PERPLEXITY_PROXY=1 not set; skipping proxy bootstrap.", flush=True)
        return 0
    v2_items = json.loads(v2_path.read_text())
    built = []
    for i in range(min(n, len(v2_items))):
        it = v2_items[i]
        doms = it.get("required_adapters") or it.get("domains") or []
        if len(doms) < 2:
            continue
        d1, d2 = doms[0], doms[1]
        eid = f"ext_proxy_{i+1:02d}"
        spec = (
            f"Design one benchmark: domains {d1} and {d2}. "
            "Format exactly:\nQUESTION: <one paragraph>\nREFERENCE: <one paragraph>\n"
        )
        text = bridge.ask(spec, mode="reasoning") or ""
        q, ref = "", ""
        for line in text.splitlines():
            s = line.strip()
            if s.upper().startswith("QUESTION:"):
                q = s.split(":", 1)[1].strip()
            elif s.upper().startswith("REFERENCE:"):
                ref = s.split(":", 1)[1].strip()
        if not q or not ref:
            print(f"Proxy parse fail item {eid}; stopping proxy bootstrap.", flush=True)
            break
        built.append(
            {
                "id": eid,
                "domains": [d1, d2],
                "required_adapters": [d1, d2],
                "question": q,
                "reference_answer": ref,
                "provenance": {
                    "question_author": "perplexity-proxy",
                    "reference_author": "perplexity-proxy",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
            }
        )
        print(f"  proxy item {len(built)}/{n} ok: {eid}", flush=True)
    if not built:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(built, indent=2))
    return len(built)


def summarize_mistral_json(path: Path, label: str, lines: list[str]) -> None:
    try:
        m = json.loads(path.read_text())
    except Exception:
        return
    lines.append(f"\n--- {label} (Mistral JSON) ---\n")
    lines.append(
        f"  model={m.get('model', '?')}  n={m.get('n_items')}  "
        f"n_errors={m.get('n_ollama_errors', '?')}  "
        f"mean_sim={m.get('mean_similarity', 0):.4f}  mean_lat={m.get('mean_latency_s', 0):.3f}s\n"
    )


def run_aggregate_eval_jsonl(jsonl: Path, label: str, lines: list[str]) -> None:
    if not jsonl.is_file():
        return
    r = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src/eval/aggregate_eval_jsonl.py"),
            str(jsonl),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    lines.append(f"\n--- {label} ({jsonl.name}) ---\n")
    lines.append(r.stdout or "")
    if r.stderr:
        lines.append(r.stderr)


def run_aggregate(jsonl: Path, label: str, lines: list[str]) -> None:
    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src/eval/aggregate_injection_jsonl.py"), str(jsonl)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    lines.append(f"\n--- {label} ({jsonl.name}) ---\n")
    lines.append(r.stdout or "")
    if r.stderr:
        lines.append(r.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="Run GPU + Ollama steps (required for full run).")
    parser.add_argument("--limit", type=int, default=0, help="Forward to injection eval (0 = all).")
    parser.add_argument(
        "--proxy-items",
        type=int,
        default=0,
        help="If >0 and Perplexity proxy enabled, build Track B from N proxy drafts (else clone v2).",
    )
    parser.add_argument("--skip-mistral", action="store_true")
    parser.add_argument("--skip-injection", action="store_true")
    parser.add_argument(
        "--with-sats",
        action="store_true",
        help="After injection, run run_eval_sats.py --real (same --limit if set).",
    )
    parser.add_argument(
        "--with-cluster",
        action="store_true",
        help="After injection, run run_eval_cluster_strict.py --real (same --data/limit).",
    )
    parser.add_argument(
        "--mistral-skip-preflight",
        action="store_true",
        help="Forward --skip-preflight to mistral_eval_md.py (debug only).",
    )
    parser.add_argument(
        "--mistral-model",
        type=str,
        default="mistral:7b",
        help="Ollama model tag for Mistral steps.",
    )
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    v2_path = PROJECT_ROOT / TRACK_A_DATA
    b_path = PROJECT_ROOT / TRACK_B_DATA
    started = datetime.now(timezone.utc).isoformat()
    log: list[dict] = []
    txt_lines: list[str] = [
        f"Showcase pipeline summary  utc={started}",
        f"PROJECT_ROOT={PROJECT_ROOT}",
        "",
    ]

    # --- Track B file ---
    if args.proxy_items > 0:
        n = try_proxy_bootstrap(args.proxy_items, v2_path, b_path)
        txt_lines.append(f"Track B: proxy bootstrap wrote {n} items -> {TRACK_B_DATA}\n")
        if n == 0:
            n2 = bootstrap_track_b_from_v2(v2_path, b_path)
            txt_lines.append(f"Track B: fell back to v2 clone, n={n2}\n")
            log.append({"step": "bootstrap_track_b", "mode": "clone_v2_fallback", "n": n2})
        else:
            log.append({"step": "bootstrap_track_b", "mode": "proxy", "n": n})
    else:
        n = bootstrap_track_b_from_v2(v2_path, b_path)
        txt_lines.append(f"Track B: cloned v2 with provenance, n={n} -> {TRACK_B_DATA}\n")
        log.append({"step": "bootstrap_track_b", "mode": "clone_v2", "n": n})

    if not args.real:
        print("Dry run: Track B JSON written. Use --real for Mistral + Qwen injection.")
        (PROJECT_ROOT / SUMMARY_TXT).write_text("\n".join(txt_lines))
        return

    py = sys.executable

    # --- Mistral ---
    if not args.skip_mistral:
        margs_base = [
            py,
            str(PROJECT_ROOT / "backend/mistral_eval_md.py"),
            "--model",
            args.mistral_model,
            "--num-predict",
            "180",
        ]
        if args.mistral_skip_preflight:
            margs_base.append("--skip-preflight")
        if args.limit and args.limit > 0:
            margs_base += ["--limit", str(args.limit)]
        log.append(
            run_cmd(
                "Mistral Track A (v2, num_predict=180)",
                margs_base
                + [
                    "--data",
                    TRACK_A_DATA,
                    "--output",
                    MISTRAL_A_OUT,
                ],
                allow_fail=True,
            )
        )
        log.append(
            run_cmd(
                "Mistral Track B (external json, num_predict=180)",
                margs_base
                + [
                    "--data",
                    TRACK_B_DATA,
                    "--output",
                    MISTRAL_B_OUT,
                ],
                allow_fail=True,
            )
        )
    else:
        txt_lines.append("Mistral steps skipped.\n")

    # --- Injection ---
    if not args.skip_injection:
        inj_args = [py, str(PROJECT_ROOT / "src/eval/run_eval_injection_hypotheses.py"), "--real", "--extra", "--more"]
        if args.limit and args.limit > 0:
            inj_args += ["--limit", str(args.limit)]
        log.append(
            run_cmd(
                "Qwen injection Track A",
                inj_args + ["--data", TRACK_A_DATA, "--output", INJ_A_OUT],
            )
        )
        log.append(
            run_cmd(
                "Qwen injection Track B",
                inj_args + ["--data", TRACK_B_DATA, "--output", INJ_B_OUT],
            )
        )
    else:
        txt_lines.append("Injection steps skipped.\n")

    # --- SATS (optional, heavy) ---
    if args.with_sats and args.real:
        sargs = [
            py,
            str(PROJECT_ROOT / "src/eval/run_eval_sats.py"),
            "--real",
            "--data",
            TRACK_A_DATA,
            "--output",
            SATS_OUT,
        ]
        if args.limit and args.limit > 0:
            sargs += ["--limit", str(args.limit)]
        log.append(run_cmd("SATS vs baselines", sargs, allow_fail=True))

    if args.with_cluster and args.real:
        cargs = [
            py,
            str(PROJECT_ROOT / "src/eval/run_eval_cluster_strict.py"),
            "--real",
            "--data",
            TRACK_A_DATA,
            "--output",
            CLUSTER_OUT,
        ]
        if args.limit and args.limit > 0:
            cargs += ["--limit", str(args.limit)]
        log.append(run_cmd("Cluster strict vs standard", cargs, allow_fail=True))

    # --- Aggregates ---
    for path, label in [
        (PROJECT_ROOT / INJ_A_OUT, "injection Track A"),
        (PROJECT_ROOT / INJ_B_OUT, "injection Track B"),
    ]:
        if path.is_file():
            run_aggregate(path, label, txt_lines)

    run_aggregate_eval_jsonl(PROJECT_ROOT / SATS_OUT, "SATS aggregate", txt_lines)
    run_aggregate_eval_jsonl(PROJECT_ROOT / CLUSTER_OUT, "Cluster strict aggregate", txt_lines)

    for label, mj in [
        ("Mistral Track A", PROJECT_ROOT / MISTRAL_A_OUT),
        ("Mistral Track B", PROJECT_ROOT / MISTRAL_B_OUT),
    ]:
        if mj.is_file():
            summarize_mistral_json(mj, label, txt_lines)
        if not mj.is_file():
            continue
        try:
            m = json.loads(mj.read_text())
            lat = float(m.get("mean_latency_s") or 0)
            n_err = int(m.get("n_ollama_errors") or 0)
            if lat < 0.05 or n_err > (m.get("n_items") or 0) // 2:
                txt_lines.append(
                    f"\nWARNING: {label} ({mj.name}): mean_latency_s={lat}, n_ollama_errors={n_err} — "
                    "check OLLAMA_HOST, `ollama serve`, and `ollama pull` for the model.\n"
                )
        except Exception:
            pass

    finished = datetime.now(timezone.utc).isoformat()
    artifacts = {
        "mistral_a": MISTRAL_A_OUT,
        "mistral_b": MISTRAL_B_OUT,
        "injection_a": INJ_A_OUT,
        "injection_b": INJ_B_OUT,
    }
    if args.with_sats:
        artifacts["sats"] = SATS_OUT
    if args.with_cluster:
        artifacts["cluster_strict"] = CLUSTER_OUT
    payload = {
        "started_utc": started,
        "finished_utc": finished,
        "track_b_data": TRACK_B_DATA,
        "steps": log,
        "artifacts": artifacts,
    }
    (PROJECT_ROOT / SUMMARY_JSON).write_text(json.dumps(payload, indent=2))
    txt_lines.append(f"\nDone utc={finished}\n")
    (PROJECT_ROOT / SUMMARY_TXT).write_text("\n".join(txt_lines))
    print(f"\nWrote {SUMMARY_JSON} and {SUMMARY_TXT}", flush=True)


if __name__ == "__main__":
    main()
