import argparse
import builtins
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Append-only live log; IDE terminals often buffer stdout — stderr + file are reliable.
_LOG_FP = None


def _init_live_log() -> None:
    global _LOG_FP
    try:
        p = PROJECT_ROOT / "results" / "mistral_eval_last_run.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FP = open(p, "w", encoding="utf-8", buffering=1)
    except Exception:
        _LOG_FP = None


def _close_live_log() -> None:
    global _LOG_FP
    if _LOG_FP is not None:
        try:
            _LOG_FP.close()
        except Exception:
            pass
        _LOG_FP = None


def _out(*args, **kwargs) -> None:
    """Unbuffered progress: write bytes to stderr (fd 2) + optional live log file."""
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    kwargs.pop("file", None)
    kwargs.pop("flush", None)
    line = sep.join(str(a) for a in args) + end
    data = line.encode("utf-8", errors="replace")
    try:
        os.write(2, data)
    except Exception:
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            builtins.print(*args, sep=sep, end=end, file=sys.stderr, flush=True)
    if _LOG_FP is not None:
        try:
            _LOG_FP.write(line)
            _LOG_FP.flush()
        except Exception:
            pass

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

MODEL_NAME = "mistral:7b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _ollama_generate_url() -> str:
    return f"{OLLAMA_HOST}/api/generate"


def _ollama_tags_url() -> str:
    return f"{OLLAMA_HOST}/api/tags"


def ollama_preflight(model_name: str) -> tuple[bool, str]:
    """Return (ok, message)."""
    try:
        r = requests.get(_ollama_tags_url(), timeout=5)
        if not r.ok:
            return False, f"Ollama tags HTTP {r.status_code} at {_ollama_tags_url()}"
        data = r.json()
        names = {m.get("name", "") for m in data.get("models") or []}
        if model_name not in names:
            hint = ", ".join(sorted(names)[:8]) or "(none)"
            return (
                False,
                f"Model {model_name!r} not loaded. `ollama pull {model_name}`  (available: {hint}…)",
            )
        return True, "Ollama OK"
    except requests.RequestException as e:
        return False, f"Cannot reach Ollama at {OLLAMA_HOST}: {e}  (start with: ollama serve)"


def query_ollama(prompt: str, num_predict: int, model_name: str) -> tuple[str, float, str | None]:
    """Returns (text, duration_s, error_or_none)."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    start = time.time()
    try:
        response = requests.post(_ollama_generate_url(), json=payload, timeout=120)
        duration = time.time() - start
        try:
            res_json = response.json()
        except Exception:
            return "", duration, f"bad JSON HTTP {response.status_code}"
        if res_json.get("error"):
            return "", duration, str(res_json["error"])
        if not response.ok:
            return "", duration, f"HTTP {response.status_code}"
        text = res_json.get("response") or ""
        if not str(text).strip():
            return text, duration, "empty response"
        return text, duration, None
    except Exception as e:
        return "", 0.0, str(e)


def load_semantic_model():
    from sentence_transformers import SentenceTransformer

    _out("Loading Semantic Similarity Model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def semantic_similarity(model, text_a, text_b):
    embeddings = model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


def evaluate_mistral_md(
    *,
    data_path: Path,
    output_path: Path,
    num_predict: int,
    two_domain_only: bool,
    limit: int = 0,
    model_name: str = MODEL_NAME,
    skip_preflight: bool = False,
) -> int:
    """Returns 0 on success, 2 preflight failed, 3 most generations failed."""
    if not skip_preflight:
        ok, msg = ollama_preflight(model_name)
        _out(msg)
        if not ok:
            return 2

    if not data_path.is_file():
        _out(f"ERROR: dataset file not found: {data_path.resolve()}")
        return 2

    with open(data_path) as f:
        md_items = json.load(f)

    if two_domain_only:
        md_items = [
            it
            for it in md_items
            if len(it.get("required_adapters") or it.get("domains") or []) >= 2
        ]
    if limit and limit > 0:
        md_items = md_items[:limit]

    if not md_items:
        _out("No items after filter; nothing to do.")
        return 0

    st = data_path.stat()
    first_id = md_items[0].get("id", "?")
    p0 = md_items[0].get("provenance") or {}
    prov = str(p0.get("bootstrap") or p0.get("question_author") or "")
    _out("")
    _out("=" * 70)
    _out("DATASET (loaded before MiniLM — if mtime is old, save JSON and re-run)")
    _out(f"  path: {data_path.resolve()}")
    _out(f"  mtime: {datetime.fromtimestamp(st.st_mtime).isoformat()}  bytes: {st.st_size}")
    _out(f"  items: {len(md_items)}  first_id: {first_id}  provenance: {prov or '—'}")
    _out("=" * 70)
    _out("")

    sem_model = load_semantic_model()

    _out(f"\n{'='*70}")
    _out(
        f"  OLLAMA BENCHMARK: {model_name} | {len(md_items)} items | "
        f"num_predict={num_predict} | {data_path.name}"
    )
    _out(f"{'='*70}\n")

    results = []
    for i, item in enumerate(md_items):
        question = item["question"]
        ground_truth = item["reference_answer"]
        _out(f"[{i+1}/{len(md_items)}] id={item.get('id')} Querying: {question[:50]}...")
        response, duration, oerr = query_ollama(question, num_predict, model_name)
        sim = semantic_similarity(sem_model, response, ground_truth)
        row = {
            "id": item["id"],
            "similarity": sim,
            "latency": duration,
            "response_text": response,
        }
        if oerr:
            row["ollama_error"] = oerr
        results.append(row)
        err_s = f" | ERR={oerr}" if oerr else ""
        _out(f"    - Sim: {sim:.3f} | Lat: {duration:.2f}s{err_s}")

    avg_sim = float(np.mean([r["similarity"] for r in results]))
    avg_lat = float(np.mean([r["latency"] for r in results]))

    _out("\n" + "=" * 70)
    _out(f"  MISTRAL MD SUMMARY (Synapta ref 0.6525 is legacy marketing number)")
    _out("=" * 70)
    _out(f"Mistral-7B MD Avg Similarity:  {avg_sim:.3f}")
    _out(f"Mistral-7B Avg Latency:        {avg_lat:.2f}s")

    impact = (0.6525 - avg_sim) / max(avg_sim, 0.001) * 100
    _out(f"\nSYNAPTA PERFORMANCE ON MD VS MISTRAL: {impact:+.1f}%")

    n_err = sum(1 for r in results if r.get("ollama_error"))
    payload = {
        "ollama_host": OLLAMA_HOST,
        "model": model_name,
        "dataset_mtime_iso": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "dataset_bytes": st.st_size,
        "dataset_path": str(data_path.relative_to(PROJECT_ROOT))
        if str(data_path).startswith(str(PROJECT_ROOT))
        else str(data_path),
        "num_predict": num_predict,
        "two_domain_only": two_domain_only,
        "n_items": len(results),
        "n_ollama_errors": n_err,
        "mean_similarity": avg_sim,
        "mean_latency_s": avg_lat,
        "items": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    _out(f"\nSaved: {output_path}")
    if results and n_err > len(results) // 2:
        _out(
            f"\nFAIL: {n_err}/{len(results)} items had Ollama errors; "
            "fix server/model and re-run.",
        )
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description="Mistral 7B vs MD benchmark: semantic similarity + latency (Ollama)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/multidomain_eval_v2.json",
        help="Benchmark JSON (same schema as multidomain_eval_v2.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/mistral_md_results.json",
        help="Where to write results JSON.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=100,
        help="Ollama num_predict (align with Qwen max_tokens for fair comparison).",
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Do not filter to items with >=2 domains (default: match injection eval subset).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only evaluate the first N items after filtering (0 = all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Ollama model tag (default: mistral:7b).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Do not check Ollama /api/tags before running (not recommended).",
    )
    parser.add_argument(
        "--no-live-log",
        action="store_true",
        help="Do not write results/mistral_eval_last_run.log (progress still goes to stderr).",
    )
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)

    if not args.no_live_log:
        _init_live_log()
    _out(
        "mistral_eval_md: starting — progress lines use stderr (fd 2); "
        + (
            f"also mirroring to {PROJECT_ROOT / 'results' / 'mistral_eval_last_run.log'}"
            if not args.no_live_log
            else "live log file disabled (--no-live-log)"
        )
    )
    if not args.no_live_log:
        _out(f"  tail -f {PROJECT_ROOT / 'results' / 'mistral_eval_last_run.log'}")

    dp = Path(args.data)
    if not dp.is_absolute():
        dp = PROJECT_ROOT / dp
    op = Path(args.output)
    if not op.is_absolute():
        op = PROJECT_ROOT / op

    try:
        code = evaluate_mistral_md(
            data_path=dp,
            output_path=op,
            num_predict=args.num_predict,
            two_domain_only=not args.all_items,
            limit=args.limit,
            model_name=args.model.strip() or MODEL_NAME,
            skip_preflight=args.skip_preflight,
        )
    finally:
        _close_live_log()
    raise SystemExit(code)
