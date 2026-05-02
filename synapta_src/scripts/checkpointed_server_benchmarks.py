#!/usr/bin/env python3
"""
Controlled benchmark runner against the live Synapta demo server.

Goals:
- deterministic generation for benchmark reruns
- save every prompt/response pair immediately
- checkpoint human-readable summaries every N examples
- resume-safe when pointed at an existing run directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import websockets
from datasets import load_dataset


PROJECT = Path("/home/learner/Desktop/mewtwo")
RESULTS_ROOT = PROJECT / "results" / "controlled_benchmarks"
LOG_ROOT = PROJECT / "logs" / "controlled_benchmarks"

DEFAULT_SIZES = {
    "math500": 500,
    "humaneval": 164,
    "arc_challenge": 1172,
}

CONFIGS = {
    "base": {"mode": "naked", "adapter": "code"},
    "math": {"mode": "single", "adapter": "math"},
    "code": {"mode": "single", "adapter": "code"},
    "science": {"mode": "single", "adapter": "science"},
    "routed": {"mode": "routed", "adapter": "code"},
}


@dataclass
class ExampleRecord:
    key: str
    benchmark: str
    config: str
    example_index: int
    question: str
    prompt: str
    response: str
    parsed_prediction: str
    prediction_method: str
    gold: str
    correct: bool
    elapsed_s: float
    total_tokens: int
    swaps: int
    final_domain: str
    route_events: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def load_status(status_url: str) -> dict[str, Any]:
    with urlopen(status_url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def normalize_ws_url(server: str) -> tuple[str, str]:
    server = server.rstrip("/")
    if server.startswith("ws://") or server.startswith("wss://"):
        base = server
    elif server.startswith("http://") or server.startswith("https://"):
        base = "ws" + server[4:]
    else:
        base = f"ws://{server}"
    if not base.endswith("/ws/generate"):
        ws_url = base + "/ws/generate"
        status_url = base.replace("ws://", "http://").replace("wss://", "https://") + "/api/status"
    else:
        ws_url = base
        status_url = base.replace("ws://", "http://").replace("wss://", "https://").removesuffix("/ws/generate") + "/api/status"
    return ws_url, status_url


def extract_choice(text: str) -> tuple[str, str]:
    patterns = [
        r"(?:answer|correct answer)\s*(?:is|:)?\s*([ABCD])\b",
        r"^\s*([ABCD])\s*$",
    ]
    for pattern in patterns:
        hits = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if hits:
            return hits[-1].upper(), "pattern"
    letters = re.findall(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    return (letters[-1].upper(), "fallback_letter") if letters else ("", "missing")


def strip_outer_braces(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        text = text[1:-1].strip()
    return text


def normalize_math(text: str) -> str:
    text = text.strip()
    text = text.replace("$", "")
    text = re.sub(r"\\boxed\s*\{", "{", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mbox\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("\\tfrac", "\\frac")
    text = text.replace("\\dfrac", "\\frac")
    text = text.replace("\\pi", "pi")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\,", "")
    text = text.replace("^{\\circ}", "")
    text = text.replace("^\\circ", "")
    text = text.replace("°", "")
    text = strip_outer_braces(text)
    text = re.sub(r"\(([A-E])\)", r"\1", text)
    text = re.sub(r"\s+", "", text.lower())
    return text


def extract_boxed(text: str) -> list[str]:
    matches = []
    start = 0
    needle = "\\boxed{"
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        depth = 1
        j = idx + len(needle)
        buf = []
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(ch)
            j += 1
        if buf:
            matches.append("".join(buf).strip())
        start = j + 1
    return matches


def extract_math_answer(text: str) -> tuple[str, str]:
    boxed = extract_boxed(text)
    if boxed:
        return boxed[-1], "boxed"

    hash_hits = re.findall(r"####\s*(.+?)(?:\n|$)", text)
    if hash_hits:
        return hash_hits[-1].strip(), "hashes"

    answer_hits = re.findall(r"(?:the answer is|answer:|therefore|thus|hence)\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
    if answer_hits:
        return answer_hits[-1].strip(" ."), "answer_phrase"

    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty:
        return nonempty[-1].strip(" ."), "last_nonempty_line"

    return "", "missing"


def math_equal(pred: str, gold: str) -> bool:
    return normalize_math(pred) == normalize_math(gold)


def extract_humaneval_code(response: str, entry_point: str, prompt_code: str) -> tuple[str, str]:
    block = re.search(r"```(?:python)?\s*\n(.*?)```", response, flags=re.DOTALL)
    if block:
        code = block.group(1).strip()
        if f"def {entry_point}" in code:
            code = code[code.index(f"def {entry_point}"):]
            return code, "fenced_function"
        return prompt_code + code, "fenced_completion"

    if f"def {entry_point}" in response:
        code = response[response.index(f"def {entry_point}"):]
        return code.strip(), "inline_function"

    return prompt_code + response, "raw_completion"


def run_humaneval_check(code: str, test_code: str, entry_point: str) -> bool:
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as handle:
            handle.write(full)
            handle.flush()
            proc = subprocess.run(
                ["python3", handle.name],
                capture_output=True,
                text=True,
                timeout=15,
            )
        return proc.returncode == 0
    except Exception:
        return False


def build_math500_examples(limit: int) -> list[dict[str, Any]]:
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        ds = load_dataset("lighteval/MATH", split="test")
    q_key = "problem" if "problem" in ds.column_names else "question"
    a_key = "solution" if "solution" in ds.column_names else "answer"
    rows = []
    for i, ex in enumerate(ds.select(range(min(limit, len(ds))))):
        gold, _ = extract_math_answer(ex[a_key])
        question = ex[q_key]
        prompt = (
            "Solve the following math problem carefully. "
            "Put only the final answer in \\boxed{} at the end.\n\n"
            f"{question}"
        )
        rows.append({"index": i, "question": question, "prompt": prompt, "gold": gold})
    return rows


def build_humaneval_examples(limit: int) -> list[dict[str, Any]]:
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for i, ex in enumerate(ds.select(range(min(limit, len(ds))))):
        prompt_code = ex["prompt"]
        prompt = (
            "Complete the following Python code. "
            "Return only valid Python code with no markdown fences and no explanation.\n\n"
            f"{prompt_code}"
        )
        rows.append(
            {
                "index": i,
                "question": prompt_code,
                "prompt": prompt,
                "gold": ex["entry_point"],
                "test": ex["test"],
                "entry_point": ex["entry_point"],
                "prompt_code": prompt_code,
            }
        )
    return rows


def build_arc_examples(limit: int) -> list[dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, ex in enumerate(ds.select(range(min(limit, len(ds))))):
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])
        )
        question = f"{ex['question']}\n\n{choices}"
        prompt = question + "\n\nAnswer with only one letter: A, B, C, or D."
        rows.append({"index": i, "question": question, "prompt": prompt, "gold": ex["answerKey"].strip().upper()})
    return rows


def load_benchmark_examples(name: str, limit: int) -> list[dict[str, Any]]:
    if name == "math500":
        return build_math500_examples(limit)
    if name == "humaneval":
        return build_humaneval_examples(limit)
    if name == "arc_challenge":
        return build_arc_examples(limit)
    raise ValueError(f"Unsupported benchmark: {name}")


def score_example(benchmark: str, example: dict[str, Any], response: str) -> tuple[str, str, bool]:
    if benchmark == "math500":
        pred, method = extract_math_answer(response)
        return pred, method, bool(pred and math_equal(pred, example["gold"]))
    if benchmark == "humaneval":
        code, method = extract_humaneval_code(response, example["entry_point"], example["prompt_code"])
        correct = run_humaneval_check(code, example["test"], example["entry_point"])
        return code, method, correct
    if benchmark == "arc_challenge":
        pred, method = extract_choice(response)
        return pred, method, pred == example["gold"]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


async def generate_once(
    ws_url: str,
    prompt: str,
    mode: str,
    adapter: str,
    max_tokens: int,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt,
        "mode": mode,
        "adapter": adapter,
        "max_tokens": max_tokens,
        "thinking": False,
        "do_sample": False,
        "chunk_size": 16,
        "repetition_penalty": 1.0,
    }
    if extra_payload:
        payload.update(extra_payload)

    route_events: list[dict[str, Any]] = []
    tokens: list[str] = []
    done: dict[str, Any] | None = None

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20, max_size=2**22) as ws:
        await ws.send(json.dumps(payload))
        while True:
            message = json.loads(await ws.recv())
            msg_type = message.get("type")
            if msg_type == "token":
                tokens.append(message.get("text", ""))
            elif msg_type in {"route", "swap"}:
                route_events.append(message)
            elif msg_type == "error":
                raise RuntimeError(message.get("message", "unknown generation error"))
            elif msg_type == "done":
                done = message
                break

    if done is None:
        raise RuntimeError("generation finished without done message")

    response = done.get("full_text", "".join(tokens))
    return {
        "response": response,
        "elapsed_s": float(done.get("elapsed_s", 0.0)),
        "total_tokens": int(done.get("total_tokens", len(tokens))),
        "swaps": int(done.get("swaps", 0)),
        "final_domain": done.get("final_domain", ""),
        "route_events": route_events,
    }


def append_jsonl(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_summary(records: list[ExampleRecord], path: Path) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = f"{record.config}:{record.benchmark}"
        bucket = grouped.setdefault(
            key,
            {
                "config": record.config,
                "benchmark": record.benchmark,
                "correct": 0,
                "total": 0,
                "avg_elapsed_s": 0.0,
                "avg_tokens": 0.0,
                "avg_swaps": 0.0,
            },
        )
        bucket["correct"] += int(record.correct)
        bucket["total"] += 1
        bucket["avg_elapsed_s"] += record.elapsed_s
        bucket["avg_tokens"] += record.total_tokens
        bucket["avg_swaps"] += record.swaps

    for bucket in grouped.values():
        total = max(bucket["total"], 1)
        bucket["accuracy"] = round(bucket["correct"] / total, 4)
        bucket["avg_elapsed_s"] = round(bucket["avg_elapsed_s"] / total, 3)
        bucket["avg_tokens"] = round(bucket["avg_tokens"] / total, 2)
        bucket["avg_swaps"] = round(bucket["avg_swaps"] / total, 2)

    summary = {
        "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "examples_logged": len(records),
        "results": sorted(grouped.values(), key=lambda x: (x["benchmark"], x["config"])),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_markdown(records: list[ExampleRecord], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Controlled Benchmark Rerun",
        "",
        f"Updated: {summary['updated_at']}",
        "",
        "## Summary",
        "",
        "| Benchmark | Config | Correct | Total | Accuracy | Avg s | Avg toks | Avg swaps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['benchmark']} | {row['config']} | {row['correct']} | {row['total']} | "
            f"{row['accuracy']:.4f} | {row['avg_elapsed_s']:.3f} | {row['avg_tokens']:.2f} | {row['avg_swaps']:.2f} |"
        )

    lines.extend(["", "## Examples", ""])
    for record in records:
        lines.extend(
            [
                f"### {record.benchmark} / {record.config} / #{record.example_index}",
                "",
                f"- Correct: `{record.correct}`",
                f"- Parsed prediction method: `{record.prediction_method}`",
                f"- Parsed prediction: `{record.parsed_prediction}`",
                f"- Gold: `{record.gold}`",
                f"- Elapsed: `{record.elapsed_s}`s",
                f"- Tokens: `{record.total_tokens}`",
                f"- Swaps: `{record.swaps}`",
                "",
                "#### Question",
                "",
                "```text",
                record.question,
                "```",
                "",
                "#### Response",
                "",
                "```text",
                record.response,
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def max_tokens_for_benchmark(name: str) -> int:
    return {"math500": 384, "humaneval": 512, "arc_challenge": 32}[name]


async def run(args: argparse.Namespace) -> None:
    ws_url, status_url = normalize_ws_url(args.server)
    status = load_status(status_url)
    if not status.get("ready"):
        raise RuntimeError(f"Server is not ready: {status}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_dir = RESULTS_ROOT / f"server_rerun_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = run_dir / "examples.jsonl"
    summary_path = run_dir / "summary.json"
    markdown_path = run_dir / "examples.md"
    progress_path = run_dir / "progress.json"

    records: list[ExampleRecord] = []
    seen_keys: set[str] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Rescore lightweight benchmarks on resume so parser/normalizer fixes
            # can repair prior false negatives without re-running generations.
            if row.get("benchmark") == "math500":
                row["correct"] = math_equal(row.get("parsed_prediction", ""), row.get("gold", ""))
            elif row.get("benchmark") == "arc_challenge":
                row["correct"] = row.get("parsed_prediction", "").strip().upper() == row.get("gold", "").strip().upper()
            records.append(ExampleRecord(**row))
            seen_keys.add(row["key"])

    if records:
        summary = write_summary(records, summary_path)
        write_markdown(records, summary, markdown_path)

    processed_since_checkpoint = 0
    extra_payload = {}
    for key in [
        "chunk_size",
        "routing_interval",
        "min_tokens_before_swap",
        "domain_hint_strength",
        "swap_margin",
        "lock_prompt_domain",
        "prompt_anchor",
    ]:
        value = getattr(args, key)
        if value is not None:
            extra_payload[key] = value

    for benchmark in args.benchmarks:
        limit = args.max_samples if args.max_samples > 0 else DEFAULT_SIZES[benchmark]
        examples = load_benchmark_examples(benchmark, limit)

        for config in args.configs:
            mode = CONFIGS[config]["mode"]
            adapter = CONFIGS[config]["adapter"]

            for example in examples:
                key = f"{config}|{benchmark}|{example['index']}"
                if key in seen_keys:
                    continue

                started = time.time()
                result = await generate_once(
                    ws_url=ws_url,
                    prompt=example["prompt"],
                    mode=mode,
                    adapter=adapter,
                    max_tokens=max_tokens_for_benchmark(benchmark),
                    extra_payload=extra_payload,
                )
                parsed_prediction, method, correct = score_example(benchmark, example, result["response"])
                elapsed = round(time.time() - started, 3)

                record = ExampleRecord(
                    key=key,
                    benchmark=benchmark,
                    config=config,
                    example_index=example["index"],
                    question=example["question"],
                    prompt=example["prompt"],
                    response=result["response"],
                    parsed_prediction=parsed_prediction,
                    prediction_method=method,
                    gold=example["gold"],
                    correct=correct,
                    elapsed_s=elapsed,
                    total_tokens=result["total_tokens"],
                    swaps=result["swaps"],
                    final_domain=result["final_domain"],
                    route_events=result["route_events"],
                )

                records.append(record)
                seen_keys.add(key)
                append_jsonl(jsonl_path, record.to_json())

                progress = {
                    "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "last_record": record.__dict__,
                    "examples_logged": len(records),
                    "run_dir": str(run_dir),
                }
                progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

                processed_since_checkpoint += 1
                if processed_since_checkpoint >= args.checkpoint_every:
                    summary = write_summary(records, summary_path)
                    write_markdown(records, summary, markdown_path)
                    processed_since_checkpoint = 0

    summary = write_summary(records, summary_path)
    write_markdown(records, summary, markdown_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed benchmark rerun via live demo server")
    parser.add_argument("--server", default="ws://localhost:8765", help="Server base URL or websocket URL")
    parser.add_argument("--benchmarks", nargs="+", default=["math500", "humaneval", "arc_challenge"])
    parser.add_argument("--configs", nargs="+", default=["base", "math", "code", "science", "routed"])
    parser.add_argument("--max-samples", type=int, default=0, help="Per benchmark cap. 0 = default full size.")
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--run-dir", help="Resume or write into an existing run directory")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--routing-interval", type=int)
    parser.add_argument("--min-tokens-before-swap", type=int)
    parser.add_argument("--domain-hint-strength", type=float)
    parser.add_argument("--swap-margin", type=float)
    parser.add_argument("--lock-prompt-domain", action="store_true", default=None)
    parser.add_argument("--prompt-anchor", action="store_true", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
