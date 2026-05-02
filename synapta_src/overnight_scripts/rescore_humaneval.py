#!/usr/bin/env python3
"""Offline re-scoring of HumanEval results using saved raw_output.

Improvements over inline extract_code in run_humaneval_n164.py:
  Strategy 1: fenced ```python``` block (most reliable)
  Strategy 2: search for `def <entry_point>` directly in raw text and take from there
  Strategy 3: indented-block extraction (assume text after preamble is the body)

Reuses the prompt's import header for cases where the body is just the def block.

Run after the main HumanEval finishes:
  python rescore_humaneval.py
"""
import os, sys, json, re, tempfile, subprocess, datetime
from pathlib import Path

OVERNIGHT = Path("/home/learner/Desktop/mewtwo/overnight_run")
QA = OVERNIGHT / "qa_pairs"
LOG = OVERNIGHT / "gpu_jobs" / "logs" / "rescore.log"

# Reload HumanEval prompts (we need them since the saved JSONL only has task_id)
from datasets import load_dataset


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)
    with open(LOG, "a") as f:
        f.write(f"[{ts}] {msg}\n")


def _prompt_header(prompt):
    """Extract everything in the prompt before the first def/class — typically imports."""
    header_lines = []
    for line in prompt.split("\n"):
        if line.strip().startswith(("def ", "class ")):
            break
        header_lines.append(line)
    return "\n".join(header_lines).strip()


def extract_code_v2(raw_output, prompt, entry_point):
    """Robust extraction. Returns the full executable code.

    Strategy ordering:
    1. Fenced code block — if present, trust it. If it contains the entry_point def,
       prepend imports header. Otherwise, prepend the entire prompt (treat block as body).
    2. No code block — search for `^def <entry_point>` anchored at line start (avoids
       prose mentions like 'def longest(...):'). If found, prepend imports header.
    3. No code block, no def — extract indented lines as function body, prepend full prompt.
    4. Total fallback — prepend prompt + 'pass' so the test still runs and clearly fails.
    """
    header = _prompt_header(prompt)

    # Strategy 1: fenced code block — DON'T strip; leading whitespace is meaningful
    # (e.g. "    return [...]" needs its 4-space indent preserved when the body
    #  is going to be continuation of a `def f():` from the prompt).
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_output, re.DOTALL)
    if m:
        body = m.group(1).rstrip()  # only trailing
        if f"def {entry_point}" in body:
            return (header + "\n\n" + body) if header else body
        return prompt + "\n" + body

    # Strategy 2: anchor-line def search in raw text (only if no code block)
    def_match = re.search(rf"^def\s+{re.escape(entry_point)}\b", raw_output, re.MULTILINE)
    if def_match:
        body = raw_output[def_match.start():].rstrip()
        # Cut off at the next non-indented, non-def line that doesn't look like Python
        # to avoid picking up trailing prose
        lines = body.split("\n")
        kept = []
        for i, line in enumerate(lines):
            if i == 0 or line.startswith((" ", "\t")) or line.strip() == "" or line.startswith(("def ", "class ", "import ", "from ", "@")):
                kept.append(line)
            else:
                break
        body = "\n".join(kept)
        return (header + "\n\n" + body) if header else body

    # Strategy 3: indented-block extraction
    lines = raw_output.split("\n")
    body_lines = []
    in_body = False
    for line in lines:
        if line.startswith(("    ", "\t")) and line.strip():
            in_body = True
            body_lines.append(line)
        elif in_body and line.strip() == "":
            body_lines.append(line)
        elif in_body and line.strip() and not line.startswith((" ", "\t")):
            break
    if body_lines:
        return prompt + "\n" + "\n".join(body_lines)

    # Strategy 4: total fallback
    return prompt + "\n    pass  # extraction failed\n"


def run_test(code, test_code, entry_point, timeout=10):
    full = code + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full)
            fpath = f.name
        try:
            r = subprocess.run(
                ["/home/learner/Desktop/mewtwo/.venv/bin/python", fpath],
                timeout=timeout, capture_output=True, text=True,
            )
            return r.returncode == 0, r.stderr[:300] if r.returncode != 0 else ""
        finally:
            try:
                os.unlink(fpath)
            except OSError:
                pass
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"exec_err: {e}"


def main():
    log("=== Offline HumanEval re-scoring ===")
    log("Loading HumanEval...")
    ds = load_dataset("openai_humaneval", split="test")
    by_id = {ex["task_id"]: ex for ex in ds}
    log(f"Loaded {len(by_id)} reference problems.")

    summary = {}
    for mode in ("base", "format_guard"):
        f_in = QA / f"humaneval_full_{mode}.jsonl"
        f_out = QA / f"humaneval_full_{mode}_rescored.jsonl"
        if not f_in.exists():
            log(f"SKIP {mode} — file missing")
            continue
        log(f"\n--- Re-scoring {mode} ---")

        rows = []
        with open(f_in) as f:
            for l in f:
                rows.append(json.loads(l))

        passed_old = sum(1 for r in rows if r.get("passed"))
        passed_new = 0
        improved = 0
        worsened = 0

        with open(f_out, "w") as fo:
            for r in rows:
                tid = r.get("task_id")
                if tid not in by_id:
                    fo.write(json.dumps(r) + "\n")
                    continue
                problem = by_id[tid]
                raw = r.get("raw_output", "")
                if not raw:
                    # No saved raw output — keep original score
                    fo.write(json.dumps(r) + "\n")
                    if r.get("passed"):
                        passed_new += 1
                    continue
                code_v2 = extract_code_v2(raw, problem["prompt"], problem["entry_point"])
                ok, err = run_test(code_v2, problem["test"], problem["entry_point"])
                if ok:
                    passed_new += 1
                if ok and not r.get("passed"):
                    improved += 1
                elif not ok and r.get("passed"):
                    worsened += 1
                new_row = dict(r)
                new_row["passed_v2"] = ok
                new_row["error_v2"] = err[:200] if err else ""
                new_row["code_tested_v2"] = code_v2[:2000]
                fo.write(json.dumps(new_row) + "\n")

        log(f"{mode}: original {passed_old}/{len(rows)} = {passed_old/max(len(rows),1):.1%}")
        log(f"{mode}: rescored {passed_new}/{len(rows)} = {passed_new/max(len(rows),1):.1%}")
        log(f"{mode}: improved={improved} worsened={worsened} delta={passed_new - passed_old:+d}")
        summary[mode] = {
            "n": len(rows),
            "pass_at_1_v1": passed_old / max(len(rows), 1),
            "pass_at_1_v2": passed_new / max(len(rows), 1),
            "improved": improved,
            "worsened": worsened,
        }

    out_path = QA / "humaneval_rescored_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary -> {out_path}")
    log(json.dumps(summary, indent=2))

    # Write findings if both base + format_guard rescored
    if "base" in summary and "format_guard" in summary:
        base_acc = summary["base"]["pass_at_1_v2"]
        fg_acc = summary["format_guard"]["pass_at_1_v2"]
        delta = fg_acc - base_acc
        findings_path = OVERNIGHT / "findings" / "humaneval_n164.md"
        with open(findings_path, "w") as f:
            f.write("# HumanEval n=164 — corrected scoring\n\n")
            f.write(f"**Sample size:** n={summary['base']['n']} per mode\n\n")
            f.write("## Results (re-scored with robust extract_code)\n\n")
            f.write(f"| Mode | Pass@1 (v1, raw) | Pass@1 (v2, fixed) | Improved | Worsened |\n")
            f.write(f"|---|---|---|---|---|\n")
            f.write(f"| base | {summary['base']['pass_at_1_v1']:.1%} | **{base_acc:.1%}** | {summary['base']['improved']} | {summary['base']['worsened']} |\n")
            f.write(f"| format_guard | {summary['format_guard']['pass_at_1_v1']:.1%} | **{fg_acc:.1%}** | {summary['format_guard']['improved']} | {summary['format_guard']['worsened']} |\n\n")
            f.write(f"**Delta (FG − base): {delta:+.1%}**\n\n")
            f.write("## Comparison vs deck claim\n\n")
            f.write("Deck claims (from grand_comparison_v2_results.json, n=25):\n")
            f.write("- base: 24%\n- format_guard: 48%\n- delta: +24 points\n\n")
            f.write(f"At n=164:\n")
            f.write(f"- base: {base_acc:.1%}\n- format_guard: {fg_acc:.1%}\n- delta: {delta:+.1%}\n\n")
            if delta >= 0.10:
                f.write("**The deck claim survives at scale.** Format Guard maintains a meaningful advantage over base.\n")
            elif delta >= 0:
                f.write("**Partial replication.** Format Guard is slightly better than base but the gap shrinks at n=164. Re-frame deck claim as 'modest improvement at full scale'.\n")
            else:
                f.write("**Deck claim does not hold at n=164.** Format Guard underperforms base. Need to revise the deck.\n")
        log(f"Findings -> {findings_path}")


if __name__ == "__main__":
    main()
