"""Phase E: Heuristic validation of the QA dataset.

Runs 10 validation checks per row, writes failure log + cleaned splits,
and prints stats + a GREEN/YELLOW/RED training-readiness recommendation.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

QA_DIR = Path("/home/learner/Desktop/mewtwo/data/rbi_corpus/qa")
TRAIN_IN = QA_DIR / "train.jsonl"
EVAL_IN = QA_DIR / "eval.jsonl"
TRAIN_OUT = QA_DIR / "train_clean.jsonl"
EVAL_OUT = QA_DIR / "eval_clean.jsonl"
FAILURE_LOG = QA_DIR / "validation_failures.jsonl"

# Generic section headings that produce low-information questions.
GENERIC_HEADINGS = {
    "introduction",
    "definitions",
    "short title",
    "commencement",
    "applicability",
    "scope",
}

# Literal template-failure markers (case-sensitive matched as-is).
TEMPLATE_MARKERS = ["{}", "{{", "}}", "[None]", "(None)", "<UNK>"]
# Tokens we look for as standalone words (case-insensitive) to avoid false hits
# inside legitimate prose.
TEMPLATE_WORD_MARKERS = ["None", "null", "nan"]

UNIT_HINTS = [
    "rs", "rs.", "inr", "₹", "%", "percent", "per cent",
    "day", "days", "month", "months", "year", "years",
    "lakh", "crore", "cr", "million", "billion", "bn", "mn",
    "bps", "basis point", "section", "para", "paragraph", "regulation",
    "clause", "article", "schedule", "rule", "act",
]
ALLOWED_SOLO_YEARS = {"1934", "1949", "1956", "1992", "1999", "2000",
                      "2002", "2013", "2018", "2023", "2024", "2025", "2026"}

# Repeating-token pattern: same word repeated 3+ times in a row.
REPEAT_RE = re.compile(r"\b(\w+)\b(?:\s+\1\b){2,}", re.IGNORECASE)
# "rs.\n35" style: a token containing a digit immediately broken by a newline.
NEWLINE_IN_NUMBER_RE = re.compile(r"[A-Za-z.]\n\d|\d\n[A-Za-z]")


def has_template_marker(text: str) -> bool:
    if not isinstance(text, str):
        return True
    for m in TEMPLATE_MARKERS:
        if m in text:
            return True
    lower_tokens = re.findall(r"[A-Za-z<>\[\]()_]+", text)
    for tok in lower_tokens:
        if tok.lower() in {w.lower() for w in TEMPLATE_WORD_MARKERS}:
            return True
    return False


def looks_like_section_ref(answer: str) -> bool:
    """Allow short numeric answers that reference a section/paragraph/clause."""
    a = answer.strip().lower()
    if re.search(r"^(section|para|paragraph|clause|article|schedule|rule|chapter)\s+", a):
        return True
    # things like "2.1" or "II" or "(a)"
    if re.fullmatch(r"\(?[ivx]+\)?\.?", a):  # roman numerals
        return True
    if re.fullmatch(r"\(?[a-z]\)?", a):  # single letter clause id
        return True
    if re.fullmatch(r"\d+(\.\d+)+", a):  # 2.1, 3.4.5
        return True
    return False


def has_unit(answer: str) -> bool:
    a = answer.lower()
    for hint in UNIT_HINTS:
        if hint in a:
            return True
    return False


def validate_row(row: dict[str, Any]) -> str | None:
    """Return None if row passes all checks, else short failure-reason code."""
    q = row.get("question", "")
    a = row.get("answer", "")
    ctx = row.get("context", "")
    s = row.get("answer_start_char")
    e = row.get("answer_end_char")
    heading = (row.get("section_heading") or "").strip().lower()

    # 1. Question contains a question mark.
    if "?" not in q:
        return "01_no_question_mark"

    # 2. Question length 20-400.
    if not (20 <= len(q) <= 400):
        return "02_question_length"

    # 3. Answer length 3-300.
    if not (3 <= len(a) <= 300):
        return "03_answer_length"

    # 4. Answer is a substring (case-insensitive) of context.
    if a.lower() not in ctx.lower():
        return "04_answer_not_in_context"

    # 5. Offset integrity.
    if not isinstance(s, int) or not isinstance(e, int):
        return "05_offset_integrity"
    if s < 0 or e > len(ctx) or s >= e:
        return "05_offset_integrity"
    if ctx[s:e] != a:
        return "05_offset_integrity"

    # 6. Template-failure markers in question or answer.
    if has_template_marker(q) or has_template_marker(a):
        return "06_template_marker"

    # 7. Repeating tokens / malformed numbers in question or answer.
    # Plain soft-wrap newlines in the answer are not a failure -- the spec
    # asks us to normalize them, not reject them. We only fail when a
    # newline splits a unit from its number (e.g. "Rs.\n35").
    if REPEAT_RE.search(q) or REPEAT_RE.search(a):
        return "07_repeating_or_malformed"
    if NEWLINE_IN_NUMBER_RE.search(a) or NEWLINE_IN_NUMBER_RE.search(q):
        return "07_repeating_or_malformed"

    # 8. Context length bounds.
    if not (200 < len(ctx) < 8000):
        return "08_context_length"

    # 9. Pure-digit short answer without unit (unless allowed year).
    a_strip = a.strip()
    if re.fullmatch(r"\d{1,2}", a_strip):
        return "09_bare_number"
    if re.fullmatch(r"\d{3,4}", a_strip):
        if a_strip not in ALLOWED_SOLO_YEARS:
            return "09_bare_number"
    elif re.fullmatch(r"[\d.,]+", a_strip):
        # purely numeric multi-token; require unit hint somewhere (in answer)
        if not has_unit(a_strip) and not looks_like_section_ref(a_strip):
            return "09_bare_number"

    # 10. Generic-heading template question.
    if heading in GENERIC_HEADINGS:
        # Pattern like "What does paragraph X cover regarding Y?"
        ql = q.lower()
        if ("cover regarding" in ql) or ("address regarding" in ql) or (
            "what does" in ql and ("paragraph" in ql or "section" in ql)
        ):
            return "10_generic_heading_template"

    return None


def hist(values: list[int], buckets: list[int]) -> list[tuple[str, int]]:
    """Compute a coarse histogram. buckets is the upper-edge of each bin."""
    counts = [0] * (len(buckets) + 1)
    for v in values:
        placed = False
        for i, edge in enumerate(buckets):
            if v <= edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    out = []
    prev = 0
    for i, edge in enumerate(buckets):
        out.append((f"{prev}-{edge}", counts[i]))
        prev = edge + 1
    out.append((f">{buckets[-1]}", counts[-1]))
    return out


def render_hist(label: str, values: list[int], buckets: list[int]) -> str:
    lines = [f"  {label} histogram:"]
    rows = hist(values, buckets)
    max_count = max((c for _, c in rows), default=1)
    for name, c in rows:
        bar = "#" * int(40 * c / max_count) if max_count else ""
        lines.append(f"    {name:>10s} | {c:6d} {bar}")
    return "\n".join(lines)


def validate_split(name: str, path: Path, fail_writer, clean_writer):
    rows_total = 0
    rows_pass = 0
    fail_counter: Counter[str] = Counter()
    by_tier_total: Counter[int] = Counter()
    by_tier_pass: Counter[int] = Counter()
    by_reg_total: Counter[str] = Counter()
    by_reg_pass: Counter[str] = Counter()
    sample_failures: dict[str, dict[str, Any]] = {}

    q_lens: list[int] = []
    a_lens: list[int] = []
    c_lens: list[int] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                fail_counter["00_bad_json"] += 1
                fail_writer.write(json.dumps({"split": name, "reason": "00_bad_json", "raw": line[:300]}) + "\n")
                continue

            rows_total += 1
            tier = row.get("tier", -1)
            reg = row.get("regulator", "?")
            by_tier_total[tier] += 1
            by_reg_total[reg] += 1

            q_lens.append(len(row.get("question", "")))
            a_lens.append(len(row.get("answer", "")))
            c_lens.append(len(row.get("context", "")))

            reason = validate_row(row)
            if reason is None:
                rows_pass += 1
                by_tier_pass[tier] += 1
                by_reg_pass[reg] += 1
                clean_writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                fail_counter[reason] += 1
                rec = {"split": name, "reason": reason, "qa_id": row.get("qa_id"),
                       "tier": tier, "regulator": reg,
                       "question": row.get("question"),
                       "answer": row.get("answer"),
                       "section_heading": row.get("section_heading")}
                fail_writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if reason not in sample_failures:
                    sample_failures[reason] = rec

    return {
        "name": name,
        "total": rows_total,
        "pass": rows_pass,
        "fail_counter": fail_counter,
        "by_tier_total": by_tier_total,
        "by_tier_pass": by_tier_pass,
        "by_reg_total": by_reg_total,
        "by_reg_pass": by_reg_pass,
        "sample_failures": sample_failures,
        "q_lens": q_lens,
        "a_lens": a_lens,
        "c_lens": c_lens,
    }


def fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100.0 * num / den:.2f}%"


def main():
    print("=" * 72)
    print("Phase E: QA validation")
    print("=" * 72)
    FAILURE_LOG.unlink(missing_ok=True)

    with FAILURE_LOG.open("w") as fail_f, \
         TRAIN_OUT.open("w") as train_clean_f, \
         EVAL_OUT.open("w") as eval_clean_f:
        train_stats = validate_split("train", TRAIN_IN, fail_f, train_clean_f)
        eval_stats = validate_split("eval", EVAL_IN, fail_f, eval_clean_f)

    combined_total = train_stats["total"] + eval_stats["total"]
    combined_pass = train_stats["pass"] + eval_stats["pass"]
    overall_rate = combined_pass / combined_total if combined_total else 0.0

    for stats in (train_stats, eval_stats):
        print()
        print(f"--- {stats['name']} split ---")
        print(f"  rows={stats['total']}  pass={stats['pass']}  "
              f"pass_rate={fmt_pct(stats['pass'], stats['total'])}")
        print("  pass rate per tier:")
        for tier in sorted(stats["by_tier_total"]):
            tot = stats["by_tier_total"][tier]
            ps = stats["by_tier_pass"][tier]
            print(f"    tier={tier}  {ps}/{tot}  {fmt_pct(ps, tot)}")
        print("  pass rate per regulator:")
        for reg in sorted(stats["by_reg_total"]):
            tot = stats["by_reg_total"][reg]
            ps = stats["by_reg_pass"][reg]
            print(f"    {reg}  {ps}/{tot}  {fmt_pct(ps, tot)}")
        print(render_hist("question length", stats["q_lens"],
                          [40, 80, 120, 160, 200, 280, 400]))
        print(render_hist("answer length", stats["a_lens"],
                          [10, 30, 60, 100, 150, 220, 300]))
        print(render_hist("context length", stats["c_lens"],
                          [500, 1000, 2000, 3000, 4000, 6000, 8000]))

    # Combined failure breakdown.
    combined_fail: Counter[str] = Counter()
    for s in (train_stats, eval_stats):
        combined_fail.update(s["fail_counter"])

    print()
    print("=" * 72)
    print("OVERALL")
    print("=" * 72)
    print(f"  total rows           : {combined_total}")
    print(f"  total pass           : {combined_pass}")
    print(f"  overall pass rate    : {fmt_pct(combined_pass, combined_total)}")
    print()
    print("  top failure reasons:")
    for reason, n in combined_fail.most_common(10):
        print(f"    {reason:40s}  {n:6d}  {fmt_pct(n, combined_total)}")

    # Sample failures (one per reason category if possible) from both splits.
    print()
    print("  sample failures (one per reason if available):")
    seen: set[str] = set()
    samples_pool = list(train_stats["sample_failures"].items()) + list(eval_stats["sample_failures"].items())
    shown = 0
    for reason, rec in samples_pool:
        if reason in seen:
            continue
        seen.add(reason)
        print(f"    [{reason}] qa_id={rec.get('qa_id')} tier={rec.get('tier')} reg={rec.get('regulator')}")
        q = (rec.get("question") or "").replace("\n", " ")[:160]
        a = (rec.get("answer") or "").replace("\n", " ")[:160]
        print(f"        Q: {q}")
        print(f"        A: {a}")
        shown += 1
        if shown >= 5:
            break

    # Decision.
    print()
    print("=" * 72)
    print("DECISION")
    print("=" * 72)
    pct = overall_rate * 100
    if pct >= 85:
        decision = "GREEN"
        msg = "Quality is good enough to train. Proceed on the cleaned splits."
    elif pct >= 70:
        decision = "YELLOW"
        msg = ("Quality is borderline. Train on cleaned data but log specific failure "
               "patterns and recommend fixes to the QA builder.")
    else:
        decision = "RED"
        msg = "Quality is too low. Abort training and fix the QA builder first."

    print(f"  overall pass rate = {pct:.2f}%  ->  {decision}")
    print(f"  recommendation    : {msg}")
    print()
    print(f"  cleaned train rows: {train_stats['pass']}  -> {TRAIN_OUT}")
    print(f"  cleaned eval rows : {eval_stats['pass']}  -> {EVAL_OUT}")
    print(f"  failure log       : {FAILURE_LOG}")
    print()
    print(f"FINAL: {decision}")


if __name__ == "__main__":
    main()
