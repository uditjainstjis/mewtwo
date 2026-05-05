"""Deterministic builder for Synapta Indian BFSI Benchmark v1.

NOTE FOR PUBLIC USERS:
    This file is shipped as DOCUMENTATION OF CONSTRUCTION, not as a
    runnable script for the published bundle. It depends on Synapta's
    internal RBI/SEBI corpus path
    (data/rbi_corpus/qa/eval_clean.jsonl) which is not redistributed.
    Read it to verify how questions.jsonl was produced (deterministic
    templates over headings / numeric spans, no LLM authorship,
    stratified greedy selection with per-PDF caps).

Original developer-facing usage:
    python build_benchmark.py
    Reads:  /home/learner/Desktop/mewtwo/data/rbi_corpus/qa/eval_clean.jsonl
    Writes: /home/learner/Desktop/mewtwo/data/benchmark/synapta_indian_bfsi_v1/questions.jsonl

The selection is fully deterministic given the input file: we sort by qa_id,
apply a strict quality filter, then greedy-pick by stratified quotas with
a per-PDF cap and topic diversity. No LLM is involved.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/learner/Desktop/mewtwo")
EVAL = ROOT / "data/rbi_corpus/qa/eval_clean.jsonl"
SPLIT = ROOT / "data/rbi_corpus/qa/split_manifest_v2.json"
OUT_DIR = ROOT / "data/benchmark/synapta_indian_bfsi_v1"
OUT = OUT_DIR / "questions.jsonl"

# Per-PDF topic mapping (hand assigned from inspection of source documents).
# Topics restricted to closed schema:
#   kyc, aml, fraud, capital, derivatives, mutual_fund, insurance,
#   payments, banking_ops, foreign_exchange, governance, reporting
PDF_TOPIC = {
    # RBI
    "05MD56DF4232873F466191DC23F39F1012DD.PDF": "insurance",            # MD - Insurance / Foreign Exchange
    "06MDE170516F633150EBCFE438084174F7DECCDC20C.PDF": "foreign_exchange", # MD - Establishment of Branch Office
    "08MDROA112016989330DF1FF9494784DAC32B488E6AE4.PDF": "foreign_exchange", # MD - Remittance of Assets
    "108MDINTERNALOMBUDSMANCC05402F77BE4F229B59877F341386A4.PDF": "governance", # Internal Ombudsman
    "10MD06102016E550559916C346E0BC93720658286729.PDF": "foreign_exchange", # MD - Direct Investment by Residents (FEMA ODI)
    "115MDCN01042024F088EFA4F3F04DC9968AD7EC6844AFE9.PDF": "banking_ops", # Counterfeit Notes 2024
    "12MDFB8AD1B34BCB4D0A8F6869DA4A53082E.PDF": "foreign_exchange",     # MD - Imports / Advance Remittance
    "136MD25042025D32CA11EFBFC425288223F9E7FA96AF4.PDF": "banking_ops", # Incentive for soiled notes
    "137MDEF04E0A142F948C58A9503C136E565AF.PDF": "governance",          # ETP operators
    "13MDRD77DCF42C4E64B6C9A83C24EF5D4E188.PDF": "reporting",           # Reporting under FEMA
    "14MDM1120153F2FD5CC455640E78D139304BC7C080F.PDF": "foreign_exchange", # MD - Miscellaneous (current acct remittances)
    "91MDPPDR01042022EADFF060516C47E49C34DB2C9AA96CE1.PDF": "reporting", # Penal Provisions in deficiencies in reporting
    "MD126B9CF2E0CABD14471955E50A54D8291F2.PDF": "foreign_exchange",    # Sovereign Green Bonds at IFSC
    "MD28A4C421E7F7724C07B38E3C6207F3548E.PDF": "fraud",                # Frauds Classification & Reporting
    "MD3191FD1C01B7704FB9B24E7073F651AB51.PDF": "derivatives",          # Risk Management & Inter-Bank Dealings
    "MD39472AB49ECE87F40868FBB8F0714E1627F.PDF": "banking_ops",         # Counterfeit Notes (older)
    "MD63948FC3860C834CF78DB37990DA108B23.PDF": "banking_ops",          # Currency Distribution & Exchange Scheme
    "MDCND0104202580F6E20BEB804BF6A10E8551EFDCA39F.PDF": "banking_ops", # Counterfeit Notes 2025
    # SEBI
    "dec-2024_1733233982158.pdf": "governance",     # MC for Depositories (governance + ops)
    "feb-2026_1770375507051.pdf": "mutual_fund",    # MC for Research Analysts
    "jun-2025_1751022988074.pdf": "mutual_fund",    # IA fee schedule (Investment Advisers)
    "jun-2025_1751025804218.pdf": "mutual_fund",    # RA / IA fee limits
    "sep-2024_1727094551693.pdf": "governance",     # SEBI Insider Trading Regulations
}

DIFFICULTY_HINT = {
    # heuristic: very short numeric -> easy; multi-condition timeline -> hard; rest medium
}

def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open()]


def sha10(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def is_high_value_question(r: dict) -> bool:
    """Strong quality gate. Reject low-signal templated retrieval questions."""
    q = r["question"]
    a = r["answer"].strip()
    ctx = r["context"]

    # Hard quality gates from the spec
    if len(ctx) < 200 or len(ctx) > 4000:
        return False
    if "passage above" in q.lower() or "above passage" in q.lower():
        return False
    if len(a) < 3:
        return False
    # bare number with no unit
    if re.fullmatch(r"[\d,\.]+%?", a) and not a.endswith("%"):
        return False
    # truncated mid-word (long answer ending without sentence terminator and last token long)
    last_token = a.split()[-1] if a.split() else ""
    if len(a) > 60 and a[-1].isalpha() and len(last_token) > 4 and not a.endswith(("s", "y", "d", "n", "e", "g", "r", "t", "l")):
        # might be truncated; reject
        return False

    # tier 3: heading-based answers are often dumped chunks. Keep only if answer
    # is < 250 chars and starts capitalised / numbered (looks like a real heading
    # body, not a runaway sentence).
    if r["tier"] == 3:
        if len(a) < 50 or len(a) > 250:
            # too short = chunk artifact ("machines.", "SEBI?", "A.D.", "SEBI.")
            # too long = runaway chunk
            return False
        # reject if answer starts with a lowercase preposition / conjunction (clear chunk artifact)
        first = a.split()[0].lower() if a.split() else ""
        if first in {"and", "or", "of", "the", "in", "to", "for", "by", "with", "from",
                     "after", "before", "while", "as", "that", "which", "such", "an",
                     "a", "is", "are", "was", "were", "be", "been", "being",
                     "their", "answer", "during", "permitted", "machines", "intermediaries",
                     "counter", "framed", "identified", "restructuring", "residual"}:
            return False
        # answers that read like raw OCR (lots of stray punctuation)
        if a.count('"') + a.count("”") + a.count("“") >= 2:
            return False
        # answers that are bare references (end with stray section number + period)
        if re.search(r"\b\d+\.\d+(\.\d+)*\.?$", a) and len(a.split()) < 12:
            return False
        # answers that are clearly mid-sentence chunks (start with a digit/section marker
        # but only because chunking sliced the sentence)
        if re.match(r"^\d+\.\d+\.", a) and "shall" not in a and "may" not in a and "is" not in a:
            return False
        # answers ending with a hanging "Circular No." / "circular no." / "FAQ No." / "Annex"
        if re.search(r"(?i)\b(Circular No|circular no|FAQ No|Annexure|Annex)\.?\s*$", a):
            return False
        # answers ending with stray "etc." / "i." / "ii." / "iii." / "iv." / "1." / "2." / "a." / "b."
        if re.search(r"(?i)\b(etc|iv|iii|ii|vi|vii|viii)\.\s*$", a):
            return False
        if re.search(r"^\s*[ivxlcdm]+\.|\b[ivx]+\.\s*$", a):
            return False
        # URL answers
        if "http://" in a or "https://" in a or "www." in a:
            return False
        # Answers that are themselves questions
        if a.endswith("?"):
            return False
        # Answers that start with example/illustration markers
        if re.match(r"(?i)^(Example|Illustration|Note|Annex)", a):
            return False
        # Answers that contain odd table-of-contents-style fragments
        if re.search(r"\bTimeline for implementation\b", a):
            return False
        # answer ending with a section marker followed by period (mid-sentence cut)
        if re.search(r"\b\d+\.?\s*$", a) and not re.search(r"%|years|months|days|crore|lakh|million|rupees|Rs", a, re.I):
            # but allow if the rest of the answer looks like a complete sentence
            if a.count(".") >= 2 or len(a) < 80:
                return False

    # tier 2 specific: answer should contain a digit, currency symbol, percent,
    # named regulation/section, or a clear named entity. Rules out vacuous answers.
    if r["tier"] == 2:
        looks_concrete = bool(
            re.search(r"\d", a) or
            re.search(r"(?i)\b(regulation|section|para(graph)?|chapter|annex)\b", a) or
            re.search(r"[%₹]", a) or
            "Rs" in a or "INR" in a or "lakh" in a or "crore" in a or "million" in a or "billion" in a
        )
        if not looks_concrete:
            return False
        # answer shouldn't be longer than 80 chars for tier 2 (extractive numeric/short-span)
        if len(a) > 80:
            return False
        # reject obvious OCR garbage like "rs37" with no separator
        if re.match(r"(?i)^rs\d", a) and " " not in a:
            return False
        # tier-2 answer must contain at least one digit OR be a regulation reference
        if not re.search(r"\d", a) and not re.match(r"(?i)^(Regulation|Section|Para|Paragraph|Chapter|Clause)\s+\S", a):
            return False

    # Reject questions phrased as "what does X cover regarding Y" where Y is a
    # full-sentence run (not a topic) - these are weak retrieval prompts that
    # only make sense after seeing context.
    if re.search(r"\b(cover|state|provided|concerning)\b.*\b(regarding|about|concerning)\b", q):
        # length of trailing topic phrase
        m = re.search(r"\b(regarding|about|concerning)\s+(.+?)\??$", q)
        if m and len(m.group(2)) > 80:
            return False

    # Topic-phrase sanity. Many templates substitute a sentence fragment in
    # place of a topic noun phrase, producing gibberish like
    #   "Under which section is Banks should ensure that ... so dealt with?"
    # Identify the substituted phrase {Y} for known patterns and reject if it
    # is itself a sentence fragment (starts with subject+modal/verb, ends with
    # connector, etc).
    bad_starts = (
        "Banks should", "Entities from", "Importers are", "Listed company",
        "Investment advice", "Account opening", "Investment adviser shall",
        "What are the", "What is the", "Pursuant to", "Whether the",
        "Who are all", "On the basis", "In case of", "Designating ",
    )
    bad_ends = (
        " so", " and", " or", " for", " with", " from", " of", " across",
        " which", " that", " the", " under", " by", " to", " on", " in",
        " an", " a", " is", " are", " was", " were", " be", " been",
        " including", " concerning", " regarding", " about",
    )
    # Pattern 1: "Under which X is {Y} dealt with?"
    m = re.search(r"Under which \w+ is (.+?) dealt with\??$", q)
    if m:
        y = m.group(1).strip()
        if y.startswith(bad_starts) or y.endswith(bad_ends):
            return False
    # Pattern 2: "What X is specified for {Y}..."
    # Pattern 3: "What X is referenced in connection with {Y}?"
    m = re.search(r"in connection with (.+?)\??$", q)
    if m:
        y = m.group(1).strip()
        if y.startswith(bad_starts) or y.endswith(bad_ends):
            return False
    # Pattern 4: trailing topic after "regarding/about/concerning {Y}"
    m = re.search(r"\b(regarding|about|concerning|for|specified for)\s+(.+?)\??$", q)
    if m:
        y = m.group(2).strip().rstrip(",.;?")
        # Strip trailing "under paragraph/section/clause N(.M)*" qualifier
        y = re.sub(r"\s+under\s+(paragraph|section|clause|para|chapter)\s+\S+\s*$", "", y, flags=re.I).strip()
        if y.startswith(bad_starts) or y.endswith(bad_ends):
            return False
        # "Each stock exchange shall consolidate..." or "Listed company shall provide..."
        # — sentence-shaped substitutions, not headings.
        if re.match(r"^(Each|Every|Any|All|The|A|An)\s+\w+\s+(shall|may|must|should)", y):
            return False
        # SEBI MCs use a Q&A layout where headings legitimately start with "Whether".
        # Don't reject those.
        # Truncated section-heading topic ending with comma+date+connector
        if re.search(r",\s*\d{4}\s*[–-]\s*[A-Z][\w ,]*\band\s*$", y):
            return False
        # Strip trailing connector
        if re.search(r"\b(and|or|of|for|to)\s*$", y):
            return False
        # Unbalanced parentheses (e.g. "SEBI (Prohibition")
        if y.count("(") != y.count(")"):
            return False

    # Catch truncated section titles like "Counterfeit Notes, 2025 – Detection,
    # Reporting and" embedded anywhere in the question
    if re.search(r"\b(Reporting|Detection|Monitoring|Distribution)\s+and\s+(cover|concerning|state|provided|dealt|under|specified)", q):
        return False
    if re.search(r"&\s+(cover|concerning|state|provided|dealt|under|specified)", q):
        return False

    # Sanity: no orphan opening parens in question
    if q.count("(") != q.count(")"):
        return False

    # "Master Direction on X of cover" - "of" connector before template verb is gibberish
    if re.search(r"\b(of|for|to|with|by|on|in)\s+(cover|state|provided|concerning|dealt|specified)", q):
        return False

    return True


def alt_answers_for(answer: str, tier: int = 2) -> list[str]:
    """Generate deterministic acceptable variants for substring/F1 scoring.

    Goal: at least 2 variants per answer where reasonable, to absorb
    surface-form noise (currency symbols, lakh/crore, "Rs." vs "₹",
    bare numbers vs "Section N", "within N days" vs "no later than N days").
    """
    a = answer.strip().rstrip(".")
    alts = []

    # ---- Currency variants ----
    if "₹" in a:
        alts.append(a.replace("₹", "Rs. "))
        alts.append(a.replace("₹", "Rs."))
        alts.append(a.replace("₹", "INR "))
        alts.append(a.replace("₹", ""))   # bare number with unit
    if a.startswith("Rs."):
        alts.append(a.replace("Rs.", "₹").strip())
    # 0.1 million <-> 1 lakh
    if re.match(r"^₹?\s*0\.1\s*million$", a, re.I):
        alts.extend(["1 lakh", "Rs. 1 lakh", "₹1 lakh", "Rupees 1 lakh"])
    if re.match(r"^₹?\s*1\s*crore$", a, re.I):
        alts.extend(["10 million", "Rs. 1 crore", "₹1 crore"])

    # ---- Percent variants ----
    if re.search(r"\d+\s*per\s*cent", a, re.I):
        alts.append(re.sub(r"\s*per\s*cent", "%", a, flags=re.I))
        alts.append(re.sub(r"per\s*cent", "percent", a, flags=re.I))
    if a.endswith("%"):
        alts.append(re.sub(r"%$", " per cent", a))
        alts.append(re.sub(r"%$", " percent", a))

    # ---- Regulation / Section / Para number normalization ----
    m = re.match(r"^(Regulation|Section|Para|Paragraph|Chapter|Clause)\s+(\S+)$", a, re.I)
    if m:
        kw, num = m.group(1), m.group(2)
        # bare number
        alts.append(num)
        # short forms
        if kw.lower() == "section":
            alts.extend([f"S. {num}", f"Sec. {num}", f"section {num}"])
        elif kw.lower() == "regulation":
            alts.extend([f"Reg. {num}", f"regulation {num}"])
        elif kw.lower() in ("para", "paragraph"):
            alts.extend([f"Paragraph {num}", f"para {num}", f"paragraph {num}"])
        elif kw.lower() == "chapter":
            alts.extend([f"Ch. {num}", f"chapter {num}"])
        elif kw.lower() == "clause":
            alts.extend([f"clause {num}"])

    # ---- "not later than N days" / "within N days" ----
    m = re.match(r"(?i)^(not later than|no later than|within)\s+(\d+)\s+(day|days|month|months|year|years)$", a)
    if m:
        n, unit = m.group(2), m.group(3).lower()
        alts.extend([
            f"within {n} {unit}",
            f"not later than {n} {unit}",
            f"no later than {n} {unit}",
            f"{n} {unit}",
        ])

    # ---- "for a period of N years" ----
    m = re.match(r"(?i)^for a period of\s+(\d+)\s+(day|days|month|months|year|years)$", a)
    if m:
        n, unit = m.group(1), m.group(2).lower()
        alts.extend([f"{n} {unit}", f"period of {n} {unit}"])

    # ---- Tier 3 long answers: use first clause / first noun phrase as alts ----
    if tier == 3 and len(a) > 40:
        # first clause up to first comma / semicolon
        first_clause = re.split(r"[;,]", a, maxsplit=1)[0].strip()
        if 8 <= len(first_clause) < len(a) - 5:
            alts.append(first_clause)
        # first 8 words
        words = a.split()
        if len(words) > 10:
            alts.append(" ".join(words[:8]))
            alts.append(" ".join(words[:5]))

    # ---- de-dup, drop canonical ----
    seen, out = {a.lower().strip()}, []
    for x in alts:
        x = x.strip()
        if x and x.lower() not in seen:
            out.append(x)
            seen.add(x.lower())
    return out[:4]


def difficulty_for(r: dict) -> str:
    a = r["answer"].strip()
    ctx_len = len(r["context"])
    if r["tier"] == 2 and re.fullmatch(r"[\d\.,]+\s*(per\s*cent|%)", a, re.I):
        return "easy"
    if r["tier"] == 2 and ("₹" in a or "Rs" in a or "lakh" in a or "crore" in a or "million" in a):
        return "easy"
    if r["tier"] == 2 and re.match(r"^(Regulation|Section|Para|Paragraph)\s+\S+$", a, re.I):
        return "medium"
    if r["tier"] == 3 and ctx_len > 1500:
        return "hard"
    if r["tier"] == 3:
        return "medium"
    return "medium"


def scoring_method_for(r: dict) -> str:
    a = r["answer"].strip()
    if r["tier"] == 2:
        # short numeric / regulation reference → substring tolerant
        return "substring"
    # tier 3 heading body → token F1
    if len(a.split()) >= 6:
        return "token_f1_threshold_0.5"
    return "substring"


def main():
    # Sanity: enforce eval/train PDF disjointness (the manifest already promises this).
    split = json.loads(SPLIT.read_text())
    train_pdfs = set(split["train_pdfs"])
    eval_pdfs = set(split["eval_pdfs"])
    overlap = train_pdfs & eval_pdfs
    assert not overlap, f"DOC OVERLAP: {overlap}"

    rows = load_jsonl(EVAL)
    # Sort deterministically by qa_id
    rows.sort(key=lambda r: r["qa_id"])

    # Filter
    candidates = [r for r in rows if is_high_value_question(r)]
    # Dedup by question text (some templated questions repeat verbatim with
    # different gold answers from different chunks). Keep first occurrence
    # by qa_id sort order.
    seen_q = set()
    deduped = []
    for r in candidates:
        key = r["question"].strip().lower()
        if key in seen_q:
            continue
        seen_q.add(key)
        deduped.append(r)
    candidates = deduped
    # Verify all candidates' source PDFs are in eval_pdfs (not train)
    for r in candidates:
        assert r["source_pdf"] in eval_pdfs, f"LEAK: {r['source_pdf']}"

    # Stratified greedy pick: target 30 RBI + 30 SEBI, 30 T2 + 30 T3.
    # PDF caps are per (PDF, tier) pair to prevent any single PDF from
    # dominating one tier of the benchmark while still allowing the same PDF
    # to contribute to both tiers (e.g. dec-2024 SEBI Depositories MC has
    # plenty of well-formed tier-2 and tier-3 candidates).
    SEBI_PDFS = {"dec-2024_1733233982158.pdf",
                 "feb-2026_1770375507051.pdf",
                 "jun-2025_1751022988074.pdf",
                 "jun-2025_1751025804218.pdf",
                 "sep-2024_1727094551693.pdf"}
    def per_pdf_tier_cap(pdf, tier):
        # SEBI side: 5 PDFs total but tier-3 candidates are concentrated in
        # the depositories MC. Lift the dec-2024 SEBI tier-3 cap to 12 so we
        # can hit 15 SEBI tier-3 questions while still drawing from at least
        # 2 SEBI PDFs.
        if pdf == "dec-2024_1733233982158.pdf" and tier == 3:
            return 12
        return 8 if pdf in SEBI_PDFS else 5
    target = {("RBI", 2): 15, ("RBI", 3): 15, ("SEBI", 2): 15, ("SEBI", 3): 15}
    picked_by_pdf_tier = defaultdict(int)
    picked_by_pdf = defaultdict(int)
    picked_by_cell = defaultdict(int)
    picked_topics = defaultdict(int)
    selected = []

    # Group candidates by cell
    cells = defaultdict(list)
    for r in candidates:
        cells[(r["regulator"], r["tier"])].append(r)

    # Round-robin pass: cycle through cells, pick best candidate from least-touched PDF
    # within each cell, prefer under-represented topics.
    def topic_for(r):
        return PDF_TOPIC.get(r["source_pdf"], "banking_ops")

    while sum(picked_by_cell[k] for k in target) < sum(target.values()):
        progressed = False
        for cell, quota in target.items():
            if picked_by_cell[cell] >= quota:
                continue
            pool = cells[cell]
            # rank: PDF×tier count asc, topic rarity asc, qa_id asc (deterministic)
            pool_ranked = sorted(
                [r for r in pool if r["qa_id"] not in {s["qa_id"] for s in selected}
                 and picked_by_pdf_tier[(r["source_pdf"], r["tier"])] < per_pdf_tier_cap(r["source_pdf"], r["tier"])],
                key=lambda r: (picked_by_pdf_tier[(r["source_pdf"], r["tier"])],
                               picked_by_pdf[r["source_pdf"]],
                               picked_topics[topic_for(r)],
                               r["qa_id"])
            )
            if not pool_ranked:
                continue
            choice = pool_ranked[0]
            selected.append(choice)
            picked_by_pdf_tier[(choice["source_pdf"], choice["tier"])] += 1
            picked_by_pdf[choice["source_pdf"]] += 1
            picked_by_cell[cell] += 1
            picked_topics[topic_for(choice)] += 1
            progressed = True
        if not progressed:
            break

    assert len(selected) == 60, f"selected {len(selected)} not 60"

    # Build records
    out_records = []
    for i, r in enumerate(sorted(selected, key=lambda x: (x["regulator"], x["tier"], x["qa_id"]))):
        bench_id = f"sib1-{i+1:03d}-{sha10(r['qa_id'])}"
        rec = {
            "benchmark_id": bench_id,
            "regulator": r["regulator"],
            "tier": r["tier"],
            "source_pdf": r["source_pdf"],
            "source_section": r.get("section_heading", ""),
            "context": r["context"],
            "question": r["question"],
            "gold_answer": r["answer"],
            "alternative_answers": alt_answers_for(r["answer"], tier=r["tier"]),
            "scoring_method": scoring_method_for(r),
            "difficulty": difficulty_for(r),
            "topic_tag": PDF_TOPIC.get(r["source_pdf"], "banking_ops"),
            "_provenance": {"qa_id": r["qa_id"], "split": "eval_clean.jsonl"},
        }
        out_records.append(rec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Diagnostics
    from collections import Counter
    print(f"Wrote {len(out_records)} records to {OUT}")
    print("Regulator:", Counter(r["regulator"] for r in out_records))
    print("Tier:     ", Counter(r["tier"] for r in out_records))
    print("Topic:    ", Counter(r["topic_tag"] for r in out_records))
    print("Difficulty:", Counter(r["difficulty"] for r in out_records))
    print("Scoring:  ", Counter(r["scoring_method"] for r in out_records))
    print("PDF cnt:  ", len({r["source_pdf"] for r in out_records}))
    print("\nPer PDF:")
    by_pdf = Counter(r["source_pdf"] for r in out_records)
    for p, c in sorted(by_pdf.items(), key=lambda x: -x[1]):
        print(f"  {p[:60]:<62} {c}")


if __name__ == "__main__":
    main()
