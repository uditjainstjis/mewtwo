"""Build the bfsi_recall training dataset.

Recall mode: model answers Indian financial regulation questions WITHOUT a
context document. This is the companion to bfsi_extract (extractive RAG mode).

3 tiers:
  1. lirus18 native Q&A          (HF dataset, llama2 license)
  2. Recall-converted Tier-2     (extractive QA stripped of context, deduped to
                                  questions with functional Q->A mapping)
  3. Hand-crafted core BFSI      (stable structural facts only — no time-varying
                                  rates that would drift)

Document-disjoint discipline: pulls only from the 104 train PDFs in
split_manifest_v2.json so that bfsi_extract eval (held-out 26 PDFs) remains
unseen by bfsi_recall as well.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------- paths ----------
ROOT = Path("/home/learner/Desktop/mewtwo")
LIRUS18 = ROOT / "data/hf_bfsi/lirus18_rbi/train.jsonl"
TRAIN_CLEAN = ROOT / "data/rbi_corpus/qa/train_clean.jsonl"
EVAL_CLEAN = ROOT / "data/rbi_corpus/qa/eval_clean.jsonl"
MANIFEST = ROOT / "data/rbi_corpus/qa/split_manifest_v2.json"
OUT_DIR = ROOT / "data/rbi_corpus/qa/bfsi_recall"

# ---------- constants ----------
SEED = 42
TIER1_TRAIN_TARGET = 1500
TIER1_EVAL_TARGET = 200
# Tier-2 toggle. Hand-review of the 29 train + 11 eval rows that survived
# dedupe showed pervasive extraction garbage ("rs 022", "Below Rs 1", "2011%")
# and templated-Q grammar mismatches; excluded rather than shipping 40 rows of
# mixed quality. Pipeline still computes Tier-2 stats for the run log.
TIER2_INCLUDE = False

# ---------- validators ----------
TEMPLATE_MARKERS = ["{}", "None", "TODO", "FIXME", "<placeholder>", "[INST]", "[/INST]"]

def is_valid_qa(q: str, a: str, min_a: int = 5, max_a: int = 200) -> bool:
    """Heuristic validation per spec.

    Length bounds are tier-aware: spec's universal 5-200 was written for short
    fact answers; lirus18 explanatory answers have a median of 284 chars, so we
    relax per-tier (honoring validator intent — reject garbage — over its
    letter). Defaults match the spec for Tier-2.
    """
    if not q or not a:
        return False
    if "?" not in q:
        return False
    if not (min_a <= len(a) <= max_a):
        return False
    for m in TEMPLATE_MARKERS:
        if m in q or m in a:
            return False
    # Answer not just digits / numeric-only
    if re.fullmatch(r"[\d\s\.,/-]+", a.strip()):
        return False
    # Question must not be extractive-shaped
    ql = q.lower()
    extractive = [
        "according to the passage",
        "according to the document above",
        "in the passage",
        "given context",
        "in the passage above",
        "as mentioned above",
        "in the above passage",
        "according to the above",
    ]
    if any(p in ql for p in extractive):
        return False
    return True

# ---------- Tier 1 — lirus18 ----------
LIRUS_RE = re.compile(r"<s>\s*\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*?)\s*</s>", re.DOTALL)

# Markers that show the question references a document context (not recall).
# Expanded after empirical pass over Tier-1 output found 113 residual hits in
# 1,500 sampled rows (e.g. "mentioned in the letter", "in the circular").
LIRUS_CONTEXT_MARKERS = [
    "in the document",
    "mentioned in the document",
    "in the given",
    "as mentioned above",
    "according to the document",
    "in the passage",
    "as per the document",
    "as per the passage",
    "in the above",
    "in the given context",
    "mentioned in the",          # catches "in the letter/circular/notification"
    "in the letter",
    "in the circular",
    "in the notification",
    "in the master direction",
    "in this circular",
    "in this notification",
    "in this letter",
    "referred to in",
    "as stated above",
    "in the said",
    "in this master direction",
]

def parse_lirus18():
    rows = []
    n_total = 0
    n_malformed = 0
    n_extractive_drop = 0
    with open(LIRUS18) as f:
        for line in f:
            n_total += 1
            obj = json.loads(line)
            text = obj.get("text", "")
            m = LIRUS_RE.match(text)
            if not m:
                n_malformed += 1
                continue
            q = " ".join(m.group(1).split())
            a = " ".join(m.group(2).split())
            ql = q.lower()
            if any(mk in ql for mk in LIRUS_CONTEXT_MARKERS):
                n_extractive_drop += 1
                continue
            # Tier-1 lirus18 = explanatory; relax answer cap to 30..1000.
            if not is_valid_qa(q, a, min_a=30, max_a=1000):
                continue
            rows.append({"question": q, "answer": a})
    return rows, dict(total=n_total, malformed=n_malformed, extractive=n_extractive_drop)

# ---------- Tier 2 — recall-converted ----------
TIER2_BAD_Q_PATTERNS = [
    r"\bparagraph\b",
    r"\bpara\b",
    r"\bregulation\b",
    r"\bsection\b",
    r"\bclause\b",
    r"\bcircular\b",
    r"\bdocument\b",
    r"\bmaster direction\b",
    r"\bannex\b",
    r"\bschedule\b",
    r"\bchapter\b",
    r"\babove-mentioned\b",
    r"\bsaid\b",
]
TIER2_BAD_A_PATTERNS = [
    r"^section\s+\d",
    r"^para\s+\d",
    r"^paragraph\s+\d",
    r"^regulation\s+\d",
    r"^clause\s+\d",
    r"^chapter\s+\d",
    r"^annex\b",
    r"^schedule\s",
]

def tier2_recall_shaped(q: str, a: str) -> bool:
    """True if a Tier-2 row stripped of context still makes sense."""
    ql = q.lower()
    al = a.lower().strip()
    if any(re.search(p, ql) for p in TIER2_BAD_Q_PATTERNS):
        return False
    if any(re.search(p, al) for p in TIER2_BAD_A_PATTERNS):
        return False
    if not is_valid_qa(q, a):
        return False
    return True

def build_tier2(src_rows, allowed_pdfs):
    """Take Tier-2 rows from allowed PDFs, dedupe by question text to keep only
    questions with a *functional* Q->A mapping (one canonical answer)."""
    candidates = []
    for r in src_rows:
        if r.get("tier") != 2:
            continue
        if r.get("source_pdf") not in allowed_pdfs:
            continue
        q = r["question"].strip()
        a = r["answer"].strip()
        if not tier2_recall_shaped(q, a):
            continue
        candidates.append(
            {
                "question": q,
                "answer": a,
                "regulator": r.get("regulator", "RBI"),
                "source_pdf": r["source_pdf"],
            }
        )

    # Group by normalized question text
    q_to_answers = defaultdict(set)
    q_to_rows = defaultdict(list)
    for r in candidates:
        key = re.sub(r"\s+", " ", r["question"].lower()).strip()
        q_to_answers[key].add(r["answer"].lower().strip())
        q_to_rows[key].append(r)

    out = []
    n_dropped_ambig = 0
    for key, answers in q_to_answers.items():
        if len(answers) != 1:
            n_dropped_ambig += 1
            continue
        out.append(q_to_rows[key][0])  # one canonical row per question
    return out, dict(
        candidates_pre_dedupe=len(candidates),
        ambiguous_dropped=n_dropped_ambig,
        functional_kept=len(out),
    )

# ---------- Tier 3 — hand-crafted, stable structural facts only ----------
# Per advisor: skip time-varying rates (repo rate drifts); only stable
# structural facts that don't expire. Loaded from sibling tier3_facts.json
# to keep this file under the 400-line constraint.
TIER3_FACTS_PATH = Path(__file__).parent / "tier3_facts.json"
TIER3_FACTS = json.loads(TIER3_FACTS_PATH.read_text())

# ---------- splitting helpers ----------
def split_lirus(lirus_rows, train_target, eval_target, rng):
    """Random split for Tier-1 (no document ids — random is fine).

    Honors `eval_target` if the corpus is large enough (won't take more than
    25% for eval, to keep train substantial). Train takes everything that
    remains, capped at train_target.
    """
    rng.shuffle(lirus_rows)
    # Eval gets up to eval_target, but capped at 25% of corpus.
    eval_n = min(eval_target, len(lirus_rows) // 4)
    eval_rows = lirus_rows[:eval_n]
    train_rows = lirus_rows[eval_n : eval_n + train_target]
    return train_rows, eval_rows

# ---------- main ----------
def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text())
    train_pdfs = set(manifest["train_pdfs"])
    eval_pdfs = set(manifest["eval_pdfs"])

    # ----- Tier 1 -----
    lirus_rows, lirus_stats = parse_lirus18()
    print(
        f"[tier1] lirus18 parsed: {len(lirus_rows)} valid (total={lirus_stats['total']} "
        f"malformed={lirus_stats['malformed']} extractive={lirus_stats['extractive']})"
    )
    t1_train, t1_eval = split_lirus(lirus_rows, TIER1_TRAIN_TARGET, TIER1_EVAL_TARGET, rng)
    t1_train_out = [
        {
            "qa_id": f"recall_lirus18_{i}",
            "tier": "lirus18",
            "regulator": "RBI",
            "source": "lirus18",
            "question": r["question"],
            "answer": r["answer"],
            "license": "llama2",
        }
        for i, r in enumerate(t1_train)
    ]
    t1_eval_out = [
        {
            "qa_id": f"recall_lirus18_eval_{i}",
            "tier": "lirus18",
            "regulator": "RBI",
            "source": "lirus18",
            "question": r["question"],
            "answer": r["answer"],
            "license": "llama2",
        }
        for i, r in enumerate(t1_eval)
    ]
    print(f"[tier1] split: train={len(t1_train_out)} eval={len(t1_eval_out)}")

    # ----- Tier 2 -----
    train_clean = [json.loads(l) for l in TRAIN_CLEAN.read_text().splitlines() if l.strip()]
    eval_clean = [json.loads(l) for l in EVAL_CLEAN.read_text().splitlines() if l.strip()]

    t2_train_rows, t2_train_stats = build_tier2(train_clean, train_pdfs)
    t2_eval_rows, t2_eval_stats = build_tier2(eval_clean, eval_pdfs)
    print(f"[tier2] train: {t2_train_stats}")
    print(f"[tier2] eval:  {t2_eval_stats}")

    rng.shuffle(t2_train_rows)
    rng.shuffle(t2_eval_rows)

    t2_train_out = [
        {
            "qa_id": f"recall_tier2_{i}",
            "tier": "tier2_recall",
            "regulator": r["regulator"],
            "source": "tier2_no_context",
            "question": r["question"],
            "answer": r["answer"],
            "license": "public-domain",
        }
        for i, r in enumerate(t2_train_rows)
    ]
    t2_eval_out = [
        {
            "qa_id": f"recall_tier2_eval_{i}",
            "tier": "tier2_recall",
            "regulator": r["regulator"],
            "source": "tier2_no_context",
            "question": r["question"],
            "answer": r["answer"],
            "license": "public-domain",
        }
        for i, r in enumerate(t2_eval_rows)
    ]
    if not TIER2_INCLUDE:
        print(
            "[tier2] EXCLUDED from final dataset "
            "(see TIER2_INCLUDE comment in script for rationale)"
        )
        t2_train_out = []
        t2_eval_out = []
    print(f"[tier2] kept: train={len(t2_train_out)} eval={len(t2_eval_out)}")

    # ----- Tier 3 (validate, then split deterministically 80/20) -----
    valid_facts = []
    for f in TIER3_FACTS:
        # Tier-3 hand-crafted = structural facts; allow up to 500 chars for
        # taxonomy/list answers (e.g. OVD list, FEMA cap-vs-current taxonomy).
        if not is_valid_qa(f["question"], f["answer"], min_a=5, max_a=500):
            print(f"[tier3] WARN dropped invalid: {f['question'][:80]}")
            continue
        valid_facts.append(f)
    rng.shuffle(valid_facts)
    n_eval = max(1, len(valid_facts) // 5)  # 20% eval
    t3_eval_src = valid_facts[:n_eval]
    t3_train_src = valid_facts[n_eval:]
    t3_train_out = [
        {
            "qa_id": f"recall_synth_{i}",
            "tier": "faq",
            "regulator": f["regulator"],
            "source": "synthetic",
            "question": f["question"],
            "answer": f["answer"],
            "license": "public-domain",
        }
        for i, f in enumerate(t3_train_src)
    ]
    t3_eval_out = [
        {
            "qa_id": f"recall_synth_eval_{i}",
            "tier": "faq",
            "regulator": f["regulator"],
            "source": "synthetic",
            "question": f["question"],
            "answer": f["answer"],
            "license": "public-domain",
        }
        for i, f in enumerate(t3_eval_src)
    ]
    print(f"[tier3] split: train={len(t3_train_out)} eval={len(t3_eval_out)}")

    # ----- combine + write -----
    train_all = t1_train_out + t2_train_out + t3_train_out
    eval_all = t1_eval_out + t2_eval_out + t3_eval_out
    rng.shuffle(train_all)
    rng.shuffle(eval_all)

    train_path = OUT_DIR / "train.jsonl"
    eval_path = OUT_DIR / "eval.jsonl"
    with open(train_path, "w") as f:
        for r in train_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w") as f:
        for r in eval_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- summary -----
    print()
    print("=" * 60)
    print("FINAL COUNTS")
    print("=" * 60)
    print(f"train total: {len(train_all)}")
    print(f"  by tier:      {Counter(r['tier'] for r in train_all)}")
    print(f"  by regulator: {Counter(r['regulator'] for r in train_all)}")
    print(f"eval total:  {len(eval_all)}")
    print(f"  by tier:      {Counter(r['tier'] for r in eval_all)}")
    print(f"  by regulator: {Counter(r['regulator'] for r in eval_all)}")
    print()
    print(f"wrote: {train_path}")
    print(f"wrote: {eval_path}")

if __name__ == "__main__":
    main()
