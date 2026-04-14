"""
Draft one multi-domain benchmark item using the optional Perplexity proxy (Track B).

Requires CLUSTER_USE_PERPLEXITY_PROXY=1 and ~/Desktop/perplexity-proxy (see backend/proxy_bridge.py).

Output: single JSON object on stdout (add to data/multidomain_eval_external.json array).

Usage:
  CLUSTER_USE_PERPLEXITY_PROXY=1 python3 src/eval/draft_external_md_item.py \\
    --id ext_02 --domain-a LEGAL_ANALYSIS --domain-b PYTHON_LOGIC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from proxy_bridge import ProxyBridge  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Item id, e.g. ext_02")
    parser.add_argument("--domain-a", required=True, dest="d1")
    parser.add_argument("--domain-b", required=True, dest="d2")
    parser.add_argument(
        "--mode",
        default="reasoning",
        help="Proxy mode passed to ProxyBridge.ask (default: reasoning).",
    )
    args = parser.parse_args()

    bridge = ProxyBridge()
    if not bridge.enabled:
        print(
            "Set CLUSTER_USE_PERPLEXITY_PROXY=1 and ensure ~/Desktop/perplexity-proxy works.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    spec = (
        f"You are designing a benchmark item for evaluating a small LLM with two expert adapters: "
        f"{args.d1} and {args.d2}. "
        "Produce ONE challenging question that requires integrated knowledge from BOTH domains. "
        "Then provide a concise reference answer (3-6 sentences) suitable as a gold label. "
        "Respond in exactly this plain-text format (no markdown):\n"
        "QUESTION: <single paragraph>\n"
        "REFERENCE: <single paragraph>\n"
    )
    text = bridge.ask(spec, mode=args.mode)
    if not text or not text.strip():
        print("Proxy returned empty response.", file=sys.stderr)
        raise SystemExit(2)

    q, ref = "", ""
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("QUESTION:"):
            q = s.split(":", 1)[1].strip()
        elif s.upper().startswith("REFERENCE:"):
            ref = s.split(":", 1)[1].strip()
    if not q or not ref:
        q = text[:2000]
        ref = "(parse failed; paste REFERENCE manually — raw response below)\n" + text

    item = {
        "id": args.id,
        "domains": [args.d1, args.d2],
        "required_adapters": [args.d1, args.d2],
        "question": q,
        "reference_answer": ref,
        "provenance": {
            "question_author": "perplexity-proxy-draft",
            "reference_author": "perplexity-proxy-draft",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    print(json.dumps(item, indent=2))


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
