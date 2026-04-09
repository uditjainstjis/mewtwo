#!/usr/bin/env python3
"""
Generate an externally authored multi-domain dataset using the local Perplexity proxy
and Claude Sonnet Thinking models.

This script is designed so generation can be repeated in small batches and resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "claude_external_md_dataset_prompt.md"
DEFAULT_OUT = PROJECT_ROOT / "data" / "multidomain_eval_external_claude.json"


def load_registry_domains() -> list[str]:
    registry_path = PROJECT_ROOT / "backend" / "expert_registry.json"
    with open(registry_path) as f:
        registry = json.load(f)
    return sorted(registry.keys())


def default_pairs(domains: list[str], limit_pairs: int) -> list[tuple[str, str]]:
    pairs = list(combinations(domains, 2))
    if limit_pairs <= 0 or limit_pairs >= len(pairs):
        return pairs
    # Spread across the sorted pair list deterministically.
    step = max(1, len(pairs) // limit_pairs)
    sampled = pairs[::step][:limit_pairs]
    return sampled


def render_prompt(template: str, *, domains: list[str], pairs: list[tuple[str, str]], num_items: int) -> str:
    pair_lines = [f"- {a} + {b}" for a, b in pairs]
    return (
        template.replace("{{DOMAIN_LIST}}", "\n".join(f"- {d}" for d in domains))
        .replace("{{DOMAIN_PAIRS}}", "\n".join(pair_lines))
        .replace("{{NUM_ITEMS}}", str(num_items))
        .replace("{{CREATED_UTC}}", datetime.now(timezone.utc).isoformat())
    )


def normalize_cookie_blob(text: str) -> str:
    return text.replace("\n", "").strip()


def load_perplexity_cookies(repo_root: Path) -> dict:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(f"Missing .env at {env_path}")
    text = env_path.read_text()
    match = re.search(r"PERPLEXITY_COOKIES='(.*?)'\n", text, re.S)
    if not match:
        raise ValueError("PERPLEXITY_COOKIES not found in .env")
    return json.loads(normalize_cookie_blob(match.group(1)))


async def call_perplexity(prompt: str, *, model: str, repo_root: Path) -> str:
    sys.path.insert(0, str(repo_root))
    import perplexity_async  # type: ignore

    cookies = load_perplexity_cookies(repo_root)
    client = await perplexity_async.Client(cookies)
    resp = await client.search(
        prompt,
        mode="reasoning",
        model=model,
        sources=["web"],
        files={},
        stream=False,
        language="en-US",
        follow_up=None,
        incognito=False,
    )
    if isinstance(resp, dict):
        return str(resp.get("answer", ""))
    return str(resp)


def parse_json_array(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        raise ValueError("Empty response from generator")
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("Response did not contain a JSON array")
    payload = text[start : end + 1]
    return json.loads(payload)


def validate_items(items: Iterable[dict], allowed_domains: set[str]) -> list[dict]:
    out: list[dict] = []
    seen_ids: set[str] = set()
    for item in items:
        item_id = str(item.get("id", "")).strip()
        if not item_id or item_id in seen_ids:
            raise ValueError(f"Bad or duplicate id: {item_id!r}")
        seen_ids.add(item_id)

        domains = item.get("domains") or []
        required = item.get("required_adapters") or domains
        if len(domains) != 2 or len(required) != 2:
            raise ValueError(f"{item_id}: expected exactly two domains")
        if any(d not in allowed_domains for d in domains):
            raise ValueError(f"{item_id}: unknown domain in {domains}")

        rubric = item.get("rubric") or {}
        rubric.setdefault("must_include_all", [])
        rubric.setdefault("must_include_any", [])
        rubric.setdefault("must_not_include", [])
        rubric.setdefault("numeric_targets", [])
        rubric.setdefault("regex_targets", [])
        rubric.setdefault("judge_focus", ["correctness", "coverage", "hallucination", "usefulness"])
        item["required_adapters"] = required
        item["rubric"] = rubric
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an external MD dataset through Perplexity/Claude.")
    parser.add_argument("--items", type=int, default=24, help="Number of items to request in this batch.")
    parser.add_argument("--pair-count", type=int, default=12, help="How many domain pairs to include.")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-4.6-sonnet-thinking",
        help="Perplexity reasoning model to use.",
    )
    parser.add_argument(
        "--proxy-repo",
        type=str,
        default=str(Path.home() / "Desktop" / "perplexity-proxy"),
        help="Path to local Perplexity proxy repo.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUT),
        help="Where to write the generated dataset JSON.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Only print the rendered prompt and exit.",
    )
    args = parser.parse_args()

    domains = load_registry_domains()
    pairs = default_pairs(domains, args.pair_count)
    prompt_template = PROMPT_PATH.read_text()
    prompt = render_prompt(prompt_template, domains=domains, pairs=pairs, num_items=args.items)

    if args.prompt_only:
        print(prompt)
        return

    import asyncio

    repo_root = Path(args.proxy_repo)
    text = asyncio.run(call_perplexity(prompt, model=args.model, repo_root=repo_root))
    items = parse_json_array(text)
    validated = validate_items(items, set(domains))

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validated, indent=2))
    print(f"Wrote {out_path} ({len(validated)} items)")


if __name__ == "__main__":
    main()
