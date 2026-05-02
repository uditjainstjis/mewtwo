#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "claude_external_md_section_prompt.md"
DEFAULT_SCHEMA = PROJECT_ROOT / "data" / "md_workflow_schema_v1.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "multidomain_eval_external_sections_v1.json"
DEFAULT_PAIR_CATALOG = PROJECT_ROOT / "data" / "md_curated_pair_catalog_v1.json"

TASK_STYLES = [
    "identify_and_explain",
    "compare_two_options",
    "diagnose_and_fix",
    "choose_best_design",
    "explain_failure_mode",
    "map_formal_rule_to_application",
    "justify_decision",
    "analyze_buggy_reasoning",
    "summarize_with_constraints",
    "prioritize_under_constraints",
]

TESTING_FOCI = [
    "depth_reasoning",
    "breadth_synthesis",
    "error_diagnosis",
    "tradeoff_analysis",
    "counterfactual_reasoning",
    "edge_case_handling",
    "formalization",
    "translation_constraints",
    "stakeholder_usefulness",
    "mechanism_comparison",
]


def load_domains() -> list[str]:
    with open(PROJECT_ROOT / "backend" / "expert_registry.json") as f:
        registry = json.load(f)
    return sorted(registry.keys())


def build_balanced_pair_schedule(domains: list[str], count: int, seed: int) -> list[tuple[str, str]]:
    import random

    rng = random.Random(seed)
    all_pairs = list(combinations(domains, 2))
    rng.shuffle(all_pairs)
    usage = Counter({d: 0 for d in domains})
    chosen: list[tuple[str, str]] = []
    remaining = set(all_pairs)

    while len(chosen) < count and remaining:
        best_pair = None
        best_score = None
        for pair in remaining:
            a, b = pair
            score = (usage[a] + usage[b], max(usage[a], usage[b]), rng.random())
            if best_score is None or score < best_score:
                best_pair = pair
                best_score = score
        assert best_pair is not None
        chosen.append(best_pair)
        remaining.remove(best_pair)
        usage[best_pair[0]] += 1
        usage[best_pair[1]] += 1

    if len(chosen) < count:
        raise ValueError(f"Could only choose {len(chosen)} pairs for requested count {count}")
    return chosen


def load_pair_catalog(path: Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    out = []
    for row in data:
        if len(row) != 2:
            raise ValueError(f"Invalid pair row: {row}")
        out.append((str(row[0]), str(row[1])))
    return out


def build_catalog_schedule(pairs: list[tuple[str, str]], count: int, offset: int = 0) -> list[tuple[str, str]]:
    if not pairs:
        raise ValueError("Empty pair catalog")
    out = []
    idx = offset
    while len(out) < count:
        out.append(pairs[idx % len(pairs)])
        idx += 1
    return out


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


async def call_perplexity(prompt: str, *, mode: str, model: str, repo_root: Path) -> str:
    sys.path.insert(0, str(repo_root))
    import perplexity_async  # type: ignore

    cookies = load_perplexity_cookies(repo_root)
    client = await perplexity_async.Client(cookies)
    resp = await client.search(
        prompt,
        mode=mode,
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
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Response did not contain a JSON array")
    payload = text[start : end + 1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Claude occasionally emits LaTeX-ish text with odd backslash runs before
        # unsupported JSON escapes, e.g. `\$$` or `\\\$$`.
        repaired_chars: list[str] = []
        i = 0
        valid_escapes = set('"\\/bfnrtu')
        while i < len(payload):
            if payload[i] != "\\":
                repaired_chars.append(payload[i])
                i += 1
                continue
            j = i
            while j < len(payload) and payload[j] == "\\":
                j += 1
            run = j - i
            next_char = payload[j] if j < len(payload) else ""
            if next_char and next_char not in valid_escapes and run % 2 == 1:
                run -= 1
            repaired_chars.append("\\" * run)
            i = j
        repaired = "".join(repaired_chars)
        return json.loads(repaired)


def render_prompt(template: str, *, section: dict, item_specs: list[dict], domains: list[str]) -> str:
    return (
        template.replace("{{SECTION_NAME}}", section["name"])
        .replace("{{SECTION_GOAL}}", section["goal"])
        .replace("{{SECTION_DEMAND}}", section["cognitive_demand"])
        .replace("{{SECTION_WORKFLOW}}", section["workflow_type"])
        .replace("{{DOMAIN_LIST}}", "\n".join(f"- {d}" for d in domains))
        .replace("{{ITEM_SPECS}}", json.dumps(item_specs, indent=2))
        .replace("{{CREATED_UTC}}", datetime.now(timezone.utc).isoformat())
    )


def validate_items(items: list[dict], expected: dict[str, dict], allowed_domains: set[str]) -> list[dict]:
    out: list[dict] = []
    for item in items:
        item_id = str(item.get("id", "")).strip()
        if item_id not in expected:
            raise ValueError(f"Unexpected item id: {item_id!r}")
        spec = expected[item_id]
        item["domains"] = spec["domains"]
        item["required_adapters"] = spec["required_adapters"]
        if any(d not in allowed_domains for d in item["domains"]):
            raise ValueError(f"{item_id}: unknown domain in {item['domains']}")
        rubric = item.get("rubric") or {}
        rubric.setdefault("must_include_all", [])
        rubric.setdefault("must_include_any", [])
        rubric.setdefault("must_not_include", [])
        rubric.setdefault("numeric_targets", [])
        rubric.setdefault("regex_targets", [])
        rubric.setdefault("judge_focus", ["correctness", "coverage", "hallucination", "usefulness"])
        item["rubric"] = rubric
        prov = item.get("provenance") or {}
        prov["section"] = spec["section_name"]
        prov["workflow_type"] = spec["workflow_type"]
        prov["cognitive_demand"] = spec["cognitive_demand"]
        prov["testing_focus"] = spec["testing_focus"]
        item["provenance"] = prov
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate section-batched external MD dataset.")
    parser.add_argument("--items-per-section", type=int, default=10)
    parser.add_argument("--sections", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--schema-file", type=str, default=str(DEFAULT_SCHEMA))
    parser.add_argument("--pair-catalog", type=str, default=str(DEFAULT_PAIR_CATALOG))
    parser.add_argument("--model", type=str, default="claude-4.6-sonnet-thinking")
    parser.add_argument("--mode", type=str, default="reasoning")
    parser.add_argument("--proxy-repo", type=str, default=str(Path.home() / "Desktop" / "perplexity-proxy"))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--results-dir", type=str, default=str(PROJECT_ROOT / "results" / "md_section_generation"))
    parser.add_argument("--section-index", type=int, default=0, help="If >0, generate only this 1-based section.")
    parser.add_argument("--id-offset", type=int, default=0, help="Add this offset to item numbering.")
    parser.add_argument("--pair-offset", type=int, default=0, help="Start reading the pair catalog from this offset.")
    args = parser.parse_args()

    schema_path = Path(args.schema_file)
    if not schema_path.is_absolute():
        schema_path = PROJECT_ROOT / schema_path
    schema = json.loads(schema_path.read_text())
    if args.sections > len(schema):
        raise ValueError(f"Requested {args.sections} sections but schema only has {len(schema)}")
    schema = schema[: args.sections]

    total_items = args.items_per_section * len(schema)
    domains = load_domains()
    pair_catalog_path = Path(args.pair_catalog)
    if not pair_catalog_path.is_absolute():
        pair_catalog_path = PROJECT_ROOT / pair_catalog_path
    pair_catalog = load_pair_catalog(pair_catalog_path)
    pairs = build_catalog_schedule(pair_catalog, total_items, offset=args.pair_offset)
    prompt_template = PROMPT_PATH.read_text()
    repo_root = Path(args.proxy_repo)
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    section_indices = range(len(schema))
    if args.section_index > 0:
        section_indices = [args.section_index - 1]

    import asyncio

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    existing: dict[str, dict] = {}
    if out_path.is_file():
        existing = {item["id"]: item for item in json.loads(out_path.read_text())}
    all_items: list[dict] = []
    for sidx in section_indices:
        section = schema[sidx]
        batch_pairs = pairs[sidx * args.items_per_section : (sidx + 1) * args.items_per_section]
        item_specs = []
        for offset, (d1, d2) in enumerate(batch_pairs, start=1):
            global_idx = args.id_offset + sidx * args.items_per_section + offset
            item_specs.append(
                {
                    "id": f"ext_md_{global_idx:03d}",
                    "domains": [d1, d2],
                    "required_adapters": [d1, d2],
                    "section_name": section["name"],
                    "workflow_type": section["workflow_type"],
                    "cognitive_demand": section["cognitive_demand"],
                    "testing_focus": TESTING_FOCI[(global_idx - 1) % len(TESTING_FOCI)],
                    "task_style": TASK_STYLES[(global_idx - 1) % len(TASK_STYLES)],
                }
            )
        prompt = render_prompt(prompt_template, section=section, item_specs=item_specs, domains=domains)
        prompt_path = results_dir / f"section_{sidx+1:02d}.md"
        prompt_path.write_text(prompt)
        raw = asyncio.run(call_perplexity(prompt, mode=args.mode, model=args.model, repo_root=repo_root))
        raw_path = results_dir / f"section_{sidx+1:02d}.json"
        raw_path.write_text(raw)
        items = parse_json_array(raw)
        expected = {spec["id"]: spec for spec in item_specs}
        validated = validate_items(items, expected, set(domains))
        all_items.extend(validated)
        for item in validated:
            existing[item["id"]] = item
        merged = [existing[k] for k in sorted(existing.keys())]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, indent=2))
        print(f"section {sidx+1}: wrote {len(validated)} items")

    for item in all_items:
        existing[item["id"]] = item
    merged = [existing[k] for k in sorted(existing.keys())]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"Wrote {out_path} ({len(merged)} items)")


if __name__ == "__main__":
    main()
