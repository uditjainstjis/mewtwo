"""Shared helpers for multi-domain benchmark JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def filter_two_or_more_domains(items: list[dict]) -> list[dict]:
    """Keep items that list at least two adapters/domains (matches injection eval)."""
    return [
        it
        for it in items
        if len(it.get("required_adapters") or it.get("domains") or []) >= 2
    ]


def load_md_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def prepare_md_items(
    project_root: Path,
    data_rel: str,
    *,
    limit: int | None = None,
    two_domain_only: bool = True,
) -> tuple[Path, list[dict]]:
    """Resolve path, apply limit slice, optional two-domain filter."""
    p = Path(data_rel)
    if not p.is_absolute():
        p = project_root / p
    items = load_md_json(p)
    if limit and limit > 0:
        items = items[: int(limit)]
    if two_domain_only:
        items = filter_two_or_more_domains(items)
    return p, items
