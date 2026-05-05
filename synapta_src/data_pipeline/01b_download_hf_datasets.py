#!/usr/bin/env python3
"""Download curated HuggingFace BFSI datasets to local disk.

Apache-2.0 / OpenRAIL only — license-clean for production use.

Output: data/hf_bfsi/<dataset_name>/
Manifest: data/hf_bfsi/manifest.json
"""
import json
import sys
import time
from pathlib import Path

PROJECT = Path("/home/learner/Desktop/mewtwo")
OUT = PROJECT / "data" / "hf_bfsi"
LOG = PROJECT / "logs" / "data_pipeline" / "01b_hf.log"
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# Curated datasets — license-clean only
DATASETS = [
    {
        "name": "prakhar146/indian-finance-rag",
        "purpose": "RAG store: pre-chunked RBI+SEBI+IRDAI+NSE/BSE+tax with FAISS index",
        "license": "Apache-2.0",
        "use_for": "raw context corpus for chunking, NOT for QA pairs (no Q&A schema)",
    },
    {
        "name": "lirus18/rbi",
        "purpose": "Small clean RBI Q&A pairs",
        "license": "llama2",
        "use_for": "SFT seed (1709 examples)",
    },
    {
        "name": "iam-sathya/rbi-test",
        "purpose": "Raw RBI Master Direction PDFs as text",
        "license": "OpenRAIL",
        "use_for": "additional raw context (98 MDs)",
    },
]


def main():
    from datasets import load_dataset
    manifest = []
    log(f"=== HF BFSI dataset downloader started, {len(DATASETS)} datasets ===")
    for ds in DATASETS:
        name = ds["name"]
        local_dir = OUT / name.replace("/", "_")
        log(f"Downloading {name} ({ds['license']}) → {local_dir}")
        try:
            t0 = time.time()
            d = load_dataset(name, cache_dir=str(OUT / "_cache"))
            # Save each split as JSONL
            local_dir.mkdir(parents=True, exist_ok=True)
            row_count = 0
            for split_name in d.keys():
                split = d[split_name]
                out_jsonl = local_dir / f"{split_name}.jsonl"
                with open(out_jsonl, "w") as f:
                    for row in split:
                        f.write(json.dumps(row, default=str) + "\n")
                        row_count += 1
                log(f"  {split_name}: {len(split)} rows -> {out_jsonl.name}")
            elapsed = time.time() - t0
            log(f"  OK in {elapsed:.1f}s, total rows: {row_count}")
            manifest.append({**ds, "local_dir": str(local_dir), "row_count": row_count, "status": "ok"})
        except Exception as e:
            log(f"  ERR: {type(e).__name__}: {str(e)[:200]}")
            manifest.append({**ds, "status": "error", "error": str(e)[:300]})

    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"\nManifest saved: {OUT / 'manifest.json'}")
    log("=== Done ===")


if __name__ == "__main__":
    main()
