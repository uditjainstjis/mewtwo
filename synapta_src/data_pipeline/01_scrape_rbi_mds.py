#!/usr/bin/env python3
"""Phase 1: scrape RBI Master Direction PDFs.

Parses the MD index page, downloads PDFs concurrently with rate limiting,
records (title, url, local_path, sha256) for each.

Output:
  data/rbi_corpus/pdfs/*.PDF
  data/rbi_corpus/manifest.jsonl
"""
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

PROJECT = Path("/home/learner/Desktop/mewtwo")
OUT_PDFS = PROJECT / "data" / "rbi_corpus" / "pdfs"
MANIFEST = PROJECT / "data" / "rbi_corpus" / "manifest.jsonl"
LOG = PROJECT / "logs" / "data_pipeline" / "01_scrape.log"

OUT_PDFS.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

INDEX_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REFERER = "https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx"
CONCURRENCY = 4
RATE_DELAY = 0.5  # seconds between requests on same connection


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


async def fetch_index(session):
    log(f"Fetching index: {INDEX_URL}")
    async with session.get(INDEX_URL) as r:
        r.raise_for_status()
        return await r.text()


def parse_md_records(html):
    """Pair PDF URLs with their MD titles by table row."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    records = []
    seen_urls = set()
    if table:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            title_text = ""
            pdf_url = None
            for cell in cells:
                txt = cell.get_text(strip=True)
                if txt and len(txt) > 20 and not pdf_url:
                    title_text = txt
                for a in cell.find_all("a"):
                    href = a.get("href", "")
                    if href.lower().endswith(".pdf"):
                        pdf_url = href
            if pdf_url and title_text and pdf_url not in seen_urls:
                seen_urls.add(pdf_url)
                records.append({"title": title_text[:300], "pdf_url": pdf_url})
    return records


async def download_pdf(session, sem, record, idx, total):
    async with sem:
        url = record["pdf_url"]
        # Filename = last path segment
        fname = url.split("/")[-1]
        out_path = OUT_PDFS / fname
        if out_path.exists() and out_path.stat().st_size > 1000:
            log(f"[{idx}/{total}] CACHED {fname} ({out_path.stat().st_size//1024} KB)")
            with open(out_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            return {**record, "local_path": str(out_path), "size": out_path.stat().st_size, "sha256": sha}
        try:
            await asyncio.sleep(RATE_DELAY)
            async with session.get(url, headers={"Referer": REFERER, "Accept": "application/pdf,*/*"},
                                   timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    log(f"[{idx}/{total}] HTTP {r.status} for {url}")
                    return None
                content = await r.read()
                if len(content) < 1000 or not content.startswith(b"%PDF"):
                    log(f"[{idx}/{total}] BAD PDF (size={len(content)}) for {fname}")
                    return None
                with open(out_path, "wb") as f:
                    f.write(content)
                sha = hashlib.sha256(content).hexdigest()
                log(f"[{idx}/{total}] OK {fname} ({len(content)//1024} KB)")
                return {**record, "local_path": str(out_path), "size": len(content), "sha256": sha}
        except Exception as e:
            log(f"[{idx}/{total}] ERR {fname}: {type(e).__name__}: {str(e)[:100]}")
            return None


async def main():
    LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 60  # default 60 MDs (target ~50 successful)
    log(f"=== RBI MD scraper started, limit={LIMIT} ===")

    timeout = aiohttp.ClientTimeout(total=120)
    headers = {"User-Agent": USER_AGENT}
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        html = await fetch_index(session)
        records = parse_md_records(html)
        log(f"Parsed {len(records)} MD records from index")

        # Sort by title length (proxy: titles like "Master Direction - X" tend to be the canonical ones)
        # Prefer records starting with "Master Direction" / "Reserve Bank of India"
        def priority(r):
            t = r["title"].lower()
            if t.startswith("master direction"):
                return 0
            if t.startswith("reserve bank of india"):
                return 1
            return 2
        records.sort(key=priority)
        records = records[:LIMIT]
        log(f"Selected top {len(records)} by priority")

        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [download_pdf(session, sem, r, i+1, len(records)) for i, r in enumerate(records)]
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r]
        log(f"\nDownloaded {len(successful)}/{len(records)} successfully")

        with open(MANIFEST, "w") as f:
            for r in successful:
                f.write(json.dumps(r) + "\n")
        log(f"Wrote manifest: {MANIFEST}")

        total_size = sum(r["size"] for r in successful) // 1024 // 1024
        log(f"Total corpus size: {total_size} MB")
        log("=== Scraper complete ===")


if __name__ == "__main__":
    asyncio.run(main())
