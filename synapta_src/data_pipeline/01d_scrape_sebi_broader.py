#!/usr/bin/env python3
"""Phase 1d: scrape SEBI broader Circulars index (sid=1, ssid=7).

Uses the same iframe-PDF extractor as 01c. Capped at MAX_DOCS to be polite.

Output:
  data/sebi_corpus/pdfs/*.pdf (shared with master circulars; same naming)
  data/sebi_corpus/manifest_broader.jsonl
"""
import asyncio
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

PROJECT = Path("/home/learner/Desktop/mewtwo")
OUT_PDFS = PROJECT / "data" / "sebi_corpus" / "pdfs"
MANIFEST = PROJECT / "data" / "sebi_corpus" / "manifest_broader.jsonl"
LOG = PROJECT / "logs" / "data_pipeline" / "01d_sebi_broader.log"

OUT_PDFS.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

LISTING_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
CONCURRENCY = 4
RATE_DELAY = 0.5
MAX_DOCS = 200  # be polite — cap

# Reuse extract_pdf_url from 01c
_spec = importlib.util.spec_from_file_location(
    "sebi_master", PROJECT / "synapta_src" / "data_pipeline" / "01c_scrape_sebi_circulars.py"
)
_master = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_master)
extract_pdf_url = _master.extract_pdf_url


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


async def fetch_listing_page(session, page):
    params = {"doListing": "yes", "sid": "1", "ssid": "7", "smid": "0", "page": str(page)}
    async with session.get(LISTING_URL, params=params) as r:
        r.raise_for_status()
        return await r.text()


def parse_listing(html, page_num):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/legal/circulars/" in href and href.endswith(".html") and "/master-circulars/" not in href:
            full_url = href if href.startswith("http") else "https://www.sebi.gov.in" + href
            title = a.get_text(strip=True)[:300]
            if title and full_url not in seen:
                seen.add(full_url)
                records.append({"detail_url": full_url, "title": title, "listing_page": page_num})
    return records


async def fetch_pdf_url_from_detail(session, sem, record):
    async with sem:
        try:
            await asyncio.sleep(RATE_DELAY)
            async with session.get(record["detail_url"]) as r:
                if r.status != 200:
                    log(f"  detail HTTP {r.status}: {record['detail_url']}")
                    return None
                html = await r.text()
                pdf_url = extract_pdf_url(html)
                if pdf_url:
                    record["pdf_url"] = pdf_url
                    return record
                log(f"  no PDF: {record['detail_url']}")
        except Exception as e:
            log(f"  detail err: {str(e)[:100]}")
        return None


async def download_pdf(session, sem, record, idx, total):
    async with sem:
        url = record["pdf_url"]
        parts = url.split("?")[0].rstrip("/").split("/")
        leaf = parts[-1]
        parent = parts[-2] if len(parts) >= 2 else ""
        fname = f"{parent}_{leaf}" if parent and parent != "attachdocs" else leaf
        if not fname.lower().endswith(".pdf"):
            fname = fname + ".pdf"
        out_path = OUT_PDFS / fname
        if out_path.exists() and out_path.stat().st_size > 1000:
            log(f"[{idx}/{total}] CACHED {fname}")
            with open(out_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            return {**record, "local_path": str(out_path), "size": out_path.stat().st_size, "sha256": sha}
        try:
            await asyncio.sleep(RATE_DELAY)
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status != 200:
                    log(f"[{idx}/{total}] HTTP {r.status} {fname}")
                    return None
                content = await r.read()
                if len(content) < 1000 or not content.startswith(b"%PDF"):
                    log(f"[{idx}/{total}] BAD PDF (size={len(content)}) {fname}")
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
    log(f"=== SEBI broader Circulars scraper started, max_docs={MAX_DOCS} ===")
    headers = {"User-Agent": USER_AGENT}
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        all_records = []
        page = 1
        consecutive_errors = 0
        while len(all_records) < MAX_DOCS and page <= 20:
            try:
                html = await fetch_listing_page(session, page)
                recs = parse_listing(html, page)
                if not recs:
                    log(f"Page {page}: 0 records, stopping pagination")
                    break
                all_records.extend(recs)
                log(f"Page {page}: {len(recs)} URLs (cumulative {len(all_records)})")
                consecutive_errors = 0
                await asyncio.sleep(RATE_DELAY)
                page += 1
            except Exception as e:
                consecutive_errors += 1
                log(f"Page {page} err: {e}")
                if consecutive_errors >= 3:
                    log("3 consecutive listing errors; aborting.")
                    return
                page += 1

        all_records = all_records[:MAX_DOCS]
        log(f"Total detail URLs after cap: {len(all_records)}")
        if not all_records:
            return

        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [fetch_pdf_url_from_detail(session, sem, r) for r in all_records]
        with_pdfs = [r for r in await asyncio.gather(*tasks) if r and r.get("pdf_url")]
        log(f"Detail pages yielding PDF URLs: {len(with_pdfs)}/{len(all_records)}")

        sem2 = asyncio.Semaphore(CONCURRENCY)
        tasks = [download_pdf(session, sem2, r, i + 1, len(with_pdfs)) for i, r in enumerate(with_pdfs)]
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r]
        log(f"\nDownloaded {len(successful)}/{len(with_pdfs)} successfully")

        with open(MANIFEST, "w") as f:
            for r in successful:
                f.write(json.dumps(r) + "\n")
        log(f"Wrote manifest: {MANIFEST}")
        total_mb = sum(r["size"] for r in successful) // 1024 // 1024
        log(f"Total broader corpus added: {total_mb} MB")
        log("=== SEBI broader scraper done ===")


if __name__ == "__main__":
    asyncio.run(main())
