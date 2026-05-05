#!/usr/bin/env python3
"""Phase 1c: scrape SEBI Master Circulars.

SEBI has no anti-scraping per recon. ~133 master circulars across 6 listing pages.
Two-step: scrape listing pages → extract detail HTMLs → grep PDF link from each.

Output:
  data/sebi_corpus/pdfs/*.pdf
  data/sebi_corpus/manifest.jsonl
"""
import asyncio
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

PROJECT = Path("/home/learner/Desktop/mewtwo")
OUT_PDFS = PROJECT / "data" / "sebi_corpus" / "pdfs"
MANIFEST = PROJECT / "data" / "sebi_corpus" / "manifest.jsonl"
LOG = PROJECT / "logs" / "data_pipeline" / "01c_sebi.log"

OUT_PDFS.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

LISTING_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
CONCURRENCY = 4
RATE_DELAY = 0.5
N_PAGES = 6  # Master Circulars only, ~133 docs total


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


async def fetch_listing_page(session, page):
    """SEBI listing page params: doListing=yes&sid=1&ssid=6&smid=0&page=N"""
    params = {
        "doListing": "yes",
        "sid": "1",
        "ssid": "6",
        "smid": "0",
        "page": str(page),
    }
    async with session.get(LISTING_URL, params=params) as r:
        r.raise_for_status()
        return await r.text()


def parse_listing(html, page_num):
    """Extract detail page URLs and titles from a listing HTML."""
    soup = BeautifulSoup(html, "html.parser")
    records = []
    # Detail pages live at /legal/master-circulars/[mon-year]/[slug]_[id].html
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/legal/master-circulars/" in href and href.endswith(".html"):
            full_url = href if href.startswith("http") else "https://www.sebi.gov.in" + href
            title = a.get_text(strip=True)[:300]
            if title and full_url not in [r["detail_url"] for r in records]:
                records.append({"detail_url": full_url, "title": title, "listing_page": page_num})
    return records


# Regex matches PDF URL anywhere in HTML — anchor href, iframe src=...?file=...pdf,
# protocol-relative //www.sebi.gov.in/..., absolute https://, or site-root /sebi_data/...
PDF_URL_RE = re.compile(
    r"""(?:https?:)?//www\.sebi\.gov\.in/sebi_data/[^\s'"<>?]+?\.pdf"""
    r"""|/sebi_data/[^\s'"<>?]+?\.pdf""",
    re.IGNORECASE,
)


def extract_pdf_url(html):
    """Find the first /sebi_data/.../*.pdf reference anywhere in the HTML.

    The detail page embeds the PDF inside an <iframe src='.../web/?file=https://www.sebi.gov.in/sebi_data/.../X.pdf'>
    so anchor-only parsing misses it. We scan the raw HTML with a regex.
    """
    m = PDF_URL_RE.search(html)
    if not m:
        return None
    url = m.group(0)
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://www.sebi.gov.in" + url
    return url


async def fetch_pdf_url_from_detail(session, sem, record):
    """Open the detail HTML and find the PDF link inside."""
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
                log(f"  no PDF found in detail: {record['detail_url']}")
        except Exception as e:
            log(f"  detail fetch err: {str(e)[:100]}")
        return None


async def download_pdf(session, sem, record, idx, total):
    async with sem:
        url = record["pdf_url"]
        # filename: <mon-year>_<timestamp>.pdf to avoid collisions across months
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
    log(f"=== SEBI Master Circulars scraper started, pages={N_PAGES} ===")
    headers = {"User-Agent": USER_AGENT}
    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:
        # Step 1: scrape listing pages
        all_detail_records = []
        for page in range(1, N_PAGES + 1):
            try:
                html = await fetch_listing_page(session, page)
                recs = parse_listing(html, page)
                log(f"Page {page}: {len(recs)} detail URLs")
                all_detail_records.extend(recs)
                await asyncio.sleep(RATE_DELAY)
            except Exception as e:
                log(f"Page {page} err: {e}")
        log(f"Total detail URLs: {len(all_detail_records)}")
        if not all_detail_records:
            log("No detail URLs found; aborting.")
            return

        # Step 2: fetch PDF URL from each detail page
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = [fetch_pdf_url_from_detail(session, sem, r) for r in all_detail_records]
        with_pdfs = [r for r in await asyncio.gather(*tasks) if r and r.get("pdf_url")]
        log(f"Detail pages yielding PDF URLs: {len(with_pdfs)}/{len(all_detail_records)}")

        # Step 3: download PDFs
        sem2 = asyncio.Semaphore(CONCURRENCY)
        tasks = [download_pdf(session, sem2, r, i+1, len(with_pdfs)) for i, r in enumerate(with_pdfs)]
        results = await asyncio.gather(*tasks)
        successful = [r for r in results if r]
        log(f"\nDownloaded {len(successful)}/{len(with_pdfs)} successfully")

        with open(MANIFEST, "w") as f:
            for r in successful:
                f.write(json.dumps(r) + "\n")
        log(f"Wrote manifest: {MANIFEST}")
        total_mb = sum(r["size"] for r in successful) // 1024 // 1024
        log(f"Total corpus: {total_mb} MB")
        log("=== SEBI scraper done ===")


if __name__ == "__main__":
    asyncio.run(main())
