#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple CLI to download all PDF attachments referenced (indirectly) by a
Nasdaq results CSV file produced by nasdaq_news_cli.py.

Usage example:

    uv run --with requests python downloadpdf.py --input results.csv

Given an input like `results.csv`, this will:
  - Create a sibling directory `results/`
  - For each row, open the `messageUrl` HTML page
  - Find all links to `*.pdf`
  - Download each PDF into `results/` as:
        <company>_<cnsCategory>_<filename>.pdf
    where:
        - <company>     = value from CSV column "company"
        - <cnsCategory> = value from CSV column "cnsCategory"
        - <filename>    = the PDF filename from the URL

uv run --with requests python downloadpdf.py --input results.csv
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests


PDF_HREF_RE = re.compile(r"""href\s*=\s*["']([^"']+?\.pdf)["']""", re.IGNORECASE)
ABS_PDF_RE = re.compile(r"""https?://[^\s"'<>]+?\.pdf\b""", re.IGNORECASE)
ESCAPED_PDF_RE = re.compile(r"""https:\\/\\/[^"'\\]+?\.pdf""", re.IGNORECASE)
# Nasdaq attachment host (often no .pdf suffix)
ATTACHMENT_RE = re.compile(r"""https?://attachment\.news\.eu\.nasdaq\.com/[^\s"'<>]+""", re.IGNORECASE)
ESCAPED_ATTACHMENT_RE = re.compile(r"""https:\\/\\/attachment\.news\.eu\.nasdaq\.com\\/[^"'\\]+""", re.IGNORECASE)

# Browser-like headers to coax server into returning same content as a normal browser
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download PDFs referenced by messageUrl pages in a Nasdaq results CSV.")
    ap.add_argument(
        "--input",
        required=True,
        help="Path to input CSV (e.g. results.csv) with at least columns: messageUrl, company, cnsCategory.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each request (default: 30).",
    )
    return ap.parse_args()


def sanitize_segment(value: str) -> str:
    """
    Turn an arbitrary string (company / category) into a safe filename segment.
    """
    value = value.strip()
    if not value:
        return "unknown"
    # Replace whitespace with underscore
    value = re.sub(r"\s+", "_", value)
    # Remove characters that are not alnum, underscore or dash
    value = re.sub(r"[^A-Za-z0-9_-]+", "", value)
    return value or "unknown"


def extract_pdf_urls(html: str, base_url: str) -> List[str]:
    """
    Find all PDF links in an HTML page and return their absolute URLs.
    """
    urls: List[str] = []
    # 1) Traditional <a href="...pdf">
    for match in PDF_HREF_RE.finditer(html):
        href = match.group(1)
        if not href:
            continue
        # Make absolute if necessary
        abs_url = urljoin(base_url, href)
        urls.append(abs_url)

    # 2) Any absolute http(s)://...pdf URLs in the text/JS
    for match in ABS_PDF_RE.finditer(html):
        abs_url = match.group(0)
        urls.append(abs_url)

    # 3) Escaped https:\/\/...\.pdf in inline JSON / JS
    for match in ESCAPED_PDF_RE.finditer(html):
        esc = match.group(0)
        # Unescape minimal \\/ -> /
        url = esc.replace("\\/", "/")
        urls.append(url)

    # 4) Direct Nasdaq attachment links without .pdf suffix
    for match in ATTACHMENT_RE.finditer(html):
        urls.append(match.group(0))

    # 5) Escaped attachment links in JSON/JS
    for match in ESCAPED_ATTACHMENT_RE.finditer(html):
        esc = match.group(0)
        url = esc.replace("\\/", "/")
        urls.append(url)

    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def iter_csv_rows(path: Path) -> Iterable[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def download_pdf(url: str, dest: Path, timeout: float) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=BROWSER_HEADERS) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded: {dest} (from {url})")
        return True
    except Exception as exc:
        print(f"ERROR downloading {url}: {exc}")
        return False


def save_pdf_stream(resp: requests.Response, dest: Path, url: str) -> bool:
    """
    Save an already-opened PDF response stream to disk.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded: {dest} (from {url})")
        return True
    except Exception as exc:
        print(f"ERROR saving PDF from {url}: {exc}")
        return False


def extract_filename_from_cd(cd: str) -> Optional[str]:
    """
    Extract filename from a Content-Disposition header if present.
    Very simple parser; good enough for typical cases.
    """
    m = re.search(r'filename\*?="?([^";]+)"?', cd, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip()


def build_output_name(company: str, category: str, pdf_url: str, content_disposition: Optional[str] = None) -> str:
    filename: Optional[str] = None
    if content_disposition:
        filename = extract_filename_from_cd(content_disposition)

    if not filename:
        parsed = urlparse(pdf_url)
        filename = Path(parsed.path).name or "file.pdf"

    # Ensure we end with .pdf so files open correctly and are recognizable
    if "." not in filename:
        filename = f"{filename}.pdf"

    comp_seg = sanitize_segment(company)
    cat_seg = sanitize_segment(category)
    return f"{comp_seg}_{cat_seg}_{filename}"


def process_csv(input_csv: Path, output_dir: Path, timeout: float) -> None:
    downloaded_any = False

    for row in iter_csv_rows(input_csv):
        message_url = (row.get("messageUrl") or "").strip()
        company = (row.get("company") or "").strip()
        category = (row.get("cnsCategory") or "").strip()

        if not message_url:
            continue

        print(f"Fetching page: {message_url}")
        try:
            with requests.get(message_url, timeout=timeout, stream=True, headers=BROWSER_HEADERS) as resp:
                resp.raise_for_status()

                content_type = (resp.headers.get("Content-Type") or "").lower()
                cd_header = resp.headers.get("Content-Disposition") or ""
                cd_filename = extract_filename_from_cd(cd_header) if cd_header else None

                # Case 1: messageUrl itself returns a PDF (e.g. Content-Disposition: attachment)
                if (
                    "pdf" in content_type
                    or message_url.lower().endswith(".pdf")
                    or resp.url.lower().endswith(".pdf")
                    or (cd_filename and cd_filename.lower().endswith(".pdf"))
                ):
                    pdf_url = resp.url or message_url
                    out_name = build_output_name(
                        company,
                        category,
                        pdf_url,
                        content_disposition=cd_header if cd_header else None,
                    )
                    dest = output_dir / out_name
                    if dest.exists():
                        print(f"Skipping existing file: {dest}")
                        continue
                    print(
                        "Detected direct PDF response: "
                        f"Content-Type={content_type!r}, Content-Disposition={cd_header!r}"
                    )
                    if save_pdf_stream(resp, dest, pdf_url):
                        downloaded_any = True
                    continue

                # Case 2: HTML page with links to PDFs
                html = resp.text
        except Exception as exc:
            print(f"ERROR fetching messageUrl {message_url}: {exc}")
            continue

        pdf_urls = extract_pdf_urls(html, message_url)
        if not pdf_urls:
            pdf_count = html.lower().count("pdf")
            print(f"No PDF links found at {message_url}")
            print(f"  Content-Type: {content_type}")
            if cd_header:
                print(f"  Content-Disposition: {cd_header}")
            print(f"  HTML length: {len(html)} chars, 'pdf' occurrences: {pdf_count}")

            # Show up to first 3 contexts where 'pdf' appears to help debugging patterns
            lowered = html.lower()
            contexts_shown = 0
            for m in re.finditer("pdf", lowered):
                start = max(0, m.start() - 200)
                end = min(len(html), m.start() + 200)
                ctx = html[start:end].replace("\n", " ")
                print(f"  Context around 'pdf' #{contexts_shown + 1}: {ctx!r}")
                contexts_shown += 1
                if contexts_shown >= 3:
                    break
            continue

        for pdf_url in pdf_urls:
            out_name = build_output_name(company, category, pdf_url)
            dest = output_dir / out_name
            # Skip if already downloaded
            if dest.exists():
                print(f"Skipping existing file: {dest}")
                continue
            if download_pdf(pdf_url, dest, timeout=timeout):
                downloaded_any = True

    if not downloaded_any:
        print("No PDFs downloaded (check messageUrl column and page contents).")
    else:
        print(f"Done. PDFs saved under: {output_dir}")


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input)
    if not input_csv.is_file():
        print(f"Input CSV not found: {input_csv}")
        return 1

    # Directory named after CSV stem, e.g. results.csv -> results/
    output_dir = input_csv.with_suffix("")
    process_csv(input_csv=input_csv, output_dir=output_dir, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
