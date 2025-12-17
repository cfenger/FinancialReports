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

uv run --with requests,beautifulsoup4,pdfminer.six python downloadpdf.py --input insider.csv --to-text
"""

import argparse
import csv
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text


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
    ap.add_argument(
        "--allow-external",
        action="store_true",
        help="Allow downloading PDFs from domains different from the messageUrl base domain.",
    )
    ap.add_argument(
        "--to-text",
        action="store_true",
        help="Save extracted text (.txt) instead of PDFs/HTML (uses pdfminer.six for PDFs and BeautifulSoup for HTML).",
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


def get_base_domain(host: str) -> str:
    """
    Very simple "registrable" domain helper: takes last two labels.
    Example: view.news.eu.nasdaq.com -> nasdaq.com
    """
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


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


def download_pdf(url: str, dest: Path, timeout: float) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=BROWSER_HEADERS) as r:
            r.raise_for_status()
            content_type = (r.headers.get("Content-Type") or "").lower()
            final_url = (r.url or url).lower()
            if "pdf" not in content_type and not final_url.endswith(".pdf"):
                print(
                    f"Skipping non-PDF URL: {url} "
                    f"(resolved as {r.url!r}, Content-Type={content_type!r})"
                )
                return "skipped"
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        print(f"Downloaded: {dest} (from {url})")
        return "saved"
    except Exception as exc:
        print(f"ERROR downloading {url}: {exc}")
        return "failed"


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


def save_pdf_bytes_as_text(pdf_bytes: bytes, dest: Path) -> bool:
    """
    Extract text from PDF bytes and save to a UTF-8 .txt file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = pdf_extract_text(BytesIO(pdf_bytes))
        if not text or not text.strip():
            print(f"No text extracted from PDF; skipping save: {dest}")
            return False
        with dest.open("w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as exc:
        print(f"ERROR extracting PDF text to {dest}: {exc}")
        return False


def save_html_as_text(html: str, dest: Path) -> bool:
    """
    Extract readable text from HTML and save to a UTF-8 .txt file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        with dest.open("w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as exc:
        print(f"ERROR saving HTML text to {dest}: {exc}")
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


def process_csv(input_csv: Path, output_dir: Path, timeout: float,
                allow_external: bool, to_text: bool) -> None:
    downloaded_any = False
    saved_count = 0
    failed_count = 0
    skipped_existing_count = 0
    excluded_count = 0
    html_saved_count = 0
    failed_urls: List[str] = []
    failed_urls_seen = set()

    def record_failed_url(url: str) -> None:
        if not url:
            return
        url_lower = url.lower()
        if not url_lower.startswith("https://"):
            return
        if url_lower in failed_urls_seen:
            return
        failed_urls_seen.add(url_lower)
        failed_urls.append(url)

    for row in iter_csv_rows(input_csv):
        message_url = (row.get("messageUrl") or "").strip()
        company = (row.get("company") or "").strip()
        category = (row.get("cnsCategory") or "").strip()

        if not message_url:
            continue

        print(f"Fetching page: {message_url}")
        message_host = urlparse(message_url).hostname or ""
        allowed_domain = get_base_domain(message_host)

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
                    pdf_host = urlparse(pdf_url).hostname or ""
                    if (
                        allowed_domain
                        and get_base_domain(pdf_host) != allowed_domain
                        and not allow_external
                    ):
                        print(
                            f"Skipping direct PDF on external domain: {pdf_url} "
                            f"(host {pdf_host!r} not under {allowed_domain!r})"
                        )
                        excluded_count += 1
                        continue
                    out_name = build_output_name(
                        company,
                        category,
                        pdf_url,
                        content_disposition=cd_header if cd_header else None,
                    )
                    if to_text:
                        text_name = Path(out_name).with_suffix(".txt").name
                        dest_txt = output_dir / text_name
                        if dest_txt.exists():
                            print(f"Skipping existing file: {dest_txt}")
                            skipped_existing_count += 1
                            continue
                        print(
                            "Detected direct PDF response: "
                            f"Content-Type={content_type!r}, Content-Disposition={cd_header!r}"
                        )
                        pdf_bytes = resp.content
                        if save_pdf_bytes_as_text(pdf_bytes, dest_txt):
                            downloaded_any = True
                            saved_count += 1
                            print(f"Downloaded text: {dest_txt} (from direct PDF {pdf_url})")
                        else:
                            failed_count += 1
                            record_failed_url(pdf_url)
                            print(f"Failed: {pdf_url} -> {dest_txt}")
                    else:
                        dest = output_dir / out_name
                        if dest.exists():
                            print(f"Skipping existing file: {dest}")
                            skipped_existing_count += 1
                            continue
                        print(
                            "Detected direct PDF response: "
                            f"Content-Type={content_type!r}, Content-Disposition={cd_header!r}"
                        )
                        if save_pdf_stream(resp, dest, pdf_url):
                            downloaded_any = True
                            saved_count += 1
                        else:
                            failed_count += 1
                            record_failed_url(pdf_url)
                            print(f"Failed: {pdf_url} -> {dest}")
                    continue

                # Case 2: HTML page with links to PDFs
                html = resp.text
        except Exception as exc:
            print(f"ERROR fetching messageUrl {message_url}: {exc}")
            failed_count += 1
            record_failed_url(message_url)
            print(f"Failed: {message_url} (messageUrl fetch)")
            continue

        pdf_urls = extract_pdf_urls(html, message_url)
        # Only keep PDFs on same registrable domain as messageUrl (unless allow_external)
        filtered_urls: List[str] = []
        for pdf_url in pdf_urls:
            pdf_host = urlparse(pdf_url).hostname or ""
            if (
                allowed_domain
                and get_base_domain(pdf_host) != allowed_domain
                and not allow_external
            ):
                print(
                    f"Skipping external PDF URL: {pdf_url} "
                    f"(host {pdf_host!r} not under {allowed_domain!r})"
                )
                excluded_count += 1
                continue
            filtered_urls.append(pdf_url)
        pdf_urls = filtered_urls

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

            comp_seg = sanitize_segment(company)
            cat_seg = sanitize_segment(category)
            if to_text:
                parsed_url = urlparse(message_url)
                slug_source = Path(parsed_url.path).name or "message"
                if parsed_url.query:
                    slug_source = f"{slug_source}_{parsed_url.query}"
                slug = sanitize_segment(slug_source) or "message"
                text_filename = f"{comp_seg}_{cat_seg}_{slug}.txt"
                dest = output_dir / text_filename
                if dest.exists():
                    print(f"Skipping existing file: {dest}")
                    skipped_existing_count += 1
                    continue
                if save_html_as_text(html, dest):
                    downloaded_any = True
                    html_saved_count += 1
                    print(f"Saved HTML text (no PDFs): {dest}")
                else:
                    failed_count += 1
                    record_failed_url(message_url)
                    print(f"Failed: {message_url} -> {dest}")
            else:
                parsed_url = urlparse(message_url)
                slug_source = Path(parsed_url.path).name or "message"
                if parsed_url.query:
                    slug_source = f"{slug_source}_{parsed_url.query}"
                slug = sanitize_segment(slug_source) or "message"
                html_filename = f"{comp_seg}_{cat_seg}_{slug}"
                if not html_filename.lower().endswith(".html"):
                    html_filename = f"{html_filename}.html"
                dest = output_dir / html_filename
                base_stem = dest.stem
                ext = dest.suffix
                suffix = 1
                while dest.exists():
                    dest = dest.with_name(f"{base_stem}_{suffix}{ext}")
                    suffix += 1
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("w", encoding="utf-8") as f:
                    f.write(html)
                print(f"Saved HTML (no PDFs): {dest}")
                html_saved_count += 1
            continue

        for pdf_url in pdf_urls:
            out_name = build_output_name(company, category, pdf_url)
            if to_text:
                text_name = Path(out_name).with_suffix(".txt").name
                dest = output_dir / text_name
                if dest.exists():
                    print(f"Skipping existing file: {dest}")
                    skipped_existing_count += 1
                    continue
                try:
                    with requests.get(pdf_url, stream=True, timeout=timeout, headers=BROWSER_HEADERS) as r:
                        r.raise_for_status()
                        ct = (r.headers.get("Content-Type") or "").lower()
                        final_url = (r.url or pdf_url).lower()
                        if "pdf" not in ct and not final_url.endswith(".pdf"):
                            print(
                                f"Skipping non-PDF URL in text mode: {pdf_url} "
                                f"(resolved as {r.url!r}, Content-Type={ct!r})"
                            )
                            excluded_count += 1
                            continue
                        pdf_bytes = r.content
                except Exception as exc:
                    print(f"ERROR downloading {pdf_url}: {exc}")
                    failed_count += 1
                    record_failed_url(pdf_url)
                    print(f"Failed: {pdf_url} -> {dest}")
                    continue
                if save_pdf_bytes_as_text(pdf_bytes, dest):
                    downloaded_any = True
                    saved_count += 1
                    print(f"Downloaded text: {dest} (from {pdf_url})")
                else:
                    failed_count += 1
                    record_failed_url(pdf_url)
                    print(f"Failed: {pdf_url} -> {dest}")
            else:
                dest = output_dir / out_name
                # Skip if already downloaded
                if dest.exists():
                    print(f"Skipping existing file: {dest}")
                    skipped_existing_count += 1
                    continue
                result = download_pdf(pdf_url, dest, timeout=timeout)
                if result == "saved":
                    downloaded_any = True
                    saved_count += 1
                elif result == "failed":
                    failed_count += 1
                    record_failed_url(pdf_url)
                    print(f"Failed: {pdf_url} -> {dest}")
                elif result == "skipped":
                    excluded_count += 1

    if not downloaded_any:
        print("No PDFs downloaded (check messageUrl column and page contents).")
    if failed_urls:
        for failed_url in failed_urls:
            print(f"Failed: {failed_url}")
    print(
        f"Summary: Download {saved_count}, "
        f"failed {failed_count}, "
        f"skipped {skipped_existing_count} existing, "
        f"excluded {excluded_count} PDF(s). "
        f"Downloaded {html_saved_count} HTML file(s)."
    )


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input)
    if not input_csv.is_file():
        print(f"Input CSV not found: {input_csv}")
        return 1

    # Directory named after CSV stem, e.g. results.csv -> results/
    output_dir = input_csv.with_suffix("")
    process_csv(
        input_csv=input_csv,
        output_dir=output_dir,
        timeout=args.timeout,
        allow_external=args.allow_external,
        to_text=args.to_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
