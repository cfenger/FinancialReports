"""CLI entry point for parsing insider transaction notices into CSV output.

Example usage:
    uv run python interpret.py --task insider --input-dir insider
    uv run python interpret.py --task insider --input-dir insider --output insider_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

FIELDNAMES = ["date", "company", "name", "shares", "price", "total_value", "side", "reason"]

BUY_KEYWORDS = ("acquisition", "purchase", "buy", "receipt", "subscription", "incentive")
SELL_KEYWORDS = ("disposal", "sale", "sell", "luovutus", "divest")
NATURE_TRANSLATIONS = {
    # Finnish -> English (style aligned with other rows)
    "luovutus": "DISPOSAL",
    "hankinta": "ACQUISITION",
}

TRANSACTION_LINE_RE = re.compile(
    r"\(\d+\)\s*:\s*(?:vol(?:ume|yymi|ym)\b[^:]*:\s*([\d\s.,]+)).*?"
    r"(?:unit\s*price|pris per aktie|pris|yksikk\w*hinta|hinta)\s*:\s*([\d\s.,]+)",
    re.IGNORECASE,
)

FALLBACK_TRANSACTION_RE = re.compile(
    r"volume:\s*([\d\s.,]+)\s*(?:unit\s*price|average price|volume weighted average price|keskihinta|price)\s*[: ]\s*([\d\s.,]+)",
    re.IGNORECASE,
)

DATE_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"transaction date[:\s]+(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    re.compile(r"transaction date[:\s]+(\d{2}[./]\d{2}[./]\d{4})", re.IGNORECASE),
    re.compile(r"liiketoimen[^:]{0,30}:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    re.compile(r"liiketoimen[^:]{0,30}:\s*(\d{2}[./]\d{2}[./]\d{4})", re.IGNORECASE),
]


def normalize_line(line: str) -> str:
    """Lowercase, strip diacritics, and collapse whitespace for easier matching."""
    ascii_line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("ascii")
    lowered = ascii_line.lower()
    return re.sub(r"\s+", " ", lowered).strip()


def normalize_text_for_search(text: str) -> str:
    """Normalize an entire block of text for broad regex searches."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    lowered = ascii_text.lower()
    return re.sub(r"\s+", " ", lowered)


def value_after_colon(line: str) -> str:
    """Return the substring after the first colon, if any."""
    if ":" not in line:
        return ""
    return line.split(":", 1)[1].strip()


def parse_company_from_filename(name: str) -> str:
    """Best-effort company inference from the filename."""
    stem = Path(name).stem
    stem = re.split(r"_ledende_medarbejderes_transaktioner", stem, flags=re.IGNORECASE)[0]
    stem = re.split(r"_managers", stem, flags=re.IGNORECASE)[0]
    stem = re.split(r"_view_", stem, maxsplit=1)[0]
    stem = re.split(r"_lang", stem, maxsplit=1)[0]
    pretty = stem.replace("_", " ").strip()
    return pretty


def extract_company(lines: Sequence[Tuple[str, str]], filename: Path) -> str:
    for raw, norm in lines:
        if "issuer:" in norm or "liikkeeseenlaskija:" in norm:
            candidate = value_after_colon(raw)
            if candidate:
                return candidate
    return parse_company_from_filename(filename.name)


def extract_name(lines: Sequence[Tuple[str, str]]) -> str:
    """Prefer the name inside the person-subject section; fall back to the last seen name."""
    in_subject_block = False
    candidate = ""
    for raw, norm in lines:
        if "person subject" in norm or "notification requirement" in norm or "ilmoitusvelvollinen" in norm:
            in_subject_block = True
        if "name:" in norm or "nimi:" in norm or "namn:" in norm:
            value = value_after_colon(raw)
            if in_subject_block:
                candidate = value  # keep the latest in this section
            elif not candidate:
                candidate = value
    return candidate


def normalize_date_token(date_str: str) -> str:
    date_str = date_str.strip()
    m = re.match(r"(\d{2})[./](\d{2})[./](\d{4})", date_str)
    if m:
        day, month, year = m.groups()
        return f"{year}-{month}-{day}"
    return date_str


def extract_date(raw_text: str) -> str:
    normalized = normalize_text_for_search(raw_text)
    for pattern in DATE_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return normalize_date_token(match.group(1))
    fallback_iso = re.search(r"\d{4}-\d{2}-\d{2}", raw_text)
    if fallback_iso:
        return fallback_iso.group(0)
    fallback_eu = re.search(r"\d{2}[./]\d{2}[./]\d{4}", raw_text)
    if fallback_eu:
        return normalize_date_token(fallback_eu.group(0))
    return ""


def extract_nature(lines: Sequence[Tuple[str, str]]) -> str:
    for raw, norm in lines:
        if "nature of transaction" in norm or "liiketoimen luonne" in norm:
            value = value_after_colon(raw)
            if value:
                return value
    return ""


def translate_nature(nature: str) -> str:
    """Translate known non-English nature values into English; otherwise return as-is."""
    stripped = nature.strip()
    if not stripped:
        return nature
    key = normalize_line(stripped)
    return NATURE_TRANSLATIONS.get(key, nature)


def determine_side(nature: str) -> str:
    lowered = nature.lower()
    if any(keyword in lowered for keyword in SELL_KEYWORDS):
        return "sell"
    if any(keyword in lowered for keyword in BUY_KEYWORDS):
        return "buy"
    return ""


def clean_int(value: str) -> str:
    cleaned = re.sub(r"[^\d]", "", value)
    return cleaned


def clean_decimal(value: str) -> str:
    compact = value.replace(" ", "")
    if "," in compact and "." in compact:
        last_comma = compact.rfind(",")
        last_dot = compact.rfind(".")
        if last_comma > last_dot:
            # Treat comma as decimal separator; dots as thousands separators.
            compact = compact.replace(".", "").replace(",", ".")
        else:
            # Treat dot as decimal separator; commas as thousands separators.
            compact = compact.replace(",", "")
    elif "," in compact:
        # Only comma present -> assume decimal separator.
        compact = compact.replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", compact)
    return match.group(0) if match else ""


def decimal_to_str(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized)


def compute_total_value(shares: str, price: str) -> str:
    if not shares or not price:
        return ""
    try:
        total = Decimal(shares) * Decimal(price)
    except (InvalidOperation, ValueError):
        return ""
    return decimal_to_str(total)


def extract_transactions(lines: Sequence[Tuple[str, str]], normalized_text: str) -> List[Tuple[str, str]]:
    transactions: List[Tuple[str, str]] = []
    for raw, norm in lines:
        if "aggregated transactions" in norm or "yhdistetyt" in norm:
            continue
        match = TRANSACTION_LINE_RE.search(norm)
        if match:
            transactions.append((match.group(1), match.group(2)))
    if transactions:
        return transactions
    filtered_lines: List[str] = []
    skip_next = 0
    for _, norm in lines:
        if "aggregated transactions" in norm or "yhdistetyt" in norm:
            skip_next = 1  # skip this and the following line containing aggregated totals
            continue
        if skip_next > 0:
            skip_next -= 1
            continue
        filtered_lines.append(norm)
    filtered_text = " ".join(filtered_lines)
    fallback_matches = FALLBACK_TRANSACTION_RE.findall(filtered_text)
    for volume, price in fallback_matches:
        transactions.append((volume, price))
    return transactions


def build_lines(text: str) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append((stripped, normalize_line(stripped)))
    return lines


def parse_insider_file(text: str, path: Path) -> List[dict]:
    lines = build_lines(text)
    normalized_text = normalize_text_for_search(text)
    company = extract_company(lines, path)
    name = extract_name(lines)
    date = extract_date(text)
    nature = translate_nature(extract_nature(lines))
    side = determine_side(nature)
    transactions = extract_transactions(lines, normalized_text)

    rows: List[dict] = []
    if not transactions:
        logging.warning("No transactions parsed from %s", path.name)
        rows.append(
            {
                "date": date,
                "company": company,
                "name": name,
                "shares": "",
                "price": "",
                "total_value": "",
                "side": side,
                "reason": nature,
            }
        )
        return rows

    for volume, price in transactions:
        shares_clean = clean_int(volume)
        price_clean = clean_decimal(price)
        rows.append(
            {
                "date": date,
                "company": company,
                "name": name,
                "shares": shares_clean,
                "price": price_clean,
                "total_value": compute_total_value(shares_clean, price_clean),
                "side": side,
                "reason": nature,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse insider transaction .txt files into CSV.")
    parser.add_argument(
        "--task",
        required=True,
        choices=["insider"],
        help="Which analysis task to run.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing insider .txt files (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("insider_summary.csv"),
        help="Path to the CSV file to write.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help=(
            "If set, aggregate rows with the same date, company and name "
            "by summing shares and recomputing average price and total value."
        ),
    )
    return parser.parse_args()


def _collect_rows(rows: List[dict]) -> List[dict]:
    """Aggregate rows and optionally cancel offsetting buy/sell transactions.

    Step 1: aggregate by (date, company, name, side, reason), so multiple
    partial rows for the same actor and side are merged.
    Step 2: for each (date, company, name), if there is exactly one buy row
    and one sell row with identical share counts, drop both so fully
    offsetting transactions disappear from the summary.
    """
    grouped: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("date", ""),
            row.get("company", ""),
            row.get("name", ""),
            row.get("side", ""),
            row.get("reason", ""),
        )
        group = grouped.get(key)
        if group is None:
            group = {
                "prototype": dict(row),
                "shares": Decimal("0"),
                "total_value": Decimal("0"),
                "prices": [],
            }
            grouped[key] = group

        # Parse numeric fields best-effort.
        shares_str = row.get("shares") or ""
        price_str = row.get("price") or ""
        total_str = row.get("total_value") or ""

        try:
            shares_val = Decimal(shares_str) if shares_str else Decimal("0")
        except InvalidOperation:
            shares_val = Decimal("0")
        try:
            price_val = Decimal(price_str) if price_str else None
        except InvalidOperation:
            price_val = None
        try:
            total_val = Decimal(total_str) if total_str else None
        except InvalidOperation:
            total_val = None

        if total_val is None and price_val is not None and shares_val:
            total_val = shares_val * price_val

        group["shares"] += shares_val
        if total_val is not None:
            group["total_value"] += total_val
        if price_val is not None:
            group["prices"].append(price_val)

    # First pass: build aggregated rows per (date, company, name, side, reason).
    aggregated: List[dict] = []
    for group in grouped.values():
        base = group["prototype"]
        shares_sum: Decimal = group["shares"]
        total_sum: Decimal = group["total_value"]
        prices: List[Decimal] = group["prices"]

        if shares_sum and total_sum:
            avg_price = total_sum / shares_sum
        elif shares_sum and prices:
            # Fallback to simple average of prices.
            avg_price = sum(prices) / Decimal(len(prices))
            total_sum = shares_sum * avg_price
        else:
            avg_price = None

        base["shares"] = decimal_to_str(shares_sum) if shares_sum else ""
        base["price"] = decimal_to_str(avg_price) if avg_price is not None else ""
        base["total_value"] = decimal_to_str(total_sum) if total_sum else ""
        aggregated.append(base)

    # Second pass: cancel exact offset pairs per (date, company, name).
    by_actor: dict[tuple, List[dict]] = {}
    for row in aggregated:
        actor_key = (row.get("date", ""), row.get("company", ""), row.get("name", ""))
        by_actor.setdefault(actor_key, []).append(row)

    result: List[dict] = []
    for actor_rows in by_actor.values():
        if len(actor_rows) == 2:
            buy_row = next((r for r in actor_rows if (r.get("side") or "").lower() == "buy"), None)
            sell_row = next((r for r in actor_rows if (r.get("side") or "").lower() == "sell"), None)
            if buy_row is not None and sell_row is not None:
                try:
                    buy_shares = Decimal(buy_row.get("shares") or "0")
                    sell_shares = Decimal(sell_row.get("shares") or "0")
                except InvalidOperation:
                    buy_shares = sell_shares = Decimal("0")
                if buy_shares == sell_shares and buy_shares != 0:
                    # Fully offsetting buy/sell pair -> drop both.
                    continue
        # Default: keep all rows for this actor.
        result.extend(actor_rows)

    return result


def run_insider_task(input_dir: Path, output_csv: Path, collect: bool = False) -> int:
    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 1
    txt_files = sorted(input_dir.rglob("*.txt"))
    if not txt_files:
        logging.warning("No .txt files found in %s", input_dir)

    rows: List[dict] = []
    for path in txt_files:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logging.warning("Skipping %s: %s", path, exc)
            continue
        rows.extend(parse_insider_file(content, path))

    if collect:
        rows = _collect_rows(rows)

    try:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        logging.error("Failed to write CSV %s: %s", output_csv, exc)
        return 1
    return 0


def main() -> int:
    if load_dotenv is not None:
        load_dotenv()

    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.task == "insider":
        return run_insider_task(args.input_dir, args.output, collect=args.collect)
    logging.error("Unsupported task: %s", args.task)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
