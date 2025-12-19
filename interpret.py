"""CLI entry point for parsing insider transaction notices into CSV output.

Example usage:
    uv run python interpret.py --task insider --input-dir insider
    uv run python interpret.py --task insider --input-dir insider --output insider_summary.csv

Run the regression test with:
    python -m unittest -q test_interpret_insider_regression.py.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

FIELDNAMES = ["date", "company", "name", "shares", "price", "total_value", "side", "reason", "filename"]
REQUIRED_FIELDS = ["date", "company", "name", "shares", "price", "total_value", "side"]

BUY_KEYWORDS = (
    "acquisition",
    "purchase",
    "buy",
    "receipt",
    "subscription",
    "incentive",
    "acceptance",
    "anskaffelse",
    "kob",
    "kb",
    "erhvervelse",
    "merkinta",
)
SELL_KEYWORDS = (
    "disposal",
    "disposed",
    "sale",
    "sell",
    "sold",
    "luovutus",
    "divest",
    "salg",
    "myynti",
)
NATURE_TRANSLATIONS = {
    # Finnish -> English (style aligned with other rows)
    "luovutus": "DISPOSAL",
    "hankinta": "ACQUISITION",
    "merkinta": "SUBSCRIPTION",
    "osakepalkkion vastaanottaminen": "RECEIPT",
    "osakeoption hyvaksyminen": "ACCEPTANCE OF A STOCK OPTION",
    # Swedish -> English
    "teckning": "SUBSCRIPTION",
}

TRANSACTION_LINE_RE = re.compile(
    r"\(\d+\)\s*:\s*(?:vol(?:ume|yymi|ym)\b[^:]*:\s*([\d\s.,]+)).*?"
    r"(?:unit\s*price|pris per aktie|pris|yksikk\w*hinta|keskihinta|average price|volume weighted average price|hinta|price)\s*:\s*([\d\s.,]+)",
    re.IGNORECASE,
)

FALLBACK_TRANSACTION_RE = re.compile(
    r"volume:\s*([\d\s.,]+)\s*(?:unit\s*price|average price|volume weighted average price|keskihinta|price)\s*[: ]\s*([\d\s.,]+)",
    re.IGNORECASE,
)

NARRATIVE_PURCHASE_RE = re.compile(
    r"purchased\s+([\d][\d.,]*)\s+shares\s+.*?price\s+of\s+dkk\s+([\d.,]+)",
    re.IGNORECASE,
)

MONTH_NAME_TO_NUM = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

DATE_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"transaction date[:\s]+(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    re.compile(r"transaction date[:\s]+(\d{2}[./]\d{2}[./]\d{4})", re.IGNORECASE),
    re.compile(r"liiketoimen[^:]{0,30}:\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
    re.compile(r"liiketoimen[^:]{0,30}:\s*(\d{2}[./]\d{2}[./]\d{4})", re.IGNORECASE),
]

REFERENCE_NUMBER_RE = re.compile(
    r"(?:reference number|viitenumero|referensnummer|referencenummer|referansenummer)\s*:\s*([A-Za-z0-9][A-Za-z0-9_.-]*)",
    re.IGNORECASE,
)


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


def looks_like_alphanumeric_code(value: str) -> bool:
    """Heuristic for LEI/ISIN/reference-like tokens (letters+digits, no separators)."""
    compact = re.sub(r"\s+", "", value)
    if len(compact) < 8:
        return False
    has_letters = re.search(r"[A-Za-z]", compact) is not None
    has_digits = re.search(r"\d", compact) is not None
    if not (has_letters and has_digits):
        return False
    return re.search(r"[.,]", compact) is None


def is_date_like(value: str) -> bool:
    return bool(
        re.search(r"\b\d{4}-\d{2}-\d{2}\b", value)
        or re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", value)
    )


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


def looks_like_person_name(text: str) -> bool:
    words = [w for w in text.replace(",", " ").split() if w]
    if len(words) < 2 or len(words) > 6:
        return False
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if len(alpha_words) < 2:
        return False
    capitalized = sum(1 for w in alpha_words if w[0].isupper())
    return capitalized >= 2


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
    if candidate:
        return candidate

    # Fallback for Danish/Scandinavian templates where the person name appears
    # right after a "Nærmere oplysninger om personen ..." heading.
    for idx, (_raw, norm) in enumerate(lines):
        if "oplysninger om personen" not in norm:
            continue
        for raw2, norm2 in lines[idx + 1 : idx + 8]:
            text = raw2.strip()
            if not text or ":" in text:
                continue
            if any(ch.isdigit() for ch in text):
                continue
            if any(
                token in norm2
                for token in (
                    "navn",
                    "arsag",
                    "stilling",
                    "forste indberetning",
                    "indberetning",
                    "lei",
                )
            ):
                continue
            words = text.split()
            if len(words) < 2 or len(words) > 6:
                continue
            if looks_like_person_name(text):
                return text
            if not candidate:
                candidate = text

    # Fallback for Nordic-style "Nafn/Navn" heading followed by the name.
    for idx, (_raw, norm) in enumerate(lines):
        if "nafn" in norm or "navn" in norm:
            best_candidate = ""
            for raw2, _norm2 in lines[idx + 1 :]:
                text = raw2.strip()
                if not text or ":" in text:
                    continue
                if any(ch.isdigit() for ch in text):
                    continue
                if any(
                    token in _norm2
                    for token in (
                        "arsag",
                        "stilling",
                        "forste indberetning",
                        "indberetning",
                        "lei",
                        "identifikationskode",
                        "transaktionens art",
                        "pris",
                        "maengde",
                        "mngde",
                    )
                ):
                    continue
                if looks_like_person_name(text):
                    best_candidate = text
                    break
                if not best_candidate:
                    best_candidate = text
            if best_candidate:
                return best_candidate

    # Fallback for ESMA-style templates without explicit "Name:" lines.
    person_idx: int | None = None
    for idx, (_raw, norm) in enumerate(lines):
        if "person discharging managerial responsibilities" in norm:
            person_idx = idx
            break
    if person_idx is not None:
        for j in range(max(0, person_idx - 3), person_idx):
            raw_j, _norm_j = lines[j]
            text = raw_j.strip()
            if not text or ":" in text:
                continue
            if any(ch.isdigit() for ch in text):
                continue
            words = text.split()
            if len(words) < 2 or len(words) > 4:
                continue
            candidate = text
            break
    return candidate


def extract_narrative_name_from_text(text: str) -> str:
    """Extract a person name from narrative sentences like '..., Lars Kristensen has on 8 December ...'."""
    m = re.search(r",\s*([^,\n]+?)\s+has on\s+\d", text)
    if not m:
        return ""
    return m.group(1).strip()


def normalize_date_token(date_str: str) -> str:
    date_str = date_str.strip()
    m = re.match(r"(\d{2})[./-](\d{2})[./-](\d{4})", date_str)
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
    fallback_eu = re.search(r"\d{2}[./-]\d{2}[./-]\d{4}", raw_text)
    if fallback_eu:
        return normalize_date_token(fallback_eu.group(0))
    # Fallback: textual dates like "8 December 2025" or "8 December".
    m = re.search(
        r"\b(\d{1,2})\s+("
        r"january|february|march|april|may|june|july|august|september|october|november|december"
        r")\b(?:\s+(\d{4}))?",
        normalized,
    )
    if m:
        day_str, month_name, year = m.groups()
        # If the first occurrence has no year, fall back to the first year
        # mentioned anywhere in the text.
        if not year:
            year_match = re.search(r"\b(\d{4})\b", normalized)
            year = year_match.group(1) if year_match else "0000"
        try:
            day = int(day_str)
        except ValueError:
            return ""
        month = MONTH_NAME_TO_NUM.get(month_name.lower())
        if not month:
            return ""
        return f"{year}-{month}-{day:02d}"
    return ""


def extract_reference_number(raw_text: str) -> str:
    match = REFERENCE_NUMBER_RE.search(raw_text)
    return match.group(1).strip() if match else ""


def extract_nature(lines: Sequence[Tuple[str, str]]) -> str:
    def is_section_label(text: str) -> bool:
        return bool(re.fullmatch(r"[a-z]\)", normalize_line(text)))

    def is_label_line(norm: str) -> bool:
        return (
            ("price(s)" in norm and "volume(s)" not in norm)
            or ("volume(s)" in norm and "price(s)" not in norm)
            or ("pris" in norm and ("maengde" in norm or "mngde" in norm))
            or norm.startswith("mngde")
            or norm.startswith("pris")
            or norm.startswith("prise")
            or "aggregerede oplysninger" in norm
            or "aggregated information" in norm
            or "aggregeret maengde" in norm
            or "aggregeret mngde" in norm
            or "dato for transaktionen" in norm
            or "transaction date" in norm
            or "sted for transaktionen" in norm
            or "place of transaction" in norm
        )

    # Primary: value on the same line as the label.
    for raw, norm in lines:
        if (
            "nature of transaction" in norm
            or "nature of the transaction" in norm
            or "liiketoimen luonne" in norm
            or "transaktionens art" in norm
            or "transaktionens karaktar" in norm
        ):
            value = value_after_colon(raw)
            if value:
                return value
    # Fallback for ESMA-style layouts where the description appears on later lines.
    nature_start: int | None = None
    for idx, (_raw, norm) in enumerate(lines):
        if (
            "nature of transaction" in norm
            or "nature of the transaction" in norm
            or "liiketoimen luonne" in norm
            or "transaktionens art" in norm
            or "transaktionens karaktar" in norm
        ):
            nature_start = idx
            break
    if nature_start is not None:
        candidates: List[str] = []
        for raw, norm in lines[nature_start + 1 :]:
            text = raw.strip()
            if not text or text.endswith(":"):
                continue
            if is_section_label(text):
                continue
            if is_label_line(norm):
                continue
            if looks_like_alphanumeric_code(text):
                continue
            if not re.search(r"[A-Za-z]", text):
                # If we've already seen a descriptive line, stop when hitting the numeric block.
                if candidates and re.search(r"\d", text):
                    break
                continue
            candidates.append(text)
        if candidates:
            # Prefer the last descriptive line before the numeric block.
            return candidates[-1]
        # Some Danish templates list field labels first and values later. In that case,
        # look ahead for the first descriptive line after "Transaktionens art".
        if "transaktionens art" in lines[nature_start][1]:
            for raw, norm in lines[nature_start + 1 : nature_start + 25]:
                if is_label_line(norm):
                    continue
                if any(
                    token in norm
                    for token in (
                        "identifikationskode",
                        "isin",
                        "pris",
                        "maengde",
                        "mngde",
                        "dato for transaktionen",
                        "sted for transaktionen",
                    )
                ):
                    continue
                text = raw.strip()
                if not text or text.endswith(":"):
                    continue
                if is_section_label(text):
                    continue
                if not re.search(r"[A-Za-z]", text):
                    continue
                if re.search(r"\b[A-Z]{2}\d{6,}\b", text) or looks_like_alphanumeric_code(text):
                    continue
                return text

    # Fallback: look for common single-word nature labels anywhere in the text
    # (e.g. Danish "Anskaffelse" in standard forms, or simple "purchased").
    for raw, norm in lines:
        if "anskaffelse" in norm or "purchased" in norm:
            return raw.strip()
    return ""


def translate_nature(nature: str) -> str:
    """Translate known non-English nature values into English; otherwise return as-is."""
    stripped = nature.strip()
    if not stripped:
        return nature
    key = normalize_line(stripped)
    return NATURE_TRANSLATIONS.get(key, nature)


def determine_side(nature: str) -> str:
    lowered = normalize_line(nature)
    if any(keyword in lowered for keyword in SELL_KEYWORDS):
        return "sell"
    if any(keyword in lowered for keyword in BUY_KEYWORDS):
        return "buy"
    return ""


def clean_int(value: str) -> str:
    digits = re.sub(r"[^\d]", "", value)
    return digits


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
    aggregated_transactions: List[Tuple[str, str]] = []
    in_aggregated = False
    for _raw, norm in lines:
        if "yhdistetyt" in norm or "aggregated" in norm or "aggreger" in norm:
            in_aggregated = True
            continue
        match = TRANSACTION_LINE_RE.search(norm)
        if match:
            target = aggregated_transactions if in_aggregated else transactions
            target.append((match.group(1), match.group(2)))
    if transactions:
        return transactions
    if aggregated_transactions:
        return aggregated_transactions
    filtered_lines: List[str] = []
    skip_next = 0
    for _, norm in lines:
        if "yhdistetyt" in norm or "aggregated" in norm or "aggreger" in norm:
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
    if transactions:
        return transactions

    # Fallback 2: ESMA-style layouts with separate "Price(s)" and "Volume(s)" headings,
    # where the numeric values appear on following lines.
    price_indices: List[int] = []
    volume_indices: List[int] = []
    for idx, (_raw, norm) in enumerate(lines):
        if "price(s)" in norm and "volume(s)" not in norm:
            price_indices.append(idx)
        if "volume(s)" in norm and "price(s)" not in norm:
            volume_indices.append(idx)

    price_raw: str | None = None
    volume_raw: str | None = None
    if price_indices:
        for idx in price_indices:
            for raw, _norm in lines[idx + 1 :]:
                if re.search(r"\d", raw):
                    price_raw = raw
                    break
            if price_raw:
                break

    if volume_indices:
        for idx in volume_indices:
            for raw, _norm in lines[idx + 1 :]:
                if re.search(r"\d", raw):
                    volume_raw = raw
                    break
            if volume_raw:
                break

    if price_raw and volume_raw:
        transactions.append((volume_raw, price_raw))
        return transactions

    # Fallback 3: Icelandic-style layout with "Verð og magn" and separate
    # "Verð" / "Samanlagt magn" or "Magn" labels.
    has_ver_og_magn = any("verd og magn" in norm for _raw, norm in lines)
    if has_ver_og_magn:
        price_header_idx: int | None = None
        for idx, (_raw, norm) in enumerate(lines):
            if norm == "verd":
                price_header_idx = idx
                break

        price_raw = None
        if price_header_idx is not None:
            for raw, _norm in lines[price_header_idx + 1 :]:
                if re.search(r"\d", raw):
                    price_raw = raw
                    break

        volume_header_idx: int | None = None
        for idx, (_raw, norm) in enumerate(lines):
            if "samanlagt magn" in norm:
                volume_header_idx = idx
            elif norm == "magn" and volume_header_idx is None:
                volume_header_idx = idx

        volume_raw = None
        if volume_header_idx is not None:
            for raw, _norm in lines[volume_header_idx + 1 :]:
                if re.search(r"\d", raw):
                    volume_raw = raw
                    break

        if price_raw and volume_raw:
            transactions.append((volume_raw, price_raw))

    # Fallback 4: Danish-style layout with "Pris(er)" and "Mængde(r)" headings.
    has_pris = any("pris" in norm for _raw, norm in lines)
    has_maengde = any(("maengde" in norm) or ("mngde" in norm) for _raw, norm in lines)
    if has_pris and has_maengde:
        def should_skip_numeric_line(raw: str) -> bool:
            return looks_like_alphanumeric_code(raw) or is_date_like(raw)

        def first_numeric_after(idx: int | None) -> str | None:
            if idx is None:
                return None
            for raw, _norm in lines[idx + 1 :]:
                if re.search(r"\d", raw):
                    if should_skip_numeric_line(raw):
                        continue
                    return raw
            return None

        def split_price_volume_from_line(line: str) -> Tuple[str, str] | None:
            number_re = r"-?(?:\d{1,3}(?:[.,]\d{3}|\s+\d{3})+|\d+)(?:[.,]\d+)?"
            date_stripped = re.sub(
                r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b",
                " ",
                line,
            )
            # Strip date-like tokens; if extra numbers remain, take the last two.
            tokens = re.findall(number_re, date_stripped)
            if len(tokens) < 2:
                return None
            if len(tokens) > 2:
                tokens = tokens[-2:]

            def has_fractional(token: str) -> bool:
                compact = token.replace(" ", "")
                sep_pos = max(compact.rfind("."), compact.rfind(","))
                if sep_pos == -1:
                    return False
                frac = compact[sep_pos + 1 :]
                if not frac.isdigit():
                    return False
                # Treat 3-digit tails as thousands separators, not decimals.
                if len(frac) == 3:
                    return False
                return len(frac) > 0

            first, second = tokens[0], tokens[1]
            first_frac = has_fractional(first)
            second_frac = has_fractional(second)
            if first_frac and not second_frac:
                return first, second
            if second_frac and not first_frac:
                return second, first
            return first, second

        combined_idx: int | None = None
        price_header_idx: int | None = None
        volume_header_idx: int | None = None
        for idx, (_raw, norm) in enumerate(lines):
            has_price = ("pris" in norm) or ("prise" in norm)
            has_volume = ("maengde" in norm) or ("mngde" in norm)
            if combined_idx is None and has_price and has_volume:
                combined_idx = idx
            if price_header_idx is None and (norm.startswith("pris") or norm.startswith("prise")) and not has_volume:
                price_header_idx = idx
            if volume_header_idx is None and (norm.startswith("maengde") or norm.startswith("mngde")) and not has_price:
                volume_header_idx = idx

        # Common layout: "Pris(er)" and "Mængde(r)" headings followed by the
        # numeric values on subsequent lines (price first, then volume).
        if price_header_idx is not None and volume_header_idx is not None:
            numeric_lines: List[str] = []
            scan_start = max(price_header_idx, volume_header_idx) + 1
            for raw, norm in lines[scan_start:]:
                if re.search(r"\d", raw):
                    if should_skip_numeric_line(raw):
                        continue
                    split = split_price_volume_from_line(raw)
                    if split:
                        price_raw, volume_raw = split
                        transactions.append((volume_raw, price_raw))
                        return transactions
                    numeric_lines.append(raw)
                if len(numeric_lines) >= 2:
                    break
            if len(numeric_lines) >= 2:
                price_raw = numeric_lines[0]
                volume_raw = numeric_lines[1]
                transactions.append((volume_raw, price_raw))
                return transactions
            if len(numeric_lines) == 1:
                split = split_price_volume_from_line(numeric_lines[0])
                if split:
                    price_raw, volume_raw = split
                    transactions.append((volume_raw, price_raw))
                    return transactions

        price_raw = first_numeric_after(price_header_idx) if price_header_idx is not None else None
        volume_raw = first_numeric_after(volume_header_idx) if volume_header_idx is not None else None

        if price_raw and volume_raw:
            transactions.append((volume_raw, price_raw))
            return transactions

        # Some Danish templates only have a combined "Pris(er) og mængde(r)" heading.
        if combined_idx is not None:
            numeric_lines: List[str] = []
            for raw, norm in lines[combined_idx + 1 :]:
                if re.search(r"\d", raw):
                    if should_skip_numeric_line(raw):
                        continue
                    numeric_lines.append(raw)
                if len(numeric_lines) >= 2:
                    break
            if len(numeric_lines) >= 2:
                price_raw = numeric_lines[0]
                volume_raw = numeric_lines[1]
                transactions.append((volume_raw, price_raw))
                return transactions
            if len(numeric_lines) == 1:
                split = split_price_volume_from_line(numeric_lines[0])
                if split:
                    price_raw, volume_raw = split
                    transactions.append((volume_raw, price_raw))
                    return transactions

    if transactions:
        return transactions

    # Fallback 5: narrative summaries such as
    # "purchased 2,141,911 shares ... at an average price of DKK 1.60 per share".
    m = NARRATIVE_PURCHASE_RE.search(normalized_text)
    if m:
        volume, price = m.groups()
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


def determine_side_from_text(nature: str, normalized_text: str) -> str:
    """Determine buy/sell side from nature field, falling back to full text."""
    side = determine_side(nature)
    if side:
        return side
    # Fallback: inspect full normalized text for cues.
    lowered = normalized_text
    if "purchased" in lowered:
        return "buy"
    if "disposed" in lowered or "sold" in lowered:
        return "sell"
    return determine_side(lowered)


def parse_insider_file(text: str, path: Path) -> List[dict]:
    lines = build_lines(text)
    normalized_text = normalize_text_for_search(text)
    company = extract_company(lines, path)
    name = extract_name(lines)
    if not name:
        # Narrative-style notices may not follow the ESMA tabular layout
        # and only mention the person in free text.
        name = extract_narrative_name_from_text(text)
    date = extract_date(text)
    reference_number = extract_reference_number(text)
    nature = translate_nature(extract_nature(lines))
    side = determine_side_from_text(nature, normalized_text)
    transactions = extract_transactions(lines, normalized_text)

    rows: List[dict] = []
    if not transactions:
        if (
            ("henvises til vedh" in normalized_text and "skema" in normalized_text)
            or "see attached form" in normalized_text
            or "see attached schedule" in normalized_text
        ):
            logging.warning("Skipping %s: no transaction details (attached form).", path.name)
            return []
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
                "filename": path.name,
                "_reference_number": reference_number,
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
                "filename": path.name,
                "_reference_number": reference_number,
            }
        )

    # Warn if any of the required fields could not be determined,
    # but still keep the rows in the output.
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
        if missing:
            logging.warning(
                "Incomplete row from %s: missing %s", path.name, ", ".join(missing)
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
        "--input",
        dest="input_dir",
        type=Path,
        default=Path("."),
        help="Directory containing insider .txt files (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the CSV file to write (omit to print to stdout).",
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


def _merge_sparse_dict(base: dict, other: dict) -> dict:
    merged = dict(base)
    for key, value in other.items():
        if value and not merged.get(key):
            merged[key] = value
    return merged


def _dedupe_rows(rows: List[dict]) -> List[dict]:
    """Deduplicate multi-language versions of the same notice.

    Nasdaq notices often exist in multiple languages in the input directory.
    Use the reference number (when available) to collapse duplicates so the
    summary doesn't double-count the same transaction.
    """

    def signature(row: dict) -> tuple[str, str, str] | None:
        shares = (row.get("shares") or "").strip()
        price = (row.get("price") or "").strip()
        total = (row.get("total_value") or "").strip()
        if not any((shares, price, total)):
            return None
        return (shares, price, total)

    deduped: List[dict | None] = []
    key_to_index: dict[tuple, int] = {}
    placeholder_indices_by_ref: dict[str, List[int]] = {}
    refs_with_transactions: set[str] = set()
    ref_to_tx_index: dict[str, int] = {}

    for row in rows:
        ref = (row.get("_reference_number") or "").strip()
        sig = signature(row)

        if ref:
            if sig is None:
                existing_tx_idx = ref_to_tx_index.get(ref)
                if existing_tx_idx is not None:
                    existing_tx = deduped[existing_tx_idx]
                    deduped[existing_tx_idx] = _merge_sparse_dict(existing_tx or {}, row)
                    continue
                existing_idxs = placeholder_indices_by_ref.get(ref, [])
                if existing_idxs:
                    idx = existing_idxs[0]
                    existing = deduped[idx]
                    deduped[idx] = _merge_sparse_dict(existing or {}, row)
                    continue
                idx = len(deduped)
                deduped.append(dict(row))
                placeholder_indices_by_ref.setdefault(ref, []).append(idx)
                continue
            placeholders = placeholder_indices_by_ref.get(ref, [])
            if placeholders:
                merged_row = dict(row)
                for idx in placeholders:
                    placeholder = deduped[idx]
                    if placeholder is not None:
                        merged_row = _merge_sparse_dict(merged_row, placeholder)
                row = merged_row
            key = ("ref", ref, *sig)
        else:
            key = (
                "content",
                row.get("date", ""),
                row.get("company", ""),
                row.get("name", ""),
                row.get("shares", ""),
                row.get("price", ""),
                row.get("total_value", ""),
            )

        existing_idx = key_to_index.get(key)
        if existing_idx is None:
            idx = len(deduped)
            deduped.append(dict(row))
            key_to_index[key] = idx
        else:
            existing = deduped[existing_idx]
            if existing is not None:
                deduped[existing_idx] = _merge_sparse_dict(existing, row)

        if ref and sig is not None:
            refs_with_transactions.add(ref)
            ref_to_tx_index[ref] = key_to_index.get(key, len(deduped) - 1)
            for idx in placeholder_indices_by_ref.get(ref, []):
                deduped[idx] = None
            placeholder_indices_by_ref[ref] = []

    return [row for row in deduped if row is not None]


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


def _resolve_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.txt"))


def run_insider_task(input_dir: Path, output_csv: Path | None, collect: bool = False) -> int:
    if not input_dir.exists():
        logging.error("Input directory does not exist: %s", input_dir)
        return 1
    txt_files = _resolve_input_files(input_dir)
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

    rows = _dedupe_rows(rows)

    if collect:
        rows = _collect_rows(rows)

    if output_csv is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    else:
        try:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with output_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
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
