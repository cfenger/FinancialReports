#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nasdaq Europe News CLI – generel søgning som Nasdaq UI (Category/Company/Free text/Market/Date).

Data-kilde:
- query.action (JSONP): https://api.news.eu.nasdaq.com/news/query.action

Støtte for UI dropdowns:
- Kategorier og selskaber kan:
  A) forsøges hentet automatisk fra mulige metadata endpoints, ELLER
  B) importeres fra JSONP-tekst (getCategoriesCallback(...) / getCompaniesCallback(...))
     som du kan gemme i en fil og pege på med CLI-flag.


https://github.com/cfenger/FinancialReports
https://chatgpt.com/g/g-p-693e6f04c1a481918bdcc5ab4d6c8176-flsmidth/c/693edeec-4694-8330-b760-d0d619159fb5

uv run --with requests python nasdaq_news_cli.py --help



#Bygger JSONP-tekster i filer: categories.jsonp (getCategoriesCallback({...})) og companies.jsonp (getCompaniesCallback({...}))
uv run --with requests python nasdaq_news_cli.py --import-categories-jsonp categories.jsonp --import-companies-jsonp companies.jsonp --interactive --out-csv results.csv


uv run --with requests python nasdaq_news_cli.py --import-companies-jsonp companies.jsonp --import-categories-jsonp categories.jsonp --list-companies


uv run --with requests python nasdaq_news_cli.py --import-companies-jsonp companies.jsonp --list-companies


# Interactive
uv run --with requests python nasdaq_news_cli.py --interactive --out-csv results.csv

# Non-interactive
uv run --with requests python nasdaq_news_cli.py --free-text "FLSmidth" --category "Ledende medarbejderes transaktioner" --company "FLSmidth & Co. A/S" --limit 150 --out-csv fls_transactions.csv

#Other output formats: --out-ndjson  --out-json

uv run --with requests python nasdaq_news_cli.py --market "Main Market, Copenhagen" --category "Forløb af generalforsamling" --from-date 2025-12-04 --out-csv results2.csv

uv run --with requests python nasdaq_news_cli.py --market "Main Market, Copenhagen" --category "Forløb af generalforsamling" --from-date 2015-12-04 --out-csv results2.csv --interactive --company "Novo Nordisk A/S"

uv run --with requests python nasdaq_news_cli.py --list-markets


uv run --with requests python nasdaq_news_cli.py --category "Ledende medarbejderes transaktioner"

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import fnmatch
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_BASE = "https://api.news.eu.nasdaq.com"
QUERY_PATH = "/news/query.action"

# Vi kender sikkert query.action; metadata endpoints kan variere.
# Derfor prøver vi flere. Hvis ingen virker, bruger vi import eller fallback-cache.
CANDIDATE_META_PATHS = [
    "/news/metadata.action",
    "/news/getCategories.action",
    "/news/getCompanies.action",
    "/news/categories.action",
    "/news/companies.action",
]

MARKET_CHOICES: List[str] = [
    "All",
    "Main Market, Copenhagen",
    "Main Market, Stockholm",
    "Main Market, Helsinki",
    "Main Market, Iceland",
    "First North Sweden",
    "First North Finland",
    "First North Denmark",
    "First North Iceland",
]

CACHE_DIR = Path(".")
EXCLUDED_COMPANIES_FILE = Path("excludedCompanies.txt")


def _load_excluded_company_patterns(path: Path) -> List[re.Pattern[str]]:
    """
    Load shell-style patterns (e.g. 'Nykredit*') from a text file and
    compile them to regular expressions for matching company names.

    - Empty lines and lines starting with '#' are ignored.
    - Matching is case-insensitive.
    """
    if not path.exists():
        return []
    patterns: List[re.Pattern[str]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        regex = fnmatch.translate(stripped)
        patterns.append(re.compile(regex, re.IGNORECASE))
    return patterns


def _filter_excluded_companies(companies: List[str], patterns: List[re.Pattern[str]]) -> List[str]:
    """Filter out companies that match any of the exclusion patterns."""
    if not patterns:
        return companies

    def is_excluded(name: str) -> bool:
        return any(pat.match(name) for pat in patterns)

    return [c for c in companies if not is_excluded(c)]


def _filter_excluded_items(items: List[Dict[str, Any]], patterns: List[re.Pattern[str]]) -> List[Dict[str, Any]]:
    """Filter out result items whose company matches any exclusion pattern."""
    if not patterns:
        return items

    def is_excluded_item(it: Dict[str, Any]) -> bool:
        name = (it.get("company") or "").strip()
        return any(pat.match(name) for pat in patterns)

    return [it for it in items if not is_excluded_item(it)]


@dataclass
class NasdaqQuery:
    free_text: str = ""
    cns_category: List[str] = None
    not_cns_category: List[str] = None
    company: str = ""
    market: str = ""
    from_date: str = ""  # "YYYY-MM-DD" (som UI typisk bruger)
    to_date: str = ""    # "YYYY-MM-DD"

    # UI-lignende defaults (fra dine DevTools)
    global_group: str = "exchangeNotice"
    global_name: str = "NordicAllMarkets"
    display_language: str = "da"
    time_zone: str = "CET"
    date_mask: str = "yyyy-MM-dd HH:mm:ss"
    dir: str = "DESC"  # "DESC" / "ASC"

    limit: int = 50
    start: int = 0


class NasdaqNewsClient:
    def __init__(self, base_url: str = DEFAULT_BASE, session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.s = session or requests.Session()
        self.s.headers.update({
            "User-Agent": "nasdaq-news-cli/1.0 (+python requests)",
            "Accept": "*/*",
        })

    # ---------------------------
    # JSONP helpers
    # ---------------------------
    @staticmethod
    def parse_jsonp(text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parser generisk JSONP:
          callbackName({...})
        Returnerer (callbackName, json-object)
        """
        m = re.match(r"^\s*([A-Za-z_$][\w$]*)\((.*)\)\s*;?\s*$", text, flags=re.DOTALL)
        if not m:
            raise ValueError("Input ligner ikke JSONP (callback(...)).")
        cb = m.group(1)
        payload = json.loads(m.group(2))
        return cb, payload

    @staticmethod
    def parse_jsonp_expected_callback(text: str, callback: str) -> Dict[str, Any]:
        """
        Parser JSONP med forventet callback navn, fx cb(...)
        """
        m = re.match(rf"^\s*{re.escape(callback)}\((.*)\)\s*;?\s*$", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Response ligner ikke JSONP med callback '{callback}'.")
        return json.loads(m.group(1))

    def get_jsonp(self, path: str, params: Dict[str, str], callback: str) -> Dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        params = dict(params)
        params["callback"] = callback
        r = self.s.get(url, params=params, timeout=30)
        r.raise_for_status()
        return self.parse_jsonp_expected_callback(r.text, callback)

    # ---------------------------
    # Facts extraction: { "facts": [ {"id": "..."} ], "type": "..." }
    # ---------------------------
    @staticmethod
    def extract_facts_ids(payload: Any, expected_type: Optional[str] = None) -> List[str]:
        out: List[str] = []

        def walk(obj: Any) -> None:
            if isinstance(obj, dict):
                typ = obj.get("type")
                facts = obj.get("facts")
                if isinstance(facts, list):
                    ok = True
                    if expected_type is not None:
                        ok = isinstance(typ, str) and typ.lower() == expected_type.lower()
                    if ok:
                        for f in facts:
                            if isinstance(f, dict) and isinstance(f.get("id"), str):
                                out.append(f["id"])
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)

        walk(payload)

        # unique, stable
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    # ---------------------------
    # Cache / load lists
    # ---------------------------
    @staticmethod
    def load_list_cache(path: Path) -> Optional[List[str]]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            return None
        return None

    @staticmethod
    def save_list_cache(path: Path, items: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def auto_fetch_facts_list(self, expected_type: str, display_language: str = "da") -> List[str]:
        """
        Forsøg at hente en facts-liste (company/cnscategory) automatisk ved at prøve
        forskellige metadata endpoints og lede efter facts i responsen.

        Returnerer [] hvis intet lykkes.
        """
        params = {
            "countResults": "true",
            "globalGroup": "exchangeNotice",
            "displayLanguage": display_language,
            "timeZone": "CET",
            "dateMask": "yyyy-MM-dd HH:mm:ss",
            "globalName": "NordicAllMarkets",
        }

        for p in CANDIDATE_META_PATHS:
            # Nogle endpoints bruger cb, andre getCompaniesCallback/getCategoriesCallback.
            for cb in ("cb", "getCompaniesCallback", "getCategoriesCallback"):
                try:
                    payload = self.get_jsonp(p, params=params, callback=cb)
                    items = self.extract_facts_ids(payload, expected_type=expected_type)
                    if items:
                        return items
                except Exception:
                    continue
        return []

    def get_categories(self, cache_path: Path, display_language: str = "da",
                       refresh: bool = False, import_jsonp_path: Optional[Path] = None) -> List[str]:
        """
        Returnerer liste af kategorier (strings).
        Prioritet:
          1) import_jsonp_path (hvis angivet)
          2) cache (hvis findes og ikke refresh)
          3) auto-fetch
          4) cache (sidst kendte) hvis eksisterer
          5) []
        """
        if import_jsonp_path:
            text = import_jsonp_path.read_text(encoding="utf-8")
            _, payload = self.parse_jsonp(text)
            cats = self.extract_facts_ids(payload, expected_type="cnscategory")
            if not cats:
                # nogle gange kan type mangle; prøv uden expected_type
                cats = self.extract_facts_ids(payload, expected_type=None)
            if cats:
                self.save_list_cache(cache_path, cats)
                return cats

        if not refresh:
            cached = self.load_list_cache(cache_path)
            if cached:
                return cached

        cats = self.auto_fetch_facts_list("cnscategory", display_language=display_language)
        if cats:
            self.save_list_cache(cache_path, cats)
            return cats

        cached = self.load_list_cache(cache_path)
        return cached or []

    def get_companies(self, cache_path: Path, display_language: str = "da",
                      refresh: bool = False, import_jsonp_path: Optional[Path] = None) -> List[str]:
        """
        Returnerer liste af selskabsnavne (strings).
        Samme prioritet som get_categories.
        """
        if import_jsonp_path:
            text = import_jsonp_path.read_text(encoding="utf-8")
            _, payload = self.parse_jsonp(text)
            comps = self.extract_facts_ids(payload, expected_type="company")
            if not comps:
                comps = self.extract_facts_ids(payload, expected_type=None)
            if comps:
                self.save_list_cache(cache_path, comps)
                return comps

        if not refresh:
            cached = self.load_list_cache(cache_path)
            if cached:
                return cached

        comps = self.auto_fetch_facts_list("company", display_language=display_language)
        if comps:
            self.save_list_cache(cache_path, comps)
            return comps

        cached = self.load_list_cache(cache_path)
        return cached or []

    # ---------------------------
    # query.action
    # ---------------------------
    def query(self, q: NasdaqQuery) -> Dict[str, Any]:
        """
        Kald query.action (JSONP). cnsCategory/notCnsCategory sendes som kommasepareret streng.
        Hvis Nasdaq i din UI sender flere værdier på anden måde, kan denne del justeres.
        """
        url = f"{self.base_url}{QUERY_PATH}"
        cns_cat = ",".join(q.cns_category or [])
        not_cat = ",".join(q.not_cns_category or [])

        params: Dict[str, str] = {
            "countResults": "true",
            "globalGroup": q.global_group,
            "displayLanguage": q.display_language,
            "timeZone": q.time_zone,
            "dateMask": q.date_mask,
            "limit": str(q.limit),
            "start": str(q.start),
            "dir": q.dir,
            "globalName": q.global_name,
            "freeText": q.free_text or "",
            "cnsCategory": cns_cat,
            "notCnsCategory": not_cat,
            "market": q.market or "",
            "company": q.company or "",
            "fromDate": q.from_date or "",
            "toDate": q.to_date or "",
            "callback": "cb",
        }

        r = self.s.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = self.parse_jsonp_expected_callback(r.text, "cb")
        return payload

    @staticmethod
    def iter_items(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        results = payload.get("results") or {}
        items = results.get("item") or []
        if isinstance(items, list):
            return items
        if isinstance(items, dict):
            return [items]
        return []


# ---------------------------
# Interactive helpers
# ---------------------------
def interactive_pick_from_list(items: List[str], title: str, allow_multi: bool = True) -> List[str]:
    """
    Interaktiv listevælger (for kategorier). For meget store lister (companies) anbefales search-pick.
    """
    print(title)
    for i, v in enumerate(items, 1):
        print(f"{i:3d}. {v}")
    print()
    raw = input("Vælg nummer" + ("(e) (fx 5,12)" if allow_multi else "") + " eller tom for ingen: ").strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")] if allow_multi else [raw.strip()]
    picked = []
    for p in parts:
        if not p.isdigit():
            raise SystemExit(f"Ugyldigt input: {p}")
        n = int(p)
        if n < 1 or n > len(items):
            raise SystemExit(f"Ugyldigt nummer: {n}")
        picked.append(items[n - 1])
    return picked


def interactive_search_pick(items: List[str], title: str, max_show: int = 50) -> str:
    """
    Interaktiv "autocomplete" for meget store lister (companies).
    """
    while True:
        term = input(f"{title} ").strip()
        if not term:
            return ""
        hits = [x for x in items if term.lower() in x.lower()]
        # show all matches (no truncation)
        # hits = hits[:max_show]
        if not hits:
            print("Ingen matches. Prøv igen.\n")
            continue
        for i, h in enumerate(hits, 1):
            print(f"{i:2d}. {h}")
        raw = input("Vælg nummer: ").strip()
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(hits):
                return hits[n - 1]
        print("Ugyldigt valg. Prøv igen.\n")


# ---------------------------
# Output helpers
# ---------------------------


def interactive_pick_market(default_market: str = "") -> str:
    """
    Simpel interaktiv vメlger for market-feltet.
    Returnerer den valgte market-streng eller "" for "All".
    """
    print("Vメlg market (tom for 'All'):\n")
    for i, m in enumerate(MARKET_CHOICES, 1):
        print(f"{i:2d}. {m}")
    if default_market:
        print(f"\nNuvメrende/forvalgt market: {default_market!r}")
    raw = input("Market-nummer (Enter = All): ").strip()
    if not raw:
        return ""  # All / intet filter
    if not raw.isdigit():
        raise SystemExit(f"Ugyldigt input: {raw}")
    n = int(raw)
    if n < 1 or n > len(MARKET_CHOICES):
        raise SystemExit(f"Ugyldigt nummer: {n}")
    choice = MARKET_CHOICES[n - 1]
    return "" if choice == "All" else choice


def write_csv(items: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "published",
        "releaseTime",
        "headline",
        "company",
        "market",
        "cnsCategory",
        "categoryId",
        "cnsTypeId",
        "messageUrl",
        "disclosureId",
        "attachmentCount",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in items:
            row = {k: it.get(k, "") for k in fields}
            atts = it.get("attachment") or []
            row["attachmentCount"] = len(atts) if isinstance(atts, list) else (1 if atts else 0)
            w.writerow(row)


def _parse_cli_date(value: str) -> datetime:
    """
    Parse --from-date / --to-date values.
    Accepts 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
    """
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise SystemExit(
        f"Could not parse date '{value}'. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'."
    )


def _parse_item_datetime(value: str) -> Optional[datetime]:
    """
    Parse 'published' / 'releaseTime' from API payload.
    Typically 'YYYY-MM-DD HH:MM:SS'.
    """
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Nasdaq Europe News – generel søgning som Nasdaq UI (JSONP).")

    ap.add_argument("--base-url", default=DEFAULT_BASE)

    # UI-lignende filtre
    ap.add_argument("--free-text", default="", help="Fritekst (freeText)")
    ap.add_argument("--category", action="append", default=[], help="Inkluder CNS-kategori (kan gentages)")
    ap.add_argument("--not-category", action="append", default=[], help="Ekskluder CNS-kategori (kan gentages)")
    ap.add_argument("--company", default="", help="Company filter (præcis tekst)")
    ap.add_argument("--market", default="", help="Market filter")
    ap.add_argument("--from-date", default="", help="Fra dato YYYY-MM-DD")
    ap.add_argument("--to-date", default="", help="Til dato YYYY-MM-DD")

    # Query parametre
    ap.add_argument("--language", default="da", help="displayLanguage (da/en/sv/fi/...)")
    ap.add_argument("--dir", default="DESC", choices=["DESC", "ASC"])
    ap.add_argument("--limit", type=int, default=50, help="Maks antal hits at hente (total)")
    ap.add_argument("--sleep", type=float, default=0.2)

    # Dropdown data: cache + import
    ap.add_argument("--refresh-lists", action="store_true", help="Tving genhentning af categories/companies")
    ap.add_argument("--import-categories-jsonp", default="", help="Sti til fil med getCategoriesCallback(...) JSONP")
    ap.add_argument("--import-companies-jsonp", default="", help="Sti til fil med getCompaniesCallback(...) JSONP")
    ap.add_argument("--list-categories", action="store_true")
    ap.add_argument("--list-companies", action="store_true")
    ap.add_argument("--list-markets", action="store_true")

    # Interaktiv mode
    ap.add_argument("--interactive", action="store_true", help="Interaktivt valg af category og company")

    # Output
    ap.add_argument("--out-json", default="", help="Gem resultater til JSON")
    ap.add_argument("--out-csv", default="", help="Gem resultater til CSV")

    args = ap.parse_args()

    client = NasdaqNewsClient(base_url=args.base_url)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cat_cache = CACHE_DIR / f"categories_{args.language}.json"
    comp_cache = CACHE_DIR / f"companies_{args.language}.json"

    cat_import = Path(args.import_categories_jsonp) if args.import_categories_jsonp else None
    comp_import = Path(args.import_companies_jsonp) if args.import_companies_jsonp else None

    categories = client.get_categories(
        cache_path=cat_cache,
        display_language=args.language,
        refresh=args.refresh_lists,
        import_jsonp_path=cat_import if cat_import and cat_import.exists() else None,
    )

    companies = client.get_companies(
        cache_path=comp_cache,
        display_language=args.language,
        refresh=args.refresh_lists,
        import_jsonp_path=comp_import if comp_import and comp_import.exists() else None,
    )
    excluded_patterns = _load_excluded_company_patterns(EXCLUDED_COMPANIES_FILE)
    companies = _filter_excluded_companies(companies, excluded_patterns)

    if args.list_categories:
        for c in categories:
            print(c)
        return 0

    if args.list_companies:
        for c in companies:
            print(c)
        return 0

    if args.list_markets:
        for m in MARKET_CHOICES:
            print(m)
        return 0

    chosen_categories = list(args.category)
    chosen_not = list(args.not_category)
    chosen_company = (args.company or "").strip()

    if args.interactive:
        if not chosen_categories and not chosen_not:
            if categories:
                chosen_categories = interactive_pick_from_list(categories, "Vælg CNS kategori (inkluder):", allow_multi=True)
                chosen_not = interactive_pick_from_list(categories, "Vælg CNS kategori (ekskluder):", allow_multi=True)
            else:
                print("Ingen kategoriliste tilgængelig (cache/import/auto-fetch fejlede).")

        if not chosen_company:
            if companies:
                chosen_company = interactive_search_pick(companies, "Vælg company (autocomplete):", max_show=50)
            else:
                print("Ingen companyliste tilgængelig (cache/import/auto-fetch fejlede).")

    if chosen_company and companies and chosen_company not in companies:
        print(f"WARNING: {chosen_company} is not a valid company name.", file=sys.stderr)

    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")

    max_results = args.limit
    page_size = min(200, max_results)

    q = NasdaqQuery(
        free_text=args.free_text,
        cns_category=chosen_categories,
        not_cns_category=chosen_not,
        company=chosen_company,
        market=args.market,
        from_date=args.from_date,
        to_date=args.to_date,
        display_language=args.language,
        dir=args.dir,
        limit=page_size,
        start=0,
    )

    from_dt = _parse_cli_date(args.from_date) if args.from_date else None
    to_dt = _parse_cli_date(args.to_date) if args.to_date else None

    filtered_items: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    start = 0
    while len(filtered_items) < max_results:
        before_len = len(filtered_items)
        remaining = max_results - len(filtered_items)
        q.limit = min(page_size, remaining)
        q.start = start

        payload = client.query(q)
        items = client.iter_items(payload)
        if not items:
            break

        unique_in_page = 0
        min_dt_in_page = None
        for it in items:
            key = str(it.get("disclosureId") or it.get("messageUrl") or "")
            if not key:
                key = f"{it.get('published','')}|{it.get('company','')}|{it.get('headline','')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            unique_in_page += 1

            parsed_dt = None
            if from_dt or to_dt:
                parsed_dt = _parse_item_datetime(it.get("published") or it.get("releaseTime") or "")
                if parsed_dt is None:
                    continue
                if min_dt_in_page is None or parsed_dt < min_dt_in_page:
                    min_dt_in_page = parsed_dt
                if from_dt and parsed_dt < from_dt:
                    continue
                if to_dt and parsed_dt > to_dt:
                    continue

            if excluded_patterns:
                name = (it.get("company") or "").strip()
                if any(pat.match(name) for pat in excluded_patterns):
                    continue

            filtered_items.append(it)
            if len(filtered_items) >= max_results:
                break

        start += len(items)
        if unique_in_page == 0:
            break
        if (
            from_dt
            and args.dir.upper() == "DESC"
            and min_dt_in_page is not None
            and min_dt_in_page < from_dt
            and len(filtered_items) == before_len
        ):
            # Items are strictly older than from-date; descending order means we are past the window.
            break
        if len(filtered_items) < max_results:
            time.sleep(args.sleep)

    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(filtered_items, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_csv:
        write_csv(filtered_items, Path(args.out_csv))

    print(f"Hits: {len(filtered_items)}")
    for it in filtered_items:
        print(f"- {it.get('published','')}\t{it.get('company','')}\t{it.get('headline','')}\t{it.get('messageUrl','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
