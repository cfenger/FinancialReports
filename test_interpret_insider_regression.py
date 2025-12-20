"""Regression test for the insider parsing CLI."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from interpret import (
    _dedupe_rows,
    _should_skip_stock_option_notice,
    build_lines,
    determine_side_from_text,
    extract_narrative_name_from_text,
    extract_nature,
    extract_transactions,
    normalize_text_for_search,
    parse_insider_file,
)

REPO_ROOT = Path(__file__).resolve().parent


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class InterpretInsiderRegressionTest(unittest.TestCase):
    def test_merge_placeholder_into_transactional_ref(self):
        rows = [
            {
                "_reference_number": "REF2",
                "date": "2025-01-02",
                "company": "Beta Oyj",
                "name": "Person B",
                "shares": "",
                "price": "",
                "total_value": "",
                "side": "",
                "reason": "",
            },
            {
                "_reference_number": "REF2",
                "date": "2025-01-02",
                "company": "",
                "name": "",
                "shares": "1500",
                "price": "10.5",
                "total_value": "15750",
                "side": "buy",
                "reason": "ACQUISITION",
            },
        ]
        deduped = _dedupe_rows(rows)
        self.assertEqual(len(deduped), 1, "Reference dedupe should keep a single merged row")
        merged = deduped[0]
        self.assertEqual(merged["company"], "Beta Oyj")
        self.assertEqual(merged["name"], "Person B")
        self.assertEqual(merged["shares"], "1500")
        self.assertEqual(merged["price"], "10.5")
        self.assertEqual(merged["total_value"], "15750")
        self.assertEqual(merged["side"], "buy")
        self.assertEqual(merged["reason"], "ACQUISITION")

    def test_merge_placeholder_after_transactional_ref(self):
        rows = [
            {
                "_reference_number": "REF3",
                "date": "2025-01-03",
                "company": "",
                "name": "",
                "shares": "500",
                "price": "12.5",
                "total_value": "6250",
                "side": "sell",
                "reason": "DISPOSAL",
            },
            {
                "_reference_number": "REF3",
                "date": "2025-01-03",
                "company": "Gamma",
                "name": "Person C",
                "shares": "",
                "price": "",
                "total_value": "",
                "side": "",
                "reason": "",
            },
        ]
        deduped = _dedupe_rows(rows)
        self.assertEqual(len(deduped), 1, "Reference dedupe should merge placeholders arriving later")
        merged = deduped[0]
        self.assertEqual(merged["company"], "Gamma")
        self.assertEqual(merged["name"], "Person C")
        self.assertEqual(merged["shares"], "500")
        self.assertEqual(merged["price"], "12.5")
        self.assertEqual(merged["total_value"], "6250")
        self.assertEqual(merged["side"], "sell")
        self.assertEqual(merged["reason"], "DISPOSAL")

    def test_dedupe_reference_only_rows(self):
        rows = [
            {
                "_reference_number": "REF1",
                "date": "2025-01-01",
                "company": "Alpha",
                "name": "Person A",
                "shares": "",
                "price": "",
                "total_value": "",
                "side": "",
                "reason": "",
            },
            {
                "_reference_number": "REF1",
                "date": "",
                "company": "Alpha Oy",
                "name": "",
                "shares": "",
                "price": "",
                "total_value": "",
                "side": "",
                "reason": "",
            },
        ]
        deduped = _dedupe_rows(rows)
        self.assertEqual(len(deduped), 1, "Reference-only duplicates should collapse to one row")
        self.assertEqual(deduped[0]["date"], "2025-01-01")
        self.assertEqual(deduped[0]["company"], "Alpha")
        self.assertEqual(deduped[0]["name"], "Person A")

    def test_past_tense_sell_detection(self):
        text = "The person sold 1,000 shares on 2025-12-01 at price 10.50."
        normalized = normalize_text_for_search(text)
        self.assertEqual(
            determine_side_from_text("", normalized),
            "sell",
            "Should classify past-tense 'sold' narrative as a sell",
        )

    def test_owned_by_name_extraction(self):
        text = "Vorup Invest ApS is owned by board member Lars Kristensen"
        self.assertEqual(
            extract_narrative_name_from_text(text),
            "Lars Kristensen",
            "Should pick the natural person even when the notice names an owning entity",
        )

    def test_nature_skips_section_labels_and_sets_side(self):
        sample = "\n".join(
            [
                "b.",
                "Transaktionens art",
                "Tildeling af aktier som aktieløn",
                "c.",
                "Pris(er)",
                "Mængde(r)",
                "DKK 18,37",
                "7.846 aktier",
            ]
        )
        lines = build_lines(sample)
        nature = extract_nature(lines)
        self.assertEqual(nature, "Tildeling af aktier som aktieløn")
        side = determine_side_from_text(nature, normalize_text_for_search(sample))
        self.assertEqual(side, "buy")

    def test_price_volume_after_headers_skip_lei(self):
        sample = "\n".join(
            [
                "Price(s)",
                "Volume(s)",
                "LEI: 213800ATZVDWWKJ8NI47",
                "Purchase",
                "1,6",
                "56.999",
            ]
        )
        lines = build_lines(sample)
        txs = extract_transactions(lines, normalize_text_for_search(sample))
        self.assertIn(
            ("56.999", "1,6"),
            txs,
            "Should ignore LEI-like codes and pick subsequent numeric price/volume lines",
        )

    def test_skip_portfolio_placeholder(self):
        rows = parse_insider_file(
            "For the best experience, open this PDF portfolio in Acrobat Reader",
            Path("Asetek_portfolio.txt"),
        )
        self.assertEqual(rows, [], "Portfolio placeholder files should be skipped")

    def test_extract_name_from_appended_value_block(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Asetek_AS_Ledende_medarbejderes_transaktioner_a2af337147894ca07d4b48aa69b9097d8_Jakob_Have_insider_notification_25112025.txt"
        )
        if not stub.exists():
            self.skipTest("Asetek appended-value name fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertTrue(rows, "Expected at least one parsed row for Asetek fixture")
        self.assertTrue(all(row.get("name") == "Jakob Alsted Have" for row in rows))

    def test_extract_name_prefers_person_section_over_issuer(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "AstraZeneca_PLC_Ledende_medarbejderes_transaktioner_a15113af1a69c629443a8da1616677a4f.txt"
        )
        if not stub.exists():
            self.skipTest("AstraZeneca name fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertTrue(rows, "Expected at least one parsed row for AstraZeneca fixture")
        self.assertTrue(all(row.get("name") == "Pascal Soriot" for row in rows))

    def test_skip_attachment_notice_without_data(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Ambu_AS_Ledende_medarbejderes_transaktioner_a072c1bd8fdc8d51318df641dca02ee65.txt"
        )
        if not stub.exists():
            self.skipTest("Ambu attachment-only fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(
            rows,
            [],
            "Notices that only point to attached PDFs should be skipped until the attachment is parsed.",
        )

    def test_skip_attachment_notice_bactiquant(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "BactiQuant_AS_Ledende_medarbejderes_transaktioner_a24a85b8255dadb1e16cab1af5caadba2.txt"
        )
        if not stub.exists():
            self.skipTest("BactiQuant attachment-only fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(
            rows,
            [],
            "Notices that only point to attached forms (without transaction data) should be skipped.",
        )

    def test_skip_stock_option_notice_better_collective(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Better_Collective_AS_Ledende_medarbejderes_transaktioner_af82bf2db67fe44b2b7740ed3da9ba567.txt"
        )
        if not stub.exists():
            self.skipTest("Better Collective option fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(rows, [], "Stock option grants should be ignored.")

    def test_skip_stock_option_notice_componenta(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Componenta_Oyj_Ledende_medarbejderes_transaktioner_view_idb03568cc156ce0ad41f725ecda9398870langensrclisted.txt"
        )
        if not stub.exists():
            self.skipTest("Componenta option fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(rows, [], "Stock option programme instruments should be ignored.")

    def test_keep_share_transaction_linked_to_stock_option_program(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Scanfil_Oyj_Ledende_medarbejderes_transaktioner_aae4a5a18142188691e9cabb209cb35d7.txt"
        )
        if not stub.exists():
            self.skipTest("Scanfil share fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertTrue(
            rows and rows[0].get("shares"),
            "Share transactions linked to an option programme should still be kept.",
        )

    def test_skip_share_based_incentive_receipt_aspo(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Aspo_Ledende_medarbejderes_transaktioner_view_idbb2abca53702288efe248c8f52b78b66flangensrclisted.txt"
        )
        if not stub.exists():
            self.skipTest("Aspo share-based incentive fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(rows, [], "Share-based incentive receipts should be ignored.")

    def test_skip_rsu_vesting_notice(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Oculis_Holding_AG_Ledende_medarbejderes_transaktioner_a645960f1aad528847e47c1653f755a26.txt"
        )
        if not stub.exists():
            self.skipTest("Oculis RSU vesting fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(rows, [], "RSU vesting transactions should be ignored.")

    def test_skip_restricted_share_unit_plan_treasury_transfer_notice(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Duell_Oyj_Ledende_medarbejderes_transaktioner_view_idb8f69b244069bbbe187f2bdb846c547b7langensrclisted.txt"
        )
        if not stub.exists():
            self.skipTest("Duell RSU plan fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertEqual(rows, [], "RSU plan treasury share transfers should be ignored.")

    def test_keep_sale_motivated_by_incentive_plan_taxes(self):
        stub = (
            REPO_ROOT
            / "test"
            / "insider"
            / "nasdaq_news_cli"
            / "Pandora_AS_Ledende_medarbejderes_transaktioner_af00e637211abeda71630155fdfac5c9e.txt"
        )
        if not stub.exists():
            self.skipTest("Pandora sale fixture missing")

        rows = parse_insider_file(stub.read_text(encoding="utf-8", errors="ignore"), stub)
        self.assertTrue(rows and rows[0].get("side") == "sell", "Sale transactions should not be skipped.")

    def test_share_instrument_not_skipped_by_option_hint(self):
        sample = "\n".join(
            [
                "Name of the instrument: Shares",
                "Nature of transaction: Purchase of shares under stock option programme",
                "Description of the financial instrument",
                "Shares",
            ]
        )
        lines = build_lines(sample)
        nature = extract_nature(lines)
        should_skip = _should_skip_stock_option_notice(
            lines=lines,
            normalized_text=normalize_text_for_search(sample),
            nature=nature,
        )
        self.assertFalse(
            should_skip,
            "Share trades that merely mention an option programme should not be skipped as stock options.",
        )

    def test_insider_cli_matches_snapshot(self):
        input_dir = REPO_ROOT / "test" / "insider" / "nasdaq_news_cli"
        expected_csv = REPO_ROOT / "test" / "insider" / "insider_summary_base.csv"

        self.assertTrue(input_dir.exists(), "Fixture input directory is missing")
        if not expected_csv.exists():
            self.skipTest("Expected snapshot CSV is missing")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "insider_summary.csv"
            cmd = [
                sys.executable,
                "interpret.py",
                "--task",
                "insider",
                "--output",
                str(output_csv),
                "--collect",
                "--input",
                str(input_dir),
            ]
            result = subprocess.run(
                cmd, cwd=REPO_ROOT, capture_output=True, text=True
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"interpret.py failed: {result.stderr or result.stdout}",
            )
            self.assertTrue(output_csv.exists(), "interpret.py did not create output CSV")

            expected_header, expected_rows = _read_csv(expected_csv)
            actual_header, actual_rows = _read_csv(output_csv)

            self.assertEqual(
                actual_header,
                expected_header,
                "CSV header changed; update the golden file if this is intentional.",
            )
            self.assertEqual(
                actual_rows,
                expected_rows,
                "CLI output changed; update test/insider/insider_summary.csv if the new output is better.",
            )


if __name__ == "__main__":
    unittest.main()
