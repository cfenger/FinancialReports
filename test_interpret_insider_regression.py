"""Regression test for the insider parsing CLI."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


class InterpretInsiderRegressionTest(unittest.TestCase):
    def test_insider_cli_matches_snapshot(self):
        input_dir = REPO_ROOT / "test" / "insider" / "nasdaq_news_cli"
        expected_csv = REPO_ROOT / "test" / "insider" / "insider_summary.csv"

        self.assertTrue(input_dir.exists(), "Fixture input directory is missing")
        self.assertTrue(expected_csv.exists(), "Expected snapshot CSV is missing")

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
