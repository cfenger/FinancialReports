"""Tests for extracting PDF text with AcroForm values included."""

import tempfile
import unittest
from pathlib import Path

from downloadpdf import save_pdf_bytes_as_text

try:
    import fitz  # type: ignore

    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

REPO_ROOT = Path(__file__).resolve().parent
PDF_WITH_FORM = REPO_ROOT / "insider" / "nasdaq_news_cli" / "AL_Sydbank_AS_Ledende_medarbejderes_transaktioner_a84ccd28f36945890697427b1445dc986.pdf"


class SavePdfAsTextFormFieldsTest(unittest.TestCase):
    @unittest.skipUnless(HAS_PYMUPDF, "PyMuPDF not installed")
    def test_includes_acroform_values(self):
        if not PDF_WITH_FORM.exists():
            self.skipTest("Fixture PDF is missing")

        pdf_bytes = PDF_WITH_FORM.read_bytes()
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "out.txt"
            ok = save_pdf_bytes_as_text(pdf_bytes, dest)

            self.assertTrue(ok, "Expected PDF text extraction to succeed with PyMuPDF")
            text = dest.read_text(encoding="utf-8")

        for expected in ("553,5", "158", "2025-12-01"):
            self.assertIn(expected, text, f"Missing AcroForm value: {expected}")


if __name__ == "__main__":
    unittest.main()
