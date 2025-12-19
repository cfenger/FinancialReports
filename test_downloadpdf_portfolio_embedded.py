"""Tests for extracting embedded PDFs from PDF portfolios."""

import tempfile
import unittest
from pathlib import Path

from downloadpdf import handle_pdf_bytes

try:
    import fitz  # type: ignore

    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False


REPO_ROOT = Path(__file__).resolve().parent
PDF_PORTFOLIO = REPO_ROOT / "insider" / "nasdaq_news_cli" / "Asetek_AS_Ledende_medarbejderes_transaktioner_abc543fd3e78e7b72be2998ed02aea5e0.pdf"


class PortfolioEmbeddedExtractionTest(unittest.TestCase):
    @unittest.skipUnless(HAS_PYMUPDF, "PyMuPDF not installed")
    def test_embedded_pdfs_extracted_to_text(self):
        if not PDF_PORTFOLIO.exists():
            self.skipTest("Fixture PDF is missing")

        pdf_bytes = PDF_PORTFOLIO.read_bytes()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            self.assertEqual(doc.embfile_count(), 3, "Expected 3 embedded PDFs in portfolio fixture")

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dest = Path(tmpdir) / "portfolio.txt"
            outputs = handle_pdf_bytes(pdf_bytes, base_dest, to_text=True)

            self.assertGreaterEqual(len(outputs), 3, "Embedded PDF texts were not extracted")

            text_outputs = [p for p in outputs if p.suffix == ".txt" and "_portfolio" not in p.stem]
            self.assertGreaterEqual(len(text_outputs), 3, "Expected text files for each embedded PDF")

            combined = "\n".join(p.read_text(encoding="utf-8") for p in text_outputs)
            self.assertTrue(combined.strip(), "Combined embedded PDF text is empty")

        for expected in ("2025-12-03", "2025-12-04", "2025-12-05"):
            self.assertIn(expected, combined, f"Missing date marker from embedded PDFs: {expected}")


if __name__ == "__main__":
    unittest.main()
