import tempfile
import unittest
from pathlib import Path
from unittest import mock

from downloadpdf import handle_pdf_bytes


class HandlePdfBytesPortfolioTest(unittest.TestCase):
    def test_saves_original_portfolio_pdf_when_embedded_present(self):
        portfolio_bytes = b"%PDF-1.7 mock portfolio"

        # Pretend PyMuPDF is available and we discovered two embedded PDFs.
        with mock.patch(
            "downloadpdf.extract_embedded_pdfs",
            return_value=(2, [("a.pdf", b"%PDF-1.4 a"), ("b.pdf", b"%PDF-1.4 b")], True),
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                base_dest = Path(tmpdir) / "portfolio.pdf"
                outputs = handle_pdf_bytes(portfolio_bytes, base_dest, to_text=False)

                saved_paths = {p.name for p in outputs}
                self.assertIn("portfolio.pdf", saved_paths, "Original portfolio container was not saved")
                self.assertTrue(base_dest.exists(), "Expected original portfolio file on disk")

                # Embedded PDFs should also be saved.
                self.assertGreaterEqual(
                    len([p for p in outputs if p.suffix == ".pdf" and p.name != "portfolio.pdf"]),
                    2,
                )


if __name__ == "__main__":
    unittest.main()
