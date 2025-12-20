import unittest

from downloadpdf import _is_pdf_host_allowed


class PdfHostFilterTest(unittest.TestCase):
    def test_trusted_cross_domain_attachment(self):
        # messageUrl on omxgroup.com pointing to attachment on nasdaq.com should be allowed
        allowed = _is_pdf_host_allowed(
            pdf_host="attachment.news.eu.nasdaq.com",
            allowed_domain="omxgroup.com",
            allow_external=False,
        )
        self.assertTrue(allowed)

    def test_external_blocked_without_flag(self):
        allowed = _is_pdf_host_allowed(
            pdf_host="example.com",
            allowed_domain="nasdaq.com",
            allow_external=False,
        )
        self.assertFalse(allowed)

    def test_external_allowed_with_flag(self):
        allowed = _is_pdf_host_allowed(
            pdf_host="example.com",
            allowed_domain="nasdaq.com",
            allow_external=True,
        )
        self.assertTrue(allowed)

    def test_multi_label_public_suffix_not_collapsed(self):
        allowed = _is_pdf_host_allowed(
            pdf_host="malicious.co.uk",
            allowed_domain="example.co.uk",
            allow_external=False,
        )
        self.assertFalse(allowed)


if __name__ == "__main__":
    unittest.main()
