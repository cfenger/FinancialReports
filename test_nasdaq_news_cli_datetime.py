import unittest

from nasdaq_news_cli import _parse_item_datetime


class ParseItemDatetimeTest(unittest.TestCase):
    def test_parses_slash_format(self):
        dt = _parse_item_datetime("10/12/2025 16:52")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.month, 12)
        self.assertEqual(dt.day, 10)
        self.assertEqual(dt.hour, 16)
        self.assertEqual(dt.minute, 52)

    def test_parses_iso_with_millis(self):
        dt = _parse_item_datetime("2025-11-14 15:21:00.123")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2025)
        self.assertEqual(dt.month, 11)
        self.assertEqual(dt.day, 14)


if __name__ == "__main__":
    unittest.main()
