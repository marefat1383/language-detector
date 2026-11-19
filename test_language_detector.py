import unittest

from language_detector import detect_language


class LanguageDetectorTests(unittest.TestCase):
    def test_english_sentence(self):
        self.assertEqual(detect_language("hello how are you"), "english")

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_language("   ")


if __name__ == "__main__":
    unittest.main()
