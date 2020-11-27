import unittest
from src.model.eval_model import tonality

class TonalityTest(unittest.TestCase):
    def test_iloveu(self):
        self.assertEqual(tonality("Я тебя люблю"), "+")

if __name__ == '__main__':
    unittest.main()