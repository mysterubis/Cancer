# tests/test_sample.py

import unittest
from main import add

class SimpleTest(unittest.TestCase):

    def test_add(self):
        self.assertEqual(add(1, 2), 3)

if __name__ == '__main__':
    unittest.main()

