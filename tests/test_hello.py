import unittest
from miniball.example import hello


class Testminiball(unittest.TestCase):
    def test_hello(self):
        assert hello("Giulio") == "Hello Giulio!"


if __name__ == "__main__":
    unittest.main()
