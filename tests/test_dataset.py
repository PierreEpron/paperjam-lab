import unittest

from src.dataset import RelDataLoader

class RelDataLoaderTest(unittest.TestCase):
    def test_get_onehot(self):
        self.assertEqual(RelDataLoader.get_onehot([], [0,1,2]), [0,0,0])
        self.assertEqual(RelDataLoader.get_onehot([1], [0,1,2]), [0,1,0])
        self.assertEqual(RelDataLoader.get_onehot([0,1,2], [0,1,2]), [1,1,1])

if __name__ == '__main__':
    unittest.main()