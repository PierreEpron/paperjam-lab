import unittest

from src.dataset import BaseDataLoader, RelDataLoader

class BaseDataLoaderText(unittest.TestCase):
    def test_get_ent_type(self):
        self.assertEqual(BaseDataLoader.get_ent_type((0,10), {'ner':[(0, 10, 'a'), (5, 12, 'b'), (42, 52, 'c')]}), 'a')
        self.assertEqual(BaseDataLoader.get_ent_type((5, 12), {'ner':[(0, 10, 'a'), (5, 12, 'b'), (42, 52, 'c')]}), 'b')
        self.assertEqual(BaseDataLoader.get_ent_type((0,10), {'ner':[(0, 10, 'a'), (5, 12, 'b'), (42, 52, 'c')], 'd':'e'}), 'a')
        with self.assertRaises(RuntimeError):
            BaseDataLoader.get_ent_type((0,12), {'ner':[(0, 10, 'a'), (5, 12, 'b'), (42, 52, 'c')], 'doc_id':'id'})

class RelDataLoaderTest(unittest.TestCase):

    def test_get_onehot(self):
        self.assertEqual(RelDataLoader.get_onehot([], [0,1,2]), [0,0,0])
        self.assertEqual(RelDataLoader.get_onehot([1], [0,1,2]), [0,1,0])
        self.assertEqual(RelDataLoader.get_onehot([0,1,2], [0,1,2]), [1,1,1])

    def test_get_map_lists(self):
        self.assertEqual(RelDataLoader.get_map_lists([0,1,2], [2,1,0]), ({0:2,1:1,2:0}, {2:0,1:1,0:2}))
        self.assertEqual(RelDataLoader.get_map_lists([2,1,0], [0,1,2]), ({2:0,1:1,0:2}, {0:2,1:1,2:0}))
        self.assertEqual(RelDataLoader.get_map_lists([2,1,0]), ({2:0,1:1,0:2}, {0:2,1:1,2:0}))
        self.assertEqual(RelDataLoader.get_map_lists([]), ({}, {}))
        self.assertEqual(RelDataLoader.get_map_lists(['a', 'b', 'c'], [(0), (1), (2)]), ({'a':(0),'b':(1),'c':(2)}, {(0):'a',(1):'b',(2):'c'}))
        self.assertEqual(RelDataLoader.get_map_lists({'a':'z', 'b':'y', 'c':'x'}), ({'a':0, 'b':1, 'c':2}, {0:'a', 1:'b', 2:'c'}))

if __name__ == '__main__':
    unittest.main()