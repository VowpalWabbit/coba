import pickle
import unittest

from collections import Counter
from math import isnan

from coba.pipes      import LazyDense, LazySparse, HeadDense, ListSink
from coba.context    import CobaContext, NullLogger, BasicLogger
from coba.exceptions import CobaException
from coba.primitives import Categorical
from coba.primitives import LoggedInteraction, SimulatedInteraction, GroundedInteraction
from coba.primitives import DiscreteReward, L1Reward, BinaryReward
from coba.learners   import FixedLearner
from coba.utilities  import peek_first, PackageChecker

from coba.environments.filters import Sparsify, Sort, Scale, Cycle, Impute, Binary, Flatten, Params, Batch
from coba.environments.filters import Densify, Shuffle, Take, Reservoir, Where, Noise, Riffle, Grounded, Slice
from coba.environments.filters import Finalize, Repr, BatchSafe, Cache, Logged, Unbatch, Mutable, OpeRewards

class TestEnvironment:
    def __init__(self, id) -> None:
        self._id = id

    @property
    def params(self):
        return {'id':self._id}

    def read(self):
        return [
            SimulatedInteraction(1, [None,None], [1,2]),
            SimulatedInteraction(2, [None,None], [2,3]),
            SimulatedInteraction(3, [None,None], [3,4]),
        ]

    def __str__(self) -> str:
        return str(self.params)

class NoParamIdent:
    def filter(self,item):
        return item

    def __str__(self) -> str:
        return 'NoParamIdent'

CobaContext.logger = NullLogger()

class Shuffle_Tests(unittest.TestCase):

    def test_str(self):
        self.assertEqual("Shuffle('shuffle_seed': 1)", str(Shuffle(1)))

    def test_empty(self):
        self.assertEqual(list(Shuffle(1).filter([])),[])

    def test_logged(self):
        logged_interactions = [{'context':0,'action':1,'reward':2},{'context':1,'action':1,'reward':2},{'context':2,'action':1,'reward':2}]
        normal_interactions = [{'context':0},{'context':1},{'context':2}]

        logged_order = [i['context'] for i in Shuffle(1).filter(logged_interactions)]
        normal_order = [i['context'] for i in Shuffle(1).filter(normal_interactions)]

        self.assertEqual(normal_interactions,[{'context':0},{'context':1},{'context':2}])
        self.assertEqual(logged_interactions,[{'context':0,'action':1,'reward':2},{'context':1,'action':1,'reward':2},{'context':2,'action':1,'reward':2}])
        self.assertNotEqual(normal_order,normal_interactions)
        self.assertNotEqual(logged_order,logged_interactions)
        self.assertNotEqual(logged_order,normal_order)

class Sort_Tests(unittest.TestCase):

    def test_sort_missing(self):
        interactions = [ {'action':1}, {'action':2} ]

        mem_interactions = interactions
        srt_interactions = list(Sort().filter(mem_interactions))

        self.assertEqual(mem_interactions,interactions)
        self.assertEqual(srt_interactions,interactions)

    def test_sort_empty(self):
        interactions = []

        mem_interactions = interactions
        srt_interactions = list(Sort().filter(mem_interactions))

        self.assertEqual(mem_interactions,interactions)
        self.assertEqual(srt_interactions,interactions)


    def test_sort1_logged(self):
        interactions = [
            LoggedInteraction((7,2), 1, 1),
            LoggedInteraction((1,9), 1, 1),
            LoggedInteraction((8,3), 1, 1)
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0]).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])
        self.assertEqual((1,9), srt_interactions[0]['context'])
        self.assertEqual((7,2), srt_interactions[1]['context'])
        self.assertEqual((8,3), srt_interactions[2]['context'])

    def test_sort1(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0]).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])
        self.assertEqual((1,9), srt_interactions[0]['context'])
        self.assertEqual((7,2), srt_interactions[1]['context'])
        self.assertEqual((8,3), srt_interactions[2]['context'])

    def test_sort2(self):
        interactions = [
            SimulatedInteraction((1,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((1,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0,1]).filter(mem_interactions))

        self.assertEqual((1,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((1,3), mem_interactions[2]['context'])
        self.assertEqual((1,2), srt_interactions[0]['context'])
        self.assertEqual((1,3), srt_interactions[1]['context'])
        self.assertEqual((1,9), srt_interactions[2]['context'])

    def test_sort3(self):
        interactions = [
            SimulatedInteraction((1,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((1,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort(*[0,1]).filter(mem_interactions))

        self.assertEqual((1,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((1,3), mem_interactions[2]['context'])
        self.assertEqual((1,2), srt_interactions[0]['context'])
        self.assertEqual((1,3), srt_interactions[1]['context'])
        self.assertEqual((1,9), srt_interactions[2]['context'])

    def test_params(self):
        self.assertEqual({'sort_keys':'*'}, Sort().params)
        self.assertEqual({'sort_keys':[0]}, Sort(0).params)
        self.assertEqual({'sort_keys':[0]}, Sort([0]).params)
        self.assertEqual({'sort_keys':[1,2]}, Sort([1,2]).params)

    def test_str(self):
        self.assertEqual("Sort('sort_keys': [0])", str(Sort([0])))

class Take_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Take(-1)

        with self.assertRaises(ValueError):
            Take('A')

        with self.assertRaises(ValueError):
            Take((-1,5))

    def test_take_1_not_strict(self):

        items = [ 1,2,3 ]
        take_items = list(Take(1).filter(items))

        self.assertEqual([1    ], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_2_not_strict(self):
        items = [ 1,2,3 ]
        take_items = list(Take(2).filter(items))

        self.assertEqual([1,2  ], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_3_not_strict(self):
        items = [ 1,2,3 ]
        take_items = list(Take(3).filter(items))

        self.assertEqual([1,2,3], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_4_not_strict(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4).filter(items))

        self.assertEqual([1,2,3], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_4_strict(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4,strict=True).filter(items))

        self.assertEqual([]     , take_items)
        self.assertEqual([1,2,3], items     )

class Resevoir_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Reservoir(-1)

        with self.assertRaises(ValueError):
            Reservoir('A')

        with self.assertRaises(ValueError):
            Reservoir((-1,5))

    def test_not_strict(self):
        items = [1,2,3,4,5]

        take_items = list(Reservoir(2,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([4,2]      , take_items)

        take_items = list(Reservoir(None,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(5,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(6,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(0,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([]         , take_items)

    def test_strict(self):
        items = [1,2,3,4,5]

        take_items = list(Reservoir(2,strict=True,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([4,2]      , take_items)

        take_items = list(Reservoir(None,strict=True,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(5,strict=True,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([1,5,4,3,2], take_items)

        take_items = list(Reservoir(6,strict=True,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([]         , take_items)

        take_items = list(Reservoir(0,strict=True,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([]         , take_items)

class Where_Tests(unittest.TestCase):

    def test_n_interactions(self):
        items = [ 1,2,3 ]
        self.assertEqual([]     , list(Where(n_interactions=2       ).filter([])))
        self.assertEqual([]     , list(Where(n_interactions=2       ).filter(items)))
        self.assertEqual([]     , list(Where(n_interactions=4       ).filter(items)))
        self.assertEqual([]     , list(Where(n_interactions=(None,1)).filter(items)))
        self.assertEqual([]     , list(Where(n_interactions=(4,None)).filter(items)))

        self.assertEqual([1,2,3], list(Where(n_interactions=(1,None)).filter(items)))
        self.assertEqual([1,2,3], list(Where(n_interactions=(None,4)).filter(items)))
        self.assertEqual([1,2,3], list(Where(n_interactions=(1,4)   ).filter(items)))
        self.assertEqual([1,2,3], list(Where(n_interactions=3       ).filter(items)))
        self.assertEqual([1,2,3], list(Where(                       ).filter(items)))

    def test_n_actions(self):
        items = [ {'actions':[1,2,3]} ] * 3
        self.assertEqual([]     , list(Where(n_actions=2       ).filter([])))
        self.assertEqual([]     , list(Where(n_actions=2       ).filter(items)))
        self.assertEqual([]     , list(Where(n_actions=4       ).filter(items)))
        self.assertEqual([]     , list(Where(n_actions=(None,1)).filter(items)))
        self.assertEqual([]     , list(Where(n_actions=(4,None)).filter(items)))

        self.assertEqual(items, list(Where(n_actions=(1,None)).filter(items)))
        self.assertEqual(items, list(Where(n_actions=(None,4)).filter(items)))
        self.assertEqual(items, list(Where(n_actions=(1,4)   ).filter(items)))
        self.assertEqual(items, list(Where(n_actions=3       ).filter(items)))
        self.assertEqual(items, list(Where(                  ).filter(items)))

    def test_n_features(self):
        items = [ {'context':[1,2,3]} ] * 3
        self.assertEqual([]     , list(Where(n_features=2       ).filter([])))
        self.assertEqual([]     , list(Where(n_features=2       ).filter(items)))
        self.assertEqual([]     , list(Where(n_features=4       ).filter(items)))
        self.assertEqual([]     , list(Where(n_features=(None,1)).filter(items)))
        self.assertEqual([]     , list(Where(n_features=(4,None)).filter(items)))

        self.assertEqual(items, list(Where(n_features=(1,None)).filter(items)))
        self.assertEqual(items, list(Where(n_features=(None,4)).filter(items)))
        self.assertEqual(items, list(Where(n_features=(1,4)   ).filter(items)))
        self.assertEqual(items, list(Where(n_features=3       ).filter(items)))
        self.assertEqual(items, list(Where(                  ).filter(items)))

    def test_params(self):
        self.assertEqual({'where_n_interactions':(None,1)}, Where(n_interactions=(None,1)).params)
        self.assertEqual({'where_n_interactions':(4,None)}, Where(n_interactions=(4,None)).params)
        self.assertEqual({'where_n_interactions':1       }, Where(n_interactions=1       ).params)
        self.assertEqual({'where_n_actions'     :(None,1)}, Where(n_actions     =(None,1)).params)
        self.assertEqual({'where_n_actions'     :(4,None)}, Where(n_actions     =(4,None)).params)
        self.assertEqual({'where_n_actions'     :1       }, Where(n_actions     =1       ).params)
        self.assertEqual({'where_n_features'    :(None,1)}, Where(n_features    =(None,1)).params)
        self.assertEqual({'where_n_features'    :(4,None)}, Where(n_features    =(4,None)).params)
        self.assertEqual({'where_n_features'    :1       }, Where(n_features    =1       ).params)

        self.assertEqual({                               }, Where(                       ).params)


class Scale_Tests(unittest.TestCase):

    def test_scale_empty(self):
        self.assertEqual(list(Scale("min","minmax").filter([])),[])

    def test_scale_no_context(self):
        interactions = [
            {'action':1,'reward':1},
            {'action':1,'reward':1},
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(mem_interactions,interactions)
        self.assertEqual(scl_interactions,interactions)

    def test_scale_partial_none_context(self):
        interactions = [
            LoggedInteraction(None, 1, 1),
            LoggedInteraction(1   , 1, 1),
            LoggedInteraction(2   , 1, 1)
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(None, mem_interactions[0]['context'])
        self.assertEqual(1   , mem_interactions[1]['context'])
        self.assertEqual(2   , mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0]['context'])
        self.assertEqual(0   , scl_interactions[1]['context'])
        self.assertEqual(1   , scl_interactions[2]['context'])

    def test_scale_none_context(self):
        interactions = [
            LoggedInteraction(None, 1, 1),
            LoggedInteraction(None, 1, 1),
            LoggedInteraction(None, 1, 1)
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(None, mem_interactions[0]['context'])
        self.assertEqual(None, mem_interactions[1]['context'])
        self.assertEqual(None, mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0]['context'])
        self.assertEqual(None, scl_interactions[1]['context'])
        self.assertEqual(None, scl_interactions[2]['context'])

    def test_scale_min_and_minmax_using_all_logged(self):
        interactions = [
            LoggedInteraction((7,2), 1, 1),
            LoggedInteraction((1,9), 1, 1),
            LoggedInteraction((8,3), 1, 1)
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([6/7,0  ], scl_interactions[0]['context'])
        self.assertEqual([0  ,1  ], scl_interactions[1]['context'])
        self.assertEqual([1  ,1/7], scl_interactions[2]['context'])

    def test_scale_min_and_minmax_using_all_simulated(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([6/7,0  ], scl_interactions[0]['context'])
        self.assertEqual([0  ,1  ], scl_interactions[1]['context'])
        self.assertEqual([1  ,1/7], scl_interactions[2]['context'])

    def test_scale_min_and_minmax_using_2(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax",using=2).filter(interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([1  ,0], scl_interactions[0]['context'])
        self.assertEqual([0  ,1], scl_interactions[1]['context'])

        self.assertAlmostEqual(7/6, scl_interactions[2]['context'][0])
        self.assertAlmostEqual(1/7, scl_interactions[2]['context'][1])

    def test_scale_lazy_dense_0_and_2(self):
        interactions = [
            SimulatedInteraction(LazyDense((8,2)), [1], [1]),
            SimulatedInteraction(LazyDense((4,4)), [1], [1]),
            SimulatedInteraction(LazyDense((2,6)), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,4), mem_interactions[1]['context'])
        self.assertEqual((2,6), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([4,1], scl_interactions[0]['context'])
        self.assertEqual([2,2], scl_interactions[1]['context'])
        self.assertEqual([1,3], scl_interactions[2]['context'])

    def test_scale_lazy_sparse_0_and_2(self):
        interactions = [
            SimulatedInteraction(LazySparse({'a':8,'b':2}), [1], [1]),
            SimulatedInteraction(LazySparse({'a':4,'b':4}), [1], [1]),
            SimulatedInteraction(LazySparse({'a':2,'b':6}), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual({'a':8,'b':2}, mem_interactions[0]['context'])
        self.assertEqual({'a':4,'b':4}, mem_interactions[1]['context'])
        self.assertEqual({'a':2,'b':6}, mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual({'a':4,'b':1}, scl_interactions[0]['context'])
        self.assertEqual({'a':2,'b':2}, scl_interactions[1]['context'])
        self.assertEqual({'a':1,'b':3}, scl_interactions[2]['context'])

    def test_scale_0_and_2_tuples(self):
        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((2,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,4), mem_interactions[1]['context'])
        self.assertEqual((2,6), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([4,1], scl_interactions[0]['context'])
        self.assertEqual([2,2], scl_interactions[1]['context'])
        self.assertEqual([1,3], scl_interactions[2]['context'])

    def test_scale_0_and_2_lists(self):
        interactions = [
            SimulatedInteraction([8,2], [1], [1]),
            SimulatedInteraction([4,4], [1], [1]),
            SimulatedInteraction([2,6], [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual([8,2], mem_interactions[0]['context'])
        self.assertEqual([4,4], mem_interactions[1]['context'])
        self.assertEqual([2,6], mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([4,1], scl_interactions[0]['context'])
        self.assertEqual([2,2], scl_interactions[1]['context'])
        self.assertEqual([1,3], scl_interactions[2]['context'])

    def test_scale_0_and_2_single_number(self):
        interactions = [
            SimulatedInteraction(2, [1], [1]),
            SimulatedInteraction(4, [1], [1]),
            SimulatedInteraction(6, [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual(2, mem_interactions[0]['context'])
        self.assertEqual(4, mem_interactions[1]['context'])
        self.assertEqual(6, mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(1, scl_interactions[0]['context'])
        self.assertEqual(2, scl_interactions[1]['context'])
        self.assertEqual(3, scl_interactions[2]['context'])

    def test_scale_mean_and_std(self):
        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((0,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="mean",scale="std").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,4), mem_interactions[1]['context'])
        self.assertEqual((0,6), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([ 4/4,-2/2], scl_interactions[0]['context'])
        self.assertEqual([ 0/4, 0/2], scl_interactions[1]['context'])
        self.assertEqual([-4/4, 2/2], scl_interactions[2]['context'])

    def test_scale_maxabs(self):
        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((0,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(scale="maxabs").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,4), mem_interactions[1]['context'])
        self.assertEqual((0,6), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([1.,1/3], scl_interactions[0]['context'])
        self.assertEqual([.5,2/3], scl_interactions[1]['context'])
        self.assertEqual([0.,3/3], scl_interactions[2]['context'])

    def test_scale_med_and_iqr(self):
        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((0,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="med",scale="iqr").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,4), mem_interactions[1]['context'])
        self.assertEqual((0,6), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([ 1,-1], scl_interactions[0]['context'])
        self.assertEqual([ 0, 0], scl_interactions[1]['context'])
        self.assertEqual([-1, 1], scl_interactions[2]['context'])

    def test_scale_med_and_iqr_0(self):
        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,2), [1], [1]),
            SimulatedInteraction((0,2), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="med",scale="iqr").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,2), mem_interactions[1]['context'])
        self.assertEqual((0,2), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([ 1, 0], scl_interactions[0]['context'])
        self.assertEqual([ 0, 0], scl_interactions[1]['context'])
        self.assertEqual([-1, 0], scl_interactions[2]['context'])

    def test_scale_min_and_minmax_with_str(self):
        interactions = [
            SimulatedInteraction((7,2,"A"), [1], [1]),
            SimulatedInteraction((1,9,"B"), [1], [1]),
            SimulatedInteraction((8,3,"C"), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2,"A"), mem_interactions[0]['context'])
        self.assertEqual((1,9,"B"), mem_interactions[1]['context'])
        self.assertEqual((8,3,"C"), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([6/7,0  ,"A"], scl_interactions[0]['context'])
        self.assertEqual([0  ,1  ,"B"], scl_interactions[1]['context'])
        self.assertEqual([1  ,1/7,"C"], scl_interactions[2]['context'])

    def test_scale_min_and_minmax_with_nan(self):
        interactions = [
            SimulatedInteraction((float('nan'), 2           ), [1], [1]),
            SimulatedInteraction((1           , 9           ), [1], [1]),
            SimulatedInteraction((8           , float('nan')), [1], [1])
        ]

        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertTrue(isnan(scl_interactions[0]['context'][0]))
        self.assertEqual(0,   scl_interactions[0]['context'][1])

        self.assertEqual([0, 1], scl_interactions[1]['context'])

        self.assertEqual(1  , scl_interactions[2]['context'][0])
        self.assertTrue(isnan(scl_interactions[2]['context'][1]))

    def test_scale_min_and_minmax_with_str_and_nan(self):
        nan = float('nan')
        interactions = [
            LoggedInteraction((7,  2, 'A'), 1, 1),
            LoggedInteraction((1,  9, 'B'), 1, 1),
            LoggedInteraction((8,nan, nan), 1, 1)
        ]

        mem_interactions = interactions

        #with self.assertWarns(Warning):
        #This warning was removed because it was an expensive check.
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2,'A'), mem_interactions[0]['context'])
        self.assertEqual((1,9,'B'), mem_interactions[1]['context'])
        self.assertEqual(8,   mem_interactions[2]['context'][0] )
        self.assertTrue(isnan(mem_interactions[2]['context'][1]))
        self.assertTrue(isnan(mem_interactions[2]['context'][2]))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([6/7,0,'A'], scl_interactions[0]['context'])
        self.assertEqual([0  ,1,'B'], scl_interactions[1]['context'])
        self.assertEqual(1,   scl_interactions[2]['context'][0])
        self.assertTrue(isnan(scl_interactions[2]['context'][1]))
        self.assertTrue(isnan(scl_interactions[2]['context'][2]))

    def test_scale_min_and_minmax_with_mixed(self):
        interactions = [
            SimulatedInteraction(("A", 2  ), [1], [1]),
            SimulatedInteraction((1  , 9  ), [1], [1]),
            SimulatedInteraction((8  , "B"), [1], [1])
        ]

        #with self.assertWarns(Warning):
        #This warning was removed because it was an expensive check.
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(["A", 2  ], scl_interactions[0]['context'])
        self.assertEqual([1  , 9  ], scl_interactions[1]['context'])
        self.assertEqual([8  , "B"], scl_interactions[2]['context'])

    def test_scale_min_and_minmax_with_none_possible(self):
        interactions = [
            SimulatedInteraction(("A", "B"), [1], [1]),
            SimulatedInteraction((1  , 2  ), [1], [1]),
            SimulatedInteraction((8  , 9  ), [1], [1])
        ]

        #with self.assertWarns(Warning):
        #This warning was removed because it was an expensive check.
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(["A", "B"], scl_interactions[0]['context'])
        self.assertEqual([1  , 2  ], scl_interactions[1]['context'])
        self.assertEqual([8  , 9  ], scl_interactions[2]['context'])

    def test_scale_0_and_minmax_with_mixed_dict(self):
        interactions = [
            SimulatedInteraction({0:"A", 1:2              }, [1], [1]),
            SimulatedInteraction({0:1  , 1:9  , 2:2       }, [1], [1]),
            SimulatedInteraction({0:8  , 1:"B", 2:1       }, [1], [1]),
            SimulatedInteraction({0:8  , 1:"B", 2:1, 3:"C"}, [1], [1])
        ]

        #with self.assertWarns(Warning):
        #This warning was removed because it was an expensive check.
        scl_interactions = list(Scale(0,"minmax").filter(interactions))

        self.assertEqual(4, len(scl_interactions))

        self.assertEqual({0:"A", 1:2               }, scl_interactions[0]['context'])
        self.assertEqual({0:1  , 1:9  , 2:1        }, scl_interactions[1]['context'])
        self.assertEqual({0:8  , 1:"B", 2:.5       }, scl_interactions[2]['context'])
        self.assertEqual({0:8  , 1:"B", 2:.5, 3:"C"}, scl_interactions[3]['context'])

    def test_scale_min_and_minmax_with_dict(self):
        interactions = [
            SimulatedInteraction({0:"A", 1:2              }, [1], [1]),
            SimulatedInteraction({0:1  , 1:9  , 2:2       }, [1], [1]),
            SimulatedInteraction({0:8  , 1:"B", 2:1       }, [1], [1]),
            SimulatedInteraction({0:8  , 1:"B", 2:1, 3:"C"}, [1], [1])
        ]

        with self.assertRaises(CobaException) as e:
            list(Scale("min","minmax").filter(interactions))

        self.assertIn("Shift is required to be 0 for sparse environments", str(e.exception))

    def test_scale_min_and_minmax_with_None(self):
        interactions = [
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1])
        ]

        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0]['context'])
        self.assertEqual(None, scl_interactions[1]['context'])
        self.assertEqual(None, scl_interactions[2]['context'])

    def test_params(self):
        self.assertEqual({"shift":"mean","scale":"std","scale_using":None}, Scale(shift="mean",scale="std").params)
        self.assertEqual({"shift":2     ,"scale":1/2  ,"scale_using":None}, Scale(shift=2,scale=1/2).params)
        self.assertEqual({"shift":2     ,"scale":1/2  ,"scale_using":10  }, Scale(shift=2,scale=1/2,using=10).params)

    def test_iter(self):
        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((2,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(iter(interactions)))

        self.assertEqual((8,2), mem_interactions[0]['context'])
        self.assertEqual((4,4), mem_interactions[1]['context'])
        self.assertEqual((2,6), mem_interactions[2]['context'])

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual([4,1], scl_interactions[0]['context'])
        self.assertEqual([2,2], scl_interactions[1]['context'])
        self.assertEqual([1,3], scl_interactions[2]['context'])

class Cycle_Tests(unittest.TestCase):

    def test_list_rewards(self):
        interactions = [
            {'context':(7,2), 'actions':[(1,0),(0,1)], 'rewards':[1,3]},
            {'context':(1,9), 'actions':[(1,0),(0,1)], 'rewards':[1,4]},
            {'context':(8,3), 'actions':[(1,0),(0,1)], 'rewards':[1,5]}
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle().filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0]['rewards'])
        self.assertEqual([1,4], mem_interactions[1]['rewards'])
        self.assertEqual([1,5], mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([3,1], cyc_interactions[0]['rewards'])
        self.assertEqual([4,1], cyc_interactions[1]['rewards'])
        self.assertEqual([5,1], cyc_interactions[2]['rewards'])

    def test_callable_rewards(self):
        interactions = [
            {'context':(7,2), 'actions':[(1,0),(0,1)], 'rewards':DiscreteReward([(1,0),(0,1)],[1,3])},
            {'context':(1,9), 'actions':[(1,0),(0,1)], 'rewards':DiscreteReward([(1,0),(0,1)],[1,4])},
            {'context':(8,3), 'actions':[(1,0),(0,1)], 'rewards':DiscreteReward([(1,0),(0,1)],[1,5])}
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle().filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0]['rewards'])
        self.assertEqual([1,4], mem_interactions[1]['rewards'])
        self.assertEqual([1,5], mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([3,1], cyc_interactions[0]['rewards'])
        self.assertEqual([4,1], cyc_interactions[1]['rewards'])
        self.assertEqual([5,1], cyc_interactions[2]['rewards'])

    def test_after_0(self):
        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle().filter(mem_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), mem_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,4]), mem_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,5]), mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[3,1]), cyc_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[4,1]), cyc_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[5,1]), cyc_interactions[2]['rewards'])

    def test_after_1(self):
        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=1).filter(mem_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), mem_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,4]), mem_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,5]), mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), cyc_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[4,1]), cyc_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[5,1]), cyc_interactions[2]['rewards'])

    def test_after_2(self):
        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=2).filter(mem_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), mem_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,4]), mem_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,5]), mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), cyc_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,4]), cyc_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[5,1]), cyc_interactions[2]['rewards'])

    def test_after_10(self):
        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=10).filter(mem_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), mem_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,4]), mem_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,5]), mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,3]), cyc_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,4]), cyc_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward([(1,0),(0,1)],[1,5]), cyc_interactions[2]['rewards'])

    def test_after_10_categoricals(self):
        interactions = [
            SimulatedInteraction((7,2), ['a','b'], [1,3]),
            SimulatedInteraction((1,9), ['a','b'], [1,4]),
            SimulatedInteraction((8,3), ['a','b'], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=10).filter(mem_interactions))

        self.assertEqual(DiscreteReward(['a','b'],[1,3]), mem_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward(['a','b'],[1,4]), mem_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward(['a','b'],[1,5]), mem_interactions[2]['rewards'])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual(DiscreteReward(['a','b'],[1,3]), cyc_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward(['a','b'],[1,4]), cyc_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward(['a','b'],[1,5]), cyc_interactions[2]['rewards'])

    def test_with_action_features(self):
        interactions = [
            SimulatedInteraction((7,2), [1,2], [1,3]),
            SimulatedInteraction((1,9), [1,2], [1,4]),
            SimulatedInteraction((8,3), [1,2], [1,5])
        ]

        with self.assertWarns(Warning):
            mem_interactions = interactions
            cyc_interactions = list(Cycle(after=0).filter(mem_interactions))

            self.assertEqual(DiscreteReward([1,2],[1,3]), mem_interactions[0]['rewards'])
            self.assertEqual(DiscreteReward([1,2],[1,4]), mem_interactions[1]['rewards'])
            self.assertEqual(DiscreteReward([1,2],[1,5]), mem_interactions[2]['rewards'])

            self.assertEqual(3, len(cyc_interactions))

            self.assertEqual(DiscreteReward([1,2],[1,3]), cyc_interactions[0]['rewards'])
            self.assertEqual(DiscreteReward([1,2],[1,4]), cyc_interactions[1]['rewards'])
            self.assertEqual(DiscreteReward([1,2],[1,5]), cyc_interactions[2]['rewards'])

    def test_params(self):
        self.assertEqual({"cycle_after":0 }, Cycle().params)
        self.assertEqual({"cycle_after":2 }, Cycle(2).params)

class Impute_Tests(unittest.TestCase):

    def test_impute_empty(self):
        self.assertEqual(list(Impute("mean",False).filter([])),[])

    def test_impute_no_context(self):
        interactions = [
            {'action':1,'reward':1},
            {'action':1,'reward':1},
            {'action':1,'reward':1},
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",False).filter(interactions))

        self.assertSequenceEqual(mem_interactions, interactions)
        self.assertSequenceEqual(imp_interactions, interactions)

    def test_impute_mean_logged(self):
        interactions = [
            LoggedInteraction((7   , 2   ), 1, 1),
            LoggedInteraction((None, None), 1, 1),
            LoggedInteraction((8   , 3   ), 1, 1)
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",False).filter(interactions))

        self.assertSequenceEqual((7,2), mem_interactions[0]['context'])
        self.assertSequenceEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual((7  ,  2), imp_interactions[0]['context'])
        self.assertSequenceEqual((7.5,2.5), imp_interactions[1]['context'])
        self.assertSequenceEqual((8  ,3  ), imp_interactions[2]['context'])

    def test_impute_dense_indicator(self):
        interactions = [
            LoggedInteraction((7   , 2   ), 1, 1),
            LoggedInteraction((8   , None), 1, 1),
            LoggedInteraction((None, 3   ), 1, 1)
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",True).filter(interactions))

        self.assertSequenceEqual((7   ,2   ), mem_interactions[0]['context'])
        self.assertSequenceEqual((8   ,None), mem_interactions[1]['context'])
        self.assertSequenceEqual((None,   3), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual((7  ,  2, 0, 0), imp_interactions[0]['context'])
        self.assertSequenceEqual((8  ,2.5, 0, 1), imp_interactions[1]['context'])
        self.assertSequenceEqual((7.5,3  , 1, 0), imp_interactions[2]['context'])

    def test_one_imputable(self):
        interactions = [
            SimulatedInteraction((1   ,'B'), [1], [1]),
            SimulatedInteraction((None,'C'), [1], [1]),
            SimulatedInteraction((1   ,'E'), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",False).filter(interactions))

        self.assertSequenceEqual((1   ,'B'), mem_interactions[0]['context'])
        self.assertSequenceEqual((None,'C'), mem_interactions[1]['context'])
        self.assertSequenceEqual((1   ,'E'), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual((1,'B'), imp_interactions[0]['context'])
        self.assertSequenceEqual((1,'C'), imp_interactions[1]['context'])
        self.assertSequenceEqual((1,'E'), imp_interactions[2]['context'])

    def test_none_imputable(self):
        interactions = [
            SimulatedInteraction(('A' ,'B'), [1], [1]),
            SimulatedInteraction((None,'C'), [1], [1]),
            SimulatedInteraction(('D' ,'E'), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",False).filter(interactions))

        self.assertSequenceEqual(('A' ,'B'), mem_interactions[0]['context'])
        self.assertSequenceEqual((None,'C'), mem_interactions[1]['context'])
        self.assertSequenceEqual(('D' ,'E'), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual(('A' ,'B'), imp_interactions[0]['context'])
        self.assertSequenceEqual((None,'C'), imp_interactions[1]['context'])
        self.assertSequenceEqual(('D' ,'E'), imp_interactions[2]['context'])

    def test_impute_nothing(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",False).filter(interactions))

        self.assertSequenceEqual((7,2), mem_interactions[0]['context'])
        self.assertSequenceEqual((1,9), mem_interactions[1]['context'])
        self.assertSequenceEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual((7,2), imp_interactions[0]['context'])
        self.assertSequenceEqual((1,9), imp_interactions[1]['context'])
        self.assertSequenceEqual((8,3), imp_interactions[2]['context'])

    def test_impute_nothing_indicator(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",True).filter(interactions))

        self.assertSequenceEqual((7,2), mem_interactions[0]['context'])
        self.assertSequenceEqual((1,9), mem_interactions[1]['context'])
        self.assertSequenceEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual((7,2), imp_interactions[0]['context'])
        self.assertSequenceEqual((1,9), imp_interactions[1]['context'])
        self.assertSequenceEqual((8,3), imp_interactions[2]['context'])

    def test_impute_mean(self):
        interactions = [
            SimulatedInteraction((7   , 2   ), [1], [1]),
            SimulatedInteraction((None, None), [1], [1]),
            SimulatedInteraction((8   , 3   ), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute("mean",False).filter(interactions))

        self.assertSequenceEqual((7,2), mem_interactions[0]['context'])
        self.assertSequenceEqual((None,None), mem_interactions[1]['context'])
        self.assertSequenceEqual((8,3), mem_interactions[2]['context'])

        self.assertEqual(3, len(imp_interactions))

        self.assertSequenceEqual((7  ,  2), imp_interactions[0]['context'])
        self.assertSequenceEqual((7.5,2.5), imp_interactions[1]['context'])
        self.assertSequenceEqual((8  ,3  ), imp_interactions[2]['context'])

    def test_impute_med(self):
        interactions = [
            SimulatedInteraction((7   , 2   ), [1], [1]),
            SimulatedInteraction((7   , 2   ), [1], [1]),
            SimulatedInteraction((None, None), [1], [1]),
            SimulatedInteraction((8   , 3   ), [1], [1])
        ]

        imp_interactions = list(Impute("median",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertSequenceEqual((7, 2), imp_interactions[0]['context'])
        self.assertSequenceEqual((7, 2), imp_interactions[1]['context'])
        self.assertSequenceEqual((7, 2), imp_interactions[2]['context'])
        self.assertSequenceEqual((8, 3), imp_interactions[3]['context'])

    def test_impute_mode_None(self):
        interactions = [
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1])
        ]

        imp_interactions = list(Impute("mode",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(None, imp_interactions[0]['context'])
        self.assertEqual(None, imp_interactions[1]['context'])
        self.assertEqual(None, imp_interactions[2]['context'])
        self.assertEqual(None, imp_interactions[3]['context'])

    def test_impute_mode_None_indicator(self):
        interactions = [
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1])
        ]

        imp_interactions = list(Impute("mode",True).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual([None,1], imp_interactions[0]['context'])
        self.assertEqual([None,1], imp_interactions[1]['context'])
        self.assertEqual([None,1], imp_interactions[2]['context'])
        self.assertEqual([None,1], imp_interactions[3]['context'])

    def test_impute_med_singular(self):
        interactions = [
            SimulatedInteraction(1   , [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(5   , [1], [1]),
            SimulatedInteraction(5   , [1], [1])
        ]

        imp_interactions = list(Impute("median",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(1, imp_interactions[0]['context'])
        self.assertEqual(5, imp_interactions[1]['context'])
        self.assertEqual(5, imp_interactions[2]['context'])
        self.assertEqual(5, imp_interactions[3]['context'])

    def test_impute_mode_singular(self):
        interactions = [
            SimulatedInteraction(1   , [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(5   , [1], [1]),
            SimulatedInteraction(5   , [1], [1])
        ]

        imp_interactions = list(Impute("mode",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(1, imp_interactions[0]['context'])
        self.assertEqual(5, imp_interactions[1]['context'])
        self.assertEqual(5, imp_interactions[2]['context'])
        self.assertEqual(5, imp_interactions[3]['context'])

    def test_impute_med_singular_indicator(self):
        interactions = [
            SimulatedInteraction(1   , [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(5   , [1], [1]),
            SimulatedInteraction(5   , [1], [1])
        ]

        imp_interactions = list(Impute("median",True).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual([1,0], imp_interactions[0]['context'])
        self.assertEqual([5,1], imp_interactions[1]['context'])
        self.assertEqual([5,0], imp_interactions[2]['context'])
        self.assertEqual([5,0], imp_interactions[3]['context'])

    def test_impute_mode_dict(self):
        interactions = [
            SimulatedInteraction({1:1   }, [1], [1]),
            SimulatedInteraction({1:None}, [1], [1]),
            SimulatedInteraction({1:5   }, [1], [1]),
            SimulatedInteraction({1:5   }, [1], [1])
        ]

        imp_interactions = list(Impute("mode",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual({1:1}, imp_interactions[0]['context'])
        self.assertEqual({1:5}, imp_interactions[1]['context'])
        self.assertEqual({1:5}, imp_interactions[2]['context'])
        self.assertEqual({1:5}, imp_interactions[3]['context'])

    def test_impute_median_dict(self):
        interactions = [
            SimulatedInteraction({1:1   }, [1], [1]),
            SimulatedInteraction({1:None}, [1], [1]),
            SimulatedInteraction({1:5   }, [1], [1]),
            SimulatedInteraction({1:5   }, [1], [1])
        ]

        imp_interactions = list(Impute("median",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual({1:1}, imp_interactions[0]['context'])
        self.assertEqual({1:5}, imp_interactions[1]['context'])
        self.assertEqual({1:5}, imp_interactions[2]['context'])
        self.assertEqual({1:5}, imp_interactions[3]['context'])

    def test_impute_mode_dict_indicator(self):
        interactions = [
            SimulatedInteraction({1:1   }, [1], [1]),
            SimulatedInteraction({1:None}, [1], [1]),
            SimulatedInteraction({1:5   }, [1], [1]),
            SimulatedInteraction({1:5   }, [1], [1])
        ]

        imp_interactions = list(Impute("mode",True).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual({1:1,"1_is_missing":0}, imp_interactions[0]['context'])
        self.assertEqual({1:5,"1_is_missing":1}, imp_interactions[1]['context'])
        self.assertEqual({1:5,"1_is_missing":0}, imp_interactions[2]['context'])
        self.assertEqual({1:5,"1_is_missing":0}, imp_interactions[3]['context'])

    def test_impute_mode_with_str(self):
        interactions = [
            SimulatedInteraction((7   , 2   , "A" ), [1], [1]),
            SimulatedInteraction((7   , 2   , "A" ), [1], [1]),
            SimulatedInteraction((None, None, None), [1], [1]),
            SimulatedInteraction((8   , 3   , "A" ), [1], [1])
        ]

        imp_interactions = list(Impute("mode",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertSequenceEqual((7, 2, "A"), imp_interactions[0]['context'])
        self.assertSequenceEqual((7, 2, "A"), imp_interactions[1]['context'])
        self.assertSequenceEqual((7, 2, "A"), imp_interactions[2]['context'])
        self.assertSequenceEqual((8, 3, "A"), imp_interactions[3]['context'])

    def test_impute_med_with_str(self):
        interactions = [
            SimulatedInteraction((7   , 2   , "A" ), [1], [1]),
            SimulatedInteraction((7   , 2   , "A" ), [1], [1]),
            SimulatedInteraction((None, None, None), [1], [1]),
            SimulatedInteraction((8   , 3   , "A" ), [1], [1])
        ]

        imp_interactions = list(Impute("median",False).filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertSequenceEqual((7, 2, "A"), imp_interactions[0]['context'])
        self.assertSequenceEqual((7, 2, "A"), imp_interactions[1]['context'])
        self.assertSequenceEqual((7, 2, None), imp_interactions[2]['context'])
        self.assertSequenceEqual((8, 3, "A"), imp_interactions[3]['context'])

    def test_params(self):
        self.assertEqual({ "impute_stat": "median", "impute_using": 2, "impute_indicator":False }, Impute("median",False,2).params)

class Binary_Tests(unittest.TestCase):
    def test_binary_not_discrete(self):
        interactions = [
            SimulatedInteraction((7,2), [], L1Reward(1)),
            SimulatedInteraction((1,9), [], L1Reward(1)),
            SimulatedInteraction((8,3), [], L1Reward(1))
        ]

        with self.assertWarns(UserWarning) as w:
           interactions = list(Binary().filter(interactions))

        self.assertEqual(L1Reward(1), interactions[0]['rewards'])
        self.assertEqual(L1Reward(1), interactions[1]['rewards'])
        self.assertEqual(L1Reward(1), interactions[2]['rewards'])

    def test_binary_discrete(self):
        interactions = [
            SimulatedInteraction((7,2), [1,2], [.2,.3]),
            SimulatedInteraction((1,9), [1,2], [.1,.5]),
            SimulatedInteraction((8,3), [1,2], [.5,.2])
        ]

        binary_interactions = list(Binary().filter(interactions))

        self.assertEqual([0,1], binary_interactions[0]['rewards'])
        self.assertEqual([0,1], binary_interactions[1]['rewards'])
        self.assertEqual([1,0], binary_interactions[2]['rewards'])

    def test_binary_list_rewards(self):
        interactions = [
            {'context':(7,2), 'actions':[1,2], 'rewards':[.2,.3]},
            {'context':(1,9), 'actions':[1,2], 'rewards':[.1,.5]},
            {'context':(8,3), 'actions':[1,2], 'rewards':[.5,.2]}
        ]

        binary_interactions = list(Binary().filter(interactions))
        self.assertEqual([0,1], binary_interactions[0]['rewards'])
        self.assertEqual([0,1], binary_interactions[1]['rewards'])
        self.assertEqual([1,0], binary_interactions[2]['rewards'])

    def test_binary_callable_rewards(self):
        interactions = [
            {'context':(7,2), 'actions':[1,2], 'rewards':DiscreteReward([1,2],[.2,.3])},
            {'context':(1,9), 'actions':[1,2], 'rewards':DiscreteReward([1,2],[.5,.1])},
        ]

        binary_interactions = list(Binary().filter(interactions))
        self.assertEqual(BinaryReward(2), binary_interactions[0]['rewards'])
        self.assertEqual(BinaryReward(1), binary_interactions[1]['rewards'])

    def test_params(self):
        self.assertEqual({'binary':True}, Binary().params)

class Sparsify_Tests(unittest.TestCase):
    def test_simulated_missing_context(self):
        sparse_interactions = list(Sparsify(action=True).filter([{'actions':[1,2],'rewards':[0,1]}]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual(sparse_interactions, [{'actions':[{'action':1},{'action':2}], 'rewards':[0,1]}])

    def test_simulated_no_context_and_action(self):
        sparse_interactions = list(Sparsify(action=True).filter([SimulatedInteraction(None, [1,2], [0,1]) ]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual(None, sparse_interactions[0]['context'])
        self.assertEqual([{'action':1},{'action':2}], sparse_interactions[0]['actions'])
        self.assertEqual([0,1], sparse_interactions[0]['rewards'])

    def test_simulated_str_context(self):
        sparse_interactions = list(Sparsify().filter([SimulatedInteraction("a", [{1:2},{3:4}], [0,1]) ]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({'context':"a"}, sparse_interactions[0]['context'])
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0]['actions'])
        self.assertEqual([0,1], sparse_interactions[0]['rewards'])

    def test_simulated_str_not_context_not_action(self):
        sparse_interactions = list(Sparsify(context=False).filter([SimulatedInteraction("a", [{1:2},{3:4}], [0,1]) ]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual("a", sparse_interactions[0]['context'])
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0]['actions'])
        self.assertEqual([0,1], sparse_interactions[0]['rewards'])

    def test_simulated_with_sparse_actions(self):
        sparse_interactions = list(Sparsify(context=False,action=True).filter([SimulatedInteraction("a", [{1:2},{3:4}], [0,1]) ]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual("a", sparse_interactions[0]['context'])
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0]['actions'])
        self.assertEqual([0,1], sparse_interactions[0]['rewards'])

    def test_simulated_tuple_context(self):
        sparse_interactions = list(Sparsify().filter([SimulatedInteraction((1,2,3), [{1:2},{3:4}], [0,1]) ]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({'0':1,'1':2,'2':3}, sparse_interactions[0]['context'])
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0]['actions'])
        self.assertEqual([0,1], sparse_interactions[0]['rewards'])

    def test_logged_tuple_context_and_action(self):
        sparse_interactions = list(Sparsify(action=True).filter([LoggedInteraction((1,2,3), 2, 0)]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({'0':1,'1':2,'2':3}, sparse_interactions[0]['context'])
        self.assertEqual({'action':2}, sparse_interactions[0]['action'])
        self.assertEqual(0, sparse_interactions[0]['reward'])

    def test_logged_tuple_context_and_not_action(self):
        sparse_interactions = list(Sparsify().filter([LoggedInteraction((1,2,3), 2, 0)]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({'0':1,'1':2,'2':3}, sparse_interactions[0]['context'])
        self.assertEqual(2, sparse_interactions[0]['action'])
        self.assertEqual(0, sparse_interactions[0]['reward'])

    def test_context_with_header(self):
        context = HeadDense([1,2,3],{'a':0,'b':1,'c':2})
        sparse_interactions = list(Sparsify().filter([{'context':context}]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({'a':1,'b':2,'c':3}, sparse_interactions[0]['context'])

    def test_context_without_header(self):
        context = LazyDense([1,2,3])
        sparse_interactions = list(Sparsify().filter([{'context':context}]))
        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({'0':1,'1':2,'2':3}, sparse_interactions[0]['context'])

    def test_params(self):
        self.assertEqual({'sparse_c':True, 'sparse_a':False}, Sparsify().params)

class Densify_Tests(unittest.TestCase):
    def test_dense_context(self):
        d = Densify(context=True,action=False)
        self.assertEqual(None , next(d.filter([{'context':None}]))['context'] )
        self.assertEqual(1    , next(d.filter([{'context':1}]))['context'] )
        self.assertEqual([1,2], next(d.filter([{'context':[1,2]}]))['context'] )

    def test_dense_action(self):
        d = Densify(context=True,action=True)
        self.assertEqual(None , next(d.filter([{'action':None }]))['action'] )
        self.assertEqual(1    , next(d.filter([{'action':1    }]))['action'] )
        self.assertEqual([1,2], next(d.filter([{'action':[1,2]}]))['action'] )

    def test_dense_actions(self):
        d = Densify(context=False,action=True)
        self.assertEqual([None,1,[1,2]], next(d.filter([{'actions':[None,1,[1,2]]}]))['actions'] )

    def test_lookup(self):
        d = Densify(2, method='lookup')
        self.assertEqual([1,0], list(next(d.filter([{'context':{'a':1}}]))['context']) )
        self.assertEqual([2,0], list(next(d.filter([{'context':{'a':2}}]))['context']) )
        self.assertEqual([0,3], list(next(d.filter([{'context':{'b':3}}]))['context']) )
        self.assertEqual([5,4], list(next(d.filter([{'context':{'b':1,'c':4,'d':5}}]))['context']) )

    def test_hashing(self):
        d = Densify(2, method='hashing')
        self.assertEqual([0,1], list(next(d.filter([{'context':{'a':1}}]))['context']) )
        self.assertEqual([0,2], list(next(d.filter([{'context':{'b':2}}]))['context']) )
        self.assertEqual([0,3], list(next(d.filter([{'context':{'c':3}}]))['context']) )
        self.assertEqual([4,1], list(next(d.filter([{'context':{'c':1,'d':4}}]))['context']) )

    def test_params(self):
        self.assertEqual({ 'dense_m': 'lookup', 'dense_n':400, 'dense_c':True, 'dense_a':False}, Densify().params)

class Noise_Tests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(list(Noise().filter([])),[])

    def test_missing(self):
        interactions = [ {'actions':[1,2],'rewards':[.2,.3]}, {'actions':[2,3],'rewards':[.1,.5]} ]
        actual = list(Noise().filter(interactions))
        self.assertEqual(actual,interactions)

    def test_default_noise(self):
        interactions = [
            SimulatedInteraction((7,), [1,2], [.2,.3]),
            SimulatedInteraction((1,), [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise().filter(interactions))

        self.assertEqual(2, len(actual_interactions))

        self.assertAlmostEqual(7.62598057113, actual_interactions[0]['context'][0],5)
        self.assertEqual      ([1,2]        , actual_interactions[0]['actions']     )
        self.assertEqual      ([.2,.3]      , actual_interactions[0]['rewards']     )

        self.assertAlmostEqual(-1.0118803935, actual_interactions[1]['context'][0],5)
        self.assertEqual      ([2,3]        , actual_interactions[1]['actions']     )
        self.assertEqual      ([.1,.5]      , actual_interactions[1]['rewards']     )

    def test_context_noise1(self):
        interactions = [
            SimulatedInteraction((7,5), [1,2], [.2,.3]),
            SimulatedInteraction((1,6), [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(context=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([8,5]  , actual_interactions[0]['context'])
        self.assertEqual([1,2]  , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3], actual_interactions[0]['rewards'])
        self.assertEqual([2,7]  , actual_interactions[1]['context'])
        self.assertEqual([2,3]  , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5], actual_interactions[1]['rewards'])

    def test_context_noise2(self):
        interactions = [
            SimulatedInteraction(None, [1,2], [.2,.3]),
            SimulatedInteraction(None, [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(context=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual(None   , actual_interactions[0]['context'])
        self.assertEqual([1,2]  , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3], actual_interactions[0]['rewards'])
        self.assertEqual(None   , actual_interactions[1]['context'])
        self.assertEqual([2,3]  , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5], actual_interactions[1]['rewards'])

    def test_context_noise_sparse(self):
        interactions = [
            SimulatedInteraction({'a':7, 'b':5}, [1,2], [.2,.3]),
            SimulatedInteraction({'a':1, 'b':6}, [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(context=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2             , len(actual_interactions))
        self.assertEqual({'a':8, 'b':5}, actual_interactions[0]['context'])
        self.assertEqual([1,2]         , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3]       , actual_interactions[0]['rewards'])
        self.assertEqual({'a':2, 'b':7}, actual_interactions[1]['context'])
        self.assertEqual([2,3]         , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5]       , actual_interactions[1]['rewards'])

    def test_action_noise1(self):
        interactions = [
            SimulatedInteraction([7], [1,2], [.2,.3]),
            SimulatedInteraction([1], [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(action=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]    , actual_interactions[0]['context'])
        self.assertEqual([2,2]  , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3], actual_interactions[0]['rewards'])
        self.assertEqual([1]    , actual_interactions[1]['context'])
        self.assertEqual([3,4]  , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5], actual_interactions[1]['rewards'])

    def test_action_noise2(self):
        interactions = [
            SimulatedInteraction([7], [[1],[2]], [.2,.3]),
            SimulatedInteraction([1], [[2],[3]], [.1,.5]),
        ]

        actual_interactions = list(Noise(action=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2          , len(actual_interactions))
        self.assertEqual([7]        , actual_interactions[0]['context'])
        self.assertEqual([[2],[2]]  , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3]    , actual_interactions[0]['rewards'])
        self.assertEqual([1]        , actual_interactions[1]['context'])
        self.assertEqual([[3],[4]]  , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5]    , actual_interactions[1]['rewards'])

    def test_action_noise3(self):
        interactions = [
            SimulatedInteraction((7,), ['A','B'], [.2,.3]),
            SimulatedInteraction((1,), ['A','B'], [.1,.5]),
        ]

        actual_interactions = list(Noise().filter(interactions))

        self.assertEqual(2        , len(actual_interactions))
        self.assertEqual(['A','B'], actual_interactions[0]['actions'])
        self.assertEqual(['A','B'], actual_interactions[1]['actions'])

    def test_reward_noise1(self):
        interactions = [
            SimulatedInteraction([7], [1,2], [.2,.3]),
            SimulatedInteraction([1], [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(reward=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2        , len(actual_interactions))
        self.assertEqual([7]      , actual_interactions[0]['context'])
        self.assertEqual([1,2]    , actual_interactions[0]['actions'])
        self.assertEqual([1.2,.3] , actual_interactions[0]['rewards'])
        self.assertEqual([1]      , actual_interactions[1]['context'])
        self.assertEqual([2,3]    , actual_interactions[1]['actions'])
        self.assertEqual([1.1,1.5], actual_interactions[1]['rewards'])

    def test_reward_noise2(self):
        interactions = [
            {'context':[7], 'actions':[1,2], 'rewards':[.2,.3]},
            {'context':[1], 'actions':[2,3], 'rewards':[.1,.5]},
        ]

        actual_interactions = list(Noise(reward=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2        , len(actual_interactions))
        self.assertEqual([7]      , actual_interactions[0]['context'])
        self.assertEqual([1,2]    , actual_interactions[0]['actions'])
        self.assertEqual([1.2,.3] , actual_interactions[0]['rewards'])
        self.assertEqual([1]      , actual_interactions[1]['context'])
        self.assertEqual([2,3]    , actual_interactions[1]['actions'])
        self.assertEqual([1.1,1.5], actual_interactions[1]['rewards'])

    def test_tuple_noise(self):
        interactions = [
            {'context':[7], 'actions':[1,2], 'rewards':[.2,.3]},
            {'context':[1], 'actions':[2,3], 'rewards':[.1,.5]},
        ]

        actual_interactions = list(Noise(context=('g',0,1),action=('i',0,1),reward=('i',0,1),seed=3).filter(interactions))

        self.assertAlmostEqual(5.61267      , actual_interactions[0]['context'][0],5)
        self.assertEqual      ([2,2]        , actual_interactions[0]['actions']     )
        self.assertEqual      ([.2,1.3]     , actual_interactions[0]['rewards']     )
        self.assertAlmostEqual(1.56358      , actual_interactions[1]['context'][0],5)
        self.assertEqual      ([3,4]        , actual_interactions[1]['actions']     )
        self.assertEqual      ([.1,.5]      , actual_interactions[1]['rewards']     )

    def test_tuple_noise2(self):
        interactions = [
            {'context':[7], 'actions':[1,2], 'rewards':[.2,.3]},
            {'context':[1], 'actions':[2,3], 'rewards':[.1,.5]},
        ]

        actual_interactions = list(Noise(context=(0,1),action=('i',0,1),reward=('i',0,1),seed=3).filter(interactions))

        self.assertAlmostEqual(5.61267      , actual_interactions[0]['context'][0],5)
        self.assertEqual      ([2,2]        , actual_interactions[0]['actions']     )
        self.assertEqual      ([.2,1.3]     , actual_interactions[0]['rewards']     )
        self.assertAlmostEqual(1.56358      , actual_interactions[1]['context'][0],5)
        self.assertEqual      ([3,4]        , actual_interactions[1]['actions']     )
        self.assertEqual      ([.1,.5]      , actual_interactions[1]['rewards']     )


    def test_noise_repeatable(self):
        interactions = [
            SimulatedInteraction([7], [1,2], [.2,.3]),
            SimulatedInteraction([1], [2,3], [.1,.5]),
        ]

        noise_filter = Noise(action=lambda v,r: v+r.randint(0,1), seed=5)

        actual_interactions = list(noise_filter.filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]    , actual_interactions[0]['context'])
        self.assertEqual([2,2]  , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3], actual_interactions[0]['rewards'])
        self.assertEqual([1]    , actual_interactions[1]['context'])
        self.assertEqual([3,4]  , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5], actual_interactions[1]['rewards'])

        actual_interactions = list(noise_filter.filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]    , actual_interactions[0]['context'])
        self.assertEqual([2,2]  , actual_interactions[0]['actions'])
        self.assertEqual([.2,.3], actual_interactions[0]['rewards'])
        self.assertEqual([1]    , actual_interactions[1]['context'])
        self.assertEqual([3,4]  , actual_interactions[1]['actions'])
        self.assertEqual([.1,.5], actual_interactions[1]['rewards'])

    def test_params(self):
        self.assertEqual({"context_noise": True, "noise_seed":1}, Noise().params)

    def test_pickle_default(self):
        self.assertEqual({"context_noise": True, "noise_seed":1}, pickle.loads(pickle.dumps(Noise())).params)

    def test_pickle_failure(self):
        with self.assertRaises(CobaException):
            pickle.dumps(Noise(lambda x,_: x))

class Riffle_Tests(unittest.TestCase):
    def test_riffle0(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        cov_interactions = list(Riffle(0).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])
        self.assertEqual((7,2), cov_interactions[0]['context'])
        self.assertEqual((1,9), cov_interactions[1]['context'])
        self.assertEqual((8,3), cov_interactions[2]['context'])

    def test_riffle1(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        cov_interactions = list(Riffle(1,seed=5).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])
        self.assertEqual((7,2), cov_interactions[0]['context'])
        self.assertEqual((8,3), cov_interactions[1]['context'])
        self.assertEqual((1,9), cov_interactions[2]['context'])

        cov_interactions = list(Riffle(1,seed=4).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])
        self.assertEqual((8,3), cov_interactions[0]['context'])
        self.assertEqual((7,2), cov_interactions[1]['context'])
        self.assertEqual((1,9), cov_interactions[2]['context'])

    def test_riffle5(self):
        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        cov_interactions = list(Riffle(5).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0]['context'])
        self.assertEqual((1,9), mem_interactions[1]['context'])
        self.assertEqual((8,3), mem_interactions[2]['context'])
        self.assertEqual((7,2), cov_interactions[0]['context'])
        self.assertEqual((1,9), cov_interactions[1]['context'])
        self.assertEqual((8,3), cov_interactions[2]['context'])

    def test_params(self):
        self.assertEqual({'riffle_spacing':2, 'riffle_seed':3}, Riffle(2,3).params)

class Flatten_Tests(unittest.TestCase):
    def test_flatten_empty(self):
        self.assertEqual(list(Flatten().filter([])),[])

    def test_flatten_with_missing_fields(self):
        flatten_filter = Flatten()

        interactions = [ {'context':(7,(1,0))}, {'context':(1,(0,1))} ]
        actual_interactions = list(flatten_filter.filter(interactions))
        self.assertEqual(actual_interactions, [{'context':(7,1,0)},{'context':(1,0,1)}])

        interactions = [ {'actions':[(1,("d",1)),(1,("j",1))]}, {'actions':[(1,("g",2)),(1,("l",1))]} ]
        actual_interactions = list(flatten_filter.filter(interactions))
        self.assertEqual(actual_interactions, [{'actions':[(1,"d",1),(1,"j",1)]},{'actions':[(1,"g",2),(1,"l",1)]}])

        interactions = [ {'d':[(1,("d",1)),(1,("j",1))]}, {'d':[(1,("g",2)),(1,("l",1))]} ]
        actual_interactions = list(flatten_filter.filter(interactions))
        self.assertEqual(actual_interactions, [ {'d':[(1,("d",1)),(1,("j",1))]}, {'d':[(1,("g",2)),(1,("l",1))]} ])

    def test_flatten_context(self):
        interactions = [
            SimulatedInteraction((7,(1,0)), [(1,"def"),2], [.2,.3]),
            SimulatedInteraction((1,(0,1)), [(1,"ghi"),3], [.1,.5]),
        ]

        flatten_filter = Flatten()
        actual_interactions = list(flatten_filter.filter(interactions))

        self.assertEqual(2            , len(actual_interactions))
        self.assertEqual((7,1,0)      , actual_interactions[0]['context'])
        self.assertEqual([(1,"def"),2], actual_interactions[0]['actions'])
        self.assertEqual([.2,.3]      , actual_interactions[0]['rewards'])
        self.assertEqual((1,0,1)      , actual_interactions[1]['context'])
        self.assertEqual([(1,"ghi"),3], actual_interactions[1]['actions'])
        self.assertEqual([.1,.5]      , actual_interactions[1]['rewards'])

    def test_flatten_actions(self):
        interactions = [
            GroundedInteraction((7,1,0), [(1,("d",1)),(1,("j",1))], [.2,.3],[.3,.4]),
            GroundedInteraction((1,0,1), [(1,("g",2)),(1,("l",1))], [.1,.5],[.2,.6]),
        ]

        flatten_filter = Flatten()
        actual_interactions = list(flatten_filter.filter(interactions))

        self.assertEqual(2                    , len(actual_interactions))
        self.assertEqual((7,1,0)              , actual_interactions[0]['context'])
        self.assertEqual([(1,"d",1),(1,"j",1)], actual_interactions[0]['actions'])
        self.assertEqual([.2,.3]              , actual_interactions[0]['rewards'])
        self.assertEqual([.3,.4]              , actual_interactions[0]['feedbacks'])
        self.assertEqual((1,0,1)              , actual_interactions[1]['context'])
        self.assertEqual([(1,"g",2),(1,"l",1)], actual_interactions[1]['actions'])
        self.assertEqual([.1,.5]              , actual_interactions[1]['rewards'])
        self.assertEqual([.2,.6]              , actual_interactions[1]['feedbacks'])

    def test_flatten_rewards_feedbacks(self):
        actions1 = [(1,("d",1)),(1,("j",1))]
        actions2 = [(1,("g",2)),(1,("l",1))]
        interactions = [
            GroundedInteraction((7,1,0),actions1,DiscreteReward(actions1,[.2,.3]),DiscreteReward(actions1,[.3,.4])),
            GroundedInteraction((1,0,1),actions2,DiscreteReward(actions2,[.1,.5]),DiscreteReward(actions2,[.2,.6])),
        ]

        flatten_filter = Flatten()
        actual_interactions = list(flatten_filter.filter(interactions))

        expected_a1 = [(1,"d",1),(1,"j",1)]
        expected_a2 = [(1,"g",2),(1,"l",1)]

        self.assertEqual(2                                  , len(actual_interactions))
        self.assertEqual((7,1,0)                            , actual_interactions[0]['context'])
        self.assertEqual(expected_a1                        , actual_interactions[0]['actions'])
        self.assertEqual(DiscreteReward(expected_a1,[.2,.3]), actual_interactions[0]['rewards'])
        self.assertEqual(DiscreteReward(expected_a1,[.3,.4]), actual_interactions[0]['feedbacks'])
        self.assertEqual((1,0,1)                            , actual_interactions[1]['context'])
        self.assertEqual(expected_a2                        , actual_interactions[1]['actions'])
        self.assertEqual(DiscreteReward(expected_a2,[.1,.5]), actual_interactions[1]['rewards'])
        self.assertEqual(DiscreteReward(expected_a2,[.2,.6]), actual_interactions[1]['feedbacks'])

    def test_params(self):
        self.assertEqual({'flat':True}, Flatten().params)

class Params_Tests(unittest.TestCase):
    def test_params(self):
        params_filter = Params({'a':123})
        self.assertEqual({'a':123}, params_filter.params)
        self.assertEqual(1, params_filter.filter(1))

class Grounded_Tests(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(list(Grounded(10,5,4,2,4).filter([])),[])

    def test_missing_context(self):
        interactions = [
            {'actions':[1,2,3],'rewards':[1,0,0]},
            {'actions':[1,2,3],'rewards':[0,1,0]},
        ]
        to_igl_filter    = Grounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        feedbacks_0 = [igl_interactions[0]['feedbacks'](a) for a in [1,2,3]]

        self.assertNotIn('context', igl_interactions[0])
        self.assertEqual(True,igl_interactions[0]['isnormal'])
        self.assertEqual(4, igl_interactions[0]['userid'])
        self.assertEqual([(0,), (2,), (3,)], feedbacks_0)

        feedbacks_1 = [igl_interactions[1]['feedbacks'](a) for a in [1,2,3]]

        self.assertNotIn('context', igl_interactions[1])
        self.assertEqual(True,igl_interactions[1]['isnormal'])
        self.assertEqual(2, igl_interactions[1]['userid'])
        self.assertEqual([(3,),(0,),(3,)], feedbacks_1)

    def test_bad_users(self):
        with self.assertRaises(CobaException) as e:
            Grounded(10,20,5,2,1)
        self.assertIn("Igl conversion can't have more normal users", str(e.exception))

    def test_bad_words(self):
        with self.assertRaises(CobaException) as e:
            Grounded(10,5,5,10,1)
        self.assertIn("Igl conversion can't have more good words", str(e.exception))

    def test_bad_n_users(self):
        with self.assertRaises(CobaException) as e:
            Grounded(10.5,20,5,2,1)
        self.assertIn("n_users must be a whole number and not 10.5.", str(e.exception))

    def test_n_users_float_but_integer(self):
        filter = Grounded(10.0,5.0,5.0,2.0,1)
        self.assertEqual(filter._n_users, 10)

    def test_interaction_count(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[1,0,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
        ]
        to_igl_filter    = Grounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        self.assertEqual(2, len(igl_interactions))

    def test_fixed_for_seed(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[1,0,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
        ]
        to_igl_filter    = Grounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        feedbacks_0 = [igl_interactions[0]['feedbacks'](a) for a in [1,2,3]]

        self.assertEqual(True,igl_interactions[0]['isnormal'])
        self.assertEqual(4, igl_interactions[0]['userid'])
        self.assertEqual([(0,), (2,), (3,)], feedbacks_0)

        feedbacks_1 = [igl_interactions[1]['feedbacks'](a) for a in [1,2,3]]

        self.assertEqual(True,igl_interactions[1]['isnormal'])
        self.assertEqual(2, igl_interactions[1]['userid'])
        self.assertEqual([(3,),(0,),(3,)], feedbacks_1)

    def test_01_reward(self):
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[1,0,0]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0]['rewards'], [1,0,0])

    def test_not_01_reward(self):
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[0,.2,.5]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0]['rewards'], BinaryReward(3))

    def test_list_reward(self):
        sim_interactions = [ {'context':0,'actions':[1,2,3],'rewards':[0,.2,.5]} ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0]['rewards'], BinaryReward(3))

    def test_normal_bizzaro_users(self):
        to_igl           = Grounded(10,5,4,2,1)
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[0,.2,.5]) ] * 6000
        igl_interactions = list(to_igl.filter(sim_interactions))

        c = Counter()
        for i in igl_interactions:
            self.assertEqual(i['isnormal'],i['userid'] in to_igl.normalids)
            c.update(['normal' if i['isnormal'] else 'bizzaro'])

        self.assertAlmostEqual(1,c['normal']/c['bizzaro'],1)

    def test_feedbacks_repr(self):
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[1,0,0]) ]
        igl_interactions = next(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(repr(igl_interactions['feedbacks']), 'GroundedFeedback(1)')

    def test_feedbacks(self):
        n_words          = 50
        to_igl           = Grounded(10,5,n_words,25,1)
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[0,.2,.5]) ] * 50000
        igl_interactions = list(to_igl.filter(sim_interactions))

        c = Counter()
        for i in igl_interactions:
            feedbacks = [i['feedbacks'](a) for a in i['actions'] ]
            rewards   = [i['rewards'](a)   for a in i['actions'] ]

            for word,reward in zip(feedbacks,rewards):
                if reward == 1: self.assertEqual(i['isnormal'], word[0] in to_igl.goodwords)
                if reward == 0: self.assertEqual(i['isnormal'], word[0] in to_igl.badwords)

            c.update(feedbacks)

        self.assertEqual(len(c),n_words)
        self.assertLess(abs(1-min(c.values())/max(c.values())), .15)

    def test_callable_rewards(self):
        n_words          = 50
        to_igl           = Grounded(10,5,n_words,25,1)
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],DiscreteReward([1,2,3],[0,.2,.5])) ] * 50000
        igl_interactions = list(to_igl.filter(sim_interactions))

        c = Counter()
        for i in igl_interactions:
            feedbacks = [i['feedbacks'](a) for a in i['actions'] ]
            rewards   = [i['rewards'](a)   for a in i['actions'] ]

            for word,reward in zip(feedbacks,rewards):
                if reward == 1: self.assertEqual(i['isnormal'], word[0] in to_igl.goodwords)
                if reward == 0: self.assertEqual(i['isnormal'], word[0] in to_igl.badwords)

            c.update(feedbacks)

        self.assertEqual(len(c),n_words)
        self.assertLess(abs(1-min(c.values())/max(c.values())), .15)


    def test_feedback_repeatable(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[.1,0,0]),
            SimulatedInteraction(1,[1,2,3],[0,.1,0]),
            SimulatedInteraction(2,[1,2,3],[0,0,.1]),
        ] * 3000
        to_igl_filter    = Grounded(10,5,4,2,1)
        igl_interactions = list(to_igl_filter.filter(interactions))

        for interaction in igl_interactions:
            f1 = [interaction['feedbacks'](a) for a in interaction['actions'] ]
            f2 = [interaction['feedbacks'](a) for a in interaction['actions'] ]
            self.assertEqual(f1,f2)

    def test_params(self):
        params = Grounded(10,5,4,2,1).params
        self.assertEqual(10, params['n_users'])
        self.assertEqual( 5, params['n_normal'])
        self.assertEqual( 4, params['n_words'])
        self.assertEqual( 2, params['n_good'])
        self.assertEqual( 1, params['igl_seed'])

class Repr_Tests(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual([],list(Repr('onehot','onehot').filter([])))

    def test_context_none(self):
        out = next(Repr('onehot','onehot').filter([{'context':None,'actions':[[Categorical('1',['1','2'])],[Categorical('2',['1','2'])]],'rewards':[1,2]}]))
        self.assertEqual(out,{'context':None, 'actions':[[1,0],[0,1]], 'rewards':[1,2] })

    def test_context_missing(self):
        out = next(Repr('onehot','onehot').filter([{'actions':[[Categorical('1',['1','2'])],[Categorical('2',['1','2'])]],'rewards':[1,2]}]))
        self.assertEqual(out,{'actions':[[1,0],[0,1]], 'rewards':[1,2] })

    def test_repr_none_none(self):
        out = next(Repr(None,None).filter([SimulatedInteraction(Categorical('1',['1','2']),[1,2],[1,2])]))
        self.assertEqual(Categorical('1',['1','2']),out['context'])
        self.assertEqual([1,2],out['actions'])
        self.assertEqual([1,2],out['rewards'])

    def test_no_categorical(self):
        out = next(Repr('onehot','onehot').filter([SimulatedInteraction([1,2,3],[1,2],[1,2])]))
        self.assertEqual([1,2,3],out['context'])
        self.assertEqual([1,2]  ,out['actions'])
        self.assertEqual([1,2]  ,out['rewards'])

    def test_context_categorical_onehot(self):
        out = next(Repr('onehot','onehot').filter([SimulatedInteraction([1,2,Categorical('1',['1','2'])],[1,2],[1,2])]))
        self.assertEqual([1,2,1,0],out['context'])
        self.assertEqual([1,2]    ,out['actions'])
        self.assertEqual([1,2]    ,out['rewards'])

    def test_context_categorical_unflat(self):
        out = next(Repr('onehot_tuple','onehot').filter([SimulatedInteraction([1,2,Categorical('1',['1','2'])],[1,2],[1,2])]))
        self.assertEqual([1,2,(1,0)],out['context'])
        self.assertEqual([1,2]      ,out['actions'])
        self.assertEqual([1,2]      ,out['rewards'])

    def test_context_categorical_string(self):
        out = next(Repr('string','onehot').filter([SimulatedInteraction([Categorical('1',['1','2'])],[1,2],[1,2])]))
        self.assertEqual(['1'],out['context'])
        self.assertNotIsInstance(out['context'][0],Categorical)
        self.assertEqual([1,2],out['actions'])
        self.assertEqual([1,2],out['rewards'])

    def test_context_categorical_value_string(self):
        out = next(Repr('string','onehot').filter([SimulatedInteraction(Categorical('1',['1','2']),[1,2],[1,2])]))
        self.assertEqual('1',out['context'])
        self.assertNotIsInstance(out['context'],Categorical)
        self.assertEqual([1,2],out['actions'])
        self.assertEqual([1,2],out['rewards'])

    def test_context_categorical_value_onehot(self):
        out = next(Repr('onehot','onehot').filter([SimulatedInteraction(Categorical('1',['1','2']),[1,2],[1,2])]))
        self.assertEqual((1,0),out['context'])
        self.assertEqual([1,2],out['actions'])
        self.assertEqual([1,2],out['rewards'])

    def test_actions_categorical_to_onehot_tuple_discrete_reward(self):
        context = [1,2,3]
        actions = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        rewards = DiscreteReward(actions,[1,2])

        out = next(Repr('onehot','onehot_tuple').filter([{'context':context,'actions':actions,'rewards':rewards}]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual(out['rewards']((1,0)), 1)
        self.assertEqual(out['rewards']((0,1)), 2)

    def test_actions_categorical_to_onehot_tuple_binary_reward(self):
        context = [1,2,3]
        actions = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        rewards = BinaryReward(actions[1])

        out = next(Repr('onehot','onehot_tuple').filter([{'context':context,'actions':actions,'action':Categorical('1',['1','2']),'rewards':rewards}]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual((1,0)        , out['action' ])
        self.assertEqual(out['rewards'], BinaryReward((0,1)))

    def test_actions_categorical_value_onehot_tuple(self):
        out = next(Repr('onehot','onehot_tuple').filter([SimulatedInteraction([1,2,3],[Categorical('1',['1','2']),Categorical('2',['1','2'])],[1,2])]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual(out['rewards'], [1,2])

    def test_actions_categorical_value_onehot(self):
        out = next(Repr('onehot','onehot').filter([SimulatedInteraction([1,2,3],[Categorical('1',['1','2']),Categorical('2',['1','2'])],[1,2])]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual(out['rewards'], [1,2])

    def test_actions_categorical_value_string(self):
        out = next(Repr('onehot','string').filter([SimulatedInteraction([1,2,3],[Categorical('1',['1','2']),Categorical('2',['1','2'])],[1,2])]))
        self.assertEqual([1,2,3]  , out['context'])
        self.assertEqual(['1','2'], out['actions'])
        self.assertEqual(out['rewards'], [1,2])

    def test_actions_categorical_dense_onehot_tuple_with_list(self):
        out = next(Repr('onehot','onehot_tuple').filter([SimulatedInteraction([1,2,3],[[Categorical('1',['1','2']),[1]],[Categorical('2',['1','2']),[2]]],[1,2])]))
        self.assertEqual([1,2,3]                  , out['context'])
        self.assertEqual([[(1,0),[1]],[(0,1),[2]]], out['actions'])
        self.assertEqual(out['rewards'], [1,2])

    def test_actions_categorical_dense_onehot_tuple(self):
        out = next(Repr('onehot','onehot_tuple').filter([SimulatedInteraction([1,2,3],[[Categorical('1',['1','2'])],[Categorical('2',['1','2'])]],[1,2])]))
        self.assertEqual([1,2,3]          , out['context'])
        self.assertEqual([[(1,0)],[(0,1)]], out['actions'])
        self.assertEqual(out['rewards'], [1,2])

    def test_actions_categorical_dense_onehot(self):
        actions = [[Categorical('1',['1','2'])],[Categorical('2',['1','2'])]]
        out = next(Repr('onehot','onehot').filter([{'context':[1,2,3],'actions':actions,'rewards':DiscreteReward(actions,[1,2])}]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([[1,0],[0,1]], out['actions'])
        self.assertEqual(out['rewards']([1,0]), 1)
        self.assertEqual(out['rewards']([0,1]), 2)

    def test_actions_categorical_dense_onehot_rewards_list(self):
        out = next(Repr('onehot','onehot').filter([{'context':[1,2,3],'actions':[[Categorical('1',['1','2'])],[Categorical('2',['1','2'])]],'rewards':[1,2]}]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([[1,0],[0,1]], out['actions'])
        self.assertEqual(out['rewards'], [1,2])

    def test_actions_categorical_dense_string(self):
        out = next(Repr('onehot','string').filter([SimulatedInteraction([1,2,3],[[Categorical('1',['1','2'])],[Categorical('2',['1','2'])]],[1,2])]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([['1'],['2']], out['actions'])
        self.assertFalse(isinstance(out['actions'][0][0],Categorical))
        self.assertEqual(out['rewards'],[1,2])

    def test_actions_categorical_with_feedbacks(self):
        out = next(Repr('onehot','onehot_tuple').filter([GroundedInteraction([1,2,3],[Categorical('1',['1','2']),Categorical('2',['1','2'])],[1,2],[3,4])]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual(out['rewards'],[1,2])
        self.assertEqual(out['feedbacks'],[3,4])

    def test_actions_categorical_with_only_feedbacks(self):
        actions = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        out = next(Repr('onehot','onehot_tuple').filter([{'context':[1,2,3],'actions':actions,'feedbacks':DiscreteReward(actions,[3,4])}]))
        self.assertEqual([1,2,3]      , out['context'])
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual(out['feedbacks']((1,0)), 3)
        self.assertEqual(out['feedbacks']((0,1)), 4)

    def test_actions_categorical_str(self):
        out = next(Repr('onehot','string').filter([GroundedInteraction([1,2,3],[Categorical('1',['1','2']),Categorical('2',['1','2'])],[1,2],[3,4])]))
        self.assertEqual([1,2,3]  , out['context'])
        self.assertEqual(['1','2'], out['actions'])
        self.assertEqual(out['rewards'], [1,2])
        self.assertEqual(out['feedbacks'], [3,4])
        self.assertNotIsInstance(out['actions'][0],Categorical)
        self.assertNotIsInstance(out['actions'][1],Categorical)

    def test_repr_binary_reward(self):
        out = next(Repr(None,'string').filter([{'actions':[Categorical('1',['1','2']),Categorical('2',['1','2'])], 'rewards':BinaryReward(Categorical('2',['1','2']),2)}]))
        self.assertEqual(['1','2'], out['actions'])
        self.assertEqual(out['rewards']('1'), 0)
        self.assertEqual(out['rewards']('2'), 2)

    def test_repr_discrete_reward(self):
        actions = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        out = next(Repr(None,'string').filter([{'actions':actions, 'rewards':DiscreteReward(actions,[1,2])}]))
        self.assertEqual(['1','2'], out['actions'])
        self.assertEqual(out['rewards']('1'), 1)
        self.assertEqual(out['rewards']('2'), 2)

    def test_repr_exotic_reward(self):
        def my_reward(action):
            if action == '1': return 5
            if action == '2': return 6
        actions = [Categorical('1',['1','2']),Categorical('2',['1','2'])]
        out = next(Repr(None,'onehot').filter([{'actions':actions, 'rewards':my_reward}]))
        self.assertEqual([(1,0),(0,1)], out['actions'])
        self.assertEqual(out['rewards']((1,0)), 5)
        self.assertEqual(out['rewards']((0,1)), 6)

    def test_namespaced_context_nothing(self):
        out = next(Repr('onehot',None).filter([{'context':{'a':[1,2],'b':[3,4]}}]))
        self.assertEqual(out, {'context':{'a':[1,2],'b':[3,4]}})

    def test_namespaced_context_dense_onehot(self):
        out = next(Repr('onehot',None).filter([{'context':{'a':[Categorical('b',['a','b']),Categorical('a',['a','b'])],'b':[3,4]}}]))
        self.assertEqual(out, {'context':{'a':[0,1,1,0],'b':[3,4]}})

    def test_namespaced_context_dense_onehot_tuple(self):
        out = next(Repr('onehot_tuple',None).filter([{'context':{'a':[Categorical('b',['a','b']),Categorical('a',['a','b'])],'b':[3,4]}}]))
        self.assertEqual(out, {'context':{'a':[(0,1),(1,0)],'b':[3,4]}})

    def test_namespaced_context_sparse_onehot(self):
        out = next(Repr('onehot_tuple',None).filter([{'context':{'a':{'c':Categorical('b',['a','b'])},'b':{'d':[3,4]}}}]))
        self.assertEqual(out, {'context':{'a':{'c':(0,1)},'b':{'d':[3,4]}}})

    def test_namespaced_context_sparse_onehot_tuple(self):
        out = next(Repr('onehot',None).filter([{'context':{'a':{'c':Categorical('b',['a','b'])},'b':{'d':[3,4]}}}]))
        self.assertEqual(out, {'context':{'a':{'c_1':1},'b':{'d':[3,4]}}})

class Finalize_Tests(unittest.TestCase):

    def test_empty_list(self):
        self.assertEqual(list(Finalize().filter([])),[])

    def test_dense_encodes(self):
        context = [Categorical('a',['a','b']),5]
        actions = [3,4]
        interactions = [SimulatedInteraction(context, actions,[1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,0,5])
        self.assertEqual(actual[0]['actions'], [3,4])

    def test_dense_materializes(self):
        context = LazyDense([1])
        actions = [LazyDense([2]),LazyDense([3])]
        interactions = [SimulatedInteraction(context, actions,[1,2], action=actions[0])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertIsInstance(actual[0]['context']   , list)
        self.assertIsInstance(actual[0]['actions'][0], list)
        self.assertIsInstance(actual[0]['action' ]   , list)
        self.assertEqual(actual[0]['context'], [1])
        self.assertEqual(actual[0]['actions'], [[2],[3]])
        self.assertEqual(actual[0]['action' ], [2])

    def test_sparse_materializes(self):
        context = LazySparse({1:1})
        actions = [LazySparse({1:2}),LazySparse({1:3})]
        interactions = [SimulatedInteraction(context, actions,[1,2],action=actions[0])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertIsInstance(actual[0]['context']   , dict)
        self.assertIsInstance(actual[0]['actions'][0], dict)
        self.assertIsInstance(actual[0]['action' ]   , dict)
        self.assertEqual(actual[0]['context'], {1:1})
        self.assertEqual(actual[0]['actions'], [{1:2},{1:3}])
        self.assertEqual(actual[0]['action' ], {1:2})

    def test_None_simulated(self):
        interactions = [SimulatedInteraction(None,[[1,2],[3,4]], [1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], None)
        self.assertEqual(actual[0]['actions'], [[1,2],[3,4]])

    def test_dense_simulated(self):
        interactions = [SimulatedInteraction([1,2,3],[[1,2],[3,4]], [1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['actions'], [[1,2],[3,4]])

    def test_sparse_simulated(self):
        interactions = [SimulatedInteraction({1:2},[[1,2],[3,4]], [1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertIsInstance(actual[0]['context'], dict)
        self.assertEqual(actual[0]['actions'], [[1,2],[3,4]])

    def test_sparse_actions(self):
        interactions = [SimulatedInteraction({1:2},[{1:2},{2:3}], [1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['actions'], [{1:2},{2:3}])

    def test_logged_with_actions(self):
        interactions = [LoggedInteraction([1,2,3], [1,2], 1, probability=1, actions=[[1,2],[3,4]], rewards=[1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['action'], [1,2])
        self.assertEqual(actual[0]['actions'], [[1,2],[3,4]])

    def test_logged_with_action_int(self):
        interactions = [LoggedInteraction([1,2,3], 1, 1, probability=1, actions=[1,2,3], rewards=[1,2,3])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['action'], 1)
        self.assertEqual(actual[0]['actions'], [1,2,3])

    def test_logged_tuple_actions_with_action_int(self):
        interactions = [LoggedInteraction([1,2,3], 1, 1, probability=1, actions=[(1,0,0),(0,1,0),(0,0,1)], rewards=[1,2,3])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['action'], 1)
        self.assertEqual(actual[0]['actions'], [(1,0,0),(0,1,0),(0,0,1)])

    def test_logged_sparse_actions(self):
        interactions = [LoggedInteraction([1,2,3], {1:2}, 1, probability=1, actions=[{1:2},{3:4}], rewards=[1,2])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['action'], {1:2})
        self.assertEqual(actual[0]['actions'], [{1:2},{3:4}])

    def test_logged_with_action_only(self):
        interactions = [LoggedInteraction([1,2,3], [1,2], 1, probability=1)]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['action'], [1,2])

    def test_grounded(self):
        interactions = [GroundedInteraction([1,2,3],[[1,2],[3,4]], [1,2], [3,4])]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['context'], [1,2,3])
        self.assertEqual(actual[0]['actions'], [[1,2],[3,4]])

    def test_list_rewards(self):
        context = LazySparse({1:1})
        actions = [(1,0),(0,1)]
        interactions = [{'context':context, 'actions':actions, 'rewards':[1,2]}]
        actual = list(Finalize().filter(interactions))
        self.assertEqual(len(actual),1)
        self.assertIsInstance(actual[0]['context']   , dict)
        self.assertIsInstance(actual[0]['actions'][0], tuple)
        self.assertEqual(actual[0]['context'], {1:1})
        self.assertEqual(actual[0]['actions'], actions)
        self.assertEqual(actual[0]['rewards'], DiscreteReward(actions,[1,2]))

    def test_empty_logs_once_and_caches(self):
        finalizer = Finalize()
        loggersink = ListSink()
        CobaContext.logger = BasicLogger(loggersink)
        list(finalizer.filter([]))
        list(finalizer.filter([]))
        self.assertEqual(loggersink.items, ["An environment had nothing to evaluate (this is often due to having too few interactions)."] )
        self.assertFalse(list(finalizer.filter([1,2,3])))

class Batch_Tests(unittest.TestCase):
    def test_simple(self):
        batch = Batch(3)
        batches = list(batch.filter([{'a':1,'b':2}]*4))
        self.assertEqual(batch.params,{'batch_size':3,'batch_type':'list'})
        self.assertEqual(batches[0], {'a':[1,1,1],'b':[2,2,2]})
        self.assertEqual(batches[1], {'a':[1]    ,'b':[2]})
        self.assertEqual(batches[0]['a'].is_batch, True)
        self.assertEqual(batches[0]['b'].is_batch, True)
        self.assertEqual(batches[1]['a'].is_batch, True)
        self.assertEqual(batches[1]['b'].is_batch, True)

    def test_batch_None(self):
        batch = Batch(None)
        items = list(batch.filter([{'a':1,'b':2}]*2))
        self.assertEqual(batch.params,{'batch_size':None,'batch_type':'list'})
        self.assertEqual(items,[{'a':1,'b':2}]*2)

    def test_batch_1(self):
        batch = Batch(1)
        batches = list(batch.filter([{'a':1,'b':2}]*2))
        self.assertEqual(batch.params,{'batch_size':1,'batch_type':'list'})
        self.assertEqual(batches[0], {'a':[1],'b':[2]})
        self.assertEqual(batches[1], {'a':[1],'b':[2]})
        self.assertEqual(batches[0]['a'].is_batch, True)
        self.assertEqual(batches[0]['b'].is_batch, True)
        self.assertEqual(batches[1]['a'].is_batch, True)
        self.assertEqual(batches[1]['b'].is_batch, True)

    def test_batch_2(self):
        batch = Batch(2)
        batches = list(batch.filter([{'a':1,'b':2}]*2))
        self.assertEqual(batch.params,{'batch_size':2,'batch_type':'list'})
        self.assertEqual(batches, [{'a':[1,1],'b':[2,2]}])
        self.assertEqual(batches[0]['a'].is_batch, True)
        self.assertEqual(batches[0]['b'].is_batch, True)

    def test_batch_rewards(self):
        batch = list(Batch(2).filter([{'rewards':L1Reward(2),'b':2}]*2))[0]
        self.assertEqual(batch['rewards']([1,2]), [-1,0])

    def test_batch_feedbacks(self):
        batch = list(Batch(2).filter([{'feedbacks':DiscreteReward([0,1,2],[1,2,3]),'b':2}]*2))[0]
        self.assertEqual(batch['feedbacks']([1,2]), [2,3])

    @unittest.skipUnless(PackageChecker.torch(strict=False), "This test requires pytorch")
    def test_torch(self):
        import torch
        batch = list(Batch(2,'torch').filter([{'a':1,'b':None,'rewards':L1Reward(2)}]*2))[0]
        self.assertTrue(torch.equal(batch['a'],torch.tensor([1,1])))
        self.assertEqual(batch['b'],[None,None])
        self.assertTrue(batch['a'].is_batch)
        self.assertTrue(batch['b'].is_batch)
        self.assertEqual(batch['rewards']([1,2]), [-1,0])
        self.assertTrue(torch.equal(batch['rewards'](torch.tensor([1,2])), torch.tensor([-1,0])))
        self.assertTrue(torch.equal(batch['rewards'](torch.tensor([[1],[2]])), torch.tensor([[-1],[0]])))

    def test_empty(self):
        self.assertEqual([],list(Batch(3).filter([])))

class Unbatch_Tests(unittest.TestCase):
    def test_not_batched(self):
        interaction = {'a':1,'b':2}
        unbatched = list(Unbatch().filter([interaction]*3))
        self.assertEqual(unbatched, [interaction]*3)

    def test_batched(self):
        interaction = {'a':1,'b':2}
        batched     = list(Batch(3).filter([interaction]*3))
        unbatched   = list(Unbatch().filter(batched))
        self.assertEqual(unbatched, [{'a':1,'b':2}]*3)
        batched[0]['c'] = 3
        unbatched = list(Unbatch().filter(batched))
        self.assertEqual(unbatched, [{'a':1,'b':2,'c':3}]*3)

    def test_empty(self):
        self.assertEqual(list(Unbatch().filter([])),[])

class BatchSafe_Tests(unittest.TestCase):
    def test_simple_batched(self):
        initial_rows = [{'a':1,'b':2}]*4
        expected_rows = initial_rows

        class TestFilter:
            def filter(self, actual_rows):
                actual_rows = list(actual_rows)
                assert expected_rows == actual_rows
                return actual_rows

        in_batches  = list(Batch(3).filter(initial_rows))
        out_batches = list(BatchSafe(TestFilter()).filter(in_batches))

        self.assertEqual(in_batches,out_batches)

    def test_is_fully_lazy(self):
        def generator():
            raise Exception()
            yield None

        class TestFilter:
            def filter(self, actual_rows):
                return actual_rows

        BatchSafe(TestFilter()).filter(generator())

    def test_not_batched(self):
        initial_rows = [{'a':1,'b':2}]*4
        expected_rows = initial_rows

        class TestFilter:
            def filter(self, actual_rows):
                actual_rows = list(actual_rows)
                assert expected_rows == actual_rows
                return actual_rows

        final_rows = BatchSafe(TestFilter()).filter(initial_rows)

        self.assertEqual(list(initial_rows),list(final_rows))

    def test_empty(self):
        initial_rows = []
        expected_rows = initial_rows

        class TestFilter:
            def filter(self, actual_rows):
                assert expected_rows == list(actual_rows)
                return actual_rows

        in_batches  = Batch(3).filter(initial_rows)
        out_batches = BatchSafe(TestFilter()).filter(in_batches)

        self.assertEqual(list(in_batches),list(out_batches))

    def test_str(self):
        self.assertEqual("BatchSafe(Finalize())",str(BatchSafe(Finalize())))

class Cache_Tests(unittest.TestCase):
    def test_simple_cached(self):
        initial_rows  = [{'a':1,'b':2}]*4
        cacher        = Cache(3)
        filtered_rows = list(cacher.filter(initial_rows))
        self.assertEqual(filtered_rows, initial_rows)
        self.assertIsNot(initial_rows[0],filtered_rows[0])
        self.assertIsNot(filtered_rows[0],list(cacher.filter(initial_rows))[0])

    def test_cache_peek_then_read(self):
        initial_rows  = [{'a':1,'b':2}]*4
        cacher        = Cache(3)
        rows          = peek_first(iter(cacher.filter(initial_rows)))[1]
        self.assertEqual(list(rows), initial_rows)

    def test_cache_peek_then_reread(self):
        initial_rows  = [{'a':1,'b':2}]*4
        cacher        = Cache(3)
        first,rows    = peek_first(iter(cacher.filter(initial_rows)))
        self.assertEqual(list(cacher.filter(initial_rows)), initial_rows)

    def test_empty(self):
        self.assertEqual(list(Cache(3).filter([])), [])

class Logged_Tests(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(list(Logged(FixedLearner([1,0,0])).filter([])),[])

    def test_missing_context(self):
        initial_input   = {'actions':[0,1,2], "rewards":L1Reward(1)}
        expected_output = {'action':0, "reward":-1, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1)}
        actual_output   = list(Logged(FixedLearner([1,0,0])).filter([initial_input]*2))
        self.assertEqual(actual_output, [expected_output]*2)

    def test_not_batched(self):
        initial_input = {'context':None, 'actions':[0,1,2], "rewards":L1Reward(1)}
        expected_output = {'context':None, 'action':0, "reward":-1, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1)}
        actual_output = list(Logged(FixedLearner([1,0,0])).filter([initial_input]*2))
        self.assertEqual(actual_output, [expected_output]*2)

    def test_batched(self):
        class TestLearner:
            def predict(self,*args):
                return [[1,0,0],[1,0,0]]
            def learn(self,*args):
                pass

        initial_input   = {'context':None, 'actions':[0,1,2], "rewards":L1Reward(1)}
        expected_output = [{'context':None, 'action':0, "reward":-1, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1)}]*2
        actual_output = list(Logged(TestLearner()).filter(Batch(2).filter([initial_input]*2)))
        self.assertEqual(actual_output, expected_output)

    def test_learning_info(self):
        class TestLearner:
            def predict(self,*args):
                CobaContext.learning_info['a'] = 1
                return [1,0,0]
            def learn(self,*args):
                CobaContext.learning_info['b'] = 2
                pass

        initial_input   = {'context':None, 'actions':[0,1,2], "rewards":L1Reward(1)}
        expected_output = {'context':None, 'action':0, "reward":-1, 'probability':1, 'actions':[0,1,2], "rewards":L1Reward(1), 'a':1, 'b':2}
        actual_output = list(Logged(TestLearner()).filter([initial_input]))
        self.assertEqual(actual_output, [expected_output])

    def test_bad_type(self):
        initial_input = {'context':None, "rewards":L1Reward(1)}

        with self.assertRaises(CobaException):
            list(Logged(FixedLearner([1,0,0])).filter([initial_input]*2))

    def test_params(self):
        learner = FixedLearner([1,0,0])
        logged  = Logged(learner)
        self.assertEqual(logged.params,{**learner.params,"learner":learner.params,"logged":True,"log_seed":1.23})

class Mutable_Tests(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(list(Mutable().filter([])), [])

    def test_no_context(self):
        input_interactions = [{'action':1}, {'action':2}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertEqual(input_interactions, output_interactions)

    def test_value_contexts(self):
        input_interactions = [{'context':None}, {'context':1}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertEqual(input_interactions, output_interactions)
        self.assertIsNot(input_interactions[0], output_interactions[0])
        self.assertIsNot(input_interactions[1], output_interactions[1])

    def test_lazy_dense_contexts(self):
        input_interactions = [{'context':LazyDense([1,2])}, {'context':LazyDense([3,4])}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertEqual(input_interactions, output_interactions)
        self.assertIsNot(input_interactions[0], output_interactions[0])
        self.assertIsNot(input_interactions[1], output_interactions[1])
        self.assertIsInstance(input_interactions[0]['context'], LazyDense)
        self.assertIsInstance(input_interactions[1]['context'], LazyDense)
        self.assertIsInstance(output_interactions[0]['context'], list)
        self.assertIsInstance(output_interactions[1]['context'], list)

    def test_lazy_sparse_contexts(self):
        input_interactions = [{'context':LazySparse({1:2})}, {'context':LazySparse({3:4})}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertEqual(input_interactions, output_interactions)
        self.assertIsNot(input_interactions[0], output_interactions[0])
        self.assertIsNot(input_interactions[1], output_interactions[1])
        self.assertIsInstance(input_interactions[0]['context'], LazySparse)
        self.assertIsInstance(input_interactions[1]['context'], LazySparse)
        self.assertIsInstance(output_interactions[0]['context'], dict)
        self.assertIsInstance(output_interactions[1]['context'], dict)

    def test_tuple_contexts(self):
        input_interactions = [{'context':(1,2)}, {'context':(3,4)}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertSequenceEqual(output_interactions,[{'context':[1,2]}, {'context':[3,4]}])
        self.assertIsNot(input_interactions[0], output_interactions[0])
        self.assertIsNot(input_interactions[1], output_interactions[1])
        self.assertIsInstance(input_interactions[0]['context'], tuple)
        self.assertIsInstance(input_interactions[1]['context'], tuple)
        self.assertIsInstance(output_interactions[0]['context'], list)
        self.assertIsInstance(output_interactions[1]['context'], list)

    def test_list_contexts(self):
        input_interactions = [{'context':[1,2]}, {'context':[3,4]}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertSequenceEqual(input_interactions, output_interactions)
        self.assertIsNot(input_interactions[0], output_interactions[0])
        self.assertIsNot(input_interactions[1], output_interactions[1])
        self.assertIsNot(input_interactions[0]['context'], output_interactions[0]['context'])
        self.assertIsNot(input_interactions[1]['context'], output_interactions[1]['context'])
        self.assertIsInstance(input_interactions[0]['context'], list)
        self.assertIsInstance(input_interactions[1]['context'], list)
        self.assertIsInstance(output_interactions[0]['context'], list)
        self.assertIsInstance(output_interactions[1]['context'], list)

    def test_dict_contexts(self):
        input_interactions = [{'context':{1:2}}, {'context':{3:4}}]
        output_interactions = list(Mutable().filter(input_interactions))

        self.assertSequenceEqual(input_interactions, output_interactions)
        self.assertIsNot(input_interactions[0], output_interactions[0])
        self.assertIsNot(input_interactions[1], output_interactions[1])
        self.assertIsNot(input_interactions[0]['context'], output_interactions[0]['context'])
        self.assertIsNot(input_interactions[1]['context'], output_interactions[1]['context'])
        self.assertIsInstance(input_interactions[0]['context'], dict)
        self.assertIsInstance(input_interactions[1]['context'], dict)
        self.assertIsInstance(output_interactions[0]['context'], dict)
        self.assertIsInstance(output_interactions[1]['context'], dict)

class Slice_Tests(unittest.TestCase):
    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Slice(-1,1)

        with self.assertRaises(ValueError):
            Slice(1,-1)

    def test_take_slice_0_1(self):
        items = [ 1,2,3 ]
        slice_items = list(Slice(0,1).filter(items))
        self.assertEqual([1    ], slice_items)
        self.assertEqual([1,2,3], items     )

    def test_take_slice_None_2(self):
        items = [ 1,2,3 ]
        slice_items = list(Slice(None,2).filter(items))
        self.assertEqual([1,2  ], slice_items)
        self.assertEqual([1,2,3], items     )

    def test_take_slice_2_None(self):
        items = [ 1,2,3 ]
        slice_items = list(Slice(2,None).filter(items))
        self.assertEqual([    3], slice_items)
        self.assertEqual([1,2,3], items     )

    def test_take_slice_None_None_2(self):
        items = [ 1,2,3 ]
        slice_items = list(Slice(None,None,2).filter(items))
        self.assertEqual([1,  3], slice_items)
        self.assertEqual([1,2,3], items     )

    def test_params(self):
        self.assertEqual(Slice(None,None).params, {"slice_start":None, "slice_stop":None})
        self.assertEqual(Slice(1,2).params, {"slice_start":1, "slice_stop":2})
        self.assertEqual(Slice(1,2,2).params, {"slice_start":1, "slice_stop":2, "slice_step":2})

class OpeRewards_Tests(unittest.TestCase):
    def test_simple(self):
        interactions = [{'a':1},{'b':2}]
        self.assertEqual(interactions,list(OpeRewards().filter(interactions)))

    def test_IPS(self):
        interactions = [
            {'action':1,'actions':[1,2],'reward':1  ,'probability':.5 },
            {'action':2,'actions':[1,2],'reward':.25,'probability':.25},
        ]

        new_interactions = list(OpeRewards("IPS").filter(interactions))

        self.assertEqual(new_interactions[0]['rewards'](1),2)
        self.assertEqual(new_interactions[0]['rewards'](2),0)

        self.assertEqual(new_interactions[1]['rewards'](1),0)
        self.assertEqual(new_interactions[1]['rewards'](2),1)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed.")
    def test_DM_context_missing(self):
        interactions = [
            {'action':'c','actions':['c','d'],'reward':1  ,'probability':.5 },
            {'action':'f','actions':['e','f'],'reward':.25,'probability':.25},
        ]

        new_interactions = list(OpeRewards("DM").filter(interactions))
        self.assertAlmostEqual(new_interactions[0]['rewards']('c'),.62640, places=3)
        self.assertAlmostEqual(new_interactions[0]['rewards']('d'),.31039, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards']('e'),.31039, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards']('f'),.25000, places=3)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed.")
    def test_DM_context_None(self):
        interactions = [
            {'context':None, 'action':'c','actions':['c','d'],'reward':1  ,'probability':.5 },
            {'context':None, 'action':'f','actions':['e','f'],'reward':.25,'probability':.25},
        ]

        new_interactions = list(OpeRewards("DM").filter(interactions))
        self.assertAlmostEqual(new_interactions[0]['rewards']('c'),.62640, places=3)
        self.assertAlmostEqual(new_interactions[0]['rewards']('d'),.31039, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards']('e'),.31039, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards']('f'),.25000, places=3)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed.")
    def test_DM_hashable(self):
        interactions = [
            {'action':'c','context':'a','actions':['c','d'],'reward':1  ,'probability':.5 },
            {'action':'f','context':'b','actions':['e','f'],'reward':.25,'probability':.25},
        ]

        new_interactions = list(OpeRewards("DM").filter(interactions))

        self.assertAlmostEqual(new_interactions[0]['rewards']('c'),.79699, places=3)
        self.assertAlmostEqual(new_interactions[0]['rewards']('d'),.32049, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards']('e'),.18374, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards']('f'),.25000, places=3)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed.")
    def test_DM_not_hashable(self):
        interactions = [
            {'action':['c'],'context':'a','actions':[['c'],['d']],'reward':1  ,'probability':.5 },
            {'action':['f'],'context':'b','actions':[['e'],['f']],'reward':.25,'probability':.25},
        ]

        new_interactions = list(OpeRewards("DM").filter(interactions))

        self.assertAlmostEqual(new_interactions[0]['rewards'](['c']),.79699, places=3)
        self.assertAlmostEqual(new_interactions[0]['rewards'](['d']),.32049, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards'](['e']),.18374, places=3)
        self.assertAlmostEqual(new_interactions[1]['rewards'](['f']),.25000, places=3)

    @unittest.skipUnless(PackageChecker.vowpalwabbit(strict=False), "VW is not installed.")
    def test_DR(self):
        interactions = [
            {'action':'c','context':'a','actions':['c','d'],'reward':1  ,'probability':.5 },
            {'action':'f','context':'b','actions':['e','f'],'reward':.25,'probability':.25},
        ]

        dm_interactions = list(OpeRewards("DM").filter(interactions))
        dr_interactions = list(OpeRewards("DR").filter(interactions))

        c = dm_interactions[0]['rewards']('c')
        d = dm_interactions[0]['rewards']('d')
        e = dm_interactions[1]['rewards']('e')
        f = dm_interactions[1]['rewards']('f')

        self.assertAlmostEqual(dr_interactions[0]['rewards']('c'), c+(1-c)/.5   , places=3)
        self.assertAlmostEqual(dr_interactions[0]['rewards']('d'), d+0          , places=3)
        self.assertAlmostEqual(dr_interactions[1]['rewards']('e'), e+0          , places=3)
        self.assertAlmostEqual(dr_interactions[1]['rewards']('f'), f+(.25-f)/.25, places=3)

    def test_params(self):
        self.assertEqual(OpeRewards().params,{"ope_reward":"None"})
        self.assertEqual(OpeRewards("IPS").params,{"ope_reward":"IPS"})

if __name__ == '__main__':
    unittest.main()
