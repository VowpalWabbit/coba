import pickle
import unittest

from collections import Counter
from math import isnan

from coba.contexts     import CobaContext, NullLogger
from coba.exceptions   import CobaException
from coba.environments import LoggedInteraction, SimulatedInteraction, GroundedInteraction
from coba.environments import L1Reward
from coba.environments import Sparse, Sort, Scale, Cycle, Impute, Binary, Flatten, Params
from coba.environments import Warm, Shuffle, Take, Reservoir, Where, Noise, Riffle, Grounded
from coba.environments import Finalize, HashableSeq, HashableMap

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
        self.assertEqual("{'shuffle': 1}", str(Shuffle(1)))

class Sort_Tests(unittest.TestCase):

    def test_sort1_logged(self):

        interactions = [
            LoggedInteraction((7,2), 1, 1),
            LoggedInteraction((1,9), 1, 1),
            LoggedInteraction((8,3), 1, 1)
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0]).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((1,9), srt_interactions[0].context)
        self.assertEqual((7,2), srt_interactions[1].context)
        self.assertEqual((8,3), srt_interactions[2].context)

    def test_sort1(self):

        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0]).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((1,9), srt_interactions[0].context)
        self.assertEqual((7,2), srt_interactions[1].context)
        self.assertEqual((8,3), srt_interactions[2].context)

    def test_sort2(self):

        interactions = [
            SimulatedInteraction((1,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((1,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0,1]).filter(mem_interactions))

        self.assertEqual((1,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((1,3), mem_interactions[2].context)

        self.assertEqual((1,2), srt_interactions[0].context)
        self.assertEqual((1,3), srt_interactions[1].context)
        self.assertEqual((1,9), srt_interactions[2].context)

    def test_sort3(self):
        interactions = [
            SimulatedInteraction((1,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((1,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort(*[0,1]).filter(mem_interactions))

        self.assertEqual((1,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((1,3), mem_interactions[2].context)

        self.assertEqual((1,2), srt_interactions[0].context)
        self.assertEqual((1,3), srt_interactions[1].context)
        self.assertEqual((1,9), srt_interactions[2].context)

    def test_params(self):
        self.assertEqual({'sort':'*'}, Sort().params)
        self.assertEqual({'sort':[0]}, Sort(0).params)
        self.assertEqual({'sort':[0]}, Sort([0]).params)
        self.assertEqual({'sort':[1,2]}, Sort([1,2]).params)

    def test_str(self):
        self.assertEqual("{'sort': [0]}", str(Sort([0])))

class Take_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Take(-1)

        with self.assertRaises(ValueError):
            Take('A')

        with self.assertRaises(ValueError):
            Take((-1,5))

    def test_take_exact_1(self):

        items = [ 1,2,3 ]
        take_items = list(Take(1).filter(items))

        self.assertEqual([1    ], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_exact_2(self):
        items = [ 1,2,3 ]
        take_items = list(Take(2).filter(items))

        self.assertEqual([1,2  ], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_exact_3(self):
        items = [ 1,2,3 ]
        take_items = list(Take(3).filter(items))

        self.assertEqual([1,2,3], take_items)
        self.assertEqual([1,2,3], items     )

    def test_take_exact_4(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4).filter(items))

        self.assertEqual([1,2,3], take_items)
        self.assertEqual([1,2,3], items     )

class Resevoir_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Reservoir(-1)

        with self.assertRaises(ValueError):
            Reservoir('A')

        with self.assertRaises(ValueError):
            Reservoir((-1,5))

    def test_take_exacts(self):
        items = [1,2,3,4,5]

        take_items = list(Reservoir(2,seed=1).filter(items))
        self.assertEqual([1,2,3,4,5], items)
        self.assertEqual([4, 2], take_items)

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

class Where_Tests(unittest.TestCase):

    def test_filter(self):

        items = [ 1,2,3 ]

        self.assertEqual([]     , list(Where(n_interactions=2       ).filter(items)))
        self.assertEqual([]     , list(Where(n_interactions=4       ).filter(items)))
        self.assertEqual([]     , list(Where(n_interactions=(None,1)).filter(items)))
        self.assertEqual([]     , list(Where(n_interactions=(4,None)).filter(items)))

        self.assertEqual([1,2,3], list(Where(n_interactions=(1,None)).filter(items)))
        self.assertEqual([1,2,3], list(Where(n_interactions=(None,4)).filter(items)))
        self.assertEqual([1,2,3], list(Where(n_interactions=(1,4)   ).filter(items)))
        self.assertEqual([1,2,3], list(Where(n_interactions=3       ).filter(items)))
        self.assertEqual([1,2,3], list(Where(                       ).filter(items)))

    def test_params(self):
        self.assertEqual({'where_n_interactions':(None,1)}, Where(n_interactions=(None,1)).params)
        self.assertEqual({'where_n_interactions':(4,None)}, Where(n_interactions=(4,None)).params)
        self.assertEqual({'where_n_interactions':1       }, Where(n_interactions=1       ).params)
        self.assertEqual({                               }, Where(                       ).params)

class Scale_Tests(unittest.TestCase):

    def test_scale_min_and_minmax_using_all_logged(self):

        interactions = [
            LoggedInteraction((7,2), 1, 1),
            LoggedInteraction((1,9), 1, 1),
            LoggedInteraction((8,3), 1, 1)
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertIsInstance(scl_interactions[0], LoggedInteraction)
        self.assertIsInstance(scl_interactions[1], LoggedInteraction)
        self.assertIsInstance(scl_interactions[2], LoggedInteraction)

        self.assertEqual((6/7,0  ), scl_interactions[0].context)
        self.assertEqual((0  ,1  ), scl_interactions[1].context)
        self.assertEqual((1  ,1/7), scl_interactions[2].context)

    def test_scale_min_and_minmax_using_all_simulated(self):

        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual((6/7,0  ), scl_interactions[0].context)
        self.assertEqual((0  ,1  ), scl_interactions[1].context)
        self.assertEqual((1  ,1/7), scl_interactions[2].context)

    def test_scale_min_and_minmax_using_2(self):

        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax",using=2).filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual((1  ,0  ), scl_interactions[0].context)
        self.assertEqual((0  ,1  ), scl_interactions[1].context)

        self.assertAlmostEqual(7/6, scl_interactions[2].context[0])
        self.assertAlmostEqual(1/7, scl_interactions[2].context[1])

    def test_scale_0_and_2(self):

        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((2,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual((8,2), mem_interactions[0].context)
        self.assertEqual((4,4), mem_interactions[1].context)
        self.assertEqual((2,6), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual((4,1), scl_interactions[0].context)
        self.assertEqual((2,2), scl_interactions[1].context)
        self.assertEqual((1,3), scl_interactions[2].context)

    def test_scale_0_and_2_single_number(self):

        interactions = [
            SimulatedInteraction(2, [1], [1]),
            SimulatedInteraction(4, [1], [1]),
            SimulatedInteraction(6, [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift=0,scale=1/2,using=2).filter(interactions))

        self.assertEqual(2, mem_interactions[0].context)
        self.assertEqual(4, mem_interactions[1].context)
        self.assertEqual(6, mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(1, scl_interactions[0].context)
        self.assertEqual(2, scl_interactions[1].context)
        self.assertEqual(3, scl_interactions[2].context)

    def test_scale_mean_and_std(self):

        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((0,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="mean",scale="std").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0].context)
        self.assertEqual((4,4), mem_interactions[1].context)
        self.assertEqual((0,6), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(( 4/4,-2/2), scl_interactions[0].context)
        self.assertEqual(( 0/4, 0/2), scl_interactions[1].context)
        self.assertEqual((-4/4, 2/2), scl_interactions[2].context)

    def test_scale_med_and_iqr(self):

        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,4), [1], [1]),
            SimulatedInteraction((0,6), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="med",scale="iqr").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0].context)
        self.assertEqual((4,4), mem_interactions[1].context)
        self.assertEqual((0,6), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(( 1,-1), scl_interactions[0].context)
        self.assertEqual(( 0, 0), scl_interactions[1].context)
        self.assertEqual((-1, 1), scl_interactions[2].context)

    def test_scale_med_and_iqr_0(self):

        interactions = [
            SimulatedInteraction((8,2), [1], [1]),
            SimulatedInteraction((4,2), [1], [1]),
            SimulatedInteraction((0,2), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="med",scale="iqr").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0].context)
        self.assertEqual((4,2), mem_interactions[1].context)
        self.assertEqual((0,2), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(( 1, 0), scl_interactions[0].context)
        self.assertEqual(( 0, 0), scl_interactions[1].context)
        self.assertEqual((-1, 0), scl_interactions[2].context)

    def test_scale_min_and_minmax_with_str(self):

        interactions = [
            SimulatedInteraction((7,2,"A"), [1], [1]),
            SimulatedInteraction((1,9,"B"), [1], [1]),
            SimulatedInteraction((8,3,"C"), [1], [1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2,"A"), mem_interactions[0].context)
        self.assertEqual((1,9,"B"), mem_interactions[1].context)
        self.assertEqual((8,3,"C"), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual((6/7,0  ,"A"), scl_interactions[0].context)
        self.assertEqual((0  ,1  ,"B"), scl_interactions[1].context)
        self.assertEqual((1  ,1/7,"C"), scl_interactions[2].context)

    def test_scale_min_and_minmax_with_nan(self):

        interactions = [
            SimulatedInteraction((float('nan'), 2           ), [1], [1]),
            SimulatedInteraction((1           , 9           ), [1], [1]),
            SimulatedInteraction((8           , float('nan')), [1], [1])
        ]

        scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertTrue(isnan(scl_interactions[0].context[0]))
        self.assertEqual(0,   scl_interactions[0].context[1])

        self.assertEqual((0, 1), scl_interactions[1].context)

        self.assertEqual(1  , scl_interactions[2].context[0])
        self.assertTrue(isnan(scl_interactions[2].context[1]))

    def test_scale_min_and_minmax_with_str_and_nan(self):

        nan = float('nan')

        interactions = [
            LoggedInteraction((7,  2, 'A'), 1, 1),
            LoggedInteraction((1,  9, 'B'), 1, 1),
            LoggedInteraction((8,nan, nan), 1, 1)
        ]

        mem_interactions = interactions
    
        with self.assertWarns(Warning):
            scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual((7,2,'A'), mem_interactions[0].context)
        self.assertEqual((1,9,'B'), mem_interactions[1].context)
        self.assertEqual(8,   mem_interactions[2].context[0] )
        self.assertTrue(isnan(mem_interactions[2].context[1]))
        self.assertTrue(isnan(mem_interactions[2].context[2]))

        self.assertEqual(3, len(scl_interactions))

        self.assertIsInstance(scl_interactions[0], LoggedInteraction)
        self.assertIsInstance(scl_interactions[1], LoggedInteraction)
        self.assertIsInstance(scl_interactions[2], LoggedInteraction)

        self.assertEqual((6/7,0,'A'), scl_interactions[0].context)
        self.assertEqual((0  ,1,'B'), scl_interactions[1].context)
        self.assertEqual(1,   scl_interactions[2].context[0])
        self.assertTrue(isnan(scl_interactions[2].context[1]))
        self.assertTrue(isnan(scl_interactions[2].context[2]))

    def test_scale_min_and_minmax_with_mixed(self):

        interactions = [
            SimulatedInteraction(("A", 2  ), [1], [1]),
            SimulatedInteraction((1  , 9  ), [1], [1]),
            SimulatedInteraction((8  , "B"), [1], [1])
        ]

        with self.assertWarns(Warning):
            scl_interactions = list(Scale("min","minmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(("A", 2  ), scl_interactions[0].context)
        self.assertEqual((1  , 9  ), scl_interactions[1].context)
        self.assertEqual((8  , "B"), scl_interactions[2].context)

    def test_scale_0_and_minmax_with_mixed_dict(self):

        interactions = [
            SimulatedInteraction({0:"A", 1:2              }, [1], [1]),
            SimulatedInteraction({0:1  , 1:9  , 2:2       }, [1], [1]),
            SimulatedInteraction({0:8  , 1:"B", 2:1       }, [1], [1]),
            SimulatedInteraction({0:8  , 1:"B", 2:1, 3:"C"}, [1], [1])
        ]

        with self.assertWarns(Warning):
            scl_interactions = list(Scale(0,"minmax").filter(interactions))

        self.assertEqual(4, len(scl_interactions))

        self.assertEqual({0:"A", 1:2               }, scl_interactions[0].context)
        self.assertEqual({0:1  , 1:9  , 2:1        }, scl_interactions[1].context)
        self.assertEqual({0:8  , 1:"B", 2:.5       }, scl_interactions[2].context)
        self.assertEqual({0:8  , 1:"B", 2:.5, 3:"C"}, scl_interactions[3].context)

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

        self.assertEqual(None, scl_interactions[0].context)
        self.assertEqual(None, scl_interactions[1].context)
        self.assertEqual(None, scl_interactions[2].context)

    def test_scale_mean_and_minmax_target_rewards(self):

        interactions = [
            SimulatedInteraction(None, [1,2], [1,3]),
            SimulatedInteraction(None, [1,2], [1,3]),
            SimulatedInteraction(None, [1,2], [1,3])
        ]

        scl_interactions = list(Scale("mean","minmax", target="rewards").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0].context)
        self.assertEqual(None, scl_interactions[1].context)
        self.assertEqual(None, scl_interactions[2].context)

        self.assertEqual([-1/2,1/2], [scl_interactions[0].rewards.eval(a) for a in [1,2]])
        self.assertEqual([-1/2,1/2], [scl_interactions[1].rewards.eval(a) for a in [1,2]])
        self.assertEqual([-1/2,1/2], [scl_interactions[2].rewards.eval(a) for a in [1,2]])

    def test_scale_number_and_absmax_target_discrete_rewards(self):

        interactions = [
            SimulatedInteraction(None, [1,3], [1,3]),
            SimulatedInteraction(None, [1,3], [3,1]),
            SimulatedInteraction(None, [1,3], [1,3])
        ]

        scl_interactions = list(Scale(3,"maxabs", target="rewards").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0].context)
        self.assertEqual(None, scl_interactions[1].context)
        self.assertEqual(None, scl_interactions[2].context)

        self.assertEqual([-1,0], [scl_interactions[0].rewards.eval(a) for a in [1,3]])
        self.assertEqual([0,-1], [scl_interactions[1].rewards.eval(a) for a in [1,3]])
        self.assertEqual([-1,0], [scl_interactions[2].rewards.eval(a) for a in [1,3]])

    def test_scale_min_and_minmax_target_argmax(self):

        interactions = [
            SimulatedInteraction(None, [], L1Reward(3)),
            SimulatedInteraction(None, [], L1Reward(1)),
            SimulatedInteraction(None, [], L1Reward(2))
        ]

        scl_interactions = list(Scale("min", "minmax", target="argmax").filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0].context)
        self.assertEqual(None, scl_interactions[1].context)
        self.assertEqual(None, scl_interactions[2].context)

        self.assertEqual(1 , scl_interactions[0].rewards.argmax())
        self.assertEqual(0 , scl_interactions[1].rewards.argmax())
        self.assertEqual(.5, scl_interactions[2].rewards.argmax())

    def test_params(self):
        self.assertEqual({"scale_shift":"mean","scale_scale":"std","scale_using":None,"scale_target":"features"}, Scale(shift="mean",scale="std").params)
        self.assertEqual({"scale_shift":2     ,"scale_scale":1/2  ,"scale_using":None,"scale_target":"features"}, Scale(shift=2,scale=1/2).params)
        self.assertEqual({"scale_shift":2     ,"scale_scale":1/2  ,"scale_using":10  ,"scale_target":"features"}, Scale(shift=2,scale=1/2,using=10).params)

class Cycle_Tests(unittest.TestCase):

    def test_after_0(self):

        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle().filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].rewards)
        self.assertEqual([1,4], mem_interactions[1].rewards)
        self.assertEqual([1,5], mem_interactions[2].rewards)
        self.assertEqual([1,3], mem_interactions[0].rewards)
        self.assertEqual([1,4], mem_interactions[1].rewards)
        self.assertEqual([1,5], mem_interactions[2].rewards)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([3,1], cyc_interactions[0].rewards)
        self.assertEqual([4,1], cyc_interactions[1].rewards)
        self.assertEqual([5,1], cyc_interactions[2].rewards)
        self.assertEqual([3,1], cyc_interactions[0].rewards)
        self.assertEqual([4,1], cyc_interactions[1].rewards)
        self.assertEqual([5,1], cyc_interactions[2].rewards)

    def test_after_1(self):

        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=1).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].rewards)
        self.assertEqual([1,4], mem_interactions[1].rewards)
        self.assertEqual([1,5], mem_interactions[2].rewards)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].rewards)
        self.assertEqual([4,1], cyc_interactions[1].rewards)
        self.assertEqual([5,1], cyc_interactions[2].rewards)

    def test_after_2(self):

        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=2).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].rewards)
        self.assertEqual([1,4], mem_interactions[1].rewards)
        self.assertEqual([1,5], mem_interactions[2].rewards)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].rewards)
        self.assertEqual([1,4], cyc_interactions[1].rewards)
        self.assertEqual([5,1], cyc_interactions[2].rewards)

    def test_after_10(self):

        interactions = [
            SimulatedInteraction((7,2), [(1,0),(0,1)], [1,3]),
            SimulatedInteraction((1,9), [(1,0),(0,1)], [1,4]),
            SimulatedInteraction((8,3), [(1,0),(0,1)], [1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=10).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].rewards)
        self.assertEqual([1,4], mem_interactions[1].rewards)
        self.assertEqual([1,5], mem_interactions[2].rewards)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].rewards)
        self.assertEqual([1,4], cyc_interactions[1].rewards)
        self.assertEqual([1,5], cyc_interactions[2].rewards)

    def test_with_action_features(self):

        interactions = [
            SimulatedInteraction((7,2), [1,2], [1,3]),
            SimulatedInteraction((1,9), [1,2], [1,4]),
            SimulatedInteraction((8,3), [1,2], [1,5])
        ]

        with self.assertWarns(Warning):

            mem_interactions = interactions
            cyc_interactions = list(Cycle(after=0).filter(mem_interactions))

            self.assertEqual([1,3], mem_interactions[0].rewards)
            self.assertEqual([1,4], mem_interactions[1].rewards)
            self.assertEqual([1,5], mem_interactions[2].rewards)

            self.assertEqual(3, len(cyc_interactions))

            self.assertEqual([1,3], cyc_interactions[0].rewards)
            self.assertEqual([1,4], cyc_interactions[1].rewards)
            self.assertEqual([1,5], cyc_interactions[2].rewards)

    def test_params(self):
        self.assertEqual({"cycle_after":0 }, Cycle().params)
        self.assertEqual({"cycle_after":2 }, Cycle(2).params)

class Impute_Tests(unittest.TestCase):

    def test_impute_mean_logged(self):

        interactions = [
            LoggedInteraction((7           , 2           ), 1, 1),
            LoggedInteraction((float('nan'), float('nan')), 1, 1),
            LoggedInteraction((8           , 3           ), 1, 1)
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute().filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(imp_interactions))

        self.assertEqual((7  ,  2), imp_interactions[0].context)
        self.assertEqual((7.5,2.5), imp_interactions[1].context)
        self.assertEqual((8  ,3  ), imp_interactions[2].context)

    def test_impute_nothing(self):

        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute().filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(imp_interactions))

        self.assertEqual((7,2), imp_interactions[0].context)
        self.assertEqual((1,9), imp_interactions[1].context)
        self.assertEqual((8,3), imp_interactions[2].context)

    def test_impute_mean(self):

        interactions = [
            SimulatedInteraction((7           , 2           ), [1], [1]),
            SimulatedInteraction((float('nan'), float('nan')), [1], [1]),
            SimulatedInteraction((8           , 3           ), [1], [1])
        ]

        mem_interactions = interactions
        imp_interactions = list(Impute().filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        #self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(imp_interactions))

        self.assertEqual((7  ,  2), imp_interactions[0].context)
        self.assertEqual((7.5,2.5), imp_interactions[1].context)
        self.assertEqual((8  ,3  ), imp_interactions[2].context)

    def test_impute_med(self):

        interactions = [
            SimulatedInteraction((7           , 2           ), [1], [1]),
            SimulatedInteraction((7           , 2           ), [1], [1]),
            SimulatedInteraction((float('nan'), float('nan')), [1], [1]),
            SimulatedInteraction((8           , 3           ), [1], [1])
        ]

        imp_interactions = list(Impute("median").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual((7, 2), imp_interactions[0].context)
        self.assertEqual((7, 2), imp_interactions[1].context)
        self.assertEqual((7, 2), imp_interactions[2].context)
        self.assertEqual((8, 3), imp_interactions[3].context)

    def test_impute_mode_None(self):

        interactions = [
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1]),
            SimulatedInteraction(None, [1], [1])
        ]

        imp_interactions = list(Impute("mode").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(None, imp_interactions[0].context)
        self.assertEqual(None, imp_interactions[1].context)
        self.assertEqual(None, imp_interactions[2].context)
        self.assertEqual(None, imp_interactions[3].context)

    def test_impute_mode_singular(self):

        interactions = [
            SimulatedInteraction(1           , [1], [1]),
            SimulatedInteraction(float('nan'), [1], [1]),
            SimulatedInteraction(5           , [1], [1]),
            SimulatedInteraction(5           , [1], [1])
        ]

        imp_interactions = list(Impute("mode").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(1, imp_interactions[0].context)
        self.assertEqual(5, imp_interactions[1].context)
        self.assertEqual(5, imp_interactions[2].context)
        self.assertEqual(5, imp_interactions[3].context)

    def test_impute_mode_dict(self):

        interactions = [
            SimulatedInteraction({1:1           }, [1], [1]),
            SimulatedInteraction({1:float('nan')}, [1], [1]),
            SimulatedInteraction({1:5           }, [1], [1]),
            SimulatedInteraction({1:5           }, [1], [1])
        ]

        imp_interactions = list(Impute("mode").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual({1:1}, imp_interactions[0].context)
        self.assertEqual({1:5}, imp_interactions[1].context)
        self.assertEqual({1:5}, imp_interactions[2].context)
        self.assertEqual({1:5}, imp_interactions[3].context)

    def test_impute_med_with_str(self):

        interactions = [
            SimulatedInteraction((7           , 2           , "A"), [1], [1]),
            SimulatedInteraction((7           , 2           , "A"), [1], [1]),
            SimulatedInteraction((float('nan'), float('nan'), "A"), [1], [1]),
            SimulatedInteraction((8           , 3           , "A"), [1], [1])
        ]

        imp_interactions = list(Impute("median").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual((7, 2, "A"), imp_interactions[0].context)
        self.assertEqual((7, 2, "A"), imp_interactions[1].context)
        self.assertEqual((7, 2, "A"), imp_interactions[2].context)
        self.assertEqual((8, 3, "A"), imp_interactions[3].context)

    def test_params(self):
        self.assertEqual({ "impute_stat": "median", "impute_using": 2 }, Impute("median",2).params)

class Binary_Tests(unittest.TestCase):
    def test_binary(self):
        interactions = [
            SimulatedInteraction((7,2), [1,2], [.2,.3]),
            SimulatedInteraction((1,9), [1,2], [.1,.5]),
            SimulatedInteraction((8,3), [1,2], [.5,.2])
        ]

        binary_interactions = list(Binary().filter(interactions))

        self.assertEqual([0,1], [binary_interactions[0].rewards.eval(a) for a in [1,2]])
        self.assertEqual([0,1], [binary_interactions[1].rewards.eval(a) for a in [1,2]])
        self.assertEqual([1,0], [binary_interactions[2].rewards.eval(a) for a in [1,2]])

    def test_params(self):
        self.assertEqual({'binary':True}, Binary().params)

class Sparse_Tests(unittest.TestCase):
    def test_sparse_simulated_no_context_and_action(self):

        sparse_interactions = list(Sparse(action=True).filter([SimulatedInteraction(None, [1,2], [0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual(None, sparse_interactions[0].context)
        self.assertEqual([{0:1},{0:2}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].rewards)

    def test_sparse_simulated_str_context(self):

        sparse_interactions = list(Sparse().filter([SimulatedInteraction("a", [{1:2},{3:4}], [0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:"a"}, sparse_interactions[0].context)
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].rewards)

    def test_sparse_simulated_str_not_context_not_action(self):

        sparse_interactions = list(Sparse(context=False).filter([SimulatedInteraction("a", [{1:2},{3:4}], [0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual("a", sparse_interactions[0].context)
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].rewards)

    def test_sparse_simulated_tuple_context(self):

        sparse_interactions = list(Sparse().filter([SimulatedInteraction((1,2,3), [{1:2},{3:4}], [0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:1,1:2,2:3}, sparse_interactions[0].context)
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].rewards)

    def test_sparse_logged_tuple_context_and_action(self):

        sparse_interactions = list(Sparse(action=True).filter([LoggedInteraction((1,2,3), 2, 0)]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:1,1:2,2:3}, sparse_interactions[0].context)
        self.assertEqual({0:2}, sparse_interactions[0].action)
        self.assertEqual(0, sparse_interactions[0].reward)

    def test_sparse_logged_tuple_context_and_not_action(self):

        sparse_interactions = list(Sparse().filter([LoggedInteraction((1,2,3), 2, 0)]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:1,1:2,2:3}, sparse_interactions[0].context)
        self.assertEqual(2, sparse_interactions[0].action)
        self.assertEqual(0, sparse_interactions[0].reward)

    def test_params(self):
        self.assertEqual({'sparse_C':True, 'sparse_A':False}, Sparse().params)

class Warm_Tests(unittest.TestCase):

    def test_to_warmstart(self):
        interactions = [
            SimulatedInteraction((7,2), [1,2], [.2,.3]),
            SimulatedInteraction((1,9), [1,2], [.1,.5]),
            SimulatedInteraction((8,3), [1,2], [.5,.2])
        ]

        warmstart_interactions = list(Warm(2).filter(interactions))

        self.assertIsInstance(warmstart_interactions[0], LoggedInteraction)
        self.assertIsInstance(warmstart_interactions[1], LoggedInteraction)
        self.assertIsInstance(warmstart_interactions[2], SimulatedInteraction)

        self.assertEqual((7,2), warmstart_interactions[0].context)
        self.assertEqual(1, warmstart_interactions[0].action)
        self.assertEqual([1,2], warmstart_interactions[0].actions)
        self.assertEqual(1/2, warmstart_interactions[0].probability)
        self.assertEqual(.2, warmstart_interactions[0].reward)

        self.assertEqual((1,9), warmstart_interactions[1].context)
        self.assertEqual(2, warmstart_interactions[1].action)
        self.assertEqual([1,2], warmstart_interactions[1].actions)
        self.assertEqual(1/2, warmstart_interactions[1].probability)
        self.assertEqual(.5, warmstart_interactions[1].reward)

        self.assertEqual((8,3), warmstart_interactions[2].context)
        self.assertEqual([1,2], warmstart_interactions[2].actions)
        self.assertEqual([.5,.2], warmstart_interactions[2].rewards)

    def test_params(self):
        self.assertEqual({"n_warm": 10}, Warm(10).params)

class Noise_Tests(unittest.TestCase):

    def test_default_noise(self):

        interactions = [
            SimulatedInteraction((7,), [1,2], [.2,.3]),
            SimulatedInteraction((1,), [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise().filter(interactions))

        self.assertEqual(2, len(actual_interactions))

        self.assertAlmostEqual(7.62598057113, actual_interactions[0].context[0],5)
        self.assertEqual      ([1,2]        , actual_interactions[0].actions     )
        self.assertEqual      ([.2,.3]      , actual_interactions[0].rewards     )

        self.assertAlmostEqual(-1.0118803935, actual_interactions[1].context[0],5)
        self.assertEqual      ([2,3]        , actual_interactions[1].actions     )
        self.assertEqual      ([.1,.5]      , actual_interactions[1].rewards     )

    def test_context_noise1(self):

        interactions = [
            SimulatedInteraction((7,5), [1,2], [.2,.3]),
            SimulatedInteraction((1,6), [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(context=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([8,5]  , actual_interactions[0].context)
        self.assertEqual([1,2]  , actual_interactions[0].actions)
        self.assertEqual([.2,.3], actual_interactions[0].rewards)
        self.assertEqual([2,7]  , actual_interactions[1].context)
        self.assertEqual([2,3]  , actual_interactions[1].actions)
        self.assertEqual([.1,.5], actual_interactions[1].rewards)

    def test_context_noise2(self):

        interactions = [
            SimulatedInteraction(None, [1,2], [.2,.3]),
            SimulatedInteraction(None, [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(context=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual(None   , actual_interactions[0].context)
        self.assertEqual([1,2]  , actual_interactions[0].actions)
        self.assertEqual([.2,.3], actual_interactions[0].rewards)
        self.assertEqual(None   , actual_interactions[1].context)
        self.assertEqual([2,3]  , actual_interactions[1].actions)
        self.assertEqual([.1,.5], actual_interactions[1].rewards)

    def test_context_noise_sparse(self):

        interactions = [
            SimulatedInteraction({'a':7, 'b':5}, [1,2], [.2,.3]),
            SimulatedInteraction({'a':1, 'b':6}, [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(context=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2             , len(actual_interactions))
        self.assertEqual({'a':8, 'b':5}, actual_interactions[0].context)
        self.assertEqual([1,2]         , actual_interactions[0].actions)
        self.assertEqual([.2,.3]       , actual_interactions[0].rewards)
        self.assertEqual({'a':2, 'b':7}, actual_interactions[1].context)
        self.assertEqual([2,3]         , actual_interactions[1].actions)
        self.assertEqual([.1,.5]       , actual_interactions[1].rewards)

    def test_action_noise1(self):

        interactions = [
            SimulatedInteraction([7], [1,2], [.2,.3]),
            SimulatedInteraction([1], [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(action=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]    , actual_interactions[0].context)
        self.assertEqual([2,2]  , actual_interactions[0].actions)
        self.assertEqual([.2,.3], actual_interactions[0].rewards)
        self.assertEqual([1]    , actual_interactions[1].context)
        self.assertEqual([3,4]  , actual_interactions[1].actions)
        self.assertEqual([.1,.5], actual_interactions[1].rewards)

    def test_action_noise2(self):

        interactions = [
            SimulatedInteraction([7], [[1],[2]], [.2,.3]),
            SimulatedInteraction([1], [[2],[3]], [.1,.5]),
        ]

        actual_interactions = list(Noise(action=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]   , actual_interactions[0].context)
        self.assertEqual([[2],[2]]  , actual_interactions[0].actions)
        self.assertEqual([.2,.3], actual_interactions[0].rewards)
        self.assertEqual([1]   , actual_interactions[1].context)
        self.assertEqual([[3],[4]]  , actual_interactions[1].actions)
        self.assertEqual([.1,.5], actual_interactions[1].rewards)

    def test_action_noise3(self):

        interactions = [
            SimulatedInteraction((7,), ['A','B'], [.2,.3]),
            SimulatedInteraction((1,), ['A','B'], [.1,.5]),
        ]

        actual_interactions = list(Noise().filter(interactions))

        self.assertEqual(2        , len(actual_interactions))
        self.assertEqual(['A','B'], actual_interactions[0].actions)
        self.assertEqual(['A','B'], actual_interactions[1].actions)

    def test_reward_noise(self):

        interactions = [
            SimulatedInteraction([7], [1,2], [.2,.3]),
            SimulatedInteraction([1], [2,3], [.1,.5]),
        ]

        actual_interactions = list(Noise(reward=lambda v,r: v+r.randint(0,1), seed=5).filter(interactions))

        self.assertEqual(2        , len(actual_interactions))
        self.assertEqual([7]      , actual_interactions[0].context)
        self.assertEqual([1,2]    , actual_interactions[0].actions)
        self.assertEqual([1.2,.3] , actual_interactions[0].rewards)
        self.assertEqual([1]      , actual_interactions[1].context)
        self.assertEqual([2,3]    , actual_interactions[1].actions)
        self.assertEqual([1.1,1.5], actual_interactions[1].rewards)

    def test_noise_repeatable(self):

        interactions = [
            SimulatedInteraction([7], [1,2], [.2,.3]),
            SimulatedInteraction([1], [2,3], [.1,.5]),
        ]

        noise_filter = Noise(action=lambda v,r: v+r.randint(0,1), seed=5)

        actual_interactions = list(noise_filter.filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]    , actual_interactions[0].context)
        self.assertEqual([2,2]  , actual_interactions[0].actions)
        self.assertEqual([.2,.3], actual_interactions[0].rewards)
        self.assertEqual([1]    , actual_interactions[1].context)
        self.assertEqual([3,4]  , actual_interactions[1].actions)
        self.assertEqual([.1,.5], actual_interactions[1].rewards)

        actual_interactions = list(noise_filter.filter(interactions))

        self.assertEqual(2      , len(actual_interactions))
        self.assertEqual([7]    , actual_interactions[0].context)
        self.assertEqual([2,2]  , actual_interactions[0].actions)
        self.assertEqual([.2,.3], actual_interactions[0].rewards)
        self.assertEqual([1]    , actual_interactions[1].context)
        self.assertEqual([3,4]  , actual_interactions[1].actions)
        self.assertEqual([.1,.5], actual_interactions[1].rewards)

    def test_noise_error(self):
        with self.assertRaises(CobaException):
            list(Noise().filter([LoggedInteraction((7,), 1, 1)]))

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

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((7,2), cov_interactions[0].context)
        self.assertEqual((1,9), cov_interactions[1].context)
        self.assertEqual((8,3), cov_interactions[2].context)

    def test_riffle1(self):

        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        cov_interactions = list(Riffle(1,seed=5).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((7,2), cov_interactions[0].context)
        self.assertEqual((8,3), cov_interactions[1].context)
        self.assertEqual((1,9), cov_interactions[2].context)

        cov_interactions = list(Riffle(1,seed=4).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((8,3), cov_interactions[0].context)
        self.assertEqual((7,2), cov_interactions[1].context)
        self.assertEqual((1,9), cov_interactions[2].context)

    def test_riffle5(self):

        interactions = [
            SimulatedInteraction((7,2), [1], [1]),
            SimulatedInteraction((1,9), [1], [1]),
            SimulatedInteraction((8,3), [1], [1])
        ]

        mem_interactions = interactions
        cov_interactions = list(Riffle(5).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((7,2), cov_interactions[0].context)
        self.assertEqual((1,9), cov_interactions[1].context)
        self.assertEqual((8,3), cov_interactions[2].context)

    def test_params(self):
        self.assertEqual({'riffle_spacing':2, 'riffle_seed':3}, Riffle(2,3).params)

class Flatten_Tests(unittest.TestCase):
    def test_flatten_context(self):
        
        interactions = [
            SimulatedInteraction((7,(1,0)), [(1,"def"),2], [.2,.3]),
            SimulatedInteraction((1,(0,1)), [(1,"ghi"),3], [.1,.5]),
        ]

        flatten_filter = Flatten()
        actual_interactions = list(flatten_filter.filter(interactions))

        self.assertEqual(2            , len(actual_interactions))
        self.assertEqual((7,1,0)      , actual_interactions[0].context)
        self.assertEqual([(1,"def"),2], actual_interactions[0].actions)
        self.assertEqual([.2,.3]      , actual_interactions[0].rewards)
        self.assertEqual((1,0,1)      , actual_interactions[1].context)
        self.assertEqual([(1,"ghi"),3], actual_interactions[1].actions)
        self.assertEqual([.1,.5]      , actual_interactions[1].rewards)

    def test_flatten_actions(self):
        
        interactions = [
            SimulatedInteraction((7,1,0), [(1,("def",1)),2], [.2,.3]),
            SimulatedInteraction((1,0,1), [(1,("ghi",2)),3], [.1,.5]),
        ]

        flatten_filter = Flatten()
        actual_interactions = list(flatten_filter.filter(interactions))

        self.assertEqual(2              , len(actual_interactions))
        self.assertEqual((7,1,0)        , actual_interactions[0].context)
        self.assertEqual([(1,"def",1),2], actual_interactions[0].actions)
        self.assertEqual([.2,.3]        , actual_interactions[0].rewards)
        self.assertEqual((1,0,1)        , actual_interactions[1].context)
        self.assertEqual([(1,"ghi",2),3], actual_interactions[1].actions)
        self.assertEqual([.1,.5]        , actual_interactions[1].rewards)

    def test_params(self):
        self.assertEqual({'flat':True}, Flatten().params)

class Params_Tests(unittest.TestCase):
    def test_params(self): 
        params_filter = Params({'a':123})
        self.assertEqual({'a':123}, params_filter.params)
        self.assertEqual(1, params_filter.filter(1))

class Grounded_Tests(unittest.TestCase):

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
            SimulatedInteraction(0,[1,2,3],[1,0,0],c=0),
            SimulatedInteraction(1,[1,2,3],[0,1,0],c=1),
        ]
        to_igl_filter    = Grounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        self.assertEqual(2, len(igl_interactions))

    def test_fixed_for_seed_new_version(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[1,0,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
        ]
        to_igl_filter    = Grounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        feedbacks_0 = [igl_interactions[0].feedbacks.eval(a) for a in igl_interactions[0].actions ]

        self.assertEqual(True,igl_interactions[0].kwargs['isnormal'])
        self.assertEqual(4, igl_interactions[0].kwargs['userid'])
        self.assertEqual([(0,), (2,), (3,)], feedbacks_0)

        feedbacks_1 = [igl_interactions[1].feedbacks.eval(a) for a in igl_interactions[1].actions ]

        self.assertEqual(True,igl_interactions[1].kwargs['isnormal'])
        self.assertEqual(2, igl_interactions[1].kwargs['userid'])
        self.assertEqual([(3,),(0,),(3,)], feedbacks_1)

    @unittest.skip("This is correct for the new version but we're using old version for now")
    def test_fixed_for_seed_old_version(self):
        interactions = [
            SimulatedInteraction(0,[1,2,3],[1,0,0]),
            SimulatedInteraction(1,[1,2,3],[0,1,0]),
        ]

        to_igl_filter    = Grounded(10,5,4,2,4)
        igl_interactions = list(to_igl_filter.filter(interactions))

        feedbacks_0 = [igl_interactions[0].feedbacks.eval(a) for a in igl_interactions[0].actions ]

        self.assertEqual(True,igl_interactions[0].kwargs['isnormal'])
        self.assertEqual(4, igl_interactions[0].kwargs['userid'])
        self.assertEqual([(1,),(2,),(2,)], feedbacks_0)

        feedbacks_1 = [igl_interactions[1].feedbacks.eval(a) for a in igl_interactions[0].actions ]

        self.assertEqual(True,igl_interactions[1].kwargs['isnormal'])
        self.assertEqual(2, igl_interactions[1].kwargs['userid'])
        self.assertEqual([(2,),(1,),(3,)], feedbacks_1)

    def test_number_context(self):
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[1,0,0]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0].context, (igl_interactions[0].kwargs['userid'], 0))

    def test_list_context(self):
        sim_interactions = [ SimulatedInteraction([0],[1,2,3],[1,0,0]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0].context, (igl_interactions[0].kwargs['userid'], 0))

    def test_dict_context(self):
        sim_interactions = [ SimulatedInteraction({'a':1},[1,2,3],[1,0,0]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0].context, {'userid':igl_interactions[0].kwargs['userid'], 'a':1})
    
    def test_01_reward(self):
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[1,0,0]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0].rewards, [1,0,0])

    def test_not_01_reward(self):
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[0,.2,.5]) ]
        igl_interactions = list(Grounded(10,5,4,2,1).filter(sim_interactions))
        self.assertEqual(igl_interactions[0].rewards.eval(1), 0)
        self.assertEqual(igl_interactions[0].rewards.eval(2), 0)
        self.assertEqual(igl_interactions[0].rewards.eval(3), 1)

    def test_normal_bizzaro_users(self):
        to_igl           = Grounded(10,5,4,2,1)
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[0,.2,.5]) ] * 6000
        igl_interactions = list(to_igl.filter(sim_interactions))

        c = Counter()
        for i in igl_interactions:
            self.assertEqual(i.kwargs['isnormal'],i.kwargs['userid'] in to_igl.normalids)
            c.update(['normal' if i.kwargs['isnormal'] else 'bizzaro'])

        self.assertAlmostEqual(1,c['normal']/c['bizzaro'],1)

    def test_feedbacks(self):
        n_words          = 50
        to_igl           = Grounded(10,5,n_words,25,1)
        sim_interactions = [ SimulatedInteraction(0,[1,2,3],[0,.2,.5]) ] * 50000
        igl_interactions = list(to_igl.filter(sim_interactions))

        c = Counter()
        for i in igl_interactions:
            feedbacks = [i.feedbacks.eval(a) for a in i.actions ]
            rewards   = [i.rewards.eval(a) for a in i.actions ]

            for word,reward in zip(feedbacks,rewards):
                if reward == 1: self.assertEqual(i.kwargs['isnormal'], word[0] in to_igl.goodwords)
                if reward == 0: self.assertEqual(i.kwargs['isnormal'], word[0] in to_igl.badwords)

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
            f1 = [interaction.feedbacks.eval(a) for a in interaction.actions ]
            f2 = [interaction.feedbacks.eval(a) for a in interaction.actions ]
            self.assertEqual(f1,f2)
    
    def test_params(self):
        params = Grounded(10,5,4,2,1).params

        self.assertEqual(10, params['n_users'])
        self.assertEqual( 5, params['n_normal'])
        self.assertEqual( 4, params['n_words'])
        self.assertEqual( 2, params['n_good'])
        self.assertEqual( 1, params['igl_seed'])

class Finalize_Tests(unittest.TestCase):

    def test_dense_simulated(self):
        interactions = [SimulatedInteraction([1,2,3],[[1,2],[3,4]], [1,2])]

        actual = list(Finalize().filter(interactions))

        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0].context, (1,2,3))
        self.assertEqual(actual[0].actions, [(1,2),(3,4)])

    def test_sparse_simulated(self):
        interactions = [SimulatedInteraction({1:2},[[1,2],[3,4]], [1,2])]

        actual = list(Finalize().filter(interactions))

        self.assertEqual(len(actual),1)
        self.assertIsInstance(actual[0].context, HashableMap)
        self.assertEqual(actual[0].actions, [(1,2),(3,4)])

    def test_logged(self):
        interactions = [LoggedInteraction([1,2,3], [1,2], 1, 1, [[1,2],[3,4]], [1,2])]

        actual = list(Finalize().filter(interactions))

        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0].context, (1,2,3))
        self.assertEqual(actual[0].action, (1,2))
        self.assertEqual(actual[0].actions, [(1,2),(3,4)])

    def test_grounded(self):
        interactions = [GroundedInteraction([1,2,3],[[1,2],[3,4]], [1,2], [3,4])]

        actual = list(Finalize().filter(interactions))

        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0].context, (1,2,3))
        self.assertEqual(actual[0].actions, [(1,2),(3,4)])

if __name__ == '__main__':
    unittest.main()
