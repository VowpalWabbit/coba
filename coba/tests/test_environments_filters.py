import unittest

from math import isnan

from coba.contexts     import CobaContext, NullLogger
from coba.environments import LoggedInteraction, SimulatedInteraction
from coba.environments import FilteredEnvironment
from coba.environments import Identity, Sparse, Sort, Scale, Cycle, Impute, Binary, WarmStart, Shuffle, Take, Reservoir

class TestEnvironment:

    def __init__(self, id) -> None:
        self._id = id

    @property
    def params(self):
        return {'id':self._id}

    def read(self):
        return [
            SimulatedInteraction(1, [None,None], rewards=[1,2]),
            SimulatedInteraction(2, [None,None], rewards=[2,3]),
            SimulatedInteraction(3, [None,None], rewards=[3,4]),
        ]

    def __str__(self) -> str:
        return str(self.params)

class NoParamIdent:
    def filter(self,item):
        return item

    def __str__(self) -> str:
        return 'NoParamIdent'

CobaContext.logger = NullLogger()

class FilteredEnvironment_Tests(unittest.TestCase):

    def test_environment_no_filters(self):
        ep = FilteredEnvironment(TestEnvironment("A"))

        self.assertEqual(3, len(ep.read()))
        self.assertEqual({"id":"A"}, ep.params)
        self.assertEqual(str({"id":"A"}), str(ep))

    def test_environment_one_filter(self):
        ep = FilteredEnvironment(TestEnvironment("A"), Take(1))

        self.assertEqual(1, len(list(ep.read())))
        self.assertEqual({'id':'A', 'take':1}, ep.params)
        self.assertEqual("{'id': 'A'},{'take': 1}", str(ep))

    def test_environment_ident_filter_removed(self):
        ep = FilteredEnvironment(TestEnvironment("A"), Identity(), Take(1))

        self.assertEqual(1, len(list(ep.read())))
        self.assertEqual({'id':'A', 'take':1}, ep.params)
        self.assertEqual("{'id': 'A'},{'take': 1}", str(ep))

    def test_environment_no_param_ident(self):
        ep = FilteredEnvironment(TestEnvironment("A"), NoParamIdent())

        self.assertEqual(3, len(list(ep.read())))
        self.assertEqual({'id':'A'}, ep.params)
        self.assertEqual("{'id': 'A'},NoParamIdent", str(ep))

class Shuffle_Tests(unittest.TestCase):

    def test_str(self):
        self.assertEqual("{'shuffle': 1}", str(Shuffle(1)))

class Sort_Tests(unittest.TestCase):

    def test_sort1_logged(self):

        interactions = [
            LoggedInteraction((7,2), 1, reward=1),
            LoggedInteraction((1,9), 1, reward=1),
            LoggedInteraction((8,3), 1, reward=1)
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
            SimulatedInteraction((7,2), [1], rewards=[1]),
            SimulatedInteraction((1,9), [1], rewards=[1]),
            SimulatedInteraction((8,3), [1], rewards=[1])
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
            SimulatedInteraction((1,2), [1], rewards=[1]),
            SimulatedInteraction((1,9), [1], rewards=[1]),
            SimulatedInteraction((1,3), [1], rewards=[1])
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
            SimulatedInteraction((1,2), [1], rewards=[1]),
            SimulatedInteraction((1,9), [1], rewards=[1]),
            SimulatedInteraction((1,3), [1], rewards=[1])
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

    def test_take1(self):

        items = [ 1,2,3 ]
        take_items = list(Take(1).filter(items))

        self.assertEqual(1, len(take_items))
        self.assertEqual(items[0], take_items[0])

        self.assertEqual(3, len(items))
        self.assertEqual(items[0], items[0])
        self.assertEqual(items[1], items[1])
        self.assertEqual(items[2], items[2])

    def test_take4(self):
        items = [ 1,2,3 ]
        take_items = list(Take(4).filter(items))

        self.assertEqual(3, len(items))
        self.assertEqual(0, len(take_items))

    def test_params(self):
        self.assertEqual({'take':None}, Take(None).params)
        self.assertEqual({'take':2}, Take(2).params)

    def test_str(self):
        self.assertEqual("{'take': None}", str(Take(None)))

class Resevoir_Tests(unittest.TestCase):

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            Reservoir(-1)

        with self.assertRaises(ValueError):
            Reservoir('A')

    def test_take_seed(self):
        take_items = list(Reservoir(2,seed=1).filter(range(10000)))
        self.assertEqual(2, len(take_items))
        self.assertLess(0, take_items[0])
        self.assertLess(0, take_items[1])

    def test_params(self):
        self.assertEqual({"reservoir_count":2, "reservoir_seed":3}, Reservoir(2,3).params)
    
    def test_str(self):
        self.assertEqual(str({"reservoir_count":2, "reservoir_seed":3}), str(Reservoir(2,3)))

class Scale_Tests(unittest.TestCase):

    def test_scale_min_and_minmax_using_all_logged(self):

        interactions = [
            LoggedInteraction((7,2), 1, reward=1),
            LoggedInteraction((1,9), 1, reward=1),
            LoggedInteraction((8,3), 1, reward=1)
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale().filter(interactions))

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
            SimulatedInteraction((7,2), [1], rewards=[1]),
            SimulatedInteraction((1,9), [1], rewards=[1]),
            SimulatedInteraction((8,3), [1], rewards=[1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual((6/7,0  ), scl_interactions[0].context)
        self.assertEqual((0  ,1  ), scl_interactions[1].context)
        self.assertEqual((1  ,1/7), scl_interactions[2].context)

    def test_scale_min_and_minmax_using_2(self):

        interactions = [
            SimulatedInteraction((7,2), [1], rewards=[1]),
            SimulatedInteraction((1,9), [1], rewards=[1]),
            SimulatedInteraction((8,3), [1], rewards=[1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(using=2).filter(interactions))

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
            SimulatedInteraction((8,2), [1], rewards=[1]),
            SimulatedInteraction((4,4), [1], rewards=[1]),
            SimulatedInteraction((2,6), [1], rewards=[1])
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
            SimulatedInteraction(2, [1], rewards=[1]),
            SimulatedInteraction(4, [1], rewards=[1]),
            SimulatedInteraction(6, [1], rewards=[1])
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
            SimulatedInteraction((8,2), [1], rewards=[1]),
            SimulatedInteraction((4,4), [1], rewards=[1]),
            SimulatedInteraction((0,6), [1], rewards=[1])
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
            SimulatedInteraction((8,2), [1], rewards=[1]),
            SimulatedInteraction((4,4), [1], rewards=[1]),
            SimulatedInteraction((0,6), [1], rewards=[1])
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
            SimulatedInteraction((8,2), [1], rewards=[1]),
            SimulatedInteraction((4,2), [1], rewards=[1]),
            SimulatedInteraction((0,2), [1], rewards=[1])
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
            SimulatedInteraction((7,2,"A"), [1], rewards=[1]),
            SimulatedInteraction((1,9,"B"), [1], rewards=[1]),
            SimulatedInteraction((8,3,"C"), [1], rewards=[1])
        ]
 
        mem_interactions = interactions
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual((7,2,"A"), mem_interactions[0].context)
        self.assertEqual((1,9,"B"), mem_interactions[1].context)
        self.assertEqual((8,3,"C"), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual((6/7,0  ,"A"), scl_interactions[0].context)
        self.assertEqual((0  ,1  ,"B"), scl_interactions[1].context)
        self.assertEqual((1  ,1/7,"C"), scl_interactions[2].context)

    def test_scale_min_and_minmax_with_(self):

        interactions = [
            SimulatedInteraction((float('nan'), 2           ), [1], rewards=[1]),
            SimulatedInteraction((1           , 9           ), [1], rewards=[1]),
            SimulatedInteraction((8           , float('nan')), [1], rewards=[1])
        ]
 
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertTrue(isnan(scl_interactions[0].context[0]))
        self.assertEqual(0,   scl_interactions[0].context[1])

        self.assertEqual((0, 1), scl_interactions[1].context)

        self.assertEqual(1  , scl_interactions[2].context[0])
        self.assertTrue(isnan(scl_interactions[2].context[1]))

    def test_scale_min_and_minmax_with_mixed(self):

        interactions = [
            SimulatedInteraction(("A", 2  ), [1], rewards=[1]),
            SimulatedInteraction((1  , 9  ), [1], rewards=[1]),
            SimulatedInteraction((8  , "B"), [1], rewards=[1])
        ]
 
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual("A", scl_interactions[0].context[0])
        self.assertEqual(0  , scl_interactions[0].context[1])
        
        self.assertEqual((0, 1), scl_interactions[1].context)
        
        self.assertEqual(1  , scl_interactions[2].context[0])
        self.assertTrue("B" , scl_interactions[2].context[1])

    def test_scale_min_and_minmax_with_dict(self):

        interactions = [
            SimulatedInteraction({0:"A", 1:2  }, [1], rewards=[1]),
            SimulatedInteraction({0:1  , 1:9  }, [1], rewards=[1]),
            SimulatedInteraction({0:8  , 1:"B"}, [1], rewards=[1])
        ]
 
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual("A", scl_interactions[0].context[0])
        self.assertEqual(0  , scl_interactions[0].context[1])
        
        self.assertEqual({0:0, 1:1}, scl_interactions[1].context)
        
        self.assertEqual(1  , scl_interactions[2].context[0])
        self.assertTrue("B" , scl_interactions[2].context[1])

    def test_scale_min_and_minmax_with_None(self):

        interactions = [
            SimulatedInteraction(None, [1], rewards=[1]),
            SimulatedInteraction(None, [1], rewards=[1]),
            SimulatedInteraction(None, [1], rewards=[1])
        ]
 
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(None, scl_interactions[0].context)
        self.assertEqual(None, scl_interactions[1].context)
        self.assertEqual(None, scl_interactions[2].context)
        
    def test_params(self):
        self.assertEqual({"scale_shift":"mean","scale_scale":"std","scale_using":None}, Scale(shift="mean",scale="std").params)
        self.assertEqual({"scale_shift":2,"scale_scale":1/2,"scale_using":None}, Scale(shift=2,scale=1/2).params)
        self.assertEqual({"scale_shift":2,"scale_scale":1/2,"scale_using":10}, Scale(shift=2,scale=1/2,using=10).params)

class Cycle_Tests(unittest.TestCase):

    def test_after_0(self):

        interactions = [
            SimulatedInteraction((7,2), [1,2], rewards=[1,3]),
            SimulatedInteraction((1,9), [1,2], rewards=[1,4]),
            SimulatedInteraction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle().filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], mem_interactions[1].kwargs["rewards"])
        self.assertEqual([1,5], mem_interactions[2].kwargs["rewards"])
        self.assertEqual([1,3], mem_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], mem_interactions[1].kwargs["rewards"])
        self.assertEqual([1,5], mem_interactions[2].kwargs["rewards"])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([3,1], cyc_interactions[0].kwargs["rewards"])
        self.assertEqual([4,1], cyc_interactions[1].kwargs["rewards"])
        self.assertEqual([5,1], cyc_interactions[2].kwargs["rewards"])
        self.assertEqual([3,1], cyc_interactions[0].kwargs["rewards"])
        self.assertEqual([4,1], cyc_interactions[1].kwargs["rewards"])
        self.assertEqual([5,1], cyc_interactions[2].kwargs["rewards"])

    def test_after_1(self):

        interactions = [
            SimulatedInteraction((7,2), [1,2], rewards=[1,3]),
            SimulatedInteraction((1,9), [1,2], rewards=[1,4]),
            SimulatedInteraction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=1).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], mem_interactions[1].kwargs["rewards"])
        self.assertEqual([1,5], mem_interactions[2].kwargs["rewards"])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].kwargs["rewards"])
        self.assertEqual([4,1], cyc_interactions[1].kwargs["rewards"])
        self.assertEqual([5,1], cyc_interactions[2].kwargs["rewards"])

    def test_after_2(self):

        interactions = [
            SimulatedInteraction((7,2), [1,2], rewards=[1,3]),
            SimulatedInteraction((1,9), [1,2], rewards=[1,4]),
            SimulatedInteraction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=2).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], mem_interactions[1].kwargs["rewards"])
        self.assertEqual([1,5], mem_interactions[2].kwargs["rewards"])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], cyc_interactions[1].kwargs["rewards"])
        self.assertEqual([5,1], cyc_interactions[2].kwargs["rewards"])

    def test_after_10(self):

        interactions = [
            SimulatedInteraction((7,2), [1,2], rewards=[1,3]),
            SimulatedInteraction((1,9), [1,2], rewards=[1,4]),
            SimulatedInteraction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=10).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], mem_interactions[1].kwargs["rewards"])
        self.assertEqual([1,5], mem_interactions[2].kwargs["rewards"])

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].kwargs["rewards"])
        self.assertEqual([1,4], cyc_interactions[1].kwargs["rewards"])
        self.assertEqual([1,5], cyc_interactions[2].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({"cycle_after":0 }, Cycle().params)
        self.assertEqual({"cycle_after":2 }, Cycle(2).params)

class Impute_Tests(unittest.TestCase):

    def test_impute_mean_logged(self):

        interactions = [
            LoggedInteraction((7           , 2           ), 1, reward=1),
            LoggedInteraction((float('nan'), float('nan')), 1, reward=1),
            LoggedInteraction((8           , 3           ), 1, reward=1)
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
            SimulatedInteraction((7,2), [1], rewards=[1]),
            SimulatedInteraction((1,9), [1], rewards=[1]),
            SimulatedInteraction((8,3), [1], rewards=[1])
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
            SimulatedInteraction((7           , 2           ), [1], rewards=[1]),
            SimulatedInteraction((float('nan'), float('nan')), [1], rewards=[1]),
            SimulatedInteraction((8           , 3           ), [1], rewards=[1])
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
            SimulatedInteraction((7           , 2           ), [1], rewards=[1]),
            SimulatedInteraction((7           , 2           ), [1], rewards=[1]),
            SimulatedInteraction((float('nan'), float('nan')), [1], rewards=[1]),
            SimulatedInteraction((8           , 3           ), [1], rewards=[1])
        ]

        imp_interactions = list(Impute("median").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual((7, 2), imp_interactions[0].context)
        self.assertEqual((7, 2), imp_interactions[1].context)
        self.assertEqual((7, 2), imp_interactions[2].context)
        self.assertEqual((8, 3), imp_interactions[3].context)

    def test_impute_mode_None(self):

        interactions = [
            SimulatedInteraction(None, [1], rewards=[1]),
            SimulatedInteraction(None, [1], rewards=[1]),
            SimulatedInteraction(None, [1], rewards=[1]),
            SimulatedInteraction(None, [1], rewards=[1])
        ]

        imp_interactions = list(Impute("mode").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(None, imp_interactions[0].context)
        self.assertEqual(None, imp_interactions[1].context)
        self.assertEqual(None, imp_interactions[2].context)
        self.assertEqual(None, imp_interactions[3].context)

    def test_impute_mode_singular(self):

        interactions = [
            SimulatedInteraction(1           , [1], rewards=[1]),
            SimulatedInteraction(float('nan'), [1], rewards=[1]),
            SimulatedInteraction(5           , [1], rewards=[1]),
            SimulatedInteraction(5           , [1], rewards=[1])
        ]

        imp_interactions = list(Impute("mode").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual(1, imp_interactions[0].context)
        self.assertEqual(5, imp_interactions[1].context)
        self.assertEqual(5, imp_interactions[2].context)
        self.assertEqual(5, imp_interactions[3].context)

    def test_impute_mode_dict(self):

        interactions = [
            SimulatedInteraction({1:1           }, [1], rewards=[1]),
            SimulatedInteraction({1:float('nan')}, [1], rewards=[1]),
            SimulatedInteraction({1:5           }, [1], rewards=[1]),
            SimulatedInteraction({1:5           }, [1], rewards=[1])
        ]

        imp_interactions = list(Impute("mode").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual({1:1}, imp_interactions[0].context)
        self.assertEqual({1:5}, imp_interactions[1].context)
        self.assertEqual({1:5}, imp_interactions[2].context)
        self.assertEqual({1:5}, imp_interactions[3].context)

    def test_impute_med_with_str(self):

        interactions = [
            SimulatedInteraction((7           , 2           , "A"), [1], rewards=[1]),
            SimulatedInteraction((7           , 2           , "A"), [1], rewards=[1]),
            SimulatedInteraction((float('nan'), float('nan'), "A"), [1], rewards=[1]),
            SimulatedInteraction((8           , 3           , "A"), [1], rewards=[1])
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
            SimulatedInteraction((7,2), [1,2], rewards=[.2,.3]),
            SimulatedInteraction((1,9), [1,2], rewards=[.1,.5]),
            SimulatedInteraction((8,3), [1,2], rewards=[.5,.2])
        ]
        
        binary_interactions = list(Binary().filter(interactions))

        self.assertEqual([0,1], binary_interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], binary_interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], binary_interactions[2].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'binary':True}, Binary().params)

class Sparse_Tests(unittest.TestCase):
    def test_sparse_simulated_no_context_and_action(self):
        
        sparse_interactions = list(Sparse(action=True).filter([SimulatedInteraction(None, [1,2], rewards=[0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual(None, sparse_interactions[0].context)
        self.assertEqual([{0:1},{0:2}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].kwargs["rewards"])

    def test_sparse_simulated_str_context(self):

        sparse_interactions = list(Sparse().filter([SimulatedInteraction("a", [{1:2},{3:4}], rewards=[0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:"a"}, sparse_interactions[0].context)
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].kwargs["rewards"])

    def test_sparse_simulated_str_not_context_not_action(self):

        sparse_interactions = list(Sparse(context=False).filter([SimulatedInteraction("a", [{1:2},{3:4}], rewards=[0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual("a", sparse_interactions[0].context)
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].kwargs["rewards"])

    def test_sparse_simulated_tuple_context(self):

        sparse_interactions = list(Sparse().filter([SimulatedInteraction((1,2,3), [{1:2},{3:4}], rewards=[0,1]) ]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:1,1:2,2:3}, sparse_interactions[0].context)
        self.assertEqual([{1:2},{3:4}], sparse_interactions[0].actions)
        self.assertEqual([0,1], sparse_interactions[0].kwargs["rewards"])

    def test_sparse_logged_tuple_context_and_action(self):

        sparse_interactions = list(Sparse(action=True).filter([LoggedInteraction((1,2,3), 2, reward=0)]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:1,1:2,2:3}, sparse_interactions[0].context)
        self.assertEqual({0:2}, sparse_interactions[0].action)
        self.assertEqual(0, sparse_interactions[0].kwargs["reward"])

    def test_sparse_logged_tuple_context_and_not_action(self):

        sparse_interactions = list(Sparse().filter([LoggedInteraction((1,2,3), 2, reward=0)]))

        self.assertEqual(1, len(sparse_interactions))
        self.assertEqual({0:1,1:2,2:3}, sparse_interactions[0].context)
        self.assertEqual(2, sparse_interactions[0].action)
        self.assertEqual(0, sparse_interactions[0].kwargs["reward"])

    def test_params(self):
        self.assertEqual({'sparse_C':True, 'sparse_A':False}, Sparse().params)

class ToWarmStart_Tests(unittest.TestCase):
    
    def test_to_warmstart(self):
        interactions = [
            SimulatedInteraction((7,2), [1,2], rewards=[.2,.3], reveals=[1,2]),
            SimulatedInteraction((1,9), [1,2], rewards=[.1,.5], reveals=[3,4]),
            SimulatedInteraction((8,3), [1,2], rewards=[.5,.2], reveals=[5,6])
        ]

        warmstart_interactions = list(WarmStart(2).filter(interactions))

        self.assertIsInstance(warmstart_interactions[0], LoggedInteraction)
        self.assertIsInstance(warmstart_interactions[1], LoggedInteraction)
        self.assertIsInstance(warmstart_interactions[2], SimulatedInteraction)

        self.assertEqual((7,2), warmstart_interactions[0].context)
        self.assertEqual(1, warmstart_interactions[0].action)
        self.assertEqual([1,2], warmstart_interactions[0].kwargs["actions"])
        self.assertEqual(1/2, warmstart_interactions[0].kwargs["probability"])
        self.assertEqual(.2, warmstart_interactions[0].kwargs["reward"])
        self.assertEqual(1, warmstart_interactions[0].kwargs["reveal"])

        self.assertEqual((1,9), warmstart_interactions[1].context)
        self.assertEqual(2, warmstart_interactions[1].action)
        self.assertEqual([1,2], warmstart_interactions[1].kwargs["actions"])
        self.assertEqual(1/2, warmstart_interactions[1].kwargs["probability"])
        self.assertEqual(.5, warmstart_interactions[1].kwargs["reward"])
        self.assertEqual(4, warmstart_interactions[1].kwargs["reveal"])

        self.assertEqual((8,3), warmstart_interactions[2].context)
        self.assertEqual([1,2], warmstart_interactions[2].actions)
        self.assertEqual([.5,.2], warmstart_interactions[2].kwargs["rewards"])
        self.assertEqual([5,6], warmstart_interactions[2].kwargs["reveals"])

    def test_params(self):
        self.assertEqual({"n_warmstart": 10}, WarmStart(10).params)

if __name__ == '__main__':
    unittest.main()