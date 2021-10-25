import unittest

from math import isnan
from itertools import repeat

from coba.config import CobaConfig, NoneLogger
from coba.simulations import Interaction, Identity, Shuffle, Take, Sort, Scale, Cycle, Impute

CobaConfig.Logger = NoneLogger()

class Identity_Tests(unittest.TestCase):
    
    def test_ident(self):
        interactions = [
            Interaction((8,2), [1], rewards=[1]),
            Interaction((4,4), [1], rewards=[1]),
            Interaction((0,6), [1], rewards=[1])
        ]

        mem_interactions = interactions
        idn_interactions = list(Identity().filter(interactions))

        self.assertEqual(idn_interactions, mem_interactions)

    def test_params(self):
        self.assertEqual({}, Identity().params)
        self.assertEqual({'shuffle':None}, Shuffle(None).params)

class Shuffle_Tests(unittest.TestCase):
    
    def test_shuffle(self):
        interactions = [ Interaction(c,a,rewards=r) for c,a,r in zip(repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]) ]
        shuffled_interactions = list(Shuffle(40).filter(interactions))

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

        self.assertEqual(3, len(shuffled_interactions))
        self.assertEqual(interactions[1], shuffled_interactions[0])
        self.assertEqual(interactions[2], shuffled_interactions[1])
        self.assertEqual(interactions[0], shuffled_interactions[2])

    def test_params(self):
        self.assertEqual({'shuffle':2}, Shuffle(2).params)
        self.assertEqual({'shuffle':None}, Shuffle(None).params)

class Take_Tests(unittest.TestCase):
    
    def test_take1(self):

        interactions = [ Interaction(c,a,rewards=r) for c,a,r in zip(repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]) ]
        take_interactions = list(Take(1).filter(interactions))

        self.assertEqual(1, len(take_interactions))
        self.assertEqual(interactions[0], take_interactions[0])

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

    def test_take2(self):
        interactions = [ Interaction(c,a,rewards=r) for c,a,r in zip(repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]) ]
        take_interactions = list(Take(2).filter(interactions))

        self.assertEqual(2, len(take_interactions))
        self.assertEqual(interactions[0], take_interactions[0])
        self.assertEqual(interactions[1], take_interactions[1])

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

    def test_take3(self):
        interactions = [ Interaction(c,a,rewards=r) for c,a,r in zip(repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]) ]
        take_interactions = list(Take(3).filter(interactions))

        self.assertEqual(3, len(take_interactions))
        self.assertEqual(interactions[0], take_interactions[0])
        self.assertEqual(interactions[1], take_interactions[1])
        self.assertEqual(interactions[2], take_interactions[2])

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

    def test_take4(self):
        interactions = [ Interaction(c,a,rewards=r) for c,a,r in zip(repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]) ]
        take_interactions = list(Take(4).filter(interactions))

        self.assertEqual(3, len(interactions))
        self.assertEqual(0, len(take_interactions))

    def test_params(self):
        self.assertEqual({'take':2}, Take(2).params)
        self.assertEqual({'take':None}, Take(None).params)

class Sort_tests(unittest.TestCase):

    def test_sort1(self) -> None:

        interactions = [
            Interaction((7,2), [1], rewards=[1]),
            Interaction((1,9), [1], rewards=[1]),
            Interaction((8,3), [1], rewards=[1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0]).filter(mem_interactions))

        self.assertEqual((7,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((8,3), mem_interactions[2].context)

        self.assertEqual((1,9), srt_interactions[0].context)
        self.assertEqual((7,2), srt_interactions[1].context)
        self.assertEqual((8,3), srt_interactions[2].context)

    def test_sort2(self) -> None:

        interactions = [
            Interaction((1,2), [1], rewards=[1]),
            Interaction((1,9), [1], rewards=[1]),
            Interaction((1,3), [1], rewards=[1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort([0,1]).filter(mem_interactions))

        self.assertEqual((1,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((1,3), mem_interactions[2].context)

        self.assertEqual((1,2), srt_interactions[0].context)
        self.assertEqual((1,3), srt_interactions[1].context)
        self.assertEqual((1,9), srt_interactions[2].context)

    def test_sort3(self) -> None:
        interactions = [
            Interaction((1,2), [1], rewards=[1]),
            Interaction((1,9), [1], rewards=[1]),
            Interaction((1,3), [1], rewards=[1])
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

class Scale_tests(unittest.TestCase):

    def test_scale_min_and_minmax_using_all(self) -> None:

        interactions = [
            Interaction((7,2), [1], rewards=[1]),
            Interaction((1,9), [1], rewards=[1]),
            Interaction((8,3), [1], rewards=[1])
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

    def test_scale_min_and_minmax_using_2(self) -> None:

        interactions = [
            Interaction((7,2), [1], rewards=[1]),
            Interaction((1,9), [1], rewards=[1]),
            Interaction((8,3), [1], rewards=[1])
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

    def test_scale_0_and_2(self) -> None:

        interactions = [
            Interaction((8,2), [1], rewards=[1]),
            Interaction((4,4), [1], rewards=[1]),
            Interaction((2,6), [1], rewards=[1])
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

    def test_scale_mean_and_std(self) -> None:

        interactions = [
            Interaction((8,2), [1], rewards=[1]),
            Interaction((4,4), [1], rewards=[1]),
            Interaction((0,6), [1], rewards=[1])
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

    def test_scale_med_and_iqr(self) -> None:

        interactions = [
            Interaction((8,2), [1], rewards=[1]),
            Interaction((4,4), [1], rewards=[1]),
            Interaction((0,6), [1], rewards=[1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="med",scale="iqr").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0].context)
        self.assertEqual((4,4), mem_interactions[1].context)
        self.assertEqual((0,6), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(( 4/8,-2/4), scl_interactions[0].context)
        self.assertEqual(( 0/8, 0/4), scl_interactions[1].context)
        self.assertEqual((-4/8, 2/4), scl_interactions[2].context)

    def test_scale_med_and_iqr_0(self) -> None:

        interactions = [
            Interaction((8,2), [1], rewards=[1]),
            Interaction((4,2), [1], rewards=[1]),
            Interaction((0,2), [1], rewards=[1])
        ]

        mem_interactions = interactions
        scl_interactions = list(Scale(shift="med",scale="iqr").filter(interactions))

        self.assertEqual((8,2), mem_interactions[0].context)
        self.assertEqual((4,2), mem_interactions[1].context)
        self.assertEqual((0,2), mem_interactions[2].context)

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual(( 4/8, 0), scl_interactions[0].context)
        self.assertEqual(( 0/8, 0), scl_interactions[1].context)
        self.assertEqual((-4/8, 0), scl_interactions[2].context)

    def test_scale_min_and_minmax_with_str(self) -> None:

        interactions = [
            Interaction((7,2,"A"), [1], rewards=[1]),
            Interaction((1,9,"B"), [1], rewards=[1]),
            Interaction((8,3,"C"), [1], rewards=[1])
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

    def test_scale_min_and_minmax_with_nan(self) -> None:

        interactions = [
            Interaction((float('nan'), 2           ), [1], rewards=[1]),
            Interaction((1           , 9           ), [1], rewards=[1]),
            Interaction((8           , float('nan')), [1], rewards=[1])
        ]
 
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertTrue(isnan(scl_interactions[0].context[0]))
        self.assertEqual(0,   scl_interactions[0].context[1])

        self.assertEqual((0, 1), scl_interactions[1].context)

        self.assertEqual(1  , scl_interactions[2].context[0])
        self.assertTrue(isnan(scl_interactions[2].context[1]))

    def test_scale_min_and_minmax_with_nan(self) -> None:

        interactions = [
            Interaction(("A", 2  ), [1], rewards=[1]),
            Interaction((1  , 9  ), [1], rewards=[1]),
            Interaction((8  , "B"), [1], rewards=[1])
        ]
 
        scl_interactions = list(Scale().filter(interactions))

        self.assertEqual(3, len(scl_interactions))

        self.assertEqual("A", scl_interactions[0].context[0])
        self.assertEqual(0  , scl_interactions[0].context[1])
        
        self.assertEqual((0, 1), scl_interactions[1].context)
        
        self.assertEqual(1  , scl_interactions[2].context[0])
        self.assertTrue("B" , scl_interactions[2].context[1])

    def test_params(self):
        self.assertEqual({"scale_shift":"mean","scale_scale":"std","scale_using":None}, Scale(shift="mean",scale="std").params)
        self.assertEqual({"scale_shift":2,"scale_scale":1/2,"scale_using":None}, Scale(shift=2,scale=1/2).params)
        self.assertEqual({"scale_shift":2,"scale_scale":1/2,"scale_using":10}, Scale(shift=2,scale=1/2,using=10).params)

class Cycle_tests(unittest.TestCase):

    def test_after_0(self) -> None:

        interactions = [
            Interaction((7,2), [1,2], rewards=[1,3]),
            Interaction((1,9), [1,2], rewards=[1,4]),
            Interaction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle().filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].reveals)
        self.assertEqual([1,4], mem_interactions[1].reveals)
        self.assertEqual([1,5], mem_interactions[2].reveals)
        self.assertEqual([1,3], mem_interactions[0].rewards)
        self.assertEqual([1,4], mem_interactions[1].rewards)
        self.assertEqual([1,5], mem_interactions[2].rewards)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([3,1], cyc_interactions[0].reveals)
        self.assertEqual([4,1], cyc_interactions[1].reveals)
        self.assertEqual([5,1], cyc_interactions[2].reveals)
        self.assertEqual([3,1], cyc_interactions[0].rewards)
        self.assertEqual([4,1], cyc_interactions[1].rewards)
        self.assertEqual([5,1], cyc_interactions[2].rewards)

    def test_after_1(self) -> None:

        interactions = [
            Interaction((7,2), [1,2], rewards=[1,3]),
            Interaction((1,9), [1,2], rewards=[1,4]),
            Interaction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=1).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].reveals)
        self.assertEqual([1,4], mem_interactions[1].reveals)
        self.assertEqual([1,5], mem_interactions[2].reveals)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].reveals)
        self.assertEqual([4,1], cyc_interactions[1].reveals)
        self.assertEqual([5,1], cyc_interactions[2].reveals)

    def test_after_2(self) -> None:

        interactions = [
            Interaction((7,2), [1,2], rewards=[1,3]),
            Interaction((1,9), [1,2], rewards=[1,4]),
            Interaction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=2).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].reveals)
        self.assertEqual([1,4], mem_interactions[1].reveals)
        self.assertEqual([1,5], mem_interactions[2].reveals)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].reveals)
        self.assertEqual([1,4], cyc_interactions[1].reveals)
        self.assertEqual([5,1], cyc_interactions[2].reveals)

    def test_after_10(self) -> None:

        interactions = [
            Interaction((7,2), [1,2], rewards=[1,3]),
            Interaction((1,9), [1,2], rewards=[1,4]),
            Interaction((8,3), [1,2], rewards=[1,5])
        ]

        mem_interactions = interactions
        cyc_interactions = list(Cycle(after=10).filter(mem_interactions))

        self.assertEqual([1,3], mem_interactions[0].reveals)
        self.assertEqual([1,4], mem_interactions[1].reveals)
        self.assertEqual([1,5], mem_interactions[2].reveals)

        self.assertEqual(3, len(cyc_interactions))

        self.assertEqual([1,3], cyc_interactions[0].reveals)
        self.assertEqual([1,4], cyc_interactions[1].reveals)
        self.assertEqual([1,5], cyc_interactions[2].reveals)

    def test_params(self):
        self.assertEqual({"cycle_after":0 }, Cycle().params)
        self.assertEqual({"cycle_after":2 }, Cycle(2).params)

class Impute_tests(unittest.TestCase):

    def test_impute_nothing(self) -> None:

        interactions = [
            Interaction((7,2), [1], rewards=[1]),
            Interaction((1,9), [1], rewards=[1]),
            Interaction((8,3), [1], rewards=[1])
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

    def test_impute_mean(self) -> None:

        interactions = [
            Interaction((7           , 2           ), [1], rewards=[1]),
            Interaction((float('nan'), float('nan')), [1], rewards=[1]),
            Interaction((8           , 3           ), [1], rewards=[1])
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

    def test_impute_med(self) -> None:

        interactions = [
            Interaction((7           , 2           ), [1], rewards=[1]),
            Interaction((7           , 2           ), [1], rewards=[1]),
            Interaction((float('nan'), float('nan')), [1], rewards=[1]),
            Interaction((8           , 3           ), [1], rewards=[1])
        ]

        imp_interactions = list(Impute("median").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual((7, 2), imp_interactions[0].context)
        self.assertEqual((7, 2), imp_interactions[1].context)
        self.assertEqual((7, 2), imp_interactions[2].context)
        self.assertEqual((8, 3), imp_interactions[3].context)

    def test_impute_med_with_str(self) -> None:

        interactions = [
            Interaction((7           , 2           , "A"), [1], rewards=[1]),
            Interaction((7           , 2           , "A"), [1], rewards=[1]),
            Interaction((float('nan'), float('nan'), "A"), [1], rewards=[1]),
            Interaction((8           , 3           , "A"), [1], rewards=[1])
        ]

        imp_interactions = list(Impute("median").filter(interactions))

        self.assertEqual(4, len(imp_interactions))

        self.assertEqual((7, 2, "A"), imp_interactions[0].context)
        self.assertEqual((7, 2, "A"), imp_interactions[1].context)
        self.assertEqual((7, 2, "A"), imp_interactions[2].context)
        self.assertEqual((8, 3, "A"), imp_interactions[3].context)

    def test_params(self):
        self.assertEqual({"scale_shift":"mean","scale_scale":"std","scale_using":None}, Scale(shift="mean",scale="std").params)
        self.assertEqual({"scale_shift":2,"scale_scale":1/2,"scale_using":None}, Scale(shift=2,scale=1/2).params)
        self.assertEqual({"scale_shift":2,"scale_scale":1/2,"scale_using":10}, Scale(shift=2,scale=1/2,using=10).params)


if __name__ == '__main__':
    unittest.main()