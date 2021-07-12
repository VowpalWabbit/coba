import unittest

from itertools import repeat

from coba.config import CobaConfig, NoneLogger
from coba.simulations import Interaction, MemorySimulation, Shuffle, Take, PCA, Sort

CobaConfig.Logger = NoneLogger()

class Shuffle_Tests(unittest.TestCase):
    
    def test_shuffle(self):
        interactions = list(map(Interaction, repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]] ))        
        shuffled_interactions = list(Shuffle(40).filter(interactions))

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

        self.assertEqual(3, len(shuffled_interactions))
        self.assertEqual(interactions[1], shuffled_interactions[0])
        self.assertEqual(interactions[2], shuffled_interactions[1])
        self.assertEqual(interactions[0], shuffled_interactions[2])

class Take_Tests(unittest.TestCase):
    
    def test_take1(self):

        interactions = list(map(Interaction, repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]))
        take_interactions = list(Take(1).filter(interactions))

        self.assertEqual(1, len(take_interactions))
        self.assertEqual(interactions[0], take_interactions[0])

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

    def test_take2(self):
        interactions = list(map(Interaction, repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]))
        take_interactions = list(Take(2).filter(interactions))

        self.assertEqual(2, len(take_interactions))
        self.assertEqual(interactions[0], take_interactions[0])
        self.assertEqual(interactions[1], take_interactions[1])

        self.assertEqual(3, len(interactions))
        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])
        self.assertEqual(interactions[2], interactions[2])

    def test_take3(self):
        interactions = list(map(Interaction, repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]] ))
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
        interactions = list(map(Interaction, repeat(1), repeat([1,2]), [[3,3],[4,4],[5,5]]))        
        take_interactions = list(Take(4).filter(interactions))

        self.assertEqual(3, len(interactions))
        self.assertEqual(0, len(take_interactions))

    def test_repr(self):
        self.assertEqual('{"Take":2}', str(Take(2)))
        self.assertEqual('{"Take":null}', str(Take(None)))

class Interaction_Tests(unittest.TestCase):

    def test_constructor_no_context(self) -> None:
        Interaction(None, [1,2], [1,2])

    def test_constructor_context(self) -> None:
        Interaction((1,2,3,4), [1,2], [1,2])

    def test_context_correct_1(self) -> None:
        self.assertEqual(None, Interaction(None, [1,2], [1,2]).context)

    def test_actions_correct_1(self) -> None:
        self.assertSequenceEqual([1,2], Interaction(None, [1,2], [1,2]).actions)

    def test_actions_correct_2(self) -> None:
        self.assertSequenceEqual(["A","B"], Interaction(None, ["A","B"], [1,2]).actions)

    def test_actions_correct_3(self) -> None:
        self.assertSequenceEqual([(1,2), (3,4)], Interaction(None, [(1,2), (3,4)], [1,2]).actions)

class PCA_Tests(unittest.TestCase):
    def test_PCA(self):        
        interactions = [
            Interaction((1,2), [1], [1]),
            Interaction((1,9), [1], [1]),
            Interaction((7,3), [1], [1])
        ]

        mem_interactions = interactions
        pca_interactions = list(PCA().filter(interactions))

        self.assertEqual((1,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((7,3), mem_interactions[2].context)

        self.assertNotEqual((1,2), pca_interactions[0].context)
        self.assertNotEqual((1,9), pca_interactions[1].context)
        self.assertNotEqual((7,3), pca_interactions[2].context)

    def test_repr(self):
        self.assertEqual('"PCA"', str(PCA()))

class Sort_tests(unittest.TestCase):

    def test_sort1(self) -> None:

        interactions = [
            Interaction((7,2), [1], [1]),
            Interaction((1,9), [1], [1]),
            Interaction((8,3), [1], [1])
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
            Interaction((1,2), [1], [1]),
            Interaction((1,9), [1], [1]),
            Interaction((1,3), [1], [1])
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
            Interaction((1,2), [1], [1]),
            Interaction((1,9), [1], [1]),
            Interaction((1,3), [1], [1])
        ]

        mem_interactions = interactions
        srt_interactions = list(Sort(*[0,1]).filter(mem_interactions))

        self.assertEqual((1,2), mem_interactions[0].context)
        self.assertEqual((1,9), mem_interactions[1].context)
        self.assertEqual((1,3), mem_interactions[2].context)

        self.assertEqual((1,2), srt_interactions[0].context)
        self.assertEqual((1,3), srt_interactions[1].context)
        self.assertEqual((1,9), srt_interactions[2].context)
    
    def test_repr(self):
        self.assertEqual('{"Sort":[0]}', str(Sort([0])))
        self.assertEqual('{"Sort":[1,2]}', str(Sort([1,2])))

if __name__ == '__main__':
    unittest.main()