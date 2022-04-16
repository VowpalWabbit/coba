import unittest.mock
import unittest

from coba.pipes        import ListSource
from coba.contexts     import CobaContext, CobaContext, NullLogger
from coba.environments import SupervisedSimulation, CsvSource, ArffSource, LibSvmSource, ManikSource

CobaContext.logger = NullLogger()

class SupervisedSimulation_Tests(unittest.TestCase):

    def test_params(self):
        expected_params = {'source': "[X,Y]", 'label_type':'C', "type": "SupervisedSimulation"}
        self.assertEqual(expected_params, SupervisedSimulation([1,2],[1,2]).params)

    def test_source_reader_classification(self):

        source = ArffSource(ListSource("""
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {yes, no}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.3,27,1020,1,yes
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "coli").read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)) , interactions[0].context)
        self.assertEqual((8.2,29,1180,(0,1)) , interactions[1].context)
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2].context)
        
        self.assertEqual([(0,1),(1,0)], interactions[0].actions)
        self.assertEqual([(0,1),(1,0)], interactions[1].actions)
        self.assertEqual([(0,1),(1,0)], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].rewards)
        self.assertEqual([0,1], interactions[1].rewards)
        self.assertEqual([1,0], interactions[2].rewards)

    def test_source_reader_regression_less_than_10(self):

        source = ArffSource(ListSource("""
            @relation weather

            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {yes, no}

            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.3,27,1020,1,yes
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "pH", label_type="R").read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((27,1410,(1,0),(0,1)), interactions[0].context)
        self.assertEqual((29,1180,(1,0),(0,1)), interactions[1].context)
        self.assertEqual((27,1020,(0,1),(1,0)), interactions[2].context)

        self.assertEqual([(0,0,1),(1,0,0),(0,1,0)], interactions[0].actions)
        self.assertEqual([(0,0,1),(1,0,0),(0,1,0)], interactions[1].actions)
        self.assertEqual([(0,0,1),(1,0,0),(0,1,0)], interactions[2].actions)

        self.assertEqual([0 , 1, .5], interactions[0].rewards)
        self.assertEqual([.5, .5, 1], interactions[1].rewards)
        self.assertEqual([1 , 0, .5], interactions[2].rewards)

    def test_source_reader_too_large_take_no_min(self):

        source = ArffSource(ListSource("""
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {yes, no}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.3,27,1020,1,yes
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "coli", take=(None,5)).read())

        self.assertEqual(len(interactions), 3)

    def test_source_reader_too_large_take_exact_min(self):

        source = ArffSource(ListSource("""
            @relation weather
            
            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {yes, no}
            
            @data
            8.1,27,1410,2,no
            8.2,29,1180,2,no
            8.3,27,1020,1,yes
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "coli", take=(5,5)).read())

        self.assertEqual(len(interactions), 0)

    def test_X_Y_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [2,2,1]

        interactions = list(SupervisedSimulation(features, labels).read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0].context)
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1].context)
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2].context)

        self.assertEqual([1,2], interactions[0].actions)
        self.assertEqual([1,2], interactions[1].actions)
        self.assertEqual([1,2], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].rewards)
        self.assertEqual([0,1], interactions[1].rewards)
        self.assertEqual([1,0], interactions[2].rewards)

    def test_X_Y_multiclass_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [2,2,1]

        interactions = list(SupervisedSimulation(features, labels).read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0].context)
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1].context)
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2].context)

        self.assertEqual([1,2], interactions[0].actions)
        self.assertEqual([1,2], interactions[1].actions)
        self.assertEqual([1,2], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].rewards)
        self.assertEqual([0,1], interactions[1].rewards)
        self.assertEqual([1,0], interactions[2].rewards)

    def test_X_Y_multilabel_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [[1,2,3,4],[1,2],[1]]

        interactions = list(SupervisedSimulation(features, labels).read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd.context)    #make sure these are hashable
            hash(rnd.actions[0]) #make sure these are hashable
            hash(rnd.actions[1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0].context)
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1].context)
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2].context)

        self.assertEqual([1,4,3,2], interactions[0].actions)
        self.assertEqual([1,4,3,2], interactions[1].actions)
        self.assertEqual([1,4,3,2], interactions[2].actions)

        self.assertEqual([1,1,1,1], interactions[0].rewards)
        self.assertEqual([1,0,0,1], interactions[1].rewards)
        self.assertEqual([1,0,0,0], interactions[2].rewards)

    def test_X_Y_regression_more_than_10(self):
        features = list(range(12))
        labels   = list(range(12))

        interactions = list(SupervisedSimulation(features, labels, label_type='R').read())

        self.assertEqual(len(interactions), 12)

        self.assertEqual(0 , interactions[0].context)
        self.assertEqual(1 , interactions[1].context)
        self.assertEqual(2 , interactions[2].context)
        self.assertEqual(3 , interactions[3].context)
        self.assertEqual(4 , interactions[4].context)
        self.assertEqual(5 , interactions[5].context)
        self.assertEqual(6 , interactions[6].context)
        self.assertEqual(7 , interactions[7].context)
        self.assertEqual(8 , interactions[8].context)
        self.assertEqual(9 , interactions[9].context)
        self.assertEqual(10, interactions[10].context)
        self.assertEqual(11, interactions[11].context)

        actions = [
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
            (0, 0, 1, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        ]

        self.assertEqual(actions, interactions[0].actions)
        self.assertEqual(actions, interactions[1].actions)
        self.assertEqual(actions, interactions[2].actions)
        self.assertEqual(actions, interactions[3].actions)
        self.assertEqual(actions, interactions[4].actions)
        self.assertEqual(actions, interactions[5].actions)
        self.assertEqual(actions, interactions[6].actions)
        self.assertEqual(actions, interactions[7].actions)
        self.assertEqual(actions, interactions[8].actions)
        self.assertEqual(actions, interactions[9].actions)
        self.assertEqual(actions, interactions[10].actions)
        self.assertEqual(actions, interactions[11].actions)

        for i in range(12):
            self.assertEqual([1-(abs((a.index(1)+1)/11-i/11)) for a in interactions[i].actions], interactions[i].rewards)

    def test_X_Y_too_large_take(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [2,2,1]

        interactions = list(SupervisedSimulation(features, labels, take=4).read())

        self.assertEqual(len(interactions), 3)

    def test_X_Y_empty(self):
        features = []
        labels   = []

        interactions = list(SupervisedSimulation(features, labels).read())

        self.assertEqual(len(interactions), 0)

class CsvSource_Tests(unittest.TestCase):

    def test_simple(self):
        self.assertEqual([["1","2","3"]], list(CsvSource(ListSource(["1,2,3"])).read()))
        self.assertEqual({}, CsvSource(ListSource(["1,2,3"])).params)
        self.assertEqual('{},{}', str(CsvSource(ListSource(["1,2,3"]))))


class ArffSource_Tests(unittest.TestCase):

    def test_simple(self):
        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@data",
            "1,  2",
            "2,  3",
        ]

        expected = [
            [1, 2],
            [2, 3]
        ]

        self.assertEqual(expected, list(ArffSource(ListSource(lines)).read()))
        self.assertEqual({}, ArffSource(ListSource(lines)).params)
        self.assertEqual('{},{}', str(ArffSource(ListSource(lines))))

class LibsvmSource_Tests(unittest.TestCase):

    def test_simple(self):
        lines = [
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(LibSvmSource(ListSource(lines)).read()))
        self.assertEqual({}, LibSvmSource(ListSource(lines)).params)
        self.assertEqual('{},{}', str(LibSvmSource(ListSource(lines))))

class ManikSource_Tests(unittest.TestCase):

    def test_simple(self):
        lines = [
            "meta line",
            "0 1:2 2:3",
            "1 1:1 2:1",
            "2 2:1",
            "1 1:1",
        ]

        expected = [
            ({1:2, 2:3} ,['0']),
            ({1:1, 2:1}, ['1']),
            ({     2:1}, ['2']),
            ({1:1     }, ['1'])
        ]

        self.assertEqual(expected, list(ManikSource(ListSource(lines)).read()))
        self.assertEqual({}, ManikSource(ListSource(lines)).params)
        self.assertEqual('{},{}', str(ManikSource(ListSource(lines))))

if __name__ == '__main__':
    unittest.main()
