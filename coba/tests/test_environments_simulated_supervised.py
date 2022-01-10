import unittest.mock
import unittest

from coba.pipes        import ArffReader, ListIO
from coba.contexts     import CobaContext, CobaContext, NullLogger
from coba.environments import SupervisedSimulation

CobaContext.logger = NullLogger()

class SupervisedSimulation_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'super_source': "XY", 'super_type':'C'},SupervisedSimulation([1,2],[1,2]).params)

    def test_source_reader_classification(self):

        source = ListIO("""
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
        """.splitlines())

        interactions = list(SupervisedSimulation(source, ArffReader(), "coli").read())

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

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

    def test_source_reader_regression_less_than_10(self):

        source = ListIO("""
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
        """.splitlines())

        interactions = list(SupervisedSimulation(source, ArffReader(), "pH", label_type="R").read())

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

        self.assertEqual([0 , 1, .5], interactions[0].kwargs["rewards"])
        self.assertEqual([.5, .5, 1], interactions[1].kwargs["rewards"])
        self.assertEqual([1 , 0, .5], interactions[2].kwargs["rewards"])

    def test_source_reader_too_large_take(self):

        source = ListIO("""
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
        """.splitlines())

        interactions = list(SupervisedSimulation(source, ArffReader(), "coli", take=5).read())

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

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

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

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

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

        self.assertEqual([1,1,1,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0,0,1], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0,0,0], interactions[2].kwargs["rewards"])

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
            self.assertEqual([1-(abs((a.index(1)+1)/11-i/11)) for a in interactions[i].actions], interactions[i].kwargs["rewards"])

    def test_X_Y_too_large_take(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [2,2,1]

        interactions = list(SupervisedSimulation(features, labels, take=4).read())

        self.assertEqual(len(interactions), 0)

    def test_X_Y_empty(self):
        features = []
        labels   = []

        interactions = list(SupervisedSimulation(features, labels).read())

        self.assertEqual(len(interactions), 0)

if __name__ == '__main__':
    unittest.main()
