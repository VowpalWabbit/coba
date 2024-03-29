import unittest.mock
import unittest
import math

from coba.primitives   import Categorical
from coba.pipes        import IterableSource
from coba.context      import CobaContext, CobaContext, NullLogger
from coba.environments import SupervisedSimulation, CsvSource, ArffSource, LibSvmSource, ManikSource

CobaContext.logger = NullLogger()

class SupervisedSimulation_Tests(unittest.TestCase):

    def test_params_pre_read(self):
        expected_params = {'source': "[X,Y]", 'label_type':'C', "env_type": "SupervisedSimulation"}
        params = SupervisedSimulation([1,2],[1,2],label_type="C").params
        self.assertEqual(expected_params, params)

    def test_params_post_read(self):
        expected_params = {'source': "[X,Y]", 'label_type':'C', "env_type": "SupervisedSimulation", 'n_actions':2}
        env = SupervisedSimulation([1,2],[1,2],label_type="C")
        list(env.read())
        self.assertEqual(expected_params, env.params)

        expected_params = {'source': "[X,Y]", 'label_type':'C', "env_type": "SupervisedSimulation", 'n_actions':2}
        env = SupervisedSimulation([1,2],[Categorical(1,[1,2]),Categorical(2,[1,2])],label_type="C")
        list(env.read())
        self.assertEqual(expected_params, env.params)

        expected_params = {'source': "[X,Y]", 'label_type':'R', "env_type": "SupervisedSimulation", 'n_actions':float('inf')}
        env = SupervisedSimulation([1,2],[1,2],label_type="R")
        list(env.read())
        self.assertEqual(expected_params, env.params)

    def test_source_reader_classification(self):

        source = ArffSource(IterableSource("""
            @relation weather

            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {y, n}

            @data
            8.1,27,1410,2,n
            8.2,29,1180,2,n
            8.3,27,1020,1,y
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "coli").read())

        self.assertEqual(len(interactions), 3)

        self.assertEqual([8.1,27,1410,Categorical("n",["y","n"])], interactions[0]['context'])
        self.assertEqual([8.2,29,1180,Categorical("n",["y","n"])], interactions[1]['context'])
        self.assertEqual([8.3,27,1020,Categorical("y",["y","n"])], interactions[2]['context'])

        self.assertEqual([Categorical('2',["2","1"]),Categorical('1',["2","1"])], interactions[0]['actions'])
        self.assertEqual([Categorical('2',["2","1"]),Categorical('1',["2","1"])], interactions[1]['actions'])
        self.assertEqual([Categorical('2',["2","1"]),Categorical('1',["2","1"])], interactions[2]['actions'])

        self.assertEqual([1,0], list(map(interactions[0]['rewards'],['2','1'])))
        self.assertEqual([1,0], list(map(interactions[1]['rewards'],['2','1'])))
        self.assertEqual([0,1], list(map(interactions[2]['rewards'],['2','1'])))

    def test_source_reader_regression_less_than_10(self):

        source = ArffSource(IterableSource("""
            @relation weather

            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {y, n}

            @data
            8.1,27,1410,2,n
            8.2,29,1180,2,n
            8.3,27,1020,1,y
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "pH", label_type="r").read())

        self.assertEqual(len(interactions), 3)

        self.assertEqual([27,1410,Categorical('2',["2","1"]),Categorical("n",["y","n"])], interactions[0]['context'])
        self.assertEqual([29,1180,Categorical('2',["2","1"]),Categorical("n",["y","n"])], interactions[1]['context'])
        self.assertEqual([27,1020,Categorical('1',["2","1"]),Categorical("y",["y","n"])], interactions[2]['context'])

        self.assertEqual([], interactions[0]['actions'])
        self.assertEqual([], interactions[1]['actions'])
        self.assertEqual([], interactions[2]['actions'])

        self.assertEqual([-1,0], list(map(interactions[0]['rewards'],[9.1,8.1])))
        self.assertEqual([-1,0], list(map(interactions[1]['rewards'],[9.2,8.2])))
        self.assertEqual([-1,0], list(map(interactions[2]['rewards'],[9.3,8.3])))

    def test_source_reader_too_large_take_no_min(self):

        source = ArffSource(IterableSource("""
            @relation weather

            @attribute pH real
            @attribute temperature real
            @attribute conductivity real
            @attribute coli {2, 1}
            @attribute play {y, n}

            @data
            8.1,27,1410,2,n
            8.2,29,1180,2,n
            8.3,27,1020,1,y
        """.splitlines()))

        interactions = list(SupervisedSimulation(source, "coli", take=5).read())

        self.assertEqual(len(interactions), 3)

    def test_X_Y_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [2,2,1]

        interactions = list(SupervisedSimulation(features, labels, label_type='C').read())

        self.assertEqual(len(interactions), 3)

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0]['context'])
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1]['context'])
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2]['context'])

        self.assertEqual([1,2], interactions[0]['actions'])
        self.assertEqual([1,2], interactions[1]['actions'])
        self.assertEqual([1,2], interactions[2]['actions'])

        self.assertEqual(2, interactions[0]['rewards'])
        self.assertEqual(2, interactions[1]['rewards'])
        self.assertEqual(1, interactions[2]['rewards'])

    def test_X_Y_string_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = ['1','-1','1']

        interactions = list(SupervisedSimulation(features, labels, label_type='C').read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd['context'])    #make sure these are hashable
            hash(rnd['actions'][0]) #make sure these are hashable
            hash(rnd['actions'][1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0]['context'])
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1]['context'])
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2]['context'])

        self.assertEqual(['-1','1'], interactions[0]['actions'])
        self.assertEqual(['-1','1'], interactions[1]['actions'])
        self.assertEqual(['-1','1'], interactions[2]['actions'])

        self.assertEqual('1' , interactions[0]['rewards'])
        self.assertEqual('-1', interactions[1]['rewards'])
        self.assertEqual('1' , interactions[2]['rewards'])

    def test_X_Y_multiclass_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [2,2,1]

        interactions = list(SupervisedSimulation(features, labels, label_type="C").read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd['context'])    #make sure these are hashable
            hash(rnd['actions'][0]) #make sure these are hashable
            hash(rnd['actions'][1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0]['context'])
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1]['context'])
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2]['context'])

        self.assertEqual([1,2], interactions[0]['actions'])
        self.assertEqual([1,2], interactions[1]['actions'])
        self.assertEqual([1,2], interactions[2]['actions'])

        self.assertEqual(2, interactions[0]['rewards'])
        self.assertEqual(2, interactions[1]['rewards'])
        self.assertEqual(1, interactions[2]['rewards'])

    def test_X_Y_multiclass_list_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [[2],[2],[1]]

        interactions = list(SupervisedSimulation(features, labels, label_type="C").read())

        self.assertEqual(len(interactions), 3)

        for rnd in interactions:

            hash(rnd['context'])    #make sure these are hashable
            hash(rnd['actions'][0]) #make sure these are hashable
            hash(rnd['actions'][1]) #make sure these are hashable

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0]['context'])
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1]['context'])
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2]['context'])

        self.assertEqual([1,2], interactions[0]['actions'])
        self.assertEqual([1,2], interactions[1]['actions'])
        self.assertEqual([1,2], interactions[2]['actions'])

        self.assertEqual(2, interactions[0]['rewards'])
        self.assertEqual(2, interactions[1]['rewards'])
        self.assertEqual(1, interactions[2]['rewards'])

    def test_X_Y_multilabel_classification(self):
        features = [(8.1,27,1410,(0,1)), (8.2,29,1180,(0,1)), (8.3,27,1020,(1,0))]
        labels   = [[1,2,3,4],[1,2],[1]]

        interactions = list(SupervisedSimulation(features, labels, "M").read())

        self.assertEqual(len(interactions), 3)

        self.assertEqual((8.1,27,1410,(0,1)), interactions[0]['context'])
        self.assertEqual((8.2,29,1180,(0,1)), interactions[1]['context'])
        self.assertEqual((8.3,27,1020,(1,0)), interactions[2]['context'])

        self.assertEqual([1,2,3,4], interactions[0]['actions'])
        self.assertEqual([1,2,3,4], interactions[1]['actions'])
        self.assertEqual([1,2,3,4], interactions[2]['actions'])

        self.assertEqual(1, interactions[0]['rewards']([1,2,3,4]))
        self.assertEqual(1, interactions[1]['rewards']([1,2]))
        self.assertEqual(1, interactions[2]['rewards']([1]))

    def test_X_Y_regression_more_than_10(self):
        features = list(range(12))
        labels   = list(range(12))

        interactions = list(SupervisedSimulation(features, labels, label_type='R').read())

        self.assertEqual(len(interactions), 12)

        self.assertEqual(0 , interactions[0]['context'])
        self.assertEqual(1 , interactions[1]['context'])
        self.assertEqual(2 , interactions[2]['context'])
        self.assertEqual(3 , interactions[3]['context'])
        self.assertEqual(4 , interactions[4]['context'])
        self.assertEqual(5 , interactions[5]['context'])
        self.assertEqual(6 , interactions[6]['context'])
        self.assertEqual(7 , interactions[7]['context'])
        self.assertEqual(8 , interactions[8]['context'])
        self.assertEqual(9 , interactions[9]['context'])
        self.assertEqual(10, interactions[10]['context'])
        self.assertEqual(11, interactions[11]['context'])

        actions = []

        self.assertEqual(actions, interactions[0]['actions'])
        self.assertEqual(actions, interactions[1]['actions'])
        self.assertEqual(actions, interactions[2]['actions'])
        self.assertEqual(actions, interactions[3]['actions'])
        self.assertEqual(actions, interactions[4]['actions'])
        self.assertEqual(actions, interactions[5]['actions'])
        self.assertEqual(actions, interactions[6]['actions'])
        self.assertEqual(actions, interactions[7]['actions'])
        self.assertEqual(actions, interactions[8]['actions'])
        self.assertEqual(actions, interactions[9]['actions'])
        self.assertEqual(actions, interactions[10]['actions'])
        self.assertEqual(actions, interactions[11]['actions'])

        for i in range(12):
            self.assertEqual(0, interactions[i]['rewards'](i))

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
        self.assertEqual([["1","2","3"]], list(CsvSource(IterableSource(["1,2,3"])).read()))
        self.assertEqual({}, CsvSource(IterableSource(["1,2,3"])).params)
        self.assertEqual('IterableSource() | CsvReader()', str(CsvSource(IterableSource(["1,2,3"]))))


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

        self.assertEqual(expected, list(map(list,ArffSource(IterableSource(lines)).read())))
        self.assertEqual({}, ArffSource(IterableSource(lines)).params)
        self.assertEqual('IterableSource() | ArffReader()', str(ArffSource(IterableSource(lines))))

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

        self.assertEqual(expected, list(LibSvmSource(IterableSource(lines)).read()))
        self.assertEqual({}, LibSvmSource(IterableSource(lines)).params)
        self.assertEqual('IterableSource() | LibsvmReader()', str(LibSvmSource(IterableSource(lines))))

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

        self.assertEqual(expected, list(ManikSource(IterableSource(lines)).read()))
        self.assertEqual({}, ManikSource(IterableSource(lines)).params)
        self.assertEqual('IterableSource() | ManikReader()', str(ManikSource(IterableSource(lines))))

if __name__ == '__main__':
    unittest.main()
