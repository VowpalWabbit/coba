import unittest
import pickle

from pathlib import Path

from coba.typing import List
from coba.exceptions import CobaException
from coba.pipes import MemoryIO, DiskIO, NullIO
from coba.contexts import CobaContext, NullLogger
from coba.environments.simulations import ReaderSimulation
from coba.environments import (
    SimulatedInteraction, MemorySimulation, ClassificationSimulation,
    LambdaSimulation, CsvSimulation, ArffSimulation, LibsvmSimulation,
    LinearSyntheticSimulation, RegressionSimulation, ManikSimulation,
    LocalSyntheticSimulation
)
from coba.random import CobaRandom

CobaContext.logger = NullLogger()

class ClassificationSimulation_Tests(unittest.TestCase):

    def test_constructor_with_incorrect_param_count(self) -> None:

        with self.assertRaises(CobaException):
            ClassificationSimulation()

        with self.assertRaises(CobaException):
            ClassificationSimulation(1,2,3)

    def test_constructor_with_dense_data1(self) -> None:
        features   = [1,2]
        labels     = [0,.5]

        for sim in [ClassificationSimulation(features, labels), ClassificationSimulation(zip(features, labels))]:
            interactions = list(sim.read())

            self.assertEqual(2, len(interactions))

            self.assertEqual(1, interactions[0].context)
            self.assertEqual(2, interactions[1].context)

            self.assertEqual([0,.5], interactions[0].actions)
            self.assertEqual([0,.5], interactions[1].actions)

            self.assertEqual([1,0], interactions[0].kwargs["rewards"])
            self.assertEqual([0,1], interactions[1].kwargs["rewards"])

    def test_constructor_with_dense_data2(self) -> None:
        features   = ["a","b"]
        labels     = ["good","bad"]

        for sim in [ClassificationSimulation(features, labels), ClassificationSimulation(zip(features, labels))]:
            interactions = list(sim.read())

            self.assertEqual(2, len(interactions))

            self.assertEqual("a", interactions[0].context)
            self.assertEqual("b", interactions[1].context)

            self.assertEqual(["bad","good"], interactions[0].actions)
            self.assertEqual(["bad","good"], interactions[1].actions)

            self.assertEqual([0,1], interactions[0].kwargs["rewards"])
            self.assertEqual([1,0], interactions[1].kwargs["rewards"])
    
    def test_constructor_with_dense_data3(self) -> None:
        features   = [(1,2),(3,4)]
        labels     = ["good","bad"]

        for sim in [ClassificationSimulation(features, labels), ClassificationSimulation(zip(features, labels))]:
            interactions = list(sim.read())

            self.assertEqual(2, len(interactions))

            self.assertEqual((1,2), interactions[0].context)
            self.assertEqual((3,4), interactions[1].context)

            self.assertEqual(["bad","good"], interactions[0].actions)
            self.assertEqual(["bad","good"], interactions[1].actions)

            self.assertEqual([0,1], interactions[0].kwargs["rewards"])
            self.assertEqual([1,0], interactions[1].kwargs["rewards"])

    def test_constructor_with_sparse_data(self) -> None:
        features   = [{0:1},{0:2}]
        labels     = ["good","bad"]

        for sim in [ClassificationSimulation(features, labels), ClassificationSimulation(zip(features, labels))]:
            interactions = list(sim.read())

            self.assertEqual(2, len(interactions))

            self.assertEqual({0:1}, interactions[0].context)
            self.assertEqual({0:2}, interactions[1].context)

            self.assertEqual(["bad","good"], interactions[0].actions)
            self.assertEqual(["bad","good"], interactions[1].actions)

            self.assertEqual([0,1], interactions[0].kwargs["rewards"])
            self.assertEqual([1,0], interactions[1].kwargs["rewards"])

    def test_constructor_with_empty_data(self) -> None:
        
        for sim in [ClassificationSimulation([]), ClassificationSimulation([],[])]:
            self.assertEqual([],list(sim.read()))

    def test_params(self):
        self.assertEqual({}, ClassificationSimulation([]).params)

class RegressionSimulation_Tests(unittest.TestCase):

    def test_constructor_with_incorrect_param_count(self) -> None:

        with self.assertRaises(CobaException):
            RegressionSimulation()

        with self.assertRaises(CobaException):
            RegressionSimulation(1,2,3)

    def test_constructor_with_dense_data(self) -> None:
        features   = [1,2]
        labels     = [0,.5]

        for sim in [RegressionSimulation(features, labels), RegressionSimulation(zip(features, labels))]: 

            interactions = list(sim.read())

            self.assertEqual(2, len(interactions))

            self.assertEqual(1, interactions[0].context)
            self.assertEqual(2, interactions[1].context)

            self.assertEqual([0,.5], interactions[0].actions)
            self.assertEqual([0,.5], interactions[1].actions)

            self.assertEqual([1,.5], interactions[0].kwargs["rewards"])
            self.assertEqual([.5,1], interactions[1].kwargs["rewards"])

    def test_constructor_with_empty_data(self) -> None:
        
        for sim in [RegressionSimulation([]), RegressionSimulation([],[])]:
            self.assertEqual([], list(sim.read()))
    
    def test_constructor_with_sparse_data(self) -> None:
        features   = [{0:1},{0:2}]
        labels     = [0,.5]

        for sim in [RegressionSimulation(features, labels), RegressionSimulation(zip(features, labels))]: 
        
            interactions = list(sim.read())

            self.assertEqual(2, len(interactions))

            self.assertEqual({0:1}, interactions[0].context)
            self.assertEqual({0:2}, interactions[1].context)

            self.assertEqual([0,.5], interactions[0].actions)
            self.assertEqual([0,.5], interactions[1].actions)

            self.assertEqual([1,.5], interactions[0].kwargs["rewards"])
            self.assertEqual([.5,1], interactions[1].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({}, RegressionSimulation([]).params)

class MemorySimulation_Tests(unittest.TestCase):

    def test_interactions(self):

        simulation   = MemorySimulation([SimulatedInteraction(1, [1,2,3], rewards=[0,1,2]), SimulatedInteraction(2, [4,5,6], rewards=[2,3,4])])
        interactions = list(simulation.read())

        self.assertEqual(interactions[0], interactions[0])
        self.assertEqual(interactions[1], interactions[1])

    def test_params(self):
        self.assertEqual({}, MemorySimulation([]).params)
        self.assertEqual({'A':1}, MemorySimulation([],params={'A':1}).params)

    def test_str(self):
        self.assertEqual("MemorySimulation", str(MemorySimulation([])))
        self.assertEqual("MySimulation", str(MemorySimulation([], str="MySimulation")))

class LambdaSimulation_Tests(unittest.TestCase):

    def test_n_interactions_2_seed_none(self):
        
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R)
        interactions = list(simulation.read())

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_n_interactions_none_seed_none(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation   = LambdaSimulation(None,C,A,R)
        interactions = iter(simulation.read())

        interaction = next(interactions)

        self.assertEqual(1      , interaction.context)
        self.assertEqual([1,2,3], interaction.actions)
        self.assertEqual([0,1,2], interaction.kwargs["rewards"])

        interaction = next(interactions)

        self.assertEqual(2      , interaction.context)
        self.assertEqual([4,5,6], interaction.actions)
        self.assertEqual([2,3,4], interaction.kwargs["rewards"])

    def test_n_interactions_2_seed_1(self):
        
        def C(i:int, rng: CobaRandom) -> int:
            return [1,2][i]

        def A(i:int,c:int, rng: CobaRandom) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int, rng: CobaRandom) -> int:
            return a-c

        simulation = LambdaSimulation(2,C,A,R,seed=1)
        interactions = list(simulation.read())

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_params(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]

        def R(i:int,c:int,a:int) -> int:
            return a-c

        self.assertEqual({}, LambdaSimulation(2,C,A,R).params)

    def test_pickle_n_interactions_2(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        simulation = pickle.loads(pickle.dumps(LambdaSimulation(2,C,A,R)))
        interactions = list(simulation.read())

        self.assertEqual("LambdaSimulation",str(simulation))
        self.assertEqual({}, simulation.params)

        self.assertEqual(len(interactions), 2)

        self.assertEqual(1      , interactions[0].context)
        self.assertEqual([1,2,3], interactions[0].actions)
        self.assertEqual([0,1,2], interactions[0].kwargs["rewards"])

        self.assertEqual(2      , interactions[1].context)
        self.assertEqual([4,5,6], interactions[1].actions)
        self.assertEqual([2,3,4], interactions[1].kwargs["rewards"])

    def test_pickle_n_interactions_none(self):
        def C(i:int) -> int:
            return [1,2][i]

        def A(i:int,c:int) -> List[int]:
            return [[1,2,3],[4,5,6]][i]
        
        def R(i:int,c:int,a:int) -> int:
            return a-c

        with self.assertRaises(CobaException) as e:
            pickle.loads(pickle.dumps(LambdaSimulation(None,C,A,R)))
        
        self.assertIn("In general LambdaSimulation", str(e.exception))

class LinearSyntheticSimulation_Tests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(500, len(list(LinearSyntheticSimulation().read())))

    def test_params(self):
        env = LinearSyntheticSimulation(100,2,3,4,0,["xa"],2)

        self.assertEqual(2     , env.params['n_A'])
        self.assertEqual(3     , env.params['n_C_phi'])
        self.assertEqual(4     , env.params['n_A_phi'])
        self.assertEqual(0     , env.params['r_noise'])
        self.assertEqual(['xa'], env.params['X'])
        self.assertEqual(2     , env.params['seed'])
    
    def test_str(self):
        self.assertEqual("LinearSynth(A=2,c=3,a=4,X=['xa'],seed=2)", str(LinearSyntheticSimulation(100,2,3,4,0,["xa"],2)))

    def test_pickle(self):
        env = pickle.loads(pickle.dumps(LinearSyntheticSimulation(100,2,3,4,0,["xa"],2)))
        
        self.assertEqual(2     , env.params['n_A'])
        self.assertEqual(3     , env.params['n_C_phi'])
        self.assertEqual(4     , env.params['n_A_phi'])
        self.assertEqual(0     , env.params['r_noise'])
        self.assertEqual(['xa'], env.params['X'])
        self.assertEqual(2     , env.params['seed'])
        self.assertEqual("LinearSynth(A=2,c=3,a=4,X=['xa'],seed=2)", str(env))
        self.assertEqual(100, len(list(env.read())))

class LocalSyntheticSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):
        self.assertEqual(500, len(list(LocalSyntheticSimulation().read())))

    def test_params(self):
        env = LocalSyntheticSimulation(100,100,3,4,2)

        self.assertEqual(4  , env.params['n_A'])
        self.assertEqual(100, env.params['n_C'])
        self.assertEqual(3  , env.params['n_C_phi'])
        self.assertEqual(2  , env.params['seed'])

    def test_str(self):
        self.assertEqual("LocalSynth(A=4,C=100,c=3,seed=2)", str(LocalSyntheticSimulation(200,100,3,4,2)))

class ReaderSimulation_Tests(unittest.TestCase):

    def test_params(self):
        self.assertEqual({'source': 'abc'}, ReaderSimulation(None, DiskIO("abc"), None).params)
        self.assertEqual({'source': 'memory'}, ReaderSimulation(None, MemoryIO(), None).params)
        self.assertEqual({'source': 'NullIO'}, ReaderSimulation(None, NullIO(), None).params)

class CsvSimulation_Tests(unittest.TestCase):

    def test_memory_source(self):
        source       = MemoryIO(['a,b,c','1,2,3','4,5,6'])
        simulation   = CsvSimulation(source,'c')
        interactions = list(simulation.read())

        self.assertEqual(2, len(interactions))
        
        self.assertEqual(('1','2'), interactions[0].context)
        self.assertEqual(('4','5'), interactions[1].context)

        self.assertEqual(['3','6'], interactions[0].actions)
        self.assertEqual(['3','6'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])
    
    def test_file_path(self):

        Path("coba/tests/.temp/sim.csv").write_text("""
            a,b,c
            1,2,3
            4,5,6
        """)

        try:
            simulation   = CsvSimulation("coba/tests/.temp/sim.csv",'c')
            interactions = list(simulation.read())

            self.assertEqual(2, len(interactions))
            
            self.assertEqual(('1','2'), interactions[0].context)
            self.assertEqual(('4','5'), interactions[1].context)

            self.assertEqual(['3','6'], interactions[0].actions)
            self.assertEqual(['3','6'], interactions[1].actions)

            self.assertEqual([1,0], interactions[0].kwargs["rewards"])
            self.assertEqual([0,1], interactions[1].kwargs["rewards"])

        finally:
            Path("coba/tests/.temp/sim.csv").unlink()

    def test_params(self):
        self.assertEqual({'csv':'coba/tests/.temp/sim.csv'}, CsvSimulation('coba/tests/.temp/sim.csv', 'c').params)

class ArffSimulation_Tests(unittest.TestCase):

    def test_simple(self):

        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B numeric",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
        ]

        source       = MemoryIO(lines)
        simulation   = ArffSimulation(source,'c')
        interactions = list(simulation.read())

        self.assertEqual(2, len(interactions))
        
        self.assertEqual((1,2), interactions[0].context)
        self.assertEqual((2,3), interactions[1].context)

        self.assertEqual(['0','class_B'], interactions[0].actions)
        self.assertEqual(['0','class_B'], interactions[1].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0], interactions[1].kwargs["rewards"])

    def test_one_hot(self):

        lines = [
            "@relation news20",
            "@attribute a numeric",
            "@attribute B {0, 1, 2, 3}",
            "@attribute c {0, class_B, class_C, class_D}",
            "@data",
            "1,2,class_B",
            "2,3,0",
            "3,1,0"
        ]

        source       = MemoryIO(lines)
        simulation   = ArffSimulation(source,'c',)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))
        
        self.assertEqual((1,(0,0,1,0)), interactions[0].context)
        self.assertEqual((2,(0,0,0,1)), interactions[1].context)
        self.assertEqual((3,(0,1,0,0)), interactions[2].context)

        self.assertEqual(['0','class_B'], interactions[0].actions)
        self.assertEqual(['0','class_B'], interactions[1].actions)
        self.assertEqual(['0','class_B'], interactions[2].actions)

        self.assertEqual([0,1], interactions[0].kwargs["rewards"])
        self.assertEqual([1,0], interactions[1].kwargs["rewards"])
        self.assertEqual([1,0], interactions[2].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'arff':'coba/tests/.temp/sim.csv'}, ArffSimulation('coba/tests/.temp/sim.csv', 'c').params)

class LibsvmSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):

        lines = [
            "0 4:2 5:3",
            "1 1:1 2:1",
            "1 3:4"
        ]

        source       = MemoryIO(lines)
        simulation   = LibsvmSimulation(source)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))

        self.assertEqual({4:2,5:3}, interactions[0].context)
        self.assertEqual({1:1,2:1}, interactions[1].context)
        self.assertEqual({3:4    }, interactions[2].context)

        self.assertEqual(['0','1'], interactions[0].actions)
        self.assertEqual(['0','1'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'libsvm':'coba/tests/.temp/sim.csv'}, LibsvmSimulation('coba/tests/.temp/sim.csv').params)

class ManikSimulation_Tests(unittest.TestCase):
    
    def test_simple(self):

        lines = [
            "Total_Points Num_Features Num_Labels",
            "0 4:2 5:3",
            "1 1:1 2:1",
            "1 3:4"
        ]

        source       = MemoryIO(lines)
        simulation   = ManikSimulation(source)
        interactions = list(simulation.read())

        self.assertEqual(3, len(interactions))

        self.assertEqual({4:2,5:3}, interactions[0].context)
        self.assertEqual({1:1,2:1}, interactions[1].context)
        self.assertEqual({3:4    }, interactions[2].context)

        self.assertEqual(['0','1'], interactions[0].actions)
        self.assertEqual(['0','1'], interactions[1].actions)

        self.assertEqual([1,0], interactions[0].kwargs["rewards"])
        self.assertEqual([0,1], interactions[1].kwargs["rewards"])

    def test_params(self):
        self.assertEqual({'manik':'coba/tests/.temp/sim.csv'}, ManikSimulation('coba/tests/.temp/sim.csv').params)

if __name__ == '__main__':
    unittest.main()
