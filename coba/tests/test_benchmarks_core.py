import unittest
import math

from pathlib import Path
from statistics import mean
from typing import cast

from coba.simulations import LambdaSimulation
from coba.pipes import Source, MemorySink
from coba.learners import Learner
from coba.config import CobaConfig, NoneLogger, IndentLogger

from coba.benchmarks.core import Benchmark

#for testing purposes
class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def family(self):
        return "Modulo"

    @property
    def params(self):
        return {"p":self._param}

    def predict(self, key, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, key, context, action, reward, probability):
        pass

class BrokenLearner(Learner):
    
    @property
    def family(self):
        return "Broken"

    @property
    def params(self):
        return {}

    def predict(self, key, context, actions):
        raise Exception("Broken Learner")

    def learn(self, key, context, action, reward, probability):
        pass

class NotPicklableLearner(Learner):
    @property
    def family(self):
        return "0"

    @property
    def params(self):
        return {}

    def __init__(self):
        self._val = lambda x: 1

    def predict(self, key, context, actions):
        return 0

    def learn(self, key, context, action, reward, probability):
        pass

class OneTimeSource(Source):

    def __init__(self, source: Source) -> None:
        self._source = source
        self._read_count = 0

    def read(self):
        if self._read_count > 0: raise Exception("Read more than once")

        self._read_count += 1

        return self._source.read()

class Benchmark_Single_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        CobaConfig.Logger = NoneLogger()
        CobaConfig.Benchmark['processes'] = 1
        CobaConfig.Benchmark['maxtasksperchild'] = None

    def test_sims(self):
        sim1       = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        sim2       = LambdaSimulation(4, lambda r,i: i, lambda r,i,c: [3,4,5], lambda r,i,c,a: cast(float,a))
        learner    = ModuloLearner()
        benchmark  = Benchmark([sim1,sim2], batch_count=1, ignore_raise=False)

        result             = benchmark.evaluate([learner])
        actual_learners    = result.learners.to_tuples()
        actual_simulations = result.simulations.to_tuples()
        actual_batches     = result.interactions.to_tuples()
        
        expected_learners    = [(0,"Modulo(p=0)","Modulo",'0')]
        expected_simulations = [(0, '"LambdaSimulation",{"Batch":{"count":1}}'), (1, '"LambdaSimulation",{"Batch":{"count":1}}')]
        expected_batches     = [(0, 0, 1, 1, 3, 5, mean([0,1,2,0,1])), (1, 0, 1, 1, 3, 4, mean([3,4,5,3]))]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_seeds(self):
        sim1      = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        learner   = ModuloLearner()
        benchmark = Benchmark([sim1], batch_sizes=[2], ignore_raise=False, shuffle=[1,4])

        result             = benchmark.evaluate([learner])
        actual_learners    = result.learners.to_tuples()
        actual_simulations = result.simulations.to_tuples()
        actual_batches     = result.interactions.to_tuples()

        expected_learners    = [(0,"Modulo(p=0)","Modulo",'0')]
        expected_simulations = [(0, '"LambdaSimulation",{"Shuffle":1},{"Batch":{"sizes":[2]}}'), (1, '"LambdaSimulation",{"Shuffle":4},{"Batch":{"sizes":[2]}}')]
        expected_batches     = [(0, 0, 1, 1, 3, 2, mean([1,0])), (1, 0, 1, 1, 3, 2, mean([2,0]))]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_take(self):
        sim1      = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        sim2      = LambdaSimulation(2, lambda r,i: i, lambda r,i,c: [3,4,5], lambda r,i,c,a: cast(float,a))
        learner   = ModuloLearner()
        benchmark = Benchmark([sim1,sim2], take=3, ignore_raise=False)

        result             = benchmark.evaluate([learner])
        actual_learners    = result.learners.to_tuples()
        actual_simulations = result.simulations.to_tuples()
        actual_batches     = result.interactions.to_tuples()
        
        expected_learners    = [(0,"Modulo(p=0)","Modulo",'0')]
        expected_simulations = [(0, '"LambdaSimulation",{"Take":3}'), (1, '"LambdaSimulation",{"Take":3}')]
        expected_batches     = [(0, 0, 1, 1, 3, 1, 0), (0, 0, 2, 1, 3, 1, 1), (0, 0, 3, 1, 3, 1, 2)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_learners(self):
        sim       = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        learner1  = ModuloLearner("0") #type: ignore
        learner2  = ModuloLearner("1") #type: ignore
        benchmark = Benchmark([sim], batch_count=1, ignore_raise=False)

        actual_result      = benchmark.evaluate([learner1, learner2])
        actual_learners    = actual_result.learners.to_tuples()
        actual_simulations = actual_result.simulations.to_tuples()
        actual_batches     = actual_result.interactions.to_tuples()

        expected_learners     = [(0,"Modulo(p=0)","Modulo",'0'), (1,"Modulo(p=1)","Modulo",'1')]
        expected_simulations  = [(0, '"LambdaSimulation",{"Batch":{"count":1}}')]
        expected_batches      = [(0, 0, 1, 1, 3, 5, mean([0,1,2,0,1])), (0, 1, 1, 1, 3, 5, mean([0,1,2,0,1])) ]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        working_learner = ModuloLearner()
        broken_learner  = BrokenLearner()
        benchmark       = Benchmark([sim], batch_count=1)

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log
        try:
            first_result  = benchmark.evaluate([working_learner], "coba/tests/.temp/transactions.log")
            second_result = benchmark.evaluate([broken_learner], "coba/tests/.temp/transactions.log")

            actual_learners    = second_result.learners.to_tuples()
            actual_simulations = second_result.simulations.to_tuples()
            actual_batches     = second_result.interactions.to_tuples()
            
            expected_learners    = [(0,"Modulo(p=0)","Modulo",'0')]
            expected_simulations = [(0, '"LambdaSimulation",{"Batch":{"count":1}}')]
            expected_batches     = [(0, 0, 1, 1, 3, 5, mean([0,1,2,0,1]))]
        except Exception as e:
            raise
        finally:
            if Path('coba/tests/.temp/transactions.log').exists(): Path('coba/tests/.temp/transactions.log').unlink()

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

    def test_ignore_raise(self):

        log_sink = MemorySink()
        CobaConfig.Logger = IndentLogger(log_sink)

        sim1       = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        sim2       = LambdaSimulation(4, lambda r,i: i, lambda r,i,c: [3,4,5], lambda r,i,c,a: cast(float,a))
        learners   = [ModuloLearner(), BrokenLearner()]
        benchmark  = Benchmark([sim1,sim2], batch_count=1, ignore_raise=True)

        result             = benchmark.evaluate(learners)
        actual_learners    = result.learners.to_tuples()
        actual_simulations = result.simulations.to_tuples()
        actual_batches     = result.interactions.to_tuples()

        expected_learners    = [(0,"Modulo(p=0)","Modulo",'0'),(1,"Broken","Broken",float('nan'))]
        expected_simulations = [(0,'"LambdaSimulation",{"Batch":{"count":1}}'), (1, '"LambdaSimulation",{"Batch":{"count":1}}')]
        expected_batches     = [(0, 0, 1, 1, 3, 5, mean([0,1,2,0,1])), (1, 0, 1, 1, 3, 4, mean([3,4,5,3]))]

        self.assertEqual(2, sum([int("Exception after" in item) for item in log_sink.items]))

        self.assertCountEqual(actual_learners[0], expected_learners[0])
        self.assertCountEqual(actual_learners[1][:3], expected_learners[1][:3])
        self.assertTrue(math.isnan(expected_learners[1][3]))

        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_batches, expected_batches)

class Benchmark_Multi_Tests(Benchmark_Single_Tests):
    
    @classmethod
    def setUpClass(cls) -> None:
        CobaConfig.Logger = NoneLogger()
        CobaConfig.Benchmark['processes'] = 2
        CobaConfig.Benchmark['maxtasksperchild'] = None

    def test_not_picklable_learner(self):
        sim1      = LambdaSimulation(5, lambda r,i: i, lambda r,i,c: [0,1,2], lambda r,i,c,a: cast(float,a))
        learner   = NotPicklableLearner()
        benchmark = Benchmark([sim1], batch_sizes=[2], ignore_raise=False, shuffle=[1,4])

        with self.assertRaises(Exception) as cm:
            benchmark.evaluate([learner])

        self.assertTrue("Learners are required to be picklable" in str(cm.exception))

if __name__ == '__main__':
    unittest.main()