import unittest
import math

from pathlib import Path
from typing import cast

from coba.environments import LambdaSimulation, Environments
from coba.pipes import Source, MemoryIO
from coba.learners import Learner, RandomLearner
from coba.config import CobaConfig, NullLogger, IndentLogger, BasicLogger
from coba.experiments import Experiment

class ModuloLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, context, action, reward, probability, info):
        pass

class BrokenLearner(Learner):

    @property
    def params(self):
        return {"family": "Broken"}

    def predict(self, context, actions):
        raise Exception("Broken Learner")

    def learn(self, context, action, reward, probability, info):
        pass

class PredictInfoLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ], (0,1)

    def learn(self, context, action, reward, probability, info):
        assert info == (0,1)

class LearnInfoLearner(Learner):
    def __init__(self, param:str="0"):
        self._param = param

    @property
    def params(self):
        return {"family": "Modulo", "p":self._param}

    def predict(self, context, actions):
        return [ int(i == actions.index(actions[context%len(actions)])) for i in range(len(actions)) ]

    def learn(self, context, action, reward, probability, info):
        return {"Modulo": self._param}

class NotPicklableLearner(ModuloLearner):
    def __init__(self):
        self._val = lambda x: 1
        super().__init__()

class NotPicklableLearnerWithReduce(NotPicklableLearner):
    def __init__(self):
        self._val = lambda x: 1
        super().__init__()

    def __reduce__(self):
        return (NotPicklableLearnerWithReduce, ())

class WrappedLearner(Learner):

    @property
    def params(self):
        return {"family": "family"}

    def __init__(self, learner):
        self._learner = learner

    def predict(self, key, context, actions):
        return self._learner.predict(key, context, actions)
    
    def learn(self, key, context, action, reward, probability) -> None:
        return self._learner.learn(key, context, action, reward, probability)

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
        CobaConfig.logger = NullLogger()
        CobaConfig.experiment.processes = 1
        CobaConfig.experiment.maxtasksperchild = -1

    def test_sim(self):
        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = ModuloLearner()
        benchmark  = Experiment([sim1])

        result              = benchmark.evaluate([learner])
        actual_learners     = result.learners.to_tuples()
        actual_simulations  = result.simulations.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0, "Modulo","Modulo(p=0)",'0')]
        expected_simulations  = [(0, 'LambdaSimulation')]
        expected_interactions = [(0,0,1,0), (0,0,2,1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_sims(self):
        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2       = LambdaSimulation(3, lambda i: i, lambda i,c: [3,4,5], lambda i,c,a: cast(float,a))
        learner    = ModuloLearner()
        benchmark  = Experiment([sim1,sim2])

        result              = benchmark.evaluate([learner])
        actual_learners     = result.learners.to_tuples()
        actual_simulations  = result.simulations.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0,"Modulo","Modulo(p=0)",'0')]
        expected_simulations  = [(0, 'LambdaSimulation'), (1, 'LambdaSimulation')]
        expected_interactions = [(0,0,1,0), (0,0,2,1), (1,0,1,3), (1,0,2,4), (1,0,3,5)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_eval_seeds(self):
        sim1      = LambdaSimulation(3, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner   = RandomLearner()
        benchmark = Experiment(Environments(sim1).shuffle([1,4]))

        result              = benchmark.evaluate([learner], seed=1)
        actual_learners     = result.learners.to_tuples()
        actual_simulations  = result.simulations.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0, "random", "random")]
        expected_simulations  = [(0, 'LambdaSimulation', 1), (1, 'LambdaSimulation', 4)]
        expected_interactions = [(0, 0, 1, 0), (0, 0, 2, 2), (0, 0, 3, 1), (1, 0, 1, 0), (1, 0, 2, 2), (1, 0, 3, 1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_learners(self):
        sim       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1  = ModuloLearner("0") #type: ignore
        learner2  = ModuloLearner("1") #type: ignore
        benchmark = Experiment([sim])

        actual_result       = benchmark.evaluate([learner1, learner2])
        actual_learners     = actual_result._learners.to_tuples()
        actual_simulations  = actual_result._simulations.to_tuples()
        actual_interactions = actual_result._interactions.to_tuples()

        expected_learners     = [(0, "Modulo", "Modulo(p=0)", '0'), (1, "Modulo", "Modulo(p=1)", '1')]
        expected_simulations  = [(0, 'LambdaSimulation')]
        expected_interactions = [(0,0,1,0),(0,0,2,1),(0,1,1,0),(0,1,2,1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_learn_info_learners(self):
        sim       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1  = LearnInfoLearner("0") #type: ignore
        benchmark = Experiment([sim])

        actual_result       = benchmark.evaluate([learner1])
        actual_learners     = actual_result._learners.to_tuples()
        actual_simulations  = actual_result._simulations.to_tuples()
        actual_interactions = actual_result._interactions.to_tuples()

        expected_learners       = [(0, "Modulo", "Modulo(p=0)", '0')]
        expected_simulations    = [(0, 'LambdaSimulation')]
        expected_interactions_1 = [(0,0,1,0,'0'),(0,0,2,1,'0')]
        expected_interactions_2 = [(0,0,1,'0',0),(0,0,2,'0',1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        
        try:
            self.assertCountEqual(actual_interactions, expected_interactions_1)
        except:
            self.assertCountEqual(actual_interactions, expected_interactions_2)

    def test_predict_info_learners(self):
        sim       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1  = PredictInfoLearner("0") #type: ignore
        benchmark = Experiment([sim])

        actual_result       = benchmark.evaluate([learner1])
        actual_learners     = actual_result._learners.to_tuples()
        actual_simulations  = actual_result._simulations.to_tuples()
        actual_interactions = actual_result._interactions.to_tuples()

        expected_learners       = [(0, "Modulo", "Modulo(p=0)", '0')]
        expected_simulations    = [(0, 'LambdaSimulation')]
        expected_interactions_1 = [(0,0,1,0),(0,0,2,1)]
        expected_interactions_2 = [(0,0,1,0),(0,0,2,1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        
        try:
            self.assertCountEqual(actual_interactions, expected_interactions_1)
        except:
            self.assertCountEqual(actual_interactions, expected_interactions_2)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        working_learner = ModuloLearner()
        broken_learner  = BrokenLearner()
        benchmark       = Experiment([sim])

        #the second time the broken_factory() shouldn't ever be used for learning or choosing
        #because it already worked the first time and we are "resuming" benchmark from transaction.log
        try:
            first_result  = benchmark.evaluate([working_learner], "coba/tests/.temp/transactions.log")
            second_result = benchmark.evaluate([broken_learner], "coba/tests/.temp/transactions.log")

            actual_learners     = second_result.learners.to_tuples()
            actual_simulations  = second_result.simulations.to_tuples()
            actual_interactions = second_result.interactions.to_tuples()
            
            expected_learners     = [(0,"Modulo","Modulo(p=0)",'0')]
            expected_simulations  = [(0,'LambdaSimulation')]
            expected_interactions = [(0, 0, 1, 0), (0, 0, 2, 1)]
        except Exception as e:
            raise
        finally:
            if Path('coba/tests/.temp/transactions.log').exists(): Path('coba/tests/.temp/transactions.log').unlink()

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_ignore_raise(self):

        log_sink = MemoryIO()
        CobaConfig.logger = IndentLogger(log_sink)

        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2       = LambdaSimulation(3, lambda i: i, lambda i,c: [3,4,5], lambda i,c,a: cast(float,a))
        learners   = [ModuloLearner(), BrokenLearner()]
        benchmark  = Experiment([sim1,sim2])

        result              = benchmark.evaluate(learners)
        actual_learners     = result.learners.to_tuples()
        actual_simulations  = result.simulations.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0,"Modulo","Modulo(p=0)",'0'),(1,"Broken","Broken",float('nan'))]
        expected_simulations  = [(0,'LambdaSimulation'), (1,'LambdaSimulation')]
        expected_interactions = [(0, 0, 1, 0), (0, 0, 2, 1), (1, 0, 1, 3), (1, 0, 2, 4), (1, 0, 3, 5)]

        self.assertEqual(2, sum([int("Unexpected exception:" in item) for item in log_sink.items]))

        self.assertCountEqual(actual_learners[0], expected_learners[0])
        self.assertCountEqual(actual_learners[1][:3], expected_learners[1][:3])
        self.assertTrue(math.isnan(expected_learners[1][3]))

        self.assertCountEqual(actual_simulations, expected_simulations)
        self.assertCountEqual(actual_interactions, expected_interactions)

class Benchmark_Multi_Tests(Benchmark_Single_Tests):
    
    @classmethod
    def setUpClass(cls) -> None:
        CobaConfig.logger = NullLogger()
        CobaConfig.experiment.processes = 2
        CobaConfig.experiment.maxtasksperchild = None

    def test_not_picklable_learner_sans_reduce(self):
        sim1      = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner   = NotPicklableLearner()
        benchmark = Experiment([sim1])

        CobaConfig.logger = BasicLogger(MemoryIO())

        benchmark.evaluate([learner])

        self.assertEqual(1, len(CobaConfig.logger.sink.items))
        self.assertIn("pickle", CobaConfig.logger.sink.items[0])

    def test_wrapped_not_picklable_learner_sans_reduce(self):
        sim1      = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner   = WrappedLearner(NotPicklableLearner())
        benchmark = Experiment([sim1])

        CobaConfig.logger = BasicLogger(MemoryIO())
        
        benchmark.evaluate([learner])

        self.assertEqual(1, len(CobaConfig.logger.sink.items))
        self.assertIn("pickle", CobaConfig.logger.sink.items[0])

    def test_not_picklable_learner_with_reduce(self):
        sim1      = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner   = NotPicklableLearnerWithReduce()
        benchmark = Experiment([sim1])

        benchmark.evaluate([learner])

    def test_wrapped_not_picklable_learner_with_reduce(self):
        sim1      = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner   = WrappedLearner(NotPicklableLearnerWithReduce())
        benchmark = Experiment([sim1])

        benchmark.evaluate([learner])

if __name__ == '__main__':
    unittest.main()