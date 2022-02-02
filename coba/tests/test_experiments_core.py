import unittest
import math

from pathlib import Path
from typing import cast

from coba.environments import Environment, LambdaSimulation
from coba.experiments.tasks import OnlineOnPolicyEvalTask
from coba.pipes import Source, ListIO
from coba.learners import Learner
from coba.contexts import CobaContext, LearnerContext, CobaContext, IndentLogger, BasicLogger, NullLogger
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
        LearnerContext.logger.write({"Modulo": self._param})

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

class ExceptionEnvironment(Environment):

    def __init__(self, exception: Exception):
        self._exc = exception

    def params(self):
        return {}

    def read(self):
        raise self._exc

class Experiment_Single_Tests(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        CobaContext.logger = NullLogger()
        CobaContext.experiment.processes = 1
        CobaContext.experiment.maxchunksperchild = 0

    def test_sim(self):
        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = ModuloLearner()
        experiment = Experiment([sim1], [learner], evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_tuples()
        actual_environments = result.environments.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0, "Modulo","Modulo(p=0)",'0')]
        expected_environments = [(0, 'LambdaSimulation')]
        expected_interactions = [(0,0,1,0), (0,0,2,1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_sims(self):
        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2       = LambdaSimulation(3, lambda i: i, lambda i,c: [3,4,5], lambda i,c,a: cast(float,a))
        learner    = ModuloLearner()
        experiment = Experiment([sim1,sim2], [learner], evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_tuples()
        actual_environments = result.environments.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0,"Modulo","Modulo(p=0)",'0')]
        expected_environments = [(0, 'LambdaSimulation'), (1, 'LambdaSimulation')]
        expected_interactions = [(0,0,1,0), (0,0,2,1), (1,0,1,3), (1,0,2,4), (1,0,3,5)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_learners(self):
        sim        = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1   = ModuloLearner("0") #type: ignore
        learner2   = ModuloLearner("1") #type: ignore
        experiment = Experiment([sim], [learner1, learner2], evaluation_task=OnlineOnPolicyEvalTask(False))

        actual_result       = experiment.evaluate()
        actual_learners     = actual_result._learners.to_tuples()
        actual_environments = actual_result._environments.to_tuples()
        actual_interactions = actual_result.interactions.to_tuples()

        expected_learners     = [(0, "Modulo", "Modulo(p=0)", '0'), (1, "Modulo", "Modulo(p=1)", '1')]
        expected_environments = [(0, 'LambdaSimulation')]
        expected_interactions = [(0,0,1,0),(0,0,2,1),(0,1,1,0),(0,1,2,1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_learn_info_learners(self):
        sim        = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1   = LearnInfoLearner("0") #type: ignore
        experiment = Experiment([sim],[learner1], evaluation_task=OnlineOnPolicyEvalTask(False))

        actual_result       = experiment.evaluate()
        actual_learners     = actual_result._learners.to_tuples()
        actual_environments = actual_result._environments.to_tuples()
        actual_interactions = actual_result.interactions.to_tuples()

        expected_learners       = [(0, "Modulo", "Modulo(p=0)", '0')]
        expected_environments   = [(0, 'LambdaSimulation')]
        expected_interactions_1 = [(0,0,1,0,'0'),(0,0,2,1,'0')]
        expected_interactions_2 = [(0,0,1,'0',0),(0,0,2,'0',1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        
        try:
            self.assertCountEqual(actual_interactions, expected_interactions_1)
        except:
            self.assertCountEqual(actual_interactions, expected_interactions_2)

    def test_predict_info_learners(self):
        sim        = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner1   = PredictInfoLearner("0") #type: ignore
        experiment = Experiment([sim],[learner1],evaluation_task=OnlineOnPolicyEvalTask(False))

        actual_result       = experiment.evaluate()
        actual_learners     = actual_result._learners.to_tuples()
        actual_environments = actual_result._environments.to_tuples()
        actual_interactions = actual_result.interactions.to_tuples()

        expected_learners       = [(0, "Modulo", "Modulo(p=0)", '0')]
        expected_environments   = [(0, 'LambdaSimulation')]
        expected_interactions_1 = [(0,0,1,0),(0,0,2,1)]
        expected_interactions_2 = [(0,0,1,0),(0,0,2,1)]

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        
        try:
            self.assertCountEqual(actual_interactions, expected_interactions_1)
        except:
            self.assertCountEqual(actual_interactions, expected_interactions_2)

    def test_transaction_resume_1(self):
        sim             = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        working_learner = ModuloLearner()
        broken_learner  = BrokenLearner()

        #the second Experiment shouldn't ever call broken_factory() because
        #we're resuming from the first experiment's transaction.log
        try:
            first_result  = Experiment([sim],[working_learner],evaluation_task=OnlineOnPolicyEvalTask(False)).evaluate("coba/tests/.temp/transactions.log")
            second_result = Experiment([sim],[broken_learner ],evaluation_task=OnlineOnPolicyEvalTask(False)).evaluate("coba/tests/.temp/transactions.log")

            actual_learners     = second_result.learners.to_tuples()
            actual_environments = second_result.environments.to_tuples()
            actual_interactions = second_result.interactions.to_tuples()
            
            expected_learners     = [(0,"Modulo","Modulo(p=0)",'0')]
            expected_environments = [(0,'LambdaSimulation')]
            expected_interactions = [(0, 0, 1, 0), (0, 0, 2, 1)]
        except Exception as e:
            raise
        finally:
            if Path('coba/tests/.temp/transactions.log').exists(): Path('coba/tests/.temp/transactions.log').unlink()

        self.assertCountEqual(actual_learners, expected_learners)
        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_ignore_raise(self):

        CobaContext.logger = IndentLogger(ListIO())

        sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        sim2       = LambdaSimulation(3, lambda i: i, lambda i,c: [3,4,5], lambda i,c,a: cast(float,a))
        experiment = Experiment([sim1,sim2], [ModuloLearner(), BrokenLearner()],evaluation_task=OnlineOnPolicyEvalTask(False))

        result              = experiment.evaluate()
        actual_learners     = result.learners.to_tuples()
        actual_environments = result.environments.to_tuples()
        actual_interactions = result.interactions.to_tuples()

        expected_learners     = [(0,"Modulo","Modulo(p=0)",'0'),(1,"Broken","Broken",float('nan'))]
        expected_environments = [(0,'LambdaSimulation'), (1,'LambdaSimulation')]
        expected_interactions = [(0, 0, 1, 0), (0, 0, 2, 1), (1, 0, 1, 3), (1, 0, 2, 4), (1, 0, 3, 5)]

        self.assertIsInstance(CobaContext.logger, IndentLogger)
        self.assertEqual(2, sum([int("Unexpected exception:" in item) for item in CobaContext.logger.sink.items]))

        self.assertCountEqual(actual_learners[0], expected_learners[0])
        self.assertCountEqual(actual_learners[1][:3], expected_learners[1][:3])
        self.assertTrue(math.isnan(expected_learners[1][3]))

        self.assertCountEqual(actual_environments, expected_environments)
        self.assertCountEqual(actual_interactions, expected_interactions)

    def test_config_set(self):

        exp = Experiment([], [])

        CobaContext.experiment.processes = 10
        self.assertEqual(10,exp.processes)

        CobaContext.experiment.maxchunksperchild = 3
        self.assertEqual(3,exp.maxchunksperchild)

        CobaContext.experiment.chunk_by = 'source'
        self.assertEqual('source',exp.chunk_by)

        exp.config(processes=2, maxchunksperchild=5, chunk_by='task')
        self.assertEqual(2,exp.processes)
        self.assertEqual(5,exp.maxchunksperchild)
        self.assertEqual('task',exp.chunk_by)

    def test_restore_not_matched_environments(self):

        path = Path("coba/tests/.temp/experiment.log")

        if path.exists(): path.unlink()
        path.write_text('["version",4]\n["experiment",{"n_environments":1,"n_learners":1}]')

        try:
            sim1       = LambdaSimulation(2, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
            learner    = ModuloLearner()

            with self.assertRaises(AssertionError) as e:
                result = Experiment([sim1,sim1], [learner]).evaluate(str(path))

            with self.assertRaises(AssertionError) as e:
                result = Experiment([sim1], [learner,learner]).evaluate(str(path))    

        finally:
            path.unlink()

class Experiment_Multi_Tests(Experiment_Single_Tests):

    @classmethod
    def setUp(cls) -> None:
        CobaContext.logger = NullLogger()
        CobaContext.experiment.processes = 2
        CobaContext.experiment.maxchunksperchild = 0

    def test_not_picklable_learner_sans_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = NotPicklableLearner()
        experiment = Experiment([sim1],[learner])

        CobaContext.logger = BasicLogger(ListIO())

        experiment.evaluate()

        self.assertEqual(1, len(CobaContext.logger.sink.items))
        self.assertIn("pickle", CobaContext.logger.sink.items[0])

    def test_wrapped_not_picklable_learner_sans_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = WrappedLearner(NotPicklableLearner())
        experiment = Experiment([sim1],[learner])

        CobaContext.logger = BasicLogger(ListIO())
        
        experiment.evaluate()

        self.assertEqual(1, len(CobaContext.logger.sink.items))
        self.assertIn("pickle", CobaContext.logger.sink.items[0])

    def test_not_picklable_learner_with_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = NotPicklableLearnerWithReduce()
        experiment = Experiment([sim1],[learner])

        experiment.evaluate()

    def test_wrapped_not_picklable_learner_with_reduce(self):
        sim1       = LambdaSimulation(5, lambda i: i, lambda i,c: [0,1,2], lambda i,c,a: cast(float,a))
        learner    = WrappedLearner(NotPicklableLearnerWithReduce())
        experiment = Experiment([sim1],[learner])

        experiment.evaluate()

if __name__ == '__main__':
    unittest.main()